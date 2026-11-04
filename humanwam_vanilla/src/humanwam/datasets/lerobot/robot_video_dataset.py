# Modified by Open-HumanScale contributors: package namespace renamed to humanwam.
import copy
import hashlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional
import time
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from concurrent.futures import ThreadPoolExecutor, as_completed

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import (
    BaseLerobotDataset,
    _resolve_temporal_resampling_by_root)
from .lerobot.datasets.video_utils import decode_video_frames
from .temporal_resampling import (
    TEMPORAL_RESAMPLING_PROVENANCE_VERSION,
    TemporalResamplingSpec,
    canonicalize_temporal_resampling_spec,
    is_identity_temporal_resampling,
    temporal_resampling_provenance,
    temporal_resampling_spec_digest,
    parse_temporal_resampling_spec)
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from .text_embedding_pack import TextEmbeddingPack
from .object_conditioned_instruction import ObjectConditionedInstructionSampler
from .episode_instruction import EpisodeInstructionSampler
from .camera_intrinsics_warp import CameraIntrinsicsWarp
from .eef_anchor_overlay import EefAnchorOverlay
from .action_supervision import bad_segment_action_mask
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from ..task_weighted_sampling import build_task_weighted_sampling_contract
from humanwam.utils.logging_config import get_logger
from humanwam.utils import misc
from humanwam.camera_pose import camera_to_world_pose7_to_delta
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def _load_frame_exclusion_manifest(manifest, dataset_dirs):
    """Load a per-root/per-episode frame denylist used for training anchors.

    The manifest stores local episode frame indices.  We resolve them to the
    concatenated BaseLerobotDataset index after construction; this keeps the
    source parquet/video trees immutable and makes the exclusion auditable.
    """
    if manifest is None or str(manifest).strip().lower() in {"", "none", "null"}:
        return None, {}
    path = Path(str(manifest)).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    roots = value.get("roots")
    resolved = {str(Path(root).expanduser().resolve()): entry for root, entry in roots.items()}
    normalized = {}
    for root, entry in resolved.items():
        episodes = {}
        for episode, record in entry.get("episodes", {}).items():
            indices = sorted({int(i) for i in record["duplicate_frame_indices"]})
            episodes[int(episode)] = frozenset(indices)
        normalized[root] = episodes
    return path, normalized


def cover_resize_center_crop(video, size):
    """Aspect-preserving cover resize + center crop for [..., H, W] frames.

    Matches the Ego visual contract (`ResizeSmallestSideAspectPreserving` followed
    by `CenterCrop`) so a camera view can be brought onto the 256x448 grid inside a
    multi-camera composition, where the per-camera processor transforms have
    already run and cannot be configured per camera.
    """
    target_h, target_w = int(size[0]), int(size[1])
    height, width = int(video.shape[-2]), int(video.shape[-1])
    scale = max(target_h / height, target_w / width)
    resized_h = max(int(math.ceil(height * scale)), target_h)
    resized_w = max(int(math.ceil(width * scale)), target_w)
    resized = transforms_F.resize(
        video, size=[resized_h, resized_w],
        interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True,
    )
    top = (resized_h - target_h) // 2
    left = (resized_w - target_w) // 2
    return resized[..., top : top + target_h, left : left + target_w]


def _build_action_tokenizer_source_routes(dataset_dirs, rules):
    """Validate root-to-action-tokenizer routing and return exact-root lookup.

    Rules intentionally use explicit substring matches, mirroring the wrist-C source
    routing already used by the ego alignment transform. Every configured root must
    match exactly one rule when routing is enabled; silent fallback to a shared
    tokenizer would invalidate the ablation.
    """
    if rules is None:
        return None
    if OmegaConf.is_config(rules):
        rules = OmegaConf.to_container(rules, resolve=True)

    parsed = []
    for position, rule in enumerate(rules):
        match = str(rule.get("match", "")).strip()
        source = str(rule.get("source", "")).strip()
        parsed.append((match, source))

    routes = {}
    match_counts = [0] * len(parsed)
    for dataset_root in dataset_dirs:
        root = str(dataset_root)
        matches = [
            (position, source)
            for position, (match, source) in enumerate(parsed)
            if match in root
        ]
        position, source = matches[0]
        match_counts[position] += 1
        routes[str(Path(root).expanduser().resolve())] = source
    return routes


def _manifest_provenance(path: Optional[str | Path]) -> dict:
    if path is None or str(path).strip().lower() in {"", "none", "null"}:
        return {"enabled": False, "sha256": None}
    resolved = Path(str(path)).expanduser().resolve()
    return {
        "enabled": True,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _environment_sampling_provenance(contract: Optional[dict]) -> dict:
    """Return the norm-stats identity of the effective scalar sampling measure."""

    if contract is None:
        return {"mode": None}
    result = {
        "mode": contract["mode"],
        "environment_count": len(contract["environment_ids"]),
        "environment_ids_sha256": hashlib.sha256(
            "\n".join(contract["environment_ids"]).encode("utf-8")
        ).hexdigest(),
        "training_total_samples": int(contract["training_total_samples"]),
        "weighting": "equal environment mass; uniform anchors within env",
    }
    if contract.get("active_axis") is not None:
        for field in (
            "active_axis",
            "source_annotation_manifest_content_sha256",
            "registry_sha256",
            "assignment_sha256",
            "projected_population_sha256",
            "projection_policy_sha256",
            "legacy_projection_before_sha256",
            "legacy_projection_after_sha256",
        ):
            result[field] = contract[field]
    return result


def _plain_provenance_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {
            key: _plain_provenance_value(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain_provenance_value(item) for item in value]
    return value


def _expanded_temporal_resampling_provenance(
    dataset_dirs: list[str],
    specs: list,
) -> dict:
    """Canonical, rule-expression-independent per-root temporal provenance."""
    roots = []
    for dataset_root, spec in zip(dataset_dirs, specs, strict=True):
        roots.append(
            {
                "dataset_root": str(Path(dataset_root).expanduser().resolve()),
                "spec": canonicalize_temporal_resampling_spec(spec),
                "spec_sha256": temporal_resampling_spec_digest(spec),
            }
        )
    payload = {
        "version": TEMPORAL_RESAMPLING_PROVENANCE_VERSION,
        "roots": roots,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def _as_int64_numpy(value, *, name: str) -> np.ndarray:
    """Return a one-dimensional CPU int64 view/copy for compact index metadata."""

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(np.int64, copy=False)


def _live_norm_stats_corpus_provenance(
    *,
    dataset_dirs: list[str],
    base_dataset,
    sampling_specs: Optional[tuple[TemporalResamplingSpec, ...]],
) -> dict:
    """Build the selected live per-root corpus contract used by sampled stats."""

    children = list(base_dataset.multi_dataset._datasets)

    roots = []
    total_episodes = 0
    total_native_frames = 0
    total_virtual_frames = 0
    total_sampling_frames = 0
    for position, (dataset_root, child) in enumerate(
        zip(dataset_dirs, children, strict=True)
    ):
        root = str(Path(dataset_root).expanduser().resolve())
        data_spec = parse_temporal_resampling_spec(child.temporal_resampling)
        if getattr(child, "_episode_segments_enabled", False):
            native_index = child.episode_data_index
        else:
            native_index = getattr(child, "native_episode_data_index", None)
        if native_index is None:
            native_index = child.episode_data_index
        starts = _as_int64_numpy(
            native_index["from"], name=f"{root} native_episode_data_index.from"
        )
        ends = _as_int64_numpy(
            native_index["to"], name=f"{root} native_episode_data_index.to"
        )
        native_lengths = ends - starts
        temporal = temporal_resampling_provenance(
            data_spec, native_episode_lengths=native_lengths
        )
        sampling_spec = (
            sampling_specs[position] if sampling_specs is not None else data_spec
        )
        sampling_temporal = temporal_resampling_provenance(
            sampling_spec, native_episode_lengths=native_lengths
        )
        temporal_corpus = temporal["corpus"]
        sampling_corpus = sampling_temporal["corpus"]
        record = {
            "dataset_root": root,
            "episodes": int(starts.size),
            "native_frames": int(temporal_corpus["native_frames"]),
            "virtual_frames": int(temporal_corpus["virtual_frames"]),
            "sampling_frames": int(sampling_corpus["virtual_frames"]),
            "temporal_resampling": temporal,
            "sampling_temporal_resampling": sampling_temporal,
        }
        roots.append(record)
        total_episodes += record["episodes"]
        total_native_frames += record["native_frames"]
        total_virtual_frames += record["virtual_frames"]
        total_sampling_frames += record["sampling_frames"]

    return {
        "num_roots": len(roots),
        "total_episodes": total_episodes,
        "total_native_frames": total_native_frames,
        "total_virtual_frames": total_virtual_frames,
        "total_sampling_frames": total_sampling_frames,
        "roots": roots,
    }


def _sampling_lengths(
    native_lengths: np.ndarray,
    spec: TemporalResamplingSpec,
    *,
    root: str,
) -> np.ndarray:
    """Vectorized equivalent of virtual_episode_length for one validated root spec."""

    if spec.is_identity:
        return native_lengths
    scaled = native_lengths.astype(np.float64) * spec.factor
    # Python round(), used by virtual_episode_length, and np.rint both use
    # round-to-nearest/ties-to-even on float64 values.
    result = np.rint(scaled).astype(np.int64)
    return result


class _FullHorizonIndex:
    """Compact mapping from padding-free anchors to native dataset indices."""

    def __init__(
        self,
        *,
        episode_starts,
        episode_ends,
        min_query_delta: int,
        max_query_delta: int,
    ):
        starts = _as_int64_numpy(
            episode_starts, name="full_horizon episode_data_index.from"
        )
        ends = _as_int64_numpy(
            episode_ends, name="full_horizon episode_data_index.to"
        )
        min_query_delta = int(min_query_delta)
        max_query_delta = int(max_query_delta)

        valid_starts = starts + max(0, -min_query_delta)
        valid_ends = ends - max(0, max_query_delta)
        valid_counts = np.maximum(valid_ends - valid_starts, 0)
        eligible = valid_counts > 0

        self.anchor_starts = valid_starts[eligible]
        self.anchor_ends = valid_ends[eligible]
        self.episode_starts = starts[eligible]
        self.episode_ends = ends[eligible]
        self.cumulative_ends = np.cumsum(valid_counts[eligible], dtype=np.int64)
        self.num_frames = int(self.cumulative_ends[-1])
        self.native_num_frames = int(ends[-1])
        self.num_episodes = int(starts.size)
        self.eligible_episodes = int(np.count_nonzero(eligible))
        self.min_query_delta = min_query_delta
        self.max_query_delta = max_query_delta

    def __len__(self) -> int:
        return self.num_frames

    def map_index(self, index: int) -> int:
        logical_index = int(index)
        episode_pos = int(
            np.searchsorted(self.cumulative_ends, logical_index, side="right")
        )
        logical_episode_start = (
            0 if episode_pos == 0 else int(self.cumulative_ends[episode_pos - 1])
        )
        native_index = int(self.anchor_starts[episode_pos]) + (
            logical_index - logical_episode_start
        )
        return native_index


class _SamplingOnlyTemporalIndex:
    """Compact logical-anchor view over an identity/native BaseLerobotDataset.

    The index stores one cumulative int64 end offset per selected episode, rather
    than one entry per logical frame. A logical slot is first located in its episode,
    mapped to the nearest native anchor using the configured slowdown policy, and
    then handed to the unchanged native Base/Multi dataset. Consequently the loaded
    33-frame window is continuous native time even though anchor frequency matches a
    materialized slow corpus.
    """

    def __init__(
        self,
        *,
        dataset_dirs: list[str],
        base_dataset,
        specs: tuple[TemporalResamplingSpec, ...],
    ):
        children = list(base_dataset.multi_dataset._datasets)

        self.specs = specs
        self.enabled = any(not spec.is_identity for spec in specs)
        resolved_roots = [
            str(Path(root).expanduser().resolve()) for root in dataset_dirs
        ]
        root_records = []
        root_episode_ends = []
        root_native_frames = []
        root_sampling_frames = []
        episode_count = 0
        native_count = 0
        sampling_count = 0

        # The full ego corpus has about 1.7M episodes but only ~5k episodes per
        # configured chunk. Fill the single global cumulative array root by root so
        # temporary memory remains bounded by the largest chunk, not the corpus.
        total_episodes = sum(int(child.num_episodes) for child in children)
        sampling_episode_ends = (
            np.empty(total_episodes, dtype=np.int64) if self.enabled else None
        )

        for root, child, spec in zip(
            resolved_roots, children, specs, strict=True
        ):
            starts = _as_int64_numpy(
                child.episode_data_index["from"],
                name=f"{root} episode_data_index.from",
            )
            ends = _as_int64_numpy(
                child.episode_data_index["to"],
                name=f"{root} episode_data_index.to",
            )

            native_lengths = ends - starts
            root_native = int(ends[-1])

            if self.enabled:
                logical_lengths = _sampling_lengths(
                    native_lengths,
                    spec,
                    root=root,
                )
                root_sampling = int(sum(map(int, logical_lengths)))
                local_ends = np.cumsum(logical_lengths, dtype=np.int64)
                next_episode_count = episode_count + starts.size
                sampling_episode_ends[episode_count:next_episode_count] = (
                    local_ends + sampling_count
                )
                sampling_count += root_sampling
            else:
                root_sampling = root_native
                sampling_count += root_sampling
                next_episode_count = episode_count + starts.size

            episode_count = next_episode_count
            native_count += root_native
            root_episode_ends.append(episode_count)
            root_native_frames.append(root_native)
            root_sampling_frames.append(root_sampling)
            root_records.append(
                {
                    "dataset_root": root,
                    "episodes": int(starts.size),
                    "native_frames": root_native,
                    "sampling_frames": root_sampling,
                    "spec": canonicalize_temporal_resampling_spec(spec),
                    "spec_sha256": temporal_resampling_spec_digest(spec),
                }
            )

        global_starts = _as_int64_numpy(
            base_dataset.episode_data_index["from"],
            name="combined episode_data_index.from",
        )
        global_ends = _as_int64_numpy(
            base_dataset.episode_data_index["to"],
            name="combined episode_data_index.to",
        )

        self.num_episodes = total_episodes
        self.native_num_frames = native_count
        self.num_frames = sampling_count
        self.root_episode_ends = np.asarray(root_episode_ends, dtype=np.int64)
        self.root_native_frames = tuple(root_native_frames)
        self.root_sampling_frames = tuple(root_sampling_frames)
        self.sampling_episode_ends = sampling_episode_ends
        # These arrays share storage with the Base dataset's already-required
        # combined episode index when possible; no frame-sized mapping is created.
        self.native_episode_starts = global_starts
        self.native_episode_ends = global_ends
        self.corpus = {
            "num_episodes": total_episodes,
            "native_frames": native_count,
            "sampling_frames": sampling_count,
            "roots": root_records,
        }

    def __len__(self) -> int:
        return self.num_frames

    def map_index(self, index: int) -> int:
        if not self.enabled:
            return index
        logical_index = int(index)

        episode_pos = int(
            np.searchsorted(
                self.sampling_episode_ends,
                logical_index,
                side="right",
            )
        )
        logical_episode_start = (
            0
            if episode_pos == 0
            else int(self.sampling_episode_ends[episode_pos - 1])
        )
        logical_episode_end = int(self.sampling_episode_ends[episode_pos])
        logical_local = logical_index - logical_episode_start
        logical_length = logical_episode_end - logical_episode_start

        native_start = int(self.native_episode_starts[episode_pos])
        native_end = int(self.native_episode_ends[episode_pos])
        native_length = native_end - native_start
        root_pos = int(
            np.searchsorted(self.root_episode_ends, episode_pos, side="right")
        )
        spec = self.specs[root_pos]

        if spec.is_identity or native_length == 1:
            native_local = logical_local if spec.is_identity else 0
        elif spec.mapping == "endpoint":
            if logical_length <= 1:
                coordinate = 0.0
            else:
                coordinate = (
                    float(logical_local)
                    * float(native_length - 1)
                    / float(logical_length - 1)
                )
            native_local = math.floor(coordinate + 0.5)
        elif spec.mapping == "node_locked":
            coordinate = min(
                float(logical_local) / spec.factor,
                float(native_length - 1),
            )
            native_local = math.floor(coordinate + 0.5)

        native_local = min(max(int(native_local), 0), native_length - 1)
        return native_start + native_local


class _DiskVideoFrameCache:
    """Persistent, disk-backed decoded-frame cache for a single inner LeRobotDataset.

    Decodes every (episode, camera) clip ONCE into an on-disk memmap
    `[N_frames, H, W, 3] uint8` per camera (built lazily; reused across runs),
    then serves frames with zero video decoding. `lookup(query_timestamps,
    ep_idx)` is a drop-in for `LeRobotDataset._query_videos`, returning
    `{vid_key: [T, 3, H, W] float32 in [0, 1]}`.

    Storage: total_frames * H * W * 3 bytes per camera. With plentiful RAM the
    OS page cache keeps hot frames resident, so reads are RAM-speed while the
    bytes persist on disk and never need re-decoding.
    """

    CACHE_VERSION = 2  # v2: store float32 [0,1] at processed resolution (was uint8)

    def __init__(self, inner_dataset, camera_keys, cache_dir, build_workers=16, target_hw=None):
        self.fps = inner_dataset.meta.fps
        self.camera_keys = list(camera_keys)
        self.tolerance_s = inner_dataset.tolerance_s
        # Bake the (deterministic) downsize into the cache: store frames already
        # resized to the processed resolution so per-step __getitem__ skips the
        # expensive raw-resolution resize. None -> store at decoded resolution.
        self.target_hw = tuple(target_hw) if target_hw is not None else None

        episodes = (
            inner_dataset.episodes
            if inner_dataset.episodes is not None
            else list(range(inner_dataset.meta.total_episodes))
        )
        from_idx = inner_dataset.episode_data_index["from"]
        to_idx = inner_dataset.episode_data_index["to"]

        self.ep_lengths = {}
        self.ep_offsets = {}
        running = 0
        for local_pos, ep_idx in enumerate(episodes):
            length = int(to_idx[local_pos].item() - from_idx[local_pos].item())
            self.ep_lengths[int(ep_idx)] = length
            self.ep_offsets[int(ep_idx)] = running
            running += length
        self.total_frames = running

        # Probe one frame to learn (H, W).
        first_ep = int(episodes[0])
        first_path = inner_dataset.root / inner_dataset.meta.get_video_file_path(first_ep, self.camera_keys[0])
        probe = decode_video_frames(first_path, [0.0], self.tolerance_s, inner_dataset.video_backend)
        _, _, H, W = probe.shape
        if self.target_hw is not None:
            self.H, self.W = int(self.target_hw[0]), int(self.target_hw[1])
        else:
            self.H, self.W = int(H), int(W)

        # Cache directory keyed by dataset identity + exact selected-episode layout
        # (+ stored resolution), so train/val splits never alias and a changed
        # dataset or target resolution rebuilds.
        ep_sig = hashlib.sha1(
            (str(inner_dataset.root) + f"|hw={self.H}x{self.W}|" + ",".join(str(int(e)) for e in episodes)).encode()
        ).hexdigest()[:16]
        self.dir = Path(cache_dir) / f"{Path(inner_dataset.root).name}_{ep_sig}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "meta.json"

        total_gb = self.total_frames * self.H * self.W * 3 * len(self.camera_keys) / (1024 ** 3)

        # Cross-process-safe build via an O_EXCL lock: dataset construction runs
        # before Accelerate inits the process group, so dist barriers aren't
        # available -- and on multi-node each node has its own /dev/shm. A single
        # winner builds; everyone else polls until meta.json is valid (meta is
        # written last). Prevents the concurrent-build JSONDecodeError race.
        if not self._is_valid_cache():
            lock_path = self.dir / ".building.lock"
            got_lock = False
            try:
                _fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(_fd)
                got_lock = True
            except FileExistsError:
                got_lock = False

            if got_lock:
                try:
                    logger.info(
                        f"[video-cache] building cache {self.dir} "
                        f"({len(self.camera_keys)} cams x {self.total_frames} frames x {self.H}x{self.W}, {total_gb:.1f} GB)"
                    )
                    self.buffers = {cam: self._open_memmap(cam, mode="w+") for cam in self.camera_keys}
                    self._fill_buffers(inner_dataset, episodes, build_workers)
                    for cam in self.camera_keys:
                        self.buffers[cam].flush()
                    self._write_meta()  # written last -> signals completion
                    del self.buffers
                finally:
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
            else:
                logger.info(f"[video-cache] waiting for another process to build {self.dir} ...")
                waited = 0.0
                while not self._is_valid_cache():
                    time.sleep(2.0)
                    waited += 2.0

        logger.info(
            f"[video-cache] opening cache {self.dir} "
            f"({len(self.camera_keys)} cams x {self.total_frames} frames x {self.H}x{self.W}, {total_gb:.1f} GB)"
        )
        self.buffers = {cam: self._open_memmap(cam, mode="r") for cam in self.camera_keys}

    def _memmap_path(self, cam):
        return self.dir / f"{cam}.f32"

    def _open_memmap(self, cam, mode):
        return np.memmap(
            self._memmap_path(cam),
            dtype=np.float32,
            mode=mode,
            shape=(self.total_frames, self.H, self.W, 3),
        )

    def _expected_meta(self):
        return {
            "version": self.CACHE_VERSION,
            "total_frames": self.total_frames,
            "H": self.H,
            "W": self.W,
            "fps": self.fps,
            "cameras": self.camera_keys,
            "ep_lengths": {str(k): v for k, v in self.ep_lengths.items()},
            "ep_offsets": {str(k): v for k, v in self.ep_offsets.items()},
        }

    def _is_valid_cache(self):
        if not self.meta_path.exists():
            return False
        try:
            meta = json.loads(self.meta_path.read_text())
        except Exception:
            return False
        exp = self._expected_meta()
        for key in ("version", "total_frames", "H", "W", "cameras"):
            if meta.get(key) != exp[key]:
                return False
        nbytes = self.total_frames * self.H * self.W * 3 * 4  # float32
        for cam in self.camera_keys:
            p = self._memmap_path(cam)
            if not p.exists() or p.stat().st_size != nbytes:
                return False
        return True

    def _write_meta(self):
        self.meta_path.write_text(json.dumps(self._expected_meta()))

    def _fill_buffers(self, inner_dataset, episodes, build_workers):
        t0 = time.perf_counter()
        tasks = [(int(ep), cam) for ep in episodes for cam in self.camera_keys]

        def _decode_one(ep_idx, cam):
            length = self.ep_lengths[ep_idx]
            ts_offset = inner_dataset.meta.get_video_timestamp_offset(ep_idx, cam)
            timestamps = [ts_offset + i / self.fps for i in range(length)]
            path = inner_dataset.root / inner_dataset.meta.get_video_file_path(ep_idx, cam)
            frames = decode_video_frames(
                path,
                timestamps,
                self.tolerance_s,
                inner_dataset.video_backend,
            )
            if self.target_hw is not None and (frames.shape[-2], frames.shape[-1]) != (self.H, self.W):
                # Bake the processed-resolution downsize in once (float, antialiased),
                # matching the per-step torchvision Resize that this replaces.
                frames = transforms_F.resize(
                    frames, [self.H, self.W],
                    interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True,
                )
            # Store float32 [0,1] at processed resolution (full precision, and the
            # per-step ToTensor uint8->float divide is skipped). [length, H, W, 3].
            f32_frames = frames.permute(0, 2, 3, 1).contiguous().float().numpy()
            off = self.ep_offsets[ep_idx]
            self.buffers[cam][off : off + length] = f32_frames
            return length

        total_decoded = 0
        with ThreadPoolExecutor(max_workers=build_workers) as pool:
            futures = [pool.submit(_decode_one, ep, cam) for ep, cam in tasks]
            for i, fut in enumerate(as_completed(futures)):
                total_decoded += fut.result()
                if (i + 1) % 200 == 0 or (i + 1) == len(futures):
                    logger.info(f"[video-cache] decoded {i + 1}/{len(futures)} (ep, cam) pairs")
        elapsed = time.perf_counter() - t0
        logger.info(
            f"[video-cache] build complete: {total_decoded} frames across {len(self.camera_keys)} cams "
            f"in {elapsed:.1f}s ({total_decoded / max(elapsed, 1e-6):.0f} frames/s)"
        )

    def lookup(self, query_timestamps: dict, ep_idx: int) -> dict:
        """Drop-in replacement for LeRobotDataset._query_videos."""
        out = {}
        ep_idx = int(ep_idx)
        ep_len = self.ep_lengths[ep_idx]
        offset = self.ep_offsets[ep_idx]
        for vid_key, query_ts in query_timestamps.items():
            buf = self.buffers[vid_key]
            indices = np.asarray(query_ts, dtype=np.float64) * self.fps
            indices = np.clip(np.round(indices), 0, ep_len - 1).astype(np.int64)
            frames_f32 = np.ascontiguousarray(buf[offset + indices])  # [T, H, W, 3] float32 [0,1]
            # Return float [T, 3, H, W] in [0,1] at processed resolution. _get_image
            # passes float through; ToTensor is a no-op on float -> no per-step
            # resize and no uint8<->float conversions.
            frames = torch.from_numpy(frames_f32).permute(0, 3, 1, 2).contiguous()
            out[vid_key] = frames
        return out


def _install_video_cache(base_lerobot_dataset, cache_dir, build_workers=16, target_hw=None) -> list:
    """Build/open a `_DiskVideoFrameCache` for each inner LeRobotDataset and patch
    `_query_videos` to read from it. Returns the cache objects to keep references alive."""
    caches = []
    for inner in base_lerobot_dataset.multi_dataset._datasets:
        camera_keys = list(inner.meta.video_keys)
        cache = _DiskVideoFrameCache(inner, camera_keys, cache_dir, build_workers=build_workers, target_hw=target_hw)
        inner._query_videos = cache.lookup  # type: ignore[assignment]
        caches.append(cache)
    return caches


class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        text_embedding_fallback_cache_dirs=None,
        text_prompt_template: Optional[str] = DEFAULT_PROMPT,
        text_embedding_pack_dir=None,
        text_context_cache_size: int = 0,
        context_len=128,
        pretrained_norm_stats=None,
        norm_stats_weighting: str = "legacy",
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        lazy_parquet_loading: bool = False,
        video_backend: Optional[str] = None,
        content_cache_dir: Optional[str] = None,
        content_cache_manifest: Optional[str] = None,
        content_cache_max_bytes: int = 0,
        action_video_freq_ratio: int = 1,
        decode_video_sampled_frames: bool = False,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", separate-view layouts, or None
        control_representation: Optional[str] = None,
        libero_gripper_open_width: Optional[float] = None,
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        object_conditioned_instruction_manifest: Optional[str] = None,
        object_conditioned_instruction_sampling: str = "per_draw",
        object_conditioned_instruction_seed: int = 0,
        sample_unique_episode_prompts: bool = False,
        preload_video_cache: bool = False, # decode all videos once into an on-disk uint8 memmap; serve frames with zero decode
        preload_cache_workers: int = 16,
        video_cache_dir: Optional[str] = None, # where the decoded-frame memmaps live (required if preload_video_cache)
        goal_supervision: bool = False, # GoalWAM: also emit the trajectory terminal frame as `goal_image`
        merged_video_cache_dir: Optional[str] = None, # pre-merged float cache: decode+resize+concat baked offline; train-time load = memmap read (zero decode). Mutually exclusive with preload_video_cache.
        merged_video_cache_dtype: str = "float32", # float32 v2 or bit-exact model-input bfloat16 v3.
        video_latent_cache_dir: Optional[str] = None, # frozen-VAE BF16 latent mmap keyed by native sample index; requires the merged cache contract.
        episode_subset=None, # optional episode whitelist (list[int] or path to JSON list); restricts the candidate pool before the train/val split. Single dataset only.
        episode_exclusion_manifest=None, # optional v1 per-root episode denylist; applied before train/val split.
        frame_exclusion_manifest=None, # optional v1 per-root/per-episode frame denylist for training anchors.
        episode_segment_manifest=None, # optional v1 per-root logical sub-episode/caption manifest.
        mask_bad_segment_action_loss: bool = False,
        task_weighted_sampling=None, # optional exact-root per-task replacement sampling; null preserves historical order.
        temporal_resampling=None, # optional exact-root/rule-routed virtual timeline policy
        sampling_temporal_resampling=None, # optional slow-shaped anchor population over identity/native windows
        action_tokenizer_source_rules=None, # optional root -> ActionDiT I/O adapter routing
        camera_intrinsics_warp=None, # optional per-episode RGB warp to a shared virtual K
        eef_anchor_overlay=None, # optional text-free canonical EEF axes on RGB frame 0 only
        emit_camera_pose_delta: bool = False, # emit T_C0<-Ci aligned to sampled RGB frames
        camera_pose_delta_source: str = "trajectory", # trajectory metadata or an explicit fixed-camera identity route
        action_time_offset_frames: int = 0, # native-frame offset for action queries; EgoStandard uses +1
        video_timestamp_tolerance_s: float = 1e-4,
        require_full_horizon: bool = False, # expose only anchors whose state/action/video windows need no padding
    ):
        norm_stats_weighting = str(norm_stats_weighting).lower()
        self.mask_bad_segment_action_loss = bool(mask_bad_segment_action_loss)
        self.norm_stats_weighting = norm_stats_weighting
        self.text_prompt_template = text_prompt_template
        self.action_tokenizer_source_routes = _build_action_tokenizer_source_routes(
            dataset_dirs,
            action_tokenizer_source_rules,
        )
        resolved_shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        resolved_temporal_resampling = (
            OmegaConf.to_container(temporal_resampling, resolve=True)
            if OmegaConf.is_config(temporal_resampling)
            else temporal_resampling
        )
        resolved_sampling_temporal_resampling = (
            OmegaConf.to_container(sampling_temporal_resampling, resolve=True)
            if OmegaConf.is_config(sampling_temporal_resampling)
            else sampling_temporal_resampling
        )
        sampling_specs = None
        sampling_provenance = None
        if resolved_sampling_temporal_resampling is not None:
            aligned_sampling_specs = _resolve_temporal_resampling_by_root(
                list(dataset_dirs),
                resolved_sampling_temporal_resampling,
            )
            sampling_specs = tuple(
                parse_temporal_resampling_spec(spec)
                for spec in aligned_sampling_specs
            )
            sampling_provenance = _expanded_temporal_resampling_provenance(
                list(dataset_dirs),
                list(sampling_specs),
            )
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=resolved_shape_meta,
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            lazy_parquet_loading=lazy_parquet_loading,
            video_backend=video_backend,
            content_cache_dir=content_cache_dir,
            content_cache_manifest=content_cache_manifest,
            content_cache_max_bytes=content_cache_max_bytes,
            episode_subset=episode_subset,
            episode_exclusion_manifest=episode_exclusion_manifest,
            episode_segment_manifest=episode_segment_manifest,
            temporal_resampling=resolved_temporal_resampling,
        action_time_offset_frames=action_time_offset_frames,
            video_timestamp_tolerance_s=video_timestamp_tolerance_s,
        )
        self.object_conditioned_instruction_sampling = str(
            object_conditioned_instruction_sampling
        ).strip().lower()
        self.object_conditioned_instruction_seed = int(
            object_conditioned_instruction_seed
        )
        self._instruction_epoch = 0
        self._instruction_draw_ordinal = 0
        self.object_conditioned_instruction_sampler = None
        if object_conditioned_instruction_manifest is not None:
            self.object_conditioned_instruction_sampler = (
                ObjectConditionedInstructionSampler(
                    object_conditioned_instruction_manifest,
                    list(dataset_dirs),
                )
            )
            logger.info(
                "Object-conditioned runtime instruction sampling enabled: "
                "manifest=%s sha256=%s prompts=%s sources=%s",
                self.object_conditioned_instruction_sampler.manifest_path,
                self.object_conditioned_instruction_sampler.manifest_sha256,
                f"{self.object_conditioned_instruction_sampler.unique_prompt_count:,}",
                self.object_conditioned_instruction_sampler.source_ids,
            )
            logger.info(
                "Object-conditioned prompt draw policy: sampling=%s seed=%d "
                "runtime_pool_sizes=%s "
                "(a fresh compatible prompt is selected for every dataset access)",
                self.object_conditioned_instruction_sampling,
                self.object_conditioned_instruction_seed,
                self.object_conditioned_instruction_sampler.runtime_episode_pool_sizes_by_source,
            )
        self.episode_instruction_sampler = None
        if sample_unique_episode_prompts:
            children = self.lerobot_dataset.multi_dataset._datasets
            self.episode_instruction_sampler = EpisodeInstructionSampler(
                list(dataset_dirs),
                [child.episodes for child in children],
            )
            logger.info(
                "Unique episode-prompt runtime sampling enabled: "
                "pools=%s prompts=%s pool_size=[%s,%s] sources=%s",
                f"{self.episode_instruction_sampler.pool_count:,}",
                f"{self.episode_instruction_sampler.unique_prompt_count:,}",
                self.episode_instruction_sampler.min_pool_size,
                self.episode_instruction_sampler.max_pool_size,
                self.episode_instruction_sampler.source_ids,
            )
        self.frame_exclusion_manifest, frame_exclusions = _load_frame_exclusion_manifest(
            frame_exclusion_manifest, list(dataset_dirs)
        )
        self._frame_exclusion_enabled = bool(frame_exclusions)
        self._frame_valid_indices = None
        if self._frame_exclusion_enabled:
            global_excluded = []
            offset = 0
            for root, child in zip(dataset_dirs, self.lerobot_dataset.multi_dataset._datasets, strict=True):
                root_key = str(Path(root).expanduser().resolve())
                for episode_pos, source_episode in enumerate(child.episodes or range(child.num_episodes)):
                    length = int(child.episode_data_index["to"][episode_pos] - child.episode_data_index["from"][episode_pos])
                    excluded = frame_exclusions[root_key].get(int(source_episode), frozenset())
                    global_excluded.extend(offset + int(i) for i in excluded)
                    offset += length
            excluded_set = set(global_excluded)
            self._frame_valid_indices = np.asarray(
                [i for i in range(len(self.lerobot_dataset)) if i not in excluded_set],
                dtype=np.int64,
            )
            logger.info(
                "Using frame exclusion manifest %s: excluded=%s/%s native anchors",
                self.frame_exclusion_manifest,
                f"{len(excluded_set):,}",
                f"{len(self.lerobot_dataset):,}",
            )
        self.require_full_horizon = bool(require_full_horizon)
        self._full_horizon_index = None
        if self.require_full_horizon:
            incompatible = []
            if self._frame_exclusion_enabled:
                incompatible.append("frame_exclusion_manifest")
            if sampling_specs is not None:
                incompatible.append("sampling_temporal_resampling")
            if self.lerobot_dataset.environment_sampling is not None:
                incompatible.append("environment_sampling")
            max_observation_delta = (int(num_frames) - 1) * int(global_sample_stride)
            max_action_delta = (
                (int(num_frames) - 2) * int(global_sample_stride)
                + int(action_time_offset_frames)
            )
            min_query_delta = min(0, int(action_time_offset_frames))
            max_query_delta = max(max_observation_delta, max_action_delta)
            self._full_horizon_index = _FullHorizonIndex(
                episode_starts=self.lerobot_dataset.episode_data_index["from"],
                episode_ends=self.lerobot_dataset.episode_data_index["to"],
                min_query_delta=min_query_delta,
                max_query_delta=max_query_delta,
            )
            logger.info(
                "Full-horizon anchor filtering enabled: valid=%s native=%s "
                "episodes=%s eligible_episodes=%s query_delta=[%s,%s]",
                f"{len(self._full_horizon_index):,}",
                f"{self._full_horizon_index.native_num_frames:,}",
                f"{self._full_horizon_index.num_episodes:,}",
                f"{self._full_horizon_index.eligible_episodes:,}",
                min_query_delta,
                max_query_delta,
            )
        self.environment_sampling = self.lerobot_dataset.environment_sampling
        resolved_task_weighted_sampling = (
            OmegaConf.to_container(task_weighted_sampling, resolve=True)
            if OmegaConf.is_config(task_weighted_sampling)
            else task_weighted_sampling
        )
        children = self.lerobot_dataset.multi_dataset._datasets
        self.task_weighted_sampling = build_task_weighted_sampling_contract(
            dataset_dirs=list(dataset_dirs),
            root_capacities=[int(child.num_frames) for child in children],
            config=resolved_task_weighted_sampling,
        )
        if self.task_weighted_sampling is not None:
            logger.info(
                "Task-weighted replacement sampling enabled: native_samples=%s "
                "effective_samples=%s padding_equivalent_extra=%s "
                "global_batch_size=%s total_integer_mass=%s sha256=%s tasks=%s",
                f"{self.task_weighted_sampling['native_epoch_samples']:,}",
                f"{self.task_weighted_sampling['effective_epoch_samples']:,}",
                f"{self.task_weighted_sampling['padding_equivalent_extra']:,}",
                f"{self.task_weighted_sampling['global_batch_size']:,}",
                f"{self.task_weighted_sampling['total_integer_mass']:,}",
                self.task_weighted_sampling["sha256"],
                [
                    {
                        "task_id": entry["task_id"],
                        "capacity": entry["capacity"],
                        "integer_weight": entry["integer_weight"],
                        "expected_task_probability": entry[
                            "expected_task_probability"
                        ],
                    }
                    for entry in self.task_weighted_sampling["tasks"]
                ],
            )
        selected_episode_indices = None
        if camera_intrinsics_warp is not None or eef_anchor_overlay is not None:
            selected_episode_indices = []
            for child in self.lerobot_dataset.multi_dataset._datasets:
                logical_sources = getattr(child, "_logical_source_episodes", None)
                if logical_sources is not None:
                    selected_episode_indices.append(logical_sources)
                elif child.episodes is not None:
                    selected_episode_indices.append(child.episodes)
                else:
                    selected_episode_indices.append(range(child.num_episodes))
        self.camera_intrinsics_warp = None
        if camera_intrinsics_warp is not None:
            warp_config = (
                OmegaConf.to_container(camera_intrinsics_warp, resolve=True)
                if OmegaConf.is_config(camera_intrinsics_warp)
                else copy.deepcopy(camera_intrinsics_warp)
            )
            self.camera_intrinsics_warp = CameraIntrinsicsWarp(
                dataset_dirs=list(dataset_dirs),
                selected_episode_indices=selected_episode_indices or [],
                video_size=video_size,
                **warp_config,
            )
        self.eef_anchor_overlay = None
        if eef_anchor_overlay is not None:
            overlay_config = (
                OmegaConf.to_container(eef_anchor_overlay, resolve=True)
                if OmegaConf.is_config(eef_anchor_overlay)
                else copy.deepcopy(eef_anchor_overlay)
            )
            self.eef_anchor_overlay = EefAnchorOverlay(
                dataset_dirs=list(dataset_dirs),
                selected_episode_indices=selected_episode_indices or [],
                video_size=video_size,
                **overlay_config,
            )
        self._action_tokenizer_source_by_dataset_index = None
        self._action_tokenizer_source_frame_ends = ()
        if self.action_tokenizer_source_routes is not None:
            sources = []
            frame_ends = []
            total_frames = 0
            children = self.lerobot_dataset.multi_dataset._datasets
            for dataset_root, child in zip(dataset_dirs, children, strict=True):
                resolved_root = str(Path(str(dataset_root)).expanduser().resolve())
                sources.append(self.action_tokenizer_source_routes[resolved_root])
                total_frames += int(child.num_frames)
                frame_ends.append(total_frames)
            self._action_tokenizer_source_by_dataset_index = tuple(sources)
            self._action_tokenizer_source_frame_ends = tuple(frame_ends)

        aligned_temporal_specs = getattr(
            self.lerobot_dataset.multi_dataset,
            "temporal_resampling",
            getattr(self.lerobot_dataset, "temporal_resampling", None),
        )
        temporal_resampling_enabled = bool(
            getattr(
                self.lerobot_dataset.multi_dataset,
                "temporal_resampling_enabled",
                aligned_temporal_specs is not None
                and any(
                    not is_identity_temporal_resampling(spec)
                    for spec in aligned_temporal_specs
                ),
            )
        )
        # Preserve old configs and their pinned statistics bit-for-bit when the
        # option is absent.  New native/runtime-slow configs declare even identity
        # rules explicitly, which opts them into strict temporal provenance.
        temporal_provenance = None
        if resolved_temporal_resampling is not None:
            temporal_provenance = _expanded_temporal_resampling_provenance(
                list(dataset_dirs),
                aligned_temporal_specs
                if aligned_temporal_specs is not None
                else [None] * len(dataset_dirs),
            )
        strict_temporal_provenance = (
            resolved_temporal_resampling is not None
            or resolved_sampling_temporal_resampling is not None
            or episode_segment_manifest is not None
        )
        if strict_temporal_provenance and temporal_provenance is None:
            temporal_provenance = _expanded_temporal_resampling_provenance(
                list(dataset_dirs),
                aligned_temporal_specs
                if aligned_temporal_specs is not None
                else [None] * len(dataset_dirs),
            )
        if temporal_resampling_enabled:
            incompatible = []
            if preload_video_cache:
                incompatible.append("preload_video_cache")
            if merged_video_cache_dir:
                incompatible.append("merged_video_cache_dir")
            if goal_supervision:
                incompatible.append("goal_supervision")

        if episode_segment_manifest is not None:
            incompatible = []
            if merged_video_cache_dir:
                incompatible.append("merged_video_cache_dir")
            if goal_supervision:
                incompatible.append("goal_supervision")
        self.sampling_temporal_resampling = (
            [canonicalize_temporal_resampling_spec(spec) for spec in sampling_specs]
            if sampling_specs is not None
            else None
        )
        self.sampling_temporal_resampling_provenance = sampling_provenance
        self._sampling_temporal_index = None
        if sampling_specs is not None:
            self._sampling_temporal_index = _SamplingOnlyTemporalIndex(
                dataset_dirs=list(dataset_dirs),
                base_dataset=self.lerobot_dataset,
                specs=sampling_specs,
            )
            logger.info(
                "Sampling-only temporal view: roots=%s episodes=%s native_frames=%s "
                "logical_sampling_frames=%s",
                len(dataset_dirs),
                f"{self._sampling_temporal_index.num_episodes:,}",
                f"{self._sampling_temporal_index.native_num_frames:,}",
                f"{self._sampling_temporal_index.num_frames:,}",
            )
        self.sampling_temporal_resampling_enabled = bool(
            self._sampling_temporal_index is not None
            and self._sampling_temporal_index.enabled
        )
        self.sampling_temporal_resampling_corpus = (
            self._sampling_temporal_index.corpus
            if self._sampling_temporal_index is not None
            else None
        )

        self._video_caches = []
        if preload_video_cache:
            # Bake the raw->processed downsize into the cache. All cameras share
            # the processed shape ([C, H, W]); store frames at that (H, W) so the
            # per-step resize from raw resolution is skipped. Requires uniform
            # processed shape across cameras.
            sm = OmegaConf.to_container(shape_meta, resolve=True)
            proc_shapes = {tuple(im["shape"][-2:]) for im in sm["images"]}
            target_hw = next(iter(proc_shapes)) if len(proc_shapes) == 1 else None
            self._video_caches = _install_video_cache(
                self.lerobot_dataset,
                cache_dir=video_cache_dir,
                build_workers=preload_cache_workers,
                target_hw=target_hw,
            )

        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        self.control_representation = control_representation
        self.libero_gripper_open_width = libero_gripper_open_width
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.context_len = int(context_len)
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self._text_embedding_cache_dir_ready = False
        if text_embedding_fallback_cache_dirs is None:
            text_embedding_fallback_cache_dirs = []
        elif isinstance(text_embedding_fallback_cache_dirs, (str, os.PathLike)):
            text_embedding_fallback_cache_dirs = [text_embedding_fallback_cache_dirs]
        self.text_embedding_fallback_cache_dirs = [
            str(path) for path in text_embedding_fallback_cache_dirs
        ]
        self.text_embedding_pack_dir = (
            str(text_embedding_pack_dir)
            if text_embedding_pack_dir is not None
            else None
        )
        self._text_embedding_pack = (
            TextEmbeddingPack(
                self.text_embedding_pack_dir,
                context_len=self.context_len,
            )
            if self.text_embedding_pack_dir is not None
            else None
        )
        self.text_context_cache_size = int(text_context_cache_size)
        self._reset_text_context_cache_state()
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.goal_supervision = bool(goal_supervision)
        self.emit_camera_pose_delta = bool(emit_camera_pose_delta)
        self.camera_pose_delta_source = str(camera_pose_delta_source).strip().lower()
        if self.camera_intrinsics_warp is not None:
            incompatible = []
            if self.concat_multi_camera is not None:
                incompatible.append("concat_multi_camera")
            if self.goal_supervision:
                incompatible.append("goal_supervision")
            if eef_anchor_overlay is not None:
                incompatible.append("eef_anchor_overlay")

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if self.eef_anchor_overlay is not None:
                processor.emit_transformed_anchor_state = True
                processor.anchor_state_key = self.eef_anchor_overlay.state_key
            if not pretrained_norm_stats:
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(
                        processor,
                        norm_stats_weighting=self.norm_stats_weighting,
                        full_horizon_query_deltas=(
                            (self._full_horizon_index.min_query_delta, self._full_horizon_index.max_query_delta)
                            if self._full_horizon_index is not None else None
                        ),
                    )
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

        # --- pre-merged float video cache (opt-in) -------------------------------
        # Bakes decode + per-cam resize + multi-cam merge offline; at train time
        # __getitem__ just memmaps the already-merged [-1,1] float frames (zero
        # decode), eliminating the 3-camera pyav bottleneck. Strictly equivalent to
        # the on-the-fly path (built from _assemble_frames); malformed AV1 tails
        # must first be removed from local metadata/parquet. Default: disabled.
        self._merged_cache = None
        self._merged_caches = []
        self._merged_inners = []
        self._merged_buf = None
        self._video_latent_cache = None
        if merged_video_cache_dir:
            from .merged_video_cache import (
                BFloat16MergedFrameCache,
                MergedFrameCache,
                TShapeMp4FrameCache,
                image_transforms_signature)
            merged_dtype = str(merged_video_cache_dtype).strip().lower()
            cache_types = {
                "float32": MergedFrameCache,
                "f32": MergedFrameCache,
                "bfloat16": BFloat16MergedFrameCache,
                "bf16": BFloat16MergedFrameCache,
                "tshape_mp4": TShapeMp4FrameCache,
            }
            cache_type = cache_types[merged_dtype]
            selected_camera_keys = [
                str(meta["lerobot_key"]) for meta in self.lerobot_dataset.image_meta
            ]
            processor_transforms = (
                self.lerobot_dataset.processor.train_transforms
                if self.lerobot_dataset.processor.is_train
                else self.lerobot_dataset.processor.val_transforms
            )
            transform_signature = image_transforms_signature(
                processor_transforms,
                selected_camera_keys,
            )
            cache_hw = tuple(self.video_size)
            if self.concat_multi_camera in {
                "robotwin_latent_tshape", "robotwin_multiview_tokens", "robotwin_packed_multiview",
            }:
                # Cache three full-resolution views; restore the view dimension
                # before the existing model-side wrist resize and separate VAEs.
                # Include the separate-view layout in the cache signature.
                view_hw = tuple(resolved_shape_meta["images"][0]["shape"][-2:])
                cache_hw = (3 * view_hw[0], view_hw[1])
                transform_signature += ":separate_views_vertical_v1"
                if getattr(cache_type, "PREMERGED_T", False):
                    # Head over left|right wrist at half resolution: [256+128, 352].
                    cache_hw = (view_hw[0] + view_hw[0] // 2, view_hw[1])
            subs = self.lerobot_dataset.multi_dataset._datasets
            total_episodes = 0
            for inner in subs:
                cache = cache_type(
                    inner,
                    out_hw=cache_hw,
                    cache_dir=merged_video_cache_dir,
                    camera_keys=selected_camera_keys,
                    transform_signature=transform_signature,
                )
                def _merged_query(query_timestamps, ep_idx, _cache=cache):
                    # Stash only the frames the model consumes. LeRobot still requests
                    # the full 33-frame action-aligned window, but RoboTwin later
                    # selects [0,4,...,32]; selecting before the mmap read cuts cache
                    # I/O by 3.7x. Dataset workers call __getitem__ serially within a
                    # process, so one process-local scratch slot is sufficient even
                    # when the corpus contains many underlying roots.
                    if self._video_latent_cache is None:
                        self._merged_buf = _cache.lookup_window(
                            query_timestamps,
                            ep_idx,
                            frame_positions=self.video_sample_indices,
                        )  # [T_video,3,Ho,Wo]
                    else:
                        # The sample-indexed latent cache is the final training input.
                        # Keep LeRobot's metadata/action query intact but avoid touching
                        # any pixel mmap pages.
                        self._merged_buf = None
                    # Return tiny placeholders plus a marker. FastWAMProcessor skips
                    # image transforms/stacking for this marker because _get consumes
                    # the stashed merged buffer directly. Using full 240x320 dummies
                    # would otherwise allocate and discard ~91 MB per sample.
                    T = len(next(iter(query_timestamps.values())))
                    dummy = torch.zeros(T, 3, 1, 1)
                    result = {k: dummy for k in query_timestamps}
                    result["_fastwam_merged_video_cache"] = True
                    return result

                inner._query_videos = _merged_query  # type: ignore[assignment]
                self._merged_caches.append(cache)
                self._merged_inners.append(inner)
                total_episodes += len(cache.episodes)

            # Preserve the old singular attribute for compatibility with diagnostics
            # that only inspect whether the merged path is active.
            self._merged_cache = self._merged_caches[0]
            logger.info(
                f"[merged-cache] active: {len(self._merged_caches)} roots / "
                f"{total_episodes} episodes under {merged_video_cache_dir} "
                f"(out {self.video_size[0]}x{self.video_size[1]})"
            )

        if decode_video_sampled_frames or os.environ.get('FASTWAM_SPARSE_VIDEO_DECODE') == '1':
            from .sparse_video_decode import install_sparse_video_decode
            install_sparse_video_decode(self)
            logger.info('[video-decode] decoding %d of %d observation frames; action/state unchanged',
                        len(self.video_sample_indices), self.num_frames)

    def __len__(self):
        if self._frame_valid_indices is not None:
            return len(self._frame_valid_indices)
        if self._full_horizon_index is not None:
            return len(self._full_horizon_index)
        if self._sampling_temporal_index is not None:
            return len(self._sampling_temporal_index)
        return len(self.lerobot_dataset)

    def _sampling_index_to_native(self, idx):
        if self._frame_valid_indices is not None:
            idx = int(idx)
            idx = int(self._frame_valid_indices[idx])
        if self._full_horizon_index is not None:
            idx = self._full_horizon_index.map_index(idx)
        if self._sampling_temporal_index is None:
            return idx
        return self._sampling_temporal_index.map_index(idx)

    def _assemble_frames(self, multicam: torch.Tensor) -> torch.Tensor:
        """Concat cameras + final resize/crop/normalize a frame stack.

        Input:  [num_cameras, T, C, H, W]
        Output:
          - regular layouts: [C, T, H, W] in [-1, 1] at video_size
          - separate-view RoboTwin layouts: [V, C, T, H, W] in [-1, 1], with
            each camera kept separate for per-view resize and VAE encoding.
        """
        num_cameras = multicam.shape[0]
        if self.concat_multi_camera in {
            "robotwin_latent_tshape",
            "robotwin_multiview_tokens",
            "robotwin_packed_multiview",
        }:
            video = self.normalize_transform(multicam)  # [V, T, C, H, W]
            return video.permute(0, 2, 1, 3, 4).contiguous()  # [V, C, T, H, W]
        if self.concat_multi_camera == "robotwin_ego_aligned_tshape":
            # Pixel T-shape whose head keeps the Ego-aligned 256x448 contract and
            # whose wrists are half the head's height and width.  Geometrically
            # identical to `robotwin_multiview_tokens`, but composed into a single
            # canvas so the VAE encode, RoPE and token grid stay on the validated
            # pixel path.  The head is cover-resized from the 240x320 processor
            # output, so it is softer than the single-view 256x448 run, which
            # crops from the raw 480x640 frame.
            cam_top = cover_resize_center_crop(multicam[0], (256, 448))
            cam_left = transforms_F.resize(
                multicam[1], size=[128, 224],
                interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True,
            )
            cam_right = transforms_F.resize(
                multicam[2], size=[128, 224],
                interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T, C, 128, 448]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T, C, 384, 448]
        elif self.concat_multi_camera == "robotwin":
            cam_top = transforms_F.resize(
                multicam[0], size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True,
            )  # [T, C, 256, 320]
            cam_left = transforms_F.resize(
                multicam[1], size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True,
            )
            cam_right = transforms_F.resize(
                multicam[2], size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T, C, 384, 320]
        elif self.concat_multi_camera == "robotwin_tshape_528x448":
            # Large pixel T-shape requested for the Clean-50 ablation:
            # 352x448 head over two 176x224 wrist views -> 528x448 canvas.
            # The per-camera processor emits a common 240x320 tensor; resize
            # each view here so the concatenation is deterministic and does not
            # rely on the final aspect-preserving resize/crop path.
            cam_top = transforms_F.resize(
                multicam[0], size=[352, 448],
                interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True,
            )
            cam_left = transforms_F.resize(
                multicam[1], size=[176, 224],
                interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True,
            )
            cam_right = transforms_F.resize(
                multicam[2], size=[176, 224],
                interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T, C, 176, 448]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T, C, 528, 448]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([multicam[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([multicam[i] for i in range(num_cameras)], dim=-2)
        else:
            video = multicam.squeeze(0)  # [T, C, H, W]

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T, C, H, W]
        return video.permute(1, 0, 2, 3)  # [C, T, H, W]

    def _episode_terminal_goal_image(self, sample_idx: int) -> torch.Tensor:
        """Fetch the terminal RGB frame of the episode this sample belongs to, run it
        through the identical camera-concat + resize pipeline, and return [C, H, W].

        Reuses the full processor/cache pipeline by re-reading the episode's last global
        frame index (`to[ep]-1`); its query window starts at that frame (delta_timestamps
        begin at 0), so index 0 of the returned clip is the terminal frame.
        """
        sample_idx = int(sample_idx)
        to_idx = self.lerobot_dataset.episode_data_index["to"]
        ep = int(torch.searchsorted(to_idx, torch.tensor(sample_idx), right=True))
        if self._merged_cache is not None:
            remaining = sample_idx
            for inner, cache in zip(
                self._merged_inners,
                self._merged_caches,
                strict=True,
            ):
                if remaining < inner.num_frames:
                    _, ep_id, _ = inner._locate_virtual_index(remaining)
                    ep_len = cache.ep_lengths[int(ep_id)]
                    return cache.lookup_frame(int(ep_id), ep_len - 1)  # [C, H, W] in [-1, 1]
                remaining -= inner.num_frames
        terminal_idx = int(to_idx[ep]) - 1

        term = self.lerobot_dataset[terminal_idx]
        pix = term["pixel_values"]  # [num_cam, T, C, H, W] or [T, C, H, W]
        if pix.ndim == 5:
            num_cameras = pix.shape[0]
            multicam = pix[:, 0:1]  # terminal frame is query index 0 -> [num_cam, 1, C, H, W]
        else:
            num_cameras = 1
            multicam = pix[0:1].unsqueeze(0)  # [1, 1, C, H, W]
        goal = self._assemble_frames(multicam.reshape(num_cameras, 1, *pix.shape[-3:]))
        return goal[:, 0]  # [C, H, W], range [-1, 1]

    def _action_tokenizer_source_for_sample_index(
        self, sample_idx: int
    ) -> Optional[str]:
        """Resolve an exact corpus root route from the sampled global frame index.

        This deliberately does not depend on processor-retained metadata. Some ego
        roots do not expose ``cam_extrinsic``, so ``wrist_source_path`` is not always
        stamped even though the root identity is still unambiguous in the concatenated
        LeRobot dataset.
        """

        sources = self._action_tokenizer_source_by_dataset_index
        if sources is None:
            return None
        index = int(sample_idx)
        for dataset_index, frame_end in enumerate(
            self._action_tokenizer_source_frame_ends
        ):
            if index < frame_end:
                return sources[dataset_index]

    # Wan2.2 causal VAE temporal downsample (frames1-4 -> 1 latent). Combined with the
    # dataset's own temporal subsampling it converts a trajectory-frame distance into
    # latent (RoPE) index units for the long-goal position encoding (GoalWAM exp5).
    _VAE_TEMPORAL_DOWNSAMPLE = 4

    def _goal_long_rope_index(self, sample_idx: int) -> int:
        """Latent-index distance from the current chunk's last frame to the episode's
        terminal frame, for the long-goal RoPE position (GoalWAM exp5).

        Δframes = (episode last global frame) − (chunk last global frame); divided by
        the trajectory-frames-per-latent compression
            comp = global_sample_stride × action_video_freq_ratio × VAE_temporal(4)
        (= 16 for RoboTwin: stride 1 × freq 4 × VAE 4). Clamped to ≥ 0.
        """
        sample_idx = int(sample_idx)
        to_idx = self.lerobot_dataset.episode_data_index["to"]
        ep = int(torch.searchsorted(to_idx, torch.tensor(sample_idx), right=True))
        ep_last = int(to_idx[ep]) - 1
        stride = int(self.lerobot_dataset.global_sample_stride)
        chunk_last = min(sample_idx + (self.num_frames - 1) * stride, ep_last)
        delta = max(0, ep_last - chunk_last)
        comp = stride * int(self.action_video_freq_ratio) * self._VAE_TEMPORAL_DOWNSAMPLE
        return int(round(delta / comp))

    def _get(self, idx):
        sample_idx = self._sampling_index_to_native(idx)
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            # Retry in the public/logical index space so excluded anchors can
            # never be reintroduced by padding recovery.
            logical_retry = np.random.randint(len(self))
            sample_idx = self._sampling_index_to_native(logical_retry)

        image_is_pad = sample["image_is_pad"]

        video_latents = None
        if self._video_latent_cache is not None:
            image_is_pad = image_is_pad[self.video_sample_indices]
            video_latents = self._video_latent_cache.lookup(sample_idx)
            num_video_frames = len(self.video_sample_indices)
            video = None
        elif self._merged_cache is not None:
            # Pre-merged path: the query hook already selected video_sample_indices,
            # so the stashed buffer is [T_video, 3, Ho, Wo] in [-1, 1].
            merged_win = self._merged_buf
            image_is_pad = image_is_pad[self.video_sample_indices]
            video = merged_win.permute(1, 0, 2, 3).contiguous()  # [C, T_video, Ho, Wo]
            if self.concat_multi_camera in {
                "robotwin_latent_tshape", "robotwin_multiview_tokens", "robotwin_packed_multiview",
            } and not getattr(self._merged_cache, "PREMERGED_T", False):
                c, t, packed_h, w = video.shape
                video = video.reshape(c, t, 3, packed_h // 3, w).permute(2, 0, 1, 3, 4).contiguous()
        else:
            video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
            sparse_indices = sample.get('_fastwam_video_sample_indices')
            indices = slice(None) if sparse_indices is not None else self.video_sample_indices
            num_cameras = 1
            if video.ndim == 5:
                video = video[:, indices, :, :, :] # [num_cameras, T_video, C, H, W]
                num_cameras, T_video, C, H, W = video.shape
            else:
                video = video[indices, :, :, :] # [T_video, C, H, W]
                T_video, C, H, W = video.shape
            image_is_pad = image_is_pad[indices]

            video = self._assemble_frames(
                video.view(num_cameras, T_video, C, H, W)
            )  # [C,T,H,W] or [V,C,T,H,W], range [-1, 1]

        episode_key = None
        if self.camera_intrinsics_warp is not None or self.eef_anchor_overlay is not None:
            resolved_sample_idx = int(sample.get("idx", sample_idx))
            episode_key = self.lerobot_dataset._runtime_episode_key(resolved_sample_idx)

        if self.camera_intrinsics_warp is not None:
            dataset_index, episode_index = episode_key
            video = self.camera_intrinsics_warp.apply(
                video,
                dataset_index=dataset_index,
                episode_index=episode_index,
            )

        if self.eef_anchor_overlay is not None:
            anchor_state = sample.get("_eef_anchor_state")
            dataset_index, episode_index = episode_key
            self.eef_anchor_overlay.apply(
                video,
                anchor_state,
                dataset_index=dataset_index,
                episode_index=episode_index,
                operating_hand=sample.get("_eef_anchor_operating_hand", "both"),
            )

        # uint8-on-GPU: ship the video to the trainer as uint8 (4x smaller IPC)
        # and let the model decode [0,255]->[-1,1] on GPU. Opt-in via env.
        if video is not None and os.environ.get("UINT8_VIDEO", "0") == "1":
            video = ((video + 1.0) * 127.5).round().clamp_(0, 255).to(torch.uint8)

        # Proxy (from lerobot):
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video is not None:
            video_time_dim = 2 if video.ndim == 5 else 1
            num_video_frames = int(video.shape[video_time_dim])

        task = self._instruction_task_for_sample(sample, sample_idx)
        instruction = (
            task
            if self.text_prompt_template is None
            else self.text_prompt_template.format(task=task)
        )

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        camera_pose_delta = None
        camera_pose_valid = None
        if self.emit_camera_pose_delta:
            camera_pose_valid = ~torch.as_tensor(image_is_pad, dtype=torch.bool)
            if self.camera_pose_delta_source == "static_identity":
                # RoboTwin's stored head camera is fixed for the full trajectory.
                # Emit the condition explicitly instead of fabricating an absolute
                # calibration column or silently treating missing metadata as fixed.
                camera_pose_delta = torch.eye(4, dtype=torch.float32).repeat(
                    num_video_frames, 1, 1
                )
            else:
                camera_extrinsic = sample.get("cam_extrinsic")
                camera_extrinsic = torch.as_tensor(camera_extrinsic)
                sampled_camera = camera_extrinsic[self.video_sample_indices]
                camera_pose_delta = camera_to_world_pose7_to_delta(
                    sampled_camera,
                    camera_pose_valid,
                ).float()

        data = {
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
            "_fastwam_merged_video_cache": self._merged_cache is not None,
            "_fastwam_video_latent_cache": self._video_latent_cache is not None,
            "_fastwam_pyav_calls": (
                0
                if self._merged_cache is not None or self._video_latent_cache is not None
                else 1
            ),
            "_fastwam_sample_index": int(sample.get("idx", sample_idx)),
        }
        if camera_pose_delta is not None:
            data["camera_pose_delta"] = camera_pose_delta
            data["camera_pose_valid"] = camera_pose_valid
        if video_latents is not None:
            data["video_latents"] = video_latents
        else:
            data["video"] = video
        if self.concat_multi_camera in {
            "robotwin_latent_tshape",
            "robotwin_multiview_tokens",
            "robotwin_packed_multiview",
        }:
            data["video_layout"] = self.concat_multi_camera
        if self.action_tokenizer_source_routes is not None:
            resolved_sample_idx = int(sample.get("idx", sample_idx))
            source = self._action_tokenizer_source_for_sample_index(
                resolved_sample_idx
            )
            data["action_tokenizer_source"] = source
        for key in ("video_view_present", "video_view_loss_mask"):
            if key in sample:
                data[key] = sample[key]
        # per-hand supervision: carry the per-channel action mask when present
        # (homogeneous within a dataset config; absent for all sources without per-hand annotations).
        if sample.get("action_dim_valid") is not None:
            data["action_dim_valid"] = sample["action_dim_valid"]
        if sample.get("proprio_dim_is_pad") is not None:
            data["proprio_dim_is_pad"] = sample["proprio_dim_is_pad"]
        if getattr(self, "mask_bad_segment_action_loss", False):
            reason = sample["logical_bad_reason"]
            data["action_dim_valid"] = bad_segment_action_mask(
                action, reason, data.get("action_dim_valid")
            )
            data["bad_segment_action_loss_masked"] = bool(reason)
        if self.goal_supervision:
            gidx = int(sample.get("idx", sample_idx))
            data["goal_image"] = self._episode_terminal_goal_image(gidx)
            # Long-goal RoPE index (GoalWAM exp5). Cheap scalar; the model only uses it
            # when goal_long_rope_mode='distance', ignored otherwise.
            data["goal_long_rope_index"] = torch.tensor(
                self._goal_long_rope_index(gidx), dtype=torch.long
            )
        return data

    def _instruction_task_for_sample(self, sample, sample_idx: int) -> str:
        """Resolve one task for this access without a frame-to-caption binding."""

        task = sample["instruction"]
        object_sampler = getattr(self, "object_conditioned_instruction_sampler", None)
        episode_sampler = getattr(self, "episode_instruction_sampler", None)
        if object_sampler is not None or episode_sampler is not None:
            resolved_sample_idx = int(sample.get("idx", sample_idx))
            episode_key = self.lerobot_dataset._runtime_episode_key(
                resolved_sample_idx
            )
            dataset_index, episode_index = episode_key
            sampler = object_sampler if object_sampler is not None else episode_sampler
            draw_ordinal = int(getattr(self, "_instruction_draw_ordinal", 0))
            self._instruction_draw_ordinal = draw_ordinal + 1
            if (
                getattr(self, "object_conditioned_instruction_sampling", "per_draw")
                == "per_draw"
                and getattr(sampler, "supports_draw_ordinal", False)
            ):
                task = sampler.sample_task(
                    dataset_index,
                    episode_index,
                    draw_ordinal=draw_ordinal,
                    epoch=int(getattr(self, "_instruction_epoch", 0)),
                    prompt_seed=int(
                        getattr(self, "object_conditioned_instruction_seed", 0)
                    ),
                )
            else:
                # Compatibility path for legacy/custom samplers which expose
                # only the original (dataset_index, episode_index) API.
                task = sampler.sample_task(dataset_index, episode_index)
        if self.override_instruction is not None:
            task = self.override_instruction
        return task

    def set_epoch(self, epoch: int) -> None:
        """Reset the per-access prompt draw stream at an epoch boundary.

        The trainer calls this before constructing each epoch's iterator.  The
        DataLoader's ``persistent_workers=auto`` policy consequently recreates
        workers, so every worker observes the current epoch and starts a fresh
        local draw ordinal rather than retaining an episode-level caption.
        """

        self._instruction_epoch = int(epoch)
        self._instruction_draw_ordinal = 0

    def _reset_text_context_cache_state(self) -> None:
        """Create an empty process-local LRU and reset its counters.

        DataLoader workers normally receive the dataset through fork or spawn.  The
        PID guard in :meth:`_ensure_text_context_cache_state` prevents a cache that
        was warmed in the parent process from being reused as worker-local state.
        """

        self._text_context_cache = OrderedDict()
        self._text_context_cache_pid = os.getpid()
        self._text_context_cache_hits = 0
        self._text_context_cache_misses = 0
        self._text_context_cache_evictions = 0

    def _ensure_text_context_cache_state(self) -> None:
        if (
            not isinstance(getattr(self, "_text_context_cache", None), OrderedDict)
            or getattr(self, "_text_context_cache_pid", None) != os.getpid()
        ):
            self._reset_text_context_cache_state()

    def get_text_context_cache_stats(self) -> dict[str, int]:
        """Return counters for the current process's text-context LRU."""

        self._ensure_text_context_cache_state()
        return {
            "capacity": int(getattr(self, "text_context_cache_size", 0) or 0),
            "size": len(self._text_context_cache),
            "hits": self._text_context_cache_hits,
            "misses": self._text_context_cache_misses,
            "evictions": self._text_context_cache_evictions,
        }

    def _get_cached_text_context(self, prompt: str):
        text_pack = getattr(self, "_text_embedding_pack", None)
        self._ensure_text_context_cache_state()
        cache_capacity = int(getattr(self, "text_context_cache_size", 0) or 0)
        if cache_capacity > 0 and prompt in self._text_context_cache:
            self._text_context_cache_hits += 1
            context, context_mask = self._text_context_cache.pop(prompt)
            self._text_context_cache[prompt] = (context, context_mask)
            # Callers mask the returned context in-place.  Never expose the
            # cached tensors directly or one sample can corrupt later samples.
            return context.clone(), context_mask.clone()

        self._text_context_cache_misses += 1
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        filename = f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
        payload = None
        if text_pack is not None:
            try:
                payload = text_pack.load(filename)
                cache_path = f"{self.text_embedding_pack_dir}:{filename}"
            except FileNotFoundError:
                # A merged corpus may pack only its new prompts; older corpora keep
                # their per-file caches, which are consulted next when configured.
                if self.text_embedding_cache_dir is None:
                    raise
        if payload is None:
            cache_dir = self.text_embedding_cache_dir
            if not getattr(self, "_text_embedding_cache_dir_ready", False):
                os.makedirs(cache_dir, exist_ok=True)
                self._text_embedding_cache_dir_ready = True
            candidate_paths = [
                os.path.join(cache_dir, filename),
                *[
                    os.path.join(fallback_dir, filename)
                    for fallback_dir in self.text_embedding_fallback_cache_dirs
                ],
            ]
            cache_path = next(
                (path for path in candidate_paths if os.path.exists(path)),
                None,
            )
            payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()

        if cache_capacity > 0:
            cached_context = context.detach().contiguous()
            cached_mask = context_mask.detach().contiguous()
            self._text_context_cache[prompt] = (cached_context, cached_mask)
            if len(self._text_context_cache) > cache_capacity:
                self._text_context_cache.popitem(last=False)
                self._text_context_cache_evictions += 1
            return cached_context.clone(), cached_mask.clone()

        return context, context_mask

    def __getitem__(self, idx):
        return self._get(idx)
