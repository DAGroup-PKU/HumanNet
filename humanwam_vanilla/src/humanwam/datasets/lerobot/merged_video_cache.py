"""Pre-merged decoded-frame cache for RobotVideoDataset.

Moves all video decode + per-camera resize + multi-camera merge off the training
hot path into a one-time, parallelizable preprocessing pass. For each episode we
store the *already merged* frame stack -- exactly the float tensor the model
consumes -- as a per-episode float32 memmap:

    <cache_dir>/<dataset_name>_<sig>/episode_<ep:06d>.f32   # [ep_len, 3, Ho, Wo]

For valid source clips, the stored value is bit-for-bit
`RobotVideoDataset._assemble_frames` applied to the per-camera frames after the
processor's image `train_transforms`. Because _assemble_frames is per-frame
independent (resize/crop/normalize/concat have no temporal coupling),
precomputing per-frame and stacking a training window is identical to the
on-the-fly path. Stored as float32 (no uint8 quantization) so training is
numerically identical to the non-cached pipeline. A small set of pinned
RoboTwin AV1 files is shorter than its original episode metadata; the local
metadata/parquet corpus is trimmed to the shortest camera before cache build,
so no synthetic frames are introduced.

Per-episode files make the build embarrassingly parallel across nodes/shards and
idempotent (atomic rename + size check), and let lookup mmap only the episode it
needs (page cache keeps hot frames resident).  The training hot path assumes a
POSIX filesystem with efficient random mmap reads (tmpfs/local disk/GPFS).  Object
storage exposed through FUSE must be staged locally first: concurrent mmap page
faults across all DataLoader workers otherwise turn into rank stragglers and can
surface later as an apparently unrelated NCCL collective timeout.
"""
import hashlib
import json
import logging
import os
import fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

from .lerobot.datasets.video_utils import decode_video_frames

logger = logging.getLogger(__name__)


def image_transforms_signature(image_transforms, camera_keys) -> str:
    """Return a stable signature for the selected cameras' image transforms."""
    entries = []
    for camera_key in camera_keys:
        camera_name = camera_key.removeprefix("observation.images.")
        transforms = (
            image_transforms.get(camera_name)
            if isinstance(image_transforms, dict)
            else image_transforms
        )
        entries.append(
            {
                "camera_key": camera_key,
                "transforms": [
                    f"{type(transform).__module__}.{type(transform).__qualname__}:"
                    f"{transform!r}"
                    for transform in transforms
                ],
            }
        )
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _unescape_mountinfo_path(value: str) -> str:
    """Decode the octal escapes used for mount points in /proc/self/mountinfo."""
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def filesystem_type_for_path(path) -> str | None:
    """Return the filesystem type for ``path`` using the current mount namespace.

    Reading mountinfo avoids shelling out from every distributed rank and follows
    symlinks before selecting the longest matching mount point.
    """
    resolved = os.path.realpath(os.fspath(path))
    best_mount = ""
    best_fstype = None
    try:
        with open("/proc/self/mountinfo", "r", encoding="utf-8") as f:
            for line in f:
                left, right = line.rstrip("\n").split(" - ", 1)
                fields = left.split()
                if len(fields) < 5:
                    continue
                mount_point = _unescape_mountinfo_path(fields[4])
                if resolved != mount_point and not resolved.startswith(mount_point.rstrip("/") + "/"):
                    continue
                if len(mount_point) > len(best_mount):
                    best_mount = mount_point
                    best_fstype = right.split()[0]
    except (OSError, ValueError):
        return None
    return best_fstype


def is_fuse_filesystem(filesystem_type: str | None) -> bool:
    return bool(filesystem_type) and (
        filesystem_type == "fuse" or filesystem_type.startswith("fuse.")
    )


class MergedFrameCache:
    VERSION = 2
    DTYPE = np.dtype("<f4")
    LAYOUT = "TCHW"
    LOGICAL_DTYPE = None
    PAYLOAD_SUFFIX = ".f32"

    def __init__(
        self,
        inner_dataset,
        out_hw,
        cache_dir,
        *,
        camera_keys=None,
        transform_signature=None,
    ):
        """`inner_dataset` is the underlying LeRobotDataset (has .root, .meta,
        .episode_data_index, .episodes, .tolerance_s, .video_backend).
        `out_hw` = (Ho, Wo) merged output resolution (e.g. (384, 320))."""
        self.inner = inner_dataset
        self.meta = inner_dataset.meta
        self.fps = inner_dataset.meta.fps
        self.tolerance_s = inner_dataset.tolerance_s
        self.video_backend = inner_dataset.video_backend
        # Camera order MUST match shape_meta / _assemble_frames expectations.
        self.camera_keys = list(
            inner_dataset.meta.video_keys if camera_keys is None else camera_keys
        )
        self.Ho, self.Wo = int(out_hw[0]), int(out_hw[1])
        self.transform_signature = str(transform_signature or "unspecified")

        episodes = (
            inner_dataset.episodes
            if inner_dataset.episodes is not None
            else list(range(inner_dataset.meta.total_episodes))
        )
        self.episodes = [int(e) for e in episodes]
        from_idx = inner_dataset.episode_data_index["from"]
        to_idx = inner_dataset.episode_data_index["to"]
        self.ep_lengths = {}
        for local_pos, ep in enumerate(self.episodes):
            self.ep_lengths[int(ep)] = int(to_idx[local_pos].item() - from_idx[local_pos].item())

        self.manifest = {
            "schema": "fastwam_merged_frame_cache",
            "version": self.VERSION,
            "source_root": str(Path(inner_dataset.root).resolve()),
            "video_backend": str(self.video_backend),
            "fps": float(self.fps),
            "camera_keys": self.camera_keys,
            "out_hw": [self.Ho, self.Wo],
            "dtype": self.DTYPE.str,
            "layout": self.LAYOUT,
            "transform_signature": self.transform_signature,
            "episodes": self.episodes,
            "episode_lengths": {
                str(ep): self.ep_lengths[ep] for ep in self.episodes
            },
        }
        if self.LOGICAL_DTYPE is not None:
            self.manifest["dtype"] = self.LOGICAL_DTYPE
            self.manifest["storage_dtype"] = self.DTYPE.str
        manifest_bytes = json.dumps(
            self.manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        sig = hashlib.sha256(manifest_bytes).hexdigest()[:16]
        self.dir = (
            Path(cache_dir)
            / f"{Path(inner_dataset.root).name}_merged_v{self.VERSION}_{sig}"
        )
        self.dir.mkdir(parents=True, exist_ok=True)
        self._publish_manifest()
        self._buffers = {}  # ep_idx -> read-only memmap
        # Staging is an optional local read path; source identity and all model
        # transforms stay unchanged. Capacity is fixed, with independent slots.
        self._local_stage = None
        stage_root = os.environ.get("FASTWAM_MERGED_CACHE_STAGE_DIR", "").strip()
        if stage_root:
            from .local_frame_stage import LocalFrameStage
            budget = int(os.environ.get("FASTWAM_MERGED_CACHE_STAGE_MAX_BYTES", "0"))
            block = int(os.environ.get("FASTWAM_MERGED_CACHE_STAGE_CHUNK_FRAMES", "8"))
            stride = int(os.environ.get("FASTWAM_MERGED_CACHE_STAGE_STRIDE", "4"))
            identity = {
                "source_cache": str(self.dir.resolve()),
                "source_manifest_sha256": hashlib.sha256(self._manifest_path().read_bytes()).hexdigest(),
                "schema": "fixed_slots_v2", "budget": budget, "block": block, "stride": stride,
            }
            signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            stage_dir = Path(stage_root) / f"{self.dir.name}_slots_{signature}"
            self._local_stage = LocalFrameStage(
                stage_dir, max_bytes=budget,
                frame_bytes=3 * self.Ho * self.Wo * self.DTYPE.itemsize,
                block_frames=block, stride=stride,
            )
            manifest = stage_dir / "STAGE_MANIFEST.json"
            # Identical publishers can race safely; no hot-path global lock.
            tmp = manifest.with_name(f".{manifest.name}.{os.getpid()}")
            tmp.write_text(json.dumps(identity, sort_keys=True) + "\n")
            os.replace(tmp, manifest)

    # ---- paths / validity ----
    def _manifest_path(self):
        return self.dir / "CACHE_MANIFEST.json"

    def _publish_manifest(self):
        path = self._manifest_path()
        if path.exists():
            return
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        tmp.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)

    def _ep_path(self, ep):
        return self.dir / f"episode_{ep:06d}{self.PAYLOAD_SUFFIX}"


    def _ep_lock_path(self, ep):
        return self.dir / f"episode_{ep:06d}.lock"

    def _ep_nbytes(self, ep):
        return self.ep_lengths[ep] * self.Ho * self.Wo * 3 * self.DTYPE.itemsize

    def ep_done(self, ep):
        p = self._ep_path(ep)
        return p.is_file() and p.stat().st_size == self._ep_nbytes(ep)

    def missing_episodes(self, episodes=None):
        eps = self.episodes if episodes is None else episodes
        return [e for e in eps if not self.ep_done(e)]

    def is_complete(self):
        return len(self.missing_episodes()) == 0

    def expected_nbytes(self):
        return sum(self._ep_nbytes(ep) for ep in self.episodes)


    # ---- build ----
    def build(
        self,
        assemble_fn,
        image_transforms,
        shard=(0, 1),
        workers=64,
        log_every=100,
        progress_callback=None,
    ):
        """Decode + transform + merge every (missing) episode in this shard and
        write its per-episode float32 memmap.

        assemble_fn:     RobotVideoDataset._assemble_frames (bound method).
        image_transforms: the processor's image transform list (ToTensor, Resize)
                          applied per camera, exactly as training does.
        shard=(i, n):    build episodes where (position % n == i) for parallelism.
        """
        i, n = shard
        my_eps = [e for pos, e in enumerate(self.episodes) if pos % n == i]
        todo = self.missing_episodes(my_eps)
        logger.info(
            f"[merged-cache] shard {i}/{n}: {len(todo)}/{len(my_eps)} episodes to build "
            f"-> {self.dir} (out {self.Ho}x{self.Wo}, ~{self._ep_nbytes(self.episodes[0])/1e6:.0f} MB/ep)"
        )

        def _build_one(ep):
            lock_path = self._ep_lock_path(ep)
            with lock_path.open("a+b") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                if self.ep_done(ep):
                    return False, self.ep_lengths[ep]

                length = self.ep_lengths[ep]
                per_cam = []
                for cam in self.camera_keys:
                    ts_off = self.meta.get_video_timestamp_offset(ep, cam)
                    ts = [ts_off + k / self.fps for k in range(length)]
                    path = self.inner.root / self.meta.get_video_file_path(ep, cam)
                    frames = decode_video_frames(
                        path,
                        ts,
                        self.tolerance_s,
                        self.video_backend,
                    )  # [T,C,Hraw,Wraw]
                    camera_name = cam.removeprefix("observation.images.")
                    transforms = (
                        image_transforms.get(camera_name)
                        if isinstance(image_transforms, dict)
                        else image_transforms
                    )
                    for transform in transforms:
                        if hasattr(transform, "apply_with_context"):
                            tasks = self.meta.episodes[ep]["tasks"]
                            frames = transform.apply_with_context(frames, task=tasks[0])
                        else:
                            frames = transform(frames)
                    per_cam.append(frames)
                multicam = torch.stack(per_cam, dim=0)
                merged = assemble_fn(multicam)
                arr = self._encode_storage(merged)
                final_path = self._ep_path(ep)
                tmp = final_path.with_name(f"{final_path.name}.tmp.{os.getpid()}")
                try:
                    mm = np.memmap(
                        tmp,
                        dtype=self.DTYPE,
                        mode="w+",
                        shape=(length, 3, self.Ho, self.Wo),
                    )
                    mm[:] = arr
                    mm.flush()
                    del mm
                    with tmp.open("rb") as payload:
                        os.fsync(payload.fileno())
                    os.replace(tmp, final_path)
                finally:
                    tmp.unlink(missing_ok=True)
                return True, length

        done = 0
        processed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_build_one, ep): ep for ep in todo}
            for k, fut in enumerate(as_completed(futs)):
                try:
                    built, _ = fut.result()
                except Exception as e:  # noqa
                    logger.error(f"[merged-cache] episode {futs[fut]} FAILED: {e!r}")
                    raise
                done += int(built)
                processed += 1
                if processed % log_every == 0 or processed == len(todo):
                    logger.info(
                        f"[merged-cache] shard {i}/{n}: processed "
                        f"{processed}/{len(todo)}, built {done}"
                    )
                    if progress_callback is not None:
                        progress_callback()
        return done

    # ---- lookup (training hot path) ----
    def _encode_storage(self, merged: torch.Tensor) -> np.ndarray:
        return (
            merged.permute(1, 0, 2, 3)
            .contiguous()
            .to(torch.float32)
            .numpy()
            .astype(self.DTYPE, copy=False)
        )

    def _decode_storage(self, value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.array(value, copy=True))

    def _buf(self, ep_idx):
        ep_idx = int(ep_idx)
        buf = self._buffers.get(ep_idx)
        if buf is None:
            storage_path = self._ep_path(ep_idx)
            buf = np.memmap(
                storage_path, dtype=self.DTYPE, mode="r",
                shape=(self.ep_lengths[ep_idx], 3, self.Ho, self.Wo),
            )
            self._buffers[ep_idx] = buf
        return buf

    def _fd(self, ep_idx):
        # Keyed by pid so forked DataLoader workers never share descriptors;
        # bounded to stay far below the per-process open-file limit.
        fds = self.__dict__.setdefault("_fds", {})
        key = (os.getpid(), ep_idx)
        fd = fds.pop(key, None)
        if fd is None:
            fd = os.open(self._ep_path(ep_idx), os.O_RDONLY)
            while len(fds) >= 256:
                old_key = next(iter(fds))
                old_fd = fds.pop(old_key)
                if old_key[0] == key[0]:
                    os.close(old_fd)
        fds[key] = fd
        return fd

    def _pread_frames(self, ep_idx: int, indices: np.ndarray) -> np.ndarray:
        """Read selected frames with one pread per contiguous run.

        Avoids GPFS mmap page-fault serialization across many DataLoader
        workers; returned bytes are identical to the memmap path.
        """
        indices = np.asarray(indices, dtype=np.int64)
        frame_shape = (3, self.Ho, self.Wo)
        frame_bytes = 3 * self.Ho * self.Wo * self.DTYPE.itemsize
        out = np.empty((len(indices),) + frame_shape, dtype=self.DTYPE)
        if not len(indices):
            return out
        fd = self._fd(int(ep_idx))
        flat = out.reshape(len(indices), frame_bytes // self.DTYPE.itemsize)
        pos = 0
        while pos < len(indices):
            end = pos + 1
            while end < len(indices) and indices[end] == indices[end - 1] + 1:
                end += 1
            view = memoryview(flat[pos:end]).cast("B")
            offset = int(indices[pos]) * frame_bytes
            while view:
                n = os.preadv(fd, [view], offset)
                if n <= 0:
                    raise OSError(f"[merged-cache] short read in episode {ep_idx}")
                view, offset = view[n:], offset + n
            pos = end
        return out

    def _lookup_storage(self, ep_idx: int, indices: np.ndarray) -> np.ndarray:
        if self._local_stage is None:
            if os.environ.get("FASTWAM_MERGED_CACHE_READ", "mmap") == "pread":
                return self._pread_frames(ep_idx, indices)
            return self._buf(ep_idx)[indices]
        return self._local_stage.lookup(
            self._ep_path(ep_idx), ep_idx, self.ep_lengths[ep_idx], indices,
            dtype=self.DTYPE, frame_shape=(3, self.Ho, self.Wo),
        )

    def _ts_to_indices(self, query_ts, ep_len):
        # Same mapping as _DiskVideoFrameCache.lookup: round(ts * fps), clip.
        idx = np.asarray(query_ts, dtype=np.float64) * self.fps
        return np.clip(np.round(idx), 0, ep_len - 1).astype(np.int64)

    def lookup_window(
        self,
        query_timestamps: dict,
        ep_idx: int,
        frame_positions=None,
    ) -> torch.Tensor:
        """Return merged window frames [T, 3, Ho, Wo] in [-1, 1] for the given
        per-camera query timestamps (all cameras share frame indices).

        ``frame_positions`` optionally selects positions inside the query window
        before touching the memmap.  RoboTwin asks LeRobot for 33 trajectory
        frames but the model consumes only positions [0,4,...,32]; selecting here
        avoids reading and copying the other 24 full-resolution float frames.
        """
        ep_idx = int(ep_idx)
        ep_len = self.ep_lengths[ep_idx]
        any_ts = next(iter(query_timestamps.values()))
        indices = self._ts_to_indices(any_ts, ep_len)
        if frame_positions is not None:
            positions = np.asarray(frame_positions, dtype=np.int64)
            indices = indices[positions]
        return self._decode_storage(self._lookup_storage(ep_idx, indices))

    def lookup_frame(self, ep_idx: int, frame_index: int) -> torch.Tensor:
        """Return a single merged frame [3, Ho, Wo] in [-1, 1]."""
        ep_idx = int(ep_idx)
        ep_len = self.ep_lengths[ep_idx]
        fi = int(np.clip(frame_index, 0, ep_len - 1))
        return self._decode_storage(self._lookup_storage(ep_idx, np.asarray([fi], dtype=np.int64))[0])


class BFloat16MergedFrameCache(MergedFrameCache):
    """Model-input-equivalent pixel cache stored as raw BF16 payload bits."""

    VERSION = 3
    DTYPE = np.dtype("<u2")
    LOGICAL_DTYPE = "bfloat16"
    PAYLOAD_SUFFIX = ".bf16"

    def _encode_storage(self, merged: torch.Tensor) -> np.ndarray:
        return (
            merged.permute(1, 0, 2, 3)
            .contiguous()
            .to(torch.bfloat16)
            .view(torch.uint16)
            .numpy()
            .astype(self.DTYPE, copy=False)
        )

    def _decode_storage(self, value: np.ndarray) -> torch.Tensor:
        # pread already returns a private writable array; memmap slices are copied.
        if isinstance(value, np.memmap) or not value.flags.writeable or value.dtype != self.DTYPE:
            value = np.array(value, dtype=self.DTYPE, copy=True)
        return torch.from_numpy(value).view(torch.bfloat16)


class TShapeMp4FrameCache:
    """Pre-merged RoboTwin T canvas stored as one compressed MP4 per episode.

    Each frame is head [256,352] over left|right wrist [128,176] (= [384,352]),
    built offline from the BF16 separate-view cache with the model's own wrist
    resize (bilinear, antialias). Training decodes only the selected frames of a
    ~1 MB file instead of reading ~15 MB of raw BF16 per sample. Frame i has
    timestamp i/fps. Returns [T,3,384,352] bfloat16 in [-1,1], fed to
    the model with ``latent_tshape_premerged_input=True``.
    """

    SCHEMA = "fastwam_tshape_mp4_v1"
    PREMERGED_T = True

    def __init__(self, inner_dataset, out_hw, cache_dir, *, camera_keys=None, transform_signature=None):
        self.inner = inner_dataset
        self.fps = inner_dataset.meta.fps
        self.Ho, self.Wo = int(out_hw[0]), int(out_hw[1])
        root = Path(inner_dataset.root).resolve()
        self.dir = Path(cache_dir) / root.name
        episodes = inner_dataset.episodes if inner_dataset.episodes is not None else list(range(inner_dataset.meta.total_episodes))
        self.episodes = [int(e) for e in episodes]
        from_idx, to_idx = inner_dataset.episode_data_index["from"], inner_dataset.episode_data_index["to"]
        self.ep_lengths = {ep: int(to_idx[i].item() - from_idx[i].item()) for i, ep in enumerate(self.episodes)}

    def _ep_path(self, ep):
        return self.dir / f"episode_{int(ep):06d}.mp4"

    def ep_done(self, ep):
        p = self._ep_path(ep)
        return p.is_file() and p.stat().st_size == self._ep_nbytes(ep)

    def missing_episodes(self, episodes=None):
        return [e for e in (self.episodes if episodes is None else episodes) if not self.ep_done(e)]

    def is_complete(self):
        return not self.missing_episodes()


    def _ts_to_indices(self, query_ts, ep_len):
        idx = np.asarray(query_ts, dtype=np.float64) * self.fps
        return np.clip(np.round(idx), 0, ep_len - 1).astype(np.int64)

    def _decode(self, ep_idx, indices):
        import av
        wanted = sorted(set(int(i) for i in indices))
        frames = {}
        with av.open(str(self._ep_path(ep_idx))) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            tb = stream.time_base
            container.seek(int(wanted[0] / self.fps / tb), stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                i = int(round(float(frame.pts * tb) * self.fps))
                if i in frames or i not in wanted:
                    if i > wanted[-1]:
                        break
                    continue
                frames[i] = frame.to_ndarray(format="rgb24")
                if len(frames) == len(wanted):
                    break
        out = np.stack([frames[int(i)] for i in indices])  # [T,H,W,3] uint8
        return torch.from_numpy(out).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0).to(torch.bfloat16)

    def lookup_window(self, query_timestamps, ep_idx, frame_positions=None):
        ep_idx = int(ep_idx)
        indices = self._ts_to_indices(next(iter(query_timestamps.values())), self.ep_lengths[ep_idx])
        if frame_positions is not None:
            indices = indices[np.asarray(frame_positions, dtype=np.int64)]
        return self._decode(ep_idx, indices)

    def lookup_frame(self, ep_idx, frame_index):
        ep_idx = int(ep_idx)
        fi = int(np.clip(frame_index, 0, self.ep_lengths[ep_idx] - 1))
        return self._decode(ep_idx, [fi])[0]
