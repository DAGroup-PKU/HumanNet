#!/usr/bin/env python
# Modified by Open-HumanScale contributors: removed unused release runtime code.

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Includes FastWAM/OpenHumanScale adaptations; original upstream notices are retained.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import bisect
import json
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable
import warnings

import datasets
import numpy as np
import packaging.version
import torch
import torch.utils
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from datasets import load_dataset
from huggingface_hub import snapshot_download

# adapte to own path
from ..constants import HF_LEROBOT_HOME
from ..content_addressed_cache import ContentAddressedFileCache
from ..packed_parquet_cache import read_packed_parquet_table
from .datasets.compute_stats import aggregate_stats
from .datasets.utils import backward_compatible_episodes_stats, get_delta_indices, get_episode_data_index, get_hf_features_from_features, hf_transform_to_torch, load_episodes, load_episodes_stats, load_info, load_stats, load_tasks, load_annotations
from .datasets.video_utils import VideoFrame, decode_video_frames, get_safe_default_codec
from ..temporal_resampling import (
    TemporalResamplingSpec,
    build_temporal_index_map,
    interpolate_continuous_feature,
    is_temporally_continuous_feature,
    parse_temporal_resampling_spec,
    virtual_episode_length)

CODEBASE_VERSION = "v2.1"


def _stack_hf_formatted_column(column) -> torch.Tensor:
    """Stack a torch-formatted HF column across Datasets 3.x and 4.x.

    Datasets 3.x returns a plain list of tensors for a selected column, while
    Datasets 4.x wraps the same values in a lazy ``Column`` iterable. Passing
    that wrapper directly to ``torch.stack`` raises a TypeError.
    """

    if isinstance(column, torch.Tensor):
        return column
    values = list(column)
    if not values:
        return torch.empty(0)
    if isinstance(values[0], torch.Tensor):
        return torch.stack(values)
    return torch.as_tensor(values)


def _arrow_table_to_torch_dict(table: pa.Table) -> dict:
    """Convert an episode parquet table to the tensors returned by HF datasets.

    Vector columns are commonly Arrow list arrays, whose NumPy representation is
    object-typed; stack those rows into a dense array before wrapping with torch.
    String/nullable object columns remain NumPy arrays and are only indexed as
    ordinary Python metadata.
    """
    result = {}
    for col_name in table.column_names:
        col = table[col_name]
        try:
            np_arr = col.to_numpy(zero_copy_only=True)
        except Exception:
            np_arr = col.to_numpy()
        if (
            np_arr.dtype == object
            and len(np_arr)
            and np_arr[0] is not None
            and not isinstance(np_arr[0], str)
        ):
            np_arr = np.stack(np_arr)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given NumPy array is not writable",
                category=UserWarning,
            )
            result[col_name] = (
                np_arr if np_arr.dtype.kind in {"O", "U", "S"} else torch.from_numpy(np_arr)
            )
    return result


def _load_lazy_episode_index(root: Path) -> dict[int, dict]:
    """Load only episode fields required by the lazy training backend."""
    path = root / "meta" / "episodes.jsonl"
    if path.exists():
        episodes = {}
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                compact = {
                    key: value
                    for key, value in item.items()
                    if key in {"episode_index", "episode_chunk", "length", "source_hand_tag"}
                    or key.startswith("data/")
                    or key.startswith("videos/")
                }
                episodes[int(item["episode_index"])] = compact
        return dict(sorted(episodes.items()))

    # Non-v2 JSONL layouts are uncommon for this backend; preserve support by
    # compacting the standard metadata loader's result.
    full = load_episodes(root)
    return {
        int(ep_idx): {
            key: value
            for key, value in item.items()
            if key in {"episode_index", "episode_chunk", "length", "source_hand_tag"}
            or key.startswith("data/")
            or key.startswith("videos/")
        }
        for ep_idx, item in full.items()
    }


class LeRobotDatasetMetadata:
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        skip_episode_stats: bool = False,
    ):
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        self.skip_episode_stats = bool(skip_episode_stats)

        try:
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):

            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/")
            self.load_metadata()

    def load_metadata(self):
        self.info = load_info(self.root)
        # TODO add new check
        # check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        if (self.root / "annotations").exists():
            self.annotations = load_annotations(self.root)
        self.episodes = (
            _load_lazy_episode_index(self.root)
            if self.skip_episode_stats
            else load_episodes(self.root)
        )
        if self.skip_episode_stats:
            # Training with explicitly pinned normalization stats does not need the
            # per-episode metadata.  Large ego corpora contain millions of entries;
            # parsing and aggregating them in every rank defeats lazy parquet loading.
            self.episodes_stats = {}
            self.stats = {}
        elif self._version < packaging.version.parse("v2.1"):
            self.stats = load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        else:
            self.episodes_stats = load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    @property
    def _version(self) -> packaging.version.Version:
        """Codebase version used to create this dataset."""
        return packaging.version.parse(self.info["codebase_version"])

    @property
    def uses_file_indexed_episodes(self) -> bool:
        return any("data/chunk_index" in ep for ep in self.episodes.values())

    def get_data_file_path(self, ep_index: int) -> Path:
        episode = self.episodes.get(int(ep_index), {})
        if "data/chunk_index" in episode and "data/file_index" in episode:
            fpath = self.data_path.format(
                chunk_index=int(episode["data/chunk_index"]),
                file_index=int(episode["data/file_index"]),
                episode_chunk=self.get_episode_chunk(ep_index),
                episode_index=ep_index,
            )
        else:
            ep_chunk = self.get_episode_chunk(ep_index)
            fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        episode = self.episodes.get(int(ep_index), {})
        chunk_key = f"videos/{vid_key}/chunk_index"
        file_key = f"videos/{vid_key}/file_index"
        if chunk_key in episode and file_key in episode:
            fpath = self.video_path.format(
                chunk_index=int(episode[chunk_key]),
                file_index=int(episode[file_key]),
                episode_chunk=self.get_episode_chunk(ep_index),
                video_key=vid_key,
                episode_index=ep_index,
            )
        else:
            ep_chunk = self.get_episode_chunk(ep_index)
            fpath = self.video_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)
        return Path(fpath)

    def get_video_timestamp_offset(self, ep_index: int, vid_key: str) -> float:
        episode = self.episodes.get(int(ep_index), {})
        return float(episode.get(f"videos/{vid_key}/from_timestamp", 0.0) or 0.0)

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size

    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        """Formattable string for the video files."""
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        """Robot type used in recording this dataset."""
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        """Names of the various dimensions of vector modalities."""
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """Total number of different tasks performed in this dataset."""
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        """Total number of chunks (groups of episodes)."""
        return self.info["total_chunks"]

    @property
    def chunks_size(self) -> int:
        """Max number of episodes per chunk."""
        return self.info["chunks_size"]


    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        lazy_parquet_loading: bool = False,
        metadata: LeRobotDatasetMetadata | None = None,
        temporal_resampling: TemporalResamplingSpec | Mapping[str, Any] | None = None,
        episode_segments: Sequence[Mapping[str, Any]] | None = None,
        codebook_registry: Mapping[str, Mapping[str, Any]] | None = None,
        file_cache: ContentAddressedFileCache | None = None,
    ):
        """
        Load an existing LeRobot dataset from local files or the Hugging Face Hub.
        This training/evaluation reader does not record or publish datasets.

        In terms of files, LeRobotDataset encapsulates 3 main things:
            - metadata:
                - info contains various information about the dataset like shapes, keys, fps etc.
                - stats stores the dataset statistics of the different modalities for normalization
                - tasks contains the prompts for each task of the dataset, which can be used for
                  task-conditioned training.
            - hf_dataset (from datasets.Dataset), which will read any values from parquet files.
            - videos (optional) from which frames are loaded to be synchronous with data from parquet files.

        A typical LeRobotDataset looks like this from its root path:
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── episode_000000.parquet
        │   │   ├── episode_000001.parquet
        │   │   ├── episode_000002.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── episode_001000.parquet
        │   │   ├── episode_001001.parquet
        │   │   ├── episode_001002.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes.jsonl
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.jsonl
        └── videos
            ├── chunk-000
            │   ├── observation.images.laptop
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            │   ├── observation.images.phone
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            ├── chunk-001
            └── ...

        Note that this file-based structure is designed to be as versatile as possible. The files are split by
        episodes which allows a more granular control over which episodes one wants to use and download. The
        structure of the dataset is entirely described in the info.json file, which can be easily downloaded
        or viewed directly on the hub before downloading any actual data. The type of files used are very
        simple and do not need complex tools to be read, it only uses .parquet, .json and .mp4 files (and .md
        for the README).

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset. Locally, the dataset
                will be stored under root/repo_id.
            root (Path | None, optional): Local directory to use for downloading/writing files. You can also
                set the LEROBOT_HOME environment variable to point to a different location. Defaults to
                '~/.cache/huggingface/lerobot'.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list. Defaults to None.
            image_transforms (Callable | None, optional): You can pass standard v2 image transforms from
                torchvision.transforms.v2 here which will be applied to visual modalities (whether they come
                from videos or images). Defaults to None.
            delta_timestamps (dict[list[float]] | None, optional): _description_. Defaults to None.
            tolerance_s (float, optional): Tolerance in seconds used to ensure data timestamps are actually in
                sync with the fps value. It is used at the init of the dataset to make sure that each
                timestamps is separated to the next by 1/fps +/- tolerance_s. This also applies to frames
                decoded from video files. It is also used to check that `delta_timestamps` (when provided) are
                multiples of 1/fps. Defaults to 1e-4.
            revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
                commit hash. Defaults to current codebase version tag.
            sync_cache_first (bool, optional): Flag to sync and refresh local files first. If True and files
                are already present in the local cache, this will be faster. However, files loaded might not
                be in sync with the version on the hub, especially if you specified 'revision'. Defaults to
                False.
            download_videos (bool, optional): Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.
            video_backend (str | None, optional): Video backend to use for decoding videos. Defaults to torchcodec when available int the platform; otherwise, defaults to 'pyav'.
                You can also use the 'pyav' decoder used by Torchvision, which used to be the default option, or 'video_reader' which is another decoder of Torchvision.
        """
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.lazy_parquet_loading = bool(lazy_parquet_loading)
        self.file_cache = file_cache
        self.temporal_resampling = parse_temporal_resampling_spec(temporal_resampling)
        self.temporal_resampling_enabled = not self.temporal_resampling.is_identity
        self._episode_segments_enabled = episode_segments is not None
        self._requested_episode_segments = list(episode_segments or [])
        self._codebook_registry = {
            str(axis): dict(record)
            for axis, record in (codebook_registry or {}).items()
        }
        self.delta_indices = None
        self.during_training = True
        self._lazy_episode_cache = OrderedDict()
        self._lazy_cache_lock = threading.RLock()
        self._lazy_cache_pid = os.getpid()
        self._lazy_cache_size = max(
            1, int(os.environ.get("FASTWAM_LAZY_PARQUET_CACHE_EPISODES", "1"))
        )

        # Unused attributes

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = metadata or LeRobotDatasetMetadata(
            self.repo_id,
            self.root,
            self.revision,
            force_cache_sync=force_cache_sync,
            skip_episode_stats=self.lazy_parquet_loading,
        )
        if (
            not self.lazy_parquet_loading
            and self.episodes is not None
            and self.meta._version >= packaging.version.parse("v2.1")
        ):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        self.native_episode_data_index = get_episode_data_index(
            self.meta.episodes, self.episodes
        )
        if self._episode_segments_enabled:
            self._configure_episode_segments()
        elif getattr(self, "temporal_resampling_enabled", False):
            native_lengths = (
                self.native_episode_data_index["to"]
                - self.native_episode_data_index["from"]
            )
            native_lengths_np = native_lengths.numpy()
            virtual_lengths = torch.from_numpy(
                np.fromiter(
                    (
                        virtual_episode_length(int(length), self.temporal_resampling)
                        for length in native_lengths_np
                    ),
                    dtype=np.int64,
                    count=len(native_lengths_np),
                )
            )
            virtual_ends = torch.cumsum(virtual_lengths, dim=0)
            self.episode_data_index = {
                "from": virtual_ends - virtual_lengths,
                "to": virtual_ends,
            }
        else:
            # Preserve the exact historical indexing object and execution path for
            # None/identity.  This is intentionally more than numerical parity:
            # no interpolation or virtual-index lookup is entered in this mode.
            self.episode_data_index = self.native_episode_data_index
        self._native_episode_ends = self.native_episode_data_index["to"].tolist()
        self._native_selected_num_frames = (
            self._native_episode_ends[-1] if self._native_episode_ends else 0
        )
        self._episode_ends = self.episode_data_index["to"].tolist()
        self._selected_num_frames = self._episode_ends[-1] if self._episode_ends else 0

        if self.lazy_parquet_loading:
            # Defer parquet existence checks and reads until an episode is sampled.
            # This is the key property needed by corpora with millions of episode
            # files: construction touches metadata only, never every parquet.
            self.hf_dataset = None
        else:
            # Load actual data through the historical Hugging Face backend.
            try:
                self.hf_dataset = self.load_hf_dataset()
            except (AssertionError, FileNotFoundError, NotADirectoryError):
                self.revision = CODEBASE_VERSION
                self.download_episodes(download_videos)
                self.hf_dataset = self.load_hf_dataset()

        # Timestamp synchronization checking is intentionally disabled here.  Do not
        # materialize the full timestamp/episode columns when no check consumes them.

        # Setup delta_indices
        if self.delta_timestamps is not None:
            # check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)


    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def download_episodes(self, download_videos: bool = True) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version. If 'episodes' is given, this
        will only download those episodes (selected by their episode_index). If 'episodes' is None, the whole
        dataset will be downloaded. Thanks to the behavior of snapshot_download, if the files are already present
        in 'local_dir', they won't be downloaded again.
        """
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        files = None
        ignore_patterns = None if download_videos else "videos/"
        if self.episodes is not None:
            files = self.get_episodes_file_paths()

        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def get_episodes_file_paths(self) -> list[Path]:
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        elif self.meta.uses_file_indexed_episodes:
            files = sorted({str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes})
            tables = [pq.read_table(file).replace_schema_metadata(None) for file in files]
            table = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")
            hf_dataset = datasets.Dataset(table, split="train")
            ordered_indices = []
            for ep_idx in self.episodes:
                episode = self.meta.episodes[int(ep_idx)]
                ordered_indices.extend(
                    range(int(episode["dataset_from_index"]), int(episode["dataset_to_index"]))
                )
            hf_dataset = hf_dataset.select(ordered_indices)
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset


    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        if getattr(self, "_episode_segments_enabled", False):
            return self._selected_num_frames
        if getattr(self, "temporal_resampling_enabled", False):
            return self._selected_num_frames
        if self.hf_dataset is not None:
            return len(self.hf_dataset)
        if hasattr(self, "_selected_num_frames"):
            return self._selected_num_frames
        return self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        if getattr(self, "_episode_segments_enabled", False):
            return self._logical_num_episodes
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)

    def _configure_episode_segments(self) -> None:
        """Build a compact logical-episode index over selected physical episodes."""

        selected_sources = (
            [int(index) for index in self.episodes]
            if self.episodes is not None
            else list(range(self.meta.total_episodes))
        )
        source_to_physical_pos = {
            source_episode: position
            for position, source_episode in enumerate(selected_sources)
        }

        validated: list[dict[str, Any]] = []
        for raw_segment in self._requested_episode_segments:
            source_episode = int(raw_segment["episode_index"])
            start = int(raw_segment["start_frame"])
            end = int(raw_segment["end_frame"])
            task = str(raw_segment.get("task", "")).strip()
            if source_episode not in source_to_physical_pos:
                # The physical episode was removed by whitelist/exclusion/split.
                continue
            source_hand_tag = raw_segment.get("source_hand_tag")
            if source_hand_tag is not None:
                source_hand_tag = str(source_hand_tag).strip().lower()
            environment_id = raw_segment.get("environment_id")
            if environment_id is not None:
                environment_id = str(environment_id).strip()
                if not environment_id:
                    environment_id = None
            validated.append(
                {
                    "episode_index": source_episode,
                    "segment_index": int(raw_segment.get("segment_index", 0)),
                    "start_frame": start,
                    "end_frame": end,
                    "task": task,
                    "source_hand_tag": source_hand_tag,
                    "environment_id": environment_id,
                    "codebook_axes": dict(raw_segment.get("codebook_axes") or {}),
                    "bad_reason": raw_segment.get("bad_reason", ""),
                }
            )

        validated.sort(
            key=lambda item: (
                source_to_physical_pos[int(item["episode_index"])],
                int(item["start_frame"]),
                int(item["end_frame"]),
                int(item.get("segment_index", 0)),
            )
        )

        previous_end_by_source: dict[int, int] = {}
        for segment in validated:
            source_episode = int(segment["episode_index"])
            start = int(segment["start_frame"])
            previous_end_by_source[source_episode] = int(segment["end_frame"])

        self._logical_num_episodes = len(validated)
        self._logical_source_episodes = np.asarray(
            [int(segment["episode_index"]) for segment in validated],
            dtype=np.int64,
        )
        self._logical_physical_positions = np.asarray(
            [
                source_to_physical_pos[int(segment["episode_index"])]
                for segment in validated
            ],
            dtype=np.int64,
        )
        self._logical_segment_starts = np.asarray(
            [int(segment["start_frame"]) for segment in validated],
            dtype=np.int64,
        )
        self._logical_segment_ends = np.asarray(
            [int(segment["end_frame"]) for segment in validated],
            dtype=np.int64,
        )
        self._logical_segment_tasks = [str(segment["task"]) for segment in validated]
        self._logical_segment_bad_reasons = [segment["bad_reason"] for segment in validated]
        self._logical_segment_source_hand_tags = [
            (
                str(segment["source_hand_tag"]).strip().lower()
                if segment.get("source_hand_tag") is not None
                else None
            )
            for segment in validated
        ]
        self._logical_segment_environment_ids = [
            (
                str(segment["environment_id"]).strip()
                if segment.get("environment_id") is not None
                else None
            )
            for segment in validated
        ]
        self._logical_segment_codebook_axes = {}
        for axis, registry in getattr(self, "_codebook_registry", {}).items():
            atom_ids = tuple(str(value) for value in registry.get("ids", ()))
            atom_index = {value: index for index, value in enumerate(atom_ids)}
            indices = np.full(len(validated), -1, dtype=np.int32)
            confidences = np.full(len(validated), np.nan, dtype=np.float32)
            for position, segment in enumerate(validated):
                record = segment["codebook_axes"].get(axis)
                if record is None:
                    continue
                atom_id = str(record["id"])
                indices[position] = atom_index[atom_id]
                confidences[position] = float(record["confidence"])
            self._logical_segment_codebook_axes[axis] = {
                "ids": atom_ids,
                "indices": indices,
                "confidences": confidences,
            }
        # Drop the full research records (environment captions, object lists, etc.).
        # Training only needs compact boundary/task arrays; the manifest remains the
        # source of truth for object/environment subset construction.
        self._requested_episode_segments = []
        validated.clear()
        lengths = torch.as_tensor(
            self._logical_segment_ends - self._logical_segment_starts,
            dtype=torch.long,
        )
        ends = torch.cumsum(lengths, dim=0)
        self.episode_data_index = {"from": ends - lengths, "to": ends}

    def _physical_episode_position(self, episode_pos: int) -> int:
        if getattr(self, "_episode_segments_enabled", False):
            return int(self._logical_physical_positions[int(episode_pos)])
        return int(episode_pos)

    def _source_episode_index(self, episode_pos: int) -> int:
        if getattr(self, "_episode_segments_enabled", False):
            return int(self._logical_source_episodes[int(episode_pos)])
        return (
            int(self.episodes[episode_pos])
            if self.episodes is not None
            else int(episode_pos)
        )

    def _source_hand_tag_override(self, episode_pos: int) -> str | None:
        if not getattr(self, "_episode_segments_enabled", False):
            return None
        return self._logical_segment_source_hand_tags[int(episode_pos)]

    def _native_episode_length(self, episode_pos: int) -> int:
        physical_pos = self._physical_episode_position(episode_pos)
        return int(
            self.native_episode_data_index["to"][physical_pos]
            - self.native_episode_data_index["from"][physical_pos]
        )

    def _locate_logical_index(self, idx: int) -> tuple[int, int, int, int]:
        """Map a public row to logical episode, physical position/source, local row."""

        idx = int(idx)
        logical_pos = bisect.bisect_right(self._episode_ends, idx)
        logical_start = 0 if logical_pos == 0 else self._episode_ends[logical_pos - 1]
        segment_local = idx - logical_start
        physical_pos = self._physical_episode_position(logical_pos)
        source_episode = self._source_episode_index(logical_pos)
        native_local = segment_local
        if getattr(self, "_episode_segments_enabled", False):
            native_local += int(self._logical_segment_starts[logical_pos])
        return logical_pos, physical_pos, source_episode, native_local

    @staticmethod
    def _clone_row_value(value):
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.copy()
        return value

    def _locate_virtual_index(self, idx: int) -> tuple[int, int, int]:
        """Map a selected virtual row to (episode position, source episode, row)."""
        if getattr(self, "_episode_segments_enabled", False):
            logical_pos, _, source_episode, native_local = self._locate_logical_index(idx)
            return logical_pos, source_episode, native_local
        idx = int(idx)
        episode_pos = bisect.bisect_right(self._episode_ends, idx)
        episode_start = 0 if episode_pos == 0 else self._episode_ends[episode_pos - 1]
        return episode_pos, self._source_episode_index(episode_pos), idx - episode_start

    def _native_get_row(
        self, episode_pos: int, source_episode: int, native_local_idx: int
    ) -> dict:
        native_local_idx = int(native_local_idx)
        if self.lazy_parquet_loading:
            episode = self._lazy_load_episode(source_episode)
            return {
                key: self._clone_row_value(value[native_local_idx])
                for key, value in episode.items()
            }

        native_global_idx = int(
            self.native_episode_data_index["from"][episode_pos]
        ) + native_local_idx
        return {
            key: self._clone_row_value(value)
            for key, value in self.hf_dataset[native_global_idx].items()
        }

    def _native_select_column(
        self,
        episode_pos: int,
        source_episode: int,
        key: str,
        native_local_indices: Sequence[int] | np.ndarray,
    ) -> torch.Tensor | list:
        local_indices = np.asarray(native_local_indices, dtype=np.int64)
        if self.lazy_parquet_loading:
            values = self._lazy_load_episode(source_episode)[key]
            if isinstance(values, torch.Tensor):
                indices = torch.as_tensor(local_indices, dtype=torch.long)
                return values[indices].clone()
            if isinstance(values, np.ndarray):
                selected = values[local_indices]
            else:
                selected = [values[int(index)] for index in local_indices]
            return [self._clone_row_value(value) for value in selected]

        native_start = int(self.native_episode_data_index["from"][episode_pos])
        selected = self.hf_dataset.select((native_start + local_indices).tolist())[key]
        if not selected:
            return []
        if isinstance(selected[0], torch.Tensor):
            return torch.stack(list(selected)).clone()
        return [self._clone_row_value(value) for value in selected]

    @staticmethod
    def _is_temporally_continuous_key(key: str) -> bool:
        return is_temporally_continuous_feature(key)

    @staticmethod
    def _as_batched_tensor(value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.unsqueeze(0)
        array = np.asarray(value)
        return torch.as_tensor(array).unsqueeze(0)

    def _interpolate_temporal_values(
        self,
        key: str,
        lower: torch.Tensor,
        upper: torch.Tensor,
        alpha: np.ndarray,
    ) -> torch.Tensor:

        return interpolate_continuous_feature(key, lower, upper, alpha)

    @staticmethod
    def _virtual_coordinate_values(
        key: str,
        nearest_values: torch.Tensor | list,
        virtual_local_indices: np.ndarray,
        virtual_global_indices: np.ndarray,
        fps: float,
    ) -> torch.Tensor | list:
        if key not in {"frame_index", "index", "timestamp"}:
            return nearest_values
        values = {
            "frame_index": virtual_local_indices,
            "index": virtual_global_indices,
            "timestamp": virtual_local_indices.astype(np.float64) / float(fps),
        }[key]
        if isinstance(nearest_values, torch.Tensor):
            return torch.as_tensor(
                values,
                dtype=nearest_values.dtype,
                device=nearest_values.device,
            )
        return values.tolist()

    def _resample_column_for_episode(
        self,
        key: str,
        episode_pos: int,
        source_episode: int,
        virtual_local_indices: np.ndarray,
        virtual_global_indices: np.ndarray,
        *,
        expose_virtual_coordinates: bool = True,
    ) -> torch.Tensor | list:
        index_map = build_temporal_index_map(
            self._native_episode_length(episode_pos),
            self.temporal_resampling,
            virtual_indices=virtual_local_indices,
        )
        nearest_values = self._native_select_column(
            episode_pos, source_episode, key, index_map.nearest_indices
        )
        if self._is_temporally_continuous_key(key):
            lower_values = self._native_select_column(
                episode_pos, source_episode, key, index_map.lower_indices
            )
            upper_values = self._native_select_column(
                episode_pos, source_episode, key, index_map.upper_indices
            )
            values = self._interpolate_temporal_values(
                key, lower_values, upper_values, index_map.alpha
            )
        else:
            values = nearest_values
        if expose_virtual_coordinates:
            values = self._virtual_coordinate_values(
                key,
                values,
                virtual_local_indices,
                virtual_global_indices,
                self.fps,
            )
        return values

    def _temporal_select_column(
        self,
        key: str,
        virtual_global_indices: Sequence[int],
        *,
        expose_virtual_coordinates: bool = True,
    ) -> torch.Tensor | list:
        """Read and resample a column while preserving the caller's index order."""
        global_indices = np.asarray(virtual_global_indices, dtype=np.int64)
        if len(global_indices) == 0:
            return []

        grouped: dict[int, dict[str, list[int]]] = {}
        source_episodes: dict[int, int] = {}
        for output_pos, global_idx in enumerate(global_indices.tolist()):
            episode_pos, source_episode, virtual_local_idx = self._locate_virtual_index(
                global_idx
            )
            group = grouped.setdefault(
                episode_pos, {"output": [], "local": [], "global": []}
            )
            group["output"].append(output_pos)
            group["local"].append(virtual_local_idx)
            group["global"].append(global_idx)
            source_episodes[episode_pos] = source_episode

        output_rows: list[Any] = [None] * len(global_indices)
        for episode_pos, group in grouped.items():
            values = self._resample_column_for_episode(
                key,
                episode_pos,
                source_episodes[episode_pos],
                np.asarray(group["local"], dtype=np.int64),
                np.asarray(group["global"], dtype=np.int64),
                expose_virtual_coordinates=expose_virtual_coordinates,
            )
            for input_pos, output_pos in enumerate(group["output"]):
                output_rows[output_pos] = self._clone_row_value(values[input_pos])

        if isinstance(output_rows[0], torch.Tensor):
            return torch.stack(output_rows)
        return output_rows

    def _temporal_get_row(self, idx: int) -> tuple[dict, int, int]:
        episode_pos, source_episode, virtual_local_idx = self._locate_virtual_index(idx)
        index_map = build_temporal_index_map(
            self._native_episode_length(episode_pos),
            self.temporal_resampling,
            virtual_indices=np.asarray([virtual_local_idx], dtype=np.int64),
        )
        nearest_idx = int(index_map.nearest_indices[0])
        lower_idx = int(index_map.lower_indices[0])
        upper_idx = int(index_map.upper_indices[0])
        nearest_row = self._native_get_row(episode_pos, source_episode, nearest_idx)
        lower_row = (
            nearest_row
            if lower_idx == nearest_idx
            else self._native_get_row(episode_pos, source_episode, lower_idx)
        )
        upper_row = (
            nearest_row
            if upper_idx == nearest_idx
            else lower_row
            if upper_idx == lower_idx
            else self._native_get_row(episode_pos, source_episode, upper_idx)
        )
        item = {key: self._clone_row_value(value) for key, value in nearest_row.items()}
        for key in item:
            if not self._is_temporally_continuous_key(key):
                continue
            lower = self._as_batched_tensor(lower_row[key])
            upper = self._as_batched_tensor(upper_row[key])
            item[key] = self._interpolate_temporal_values(
                key, lower, upper, index_map.alpha
            )[0]

        for key in ("frame_index", "index", "timestamp"):
            if key not in item:
                continue
            value = self._virtual_coordinate_values(
                key,
                self._as_batched_tensor(item[key]),
                np.asarray([virtual_local_idx], dtype=np.int64),
                np.asarray([idx], dtype=np.int64),
                self.fps,
            )
            item[key] = value[0]
        return item, episode_pos, source_episode

    def _ensure_lazy_worker_cache(self) -> None:
        """Keep the episode LRU process-local after DataLoader forks/spawns."""
        with self._lazy_cache_lock:
            pid = os.getpid()
            if pid != self._lazy_cache_pid:
                self._lazy_episode_cache = OrderedDict()
                self._lazy_cache_pid = pid

    def __getstate__(self) -> dict:
        """Drop process-local lazy-cache state when a DataLoader uses spawn."""
        state = self.__dict__.copy()
        state["_lazy_episode_cache"] = OrderedDict()
        state["_lazy_cache_lock"] = None
        state["_lazy_cache_pid"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lazy_episode_cache = OrderedDict()
        self._lazy_cache_lock = threading.RLock()
        self._lazy_cache_pid = os.getpid()

    def _lazy_locate_index(self, idx: int) -> tuple[int, int, int]:
        """Map a selected-dataset row to (episode position, source episode, row)."""
        if getattr(self, "_episode_segments_enabled", False):
            logical_pos, _, source_episode, native_local = self._locate_logical_index(idx)
            return logical_pos, source_episode, native_local
        idx = int(idx)
        episode_pos = bisect.bisect_right(self._episode_ends, idx)
        episode_start = 0 if episode_pos == 0 else self._episode_ends[episode_pos - 1]
        source_episode = (
            int(self.episodes[episode_pos]) if self.episodes is not None else episode_pos
        )
        return episode_pos, source_episode, idx - episode_start

    def _lazy_load_episode(self, source_episode: int) -> dict:
        self._ensure_lazy_worker_cache()
        source_episode = int(source_episode)
        with self._lazy_cache_lock:
            cached = self._lazy_episode_cache.pop(source_episode, None)
            if cached is not None:
                self._lazy_episode_cache[source_episode] = cached
                return cached

        path = self.root / self.meta.get_data_file_path(source_episode)
        if self.file_cache is not None:
            path = self.file_cache.stage(path)
        episode_meta = self.meta.episodes[source_episode]
        if "data/chunk_index" in episode_meta and "data/file_index" in episode_meta:
            table = read_packed_parquet_table(path)
        else:
            table = pq.read_table(path)
        if "episode_index" in table.column_names:
            episode_col = table["episode_index"]
            # File-indexed datasets may pack multiple episodes in one parquet.
            if table.num_rows != int(self.meta.episodes[source_episode]["length"]):
                table = table.filter(pc.equal(episode_col, source_episode))
        episode = _arrow_table_to_torch_dict(table)
        # Parquet I/O and Arrow conversion intentionally happen outside the lock.
        # Another thread may load the same episode in parallel; prefer its cached
        # object if so, then mutate/evict the shared LRU atomically.
        with self._lazy_cache_lock:
            cached = self._lazy_episode_cache.pop(source_episode, None)
            if cached is not None:
                episode = cached
            self._lazy_episode_cache[source_episode] = episode
            while len(self._lazy_episode_cache) > self._lazy_cache_size:
                self._lazy_episode_cache.popitem(last=False)
            return episode

    def _lazy_get_row(self, idx: int) -> tuple[dict, int, int]:
        episode_pos, source_episode, local_idx = self._lazy_locate_index(idx)
        episode = self._lazy_load_episode(source_episode)
        item = {}
        for key, value in episode.items():
            row = value[local_idx]
            if isinstance(row, torch.Tensor):
                row = row.clone()
            elif isinstance(row, np.generic):
                row = row.item()
            item[key] = row
        return item, episode_pos, source_episode

    def _lazy_select_column(self, key: str, indices: list[int]) -> torch.Tensor | list:
        rows = []
        for idx in indices:
            _, source_episode, local_idx = self._lazy_locate_index(idx)
            value = self._lazy_load_episode(source_episode)[key][local_idx]
            rows.append(value)
        if not rows:
            return []
        if isinstance(rows[0], torch.Tensor):
            return torch.stack(rows).clone()
        return [row.item() if isinstance(row, np.generic) else row for row in rows]

    def _segment_select_column(self, key: str, indices: list[int]) -> torch.Tensor | list:
        """Select a column through logical segment indices for eager parquet mode."""

        rows = []
        for idx in indices:
            _, physical_pos, source_episode, native_local = self._locate_logical_index(idx)
            row = self._native_get_row(physical_pos, source_episode, native_local)[key]
            rows.append(row)
        if not rows:
            return []
        if isinstance(rows[0], torch.Tensor):
            return torch.stack(rows).clone()
        return [self._clone_row_value(row) for row in rows]

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                if self.temporal_resampling_enabled:
                    timestamps = self._temporal_select_column(
                        "timestamp",
                        query_indices[key],
                        expose_virtual_coordinates=False,
                    )
                    query_timestamps[key] = (
                        timestamps.tolist()
                        if isinstance(timestamps, torch.Tensor)
                        else [float(value) for value in timestamps]
                    )
                elif self._episode_segments_enabled:
                    timestamps = (
                        self._lazy_select_column("timestamp", query_indices[key])
                        if self.lazy_parquet_loading
                        else self._segment_select_column("timestamp", query_indices[key])
                    )
                    query_timestamps[key] = (
                        timestamps.tolist()
                        if isinstance(timestamps, torch.Tensor)
                        else [float(value) for value in timestamps]
                    )
                elif self.lazy_parquet_loading:
                    timestamps = self._lazy_select_column("timestamp", query_indices[key])
                    query_timestamps[key] = timestamps.tolist()
                else:
                    timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                    query_timestamps[key] = _stack_hf_formatted_column(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: _stack_hf_formatted_column(self.hf_dataset.select(q_idx)[key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def _query_hf_dataset_fast(self, query_indices: dict[str, list[int]]) -> dict:
        if self.temporal_resampling_enabled:
            return {
                key: self._temporal_select_column(key, q_idx)
                for key, q_idx in query_indices.items()
                if key not in self.meta.video_keys
                and not ("images" in key and not self.during_training)
            }
        if self._episode_segments_enabled and not self.lazy_parquet_loading:
            return {
                key: self._segment_select_column(key, q_idx)
                for key, q_idx in query_indices.items()
                if key not in self.meta.video_keys
                and not ("images" in key and not self.during_training)
            }
        if self.lazy_parquet_loading:
            return {
                key: self._lazy_select_column(key, q_idx)
                for key, q_idx in query_indices.items()
                if key not in self.meta.video_keys
                and not ("images" in key and not self.during_training)
            }

        result = {}
        processed_indices = set()
        index_to_selected = {}
        for key, q_idx in query_indices.items():
            if key not in self.meta.video_keys :
                if 'images' in key and not self.during_training:
                    continue
                q_idx_tuple = tuple(q_idx)
                if q_idx_tuple not in processed_indices:
                    selected_data = self.hf_dataset.select(q_idx)
                    index_to_selected[q_idx_tuple] = selected_data
                    processed_indices.add(q_idx_tuple)
                else:
                    selected_data = index_to_selected[q_idx_tuple]
                result[key] = _stack_hf_formatted_column(selected_data[key])
        return result

    # no videos
    def get_episode_data(self, episode_id: int) -> dict:
        if self._episode_segments_enabled:
            episode_id = int(episode_id)
            source_episode = self._source_episode_index(episode_id)
            physical_pos = self._physical_episode_position(episode_id)
            start = int(self._logical_segment_starts[episode_id])
            end = int(self._logical_segment_ends[episode_id])
            if self.lazy_parquet_loading:
                episode = self._lazy_load_episode(source_episode)
                result = {}
                for key, value in episode.items():
                    if key in self.meta.video_keys:
                        continue
                    sliced = value[start:end]
                    result[key] = (
                        sliced.clone()
                        if isinstance(sliced, torch.Tensor)
                        else sliced.copy()
                        if isinstance(sliced, np.ndarray)
                        else list(sliced)
                    )
            else:
                local_indices = np.arange(start, end, dtype=np.int64)
                result = {
                    key: self._native_select_column(
                        physical_pos,
                        source_episode,
                        key,
                        local_indices,
                    )
                    for key in self.hf_dataset.column_names
                    if key not in self.meta.video_keys
                }
            source_hand_tag = self._source_hand_tag_override(episode_id)
            result["logical_bad_reason"] = self._logical_segment_bad_reasons[episode_id]
            if source_hand_tag is not None:
                result["logical_source_hand_tag"] = source_hand_tag
            return result
        if self.temporal_resampling_enabled:
            episode_id = int(episode_id)
            source_episode = self._source_episode_index(episode_id)
            if self.lazy_parquet_loading:
                column_names = list(self._lazy_load_episode(source_episode))
            else:
                column_names = list(self.hf_dataset.column_names)
            virtual_from = int(self.episode_data_index["from"][episode_id])
            virtual_to = int(self.episode_data_index["to"][episode_id])
            virtual_global = np.arange(virtual_from, virtual_to, dtype=np.int64)
            virtual_local = np.arange(virtual_to - virtual_from, dtype=np.int64)
            return {
                key: self._resample_column_for_episode(
                    key,
                    episode_id,
                    source_episode,
                    virtual_local,
                    virtual_global,
                )
                for key in column_names
                if key not in self.meta.video_keys
            }
        if self.lazy_parquet_loading:
            source_episode = (
                int(self.episodes[episode_id]) if self.episodes is not None else int(episode_id)
            )
            episode = self._lazy_load_episode(source_episode)
            return {
                key: value.clone() if isinstance(value, torch.Tensor) else value.copy()
                for key, value in episode.items()
                if key not in self.meta.video_keys
            }
        ep_start = self.episode_data_index["from"][episode_id].item()
        ep_end = self.episode_data_index["to"][episode_id].item()
        q_idx = list(range(ep_start, ep_end))
        selected_data = self.hf_dataset[q_idx]
        res_keys = self.meta.features.keys() - set(self.meta.video_keys)
        res = {
            key: _stack_hf_formatted_column(selected_data[key])
            for key in res_keys
        }
        return res

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            if self.file_cache is not None:
                video_path = self.file_cache.stage(video_path)
            offset = self.meta.get_video_timestamp_offset(ep_idx, vid_key)
            query_ts = [float(ts) + offset for ts in query_ts]
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def _add_padding_keys(self, item: dict, padding: dict[str, list[bool]]) -> dict:
        for key, val in padding.items():
            item[key] = torch.BoolTensor(val)
        return item

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        if self._episode_segments_enabled:
            current_ep_idx, physical_pos, ep_idx, native_local = self._locate_logical_index(idx)
            if self.lazy_parquet_loading:
                episode = self._lazy_load_episode(ep_idx)
                item = {
                    key: self._clone_row_value(value[native_local])
                    for key, value in episode.items()
                }
            else:
                item = self._native_get_row(physical_pos, ep_idx, native_local)
        elif self.temporal_resampling_enabled:
            item, current_ep_idx, ep_idx = self._temporal_get_row(idx)
        elif self.lazy_parquet_loading:
            item, current_ep_idx, ep_idx = self._lazy_get_row(idx)
        else:
            item = self.hf_dataset[idx]
            ep_idx = item["episode_index"].item()
            current_ep_idx = self.episodes.index(ep_idx) if self.episodes is not None else ep_idx

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, current_ep_idx)
            query_result = self._query_hf_dataset_fast(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0 and self.during_training:
            current_ts = item["timestamp"].item()
            video_query_indices = query_indices
            if self.temporal_resampling_enabled:
                video_query_indices = dict(query_indices or {})
                for video_key in self.meta.video_keys:
                    video_query_indices.setdefault(video_key, [int(idx)])
            query_timestamps = self._get_query_timestamps(
                current_ts, video_query_indices
            )
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        if self._episode_segments_enabled:
            item["task"] = self._logical_segment_tasks[current_ep_idx]
            item["logical_segment_index"] = torch.tensor(current_ep_idx)
            item["logical_bad_reason"] = self._logical_segment_bad_reasons[current_ep_idx]
            item["segment_start_frame"] = torch.tensor(
                int(self._logical_segment_starts[current_ep_idx])
            )
            item["segment_end_frame"] = torch.tensor(
                int(self._logical_segment_ends[current_ep_idx])
            )
            if "observation.cam_extrinsic" in item:
                # A stable, causal reference for observed camera state. Reuse the
                # cached physical parquet, with native materialized coordinates;
                # never reset this reference to the randomly sampled anchor.
                item["camera_reference_pose7"] = torch.as_tensor(
                    self._native_select_column(physical_pos, ep_idx,
                        "observation.cam_extrinsic",
                        [int(self._logical_segment_starts[current_ep_idx])])[0]
                ).float()
            source_hand_tag = self._source_hand_tag_override(current_ep_idx)
            if source_hand_tag is not None:
                item["logical_source_hand_tag"] = source_hand_tag
        else:
            task_idx = item["task_index"].item()
            item["task"] = self.meta.tasks[task_idx]
        if "coarse_task_index" in item:
            coarse_task_index = item["coarse_task_index"].item()
            item["coarse_task"] = self.meta.tasks[coarse_task_index]

        if "operating_hand_index" in item:
            operating_hand_index = item["operating_hand_index"].item()
            item["operating_hand"] = self.meta.tasks[operating_hand_index]

        if "subtask_annotation" in item and hasattr(self.meta, "annotations"):
            index = item["subtask_annotation"][0].item()
            item["subtask"] = self.meta.annotations["subtask"][index]

        if "atomic_task_index" in item and item["atomic_task_index"] is not None:
            atomic_task_index = item["atomic_task_index"].item()
            item["subtask"] = self.meta.tasks[int(atomic_task_index)]

        return item

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )


    #     self.image_writer = AsyncImageWriter(
    #         num_processes=num_processes,
    #         num_threads=num_threads,
    #     )


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        dataset_dirs: list[str],
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        lazy_parquet_loading: bool = False,
        metadatas: list[LeRobotDatasetMetadata] | None = None,
        temporal_resampling: Mapping[str, Any] | Sequence[Any] | None = None,
        episode_segments: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Any] | None = None,
        codebook_registry: Mapping[str, Mapping[str, Any]] | None = None,
        file_cache: ContentAddressedFileCache | None = None,
    ):
        super().__init__()
        self.dataset_dirs = dataset_dirs
        ds_roots = [Path(ds_dir) for ds_dir in dataset_dirs]
        ds_names = [ds_dir for ds_dir in dataset_dirs]
        self.ds_names = ds_names
        self.ds_roots = ds_roots
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(ds_names, 0.0001)
        if temporal_resampling is None:
            temporal_specs = [None] * len(dataset_dirs)
        elif isinstance(temporal_resampling, Mapping):
            temporal_specs = [temporal_resampling[name] for name in ds_names]
        elif isinstance(temporal_resampling, Sequence) and not isinstance(
            temporal_resampling, (str, bytes)
        ):
            temporal_specs = list(temporal_resampling)
        if episode_segments is None:
            segment_specs = [None] * len(dataset_dirs)
        elif isinstance(episode_segments, Mapping):
            segment_specs = [episode_segments[name] for name in ds_names]
        elif isinstance(episode_segments, Sequence) and not isinstance(
            episode_segments, (str, bytes)
        ):
            segment_specs = list(episode_segments)
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self._datasets = []
        metadata_iter = metadatas if metadatas is not None else [None] * len(ds_roots)
        for ds_root, ds_name, metadata, temporal_spec, segment_spec in zip(
            ds_roots,
            ds_names,
            metadata_iter,
            temporal_specs,
            segment_specs,
            strict=True,
        ):
            _dataset = LeRobotDataset(
                ds_name,
                root=ds_root,
                episodes=episodes[ds_name] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[ds_name],
                download_videos=download_videos,
                video_backend=video_backend,
                lazy_parquet_loading=lazy_parquet_loading,
                metadata=metadata,
                temporal_resampling=temporal_spec,
                episode_segments=segment_spec,
                codebook_registry=codebook_registry,
                file_cache=file_cache,
            )
            self._datasets.append(_dataset)

        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        for ds_name, ds in zip(self.ds_names, self._datasets, strict=True):
            # Disable warning of empty extra keys do
            extra_keys = set(ds.features).difference(intersection_features)
            if extra_keys:
                logging.warning(
                    f"keys {extra_keys} of {ds_name} were disabled as they are not contained in all the "
                    "other datasets."
                )
            self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.temporal_resampling = [
            dataset.temporal_resampling for dataset in self._datasets
        ]
        self.temporal_resampling_enabled = any(
            dataset.temporal_resampling_enabled for dataset in self._datasets
        )
        # TODO(rcadene, aliberts): We should not perform this aggregation for datasets
        # with multiple robots of different ranges. Instead we should have one normalization
        # per robot.
        available_stats = [dataset.meta.stats for dataset in self._datasets if dataset.meta.stats]
        self.stats = aggregate_stats(available_stats) if available_stats else {}

    def set_during_training(self, during_training: bool):
        for dataset in self._datasets:
            dataset.during_training = during_training

    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.ds_names)}

    @property
    def repo_index_to_id(self):
        """Return the inverse mapping if repo_id_to_index."""
        return {v: k for k, v in self.repo_id_to_index}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update({k: v for k, v in dataset.hf_features.items() if k not in self.disabled_features})
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image, VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def get_episode_data(self, episode_idx: int) -> dict:
        for dataset in self._datasets:
            if episode_idx < dataset.num_episodes:
                if (
                    dataset.lazy_parquet_loading
                    or dataset.temporal_resampling_enabled
                    or dataset._episode_segments_enabled
                ):
                    return dataset.get_episode_data(episode_idx)
                ep_idx = (
                    int(dataset.episodes[episode_idx])
                    if dataset.episodes is not None
                    else int(episode_idx)
                )
                file = str(dataset.root / dataset.meta.get_data_file_path(ep_idx))
                table = pq.read_table(str(file))
                if "episode_index" in table.column_names:
                    table = table.filter(pc.equal(table["episode_index"], ep_idx))
                return _arrow_table_to_torch_dict(table)
            else:
                episode_idx -= dataset.num_episodes

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            pass
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Dataset Names: '{self.ds_names}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
