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
import io
import json
from collections.abc import Iterator
from itertools import accumulate
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import datasets
import jsonlines
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image as PILImage
from torchvision import transforms

from functools import partial


DEFAULT_CHUNK_SIZE = 1000  # Max number of episodes per chunk

INFO_PATH = "meta/info.json"
EPISODES_PATH = "meta/episodes.jsonl"
STATS_PATH = "meta/stats.json"
EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
TASKS_PATH = "meta/tasks.jsonl"

ANNOTATION_PATHS = {
    "subtask": "annotations/subtask_annotations.jsonl",
    "scene": "annotations/scene_annotations.jsonl",
    "gripper_mode": "annotations/gripper_mode_annotation.jsonl",
    "gripper_activity": "annotations/gripper_activity_annotation.jsonl",
    "eef_velocity": "annotations/eef_velocity_annotation.jsonl",
    "eef_acc_mag": "annotations/eef_acc_mag_annotation.jsonl",
    "eef_direction": "annotations/eef_direction_annotation.jsonl",
}

DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
DEFAULT_PARQUET_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_IMAGE_PATH = "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.jpeg"


DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "coarse_task_index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
    "coarse_quality_index": {"dtype": "int64", "shape": (1,), "names": None},
    "quality_index": {"dtype": "int64", "shape": (1,), "names": None},
}


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dictionary structure by collapsing nested keys into one key with a separator.

    For example:
    ```
    >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}`
    >>> print(flatten_dict(dct))
    {"a/b": 1, "a/c/d": 2, "e": 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict


#     for key in split_keys[1:]:
#         getter = getter[key]

#     return getter


def load_json(fpath: Path) -> Any:
    with open(fpath) as f:
        return json.load(f)


def load_jsonlines(fpath: Path) -> list[Any]:
    # allow \t and \n
    loose_loader = partial(json.loads, strict=False)
    with jsonlines.open(fpath, "r", loads=loose_loader) as reader:
        return list(reader)


def load_info(local_dir: Path) -> dict:
    info = load_json(local_dir / INFO_PATH)
    for ft in info["features"].values():
        ft["shape"] = tuple(ft["shape"])
    return info


def cast_stats_to_numpy(stats) -> dict[str, dict[str, np.ndarray]]:
    stats = {key: np.array(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


def load_stats(local_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    if not (local_dir / STATS_PATH).exists():
        return None
    stats = load_json(local_dir / STATS_PATH)
    return cast_stats_to_numpy(stats)


def _load_v3_parquet_records(local_dir: Path, relative_glob: str) -> list[dict]:
    records: list[dict] = []
    for fpath in sorted(local_dir.glob(relative_glob)):
        table = pq.read_table(fpath)
        records.extend(table.to_pylist())
    return records


def load_tasks(local_dir: Path) -> tuple[dict, dict]:
    tasks_path = local_dir / TASKS_PATH
    if tasks_path.exists():
        task_rows = load_jsonlines(tasks_path)
    else:
        v3_tasks_path = local_dir / "meta/tasks.parquet"
        if not v3_tasks_path.exists():
            task_rows = load_jsonlines(tasks_path)
        else:
            task_rows = []
            for row in pq.read_table(v3_tasks_path).to_pylist():
                task = row.get("task") or row.get("__index_level_0__")
                task_rows.append({"task_index": int(row["task_index"]), "task": task})

    tasks = {item["task_index"]: item["task"] for item in sorted(task_rows, key=lambda x: x["task_index"])}
    task_to_task_index = {task: task_index for task_index, task in tasks.items()}
    return tasks, task_to_task_index

def load_annotations(local_dir: Path) -> dict[str, dict[int, str]]:
    annotations = {}
    for key, path in ANNOTATION_PATHS.items():
        anno = load_jsonlines(local_dir / path)
        anno = {item[f"{key}_index"]: item[key] for item in sorted(anno, key=lambda x: x[f"{key}_index"])}
        annotations[key] = anno
    return annotations


def load_episodes(local_dir: Path) -> dict:
    episodes_path = local_dir / EPISODES_PATH
    if episodes_path.exists():
        episodes = load_jsonlines(episodes_path)
    else:
        episodes = _load_v3_parquet_records(local_dir, "meta/episodes/**/*.parquet")
        if not episodes:
            episodes = load_jsonlines(episodes_path)
    return {int(item["episode_index"]): item for item in sorted(episodes, key=lambda x: x["episode_index"])}


def load_episodes_stats(local_dir: Path) -> dict:
    episodes_stats_path = local_dir / EPISODES_STATS_PATH
    if episodes_stats_path.exists():
        episodes_stats = load_jsonlines(episodes_stats_path)
        return {
            int(item["episode_index"]): cast_stats_to_numpy(item["stats"])
            for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
        }

    episodes = _load_v3_parquet_records(local_dir, "meta/episodes/**/*.parquet")
    if not episodes:
        episodes_stats = load_jsonlines(episodes_stats_path)
        return {
            int(item["episode_index"]): cast_stats_to_numpy(item["stats"])
            for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
        }

    result = {}
    for row in sorted(episodes, key=lambda x: x["episode_index"]):
        flat_stats = {
            key[len("stats/"):]: value
            for key, value in row.items()
            if key.startswith("stats/")
        }
        result[int(row["episode_index"])] = cast_stats_to_numpy(unflatten_dict(flat_stats))
    return result


def backward_compatible_episodes_stats(
    stats: dict[str, dict[str, np.ndarray]], episodes: list[int]
) -> dict[str, dict[str, np.ndarray]]:
    return dict.fromkeys(episodes, stats)


def hf_transform_to_torch(items_dict: dict[torch.Tensor | None]):
    """Get a transform function that convert items from Hugging Face dataset (pyarrow)
    to torch tensors. Importantly, images are converted from PIL, which corresponds to
    a channel last representation (h w c) of uint8 type, to a torch image representation
    with channel first (c h w) of float32 type in range [0,1].
    """
    for key in items_dict:
        first_item = items_dict[key][0]
        if type(first_item) == dict and 'bytes' in first_item and first_item['bytes'] is not None:
            to_pil = lambda img: PILImage.open(io.BytesIO(img['bytes']))
            items_dict[key] = [to_pil(img) for img in items_dict[key]]
            first_item = items_dict[key][0]
        if isinstance(first_item, PILImage.Image):
            to_tensor = transforms.ToTensor()
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        else:
            items_dict[key] = [x if isinstance(x, str) else torch.tensor(x) for x in items_dict[key]]
    return items_dict


def get_hf_features_from_features(features: dict) -> datasets.Features:
    hf_features = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue
        elif ft["dtype"] == "image":
            hf_features[key] = datasets.Image()
        elif ft["shape"] == (1,):
            hf_features[key] = datasets.Value(dtype=ft["dtype"])
        elif len(ft["shape"]) == 1:
            hf_features[key] = datasets.Sequence(
                length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"])
            )
        elif len(ft["shape"]) == 2:
            hf_features[key] = datasets.Array2D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 3:
            hf_features[key] = datasets.Array3D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 4:
            hf_features[key] = datasets.Array4D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 5:
            hf_features[key] = datasets.Array5D(shape=ft["shape"], dtype=ft["dtype"])

    return datasets.Features(hf_features)


#             names = ft["names"]
#             # Backward compatibility for "channel" which is an error introduced in LeRobotDataset v2.0 for ported datasets.
#             if names[2] in ["channel", "channels"]:  # (h, w, c) -> (c, h, w)
#                 shape = (shape[2], shape[0], shape[1])
#         elif key == "observation.environment_state":
#             type = FeatureType.ENV
#         elif key.startswith("observation"):
#             type = FeatureType.STATE
#         elif key.startswith("action"):
#             type = FeatureType.ACTION
#         else:
#             continue

#         policy_features[key] = PolicyFeature(
#             type=type,
#             shape=shape,
#         )

#     return policy_features


def get_episode_data_index(
    episode_dicts: dict[dict], episodes: list[int] | None = None
) -> dict[str, torch.Tensor]:
    episode_lengths = {ep_idx: ep_dict["length"] for ep_idx, ep_dict in episode_dicts.items()}
    if episodes is not None:
        episode_lengths = {ep_idx: episode_lengths[ep_idx] for ep_idx in episodes}

    cumulative_lengths = list(accumulate(episode_lengths.values()))
    return {
        "from": torch.LongTensor([0] + cumulative_lengths[:-1]),
        "to": torch.LongTensor(cumulative_lengths),
    }


def get_delta_indices(delta_timestamps: dict[str, list[float]], fps: int) -> dict[str, list[int]]:
    delta_indices = {}
    for key, delta_ts in delta_timestamps.items():
        delta_indices[key] = [round(d * fps) for d in delta_ts]

    return delta_indices


def cycle(iterable):
    """The equivalent of itertools.cycle, but safe for Pytorch dataloaders.

    See https://github.com/pytorch/pytorch/issues/23900 for information on why itertools.cycle is not safe.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class IterableNamespace(SimpleNamespace):
    """
    A namespace object that supports both dictionary-like iteration and dot notation access.
    Automatically converts nested dictionaries into IterableNamespaces.

    This class extends SimpleNamespace to provide:
    - Dictionary-style iteration over keys
    - Access to items via both dot notation (obj.key) and brackets (obj["key"])
    - Dictionary-like methods: items(), keys(), values()
    - Recursive conversion of nested dictionaries

    Args:
        dictionary: Optional dictionary to initialize the namespace
        **kwargs: Additional keyword arguments passed to SimpleNamespace

    Examples:
        >>> data = {"name": "Alice", "details": {"age": 25}}
        >>> ns = IterableNamespace(data)
        >>> ns.name
        'Alice'
        >>> ns.details.age
        25
        >>> list(ns.keys())
        ['name', 'details']
        >>> for key, value in ns.items():
        ...     print(f"{key}: {value}")
        name: Alice
        details: IterableNamespace(age=25)
    """

    def __init__(self, dictionary: dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if dictionary is not None:
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    setattr(self, key, IterableNamespace(value))
                else:
                    setattr(self, key, value)

    def __iter__(self) -> Iterator[str]:
        return iter(vars(self))

    def __getitem__(self, key: str) -> Any:
        return vars(self)[key]

    def items(self):
        return vars(self).items()

    def values(self):
        return vars(self).values()

    def keys(self):
        return vars(self).keys()
