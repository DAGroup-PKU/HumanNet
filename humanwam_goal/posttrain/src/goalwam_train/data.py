"""RoboTwin 2.0 training windows read from LeRobot v3.0 `clean_50` datasets."""

import json
import math
import os
import random
from collections import OrderedDict
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from goalwam.processing.geometry import (
    propagate_relative_action_validity,
    propagate_state_arm_eef_validity,
    to_relative_action,
    transform_state_arm_eef_coordinate,
    zero_chassis_state,
)
from goalwam.processing.images import goalwam_canvas
from goalwam.processing.observation import ProcessorConfig
from goalwam.processing.robot import RobotProcessor
from goalwam.processing.text import InstructionTokenizer

from .goal_cache import TrainGoalCache
from .normalization import ACTION_MASK, STATE_MASK, discover_clean_datasets

CAMERAS = {
    "head": "observation.images.cam_high",
    "left": "observation.images.cam_left_wrist",
    "right": "observation.images.cam_right_wrist",
}
CAPTION_DROPOUT = 0.10
PREGENERATED_GOAL_PROB = 0.80
HISTORY_DROP_ALL = 0.15
HISTORY_DROP_RECENT = 0.10
HISTORY_JITTER_PROB = 0.20
HISTORY_JITTER_SCALE = (0.8, 1.2)


def read_processor(checkpoint):
    config = json.loads((Path(checkpoint) / "processor.json").read_text())
    return ProcessorConfig(
        **{
            k: tuple(map(tuple, v)) if k == "camera_sizes" else tuple(v) if isinstance(v, list) else v
            for k, v in config.items()
        }
    )


class Episodes:
    """Every episode of every RoboTwin `aloha_agilex` clean LeRobot v3.0 dataset below `root`."""

    def __init__(self, root):
        self.root = Path(root)
        self.datasets = discover_clean_datasets(self.root)
        self.episodes = []
        for index, (path, _) in enumerate(self.datasets):
            for file in sorted((path / "meta/episodes").glob("chunk-*/*.parquet")):
                columns = [c for c in pq.ParquetFile(file).schema_arrow.names if not c.startswith("stats/")]
                for episode in pq.read_table(file, columns=columns).to_pylist():
                    if episode["length"] >= 1:
                        self.episodes.append((index, episode))
        if not self.episodes:
            raise ValueError(f"No LeRobot v3.0 episodes found under {root}")
        self._tables = OrderedDict()

    def rows(self, index, episode):
        path, info = self.datasets[index]
        file = path / info["data_path"].format(
            chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"]
        )
        if file not in self._tables:
            table = pq.read_table(
                file, columns=["episode_index", "frame_index", "observation.state", "action"]
            )
            self._tables[file] = {
                "episode": table["episode_index"].to_numpy(),
                "frame": table["frame_index"].to_numpy(),
                "state": np.stack(table["observation.state"].to_numpy(zero_copy_only=False)),
                "action": np.stack(table["action"].to_numpy(zero_copy_only=False)),
            }
            while len(self._tables) > 4:
                self._tables.popitem(last=False)
        table = self._tables[file]
        selected = np.flatnonzero(table["episode"] == episode["episode_index"])
        selected = selected[np.argsort(table["frame"][selected])]
        if len(selected) != episode["length"]:
            raise ValueError(f"{path}: episode {episode['episode_index']} has missing rows")
        return table["state"][selected].astype(np.float32), table["action"][selected].astype(np.float32)


def model_state_and_actions(state, actions, action_time_valid=None):
    """Physical state `[49]` and absolute commands `[T, 49]` -> model coordinates and validity."""
    state_mask = torch.from_numpy(STATE_MASK)
    state = torch.from_numpy(state).masked_fill(~state_mask, 0)
    action_mask = torch.from_numpy(ACTION_MASK).expand(actions.shape).clone()
    if action_time_valid is not None:
        action_mask &= torch.from_numpy(np.asarray(action_time_valid, dtype=bool))[:, None]
    actions = torch.from_numpy(actions).masked_fill(~action_mask, 0)
    relative_mask = propagate_relative_action_validity(action_mask, state_mask)
    relative = to_relative_action(state, actions)
    model_mask = propagate_state_arm_eef_validity(state_mask, "head_camera")
    relative_mask[..., 42:] = False
    model_mask[42:] = False
    model_state = transform_state_arm_eef_coordinate(state, "head_camera")
    model_state = zero_chassis_state(model_state).masked_fill(~model_mask, 0)
    return model_state, model_mask, relative.masked_fill(~relative_mask, 0), relative_mask


def tokenize_instruction(tokenizer, text):
    """Tokenize an instruction or an empty string (for caption CFG dropout)."""
    if text.strip():
        return tokenizer(text)[0][0]
    raw = tokenizer.tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], tokenize=True, add_generation_prompt=True
    )
    if isinstance(raw, dict) or hasattr(raw, "keys"):
        raw = raw["input_ids"]
    if isinstance(raw, torch.Tensor):
        raw = raw.flatten().tolist()
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    return torch.tensor([*raw, tokenizer.tokenizer.eos_token_id, tokenizer.vision_start], dtype=torch.long)


def sample_recorded_history(t, fps, offsets, tolerance, rng):
    """Sample causal history frame rows and true ages `(rows, ages)` with dropout and jitter."""
    if not offsets:
        return [], ()
    slots = list(range(len(offsets)))
    draw = rng.random()
    if draw < HISTORY_DROP_ALL:
        return [], ()
    if draw < HISTORY_DROP_ALL + HISTORY_DROP_RECENT:
        slots = slots[2:]
    scale = rng.uniform(*HISTORY_JITTER_SCALE) if rng.random() < HISTORY_JITTER_PROB else 1.0
    rows, ages = [], []
    for slot in slots:
        requested = offsets[slot] * scale
        row = t - math.ceil(requested * fps - 1e-9)
        actual = (t - row) / fps
        if row >= 0 and row not in rows and actual - requested <= tolerance + 1e-9:
            rows.append(row)
            ages.append(actual)
    return rows, tuple(ages)


class VideoReader:
    def __init__(self, path, fps):
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.stream.thread_count = 1
        self.fps = fps

    def read(self, indices):
        """Exact frames by presentation time; indices are sorted frame numbers in the file."""
        wanted, frames, groups = set(indices), {}, []
        for index in indices:
            if not groups or index - groups[-1][-1] > 32:
                groups.append([])
            groups[-1].append(index)
        for group in groups:
            self.container.seek(int(group[0] / self.fps / self.stream.time_base), stream=self.stream)
            for frame in self.container.decode(self.stream):
                index = round(float(frame.pts * self.stream.time_base) * self.fps)
                if index in wanted:
                    frames[index] = frame.to_ndarray(format="rgb24")
                if index >= group[-1]:
                    break
        if len(frames) != len(wanted):
            raise ValueError(f"Missing video frames {sorted(wanted - frames.keys())}")
        return [frames[i] for i in indices]


class RoboTwinDataset(torch.utils.data.Dataset):
    """One window: instruction, pre-generated/terminal goal, history, 1+4k frames and 48 commands."""

    def __init__(self, data_root, checkpoint, robot_json, *, goal_cache=None, seed=42):
        self.processor = read_processor(checkpoint)
        self.horizon = self.processor.horizon
        self.episodes = Episodes(data_root)
        self.robot = RobotProcessor(robot_json)
        self.tokenizer_path = Path(checkpoint) / "text_tokenizer"
        self.tokenizer = None
        self.goal_cache = TrainGoalCache(goal_cache) if goal_cache is not None else None
        self.tasks = {}
        if self.goal_cache is not None:
            for idx, episode in self.episodes.episodes:
                path, _ = self.episodes.datasets[idx]
                task = self.goal_cache.resolve_task(path, episode)
                self.goal_cache.has(task, episode["episode_index"], episode["tasks"][0])
                self.tasks[(idx, episode["episode_index"])] = task
        self.windows = np.cumsum([e["length"] for _, e in self.episodes.episodes])
        self.readers = OrderedDict()
        self.seed = seed
        self._visits = 0

    def __len__(self):
        return int(self.windows[-1])

    def frames(self, index, episode, indices):
        path, info = self.episodes.datasets[index]
        fps = float(info["fps"])
        result = {}
        for name in self.processor.cameras:
            key = CAMERAS[name]
            file = path / info["video_path"].format(
                video_key=key,
                chunk_index=episode[f"videos/{key}/chunk_index"],
                file_index=episode[f"videos/{key}/file_index"],
            )
            if file not in self.readers:
                self.readers[file] = VideoReader(file, fps)
                while len(self.readers) > 8:
                    self.readers.popitem(last=False)[1].container.close()
            offset = round(episode[f"videos/{key}/from_timestamp"] * fps)
            result[name] = self.readers[file].read([offset + i for i in indices])
        return [{name: views[k] for name, views in result.items()} for k in range(len(indices))]

    def __getitem__(self, item):
        c = self.processor
        self._visits += 1
        rng = random.Random(hash((self.seed, os.getpid(), int(item), self._visits)))
        slot = int(np.searchsorted(self.windows, item, side="right"))
        index, episode = self.episodes.episodes[slot]
        t = int(item) - (int(self.windows[slot - 1]) if slot else 0)
        length = int(episode["length"])
        fps = float(self.episodes.datasets[index][1]["fps"])
        if fps != c.action_fps:
            raise ValueError(f"Datasets must be recorded at {c.action_fps} FPS")
        stride = round(c.action_fps / c.video_fps)

        history, history_ages = sample_recorded_history(
            t, fps, c.history_offsets_seconds, c.history_tolerance_seconds, rng
        )
        video_blocks = min(self.horizon // (stride * 4), (length - 1 - t) // (stride * 4))
        rollout = [t + stride * k for k in range(1 + 4 * video_blocks)]

        instruction = episode["tasks"][0]
        eid = int(episode["episode_index"])
        use_pregenerated = (
            self.goal_cache is not None
            and self.goal_cache.has(self.tasks[(index, eid)], eid, instruction)
            and rng.random() < PREGENERATED_GOAL_PROB
        )
        terminal = length - 1
        needed = sorted(set(history + rollout + ([] if use_pregenerated else [terminal])))
        views = dict(zip(needed, self.frames(index, episode, needed), strict=True))
        h, w = views[t]["head"].shape[:2]
        size = min(c.camera_sizes, key=lambda s: 1 - min(s[0] / h, s[1] / w) ** 2 * h * w / (s[0] * s[1]))

        def canvas(row, head_only=False):
            return goalwam_canvas(views[row], cell_size=size, cameras=c.cameras, head_only=head_only)

        if use_pregenerated:
            goal_rgb = self.goal_cache.read(self.tasks[(index, eid)], eid, instruction)
            goal_canvas = goalwam_canvas({"head": goal_rgb}, cell_size=size, cameras=c.cameras, head_only=True)
        else:
            goal_canvas = canvas(terminal, c.goal_head_only)

        state, action = self.episodes.rows(index, episode)
        action_idx = t + np.arange(self.horizon)
        time_valid = action_idx < length
        model_state, state_mask, relative, relative_mask = model_state_and_actions(
            state[t], action[np.minimum(action_idx, length - 1)], time_valid
        )
        values = torch.zeros(self.horizon + 1, c.action_dim)
        valid = torch.zeros(self.horizon + 1, c.action_dim, dtype=torch.bool)
        values[0, :49] = self.robot.normalize(model_state, "state", state_mask)
        values[1:, :49] = self.robot.normalize(relative, "action", relative_mask)
        valid[0, :49], valid[1:, :49] = state_mask, relative_mask

        if self.tokenizer is None:
            self.tokenizer = InstructionTokenizer(self.tokenizer_path)
        text = "" if rng.random() < CAPTION_DROPOUT else instruction
        return {
            "ids": tokenize_instruction(self.tokenizer, text),
            "goal": goal_canvas,
            "history": [canvas(row) for row in history],
            "history_ages": history_ages,
            "video": torch.stack([canvas(row) for row in rollout]),
            "action": values,
            "valid": valid,
            "domain": self.robot.config["domain_id"],
            "video_fps": c.video_fps,
            "action_fps": c.action_fps,
        }


def infinite_loader(dataset, batch_size, seed, workers):
    """Uniform random windows, independent per rank."""
    generator = torch.Generator().manual_seed(seed)
    sampler = torch.utils.data.RandomSampler(
        dataset, replacement=True, num_samples=2**62, generator=generator
    )
    return iter(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            collate_fn=list,
            persistent_workers=workers > 0,
            prefetch_factor=2 if workers > 0 else None,
        )
    )
