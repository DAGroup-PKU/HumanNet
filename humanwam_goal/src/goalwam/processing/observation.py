# Portions Copyright (c) 2026 phyai team. MIT License; see licenses/phyai-MIT.txt.
"""Camera processing and action conditioning."""

import math
from dataclasses import dataclass

import numpy as np
import torch

from .images import goalwam_canvas, select_goalwam_history


@dataclass(frozen=True)
class ProcessorConfig:
    camera_sizes: tuple[tuple[int, int], ...] = ((192, 320), (224, 288))
    cameras: tuple[str, ...] = ("head", "left", "right")
    goal_head_only: bool = True
    history_offsets_seconds: tuple[float, ...] = (0.5, 2.0)
    history_tolerance_seconds: float = 0.1
    horizon: int = 48
    action_dim: int = 64
    active_action_dim: int = 42
    output_action_dim: int = 49
    prefix_length: int = 7
    video_fps: float = 7.5
    action_fps: float = 30.0

    def __post_init__(self):
        if not self.camera_sizes or any(
            (
                len(size) != 2 or any((type(v) is not int or v <= 0 or v % 32 for v in size))
                for size in self.camera_sizes
            )
        ):
            raise ValueError("Camera sizes must be positive H,W multiples of 32")
        if "head" not in self.cameras or set(self.cameras) - {"torso", "head", "left", "right"}:
            raise ValueError("Select head and optional torso/left/right cameras")
        if any(
            (
                not math.isfinite(v) or v <= 0
                for v in (*self.history_offsets_seconds, self.video_fps, self.action_fps)
            )
        ):
            raise ValueError("Frame rates and history offsets must be finite and positive")
        if any(
            (a >= b for a, b in zip(self.history_offsets_seconds, self.history_offsets_seconds[1:]))
        ):
            raise ValueError("History offsets must be strictly increasing")
        if not math.isfinite(self.history_tolerance_seconds) or self.history_tolerance_seconds < 0:
            raise ValueError("History tolerance must be finite and nonnegative")
        if (
            not 0 <= self.prefix_length < self.horizon
            or not 0 < self.active_action_dim <= self.output_action_dim <= self.action_dim
        ):
            raise ValueError("Invalid action dimensions or RTC prefix length")


class ObservationProcessor:
    def __init__(self, config: ProcessorConfig):
        self.config = config

    def __call__(self, transition):
        c = self.config
        source = np.asarray(transition["images"]["head"])
        if source.ndim != 3 or min(source.shape[:2]) < 1:
            raise ValueError("A current head RGB image is required")
        source_hw = transition.get("source_hw", source.shape[:2])
        if len(source_hw) != 2 or any((type(v) is not int or v <= 0 for v in source_hw)):
            raise ValueError("source_hw must contain positive integer H,W dimensions")
        h, w = source_hw

        def padding(size):
            th, tw = size
            scale = min(th / h, tw / w)
            return 1 - h * scale * (w * scale) / (th * tw)

        size = min(c.camera_sizes, key=padding)
        history = select_goalwam_history(
            transition.get("history", ()),
            float(transition["observation_timestamp"]),
            c.history_offsets_seconds,
            c.history_tolerance_seconds,
        )
        frames = (
            [(transition["goal_images"], c.goal_head_only)]
            + [(capture["images"], False) for capture, _ in history]
            + [(transition["images"], False)]
        )
        canvases = tuple(
            (
                goalwam_canvas(images, cell_size=size, cameras=c.cameras, head_only=head_only)
                for images, head_only in frames
            )
        )
        pixels = tuple(
            ((canvas.float() * (1.0 / 127.5) - 1.0)[None, :, None] for canvas in canvases)
        )
        state = torch.as_tensor(transition["normalized_state"], dtype=torch.float32)
        if (
            state.ndim != 1
            or not c.active_action_dim <= state.numel() <= c.action_dim
            or (not torch.isfinite(state).all())
        ):
            raise ValueError("normalized_state must be a finite model-coordinate vector")
        action = torch.zeros(1, c.horizon + 1, c.action_dim, dtype=torch.float32)
        action[0, 0, : state.numel()] = state
        valid = torch.zeros_like(action, dtype=torch.bool)
        for field, section in (
            ("state_valid_mask", slice(0, 1)),
            ("action_valid_mask", slice(1, None)),
        ):
            mask = torch.as_tensor(
                transition.get(field, [True] * c.active_action_dim), dtype=torch.bool
            )
            if (
                mask.ndim != 1
                or not c.active_action_dim <= mask.numel() <= c.action_dim
                or mask[c.active_action_dim :].any()
            ):
                raise ValueError(f"{field} must disable padded action channels")
            valid[0, section, : mask.numel()] = mask
        prefix = torch.as_tensor(transition.get("normalized_prefix", []), dtype=torch.float32)
        if prefix.numel() and prefix.ndim != 2:
            raise ValueError("normalized_prefix must be a matrix")
        length = 0 if prefix.numel() == 0 else prefix.shape[0]
        if length:
            if (
                prefix.ndim != 2
                or not c.active_action_dim <= prefix.shape[1] <= c.action_dim
                or length > c.prefix_length
                or (not torch.isfinite(prefix).all())
            ):
                raise ValueError(
                    "normalized_prefix must fit the checkpoint's RTC prefix and action width"
                )
            action[0, 1 : 1 + length, : prefix.shape[1]] = prefix
        noisy = torch.ones(1, c.horizon + 1, dtype=torch.bool)
        noisy[:, : 1 + length] = False
        return {
            "pixels": pixels,
            "history_ages_seconds": tuple((age for _, age in history)),
            "action": action.masked_fill(~valid, 0),
            "action_noisy_mask": noisy,
            "action_valid_mask": valid,
            "action_domains": torch.full(
                (c.horizon + 1,), int(transition["domain_id"]), dtype=torch.long
            ),
            "seed": int(transition.get("seed", 0)),
            "video_fps": c.video_fps,
            "action_fps": c.action_fps,
        }
