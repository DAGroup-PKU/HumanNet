# Portions Copyright (c) 2026 phyai team. MIT License; see licenses/phyai-MIT.txt.
"""RGB camera canvases and causal visual-history selection for inference."""

import bisect
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise

import numpy as np
import torch
from torch.nn import functional as F


def select_goalwam_history(
    captures: Sequence[Mapping],
    current_time: float,
    offsets: tuple[float, ...],
    tolerance: float,
) -> tuple[tuple[Mapping, float], ...]:
    times = [float(capture["timestamp"]) for capture in captures]
    if not math.isfinite(current_time) or any(not math.isfinite(t) for t in times):
        raise ValueError("Observation timestamps must be finite")
    if any(a >= b for a, b in pairwise(times)) or any(t >= current_time for t in times):
        raise ValueError("History must be oldest first and earlier than the current frame")
    selected, used = [], set()
    for age in offsets:
        target = current_time - age
        index = bisect.bisect_right(times, target + 1e-9) - 1
        if index < 0 or index in used or target - times[index] > tolerance + 1e-9:
            continue
        used.add(index)
        selected.append((captures[index], current_time - times[index]))
    return tuple(selected)


def goalwam_canvas(
    images: Mapping[str, np.ndarray],
    *,
    cell_size: tuple[int, int],
    cameras: tuple[str, ...],
    head_only: bool = False,
) -> torch.Tensor:
    """Build one CHW uint8 canvas; absent wrist cameras retain black cells."""
    height, width = cell_size
    boxes, y = {}, 0
    for name in ("torso", "head"):
        if name in cameras:
            boxes[name] = (y, 0, height, width)
            y += height
    for name, x in (("left", 0), ("right", width // 2)):
        if name in cameras:
            boxes[name] = (y, x, height // 2, width // 2)
    if any(name in cameras for name in ("left", "right")):
        y += height // 2
    canvas_height, canvas_width = math.ceil(y / 32) * 32, math.ceil(width / 32) * 32
    canvas = torch.zeros(3, canvas_height, canvas_width, dtype=torch.uint8)
    for name, (top, left, h, w) in boxes.items():
        if head_only and name in ("left", "right"):
            continue
        value = images.get(name)
        if value is None:
            if name in ("left", "right"):
                continue
            raise ValueError(f"Required {name} RGB image is missing")
        value = np.asarray(value)
        if (
            value.dtype != np.uint8
            or value.ndim != 3
            or value.shape[-1] != 3
            or min(value.shape[:2]) < 1
        ):
            raise ValueError(f"{name} must be a nonempty HWC uint8 RGB image")
        source_h, source_w = value.shape[:2]
        scale = min(h / source_h, w / source_w)
        resized_h = min(h, max(1, round(source_h * scale)))
        resized_w = min(w, max(1, round(source_w * scale)))
        pixels = torch.from_numpy(value.copy()).permute(2, 0, 1)
        if (resized_h, resized_w) != (source_h, source_w):
            pixels = (
                F.interpolate(
                    pixels[None].float(),
                    size=(resized_h, resized_w),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                .round()
                .clamp(0, 255)
                .byte()
            )
        top += (canvas_height - y) // 2 + (h - resized_h) // 2
        left += (canvas_width - width) // 2 + (w - resized_w) // 2
        canvas[:, top : top + resized_h, left : left + resized_w] = pixels
    return canvas
