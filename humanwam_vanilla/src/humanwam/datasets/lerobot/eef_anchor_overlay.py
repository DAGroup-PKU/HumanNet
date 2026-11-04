"""First-frame EEF overlays for ego/robot visual alignment.

The training hot path deliberately stays small:

* camera intrinsics are loaded and validated once when the dataset is built;
* a sample lookup is an in-memory ``(dataset_index, episode_index)`` lookup;
* only the first model-input frame is modified, in place, with small Torch index
  writes (no PIL/OpenCV conversion and no full-frame mask allocation).

EEF poses are expected in the shared OpenGL head-camera frame used by the current
RoboTwin/Ego transform chains: +x right, +y up, +z backward.  Camera intrinsics use
the usual image/OpenCV convention, so a point ``(x, y, z)`` projects with depth
``-z`` and image y ``-fy*y/depth + cy``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .utils.rotation import quaternion_to_matrix, rotation_6d_to_matrix


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole calibration tied to a particular image raster."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


    def cover_resize_center_crop(self, output_height: int, output_width: int) -> "CameraIntrinsics":
        """Apply the loader's exact cover-resize + center-crop geometry to K."""

        scale = max(output_width / self.width, output_height / self.height)
        resized_height = int(scale * self.height + 0.5)
        resized_width = int(scale * self.width + 0.5)
        # torchvision.transforms.functional.center_crop uses round(), including
        # Python's ties-to-even behavior, for an odd crop difference.
        crop_top = int(round((resized_height - output_height) / 2.0))
        crop_left = int(round((resized_width - output_width) / 2.0))
        scale_x = resized_width / self.width
        scale_y = resized_height / self.height
        return CameraIntrinsics(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x - crop_left,
            cy=self.cy * scale_y - crop_top,
            width=int(output_width),
            height=int(output_height),
        )

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [self.fx, self.fy, self.cx, self.cy], dtype=torch.float32
        )


def _plain_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if hasattr(value, "items"):
        return {str(key): item for key, item in value.items()}


def camera_intrinsics_from_record(record: Mapping[str, Any]) -> CameraIntrinsics:
    """Parse the compact sidecar schema plus common K/dict spellings."""

    record = _plain_mapping(record, context="camera intrinsic record")
    nested = record.get("intrinsics")
    if isinstance(nested, Mapping) or hasattr(nested, "items"):
        values = {**record, **_plain_mapping(nested, context="record.intrinsics")}
    else:
        values = record

    matrix = values.get("K", values.get("camera_matrix"))
    if matrix is not None:
        matrix = torch.as_tensor(matrix, dtype=torch.float64)
        fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
        cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    else:
        aliases = {
            "fx": ("fx", "fl_x"),
            "fy": ("fy", "fl_y"),
            "cx": ("cx",),
            "cy": ("cy",),
        }
        parsed: dict[str, float] = {}
        for output_key, candidates in aliases.items():
            raw = next((values[key] for key in candidates if key in values), None)
            parsed[output_key] = float(raw)
        fx, fy, cx, cy = parsed["fx"], parsed["fy"], parsed["cx"], parsed["cy"]

    width = values.get("width", values.get("w", values.get("image_width")))
    height = values.get("height", values.get("h", values.get("image_height")))
    return CameraIntrinsics(fx, fy, cx, cy, int(width), int(height))


def load_camera_intrinsics_sidecar(path: str | Path) -> dict[int, CameraIntrinsics]:
    """Load ``meta/camera_intrinsics.jsonl`` into a compact episode table."""

    path = Path(path)
    result: dict[int, CameraIntrinsics] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            episode_index = int(record["episode_index"])
            intrinsic = camera_intrinsics_from_record(record)
            result[episode_index] = intrinsic
    return result


def _load_dense_camera_intrinsics(path: Path) -> torch.Tensor:
    """Load a dense ``[episode, fx,fy,cx,cy,width,height]`` calibration table.

    ``.npy`` is the production format: it avoids reparsing millions of JSON records
    independently on every distributed rank. JSONL remains supported for inspection,
    tests, and small datasets.
    """
    if path.suffix == ".npy":
        table = torch.from_numpy(np.load(path, allow_pickle=False)).to(torch.float32)
        return table

    sparse = load_camera_intrinsics_sidecar(path)
    table = torch.full((max(sparse) + 1, 6), torch.nan, dtype=torch.float32)
    for episode_index, calibration in sparse.items():
        table[episode_index] = torch.tensor(
            [
                calibration.fx,
                calibration.fy,
                calibration.cx,
                calibration.cy,
                calibration.width,
                calibration.height,
            ],
            dtype=torch.float32,
        )
    return table


def _cover_resize_center_crop_table(
    table: torch.Tensor, output_height: int, output_width: int
) -> torch.Tensor:
    """Vectorized calibration transform; returns compact ``[N,4]`` float32."""
    fx, fy, cx, cy, width, height = table.unbind(dim=1)
    valid = torch.isfinite(table).all(dim=1) & (fx > 0) & (fy > 0) & (width > 0) & (height > 0)
    safe_width = torch.where(valid, width, torch.ones_like(width))
    safe_height = torch.where(valid, height, torch.ones_like(height))
    scale = torch.maximum(output_width / safe_width, output_height / safe_height)
    resized_width = torch.floor(scale * safe_width + 0.5)
    resized_height = torch.floor(scale * safe_height + 0.5)
    crop_left = torch.round((resized_width - output_width) / 2.0)
    crop_top = torch.round((resized_height - output_height) / 2.0)
    result = torch.stack(
        [
            fx * resized_width / safe_width,
            fy * resized_height / safe_height,
            cx * resized_width / safe_width - crop_left,
            cy * resized_height / safe_height - crop_top,
        ],
        dim=1,
    ).to(torch.float32)
    result[~valid] = torch.nan
    return result.contiguous()


class EefAnchorOverlay:
    """Draw two canonical EEF coordinate triads on the first input frame only."""

    def __init__(
        self,
        *,
        dataset_dirs: Sequence[str],
        selected_episode_indices: Sequence[Iterable[int]],
        video_size: Sequence[int],
        intrinsics: Mapping[str, Any],
        state_key: str = "default",
        axis_length_m: float = 0.06,
        line_width_px: int = 2,
        origin_radius_px: int = 3,
        origin_outline_width_px: int = 0,
        alpha: float = 0.9,
        min_depth_m: float = 0.02,
    ) -> None:
        self.output_height, self.output_width = map(int, video_size)
        self.state_key = str(state_key)
        self.axis_length_m = float(axis_length_m)
        self.line_width_px = int(line_width_px)
        self.origin_radius_px = int(origin_radius_px)
        self.origin_outline_width_px = int(origin_outline_width_px)
        self.alpha = float(alpha)
        self.min_depth_m = float(min_depth_m)

        roots = [Path(path).expanduser().resolve() for path in dataset_dirs]
        spec = _plain_mapping(intrinsics, context="eef_anchor_overlay.intrinsics")
        mode = str(spec.get("mode", "")).strip().lower()
        self._episode_intrinsics: tuple[torch.Tensor | None, ...]
        self._fixed_intrinsics: tuple[torch.Tensor | None, ...]
        if mode == "fixed":
            fixed = camera_intrinsics_from_record(spec).cover_resize_center_crop(
                self.output_height, self.output_width
            ).as_tensor()
            self._fixed_intrinsics = tuple(fixed for _ in roots)
            self._episode_intrinsics = tuple(None for _ in roots)
        elif mode == "sidecar":
            relative_path = str(spec.get("relative_path", "meta/camera_intrinsics.npy"))
            episode_tables: list[torch.Tensor] = []
            for root, selected in zip(roots, selected_episode_indices, strict=True):
                raw_table = _load_dense_camera_intrinsics(root / relative_path)
                table = _cover_resize_center_crop_table(
                    raw_table, self.output_height, self.output_width
                )
                episode_tables.append(table)
            self._fixed_intrinsics = tuple(None for _ in roots)
            self._episode_intrinsics = tuple(episode_tables)

        # RGB local XYZ and text-free hand origin colors. There is intentionally no
        # font/text path here: neither "R" nor "L" can be rendered by this overlay.
        self._axis_colors = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.35, 1.0]],
            dtype=torch.float32,
        )
        self._origin_colors = torch.tensor(
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]], dtype=torch.float32
        )
        self._origin_outline_color = torch.ones(3, dtype=torch.float32)
        # ``line_width_px`` denotes diameter. Rasterized disks are necessarily odd,
        # so even requests round up by one pixel (2 -> 3, 5 -> 5).
        self._line_offsets = self._disk_offsets(self.line_width_px // 2)
        self._origin_offsets = self._disk_offsets(self.origin_radius_px)
        self._origin_outline_offsets = self._disk_offsets(
            self.origin_radius_px + self.origin_outline_width_px
        )

    @staticmethod
    def _disk_offsets(radius: int) -> torch.Tensor:
        offsets = [
            (dy, dx)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if dx * dx + dy * dy <= radius * radius
        ]
        return torch.tensor(offsets, dtype=torch.long)

    def intrinsics_for(self, dataset_index: int, episode_index: int) -> torch.Tensor:
        dataset_index = int(dataset_index)
        fixed = self._fixed_intrinsics[dataset_index]
        if fixed is not None:
            return fixed
        table = self._episode_intrinsics[dataset_index]
        episode_index = int(episode_index)
        result = table[episode_index]
        return result

    def _project(self, points: torch.Tensor, intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        points = points.to(dtype=torch.float32, device="cpu")
        intrinsics = intrinsics.to(dtype=torch.float32, device="cpu")
        depth = -points[:, 2]
        valid = torch.isfinite(points).all(dim=1) & (depth > self.min_depth_m)
        safe_depth = torch.where(valid, depth, torch.ones_like(depth))
        fx, fy, cx, cy = intrinsics
        pixels = torch.stack(
            [
                fx * points[:, 0] / safe_depth + cx,
                -fy * points[:, 1] / safe_depth + cy,
            ],
            dim=1,
        )
        valid &= torch.isfinite(pixels).all(dim=1)
        return pixels, valid

    def _blend_pixels(
        self,
        frame: torch.Tensor,
        ys: torch.Tensor,
        xs: torch.Tensor,
        color_01: torch.Tensor,
    ) -> None:
        if ys.numel() == 0:
            return
        keep = (
            (ys >= 0)
            & (ys < frame.shape[-2])
            & (xs >= 0)
            & (xs < frame.shape[-1])
        )
        if not bool(keep.any()):
            return
        ys, xs = ys[keep], xs[keep]
        color = color_01.to(dtype=frame.dtype, device=frame.device).mul(2.0).sub(1.0)
        if self.alpha == 1.0:
            frame[:, ys, xs] = color[:, None]
        else:
            frame[:, ys, xs] = (
                frame[:, ys, xs] * (1.0 - self.alpha)
                + color[:, None] * self.alpha
            )

    def _draw_segment(
        self,
        frame: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        color: torch.Tensor,
    ) -> None:
        delta = end - start
        steps = max(2, int(torch.ceil(delta.abs().max()).item()) + 1)
        # Bound pathological near-plane projections while preserving every visible
        # pixel of ordinary wrist axes.
        steps = min(steps, 4 * max(frame.shape[-2:]))
        alpha = torch.linspace(0.0, 1.0, steps=steps)
        points = torch.round(start[None] + alpha[:, None] * delta[None]).long()
        expanded = points[:, None, :] + self._line_offsets[None, :, [1, 0]]
        expanded = expanded.reshape(-1, 2)
        self._blend_pixels(frame, expanded[:, 1], expanded[:, 0], color)

    def _draw_origin(self, frame: torch.Tensor, center: torch.Tensor, color: torch.Tensor) -> None:
        center = torch.round(center).long()
        if self.origin_outline_width_px:
            outline_ys = center[1] + self._origin_outline_offsets[:, 0]
            outline_xs = center[0] + self._origin_outline_offsets[:, 1]
            self._blend_pixels(
                frame,
                outline_ys,
                outline_xs,
                self._origin_outline_color,
            )
        ys = center[1] + self._origin_offsets[:, 0]
        xs = center[0] + self._origin_offsets[:, 1]
        self._blend_pixels(frame, ys, xs, color)

    def apply(
        self,
        video: torch.Tensor,
        anchor_state: torch.Tensor,
        *,
        dataset_index: int,
        episode_index: int,
        operating_hand: str | None = None,
    ) -> torch.Tensor:
        """Overlay canonical EEF triads onto ``video[..., time=0, :, :]`` in place."""
        if video.ndim == 4:
            frame = video[:, 0]
        elif video.ndim == 5:
            # Separate-view layouts keep head camera first. The bridge experiment is
            # configured for single-head view, but this makes the primitive safe to reuse.
            frame = video[0, :, 0]

        state = torch.as_tensor(anchor_state, dtype=torch.float32, device="cpu").reshape(-1)
        if state.numel() == 16:
            block_width = 8
            rotation_slice = slice(3, 7)
            rotation_to_matrix = quaternion_to_matrix
        elif state.numel() == 20:
            block_width = 10
            rotation_slice = slice(3, 9)
            rotation_to_matrix = rotation_6d_to_matrix
        intrinsics = self.intrinsics_for(dataset_index, episode_index)
        hand_indices = {"left": (0,), "right": (1,), "both": (0, 1), None: (0, 1)}

        visibility_margin = self.origin_radius_px + self.origin_outline_width_px
        for hand_index in hand_indices[operating_hand]:
            block = state[
                hand_index * block_width : (hand_index + 1) * block_width
            ]
            origin = block[:3]
            rotation = rotation_to_matrix(block[rotation_slice])
            endpoints = origin[None] + rotation.transpose(0, 1) * self.axis_length_m
            points = torch.cat([origin[None], endpoints], dim=0)
            pixels, valid = self._project(points, intrinsics)
            # An origin outside the image is not a useful anchor and can create long
            # border streaks, so suppress that hand entirely.
            origin_visible = bool(
                valid[0]
                and -visibility_margin <= pixels[0, 0] < self.output_width + visibility_margin
                and -visibility_margin <= pixels[0, 1] < self.output_height + visibility_margin
            )
            if not origin_visible:
                continue
            for axis in range(3):
                if bool(valid[axis + 1]):
                    self._draw_segment(
                        frame,
                        pixels[0],
                        pixels[axis + 1],
                        self._axis_colors[axis],
                    )
            self._draw_origin(frame, pixels[0], self._origin_colors[hand_index])
        return video
