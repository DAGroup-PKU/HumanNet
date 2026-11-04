# Modified by Open-HumanScale contributors: removed unused release runtime code.
"""Normalize per-episode pinhole intrinsics by warping RGB frames.

The action/state tensors deliberately do not pass through this module.  For a
source calibration ``K_s`` and virtual calibration ``K_v`` on the same camera
axes, the inverse sampling map is::

    p_s ~ K_s K_v^-1 p_v

Dense sidecars are loaded once at dataset construction.  The hot path performs
one table lookup and reuses a sampling grid for every repeated calibration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .eef_anchor_overlay import _cover_resize_center_crop_table, _load_dense_camera_intrinsics, camera_intrinsics_from_record


def _plain_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if hasattr(value, "items"):
        return {str(key): item for key, item in value.items()}


class CameraIntrinsicsWarp:
    """Warp a single-camera video to a fixed virtual pinhole calibration."""

    def __init__(
        self,
        *,
        dataset_dirs: Sequence[str],
        selected_episode_indices: Sequence[Iterable[int]],
        video_size: Sequence[int],
        source_intrinsics: Mapping[str, Any],
        target_intrinsics: Mapping[str, Any],
        interpolation: str = "bilinear",
        fill_value: float = -1.0,
        resize_mode: str = "cover_crop",
        source_routes: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.output_height, self.output_width = map(int, video_size)

        interpolation = str(interpolation).strip().lower()
        self.interpolation = interpolation
        self.fill_value = float(fill_value)

        target = camera_intrinsics_from_record(target_intrinsics)
        self.target_intrinsics = target
        self._target = target.as_tensor()

        roots = [Path(path).expanduser().resolve() for path in dataset_dirs]
        self.resize_mode = resize_mode
        enabled = []
        for path in dataset_dirs:
            if source_routes is None:
                enabled.append(True)
                continue
            matches = [route for route in source_routes
                       if str(route.get("match", "")) and str(route["match"]) in str(path)]
            enabled.append(matches[0]["warp"])
        self._enabled = tuple(enabled)
        spec = _plain_mapping(source_intrinsics, context="source_intrinsics")
        mode = str(spec.get("mode", "")).strip().lower()
        self._episode_intrinsics: tuple[torch.Tensor | None, ...]
        self._fixed_intrinsics: tuple[torch.Tensor | None, ...]
        if mode == "fixed":
            calibration = camera_intrinsics_from_record(spec)
            if resize_mode == "direct_resize":
                fixed = calibration.as_tensor() * torch.tensor([
                    self.output_width / calibration.width,
                    self.output_height / calibration.height,
                    self.output_width / calibration.width,
                    self.output_height / calibration.height,
                ])
            else:
                fixed = calibration.cover_resize_center_crop(
                    self.output_height, self.output_width
                ).as_tensor()
            self._fixed_intrinsics = tuple(fixed for _ in roots)
            self._episode_intrinsics = tuple(None for _ in roots)
        elif mode == "sidecar":
            relative_path = str(spec.get("relative_path", "meta/camera_intrinsics.npy"))
            episode_tables: list[torch.Tensor | None] = []
            for root, selected, warp_enabled in zip(roots, selected_episode_indices, self._enabled, strict=True):
                if not warp_enabled:
                    episode_tables.append(None)
                    continue
                sidecar = root / relative_path
                raw_table = _load_dense_camera_intrinsics(sidecar)
                if resize_mode == "direct_resize":
                    table = raw_table[:, :4].clone()
                    table[:, [0, 2]] *= self.output_width / raw_table[:, 4, None]
                    table[:, [1, 3]] *= self.output_height / raw_table[:, 5, None]
                    valid = torch.isfinite(raw_table).all(dim=1)
                    valid &= (raw_table[:, [0, 1, 4, 5]] > 0).all(dim=1)
                    table[~valid] = torch.nan
                else:
                    table = _cover_resize_center_crop_table(
                        raw_table, self.output_height, self.output_width
                    )
                episode_tables.append(table)
            self._fixed_intrinsics = tuple(None for _ in roots)
            self._episode_intrinsics = tuple(episode_tables)

        # Per-worker cache.  Current MECKA has many episodes but only a small set
        # of repeated calibrations, so caching by exact float32 K avoids allocating
        # a full HxW grid per episode.
        self._grid_cache: dict[tuple[float, float, float, float], torch.Tensor] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_grid_cache"] = {}
        return state

    def intrinsics_for(self, dataset_index: int, episode_index: int) -> torch.Tensor:
        dataset_index = int(dataset_index)
        fixed = self._fixed_intrinsics[dataset_index]
        if fixed is not None:
            return fixed
        table = self._episode_intrinsics[dataset_index]
        episode_index = int(episode_index)
        result = table[episode_index]
        return result

    def _grid(self, source: torch.Tensor) -> torch.Tensor:
        source = source.to(dtype=torch.float32, device="cpu").reshape(4)
        key = tuple(float(value) for value in source)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached

        fx_s, fy_s, cx_s, cy_s = source
        fx_v, fy_v, cx_v, cy_v = self._target
        output_x = torch.arange(self.output_width, dtype=torch.float32)
        output_y = torch.arange(self.output_height, dtype=torch.float32)
        source_x = (fx_s / fx_v) * (output_x - cx_v) + cx_s
        source_y = (fy_s / fy_v) * (output_y - cy_v) + cy_s
        grid_y, grid_x = torch.meshgrid(source_y, source_x, indexing="ij")
        # align_corners=False maps pixel center x to 2*(x+0.5)/W - 1.
        grid = torch.stack(
            [
                2.0 * (grid_x + 0.5) / self.output_width - 1.0,
                2.0 * (grid_y + 0.5) / self.output_height - 1.0,
            ],
            dim=-1,
        ).unsqueeze(0).contiguous()
        self._grid_cache[key] = grid
        return grid

    def apply(
        self,
        video: torch.Tensor,
        *,
        dataset_index: int,
        episode_index: int,
    ) -> torch.Tensor:
        """Return ``video`` warped to ``target_intrinsics``; action is untouched."""
        if not self._enabled[int(dataset_index)]:
            return video
        source = self.intrinsics_for(dataset_index, episode_index)
        grid = self._grid(source).to(device=video.device)
        frames = video.permute(1, 0, 2, 3).contiguous()
        output_dtype = frames.dtype
        if frames.dtype not in {torch.float32, torch.float64}:
            frames = frames.float()
        grid = grid.to(dtype=frames.dtype).expand(frames.shape[0], -1, -1, -1)
        # Shift by the desired fill so grid_sample's zero padding becomes the
        # correct value (-1 is black for the current normalized video contract).
        warped = F.grid_sample(
            frames - self.fill_value,
            grid,
            mode=self.interpolation,
            padding_mode="zeros",
            align_corners=False,
        ).add_(self.fill_value)
        if warped.dtype != output_dtype:
            warped = warped.to(output_dtype)
        return warped.permute(1, 0, 2, 3).contiguous()
