# Portions Copyright (c) 2026 phyai team. MIT License; see licenses/phyai-MIT.txt.
"""Temporal and spatial rotary positions."""

import math

import torch


def compute_mrope_position_ids_vision(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    base_temporal_compression_factor: int | None = None,
    enable_fps_modulation: bool = True,
    start_frame_offset: int = 0,
) -> tuple[torch.Tensor, int | float]:
    fps_modulation = enable_fps_modulation and fps is not None
    if fps_modulation:
        tps = fps / temporal_compression_factor
        effective_base_tcf = (
            base_temporal_compression_factor
            if base_temporal_compression_factor is not None
            else temporal_compression_factor
        )
        base_tps = base_fps / effective_base_tcf
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        t_index = (
            ((frame_indices + start_frame_offset) / tps * base_tps + temporal_offset)
            .view(-1, 1)
            .expand(-1, grid_h * grid_w)
            .flatten()
        )
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
            + int(temporal_offset)
            + start_frame_offset
        )
    h_index = (
        torch.arange(grid_h, dtype=torch.long).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    )
    w_index = (
        torch.arange(grid_w, dtype=torch.long).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()
    )
    if fps_modulation:
        mrope_ids = torch.stack(
            [t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0
        )
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)
    next_offset = math.floor(mrope_ids.max().item()) + 1
    return (mrope_ids, next_offset)


def goalwam_position_ids(
    *,
    text_tokens: int,
    grids: tuple[tuple[int, int, int], ...],
    history_ages_seconds: tuple[float, ...],
    action_rows: int,
    video_fps: float,
    action_fps: float,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    temporal_modality_margin: int = 15000,
) -> torch.Tensor:
    if len(grids) != len(history_ages_seconds) + 2:
        raise ValueError("Provide goal, histories, then the rollout grid")
    if text_tokens < 1 or action_rows < 1 or temporal_compression_factor < 1:
        raise ValueError("Text length, action rows, and temporal compression must be positive")
    if any((not math.isfinite(value) or value <= 0 for value in (video_fps, action_fps, base_fps))):
        raise ValueError("Frame rates must be finite and positive")
    if any((len(grid) != 3 or min(grid) < 1 for grid in grids)):
        raise ValueError("Each patch grid must contain positive T,H,W dimensions")
    if any((grid[0] != 1 for grid in grids[:-1])):
        raise ValueError("Goal and histories must be independent single-frame grids")
    if any((not math.isfinite(age) or age <= 0 for age in history_ages_seconds)):
        raise ValueError("History ages must be finite and positive")
    offset = text_tokens + temporal_modality_margin
    common = {
        "fps": video_fps,
        "base_fps": base_fps,
        "temporal_compression_factor": temporal_compression_factor,
        "enable_fps_modulation": True,
    }
    goal, origin = compute_mrope_position_ids_vision(*grids[0], temporal_offset=offset, **common)
    positions = [goal]
    for grid, age in zip(grids[1:-1], history_ages_seconds, strict=True):
        history, _ = compute_mrope_position_ids_vision(
            *grid,
            temporal_offset=origin,
            start_frame_offset=-age * video_fps / temporal_compression_factor,
            **common,
        )
        positions.append(history)
    rollout, _ = compute_mrope_position_ids_vision(*grids[-1], temporal_offset=origin, **common)
    actions, _ = compute_mrope_position_ids_vision(
        action_rows,
        1,
        1,
        temporal_offset=origin,
        fps=action_fps,
        base_fps=base_fps,
        temporal_compression_factor=1,
        base_temporal_compression_factor=temporal_compression_factor,
        enable_fps_modulation=True,
        start_frame_offset=0,
    )
    return torch.cat((*positions, rollout, actions), dim=1).unsqueeze(1)
