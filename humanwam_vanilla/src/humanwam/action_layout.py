"""Explicit action components and validity for heterogeneous supervision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


EEF_CAMERA22_COMPONENTS = {
    "position": (0, 1, 2, 8, 9, 10),
    "rotation": (3, 4, 5, 6, 11, 12, 13, 14),
    "gripper": (7, 15),
    "camera_translation": (16, 17, 18),
    "camera_rotation": (19, 20, 21),
}


def resolve_action_components(
    action_dim: int,
    num_arms: int = 2,
    component_indices: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, tuple[int, ...]]:
    """Validate a complete, disjoint partition, or recover the legacy arm layout."""
    if component_indices is None:
        arm_dim = action_dim // num_arms
        return {
            "position": tuple(a * arm_dim + i for a in range(num_arms) for i in range(3)),
            "rotation": tuple(a * arm_dim + i for a in range(num_arms) for i in range(3, arm_dim - 1)),
            "gripper": tuple((a + 1) * arm_dim - 1 for a in range(num_arms)),
        }
    result = {}
    seen = set()
    for name, indices in component_indices.items():
        values = tuple(indices)
        for index in values:
            seen.add(index)
        result[name] = tuple(int(i) for i in values)
    return result


def action_valid_mask(
    action: torch.Tensor,
    action_dim_valid: torch.Tensor | None = None,
    action_is_pad: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return [B,T,D] validity; accept static [B,D] and per-step [B,T,D] masks."""
    batch, steps, dims = action.shape
    valid = torch.ones_like(action, dtype=torch.bool)
    if action_dim_valid is not None:
        dim_valid = torch.as_tensor(action_dim_valid, device=action.device, dtype=torch.bool)
        if tuple(dim_valid.shape) == (batch, dims):
            dim_valid = dim_valid[:, None, :]
        valid = valid & dim_valid
    if action_is_pad is not None:
        padding = torch.as_tensor(action_is_pad, device=action.device, dtype=torch.bool)
        valid = valid & ~padding[..., None]
    return valid


def masked_action_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean over valid dimensions, then valid time steps, independently per sample."""
    token_valid = valid.any(dim=2)
    token_mean = torch.where(valid, values, 0.0).sum(dim=2) / valid.sum(dim=2).clamp_min(1)
    return token_mean.sum(dim=1) / token_valid.sum(dim=1).clamp_min(1)
