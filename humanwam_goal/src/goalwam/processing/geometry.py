"""Robot coordinate and gripper-unit conversions."""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch

from .layout import ActionLayout, resolve_action_layout

HEAD_QUAT_NORM_TOL = 0.001
ArrayLike = Union[np.ndarray, torch.Tensor]
STATE_ARM_EEF_COORDINATES = ("base", "head_camera")


def _is_torch(x: ArrayLike) -> bool:
    return isinstance(x, torch.Tensor)


def _cos(x):
    return torch.cos(x) if _is_torch(x) else np.cos(x)


def _sin(x):
    return torch.sin(x) if _is_torch(x) else np.sin(x)


def _stack_last(components, like: ArrayLike) -> ArrayLike:
    if _is_torch(like):
        return torch.stack(components, dim=-1)
    return np.stack(components, axis=-1)


def _all_last(x: ArrayLike) -> ArrayLike:
    return x.all(dim=-1) if _is_torch(x) else np.all(x, axis=-1)


def _sum_last(x: ArrayLike) -> ArrayLike:
    return x.sum(dim=-1) if _is_torch(x) else np.sum(x, axis=-1)


def quat_conj(q: ArrayLike) -> ArrayLike:
    return _stack_last([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], q)


def quat_mul(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    ax, ay, az, aw = (a[..., 0], a[..., 1], a[..., 2], a[..., 3])
    bx, by, bz, bw = (b[..., 0], b[..., 1], b[..., 2], b[..., 3])
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return _stack_last([x, y, z, w], a)


def quat_canonical(q: ArrayLike) -> ArrayLike:
    if _is_torch(q):
        sign = torch.where(
            q[..., 3:4] < 0, -torch.ones_like(q[..., 3:4]), torch.ones_like(q[..., 3:4])
        )
        return q * sign
    sign = np.where(q[..., 3:4] < 0.0, -1.0, 1.0).astype(q.dtype)
    return q * sign


def quat_rotate(q: ArrayLike, vector: ArrayLike) -> ArrayLike:
    if _is_torch(q) != _is_torch(vector):
        raise TypeError("q and vector must use the same array type.")
    zero = vector[..., 0] * 0
    pure = _stack_last([vector[..., 0], vector[..., 1], vector[..., 2], zero], vector)
    return quat_mul(quat_mul(q, pure), quat_conj(q))[..., :3]


def check_anchor_head_quaternion(
    state: ArrayLike,
    state_valid: ArrayLike | None = None,
    action_layout: ActionLayout | None = None,
    *,
    context: str = "",
) -> None:
    layout = resolve_action_layout(action_layout)
    if layout.head_eef_quat is None:
        return
    quaternion = state[..., layout.head_eef_quat]
    deviation = abs(_sum_last(quaternion * quaternion) ** 0.5 - 1.0)
    if state_valid is not None:
        deviation = deviation * _all_last(state_valid[..., layout.head_eef_quat])
    worst = float(deviation.max())
    if worst > HEAD_QUAT_NORM_TOL:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}head quaternion is not normalized (max error {worst:.3e}). Use xyzw ordering and (0, 0, 0, 1) for the identity orientation."
        )


def _copy(x: ArrayLike) -> ArrayLike:
    return x.clone() if _is_torch(x) else x.copy()


def resolve_state_arm_eef_coordinate(value: str) -> str:
    coordinate = str(value)
    if coordinate not in STATE_ARM_EEF_COORDINATES:
        raise ValueError(
            f"state_arm_eef_coordinate must be one of {STATE_ARM_EEF_COORDINATES}, got {value!r}."
        )
    return coordinate


def resolve_action_arm_eef_coordinate(value: str) -> str:
    if value not in STATE_ARM_EEF_COORDINATES:
        raise ValueError(f"action_arm_eef_coordinate must be base or head_camera, got {value!r}.")
    return value


def transform_state_arm_eef_coordinate(
    state: ArrayLike, coordinate: str, action_layout: ActionLayout | None = None
) -> ArrayLike:
    coordinate = resolve_state_arm_eef_coordinate(coordinate)
    out = _copy(state)
    if coordinate == "base":
        return out
    layout = resolve_action_layout(action_layout)
    if layout.head_eef_pos is None or layout.head_eef_quat is None:
        raise ValueError(
            "state_arm_eef_coordinate='head_camera' requires a head EEF xyz+quaternion pose."
        )
    head_pos = state[..., layout.head_eef_pos]
    q_world_from_head = state[..., layout.head_eef_quat]
    q_head_from_world = quat_conj(q_world_from_head)
    for pos, quat in (
        (layout.left_eef_pos, layout.left_eef_quat),
        (layout.right_eef_pos, layout.right_eef_quat),
    ):
        out[..., pos] = quat_rotate(q_head_from_world, state[..., pos] - head_pos)
        out[..., quat] = quat_canonical(quat_mul(q_head_from_world, state[..., quat]))
    return out


def propagate_state_arm_eef_validity(
    state_valid: ArrayLike, coordinate: str, action_layout: ActionLayout | None = None
) -> ArrayLike:
    coordinate = resolve_state_arm_eef_coordinate(coordinate)
    out = (
        _copy(state_valid).to(torch.bool)
        if _is_torch(state_valid)
        else state_valid.astype(bool, copy=True)
    )
    if coordinate == "base":
        return out
    layout = resolve_action_layout(action_layout)
    if layout.head_eef_pos is None or layout.head_eef_quat is None:
        raise ValueError(
            "state_arm_eef_coordinate='head_camera' requires a head EEF xyz+quaternion pose."
        )
    head_pose = slice(layout.head_eef_pos.start, layout.head_eef_quat.stop)
    head_pose_valid = _all_last(out[..., head_pose])
    head_quat_valid = _all_last(out[..., layout.head_eef_quat])
    for pos, quat in (
        (layout.left_eef_pos, layout.left_eef_quat),
        (layout.right_eef_pos, layout.right_eef_quat),
    ):
        pos_valid = _all_last(out[..., pos]) & head_pose_valid
        quat_valid = _all_last(out[..., quat]) & head_quat_valid
        out[..., pos] = pos_valid[..., None]
        out[..., quat] = quat_valid[..., None]
    return out


def to_relative_action(
    state: ArrayLike,
    action: ArrayLike,
    action_layout: ActionLayout | None = None,
    *,
    action_arm_eef_coordinate: str = "head_camera",
) -> ArrayLike:
    coordinate = resolve_action_arm_eef_coordinate(action_arm_eef_coordinate)
    layout = resolve_action_layout(action_layout)
    out = _copy(action)
    joint_delta_dims = [
        d for d in range(layout.joint.start, layout.joint.stop) if d not in layout.gripper_dims
    ]
    for d in joint_delta_dims:
        out[..., d] = action[..., d] - state[..., d]
    x_t = state[..., layout.chassis_x]
    y_t = state[..., layout.chassis_y]
    theta_t = state[..., layout.chassis_angle]
    cos_t = _cos(theta_t)
    sin_t = _sin(theta_t)
    dx_g = action[..., layout.chassis_x] - x_t
    dy_g = action[..., layout.chassis_y] - y_t
    out[..., layout.chassis_x] = cos_t * dx_g + sin_t * dy_g
    out[..., layout.chassis_y] = -sin_t * dx_g + cos_t * dy_g
    out[..., layout.chassis_angle] = action[..., layout.chassis_angle] - theta_t
    for pos in (layout.left_eef_pos, layout.right_eef_pos):
        delta_world = action[..., pos] - state[..., pos]
        if coordinate == "base" or layout.head_eef_quat is None:
            out[..., pos] = delta_world
        else:
            q_world_from_camera = state[..., layout.head_eef_quat]
            out[..., pos] = quat_rotate(quat_conj(q_world_from_camera), delta_world)
    if layout.head_eef_pos is not None:
        out[..., layout.head_eef_pos] = (
            action[..., layout.head_eef_pos] - state[..., layout.head_eef_pos]
        )
    quaternions = [layout.left_eef_quat, layout.right_eef_quat]
    if layout.head_eef_quat is not None:
        quaternions.append(layout.head_eef_quat)
    for quat in quaternions:
        q_t = state[..., quat]
        q_a = action[..., quat]
        q_rel = quat_mul(quat_conj(q_t), q_a)
        out[..., quat] = quat_canonical(q_rel)
    return out


def propagate_relative_action_validity(
    action_valid: ArrayLike,
    anchor_state_valid: ArrayLike,
    action_layout: ActionLayout | None = None,
    *,
    action_arm_eef_coordinate: str = "head_camera",
) -> ArrayLike:
    if _is_torch(action_valid) != _is_torch(anchor_state_valid):
        raise TypeError("action_valid and anchor_state_valid must use the same array type.")
    coordinate = resolve_action_arm_eef_coordinate(action_arm_eef_coordinate)
    layout = resolve_action_layout(action_layout)
    out = (
        _copy(action_valid).to(torch.bool)
        if _is_torch(action_valid)
        else action_valid.astype(bool, copy=True)
    )
    anchor = (
        anchor_state_valid.to(torch.bool)
        if _is_torch(anchor_state_valid)
        else anchor_state_valid.astype(bool)
    )
    if anchor.ndim == out.ndim - 1:
        anchor = anchor[..., None, :]
    absolute = set(layout.gripper_dims)
    for dim in range(layout.joint.start, layout.joint.stop):
        if dim not in absolute:
            out[..., dim] &= anchor[..., dim]
    groups = [layout.chassis_se2, layout.left_eef_quat, layout.right_eef_quat]
    if layout.head_eef_pos is not None:
        groups.extend([layout.head_eef_pos, layout.head_eef_quat])
    for group in groups:
        valid = _all_last(out[..., group]) & _all_last(anchor[..., group])
        out[..., group] = valid[..., None]
    for group in (layout.left_eef_pos, layout.right_eef_pos):
        valid = _all_last(out[..., group]) & _all_last(anchor[..., group])
        if coordinate == "head_camera" and layout.head_eef_pos is not None:
            head_pose = slice(layout.head_eef_pos.start, layout.head_eef_quat.stop)
            valid &= _all_last(anchor[..., head_pose])
        out[..., group] = valid[..., None]
    return out


def to_global_action(
    state: ArrayLike,
    rel_action: ArrayLike,
    action_layout: ActionLayout | None = None,
    *,
    action_arm_eef_coordinate: str = "head_camera",
) -> ArrayLike:
    coordinate = resolve_action_arm_eef_coordinate(action_arm_eef_coordinate)
    layout = resolve_action_layout(action_layout)
    out = _copy(rel_action)
    joint_delta_dims = [
        d for d in range(layout.joint.start, layout.joint.stop) if d not in layout.gripper_dims
    ]
    for d in joint_delta_dims:
        out[..., d] = rel_action[..., d] + state[..., d]
    x_t = state[..., layout.chassis_x]
    y_t = state[..., layout.chassis_y]
    theta_t = state[..., layout.chassis_angle]
    cos_t = _cos(theta_t)
    sin_t = _sin(theta_t)
    dx_l = rel_action[..., layout.chassis_x]
    dy_l = rel_action[..., layout.chassis_y]
    out[..., layout.chassis_x] = cos_t * dx_l - sin_t * dy_l + x_t
    out[..., layout.chassis_y] = sin_t * dx_l + cos_t * dy_l + y_t
    out[..., layout.chassis_angle] = rel_action[..., layout.chassis_angle] + theta_t
    for pos in (layout.left_eef_pos, layout.right_eef_pos):
        delta = rel_action[..., pos]
        if coordinate == "head_camera" and layout.head_eef_quat is not None:
            q_world_from_camera = state[..., layout.head_eef_quat]
            delta = quat_rotate(q_world_from_camera, delta)
        out[..., pos] = delta + state[..., pos]
    if layout.head_eef_pos is not None:
        out[..., layout.head_eef_pos] = (
            rel_action[..., layout.head_eef_pos] + state[..., layout.head_eef_pos]
        )
    quaternions = [layout.left_eef_quat, layout.right_eef_quat]
    if layout.head_eef_quat is not None:
        quaternions.append(layout.head_eef_quat)
    for quat in quaternions:
        q_t = state[..., quat]
        q_rel = rel_action[..., quat]
        q_a = quat_mul(q_t, q_rel)
        out[..., quat] = quat_canonical(q_a)
    return out


def zero_chassis_state(state: ArrayLike, action_layout: ActionLayout | None = None) -> ArrayLike:
    layout = resolve_action_layout(action_layout)
    out = _copy(state)
    out[..., layout.chassis_se2] = 0.0
    return out


GRIPPER_SPACES: tuple[str, ...] = ("closedness_0_100", "gap_m")
DEFAULT_GRIPPER_SPACE = "closedness_0_100"
DEFAULT_GRIPPER_GAP_MAX_M = 0.1


def resolve_gripper_space(value) -> str:
    if value is None:
        return DEFAULT_GRIPPER_SPACE
    if value not in GRIPPER_SPACES:
        raise ValueError(f"gripper_space must be one of {GRIPPER_SPACES}, got {value!r}.")
    return value


def resolve_gripper_gap_max_m(value) -> float:
    if value is None:
        return DEFAULT_GRIPPER_GAP_MAX_M
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (not math.isfinite(value))
        or (value <= 0)
    ):
        raise ValueError(
            f"gripper_gap_max_m must be a positive finite number of metres, got {value!r}."
        )
    return float(value)


def transform_gripper_space(
    values: ArrayLike,
    *,
    source_space: str,
    target_space: str,
    action_layout: ActionLayout | None = None,
    gap_max_m: float = DEFAULT_GRIPPER_GAP_MAX_M,
) -> ArrayLike:
    if not (source_space in GRIPPER_SPACES and target_space in GRIPPER_SPACES):
        raise ValueError(
            f"gripper_space conversion {source_space!r} -> {target_space!r} is unsupported; expected one of {GRIPPER_SPACES}"
        )
    if source_space == target_space:
        return values
    gap_max_m = resolve_gripper_gap_max_m(gap_max_m)
    if _is_torch(values):
        if not values.is_floating_point():
            raise ValueError(f"gripper conversion requires floating values, got {values.dtype}")
    elif not np.issubdtype(np.asarray(values).dtype, np.floating):
        raise ValueError(
            f"gripper conversion requires floating values, got {np.asarray(values).dtype}"
        )
    if source_space == "closedness_0_100":

        def convert(block):
            return (1.0 - block / 100.0) * gap_max_m
    else:

        def convert(block):
            return (1.0 - block / gap_max_m) * 100.0

    layout = resolve_action_layout(action_layout)
    dims = [int(dim) for dim in layout.gripper_dims]
    out = _copy(values)
    if _is_torch(out):
        out[..., dims] = convert(out[..., dims].to(torch.float64)).to(dtype=out.dtype)
    else:
        dtype = np.asarray(out).dtype
        out[..., dims] = convert(np.asarray(out[..., dims], dtype=np.float64)).astype(
            dtype, copy=False
        )
    return out


def clamp_gripper_closedness(
    values: ArrayLike, action_layout: ActionLayout | None = None
) -> ArrayLike:
    layout = resolve_action_layout(action_layout)
    dims = [int(dim) for dim in layout.gripper_dims]
    out = _copy(values)
    if _is_torch(out):
        out[..., dims] = out[..., dims].clamp(0.0, 100.0)
    else:
        out[..., dims] = np.clip(out[..., dims], 0.0, 100.0)
    return out
