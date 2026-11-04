"""Camera-pose geometry shared by the ego dataset and Wan VideoDiT.

The canonical convention is column-vector, left-multiplication geometry:
``P_i = T_{C0<-Ci}``.  Stored ego extrinsics are camera-to-world poses in
``xyz + quaternion_wxyz`` order, so ``P_i = inv(T_W<-C0) @ T_W<-Ci``.
"""

from __future__ import annotations

import math

import torch

CAMERA_POSE_BLOCK_SIZE = 4
CAMERA_POSE_DELTA_DIM = 6
CAMERA_POSE_BLOCK_DIM = CAMERA_POSE_BLOCK_SIZE * CAMERA_POSE_DELTA_DIM
CAMERA_POSE_REPRESENTATIONS = frozenset({"euler6", "se3_log6"})


def _identity_4x4(
    leading_shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.eye(4, dtype=dtype, device=device).expand(
        leading_shape + (4, 4)
    )


def quaternion_wxyz_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert ``[..., 4]`` real-first quaternions to rotation matrices."""
    norm_sq = quaternion.square().sum(dim=-1, keepdim=True)
    w, x, y, z = (quaternion / norm_sq.sqrt()).unbind(dim=-1)
    two = 2.0
    matrix = torch.stack(
        (
            1.0 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1.0 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1.0 - two * (x * x + y * y),
        ),
        dim=-1,
    )
    return matrix.reshape(quaternion.shape[:-1] + (3, 3))


def camera_pose7_to_matrix(camera_pose: torch.Tensor) -> torch.Tensor:
    """Convert camera-to-world ``[..., 7]`` xyz+quaternion_wxyz poses."""

    camera_pose = torch.as_tensor(camera_pose)
    if not camera_pose.is_floating_point():
        camera_pose = camera_pose.float()
    matrix = torch.zeros(
        camera_pose.shape[:-1] + (4, 4),
        dtype=camera_pose.dtype,
        device=camera_pose.device,
    )
    matrix[..., :3, :3] = quaternion_wxyz_to_matrix(camera_pose[..., 3:7])
    matrix[..., :3, 3] = camera_pose[..., :3]
    matrix[..., 3, 3] = 1.0
    return matrix


def _prepare_pose_valid(
    pose_valid: torch.Tensor | None,
    pose_shape: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    if pose_valid is None:
        return torch.ones(pose_shape, dtype=torch.bool, device=device)
    valid = torch.as_tensor(pose_valid, dtype=torch.bool, device=device)
    return valid


def _replicate_invalid_tail(
    pose: torch.Tensor, pose_valid: torch.Tensor
) -> torch.Tensor:
    result = pose.clone()
    for index in range(1, pose.shape[-3]):
        result[..., index, :, :] = torch.where(
            pose_valid[..., index, None, None],
            result[..., index, :, :],
            result[..., index - 1, :, :],
        )
    return result


def camera_to_world_pose7_to_delta(
    camera_pose: torch.Tensor,
    pose_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert absolute ``T_W<-Ci`` pose7 values to canonical ``T_C0<-Ci``.

    Invalid tail poses are replicated from the last valid pose before conversion.
    The first output is then written as an exact identity matrix.
    """

    camera_pose = torch.as_tensor(camera_pose)
    valid = _prepare_pose_valid(
        pose_valid,
        tuple(camera_pose.shape[:-1]),
        device=camera_pose.device,
    )
    absolute = camera_pose7_to_matrix(camera_pose)
    absolute = _replicate_invalid_tail(absolute, valid)
    anchor_inverse = torch.linalg.inv(absolute[..., 0, :, :])
    delta = anchor_inverse.unsqueeze(-3) @ absolute
    delta[..., 0, :, :] = _identity_4x4(
        tuple(delta.shape[:-3]), dtype=delta.dtype, device=delta.device
    )
    return delta


def rotation_angle(rotation: torch.Tensor) -> torch.Tensor:
    """Return the SO(3) geodesic angle in radians for ``[..., 3, 3]``."""
    cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(
        -1.0, 1.0
    )
    return torch.acos(cosine)


def snap_camera_pose_delta_to_identity(
    camera_pose_delta: torch.Tensor,
    pose_valid: torch.Tensor | None = None,
    *,
    translation_tolerance_m: float,
    rotation_tolerance_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a cumulative-pose deadband around identity.

    A pose is treated as stationary only when *both* its translation norm and
    rotation geodesic angle are within tolerance.  This suppresses real-camera
    extrinsic jitter without erasing a small translation paired with real rotation
    (or vice versa).  Returns the snapped trajectory and its identity mask.
    """

    pose = torch.as_tensor(camera_pose_delta)
    if not pose.is_floating_point():
        pose = pose.float()
    valid = _prepare_pose_valid(
        pose_valid,
        tuple(pose.shape[:-2]),
        device=pose.device,
    )
    pose = _replicate_invalid_tail(pose, valid)
    identity = _identity_4x4(
        tuple(pose.shape[:-3]), dtype=pose.dtype, device=pose.device
    )

    translation_norm = torch.linalg.vector_norm(pose[..., :3, 3], dim=-1)
    angle = rotation_angle(pose[..., :3, :3])
    stationary = (
        (translation_norm <= translation_tolerance_m)
        & (angle <= math.radians(rotation_tolerance_deg))
        & valid
    )
    stationary[..., 0] = True
    snapped = torch.where(
        stationary[..., None, None],
        identity.unsqueeze(-3),
        pose,
    )
    snapped = _replicate_invalid_tail(snapped, valid)
    return snapped, stationary


def rotation_matrix_to_xyz_euler(rotation: torch.Tensor) -> torch.Tensor:
    """Return roll, pitch, yaw for ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

    This is the conventional fixed-axis XYZ / roll-pitch-yaw decomposition.  The
    returned angles are in radians and lie on the principal branch.  Camera-pose
    conditioning applies it only to consecutive, small rotations; the singular
    fallback is nevertheless explicit so the data contract is deterministic.
    """
    horizontal = torch.sqrt(
        rotation[..., 0, 0].square() + rotation[..., 1, 0].square()
    )
    singular = horizontal < 1.0e-6
    roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    pitch = torch.atan2(-rotation[..., 2, 0], horizontal)
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    singular_roll = torch.atan2(-rotation[..., 1, 2], rotation[..., 1, 1])
    roll = torch.where(singular, singular_roll, roll)
    yaw = torch.where(singular, torch.zeros_like(yaw), yaw)
    return torch.stack((roll, pitch, yaw), dim=-1)


def so3_log_map(rotation: torch.Tensor) -> torch.Tensor:
    """Stable SO(3) logarithm with xyz rotation-vector output."""

    angle = rotation_angle(rotation)
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sin_angle = torch.sin(angle)
    scale = torch.where(
        angle < 1.0e-4,
        0.5 + angle.square() / 12.0,
        angle / (2.0 * sin_angle.clamp_min(1.0e-7)),
    )
    return vee * scale.unsqueeze(-1)


def se3_log_map(transform: torch.Tensor) -> torch.Tensor:
    """Return ``[..., 6]`` SE(3) twists ordered as translation then rotation."""
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    omega = so3_log_map(rotation)
    theta_sq = omega.square().sum(dim=-1)
    omega_hat = torch.zeros_like(rotation)
    omega_hat[..., 0, 1] = -omega[..., 2]
    omega_hat[..., 0, 2] = omega[..., 1]
    omega_hat[..., 1, 0] = omega[..., 2]
    omega_hat[..., 1, 2] = -omega[..., 0]
    omega_hat[..., 2, 0] = -omega[..., 1]
    omega_hat[..., 2, 1] = omega[..., 0]
    theta = theta_sq.sqrt()
    coefficient = torch.where(
        theta < 1.0e-4,
        1.0 / 12.0 + theta_sq / 720.0,
        1.0 / theta_sq.clamp_min(1.0e-12)
        - (1.0 + torch.cos(theta))
        / (2.0 * theta * torch.sin(theta).clamp_min(1.0e-7)),
    )
    identity3 = torch.eye(
        3, dtype=transform.dtype, device=transform.device
    ).expand(rotation.shape)
    jacobian_inverse = (
        identity3
        - 0.5 * omega_hat
        + coefficient[..., None, None] * (omega_hat @ omega_hat)
    )
    rho = (jacobian_inverse @ translation.unsqueeze(-1)).squeeze(-1)
    return torch.cat((rho, omega), dim=-1)


def build_camera_pose_blocks(
    camera_pose_delta: torch.Tensor,
    pose_valid: torch.Tensor | None = None,
    *,
    num_latent_frames: int,
    translation_tolerance_m: float,
    rotation_tolerance_deg: float,
    representation: str = "euler6",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build four ordered consecutive 6D deltas per VAE latent.

    ``euler6`` is the baseline: each delta is translation plus XYZ roll/pitch/yaw.
    ``se3_log6`` is the representation ablation.  Both produce the same 24D block,
    use identical temporal alignment, and map identity motion to exact zeros.
    """

    pose = torch.as_tensor(camera_pose_delta)
    representation = str(representation).strip().lower()
    valid = _prepare_pose_valid(
        pose_valid,
        tuple(pose.shape[:-2]),
        device=pose.device,
    )
    snapped, stationary = snap_camera_pose_delta_to_identity(
        pose,
        valid,
        translation_tolerance_m=translation_tolerance_m,
        rotation_tolerance_deg=rotation_tolerance_deg,
    )

    if num_latent_frames == 1:
        return pose.new_zeros((pose.shape[0], 1, CAMERA_POSE_BLOCK_DIM)), stationary

    local_transform = torch.linalg.inv(snapped[:, :-1]) @ snapped[:, 1:]
    if representation == "euler6":
        local_delta = torch.cat(
            (
                local_transform[..., :3, 3],
                rotation_matrix_to_xyz_euler(local_transform[..., :3, :3]),
            ),
            dim=-1,
        )
    else:
        local_delta = se3_log_map(local_transform)
    increment_valid = valid[:, :-1] & valid[:, 1:]
    local_delta = torch.where(
        increment_valid[..., None], local_delta, torch.zeros_like(local_delta)
    )
    future_blocks = local_delta.reshape(
        pose.shape[0], num_latent_frames - 1, CAMERA_POSE_BLOCK_DIM
    )
    first_block = local_delta.new_zeros(
        (pose.shape[0], 1, CAMERA_POSE_BLOCK_DIM)
    )
    blocks = torch.cat((first_block, future_blocks), dim=1)
    return blocks, stationary


def camera_pose_fixed_normalization(
    *,
    local_translation_scale_m: float,
    local_rotation_scale_rad: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return zero-centered fixed scaling for a 4x6D delta block."""
    mean = torch.zeros(CAMERA_POSE_BLOCK_DIM, dtype=torch.float32)
    std = torch.ones(CAMERA_POSE_BLOCK_DIM, dtype=torch.float32)
    for offset in range(0, CAMERA_POSE_BLOCK_DIM, CAMERA_POSE_DELTA_DIM):
        std[offset : offset + 3] = local_translation_scale_m
        std[offset + 3 : offset + 6] = local_rotation_scale_rad
    return mean, std
