# Modified by Open-HumanScale contributors: removed internal development references.
"""Frame-change transforms for dual-arm EEF pose action/state.

Currently: express RoboTwin dual-arm EEF (stored in the sim WORLD frame) in the fixed
head-camera frame, matching the unified 16D EEF action space.
Applied BEFORE DualArmPoseRotationTransform.
"""
from typing import Dict, List

import torch

from ..utils.rotation import quaternion_to_matrix, matrix_to_quaternion


class DualArmRelativePoseTransform:
    """Camera-frame action deltas relative to the first observed EEF pose.

    Translation is p_target - p_anchor; rotation is R_target @ R_anchor.T.
    State poses and absolute gripper targets are preserved.
    """

    def __init__(self, category_keys, num_arms=2):
        self.category_keys = category_keys
        self.num_arms = num_arms

    def _run(self, batch, inverse):
        for key in self.category_keys['action']:
            actions = batch['action'][key]
            anchors = batch['state'][key][..., :1, :]
            block = actions.shape[-1] // self.num_arms
            output = []
            for pose, anchor in zip(actions.split(block, dim=-1), anchors.split(block, dim=-1)):
                rotation = quaternion_to_matrix(pose[..., 3:7])
                anchor_rotation = quaternion_to_matrix(anchor[..., 3:7])
                if inverse:
                    position = pose[..., :3] + anchor[..., :3]
                    rotation = rotation @ anchor_rotation
                else:
                    position = pose[..., :3] - anchor[..., :3]
                    rotation = rotation @ anchor_rotation.transpose(-1, -2)
                output.append(torch.cat(
                    [position, matrix_to_quaternion(rotation), pose[..., 7:]], dim=-1))
            batch['action'][key] = torch.cat(output, dim=-1)
        return batch

    def forward(self, batch):
        return self._run(batch, inverse=False)

    def backward(self, batch):
        return self._run(batch, inverse=True)


class DualArmWristBasisTransform:
    """Map a RoboTwin local EEF basis into an explicitly named native basis.

    The transform is deliberately separate from the historical human-wrist D2
    canonicalizer.  ``basis`` contains the physical signed permutation between
    two robot tool frames, and ``origin_offset`` moves the stored RoboTwin
    control point to the gripper-center origin before the basis change.

    For a pose ``(p, R)`` and local offset ``o`` the forward map is::

        p' = p + R @ o
        R' = R @ B

    where the columns of ``B`` are the output-frame axes expressed in the
    input frame.  The backward map is exact and is used by evaluation to return
    predictions to RoboTwin's world-frame EEF convention.
    """

    def __init__(self, category_keys, basis_left=None, basis_right=None,
                 origin_offset=None, basis_path=None, num_arms=2):
        if basis_path is not None:
            import json
            contract = json.loads(open(basis_path, encoding="utf-8").read())
            basis_left = contract["basis_left"]
            basis_right = contract["basis_right"]
            origin_offset = contract["origin_offset_input_frame_m"]
        if origin_offset is None:
            origin_offset = (0.0, 0.0, 0.0)
        self.category_keys = category_keys
        self.num_arms = num_arms
        self.basis = [
            torch.as_tensor(basis_left, dtype=torch.float32),
            torch.as_tensor(basis_right, dtype=torch.float32),
        ]
        self.origin_offset = torch.as_tensor(origin_offset, dtype=torch.float32)

    def _run(self, batch, inverse):
        for cat, keys in self.category_keys.items():
            if cat not in batch:
                continue
            for key in keys:
                value = batch[cat][key]
                block = value.shape[-1] // self.num_arms
                out = []
                for arm, pose in enumerate(torch.split(value, block, dim=-1)):
                    B = self.basis[arm].to(dtype=pose.dtype, device=pose.device)
                    o = self.origin_offset.to(dtype=pose.dtype, device=pose.device)
                    R = quaternion_to_matrix(pose[..., 3:7])
                    if inverse:
                        raw_R = torch.matmul(R, B.transpose(-1, -2))
                        pos = pose[..., 0:3] - torch.matmul(
                            o.expand(raw_R.shape[:-2] + (3,)).unsqueeze(-2),
                            raw_R.transpose(-1, -2),
                        ).squeeze(-2)
                        out_R = raw_R
                    else:
                        pos = pose[..., 0:3] + torch.matmul(
                            o.expand(R.shape[:-2] + (3,)).unsqueeze(-2),
                            R.transpose(-1, -2),
                        ).squeeze(-2)
                        out_R = torch.matmul(R, B)
                    out.append(torch.cat([
                        pos, matrix_to_quaternion(out_R), pose[..., 7:]
                    ], dim=-1))
                batch[cat][key] = torch.cat(out, dim=-1)
        return batch

    def forward(self, batch):
        return self._run(batch, inverse=False)

    def backward(self, batch):
        return self._run(batch, inverse=True)


class DualArmWorldToCameraTransform:
    """Re-express each arm's EEF pose from the sim WORLD frame into the head-camera frame.

    The raw RoboTwin EEF vector is two per-arm blocks ``[pos(3), quat_wxyz(4), grip(1)]``
    (16 dims). This left-multiplies each arm's world pose ``T_we`` by ``inv(cam2world)`` so
    the pose is expressed in the camera (OpenGL: x-right, y-up, z-back) frame — the same
    frame used by the action converters via ``hand_in_camera = inv(T_camera) @ T_hand``.

    ``cam2world`` is the head camera's OpenGL model matrix (``camera.get_model_matrix()``).
    For aloha-agilex it is deterministic (``random_head_camera_dis=0``), so a single constant
    matrix aligns every episode. Quaternions are real-part-first (``wxyz``); the gripper dim
    is passed through untouched.

    forward: world -> camera. backward: camera -> world (used at eval, before the sim's
    ``take_action(action_type='ee')`` which expects world-frame poses).
    """

    def __init__(self, cam2world: List[List[float]], category_keys: Dict[str, List[str]], num_arms: int = 2):
        c2w = torch.tensor(cam2world, dtype=torch.float32)
        self.cam2world = c2w
        self.world2cam = torch.inverse(c2w)
        self.category_keys = category_keys
        self.num_arms = num_arms

    def forward(self, batch):
        for cat, ks in self.category_keys.items():
            if cat not in batch:
                continue
            for k in ks:
                batch[cat][k] = self._convert(batch[cat][k], to_camera=True)
        return batch

    def backward(self, batch):
        for cat, ks in self.category_keys.items():
            if cat not in batch:
                continue
            for k in ks:
                batch[cat][k] = self._convert(batch[cat][k], to_camera=False)
        return batch

    def _convert(self, x: torch.Tensor, to_camera: bool) -> torch.Tensor:
        block = x.shape[-1] // self.num_arms
        out = [self._convert_block(b, to_camera) for b in torch.split(x, block, dim=-1)]
        return torch.cat(out, dim=-1)

    def _convert_block(self, b: torch.Tensor, to_camera: bool) -> torch.Tensor:
        position = b[..., 0:3]
        quaternion = b[..., 3:7]  # wxyz
        others = b[..., 7:]

        M = self.world2cam if to_camera else self.cam2world
        M = M.to(dtype=b.dtype, device=b.device)

        rot = quaternion_to_matrix(quaternion)  # (..., 3, 3)
        # new_R = M[:3,:3] @ rot ; new_p = M[:3,:3] @ p + M[:3,3]
        Mr = M[:3, :3]
        Mt = M[:3, 3]
        new_rot = torch.matmul(Mr, rot)
        new_pos = torch.matmul(position, Mr.transpose(-1, -2)) + Mt
        new_quat = matrix_to_quaternion(new_rot)  # wxyz
        return torch.cat([new_pos, new_quat, others], dim=-1)
