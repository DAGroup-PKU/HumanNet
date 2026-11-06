# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List
import torch
from ..utils.rotation import quaternion_to_matrix, matrix_to_quaternion

class DualArmPoseRotationTransform:
    """Canonicalize both wxyz quaternions through rotation matrices.

    Adapted from the training transform; positions and grippers are preserved.
    This release supports the quaternion representation used by RoboDojo.
    """

    def __init__(
        self,
        category_keys: Dict[str, List[str]],
        num_arms: int = 2,
        rotation_type: str = "quaternion",
    ):
        self.category_keys = category_keys
        self.num_arms = num_arms
        self.rotation_type = rotation_type
        self.rotation_dim = self._rotation_dim(rotation_type)

    def forward(self, batch):
        for cat, ks in self.category_keys.items():
            if cat not in batch:
                continue
            for k in ks:
                batch[cat][k] = self._convert(batch[cat][k], forward=True)
        return batch

    def backward(self, batch):
        for cat, ks in self.category_keys.items():
            if cat not in batch:
                continue
            for k in ks:
                batch[cat][k] = self._convert(batch[cat][k], forward=False)
        return batch

    @staticmethod
    def _rotation_dim(rotation_type):
        if rotation_type != "quaternion":
            raise ValueError("This release implements the RoboDojo quaternion contract only")
        return 4

    def _matrix_to_fitted_rotation(self, matrix):
        return matrix_to_quaternion(matrix)

    def _fitted_rotation_to_matrix(self, rotation):
        return quaternion_to_matrix(rotation)

    def _convert(self, x: torch.Tensor, forward: bool) -> torch.Tensor:
        assert x.shape[-1] % self.num_arms == 0, (
            f"pose dim {x.shape[-1]} not divisible by num_arms {self.num_arms}"
        )
        block = x.shape[-1] // self.num_arms
        expected_block = 8 if forward else 4 + self.rotation_dim
        assert block == expected_block, (
            f"expected per-arm dim {expected_block} for "
            f"{'forward' if forward else 'backward'} {self.rotation_type}, got {block}"
        )
        out = [
            self._convert_block(arm_block, forward)
            for arm_block in torch.split(x, block, dim=-1)
        ]
        return torch.cat(out, dim=-1)

    def _convert_block(self, b: torch.Tensor, forward: bool) -> torch.Tensor:
        position = b[..., 0:3]
        if forward:
            quaternion = b[..., 3:7]  # wxyz
            others = b[..., 7:]
            matrix = quaternion_to_matrix(quaternion)
            rotation = self._matrix_to_fitted_rotation(matrix)
        else:
            rotation_in = b[..., 3 : 3 + self.rotation_dim]
            others = b[..., 3 + self.rotation_dim :]
            matrix = self._fitted_rotation_to_matrix(rotation_in)
            rotation = matrix_to_quaternion(matrix)  # wxyz, non-negative real part
        return torch.cat([position, rotation, others], dim=-1)
