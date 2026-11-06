"""Invertible per-arm wrist rotation calibration for absolute EEF16 poses."""
from typing import Dict, List

import torch
from ..utils.rotation import quaternion_to_matrix, matrix_to_quaternion


class _DualArmBase:
    def __init__(self, category_keys: Dict[str, List[str]], num_arms: int = 2):
        self.category_keys = category_keys
        self.num_arms = num_arms

    def _each(self, batch, fn):
        for cat, ks in self.category_keys.items():
            if cat not in batch:
                continue
            for k in ks:
                x = batch[cat][k]
                block = x.shape[-1] // self.num_arms
                blocks = torch.split(x, block, dim=-1)
                batch[cat][k] = torch.cat([fn(cat, i, b) for i, b in enumerate(blocks)], dim=-1)
        return batch


class DualArmWristCanonicalizeTransform(_DualArmBase):
    """Right-multiply each arm's rotation by the fitted constant C (R_canon = R_raw·C):
    +X=approach, mirror-free. C_left applies to arm 0, C_right to arm 1.

    Two modes:
      * single-source (default): pass `constants_path` (or C_left/C_right) — one C pair
        applied to every sample.
      * multi-source (joint pretrain): pass `sources=[{match, constants_path}, ...]`. The
        C pair is picked per-call by substring-matching `batch['wrist_source_path']` (the
        dataset root, stamped by base_lerobot_dataset) against each entry's `match`. This
        lets one RobotVideoDataset mix egodex/egoverse/vitra chunks, each canonicalized
        with its own C, without baking C into the stored data. `match` strings must be
        mutually exclusive across the mixed roots (e.g. '/egodex', '/egoverse', '/vitra').
    """

    def __init__(self, category_keys, num_arms=2, C_left=None, C_right=None,
                 constants_path=None, sources=None):
        super().__init__(category_keys, num_arms)
        self.sources = None
        if sources is not None:
            import json
            self.sources = []
            for s in sources:
                d = json.loads(open(s["constants_path"]).read())
                pair = [torch.tensor(d["C_L"], dtype=torch.float32),
                        torch.tensor(d["C_R"], dtype=torch.float32)]
                self.sources.append((s["match"], pair))
            self.C = None
        else:
            if constants_path is not None:
                import json
                d = json.loads(open(constants_path).read())
                C_left, C_right = d["C_L"], d["C_R"]
            assert C_left is not None and C_right is not None, \
                "provide C_left/C_right, constants_path, or sources"
            self.C = [torch.tensor(C_left, dtype=torch.float32),
                      torch.tensor(C_right, dtype=torch.float32)]

    def _pick_C(self, batch):
        if self.sources is None:
            return self.C
        path = batch.get("wrist_source_path")
        if path is None:
            raise KeyError(
                "DualArmWristCanonicalizeTransform(sources=...) needs batch['wrist_source_path']; "
                "base_lerobot_dataset stamps it for cam_extrinsic datasets."
            )
        path = str(path)
        for match, pair in self.sources:
            if match in path:
                return pair
        raise ValueError(
            f"no wrist-C source matches '{path}'; configured matches: "
            f"{[m for m, _ in self.sources]}"
        )

    def _mul(self, C_i, b, inverse):
        C = C_i.to(dtype=b.dtype, device=b.device)
        if inverse:
            C = C.transpose(-1, -2)
        rot = quaternion_to_matrix(b[..., 3:7])
        new_rot = torch.matmul(rot, C)
        return torch.cat([b[..., 0:3], matrix_to_quaternion(new_rot), b[..., 7:]], dim=-1)

    def forward(self, batch):
        C = self._pick_C(batch)
        return self._each(batch, lambda cat, i, b: self._mul(C[i], b, inverse=False))

    def backward(self, batch):
        C = self._pick_C(batch)
        return self._each(batch, lambda cat, i, b: self._mul(C[i], b, inverse=True))


