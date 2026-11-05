# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Independent-frame Wan 2.2 encoding for camera conditioning."""

import json
import re

import torch
from torch import nn

from .vae_layers import CausalConv3d, Encoder3d, count_conv3d, patchify
from .weights import load_weights


def encoder_weight_name(name):
    if name.startswith(("decoder.", "post_quant_conv.")):
        return None
    name = re.sub(r"^quant_conv\.", "conv1.", name)
    name = re.sub(r"^encoder\.conv_in\.", "encoder.conv1.", name)
    name = re.sub(r"^encoder\.norm_out\.", "encoder.head.0.", name)
    name = re.sub(r"^encoder\.conv_out\.", "encoder.head.2.", name)
    name = re.sub(
        r"^encoder\.mid_block\.resnets\.(\d+)\.", lambda m: f"encoder.middle.{2 * int(m[1])}.", name
    )
    name = name.replace("encoder.mid_block.attentions.0.", "encoder.middle.1.")
    name = re.sub(
        r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.",
        r"encoder.downsamples.\1.downsamples.\2.",
        name,
    )
    name = re.sub(
        r"^encoder\.down_blocks\.(\d+)\.downsampler\.",
        r"encoder.downsamples.\1.downsamples.2.",
        name,
    )
    if ".downsamples." in name or ".middle.0." in name or ".middle.2." in name:
        for old, new in (
            ("norm1.", "residual.0."),
            ("norm2.", "residual.3."),
            ("conv1.", "residual.2."),
            ("conv2.", "residual.6."),
            ("conv_shortcut.", "shortcut."),
        ):
            name = name.replace(old, new)
    return name


class ImageEncoder(nn.Module):
    def __init__(self, checkpoint, device):
        super().__init__()
        with torch.device("meta"):
            self.encoder = Encoder3d(dim=160, z_dim=96)
            self.conv1 = CausalConv3d(96, 96, 1)
        load_weights(self, checkpoint, encoder_weight_name)
        self.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        config = json.loads((checkpoint / "config.json").read_text())
        self.mean = torch.tensor(config["latents_mean"], dtype=torch.bfloat16, device=device).view(
            1, 48, 1, 1, 1
        )
        self.inv_std = (
            1.0 / torch.tensor(config["latents_std"], dtype=torch.bfloat16, device=device)
        ).view(1, 48, 1, 1, 1)
        self.cache_size = count_conv3d(self.encoder)

    @torch.inference_mode()
    def forward(self, pixels):
        if pixels.shape[:3] != (1, 3, 1):
            raise ValueError("Encode independent camera frames with shape [1,3,1,H,W]")
        pixels = pixels.to(device=self.mean.device, dtype=torch.bfloat16)
        with torch.backends.cudnn.flags(
            enabled=True, benchmark=False, deterministic=True, allow_tf32=False
        ):
            features = self.encoder(patchify(pixels, 2), feat_cache=[None] * self.cache_size)
            mean = self.conv1(features).chunk(2, dim=1)[0]
        return (mean - self.mean) * self.inv_std
