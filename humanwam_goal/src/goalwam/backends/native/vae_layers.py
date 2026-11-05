# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Wan 2.2 image encoder building blocks."""

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

CACHE_T = 2


def _contiguous_clone(t: torch.Tensor) -> torch.Tensor:
    if t.is_contiguous():
        return t.clone()
    return t.contiguous()


def _update_cache_and_apply(
    x: torch.Tensor, layer: "CausalConv3d", feat_cache: list, feat_idx: list[int]
) -> torch.Tensor:
    idx = feat_idx[0]
    cache_x = _contiguous_clone(x[:, :, -CACHE_T:, :, :])
    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
        cache_x = torch.cat(
            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
        )
    x = layer(x, feat_cache[idx])
    feat_cache[idx] = cache_x
    feat_idx[0] += 1
    return x


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return (
            F.normalize(x, dim=1 if self.channel_first else -1) * self.scale * self.gamma
            + self.bias
        )


class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in ("none", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        if self.mode == "downsample3d":
            if feat_cache is None:
                x = F.pad(x, (0, 0, 0, 0, 2, 0))
                x = self.time_conv(x)
            else:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    if x.shape[2] == 1:
                        feat_cache[idx] = _contiguous_clone(x)
                    else:
                        cache_x = _contiguous_clone(x[:, :, -1:, :, :])
                        x_in = F.pad(x, (0, 0, 0, 0, 2, 0))
                        x = self.time_conv(x_in)
                        feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                else:
                    cache_x = _contiguous_clone(x[:, :, -1:, :, :])
                    x_cat = torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2)
                    t_cat = x_cat.shape[2]
                    if t_cat < 3:
                        x_cat = F.pad(x_cat, (0, 0, 0, 0, 3 - t_cat, 0))
                    x = self.time_conv(x_cat)
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Identity(),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                x = _update_cache_and_apply(x, layer, feat_cache, feat_idx)
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        q, k, v = (
            self.to_qkv(x)
            .reshape(b * t, 1, c * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).contiguous().reshape(b * t, c, h, w)
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity


def patchify(x, patch_size):
    if patch_size == 1:
        return x
    if patch_size == 2:
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = x.view(b, c, h // 2, 2, w // 2, 2)
            x = x.permute(0, 1, 5, 3, 2, 4).contiguous()
            return x.view(b, c * 4, h // 2, w // 2)
        if x.dim() == 5:
            b, c, f, h, w = x.shape
            x = x.view(b, c, f, h // 2, 2, w // 2, 2)
            x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
            return x.view(b, c * 4, f, h // 2, w // 2)
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(x, "b c f (h q) (w r) -> b (c r q) f h w", q=patch_size, r=patch_size)
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")
    return x


class AvgDown3D(nn.Module):
    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * self.factor, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class Down_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, mult, temperal_downsample=False, down_flag=False):
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim))
            in_dim = out_dim
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))
        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        x_shortcut = self.avg_shortcut(x)
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)
        return x + x_shortcut


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        temperal_downsample=(False, True, True),
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.temperal_downsample = temperal_downsample
        dims = [dim * u for u in (1, *dim_mult)]
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = temperal_downsample[i] if i < len(temperal_downsample) else False
            downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim),
        )
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, z_dim, 3, padding=1)
        )

    def forward(self, x, feat_cache=None):
        feat_idx = [0]
        if feat_cache is not None:
            x = _update_cache_and_apply(x, self.conv1, feat_cache, feat_idx)
        else:
            x = self.conv1(x)
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                x = _update_cache_and_apply(x, layer, feat_cache, feat_idx)
            else:
                x = layer(x)
        return x


def count_conv3d(model: nn.Module) -> int:
    return sum((1 for m in model.modules() if isinstance(m, CausalConv3d)))
