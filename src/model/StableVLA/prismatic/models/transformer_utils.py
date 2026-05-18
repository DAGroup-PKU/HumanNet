# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/functions/ms_deform_attn_func.py
# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/multi_scale_deform_attn.py
# ------------------------------------------------------------------------------------------------

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.init import constant_, xavier_uniform_
# from einops import rearrange, repeat
# from torch import einsum
# from einops_exts import rearrange_many, repeat_many

# helpers
def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class RGBDFuser(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_num: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_num = patch_num
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.channel_selection = nn.Sequential(
                            nn.Linear(out_channels, out_channels),
                            nn.Sigmoid()
                            )

    def forward(self, inputs_rgb: torch.Tensor, inputs_depth: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([inputs_rgb, inputs_depth], dim=-1)[:, 1:]
        inputs = rearrange(inputs, 'b (h w) d -> b d h w', h=self.patch_num, w=self.patch_num)
        feature = self.conv(inputs)
        feature = rearrange(feature, 'b d h w -> b (h w) d')
        selection_weights = self.channel_selection(feature.mean(dim=1))

        # channel-wise multiply
        feature = feature * selection_weights.unsqueeze(1)

        return feature


class TransFuser(nn.Module):
    def __init__(self, vis_channels: int, depth_channels: int, tgt_channels: int, num_heads: int) -> None:
        super().__init__()
        self.attn = CrossAttentionBlock(
            v_dim = tgt_channels,
            l_dim = tgt_channels,
            embed_dim = tgt_channels,
            num_heads = num_heads,
        )

        self.proj_vis = nn.Linear(vis_channels, tgt_channels)
        self.proj_depth = nn.Linear(depth_channels, tgt_channels)


    def forward(self, inputs_rgb: torch.Tensor, inputs_depth: torch.Tensor) -> torch.Tensor:
        concat_feature = torch.cat([self.proj_vis(inputs_rgb), self.proj_depth(inputs_depth)], dim=1)
        concat_feature = self.attn(concat_feature, concat_feature)

        return concat_feature

# RMSNorm -- Better, simpler alternative to LayerNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale, self.eps = dim**-0.5, eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# SwishGLU -- A Gated Linear Unit (GLU) with the Swish activation; always better than GELU MLP!
class SwishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.act, self.project = nn.SiLU(), nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)


# As defined in Set Transformers () -- basically the above, additionally taking in
# a set of $k$ learned "seed vectors" that are used to "pool" information.
class MAPAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int) -> None:
        """Multi-Input Multi-Headed Attention Operation"""
        super().__init__()
        assert embed_dim % n_heads == 0, "`embed_dim` must be divisible by `n_heads`!"
        self.n_heads, self.scale = n_heads, (embed_dim // n_heads) ** -0.5

        # Projections (no bias) --> separate for Q (seed vector), and KV ("pool" inputs)
        self.q, self.kv = nn.Linear(embed_dim, embed_dim, bias=False), nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, seed: torch.Tensor, x: torch.Tensor, attention_mask = None) -> torch.Tensor:
        (B_s, K, C_s), (B_x, N, C_x) = seed.shape, x.shape
        assert C_s == C_x, "Seed vectors and pool inputs must have the same embedding dimensionality!"

        # Project Seed Vectors to `queries`
        q = self.q(seed).reshape(B_s, K, self.n_heads, C_s // self.n_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B_x, N, 2, self.n_heads, C_x // self.n_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # Attention --> compute weighted sum over values!
        scores = q @ (k.transpose(-2, -1) * self.scale)
        # print(scores.shape)
        if attention_mask is not None:
            attention_mask = (
                attention_mask[None, None, :, :].repeat(1, self.n_heads, 1, 1) #.flatten(0, 1)
            )
            scores.masked_fill_(attention_mask == 0, float("-inf"))
        attn = scores.softmax(dim=-1)
        

        vals = (attn @ v).transpose(1, 2).reshape(B_s, K, C_s)

        # Project back to `embed_dim`
        return self.proj(vals)


class MAPBlock(nn.Module):
    def __init__(
        self,
        n_latents: int,
        vis_dim: int, 
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        do_rms_norm: bool = True,
        do_swish_glu: bool = True,
    ) -> None:
        """Multiheaded Attention Pooling Block -- note that for MAP, we adopt earlier post-norm conventions."""
        super().__init__()
        self.n_latents, self.embed_dim, self.n_heads = n_latents, embed_dim, n_heads

        # Projection Operator
        self.projection = nn.Linear(vis_dim, self.embed_dim)

        # Initialize Latents
        self.latents = nn.Parameter(torch.zeros(self.n_latents, self.embed_dim), requires_grad=True)
        nn.init.normal_(self.latents, std=0.02)

        # Custom MAP Attention (seed, encoder outputs) -> seed
        self.attn_norm = RMSNorm(self.embed_dim) if do_rms_norm else nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.attn = MAPAttention(self.embed_dim, n_heads=self.n_heads)

        # Position-wise Feed-Forward Components
        self.mlp_norm = RMSNorm(self.embed_dim) if do_rms_norm else nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            # Handle SwishGLU vs. GELU MLP...
            (
                SwishGLU(self.embed_dim, int(mlp_ratio * self.embed_dim))
                if do_swish_glu
                else nn.Sequential(nn.Linear(self.embed_dim, int(mlp_ratio * self.embed_dim)), nn.GELU())
            ),
            nn.Linear(int(mlp_ratio * self.embed_dim), self.embed_dim),
        )

    def forward(self, x: torch.Tensor, mask = None, init_embed = None) -> torch.Tensor:
        latents = repeat(self.latents, "n_latents d -> bsz n_latents d", bsz=x.shape[0])
        latents = latents + init_embed.unsqueeze(1) if init_embed is not None else latents
        latents = self.attn_norm(latents + self.attn(latents, self.projection(x), mask))
        latents = self.mlp_norm(latents + self.mlp(latents))
        return latents.squeeze(dim=1)
        

class CrossAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None, RoPE=False):
        super(CrossAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)


    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        """_summary_

        Args:
            v (_type_): bs, n_img, dim
            l (_type_): bs, n_text, dim
            attention_mask_v (_type_, optional): _description_. bs, n_img
            attention_mask_l (_type_, optional): _description_. bs, n_text

        Returns:
            _type_: _description_
        """
        bsz, tgt_len, _ = v.size()
 
        query_states = self.v_proj(v) * self.scale
        key_states = self.l_proj(l)

        key_states = self._shape(key_states, -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)

        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # bs*nhead, nimg, ntxt

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        # mask language for vision
        if attention_mask_l is not None:
            attention_mask_l = (
                attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights.masked_fill_(attention_mask_l == 0, float("-inf"))
        attn_weights_v = attn_weights.softmax(dim=-1)

        if key_states.size(1) > 1:  # only apply dropout if multiple keys
            attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        else:
            attn_probs_v = attn_weights_v

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)


        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)

        return attn_output_v
        

class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        v_dim,
        l_dim,
        embed_dim,
        num_heads,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
        cfg=None,
        RoPE=False,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(CrossAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = CrossAttention(
            v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, RoPE=RoPE
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)


    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        org_v = v
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v = self.attn(
            v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l
        )
        v = org_v + self.drop_path(self.gamma_v * delta_v)

        return v



def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, latents):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)

        q = q * self.scale

        # attention

        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        vis_dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_media_embeds = 4,
        ff_mult = 4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.num_media_embeds = num_media_embeds
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, vis_dim))
        self.proj = nn.Linear(vis_dim, dim)
        dim_head = dim // heads

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        times = x.shape[1]
        x = x + self.media_pos_emb[:times]
        x = self.proj(x)

        latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = self.num_media_embeds)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        return self.norm(latents)
