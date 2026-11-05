# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Dense text and generation towers for Shared-EEF action inference."""

import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLP, RMSNorm, TimestepEmbedder, apply_rotary_pos_emb


def attention(query, key, value, *, causal=False, packed=False):
    from flash_attn_3_nv.flash_attn_interface import flash_attn_func, flash_attn_varlen_func

    if packed:
        q_offsets = torch.arange(2, dtype=torch.int32, device=query.device) * query.shape[1]
        k_offsets = torch.arange(2, dtype=torch.int32, device=key.device) * key.shape[1]
        result = flash_attn_varlen_func(
            query[0],
            key[0],
            value[0],
            q_offsets,
            k_offsets,
            query.shape[1],
            key.shape[1],
            causal=causal,
        )
        result = result[0] if isinstance(result, tuple) else result
        return result.unsqueeze(0)
    result = flash_attn_func(query, key, value, causal=causal)
    return result[0] if isinstance(result, tuple) else result


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, positions, *, device, dtype):
        c = self.config
        inv = 1.0 / (
            c.rope_theta
            ** (torch.arange(0, c.head_dim, 2, dtype=torch.float32, device=device) / c.head_dim)
        )
        pos = positions.to(device=device, dtype=torch.float32)
        if pos.ndim == 2:
            pos = pos[None].expand(3, -1, -1)
        freqs = (
            inv[None, None, :, None].expand(3, pos.shape[1], -1, 1) @ pos[:, :, None]
        ).transpose(2, 3)
        temporal = freqs[0]
        for axis in (1, 2):
            selected = slice(axis, c.mrope_section[axis] * 3, 3)
            temporal[..., selected] = freqs[axis, ..., selected]
        embedded = torch.cat((temporal, temporal), dim=-1)
        return embedded.cos().to(dtype), embedded.sin().to(dtype)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(self, hidden, frequencies, text_kv=None):
        c = self.config
        shape = (*hidden.shape[:-1], -1, c.head_dim)
        query = self.q_norm(self.q_proj(hidden).view(shape))
        key = self.k_norm(self.k_proj(hidden).view(shape))
        value = self.v_proj(hidden).view(shape)
        query, key = apply_rotary_pos_emb(query, key, *frequencies, unsqueeze_dim=2)
        if text_kv is None:
            output = attention(query, key, value, causal=True)
        else:
            key = torch.cat((text_kv[0], key), dim=1)
            value = torch.cat((text_kv[1], value), dim=1)
            output = attention(query, key, value, packed=True)
        return self.o_proj(output.flatten(-2)), (key, value)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden, frequencies, text_kv=None):
        projected, kv = self.self_attn(self.input_layernorm(hidden), frequencies, text_kv)
        hidden = hidden + projected
        return hidden + self.mlp(self.post_attention_layernorm(hidden)), kv


class Tower(nn.Module):
    def __init__(self, config, *, text=False):
        super().__init__()
        if text:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class DomainLinear(nn.Module):
    def __init__(self, input_size, output_size, num_domains):
        super().__init__()
        self.input_size, self.output_size = input_size, output_size
        self.fc = nn.Embedding(num_domains, input_size * output_size)
        self.bias = nn.Embedding(num_domains, output_size)

    def forward(self, value, domains):
        weights = self.fc(domains).view(len(domains), self.input_size, self.output_size)
        return torch.bmm(value.unsqueeze(1), weights).squeeze(1) + self.bias(domains)


class SharedEEFLinear(nn.Module):
    def __init__(self, config, *, output=False):
        super().__init__()
        self.output = output
        hidden = config.hidden_size
        self.shared = nn.Linear(hidden, 19) if output else nn.Linear(19, hidden)
        self.joint = (
            DomainLinear(hidden, 23, config.num_embodiment_domains)
            if output
            else DomainLinear(23, hidden, config.num_embodiment_domains)
        )

    def forward(self, value, domains, valid, rows):
        joint_input = value if self.output else value[:, :23].masked_fill(~valid[:, :23], 0)
        selected = self.joint(joint_input[rows], domains[rows])
        joint = value.new_zeros((len(value), self.joint.output_size)).index_copy(0, rows, selected)
        if self.output:
            return torch.cat(
                (joint, self.shared(value), value.new_zeros((len(value), 22))), -1
            ).masked_fill(~valid, 0)
        return joint + self.shared(value[:, 23:42].masked_fill(~valid[:, 23:42], 0))


class GoalWAMTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text = Tower(config, text=True)
        self.generator = Tower(config)
        self.rotary = RotaryEmbedding(config)
        width = config.latent_channel * config.latent_patch_size**2
        self.vae2llm = nn.Linear(width, config.hidden_size)
        self.llm2vae = nn.Linear(config.hidden_size, width)
        self.action2llm = SharedEEFLinear(config)
        self.llm2action = SharedEEFLinear(config, output=True)
        self.time_embedder = TimestepEmbedder(config.hidden_size)
        self.goal_vision_embed = nn.Parameter(torch.zeros(config.hidden_size))
        self.action_modality_embed = nn.Parameter(torch.zeros(config.hidden_size))

    def prefill(self, ids):
        hidden = self.text.embed_tokens(ids)
        positions = torch.arange(ids.shape[1], device=ids.device)[None]
        frequencies = self.rotary(positions, device=ids.device, dtype=hidden.dtype)
        result = []
        for layer in self.text.layers:
            hidden, kv = layer(hidden, frequencies)
            result.append(tuple(value.clone() for value in kv))
        return result

    def forward(self, visual, actions, timestep, *, text_kv, frequencies, layout):
        dtype = self.vae2llm.weight.dtype
        hidden_video = self.vae2llm(visual.to(dtype))
        hidden_video = hidden_video + layout["goals"].to(dtype)[:, None] * self.goal_vision_embed
        hidden_action = self.action2llm(
            actions.to(dtype), layout["domains"], layout["valid"], layout["joint_rows"]
        )
        hidden_action = hidden_action + self.action_modality_embed
        time = timestep.float().reshape(1) * self.config.timestep_scale
        for hidden, rows in (
            (hidden_video, layout["video_rows"]),
            (hidden_action, layout["action_rows"]),
        ):
            embedding = self.time_embedder(time.expand(rows.numel())).to(dtype)
            hidden.index_add_(0, rows, embedding)
        hidden = torch.cat((hidden_video, hidden_action), dim=0)[None]
        for layer, kv in zip(self.generator.layers, text_kv, strict=True):
            hidden, _ = layer(hidden, frequencies, kv)
        hidden = self.generator.norm(hidden)[0]
        visual, action = hidden.split((len(hidden_video), len(hidden_action)))
        video_output = self.llm2vae(visual[layout["video_rows"]])
        rows = layout["action_rows"]
        action_output = self.llm2action(
            action[rows],
            layout["domains"][rows],
            layout["valid"][rows],
            layout["output_joint_rows"],
        )
        return video_output, action_output


def remap_weight(name):
    prefix = "language_model.model."
    if not name.startswith(prefix):
        return name
    name = name.removeprefix(prefix)
    generator = "_moe_gen" in name
    return ("generator." if generator else "text.") + name.replace("_moe_gen", "")


def patchify(value, patch):
    b, c, t, h, w = value.shape
    value = F.pad(value, (0, -w % patch, 0, -h % patch))
    return (
        value.reshape(b, c, t, (h + patch - 1) // patch, patch, (w + patch - 1) // patch, patch)
        .permute(0, 2, 3, 5, 4, 6, 1)
        .flatten(1, 3)
        .flatten(2)
    )


def unpatchify(value, shape, patch):
    b, c, t, h, w = shape
    hp, wp = (h + patch - 1) // patch, (w + patch - 1) // patch
    return (
        value.reshape(b, t, hp, wp, patch, patch, c)
        .permute(0, 6, 1, 2, 4, 3, 5)
        .reshape(b, c, t, hp * patch, wp * patch)[..., :h, :w]
    )
