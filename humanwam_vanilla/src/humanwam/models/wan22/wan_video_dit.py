# Modified by Open-HumanScale contributors: package namespace renamed to humanwam.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from functools import lru_cache
from typing import Any, Dict, Tuple, Optional, Union
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel
from .helpers.gradient import gradient_checkpoint_forward

from humanwam.camera_pose import (
    CAMERA_POSE_BLOCK_DIM,
    build_camera_pose_blocks,
    camera_pose_fixed_normalization)
from humanwam.utils.logging_config import get_logger

logger = get_logger(__name__)


_FA4_STATE: Dict[str, Any] = {"func": None, "unavailable_reason": None}


class CameraPoseConditioner(nn.Module):
    """Encode canonical per-RGB-frame camera deltas into per-latent embeddings."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        pose_hidden_dim: int,
        translation_tolerance_m: float,
        rotation_tolerance_deg: float,
        representation: str,
        dropout: float,
        local_translation_scale_m: float,
        local_rotation_scale_rad: float,
    ) -> None:
        super().__init__()
        self.translation_tolerance_m = float(translation_tolerance_m)
        self.rotation_tolerance_deg = float(rotation_tolerance_deg)
        self.representation = str(representation).strip().lower()
        self.dropout = float(dropout)
        mean, std = camera_pose_fixed_normalization(
            local_translation_scale_m=float(local_translation_scale_m),
            local_rotation_scale_rad=float(local_rotation_scale_rad),
        )
        self.register_buffer("input_mean", mean, persistent=True)
        self.register_buffer("input_std", std, persistent=True)
        self.encoder = nn.Sequential(
            nn.Linear(CAMERA_POSE_BLOCK_DIM, pose_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(pose_hidden_dim),
            nn.Linear(pose_hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(
        self,
        camera_pose_delta: torch.Tensor,
        camera_pose_valid: Optional[torch.Tensor],
        *,
        num_latent_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Geometry is deliberately evaluated in fp32 even when the DiT runs in
        # bf16. The tiny result is cast only at the pose MLP boundary.
        blocks, stationary = build_camera_pose_blocks(
            camera_pose_delta.float(),
            camera_pose_valid,
            num_latent_frames=num_latent_frames,
            translation_tolerance_m=self.translation_tolerance_m,
            rotation_tolerance_deg=self.rotation_tolerance_deg,
            representation=self.representation,
        )
        normalized = (blocks - self.input_mean.float()) / self.input_std.float()
        parameter = self.encoder[0].weight
        embedding = self.encoder(
            normalized.to(device=parameter.device, dtype=parameter.dtype)
        )
        if self.training and self.dropout > 0.0:
            keep = torch.rand(
                (embedding.shape[0], 1, 1),
                device=embedding.device,
                dtype=torch.float32,
            ) >= self.dropout
            embedding = embedding * keep.to(dtype=embedding.dtype)
        return embedding, stationary


class ActionAdaLNConditioner(nn.Module):
    """Encode per-latent-frame action chunks for VideoDiT AdaLN conditioning.

    The action sequence is grouped by the number of future latent frames.  A
    small shared encoder first embeds each raw action, then mean-pools the
    actions belonging to one latent transition.  The pooled representation
    produces both a timestep embedding and the six AdaLN modulation tensors
    used by every video DiT block.  The output heads are zero initialized so
    enabling this path does not perturb a pretrained video expert at step 0.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int,
        conditioner_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.action_dim, int(conditioner_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(conditioner_hidden_dim), int(conditioner_hidden_dim)),
            nn.LayerNorm(int(conditioner_hidden_dim)),
        )
        self.to_time = nn.Linear(int(conditioner_hidden_dim), self.hidden_dim)
        self.to_modulation = nn.Linear(
            int(conditioner_hidden_dim), 6 * self.hidden_dim
        )
        # AdaLN-zero: preserve the pretrained/no-condition path initially.
        nn.init.zeros_(self.to_time.weight)
        nn.init.zeros_(self.to_time.bias)
        nn.init.zeros_(self.to_modulation.weight)
        nn.init.zeros_(self.to_modulation.bias)

    def forward(
        self,
        action: torch.Tensor,
        *,
        num_future_groups: int,
        tokens_per_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_horizon = int(action.shape[1])

        actions_per_group = action_horizon // num_future_groups
        grouped = action.reshape(
            action.shape[0], num_future_groups, actions_per_group, self.action_dim
        )
        encoded = self.encoder(grouped)
        # Pool only within a transition group; temporal groups remain distinct.
        grouped = encoded.mean(dim=2)
        time_group = self.to_time(grouped)
        modulation_group = self.to_modulation(grouped).reshape(
            action.shape[0], num_future_groups, 6, self.hidden_dim
        )
        # Expand one condition to all spatial tokens in the latent frame.
        time_tokens = time_group.repeat_interleave(tokens_per_frame, dim=1)
        modulation_tokens = modulation_group.repeat_interleave(
            tokens_per_frame, dim=1
        )
        return time_tokens, modulation_tokens


def _fa4_func():
    """Lazily import FlashAttention-4, caching both success and failure.

    FA4 is an optional experimental backend installed in a separate prefix; the
    shared environment does not carry it.
    """
    if _FA4_STATE["func"] is not None:
        return _FA4_STATE["func"]
    from flash_attn.cute.interface import flash_attn_func
    _FA4_STATE["func"] = flash_attn_func
    return flash_attn_func


@lru_cache(maxsize=None)
def _fa4_first_frame_causal_mask_mod(video_tokens: int, first_frame_tokens: int, mode: str = "isolated"):
    """`mask_mod` equivalent of the boolean first-frame-causal joint mask.

    Mirrors `FastWAM._build_mot_attention_mask` with `video_key_gate=None`:
    first-frame video queries read only first-frame video keys, later video
    queries read all video keys, video queries never read action keys, and action
    queries read all action keys plus the first-frame video keys. Cache the
    closure by shape so all MoT layers share one callable and one FA4 JIT key.
    """

    if mode == "mutual":
        def mutual_mask_mod(b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors, *aux_scalars):
            return (q_idx >= first_frame_tokens) | (kv_idx < first_frame_tokens)
        return mutual_mask_mod

    # FA4 calls mask_mod as
    # (batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors[, aux_scalars]).
    def mask_mod(b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors, *aux_scalars):
        video_q = q_idx < video_tokens
        video_keep = (kv_idx < video_tokens) & (
            (q_idx >= first_frame_tokens) | (kv_idx < first_frame_tokens)
        )
        action_keep = (kv_idx >= video_tokens) | (kv_idx < first_frame_tokens)
        return (video_q & video_keep) | ((q_idx >= video_tokens) & action_keep)

    return mask_mod


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    ctx_mask: Optional[torch.Tensor] = None,
    compatibility_mode=True,
    joint_mask_spec: Optional[Union[Tuple[int, int], Tuple[int, int, str]]] = None,
):
    """Attention helper.

    `joint_mask_spec` is `(video_tokens, first_frame_tokens[, mode])`; absent mode
    preserves the legacy isolated mask, while `mutual` connects noisy futures.
    It is supplied only
    by the MoT video-action mixed attention, whose boolean mask FA4 can express as
    a `mask_mod`. Every other call site leaves it None and always uses SDPA, so the
    fa4 backend never touches cross-attention.
    """
    if compatibility_mode:
        default_backend_name = os.environ.get(
            "FASTWAM_SDPA_BACKEND", "default"
        ).strip().lower()
        backend_name = default_backend_name
        if joint_mask_spec is not None:
            backend_name = os.environ.get(
                "FASTWAM_MIXED_SDPA_BACKEND", default_backend_name
            ).strip().lower()
        if backend_name == "fa4":
            # FA4 takes (b, s, n, d); an additive float mask means a non-trivial
            # `video_key_gate`, which a boolean mask_mod cannot express.
            if joint_mask_spec is not None and (
                ctx_mask is None or ctx_mask.dtype == torch.bool
            ):
                video_tokens, first_frame_tokens = joint_mask_spec[:2]
                mode = joint_mask_spec[2] if len(joint_mask_spec) == 3 else "isolated"
                qh = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
                kh = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
                vh = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
                # FlashAttnFunc.apply always returns (out, lse).
                x = _fa4_func()(
                    qh, kh, vh,
                    mask_mod=_fa4_first_frame_causal_mask_mod(
                        int(video_tokens), int(first_frame_tokens), mode
                    ),
                )[0]
                return rearrange(x, "b s n d -> b s (n d)", n=num_heads)
            # Keep the non-FA4 fallback on the configured default backend. This
            # matters when mixed attention opts into FA4 while cross attention
            # must stay on the known-finite efficient SDPA path. Preserve the
            # legacy global `FASTWAM_SDPA_BACKEND=fa4` behavior by falling back
            # to cuDNN when there is no distinct default backend.
            backend_name = (
                default_backend_name
                if default_backend_name != "fa4"
                else "cudnn"
            )

        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        if backend_name == "default":
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
        else:
            backends = {
                "flash": SDPBackend.FLASH_ATTENTION,
                "efficient": SDPBackend.EFFICIENT_ATTENTION,
                "cudnn": SDPBackend.CUDNN_ATTENTION,
                "math": SDPBackend.MATH,
            }
            with sdpa_kernel(backends=[backends[backend_name]]):
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def create_group_causal_attn_mask(
    num_temporal_groups: int, num_query_per_group: int, num_key_per_group: int, mode: str = "causal"
) -> torch.Tensor:
    """
    Creates a group-based attention mask for scaled dot-product attention with two modes:
    'causal' and 'group_diagonal'.

    Parameters:
    - num_temporal_groups (int): The number of temporal groups (e.g., frames in a video sequence).
    - num_query_per_group (int): The number of query tokens per temporal group. (e.g., latent tokens in a frame, H x W).
    - num_key_per_group (int): The number of key tokens per temporal group. (e.g., action tokens per frame).
    - mode (str): The mode of the attention mask. Options are:
        - 'causal': Query tokens can attend to key tokens from the same or previous temporal groups.
        - 'group_diagonal': Query tokens can attend only to key tokens from the same temporal group.

    Returns:
    - attn_mask (torch.Tensor): A boolean tensor of shape (L, S), where:
        - L = num_temporal_groups * num_query_per_group (total number of query tokens)
        - S = num_temporal_groups * num_key_per_group (total number of key tokens)
      The mask indicates where attention is allowed (True) and disallowed (False).

    Example:
    Input:
        num_temporal_groups = 3
        num_query_per_group = 4
        num_key_per_group = 2
    Output:
        Causal Mask Shape: torch.Size([12, 6])
        Group Diagonal Mask Shape: torch.Size([12, 6])
        if mode='causal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True]])

        if mode='group_diagonal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True]])

    """

    # Generate time indices for query and key tokens (shape: [L] and [S])
    query_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_query_per_group)  # Shape: [L]
    key_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_key_per_group)  # Shape: [S]

    # Expand dimensions to compute outer comparison
    query_time_indices = query_time_indices.unsqueeze(1)  # Shape: [L, 1]
    key_time_indices = key_time_indices.unsqueeze(0)  # Shape: [1, S]

    if mode == "causal":
        # Causal Mode: Query can attend to keys where key_time <= query_time
        attn_mask = query_time_indices >= key_time_indices  # Shape: [L, S]
    elif mode == "group_diagonal":
        # Group Diagonal Mode: Query can attend only to keys where key_time == query_time
        attn_mask = query_time_indices == key_time_indices  # Shape: [L, S]
    return attn_mask


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x, freqs, self_attn_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=self_attn_mask)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6,):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)

    def compute_kv(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.norm_k(self.k(ctx)), self.v(ctx)

    def forward_with_kv_cache(self, x: torch.Tensor, cached_k: torch.Tensor, cached_v: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        x = flash_attention(q=q, k=cached_k, v=cached_v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self,  hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, hidden_dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()
        # Optional PEFT modules are injected after model construction and before
        # checkpoint load/FSDP wrapping.  ``None`` adds no checkpoint keys and
        # therefore preserves every legacy Wan/FastWAM checkpoint.
        self.post_self_attn_adapter: Optional[nn.Module] = None
        self.post_ffn_adapter: Optional[nn.Module] = None

    def forward(self, x, context, t_mod, freqs, context_mask=None, self_attn_mask: Optional[torch.Tensor] = None):
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1) # (B, 1, seq_len, context_len), 1 for heads
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask))
        if self.post_self_attn_adapter is not None:
            x = self.post_self_attn_adapter(x)
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        if self.post_ffn_adapter is not None:
            x = self.post_ffn_adapter(x)
        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanVideoDiT(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = False,
        require_clip_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
        action_conditioned: bool = False,
        action_dim: int = 7,
        action_group_causal_mask_mode = "causal",
        action_cross_attention: bool = True,
        action_adaln: bool = False,
        action_adaln_hidden_dim: int = 512,
        video_attention_mask_mode: str = "bidirectional",
        use_gradient_checkpointing: bool = False,
        num_view_embeddings: int = 0,
        camera_pose_conditioning: bool = False,
        camera_pose_hidden_dim: int = 256,
        camera_pose_translation_tolerance_m: float = 0.005,
        camera_pose_rotation_tolerance_deg: float = 0.5,
        camera_pose_representation: str = "euler6",
        camera_pose_dropout: float = 0.1,
        camera_pose_local_translation_scale_m: float = 0.02,
        camera_pose_local_rotation_scale_rad: float = 0.05,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.video_attention_mask_mode = str(video_attention_mask_mode)
        self.num_view_embeddings = int(num_view_embeddings)
        self.camera_pose_conditioning = bool(camera_pose_conditioning)

        self.action_conditioned = action_conditioned
        self.action_dim = action_dim
        self.action_cross_attention = bool(action_cross_attention)
        self.action_adaln = bool(action_adaln)
        self.action_adaln_hidden_dim = int(action_adaln_hidden_dim)

        self.patch_embedding = nn.Conv3d(
            in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.camera_pose_conditioner = (
            CameraPoseConditioner(
                hidden_dim=hidden_dim,
                pose_hidden_dim=int(camera_pose_hidden_dim),
                translation_tolerance_m=float(
                    camera_pose_translation_tolerance_m
                ),
                rotation_tolerance_deg=float(
                    camera_pose_rotation_tolerance_deg
                ),
                representation=str(camera_pose_representation),
                dropout=float(camera_pose_dropout),
                local_translation_scale_m=float(
                    camera_pose_local_translation_scale_m
                ),
                local_rotation_scale_rad=float(
                    camera_pose_local_rotation_scale_rad
                ),
            )
            if self.camera_pose_conditioning
            else None
        )
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(hidden_dim, out_dim, patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(attn_head_dim)
        self.view_embedding = None
        if self.num_view_embeddings > 0:
            # View 0 is the pretrained head-camera contract.  padding_idx keeps
            # that row exactly zero (including its gradient), while newly added
            # wrist views can learn residual identity embeddings from zero.
            self.view_embedding = nn.Embedding(
                self.num_view_embeddings,
                hidden_dim,
                padding_idx=0,
            )
            nn.init.zeros_(self.view_embedding.weight)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, hidden_dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

        if self.action_conditioned:
            self.action_group_causal_mask_mode = action_group_causal_mask_mode
            if self.action_cross_attention:
                self.action_embedding = nn.Linear(action_dim, hidden_dim)
            if self.action_adaln:
                self.action_adaln_conditioner = ActionAdaLNConditioner(
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                    conditioner_hidden_dim=self.action_adaln_hidden_dim,
                )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        if self.use_gradient_checkpointing:
            logger.info("Using gradient checkpointing for DiT blocks. This will save memory but use more computation.")


    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def _prepare_forward_inputs(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        action: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context_mask is None:
            context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)

        batch_size = x.shape[0]
        if batch_size != context.shape[0]:
            if not self.training and batch_size == 1:
                x = x.expand(context.shape[0], -1, -1, -1, -1)
                batch_size = context.shape[0]
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        return x, timestep, context_mask

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:

        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if self.video_attention_mask_mode == "per_frame_causal":
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(
                torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device)
            )
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if self.video_attention_mask_mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
        control_camera_latents_input: Optional[torch.Tensor] = None,
        temporal_index: Optional[torch.Tensor] = None,
        spatial_view_ids: Optional[torch.Tensor] = None,
        spatial_row_ids: Optional[torch.Tensor] = None,
        spatial_col_ids: Optional[torch.Tensor] = None,
        camera_pose_delta: Optional[torch.Tensor] = None,
        camera_pose_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        x, timestep, context_mask = self._prepare_forward_inputs(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
        )

        batch_size = x.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        tokens_per_frame = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)

        if self.seperated_timestep and fuse_vae_embedding_in_latents:

            token_timesteps = torch.ones(
                (batch_size, x.shape[2], tokens_per_frame),
                dtype=timestep.dtype,
                device=timestep.device,
            ) * timestep.view(batch_size, 1, 1)
            token_timesteps[:, 0, :] = 0
            token_timesteps = token_timesteps.reshape(batch_size, -1)
            token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
            t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
            pose_stationary = None
            if self.camera_pose_conditioner is not None:
                if camera_pose_delta is None:
                    camera_pose_delta = torch.eye(
                        4, dtype=torch.float32, device=x.device
                    ).reshape(1, 1, 4, 4).expand(batch_size, -1, -1, -1)
                    camera_pose_valid = torch.ones(
                        (batch_size, 1), dtype=torch.bool, device=x.device
                    )
                pose_by_latent, pose_stationary = self.camera_pose_conditioner(
                    camera_pose_delta,
                    camera_pose_valid,
                    num_latent_frames=int(x.shape[2]),
                )
                pose_by_token = pose_by_latent.repeat_interleave(
                    tokens_per_frame, dim=1
                )
                # Latent zero is intentionally included: its pose block is exact
                # identity, matching stationary future blocks after deadband.
                t = t + pose_by_token.to(dtype=t.dtype)
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        else:
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        x = self.patchify(x, control_camera_latents_input=control_camera_latents_input)
        f, h, w = x.shape[2:]

        if self.action_adaln and action is not None:
            action_time_tokens, action_mod_tokens = (
                self.action_adaln_conditioner(
                    action,
                    num_future_groups=f - 1,
                    tokens_per_frame=tokens_per_frame,
                )
            )
            null_time = torch.zeros_like(t[:, :tokens_per_frame])
            null_modulation = torch.zeros(
                (
                    batch_size,
                    tokens_per_frame,
                    6,
                    self.hidden_dim,
                ),
                dtype=t.dtype,
                device=t.device,
            )
            action_time = torch.cat([null_time, action_time_tokens], dim=1)
            action_modulation = torch.cat(
                [null_modulation, action_mod_tokens], dim=1
            )
            t = t + action_time.to(dtype=t.dtype, device=t.device)
            # Re-project after adding the action-dependent timestep stream, then
            # add the zero-initialized action-specific AdaLN residual.
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
            t_mod = t_mod + action_modulation.to(
                dtype=t_mod.dtype, device=t_mod.device
            )

        context = self.text_embedding(context) # (B, L, dim)
        context_len = context.shape[1]
        if self.action_conditioned and self.action_cross_attention and action is not None:
            action_len = action.shape[1]
            action_emb = self.action_embedding(action) # (B, action_len, dim)
            action_pos_embed = sinusoidal_embedding_1d(self.hidden_dim,
                torch.arange(action_len, device=action_emb.device)) # (action_len, dim)
            action_emb = action_emb + action_pos_embed.unsqueeze(0) # (B, action_len, dim)
            context = torch.cat([context, action_emb], dim=1) # (B, context_len + action_len, dim)

            # new mask
            num_temporal_groups = f - 1 # first latent frame do not attend to actions
            # Each latent frame (from the 2nd one) attends to the corresponding group of action tokens
            action_group_mask = create_group_causal_attn_mask(
                num_temporal_groups=num_temporal_groups,
                num_query_per_group=tokens_per_frame,
                num_key_per_group=action_len // num_temporal_groups,
                mode=self.action_group_causal_mask_mode,
            ).to(context.device) # ((f-1)*tokens_per_frame, action_len)

            seq_len = f * h * w # query length
            final_context_mask = torch.zeros((batch_size, seq_len, context.shape[1]), dtype=torch.bool, device=context.device) # (B, seq_len, L + action_len)
            # all latent frames attend to text tokens
            final_context_mask[:, :, :context_len] = context_mask.unsqueeze(1).expand(-1, seq_len, -1) # (B, seq_len, L)
            # latent frames from the 2nd one attend to action tokens
            final_context_mask[:, tokens_per_frame:, context_len:] = action_group_mask.unsqueeze(0).expand(batch_size, -1, -1) # (B, seq_len, action_len)
            context_mask = final_context_mask
        elif self.action_conditioned and self.action_cross_attention and action is None:
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1) # (B, seq_len, L)
        else:
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1) # (B, seq_len, L)

        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        custom_spatial_positions = spatial_row_ids is not None or spatial_col_ids is not None
        if custom_spatial_positions:
            spatial_row_ids = torch.as_tensor(
                spatial_row_ids, device=x_tokens.device, dtype=torch.long
            )
            spatial_col_ids = torch.as_tensor(
                spatial_col_ids, device=x_tokens.device, dtype=torch.long
            )

        if spatial_view_ids is not None:
            spatial_view_ids = torch.as_tensor(
                spatial_view_ids, device=x_tokens.device, dtype=torch.long
            )
            per_frame_view_ids = spatial_view_ids.reshape(-1)
            token_view_ids = per_frame_view_ids.repeat(f)
            x_tokens = x_tokens + self.view_embedding(token_view_ids).unsqueeze(0).to(
                dtype=x_tokens.dtype
            )

        if custom_spatial_positions:
            # `self.freqs` stays on CPU; the id tensors were moved to the token
            # device above for `view_embedding`. Index the RoPE tables with ids on
            # the tables' own device (same pattern as the `temporal_index` branch).
            rope_row_ids = spatial_row_ids.to(self.freqs[1].device)
            rope_col_ids = spatial_col_ids.to(self.freqs[2].device)

        if temporal_index is None:
            temporal_freqs = self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1)
            if custom_spatial_positions:
                height_freqs = self.freqs[1][rope_row_ids].view(1, h, w, -1).expand(f, h, w, -1)
                width_freqs = self.freqs[2][rope_col_ids].view(1, h, w, -1).expand(f, h, w, -1)
            else:
                height_freqs = self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1)
                width_freqs = self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            freqs = torch.cat(
                [temporal_freqs, height_freqs, width_freqs], dim=-1
            ).reshape(f * h * w, 1, -1).to(x_tokens.device)
        else:
            # Per-sample temporal RoPE indices (GoalWAM exp5): the long-goal latent's
            # temporal position encodes its true distance to the trajectory end, so
            # the temporal frequency varies per batch element. `temporal_index` is
            # [B, f] (long); build freqs of shape [B, f*h*w, 1, dim]. rope_apply
            # broadcasts over the head dim, so per-sample freqs are applied correctly.
            ti = temporal_index.to(self.freqs[0].device).long()  # [B, f]
            B = ti.shape[0]
            f_part = self.freqs[0][ti].view(B, f, 1, 1, -1).expand(B, f, h, w, -1)
            if custom_spatial_positions:
                h_part = self.freqs[1][rope_row_ids].view(1, 1, h, w, -1).expand(B, f, h, w, -1)
                w_part = self.freqs[2][rope_col_ids].view(1, 1, h, w, -1).expand(B, f, h, w, -1)
            else:
                h_part = self.freqs[1][:h].view(1, 1, h, 1, -1).expand(B, f, h, w, -1)
                w_part = self.freqs[2][:w].view(1, 1, 1, w, -1).expand(B, f, h, w, -1)
            freqs = torch.cat([f_part, h_part, w_part], dim=-1).reshape(
                B, f * h * w, 1, -1
            ).to(x_tokens.device)

        return {
            "tokens": x_tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "context_mask": context_mask,
            "meta": {
                "grid_size": (f, h, w),
                "tokens_per_frame": tokens_per_frame,
                "batch_size": batch_size,
                "spatial_view_ids": spatial_view_ids,
                "camera_pose_stationary": pose_stationary,
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        f, h, w = pre_state["meta"]["grid_size"]
        x = self.head(x_tokens, pre_state["t"])
        x = self.unpatchify(x, (f, h, w))
        return x

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
        temporal_index: Optional[torch.Tensor] = None,
        camera_pose_delta: Optional[torch.Tensor] = None,
        camera_pose_valid: Optional[torch.Tensor] = None,
    ):
        pre_state = self.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            temporal_index=temporal_index,
            camera_pose_delta=camera_pose_delta,
            camera_pose_valid=camera_pose_valid,
        )
        x_tokens = pre_state["tokens"]
        context_emb = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_attn_mask = pre_state["context_mask"]
        self_attn_mask = self.build_video_to_video_mask(
            video_seq_len=x_tokens.shape[1],
            video_tokens_per_frame=int(pre_state["meta"]["tokens_per_frame"]),
            device=x_tokens.device,
        ) if self.video_attention_mask_mode != "bidirectional" else None # special rule for faster speed

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x_tokens = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x_tokens, context_emb, t_mod, freqs, context_mask=context_attn_mask, self_attn_mask=self_attn_mask
                )
            else:
                x_tokens = block(x_tokens, context_emb, t_mod, freqs, context_mask=context_attn_mask, self_attn_mask=self_attn_mask)

        return self.post_dit(x_tokens, pre_state)
