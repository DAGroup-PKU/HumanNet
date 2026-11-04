# Modified by Open-HumanScale contributors: package namespace renamed to humanwam.
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .wan_video_dit import flash_attention, modulate, rope_apply
from humanwam.utils.logging_config import get_logger

logger = get_logger(__name__)

_CUDAGRAPH_COMPILE_MODES = frozenset({"reduce-overhead", "max-autotune"})


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
        compile_blocks: bool = False,
        compile_dynamic: bool = True,
        compile_mode: str = "default",
    ):
        super().__init__()

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        if mot_checkpoint_mixed_attn:
            logger.info(
                "MoT: recomputing mixed attention in backward (mot_checkpoint_mixed_attn=True). "
                "Saves activation memory but is the costliest thing to recompute (Cosmos 3 §5.2.4 "
                "keeps attention materialized); prefer False unless OOM."
            )
        else:
            logger.info("MoT: attention output materialized (no attention recompute).")

        # torch.compile the pure block compute (Cosmos 3 §5.2.5). We compile the inner
        # helper *callables* (not nn.Modules), so accelerate.unwrap_model/`_orig_mod` and
        # manual FSDP wrapping are untouched. Dynamic mode absorbs variable sequence
        # lengths from goal/video routing; fixed-shape training can disable it for a
        # smaller static graph. Composes with gradient checkpointing:
        # torch.utils.checkpoint(compiled_fn, use_reentrant=False).
        self.compile_blocks = bool(compile_blocks)
        self.compile_dynamic = bool(compile_dynamic)
        self.compile_mode = str(compile_mode)
        self.io_compile_mode: Optional[str] = None
        self.post_compile_mode: Optional[str] = None
        self.io_cudagraph_enabled = False
        self.post_cudagraph_enabled = False
        if self.compile_blocks:
            checkpoint_post = any(
                bool(getattr(expert, "use_gradient_checkpointing", False))
                for expert in self.mixtures.values()
            )
            self.io_compile_mode = self.compile_mode
            self.post_compile_mode = self.compile_mode
            self.io_cudagraph_enabled = (
                self.io_compile_mode in _CUDAGRAPH_COMPILE_MODES
            )
            if checkpoint_post and self.io_cudagraph_enabled:
                # Non-reentrant checkpoint replays this callable during backward.
                # CUDAGraph Trees cannot currently preserve the replay input
                # lifetimes and either trips its invariant or observes stale
                # in-place versions. Keep the IO helper graphed, but compile the
                # checkpointed post helper without CUDA Graph capture.
                self.post_compile_mode = "default"
            self.post_cudagraph_enabled = (
                self.post_compile_mode in _CUDAGRAPH_COMPILE_MODES
            )
            logger.info(
                "MoT: torch.compile-ing transformer block helpers "
                "(dynamic=%s, io_mode=%s, post_mode=%s, "
                "io_cudagraph=%s, post_cudagraph=%s, checkpoint_post=%s; "
                "Cosmos 3 §5.2.5).",
                self.compile_dynamic,
                self.io_compile_mode,
                self.post_compile_mode,
                self.io_cudagraph_enabled,
                self.post_cudagraph_enabled,
                checkpoint_post,
            )
            io_compile_kwargs = {
                "dynamic": self.compile_dynamic,
                "mode": self.io_compile_mode,
            }
            post_compile_kwargs = {
                "dynamic": self.compile_dynamic,
                "mode": self.post_compile_mode,
            }
            self._io_compute = torch.compile(
                self._io_compute_impl,
                **io_compile_kwargs,
            )
            self._post_compute = torch.compile(
                self._apply_expert_post_block,
                **post_compile_kwargs,
            )
        else:
            self._io_compute = self._io_compute_impl
            self._post_compute = self._apply_expert_post_block

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]

        logger.info(f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}")
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B")

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
        joint_mask_spec: Optional[Union[Tuple[int, int], Tuple[int, int, str]]] = None,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)
        if attn_mask.ndim == 3:
            # Per-sample [B,S,S] mask broadcast over heads for SDPA.
            attn_mask = attn_mask.unsqueeze(1)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(
                q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask,
                joint_mask_spec=joint_mask_spec,
            )

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
        cross_attn_kv: Optional[tuple] = None,
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))
        post_self_attn_adapter = getattr(block, "post_self_attn_adapter", None)
        if post_self_attn_adapter is not None:
            # Sequential, token-wise residual adapter.  It intentionally sits
            # before the frozen text cross-attention so the PEFT treatment
            # matches the existing self-attention + FFN LoRA target contract.
            x = post_self_attn_adapter(x)

        if cross_attn_kv is not None:
            # Inference fast path: cross-attn K/V were prefilled once from the (static)
            # text context, so per-denoise-step we only recompute the query. Bit-exact
            # vs the eager branch below. See MoT.prefill_cross_attn_cache.
            context_mask = context_payload.get("mask") if context_payload is not None else None
            if context_mask is not None and context_mask.dim() == 3:
                context_mask = context_mask.unsqueeze(1)
            x = x + block.cross_attn.forward_with_kv_cache(
                block.norm3(x), cross_attn_kv[0], cross_attn_kv[1], ctx_mask=context_mask,
            )
        elif context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        post_ffn_adapter = getattr(block, "post_ffn_adapter", None)
        if post_ffn_adapter is not None:
            x = post_ffn_adapter(x)
        return x

    @staticmethod
    def _io_compute_impl(
        block,
        x: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Pure pre-attention block compute (no module state read beyond `block`).

        Splits time modulation and applies norm1+modulate plus the q/k/v projections
        with RMSNorm. RoPE intentionally stays outside this compiled helper: its
        float64/complex implementation is unsupported by TorchInductor on the Arena
        PyTorch 2.9 stack and produced non-finite gradients when captured. The real-
        valued pointwise/projection path remains the natural unit to `torch.compile`
        (Cosmos 3 §5.2.5). Returns
        (q, k, v, gate_msa, shift_mlp, scale_mlp, gate_mlp).
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = MoT._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        return q, k, v, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        q, k, v, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._io_compute(
            block, x, t_mod
        )
        # Keep float64/complex RoPE eager. Capturing it in Inductor on PyTorch 2.9
        # emitted unsupported-complex warnings and made every rank's first-step
        # gradient norm NaN in the L20X static-compile smoke.
        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
        cross_attn_kv: Optional[tuple] = None,
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
            _cross_attn_kv=cross_attn_kv,
        ) -> torch.Tensor:
            # `self._post_compute` is `_apply_expert_post_block`, optionally torch.compiled
            # (Cosmos 3 §5.2.5). Positional args so the compiled callable matches.
            # `_cross_attn_kv` is None in training (full forward); set only on the cached
            # inference path, where compile_blocks is off so the graph captures eager.
            return self._post_compute(
                _block,
                _x,
                _mixed_slice,
                _gate_msa,
                _shift_mlp,
                _scale_mlp,
                _gate_mlp,
                _context_payload,
                _cross_attn_kv,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Prefill video branch once and cache per-layer K/V for action denoising.

        Args:
            video_tokens: Video tokens before layer 0, shape [B, Sv, D].
            video_freqs: Video RoPE frequencies, shape [Sv, 1, rope_dim].
            video_t_mod: Video time modulation tensor.
            video_context_payload: Optional dict for video cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sv, L] or [B, 1, Sv, L]
            video_attention_mask: Video self-attention mask, shape [Sv, Sv].

        Returns:
            Layer-wise cache list with length `num_layers`.
            Each entry contains:
                - `k`: video key tensor [B, Sv, H*Dh]
                - `v`: video value tensor [B, Sv, H*Dh]
        """

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Build video Q/K/V from current layer input tokens.
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            # Video prefill uses only video self-attention mask.
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            # Update video tokens for the next layer and persist current layer K/V.
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            kv_cache.append({"k": k, "v": v})
        return kv_cache

    @torch.no_grad()
    def prefill_cross_attn_cache(
        self,
        expert_name: str,
        context: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Prefill per-layer cross-attention K/V for an expert from a static context.

        The text/context K/V are constant across denoise steps, so computing them once
        and reusing via `forward_with_kv_cache` removes a full K/V projection per layer
        per step. Bit-exact. Returns a length-`num_layers` list of (k, v).
        """
        expert = self.mixtures[expert_name]
        cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            k, v = block.cross_attn.compute_kv(context)
            cache.append((k, v))
        return cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        cross_attn_kv_cache: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        # Use the action query rows from the joint [video+action] mask.
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Action query/key/value are still step-dependent and must be recomputed each step.
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            layer_cache = video_kv_cache[layer_idx]

            k_video = layer_cache["k"]
            v_video = layer_cache["v"]

            # Mixed attention: action queries attend to cached video K/V plus current action K/V.
            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
                cross_attn_kv=cross_attn_kv_cache[layer_idx] if cross_attn_kv_cache is not None else None,
            )
        return x

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        joint_mask_spec: Optional[Union[Tuple[int, int], Tuple[int, int, str]]] = None,
    ):
        # Delimit iterations only when the safe IO helper actually uses a
        # CUDAGraph mode. The checkpoint-replayed post helper is independently
        # compiled in default mode and therefore remains outside the graph.
        if self.io_cudagraph_enabled:
            torch.compiler.cudagraph_mark_step_begin()

        tokens_all = {k: v for k, v in embeds_all.items()}

        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }

            # 3. concat all tokens for mixed attention
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            mixed = self._mixed_attention(
                q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask,
                joint_mask_spec=joint_mask_spec,
            )

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                # 4. split mixed attention output and apply post-attention blocks for each expert
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                start = end

        return tokens_all
