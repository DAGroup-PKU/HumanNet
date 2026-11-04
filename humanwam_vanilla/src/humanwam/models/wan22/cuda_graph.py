# Modified by Open-HumanScale contributors: removed internal development references.
"""CUDA-graph acceleration for FastWAM action denoising (inference only).

Why this exists
---------------
In ``FastWAM.infer_action`` the per-step action forward
(``_predict_action_noise_with_cache``) is **kernel-launch-bound**: the action
tokens are tiny (~tens), so the GPU starves while the CPU launches hundreds of
small kernels per denoise step. Replaying the step as a CUDA graph reduces
CPU launch overhead while reusing the same tensor operations.

Graph reuse (important for FastWAM eval)
----------------------------------------
Eval uses only ~10 denoise steps, so re-capturing every chunk (warmup + capture ~=
3 forwards) is NOT amortized and is a net loss. Instead we capture ONCE and reuse
across chunks: the prefilled video/cross-attn KV caches, context and masks are
copied into the graph's static buffers at the start of each chunk (cheap tensor
``copy_``), then only ``latents_action``/``timestep_action`` vary per step. The
graph references fixed buffer addresses, so it stays valid across chunks as long as
the shapes match (same task -> same ``video_seq_len``/``action_horizon``); we key the
cached graph on those and rebuild only when they change. All copies are exact, so
replay is bit-identical to eager.
"""
import torch


def _clone_static(obj):
    """Deep-clone the tensor leaves of a (possibly nested list/dict/tuple) structure
    into stable-address buffers; pass non-tensors through unchanged."""
    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _clone_static(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cloned = [_clone_static(v) for v in obj]
        return type(obj)(cloned) if isinstance(obj, tuple) else cloned
    return obj


def _copy_into(dst, src):
    """Copy ``src`` tensor leaves into the matching ``dst`` static buffers in place."""
    if torch.is_tensor(dst) and torch.is_tensor(src):
        dst.copy_(src)
    elif isinstance(dst, dict):
        for k in dst:
            _copy_into(dst[k], src[k])
    elif isinstance(dst, (list, tuple)):
        for d, s in zip(dst, src):
            _copy_into(d, s)


class ActionDenoiseStepGraph:
    """Capture one ``_predict_action_noise_with_cache`` forward; replay per step,
    reusing the graph across chunks via ``update_static`` (copies new caches in)."""

    def __init__(self, model, latents_action, timestep_action, static_kwargs, warmup=2):
        self.model = model
        self.static_latents = latents_action.clone()
        self.static_timestep = timestep_action.clone()
        # Stable-address copies of every (nested) tensor the forward reads.
        self.static_kwargs = {k: _clone_static(v) for k, v in static_kwargs.items()}

        def _fwd():
            return self.model._predict_action_noise_with_cache(
                latents_action=self.static_latents,
                timestep_action=self.static_timestep,
                **self.static_kwargs,
            )

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(max(1, warmup)):
                _fwd()
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()  # private pool -> no shared-pool assert
        # 'thread_local' scopes capture-safety to this thread (a sim/render worker on
        # another thread can't invalidate the capture). Single-thread callers unaffected.
        with torch.no_grad(), torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            self.out = _fwd()

    def update_static(self, static_kwargs):
        """Copy this chunk's freshly-prefilled caches/context/masks into the static
        buffers (called once per chunk before the denoise loop)."""
        for k, v in static_kwargs.items():
            _copy_into(self.static_kwargs[k], v)

    def replay(self, latents_action, timestep_action):
        """Copy the two per-step varying inputs into the static buffers, replay, and
        return the (static) output. Clone it before the next replay if you keep it."""
        self.static_latents.copy_(latents_action)
        self.static_timestep.copy_(timestep_action)
        self.graph.replay()
        return self.out


def pin_action_rope_freqs_to_gpu(model, device):
    """Move ``action_expert.freqs`` onto ``device`` once.

    ``ActionDiT.freqs`` is a plain (non-buffer) complex attribute, so ``model.to()``
    does not move it; ``pre_dit`` then does ``self.freqs[:n].to(tokens.device)`` every
    step (a per-step H2D copy that is illegal during CUDA-graph capture). Pinning it is
    a pure win (no numeric change) and a prerequisite for capture. Idempotent.
    """
    ae = getattr(model, "action_expert", None)
    if ae is None:
        return
    f = getattr(ae, "freqs", None)
    if torch.is_tensor(f) and f.device.type != torch.device(device).type:
        ae.freqs = f.to(device)


def pin_video_rope_freqs_to_gpu(model, device):
    """Move ``video_expert.freqs`` onto ``device`` once.

    ``WanVideoDiT.freqs`` is a 3-tuple of plain (non-buffer) tensors from
    ``precompute_freqs_cis_3d``, so ``model.to()`` does not move them; ``pre_dit`` then
    does ``torch.cat([...]).to(x_tokens.device)`` every chunk (a per-chunk H2D copy that
    is illegal during CUDA-graph capture). Pinning makes that ``.to`` a no-op — pure win,
    no numeric change, and a prerequisite for capturing the video prefill. Idempotent.
    """
    ve = getattr(model, "video_expert", None)
    if ve is None:
        return
    f = getattr(ve, "freqs", None)
    dev = torch.device(device)
    if isinstance(f, (list, tuple)) and all(torch.is_tensor(t) for t in f):
        if any(t.device != dev for t in f):
            ve.freqs = type(f)(t.to(device) for t in f)


class VideoPrefillGraph:
    """Capture the once-per-chunk video prefill as a CUDA graph; replay per chunk with the
    new first-frame latents (and context) copied into static buffers.

    The prefill (``video_expert.pre_dit`` -> ``mot.prefill_video_cache`` -> action
    ``text_embedding`` -> ``mot.prefill_cross_attn_cache``) runs ONCE per chunk and is
    partly kernel-launch-bound (30 layers x ~20 ops over a few hundred video tokens). Its
    inputs that vary chunk-to-chunk are only ``first_frame_latents`` (from the VAE encode
    of the new observation) and — within an episode — the constant ``context``; the video
    attention mask, timestep (zeros) and grid shapes are fixed for an eval config. So we
    capture once, keyed on shape, and replay: copy the new latents/context into the static
    input buffers, ``replay()``, and read the prefilled caches from the static OUTPUT
    buffers (the action path consumes them). Bit-exact: replay reruns the identical
    forward. Capture requires the video RoPE freqs pinned to GPU (see above).
    """

    def __init__(self, model, prefill_inputs, warmup=2):
        self.model = model
        # Stable-address copies of every (nested) tensor the prefill reads. Non-tensors
        # (the bool fuse flag) pass through and are baked as constants.
        self.static = {k: _clone_static(v) for k, v in prefill_inputs.items()}

        def _fwd():
            s = self.static
            video_pre = self.model.video_expert.pre_dit(
                x=s["first_frame_latents"],
                timestep=s["timestep_video"],
                context=s["context"],
                context_mask=s["context_mask"],
                action=None,
                fuse_vae_embedding_in_latents=s["fuse_flag"],
            )
            video_kv_cache = self.model.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=s["video_attention_mask"],
            )
            context_embedding_cache = self.model.action_expert.text_embedding(s["context"])
            cross_attn_kv_cache = self.model.mot.prefill_cross_attn_cache("action", context_embedding_cache)
            return video_kv_cache, context_embedding_cache, cross_attn_kv_cache

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(max(1, warmup)):
                _fwd()
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            self.out = _fwd()  # (video_kv_cache, context_embedding_cache, cross_attn_kv_cache)

    def replay(self, first_frame_latents, context=None, context_mask=None):
        """Copy this chunk's varying inputs into the static buffers, replay, and return the
        (static) prefilled caches. Consume/copy them before the next replay."""
        self.static["first_frame_latents"].copy_(first_frame_latents)
        if context is not None:
            self.static["context"].copy_(context)
        if context_mask is not None:
            self.static["context_mask"].copy_(context_mask)
        self.graph.replay()
        return self.out
