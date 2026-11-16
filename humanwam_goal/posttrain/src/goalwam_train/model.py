"""Flow-matching training of the HumanWAM-Goal transformer on packed batches."""

import math

import torch
from flash_attn_3_nv.flash_attn_interface import flash_attn_varlen_func
from goalwam.backends.native import model as native
from goalwam.backends.native.layers import RMSNorm, apply_rotary_pos_emb
from goalwam.backends.native.vae import ImageEncoder
from goalwam.backends.native.vae_layers import patchify as pixel_patchify
from goalwam.processing.positions import goalwam_position_ids
from torch import nn
from torch.utils.checkpoint import checkpoint

LOSS_WEIGHT = 10.0
VIDEO_SHIFT = 5.0


def attention(query, key, value, cu_q, cu_k, max_q, max_k, causal):
    output = flash_attn_varlen_func(query, key, value, cu_q, cu_k, max_q, max_k, causal=causal)
    return output[0] if isinstance(output, tuple) else output


class Block(native.Block):
    """A transformer block over all samples of a batch, concatenated along tokens."""

    def forward(self, hidden, cos, sin, context):
        a = self.self_attn
        x = self.input_layernorm(hidden)
        shape = (len(x), -1, a.config.head_dim)
        query = a.q_norm(a.q_proj(x).view(shape))
        key = a.k_norm(a.k_proj(x).view(shape))
        value = a.v_proj(x).view(shape)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if "text" in context:
            # Generator tokens attend to their sample's text and generator tokens.
            text_key, text_value = context["text"]
            all_key = torch.cat((text_key, key))[context["order"]]
            all_value = torch.cat((text_value, value))[context["order"]]
            output = attention(query, all_key, all_value, *context["lengths"], causal=False)
        else:
            output = attention(query, key, value, *context["lengths"], causal=True)
        hidden = hidden + a.o_proj(output.flatten(1))
        return hidden + self.mlp(self.post_attention_layernorm(hidden)), (key, value)


class Tower(nn.Module):
    def __init__(self, config, *, text=False):
        super().__init__()
        if text:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class TrainingTransformer(native.GoalWAMTransformer):
    """The inference transformer's parameters with a batched training forward."""

    def __init__(self, config):
        super().__init__(config)
        self.text = Tower(config, text=True)
        self.generator = Tower(config)

    def frequencies(self, positions):
        cos, sin = self.rotary(positions, device=positions.device, dtype=torch.bfloat16)
        return cos[0], sin[0]

    def forward(self, x):
        dtype = torch.bfloat16
        with torch.no_grad():
            hidden = self.text.embed_tokens(x["ids"])
            cos, sin = self.frequencies(x["text_positions"])
            text_kv = []
            for layer in self.text.layers:
                hidden, kv = layer(hidden, cos, sin, x["text_context"])
                text_kv.append(kv)

        video = self.vae2llm(x["visual"].to(dtype))
        video = video + x["goals"].to(dtype)[:, None] * self.goal_vision_embed
        action = self.action2llm(x["actions"].to(dtype), x["domains"], x["valid"], x["joint_rows"])
        action = action + self.action_modality_embed
        time = self.time_embedder(x["times"]).to(dtype)
        video_time, action_time = time.split((len(x["video_rows"]), len(x["action_rows"])))
        if len(x["video_rows"]):
            video = video.index_add(0, x["video_rows"], video_time)
        action = action.index_add(0, x["action_rows"], action_time)
        hidden = video.new_zeros(x["generator_tokens"], video.shape[-1])
        hidden = hidden.index_copy(0, x["video_slots"], video).index_copy(0, x["action_slots"], action)

        cos, sin = self.frequencies(x["generator_positions"])
        for layer, kv in zip(self.generator.layers, text_kv, strict=True):
            context = dict(x["generator_context"], text=kv)
            hidden, _ = checkpoint(layer, hidden, cos, sin, context, use_reentrant=False)
        hidden = self.generator.norm(hidden)

        rows = x["action_rows"]
        batch = len(x["video_count"])
        if len(x["video_rows"]):
            video_pred = self.llm2vae(hidden[x["video_slots"][x["video_rows"]]])
            video_error = (video_pred.float() - x["video_target"]).square().sum(-1)
            per_sample = video_error.new_zeros(batch).index_add(0, x["video_sample"], video_error)
            active = x["video_count"] > 0
            video_loss = (per_sample[active] / x["video_count"][active]).mean()
        else:
            video_loss = 0.0 * sum(p.float().sum() for p in self.llm2vae.parameters())
        action = self.llm2action(
            hidden[x["action_slots"][rows]], x["domains"][rows], x["valid"][rows], x["output_joint_rows"]
        )
        valid = x["valid"][rows]
        action_error = ((action.float() - x["action_target"]).square() * valid).sum(-1)
        action_loss = action_error.new_zeros(batch).index_add(0, x["action_sample"], action_error)
        count = action_error.new_zeros(batch).index_add(0, x["action_sample"], valid.sum(-1).float())
        action_loss = (action_loss / count.clamp(min=1)).mean()
        loss = LOSS_WEIGHT * video_loss + LOSS_WEIGHT * action_loss
        return loss, video_loss.detach(), action_loss.detach()


class VideoEncoder(ImageEncoder):
    """Wan 2.2 causal video encoding: the first frame alone, then blocks of four frames."""

    @torch.no_grad()
    def encode(self, pixels):
        pixels = pixel_patchify(pixels.to(device=self.mean.device, dtype=torch.bfloat16), 2)
        cache = [None] * self.cache_size
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False):
            chunks = [pixels[:, :, :1], *(pixels[:, :, 1:].split(4, dim=2) if pixels.shape[2] > 1 else ())]
            features = torch.cat([self.encoder(chunk, feat_cache=cache) for chunk in chunks], dim=2)
            mean = self.conv1(features).chunk(2, dim=1)[0]
        return ((mean - self.mean) * self.inv_std).float()


def sample_sigma(count, generator):
    """Noise levels for the generated frames and actions: Waver schedule, shifted."""
    u = torch.rand(count, generator=generator)
    t = 1.0 - u - 1.29 * (torch.cos(math.pi / 2 * u) ** 2 - 1 + u)
    return VIDEO_SHIFT * t / (1 + (VIDEO_SHIFT - 1) * t)


def pixels(canvases):
    return canvases.float() * (1.0 / 127.5) - 1.0


def cumulative(lengths, device):
    return torch.tensor([0, *lengths], device=device).cumsum(0).to(torch.int32)


def _encode_rollouts(samples, vae, device, encode_batch):
    """Batch-encode rollout clips grouped by frame count (`1, 5, 9, 13`)."""
    encoded = [None] * len(samples)
    by_length = {}
    for idx, s in enumerate(samples):
        by_length.setdefault(len(s["video"]), []).append(idx)
    for indices in by_length.values():
        for start in range(0, len(indices), encode_batch):
            group = indices[start : start + encode_batch]
            clips = pixels(torch.stack([samples[i]["video"] for i in group]).to(device)).transpose(1, 2)
            latents = vae.encode(clips)
            for pos, sample_idx in enumerate(group):
                encoded[sample_idx] = latents[pos : pos + 1]
    return encoded


def prepare(samples, vae, config, generator, cuda_generator, encode_batch=4):
    """Encode, noise and pack one micro-batch."""
    device = vae.mean.device
    patch = config.latent_patch_size
    patch_dim = config.latent_channel * patch * patch
    sigma = sample_sigma(len(samples), generator).to(device)
    goals = vae.encode(pixels(torch.stack([s["goal"] for s in samples]).to(device))[:, :, None])
    history = [frame for s in samples for frame in s["history"]]
    history = vae.encode(pixels(torch.stack(history).to(device))[:, :, None]) if history else None
    rollouts = _encode_rollouts(samples, vae, device, encode_batch)
    out = {
        key: []
        for key in (
            "ids",
            "text_positions",
            "visual",
            "goals",
            "video_rows",
            "video_target",
            "video_sample",
            "video_count",
            "actions",
            "valid",
            "domains",
            "action_rows",
            "action_target",
            "action_sample",
            "generator_positions",
            "video_slots",
            "action_slots",
            "order",
        )
    }
    text_lengths, generator_lengths = [], []
    history_offset = visual_offset = text_offset = generator_offset = 0
    total_text = sum(len(s["ids"]) for s in samples)
    for i, s in enumerate(samples):
        level = sigma[i]
        clip = rollouts[i]
        if clip.shape[2] > 1:
            noise = torch.randn(clip[:, :, 1:].shape, device=device, generator=cuda_generator)
            noisy = torch.cat((clip[:, :, :1], (1 - level) * clip[:, :, 1:] + level * noise), dim=2)
            target_tokens = native.patchify(noise - clip[:, :, 1:], patch)[0]
        else:
            noisy = clip[:, :, :1]
            target_tokens = clip.new_empty((0, patch_dim))
        count = len(s["history"])
        previous = history[history_offset : history_offset + count].split(1) if count else ()
        items = [goals[i : i + 1], *previous, noisy]
        history_offset += count
        tokens = [native.patchify(item, patch)[0] for item in items]
        sizes = [len(t) for t in tokens]
        visual = sum(sizes)
        future = len(target_tokens)
        out["visual"].append(torch.cat(tokens))
        out["goals"].append(torch.arange(visual, device=device) < sizes[0])
        out["video_rows"].append(torch.arange(visual - future, visual, device=device) + visual_offset)
        out["video_target"].append(target_tokens)
        out["video_sample"].append(torch.full((future,), i, device=device, dtype=torch.long))
        out["video_count"].append(float(clip.numel()) if future else 0.0)

        actions, valid = s["action"].to(device), s["valid"].to(device)
        rows = len(actions)
        action_noise = torch.randn(actions.shape, device=device, generator=cuda_generator)
        noisy_actions = torch.cat((actions[:1], (1 - level) * actions[1:] + level * action_noise[1:]))
        out["actions"].append(noisy_actions.masked_fill(~valid, 0))
        out["valid"].append(valid)
        out["domains"].append(torch.full((rows,), s["domain"], device=device))
        out["action_rows"].append(torch.arange(1, rows, device=device) + i * rows)
        out["action_target"].append((action_noise - actions)[1:].masked_fill(~valid[1:], 0))
        out["action_sample"].append(torch.full((rows - 1,), i, device=device, dtype=torch.long))

        text = len(s["ids"])
        out["ids"].append(s["ids"].to(device))
        out["text_positions"].append(torch.arange(text, device=device))
        grids = tuple(
            (item.shape[2], math.ceil(item.shape[3] / patch), math.ceil(item.shape[4] / patch))
            for item in items
        )
        positions = goalwam_position_ids(
            text_tokens=text,
            grids=grids,
            history_ages_seconds=s["history_ages"],
            action_rows=rows,
            video_fps=s["video_fps"],
            action_fps=s["action_fps"],
        )
        out["generator_positions"].append(positions[:, 0].to(device))
        length = visual + rows
        out["video_slots"].append(torch.arange(visual, device=device) + generator_offset)
        out["action_slots"].append(torch.arange(visual, length, device=device) + generator_offset)
        out["order"].append(torch.arange(text, device=device) + text_offset)
        out["order"].append(torch.arange(length, device=device) + total_text + generator_offset)
        text_lengths.append(text)
        generator_lengths.append(length)
        visual_offset += visual
        text_offset += text
        generator_offset += length

    x = {
        key: torch.cat(value, dim=-1 if key == "generator_positions" else 0)
        for key, value in out.items()
        if key != "video_count"
    }
    x["video_count"] = torch.tensor(out["video_count"], device=device)
    x["text_positions"] = x["text_positions"][None]
    x["generator_positions"] = x["generator_positions"][:, None]
    x["times"] = (torch.cat((sigma[x["video_sample"]], sigma[x["action_sample"]])) * 1000).float()
    x["times"] = x["times"] * config.timestep_scale
    joint = x["valid"][:, :23].any(-1)
    x["joint_rows"] = joint.nonzero().flatten()
    x["output_joint_rows"] = joint[x["action_rows"]].nonzero().flatten()
    x["generator_tokens"] = generator_offset
    x["text_context"] = {
        "lengths": (
            cumulative(text_lengths, device),
            cumulative(text_lengths, device),
            max(text_lengths),
            max(text_lengths),
        ),
    }
    both = [a + b for a, b in zip(text_lengths, generator_lengths, strict=True)]
    x["generator_context"] = {
        "order": x.pop("order"),
        "lengths": (
            cumulative(generator_lengths, device),
            cumulative(both, device),
            max(generator_lengths),
            max(both),
        ),
    }
    return x
