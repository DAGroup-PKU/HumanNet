"""Native single-GPU GoalWAM inference; no accelerated-engine dependency."""

import math
from pathlib import Path

import numpy as np
import torch

from goalwam.config import ModelConfig, SamplingConfig, read_config
from goalwam.processing.positions import goalwam_position_ids

from .model import GoalWAMTransformer, patchify, remap_weight, unpatchify
from .sampler import UniPCPlan
from .vae import ImageEncoder
from .weights import load_weights


class NativeBackend:
    def __init__(self, checkpoint, *, tp=1, cuda_graph=True, device="cuda:0"):
        if tp != 1:
            raise ValueError("The native backend uses one GPU; select phyai for TP=2/4")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("GoalWAM requires a CUDA GPU")
        if torch.cuda.get_device_capability(self.device) != (9, 0):
            raise ValueError("This release is validated on Hopper (SM90) GPUs")
        self.checkpoint = Path(checkpoint)
        self.config = read_config(self.checkpoint / "transformer/config.json", ModelConfig)
        self.sampling = read_config(self.checkpoint / "sampling.json", SamplingConfig)
        with torch.device("meta"):
            self.model = GoalWAMTransformer(self.config)
        load_weights(self.model, self.checkpoint / "transformer", remap_weight)
        self.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
        self.model.time_embedder.float()
        # The full text tower resides on CPU between instruction changes.
        for name, child in self.model.named_children():
            if name != "text":
                child.to(self.device)
        self.model.goal_vision_embed.data = self.model.goal_vision_embed.data.to(self.device)
        self.model.action_modality_embed.data = self.model.action_modality_embed.data.to(
            self.device
        )
        self.vae = ImageEncoder(self.checkpoint / "vae", self.device)
        self.solver = UniPCPlan(self.sampling, self.device)
        self.text_kv = None
        self.cuda_graph = cuda_graph
        self.graph = None
        self.graph_key = None
        self.graph_input = None
        self.graph_output = None
        self.graph_conditioning = None

    @torch.inference_mode()
    def set_instruction(self, ids, mask):
        del mask
        self.graph = self.graph_output = self.graph_input = self.graph_key = None
        self.graph_conditioning = None
        self.model.text.to(self.device)
        self.text_kv = self.model.prefill(ids.to(self.device))
        self.model.text.cpu()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def predict(self, observation):
        if self.text_kv is None:
            raise RuntimeError("Set an instruction before inference")
        c = self.config
        frames = [self.vae(frame).float() for frame in observation["pixels"]]
        future = c.horizon * observation["video_fps"] / observation["action_fps"] / 4
        if not future.is_integer() or future < 1:
            raise ValueError("Action and video rates must align with temporal compression")
        frames[-1] = torch.cat(
            (
                frames[-1],
                frames[-1].new_zeros((1, c.latent_channel, int(future), *frames[-1].shape[-2:])),
            ),
            dim=2,
        )
        action = observation["action"].to(device=self.device, dtype=torch.bfloat16)
        valid = observation["action_valid_mask"].to(self.device)
        action = action.masked_fill(~valid, 0)
        grids = tuple(
            (
                x.shape[2],
                math.ceil(x.shape[3] / c.latent_patch_size),
                math.ceil(x.shape[4] / c.latent_patch_size),
            )
            for x in frames
        )
        positions = goalwam_position_ids(
            text_tokens=self.text_kv[0][0].shape[1],
            grids=grids,
            history_ages_seconds=observation["history_ages_seconds"],
            action_rows=c.horizon + 1,
            video_fps=observation["video_fps"],
            action_fps=observation["action_fps"],
        )
        frequencies = self.model.rotary(positions, device=self.device, dtype=torch.bfloat16)
        masks = [torch.zeros_like(x, dtype=torch.bool) for x in frames]
        masks[-1][:, :, 1:] = True
        masks.append(observation["action_noisy_mask"].to(self.device).unsqueeze(-1) & valid)
        clean = (*frames, action)
        initial = []
        for value, mask in zip(clean, masks, strict=True):
            noise = torch.from_numpy(
                np.random.RandomState(observation["seed"])
                .standard_normal(tuple(value.shape[1:]))
                .astype(np.float32)
            ).to(value)
            numeric_mask = mask.to(value.dtype)
            initial.append(((1.0 - numeric_mask) * value + numeric_mask * noise).flatten())
        initial = torch.cat(initial).float()
        noisy = torch.cat([mask.flatten() for mask in masks])
        counts = [math.prod(grid) for grid in grids]
        video_noisy = torch.zeros(sum(counts), dtype=torch.bool, device=self.device)
        video_noisy[-int(future) * grids[-1][1] * grids[-1][2] :] = True
        action_rows = observation["action_noisy_mask"][0].nonzero().flatten().to(self.device)
        joint_active = observation["action_valid_mask"][0, :, :23].any(-1)
        domains = observation["action_domains"]
        if (
            (domains[joint_active] < 0) | (domains[joint_active] >= c.num_embodiment_domains)
        ).any():
            raise ValueError("Active joint channels require a valid robot domain ID")
        layout = {
            "goals": torch.arange(sum(counts), device=self.device) < counts[0],
            "video_rows": video_noisy.nonzero().flatten(),
            "action_rows": action_rows,
            "valid": valid[0],
            "domains": domains.to(self.device),
            "joint_rows": joint_active.nonzero().flatten().to(self.device),
            "output_joint_rows": joint_active[observation["action_noisy_mask"][0]]
            .nonzero()
            .flatten()
            .to(self.device),
        }
        shapes = [value.shape for value in clean]
        sizes = [value.numel() for value in clean]
        times = {
            t: torch.tensor(t, device=self.device, dtype=torch.float32)
            for t in self.solver.timesteps
        }

        def velocity(latent, timestep):
            values = [
                value.reshape(shape)
                for value, shape in zip(latent.split(sizes), shapes, strict=True)
            ]
            patches = torch.cat(
                [patchify(value, c.latent_patch_size) for value in values[:-1]], dim=1
            )[0]
            video, actions = self.model(
                patches,
                values[-1][0],
                times[timestep],
                text_kv=self.text_kv,
                frequencies=frequencies,
                layout=layout,
            )
            full_video = video.new_zeros(patches.shape).index_copy(0, layout["video_rows"], video)
            full_action = actions.new_zeros(values[-1].shape[1:]).index_copy(
                0, action_rows, actions
            )
            visual = [
                unpatchify(value[None], shape, c.latent_patch_size)
                for value, shape in zip(full_video.split(counts), shapes[:-1], strict=True)
            ]
            return torch.cat([value.flatten() for value in (*visual, full_action)]) * noisy

        # Shapes and masks define a graph; pixels, actions, noise, and rotary
        # positions are refreshed for every request through owned buffers.
        if self.cuda_graph:
            signature = (
                tuple(shapes),
                observation["action_noisy_mask"].contiguous().numpy().tobytes(),
                observation["action_valid_mask"].contiguous().numpy().tobytes(),
                domains.contiguous().numpy().tobytes(),
            )
            if signature != self.graph_key:
                self.graph = self.graph_output = self.graph_input = None
                self.graph_conditioning = None
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self.solver.run(initial, velocity)
                torch.cuda.current_stream(self.device).wait_stream(stream)
                self.graph_input = initial.clone()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.graph_output = self.solver.run(self.graph_input, velocity)
                self.graph, self.graph_key = graph, signature
                self.graph_conditioning = (frequencies, layout, noisy, times)
            self.graph_input.copy_(initial)
            for target, value in zip(self.graph_conditioning[0], frequencies, strict=True):
                target.copy_(value)
            self.graph.replay()
            final = self.graph_output.clone()
        else:
            final = self.solver.run(initial, velocity)
        return final[-sizes[-1] :].reshape(shapes[-1])[:, 1:, :49].float().cpu()

    def close(self):
        self.graph = self.graph_output = self.graph_input = None
        self.graph_conditioning = None
        self.text_kv = self.model = self.vae = self.solver = None
