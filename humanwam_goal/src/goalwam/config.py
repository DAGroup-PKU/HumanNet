"""Inference-only checkpoint configuration."""

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path


def read_config(path, cls):
    values = json.loads(Path(path).read_text())
    if not isinstance(values, dict):
        raise ValueError(f"Expected a configuration object: {path}")
    known = {field.name for field in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unsupported inference settings in {path}: {sorted(unknown)}")
    return cls(**values)


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 12288
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    mrope_section: tuple[int, ...] = (24, 20, 20)
    max_position_embeddings: int = 262144
    action_gen: bool = True
    action_dim: int = 64
    joint_dim: int = 23
    shared_dim: int = 19
    horizon: int = 48
    num_embodiment_domains: int = 32
    latent_channel: int = 48
    latent_patch_size: int = 2
    timestep_scale: float = 0.001

    def __post_init__(self):
        if (self.action_dim, self.joint_dim, self.shared_dim, self.horizon) != (64, 23, 19, 48):
            raise ValueError("This release supports the 48-step Shared-EEF GoalWAM model")
        if not self.action_gen or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Invalid action or attention configuration")
        if sum(self.mrope_section) != self.head_dim // 2:
            raise ValueError("Rotary sections must cover half the head width")


@dataclass(frozen=True)
class SamplingConfig:
    num_steps: int = 10
    flow_shift: float = 2.0
    prefix_length: int = 0
    corrector_weights: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self):
        if self.num_steps != 10 or not math.isfinite(self.flow_shift) or self.flow_shift <= 0:
            raise ValueError("Use ten steps and a positive flow shift")
        if type(self.prefix_length) is not int or not 0 <= self.prefix_length < 48:
            raise ValueError("RTC prefix length must be in [0,48)")
        if self.corrector_weights is not None:
            if len(self.corrector_weights) != 10 or any(
                len(row) != min(index, 2) or not all(math.isfinite(x) for x in row)
                for index, row in enumerate(self.corrector_weights)
            ):
                raise ValueError("Invalid UniPC corrector coefficients")
