"""Physical robot state, RTC prefixes, and absolute action reconstruction."""

import json
from pathlib import Path

import numpy as np
import torch

from .geometry import (
    clamp_gripper_closedness,
    propagate_relative_action_validity,
    propagate_state_arm_eef_validity,
    to_global_action,
    to_relative_action,
    transform_gripper_space,
    transform_state_arm_eef_coordinate,
    zero_chassis_state,
)
from .layout import SHARED_EEF_LAYOUT

GRIPPER_SPACES = ("closedness_0_100", "gap_m", "command_openness")
ATOMIC_GROUPS = ((23, 26), (26, 29), (29, 33), (34, 37), (37, 41), (42, 45), (45, 49))


def validate_mask(value):
    mask = torch.as_tensor(value)
    if mask.dtype != torch.bool or mask.shape != (49,):
        raise ValueError("Validity masks must be boolean vectors of length 49")
    for lo, hi in ATOMIC_GROUPS:
        if mask[lo:hi].any() != mask[lo:hi].all():
            raise ValueError(f"Partially valid pose group {lo}:{hi}")
    return mask.clone()


def validate_values(values, valid):
    if not torch.isfinite(values).all():
        raise ValueError("Robot values must be finite")
    for group in (slice(29, 33), slice(37, 41), slice(45, 49)):
        q = values[..., group][valid[..., group].all(-1)]
        if q.numel() and not torch.allclose(
            q.norm(dim=-1), torch.ones_like(q[..., 0]), atol=1e-3, rtol=0
        ):
            raise ValueError("Valid orientations must be unit xyzw quaternions")


class RobotProcessor:
    def __init__(self, path):
        config = json.loads(Path(path).read_text())
        allowed = {
            "domain_id",
            "state_coordinate",
            "action_coordinate",
            "gripper_space",
            "open_aperture_m",
            "aperture_reference_width_m",
            "joint_channels",
            "state_valid_mask",
            "action_valid_mask",
            "normalization",
            "instruction_prefix",
        }
        if set(config) - allowed:
            raise ValueError("robot.json contains unsupported settings")
        self.config = config
        self.layout = SHARED_EEF_LAYOUT
        self.state_mask = validate_mask(config["state_valid_mask"])
        self.action_mask = validate_mask(config["action_valid_mask"])
        self.state_coordinate = config["state_coordinate"]
        self.action_coordinate = config["action_coordinate"]
        if self.state_coordinate not in ("base", "head_camera") or self.action_coordinate not in (
            "base",
            "head_camera",
        ):
            raise ValueError("Robot coordinates must be base or head_camera")
        self.gripper_space = config["gripper_space"]
        if self.gripper_space not in GRIPPER_SPACES:
            raise ValueError(f"gripper_space must be one of {GRIPPER_SPACES}")
        self.aperture_reference = config.get("aperture_reference_width_m")
        self.open_aperture = None
        if self.gripper_space == "command_openness":
            # The robot's own dimensionless gripper commands (for example RoboTwin
            # openness, 0 closed and 1 open). They are normalized by robot.json
            # bounds only: never converted, clamped or given a metric override.
            if "open_aperture_m" in config or self.aperture_reference is not None:
                raise ValueError("command_openness grippers take no aperture settings")
        else:
            self.open_aperture = float(config["open_aperture_m"])
            if not np.isfinite(self.open_aperture) or self.open_aperture <= 0:
                raise ValueError("open_aperture_m must be positive")
        if self.aperture_reference is not None and (
            not np.isfinite(self.aperture_reference) or self.aperture_reference <= 0
        ):
            raise ValueError("aperture_reference_width_m must be positive")
        self.stats = config["normalization"]
        if set(self.stats) != {"state", "action"}:
            raise ValueError("Provide separate state and action normalization")
        for name, stats in self.stats.items():
            if set(stats) - {"mode", "low", "high", "mean", "std", "clip"}:
                raise ValueError(f"Unsupported normalization fields for {name}")
            mode = stats["mode"]
            required = ("mean", "std") if mode == "meanstd" else ("low", "high")
            if mode not in ("meanstd", "bounds"):
                raise ValueError("Normalization must use meanstd or bounds")
            for key in required:
                values = torch.tensor(stats[key], dtype=torch.float32)
                if values.shape != (49,) or not torch.isfinite(values).all():
                    raise ValueError(f"{name}.{key} must contain 49 finite values")
                stats[key] = values
            if mode == "bounds" and (stats["high"] < stats["low"]).any():
                raise ValueError("Normalization bounds are reversed")
            if mode == "meanstd" and (stats["std"] < 0).any():
                raise ValueError("Normalization standard deviations must be nonnegative")

    def normalize(self, value, name, mask):
        value = value.masked_fill(~mask, 0)
        stats = self.stats[name]
        if stats["mode"] == "meanstd":
            scale = stats["std"]
            result = (value - stats["mean"]) / (scale + 1e-6)
        else:
            low, high = stats["low"], stats["high"]
            scale = high - low
            safe = torch.where(scale < 1e-4, torch.ones_like(scale), scale + 1e-6)
            result = (value - low) / safe * 2.0 - 1.0
            if stats.get("clip") is not None:
                result = result.clamp(-stats["clip"], stats["clip"])
        result = torch.where(scale < 1e-4, value, result)
        if self.aperture_reference is not None:
            result[..., [7, 15, 33, 41]] = (
                2 * value[..., [7, 15, 33, 41]] / self.aperture_reference - 1
            )
        return result.masked_fill(~mask, 0)

    def denormalize(self, value):
        value = value.float()
        stats = self.stats["action"]
        if stats["mode"] == "meanstd":
            scale = stats["std"]
            result = value * (scale + 1e-6) + stats["mean"]
        else:
            low, high = stats["low"], stats["high"]
            scale = high - low
            safe = torch.where(scale < 1e-4, torch.ones_like(scale), scale + 1e-6)
            result = ((value + 1.0) / 2.0) * safe + low
        result = torch.where(scale < 1e-4, value, result)
        if self.aperture_reference is not None:
            result[..., [7, 15, 33, 41]] = (
                0.5 * self.aperture_reference * (value[..., [7, 15, 33, 41]] + 1)
            )
        return result

    def grippers(self, value, *, inverse=False):
        if self.gripper_space == "command_openness":
            return value
        return transform_gripper_space(
            value,
            source_space=self.gripper_space if inverse else "closedness_0_100",
            target_space="closedness_0_100" if inverse else self.gripper_space,
            action_layout=self.layout,
            gap_max_m=self.open_aperture,
        )

    def preprocess(self, observation, prefix_limit):
        if "normalized_prefix" in observation or observation.get("action_prefix") is not None:
            raise ValueError("Physical requests use prev_chunk in absolute coordinates")
        state = torch.as_tensor(np.asarray(observation["state"]), dtype=torch.float32).clone()
        if state.shape != (49,):
            raise ValueError("state must contain 49 physical robot values")
        for name, expected in (
            ("state_valid_mask", self.state_mask),
            ("action_valid_mask", self.action_mask),
        ):
            if name in observation and not torch.equal(validate_mask(observation[name]), expected):
                raise ValueError(f"{name} disagrees with the checkpoint")
        validate_values(state, self.state_mask)
        model_state = transform_state_arm_eef_coordinate(state, self.state_coordinate, self.layout)
        if self.gripper_space == "gap_m":
            model_state = clamp_gripper_closedness(model_state, self.layout)
        model_state = self.grippers(model_state)
        state_mask = propagate_state_arm_eef_validity(
            self.state_mask, self.state_coordinate, self.layout
        )
        action_mask = propagate_relative_action_validity(
            self.action_mask.expand(48, -1).clone(),
            self.state_mask,
            self.layout,
            action_arm_eef_coordinate=self.action_coordinate,
        )
        state_mask[42:] = False
        action_mask[:, 42:] = False
        if not self.config.get("joint_channels", True):
            state_mask[:23] = False
            action_mask[:, :23] = False
        model_state = zero_chassis_state(model_state, self.layout).masked_fill(~state_mask, 0)
        result = dict(
            observation,
            normalized_state=self.normalize(model_state, "state", state_mask),
            state_valid_mask=state_mask,
            action_valid_mask=action_mask[0],
            domain_id=self.config["domain_id"],
            seed=observation.get("seed", observation.get("generation_seed", 0)),
        )
        result["instruction"] = (
            self.config.get("instruction_prefix", "") + observation["instruction"]
        )
        previous = observation.get("prev_chunk")
        if previous is not None:
            if prefix_limit == 0:
                raise ValueError("This checkpoint has RTC disabled")
            previous = torch.as_tensor(np.asarray(previous), dtype=torch.float32).clone()
            if (
                previous.ndim != 2
                or previous.shape[1] != 49
                or not 1 <= len(previous) <= 48
                or not torch.isfinite(previous).all()
            ):
                raise ValueError("prev_chunk must be a finite [N,49] array with 1 <= N <= 48")
            count = observation.get("s", 48)
            if type(count) is not int or not 0 <= count <= 48:
                raise ValueError("s must be an integer in [0,48]")
            length = min(prefix_limit, count, len(previous))
            if length:
                previous = previous[:length]
                active = action_mask[:length]
                for group in (slice(29, 33), slice(37, 41)):
                    q = previous[:, group].double()
                    norm = q.norm(dim=-1)
                    repair = active[:, group].all(-1) & (norm > 0.95) & ((norm - 1).abs() > 1e-3)
                    previous[repair, group] = (q[repair] / norm[repair, None]).float()
                validate_values(previous, active)
                relative = to_relative_action(
                    state, previous, self.layout, action_arm_eef_coordinate=self.action_coordinate
                )
                result["normalized_prefix"] = self.normalize(
                    self.grippers(relative), "action", active
                )
        return result, (state, action_mask)

    def postprocess(self, actions, context):
        anchor, valid = context
        relative = self.denormalize(actions).masked_fill(~valid, 0)
        relative = self.grippers(relative, inverse=True).masked_fill(~valid, 0)
        absolute = to_global_action(
            anchor, relative, self.layout, action_arm_eef_coordinate=self.action_coordinate
        ).masked_fill(~valid, 0)
        return {
            "relative_actions": relative.numpy(),
            "actions": absolute.numpy(),
            "action_valid_mask": valid.numpy(),
        }
