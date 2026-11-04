# Modified by Open-HumanScale contributors: removed internal development references.
"""Deterministic clip-locked robot-video augmentation.

The 5-task RoboTwin pipeline reads already assembled ``[-1, 1]`` float clips from
the merged cache.  These wrappers deliberately run after that cache lookup so one
sample-level RNG plan can be shared across time and, where requested, cameras.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import torch
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


# Pre-merged T-shape canvas 384x352: head 256x352 on top, two 128x176 wrists below.
_ROBOTWIN_T352_TILES = (
    ("cam_high", slice(0, 256), slice(0, 352)),
    ("cam_left_wrist", slice(256, 384), slice(0, 176)),
    ("cam_right_wrist", slice(256, 384), slice(176, 352)),
)
_ROBOTWIN_TILES = (
    ("cam_high", slice(0, 256), slice(0, 320)),
    ("cam_left_wrist", slice(256, 384), slice(0, 160)),
    ("cam_right_wrist", slice(256, 384), slice(160, 320)),
)
_RA_PHOTO_OPS = (
    "identity",
    "solarize",
    "posterize",
    "color",
    "contrast",
    "brightness",
    "sharpness",
)


def _mix_seed(base_seed: int, epoch: int, index: int) -> int:
    """Stable 64-bit mixing independent of worker/rank/global RNG state."""

    mask = (1 << 64) - 1
    x = int(base_seed) & mask
    for value in (epoch, index):
        x = (x + 0x9E3779B97F4A7C15 + (int(value) & mask)) & mask
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
        x ^= x >> 31
    return x


def _float_video_to_uint8(video: torch.Tensor) -> torch.Tensor:
    return ((video.float().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)


def _uint8_video_to_float(video: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    return video.to(torch.float32).div(127.5).sub(1.0).to(dtype=dtype)


def _replace_tile(canvas: torch.Tensor, rows: slice, cols: slice, tile: torch.Tensor) -> None:
    canvas[..., rows, cols] = tile


@dataclass(frozen=True)
class _OpenVLAPlan:
    top: int
    left: int
    crop_h: int
    crop_w: int
    brightness_delta: float
    contrast_factor: float
    saturation_factor: float
    hue_delta: float


@dataclass(frozen=True)
class _RAPhotoOp:
    name: str
    magnitude: float | int | None


# Independent RNG streams per augmentation family. Each family mixes its own salt
# into the base seed so enabling a family never changes the RA-Photo draws of the
# same (seed, epoch, index); augE stays byte-identical when the families are off.
_LIGHT_FIELD_SALT = 0x4C49474854  # "LIGHT"
_GEOMETRIC_SALT = 0x5348494654  # "SHIFT"
_INTERNW0_SALT = 0x494E54455257  # "INTERW": InternW0-Delta RoboTwin recipe

_LIGHT_FIELD_DEFAULTS: dict[str, Any] = {
    "probability": 0.5,
    "tint_strength": 0.15,
    "exposure_ev": 0.3,
    "gamma_range": (0.7, 1.4),
    "gradient_amplitude": 0.3,
    "blob_count": (1, 3),
    "blob_amplitude": 0.4,
    "blob_sigma": (0.15, 0.5),
    "field_range": (0.6, 1.4),
    "shadow_probability": 0.3,
    "extreme_probability": 0.03,
}
# InternW0-Delta augmentation recipe: crop (cam_high only), RGB gain, soft shadow,
# and colour jitter. Colour jitter has no gate; the other three are gated independently.
_INTERNW0_DEFAULTS: dict[str, Any] = {
    "probability": 1.0,          # outer gate kept for the family API; the recipe itself uses 1.0
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.25,
    "gamma": 0.1,
    "rgb_prob": 0.7,
    "rgb_gain_range": (0.8, 1.2),
    "shadow_prob": 0.7,
    "shadow_depth_range": (0.15, 0.35),
    "shadow_sigma_range": (0.25, 0.6),
    "crop_prob": 0.5,
    "crop_scale_range": (0.95, 1.0),
    "crop_head_only": True,
}
_GEOMETRIC_DEFAULTS: dict[str, Any] = {
    "probability": 0.5,
    "translate_x": (-6, 6),
    "translate_y": (-6, 10),
    "scale_range": (1.0, 1.0),
}


def _family_params(name: str, raw: Any, defaults: dict[str, Any]) -> dict[str, Any] | None:
    if raw is None or raw is False:
        return None
    if "enabled" in raw and not bool(raw["enabled"]):
        return None
    params = dict(defaults)
    params.update({k: v for k, v in raw.items() if k != "enabled"})
    for key, value in params.items():
        if isinstance(value, (list, tuple)):
            params[key] = tuple(value)
    return params


@dataclass(frozen=True)
class _LightFieldPlan:
    gain: tuple[float, float, float]
    exposure: float
    gamma: float
    gradient: tuple[float, float]  # (gx, gy) amplitude along normalized x/y in [-1,1]
    blobs: tuple[tuple[float, float, float, float], ...]  # (cx, cy, sigma, amplitude), normalized coords
    shadow: tuple[float, float, float, float, float] | None  # (px, py, theta, width, strength)
    field_range: tuple[float, float]
    extreme: str | None


@dataclass(frozen=True)
class _InternW0Plan:
    crop: tuple[float, float, float] | None  # (scale, offset_x, offset_y) in [0,1]
    rgb: tuple[float, float, float] | None  # mean-normalised per-channel gains
    shadows: tuple[tuple[float, float, float, float], ...] | None  # per view (x, y, sigma, depth)
    color: tuple[float, float, float, float]  # brightness, contrast, saturation, gamma factors


@dataclass(frozen=True)
class _GeometricPlan:
    dx: int
    dy: int
    scale: float


class ClipLockedVideoAugmentation(Dataset):
    """Wrap a robot dataset without changing its length or non-video fields."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        name: str,
        probability: float = 1.0,
        seed: int = 42,
        sync_time: bool = True,
        sync_cameras: bool | None = None,
        layout: str = "robotwin_tshape_384x320",
        magnitude_index: int = 10,
        num_ops: int = 2,
        light_field: dict[str, Any] | None = None,
        geometric_jitter: dict[str, Any] | None = None,
        internw0: dict[str, Any] | None = None,
    ) -> None:
        self.light_field = _family_params("light_field", light_field, _LIGHT_FIELD_DEFAULTS)
        self.geometric_jitter = _family_params("geometric_jitter", geometric_jitter, _GEOMETRIC_DEFAULTS)
        self.internw0 = _family_params("internw0", internw0, _INTERNW0_DEFAULTS)
        if self.internw0 is not None:
            lo, hi = (float(v) for v in self.internw0["crop_scale_range"])
        if self.geometric_jitter is not None:
            lo, hi = self.geometric_jitter["scale_range"]
            for key in ("translate_x", "translate_y"):
                a, b = self.geometric_jitter[key]
        if self.light_field is not None:
            lo, hi = self.light_field["field_range"]

        self.dataset = dataset
        self.name = str(name)
        self.probability = float(probability)
        self.seed = int(seed)
        self.layout = str(layout)
        self.magnitude_index = int(magnitude_index)
        self.num_ops = int(num_ops)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        nested = getattr(self.dataset, "set_epoch", None)
        if callable(nested):
            nested(self.epoch)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    @staticmethod
    def _sample_openvla_plan(rng: random.Random, height: int, width: int) -> _OpenVLAPlan:
        edge_scale = math.sqrt(0.9)
        crop_h = min(height, max(1, int(round(height * edge_scale))))
        crop_w = min(width, max(1, int(round(width * edge_scale))))
        return _OpenVLAPlan(
            top=rng.randint(0, height - crop_h),
            left=rng.randint(0, width - crop_w),
            crop_h=crop_h,
            crop_w=crop_w,
            brightness_delta=rng.uniform(-0.2, 0.2),
            contrast_factor=rng.uniform(0.8, 1.2),
            saturation_factor=rng.uniform(0.8, 1.2),
            hue_delta=rng.uniform(-0.05, 0.05),
        )

    @staticmethod
    def _apply_openvla_tile(tile_cthw: torch.Tensor, plan: _OpenVLAPlan) -> torch.Tensor:
        # torchvision image operators use [..., C, H, W], so time is a leading batch.
        tile = tile_cthw.permute(1, 0, 2, 3).float().add(1.0).mul(0.5).clamp_(0.0, 1.0)
        tile = F.resized_crop(
            tile,
            top=plan.top,
            left=plan.left,
            height=plan.crop_h,
            width=plan.crop_w,
            size=list(tile.shape[-2:]),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        # Match dlimp/TensorFlow's additive random_brightness, clipping after every op.
        tile = tile.add(plan.brightness_delta).clamp_(0.0, 1.0)
        tile = F.adjust_contrast(tile, plan.contrast_factor).clamp_(0.0, 1.0)
        tile = F.adjust_saturation(tile, plan.saturation_factor).clamp_(0.0, 1.0)
        tile = F.adjust_hue(tile, plan.hue_delta).clamp_(0.0, 1.0)
        return tile.mul(2.0).sub(1.0).permute(1, 0, 2, 3)

    def _apply_openvla(self, video: torch.Tensor, rng: random.Random) -> torch.Tensor:
        output = video.clone()
        shared_photo_plan = None
        for camera, rows, cols in _ROBOTWIN_TILES:
            tile = video[..., rows, cols]
            plan = self._sample_openvla_plan(rng, tile.shape[-2], tile.shape[-1])
            if self.name == "openvla_primary_crop" and camera != "cam_high":
                if shared_photo_plan is None:
                    shared_photo_plan = plan
                plan = _OpenVLAPlan(
                    top=0,
                    left=0,
                    crop_h=tile.shape[-2],
                    crop_w=tile.shape[-1],
                    brightness_delta=shared_photo_plan.brightness_delta,
                    contrast_factor=shared_photo_plan.contrast_factor,
                    saturation_factor=shared_photo_plan.saturation_factor,
                    hue_delta=shared_photo_plan.hue_delta,
                )
            augmented = self._apply_openvla_tile(tile, plan).to(dtype=video.dtype)
            _replace_tile(output, rows, cols, augmented)
        return output

    # ------------------------------------------------------------------ light field
    def _sample_light_field_plan(self, rng: random.Random) -> _LightFieldPlan:
        p = self.light_field
        g_lo, g_hi = (float(v) for v in p["gamma_range"])
        gamma = math.exp(rng.uniform(math.log(g_lo), math.log(g_hi)))
        extreme: str | None = None
        if rng.random() < float(p["extreme_probability"]):
            extreme = ("colored", "dark", "bright")[rng.randrange(3)]
        if extreme == "colored":
            dominant = rng.randrange(3)
            gain = tuple(1.0 if c == dominant else rng.uniform(0.3, 0.6) for c in range(3))
            exposure = rng.uniform(0.7, 1.2)
        elif extreme == "dark":
            gain = (1.0, 1.0, 1.0)
            exposure = rng.uniform(0.25, 0.45)
        elif extreme == "bright":
            gain = (1.0, 1.0, 1.0)
            exposure = rng.uniform(1.6, 2.2)
        else:
            t = rng.uniform(-1.0, 1.0) * float(p["tint_strength"])
            gain = (1.0 + t, 1.0, 1.0 - t)
            ev = float(p["exposure_ev"])
            exposure = 2.0 ** rng.uniform(-ev, ev)
        theta = rng.uniform(0.0, 2.0 * math.pi)
        amp = rng.uniform(0.0, float(p["gradient_amplitude"]))
        gradient = (amp * math.cos(theta), amp * math.sin(theta))
        n_lo, n_hi = (int(v) for v in p["blob_count"])
        blobs = []
        for _ in range(rng.randint(n_lo, n_hi)):
            s_lo, s_hi = (float(v) for v in p["blob_sigma"])
            blobs.append((
                rng.uniform(-1.2, 1.2),
                rng.uniform(-1.2, 1.2),
                rng.uniform(s_lo, s_hi),
                rng.uniform(-float(p["blob_amplitude"]), float(p["blob_amplitude"])),
            ))
        shadow = None
        if rng.random() < float(p["shadow_probability"]):
            shadow = (
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(0.0, math.pi),
                rng.uniform(0.05, 0.25),
                rng.uniform(0.1, 0.4),
            )
        lo, hi = (float(v) for v in p["field_range"])
        return _LightFieldPlan(
            gain=tuple(float(v) for v in gain),
            exposure=float(exposure),
            gamma=float(gamma),
            gradient=gradient,
            blobs=tuple(blobs),
            shadow=shadow,
            field_range=(lo, hi),
            extreme=extreme,
        )

    @staticmethod
    def _build_light_field(plan: _LightFieldPlan, height: int, width: int) -> torch.Tensor:
        """Low-frequency multiplicative illumination map in [field_range], shape [H, W]."""

        # Coarse grid (1/8 resolution) + bilinear upsample keeps the field smooth and cheap.
        gh, gw = max(2, height // 8), max(2, width // 8)
        ys = torch.linspace(-1.0, 1.0, gh, dtype=torch.float32).view(gh, 1).expand(gh, gw)
        xs = torch.linspace(-1.0, 1.0, gw, dtype=torch.float32).view(1, gw).expand(gh, gw)
        aspect = width / height
        xs_iso = xs * aspect if aspect >= 1.0 else xs
        ys_iso = ys / aspect if aspect >= 1.0 else ys
        field = 1.0 + plan.gradient[0] * xs + plan.gradient[1] * ys
        for cx, cy, sigma, amplitude in plan.blobs:
            cx_iso = cx * aspect if aspect >= 1.0 else cx
            cy_iso = cy / aspect if aspect >= 1.0 else cy
            d2 = (xs_iso - cx_iso) ** 2 + (ys_iso - cy_iso) ** 2
            field = field + amplitude * torch.exp(-d2 / (2.0 * sigma * sigma))
        if plan.shadow is not None:
            px, py, theta, width_n, strength = plan.shadow
            d = (xs_iso - px) * (-math.sin(theta)) + (ys_iso - py) * math.cos(theta)
            field = field * (1.0 - strength * torch.exp(-(d / width_n) ** 2))
        lo, hi = plan.field_range
        field = field.clamp(lo, hi)
        field = torch.nn.functional.interpolate(
            field.view(1, 1, gh, gw), size=(height, width), mode="bilinear", align_corners=False
        )
        return field.view(height, width)

    @classmethod
    def _apply_light_field_tile(cls, tile_cthw: torch.Tensor, plan: _LightFieldPlan) -> torch.Tensor:
        x = tile_cthw.float().add(1.0).mul(0.5).clamp_(0.0, 1.0)
        x = x.pow(plan.gamma)
        gain = x.new_tensor(plan.gain).mul(plan.exposure).view(3, 1, 1, 1)
        field = cls._build_light_field(plan, x.shape[-2], x.shape[-1]).to(x.device)
        x = x.mul(gain).mul(field.view(1, 1, *field.shape)).clamp_(0.0, 1.0)
        return x.mul(2.0).sub(1.0).to(dtype=tile_cthw.dtype)

    def _for_each_tile(self, video: torch.Tensor, fn) -> torch.Tensor:
        output = video.clone()
        if self.layout == "robotwin_three_views_256x352":
            for camera in range(3):
                output[camera] = fn(video[camera])
            return output
        tiles = (("cam_high", slice(None), slice(None)),) if self.layout == "robotwin_head_256x352" else (_ROBOTWIN_T352_TILES if self.layout == "robotwin_tshape_384x352" else _ROBOTWIN_TILES)
        for _, rows, cols in tiles:
            _replace_tile(output, rows, cols, fn(video[..., rows, cols]))
        return output

    def _apply_light_field(self, video: torch.Tensor, rng: random.Random) -> torch.Tensor:
        # One global (gain/exposure/gamma) draw per clip; the spatial field is drawn per
        # camera tile because the cameras look at different parts of the scene.
        base = self._sample_light_field_plan(rng)
        plans: list[_LightFieldPlan] = [base]
        n_tiles = 3 if self.layout != "robotwin_head_256x352" else 1
        for _ in range(n_tiles - 1):
            extra = self._sample_light_field_plan(rng)
            plans.append(_LightFieldPlan(
                gain=base.gain, exposure=base.exposure, gamma=base.gamma,
                gradient=extra.gradient, blobs=extra.blobs, shadow=extra.shadow,
                field_range=base.field_range, extreme=base.extreme,
            ))
        it = iter(plans)
        return self._for_each_tile(video, lambda tile: self._apply_light_field_tile(tile, next(it)))

    # ------------------------------------------------------------------ geometric jitter
    def _sample_geometric_plan(self, rng: random.Random) -> _GeometricPlan:
        p = self.geometric_jitter
        x_lo, x_hi = (int(v) for v in p["translate_x"])
        y_lo, y_hi = (int(v) for v in p["translate_y"])
        s_lo, s_hi = (float(v) for v in p["scale_range"])
        return _GeometricPlan(dx=rng.randint(x_lo, x_hi), dy=rng.randint(y_lo, y_hi), scale=rng.uniform(s_lo, s_hi))

    @staticmethod
    def _apply_geometric_tile(tile_cthw: torch.Tensor, plan: _GeometricPlan) -> torch.Tensor:
        height, width = tile_cthw.shape[-2:]
        margin = max(abs(plan.dx), abs(plan.dy)) + int(math.ceil(abs(plan.scale - 1.0) * max(height, width) / 2.0)) + 2
        frames = tile_cthw.permute(1, 0, 2, 3).float()
        padded = torch.nn.functional.pad(frames, (margin, margin, margin, margin), mode="replicate")
        moved = F.affine(
            padded,
            angle=0.0,
            translate=[int(plan.dx), int(plan.dy)],
            scale=float(plan.scale),
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
        )
        cropped = F.center_crop(moved, [height, width])
        return cropped.clamp_(-1.0, 1.0).permute(1, 0, 2, 3).to(dtype=tile_cthw.dtype)

    def _apply_geometric(self, video: torch.Tensor, rng: random.Random) -> torch.Tensor:
        plan = self._sample_geometric_plan(rng)  # same shift for every camera tile and frame
        return self._for_each_tile(video, lambda tile: self._apply_geometric_tile(tile, plan))

    # ------------------------------------------------------------------ InternW0-Delta recipe
    def _sample_internw0_plan(self, rng: random.Random, n_views: int) -> _InternW0Plan:
        p = self.internw0
        crop = None
        if rng.random() < float(p["crop_prob"]):
            s_lo, s_hi = (float(v) for v in p["crop_scale_range"])
            crop = (rng.uniform(s_lo, s_hi), rng.random(), rng.random())
        rgb = None
        if rng.random() < float(p["rgb_prob"]):
            g_lo, g_hi = (float(v) for v in p["rgb_gain_range"])
            gains = [rng.uniform(g_lo, g_hi) for _ in range(3)]
            mean = sum(gains) / 3.0
            rgb = tuple(g / mean for g in gains)
        shadows = None
        if rng.random() < float(p["shadow_prob"]):
            sg_lo, sg_hi = (float(v) for v in p["shadow_sigma_range"])
            d_lo, d_hi = (float(v) for v in p["shadow_depth_range"])
            shadows = tuple((rng.random(), rng.random(), rng.uniform(sg_lo, sg_hi), rng.uniform(d_lo, d_hi)) for _ in range(n_views))

        def factor(amount: float) -> float:
            amount = float(amount)
            return 1.0 if amount <= 0.0 else rng.uniform(max(0.0, 1.0 - amount), 1.0 + amount)

        color = (factor(p["brightness"]), factor(p["contrast"]), factor(p["saturation"]), factor(p["gamma"]))
        return _InternW0Plan(crop=crop, rgb=rgb, shadows=shadows, color=color)

    @staticmethod
    def _internw0_crop(tile_cthw01: torch.Tensor, crop: tuple[float, float, float]) -> torch.Tensor:
        # InternW0 RobotVideoDataset._apply_camera_jitter_to_video, random_crop branch.
        height, width = tile_cthw01.shape[-2:]
        scale, offset_x, offset_y = crop
        crop_h = max(1, min(height, int(round(height * scale))))
        crop_w = max(1, min(width, int(round(width * scale))))
        top = int(round(offset_y * max(0, height - crop_h)))
        left = int(round(offset_x * max(0, width - crop_w)))
        if crop_h == height and crop_w == width and top == 0 and left == 0:
            return tile_cthw01
        frames = tile_cthw01.permute(1, 0, 2, 3)
        frames = F.resized_crop(frames, top=top, left=left, height=crop_h, width=crop_w, size=[height, width],
                                interpolation=InterpolationMode.BILINEAR, antialias=True)
        return frames.permute(1, 0, 2, 3)

    @staticmethod
    def _internw0_shadow(tile_cthw01: torch.Tensor, shadow: tuple[float, float, float, float]) -> torch.Tensor:
        # InternW0 domain_aug_dataset.apply_domain: normalised [0,1] coordinates on both axes.
        height, width = tile_cthw01.shape[-2:]
        x, y, sigma, depth = shadow
        ys = torch.linspace(0, 1, height, dtype=tile_cthw01.dtype).view(height, 1)
        xs = torch.linspace(0, 1, width, dtype=tile_cthw01.dtype).view(1, width)
        distance = (xs - x).square() + (ys - y).square()
        field = 1.0 - depth * torch.exp(-distance / (2.0 * sigma * sigma))
        return tile_cthw01 * field.view(1, 1, height, width)

    @staticmethod
    def _internw0_color(tile_cthw01: torch.Tensor, color: tuple[float, float, float, float]) -> torch.Tensor:
        # InternW0 RobotVideoDataset._apply_color_jitter_to_video: B -> C (per-frame mean) -> S (luma) -> gamma.
        brightness, contrast, saturation, gamma = color
        flat = tile_cthw01.permute(1, 0, 2, 3).clone()  # [T,3,H,W]
        if brightness != 1.0:
            flat.mul_(brightness)
        if contrast != 1.0:
            mean = flat.mean(dim=(-3, -2, -1), keepdim=True)
            flat.sub_(mean).mul_(contrast).add_(mean)
        if saturation != 1.0:
            weights = flat.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
            gray = (flat * weights).sum(dim=1, keepdim=True)
            flat.sub_(gray).mul_(saturation).add_(gray)
        if gamma != 1.0:
            flat.clamp_(0.0, 1.0).pow_(gamma)
        flat.clamp_(0.0, 1.0)
        return flat.permute(1, 0, 2, 3)

    def _apply_internw0(self, video: torch.Tensor, rng: random.Random) -> torch.Tensor:
        n_views = 1 if self.layout == "robotwin_head_256x352" else 3
        plan = self._sample_internw0_plan(rng, n_views)
        head_only = bool(self.internw0["crop_head_only"])
        counter = iter(range(n_views))

        def fn(tile: torch.Tensor) -> torch.Tensor:
            view = next(counter)
            x = tile.float().add(1.0).mul(0.5).clamp_(0.0, 1.0)
            if plan.crop is not None and (view == 0 or not head_only):
                x = self._internw0_crop(x, plan.crop)
            if plan.rgb is not None:
                x = x * x.new_tensor(plan.rgb).view(3, 1, 1, 1)
            if plan.shadows is not None:
                x = self._internw0_shadow(x, plan.shadows[view])
            x = x.clamp_(0.0, 1.0)
            x = self._internw0_color(x, plan.color)
            return x.mul(2.0).sub(1.0).to(dtype=tile.dtype)

        return self._for_each_tile(video, fn)

    def _apply_families(self, video: torch.Tensor, index: int) -> torch.Tensor | None:
        """Apply the optional light/geometric families with their own RNG streams.

        Order: geometric -> light field -> InternW0 recipe -> (caller) RA-Photo.
        Returns None when no family fires so the caller can keep the exact legacy
        path (and the identity object) for augE-only configurations.
        """

        output: torch.Tensor | None = None
        if self.geometric_jitter is not None:
            rng = random.Random(_mix_seed(self.seed ^ _GEOMETRIC_SALT, self.epoch, int(index)))
            if rng.random() < float(self.geometric_jitter["probability"]):
                output = self._apply_geometric(video, rng)
        if self.light_field is not None:
            rng = random.Random(_mix_seed(self.seed ^ _LIGHT_FIELD_SALT, self.epoch, int(index)))
            if rng.random() < float(self.light_field["probability"]):
                output = self._apply_light_field(video if output is None else output, rng)
        if self.internw0 is not None:
            rng = random.Random(_mix_seed(self.seed ^ _INTERNW0_SALT, self.epoch, int(index)))
            if rng.random() < float(self.internw0["probability"]):
                output = self._apply_internw0(video if output is None else output, rng)
        return output

    def _sample_ra_photo_ops(self, rng: random.Random) -> tuple[_RAPhotoOp, ...]:
        index = self.magnitude_index
        delta = 0.9 * index / 30.0
        posterize_bits = 8 - int(round(4.0 * index / 30.0))
        solarize_threshold = 255.0 * (1.0 - index / 30.0)
        ops = []
        for _ in range(self.num_ops):
            name = _RA_PHOTO_OPS[rng.randrange(len(_RA_PHOTO_OPS))]
            magnitude: float | int | None = None
            if name == "solarize":
                magnitude = solarize_threshold
            elif name == "posterize":
                magnitude = posterize_bits
            elif name in {"color", "contrast", "brightness", "sharpness"}:
                signed_delta = delta if rng.random() >= 0.5 else -delta
                magnitude = 1.0 + signed_delta
            ops.append(_RAPhotoOp(name=name, magnitude=magnitude))
        return tuple(ops)

    @staticmethod
    def _apply_ra_photo_op(tile_tchw: torch.Tensor, op: _RAPhotoOp) -> torch.Tensor:
        if op.name == "identity":
            return tile_tchw
        if op.name == "solarize":
            return F.solarize(tile_tchw, float(op.magnitude))
        if op.name == "posterize":
            return F.posterize(tile_tchw, int(op.magnitude))
        if op.name == "color":
            return F.adjust_saturation(tile_tchw, float(op.magnitude))
        if op.name == "contrast":
            return F.adjust_contrast(tile_tchw, float(op.magnitude))
        if op.name == "brightness":
            return F.adjust_brightness(tile_tchw, float(op.magnitude))
        if op.name == "sharpness":
            return F.adjust_sharpness(tile_tchw, float(op.magnitude))

    def _apply_ra_photo(self, video: torch.Tensor, rng: random.Random) -> torch.Tensor:
        output = _float_video_to_uint8(video)
        ops = self._sample_ra_photo_ops(rng)
        if self.layout == "robotwin_three_views_256x352":
            # Draw one plan for all cameras and frames. Process cameras separately
            # so contrast statistics/sharpness kernels cannot cross view borders.
            for camera in range(3):
                tile = output[camera].permute(1, 0, 2, 3)
                for op in ops:
                    tile = self._apply_ra_photo_op(tile, op)
                output[camera] = tile.permute(1, 0, 2, 3)
            return _uint8_video_to_float(output, dtype=video.dtype)
        tiles = (("cam_high", slice(None), slice(None)),) if self.layout == "robotwin_head_256x352" else (_ROBOTWIN_T352_TILES if self.layout == "robotwin_tshape_384x352" else _ROBOTWIN_TILES)
        for _, rows, cols in tiles:
            tile = output[..., rows, cols].permute(1, 0, 2, 3)
            for op in ops:
                tile = self._apply_ra_photo_op(tile, op)
            _replace_tile(output, rows, cols, tile.permute(1, 0, 2, 3))
        return _uint8_video_to_float(output, dtype=video.dtype)

    @staticmethod
    def _apply_openwam_color(video: torch.Tensor, factors: tuple[float, float, float]) -> torch.Tensor:
        # Preserve PIL ImageEnhance semantics and B -> C -> S order, including
        # uint8 rounding between operators. Quantize assembled RGB once, locally;
        # the original float clip/cache is never modified.
        frames = _float_video_to_uint8(video).permute(1, 2, 3, 0).cpu().numpy()
        output = []
        for frame in frames:
            image = Image.fromarray(frame)
            for operation, factor in zip(
                (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color), factors
            ):
                image = operation(image).enhance(factor)
            output.append(torch.from_numpy(np.array(image, copy=True)))
        result = torch.stack(output).permute(3, 0, 1, 2).to(video.device)
        return _uint8_video_to_float(result, dtype=video.dtype)

    def _apply_head_photo(self, video: torch.Tensor, rng: random.Random) -> torch.Tensor:
        if self.name in {"openwam_color", "openwam_lighting"}:
            factors = tuple(rng.uniform(0.8, 1.2) for _ in range(3))
            output = self._apply_openwam_color(video, factors)
            if self.name == "openwam_lighting":
                gamma = rng.uniform(0.8, 1.2)
                gain = video.new_tensor([rng.uniform(0.85, 1.15) for _ in range(3)]).view(3, 1, 1, 1)
                output = (gain * ((output.float() + 1) * 0.5).clamp(0, 1).pow(gamma)).clamp(0, 1)
                output = (output * 2 - 1).to(video.dtype)
            return output
        # Photo-only: deliberately no resized_crop, even for the identity crop.
        tile = video.permute(1, 0, 2, 3).float().add(1).mul(0.5).clamp(0, 1)
        tile = tile.add(rng.uniform(-0.2, 0.2)).clamp(0, 1)
        tile = F.adjust_contrast(tile, rng.uniform(0.8, 1.2)).clamp(0, 1)
        tile = F.adjust_saturation(tile, rng.uniform(0.8, 1.2)).clamp(0, 1)
        tile = F.adjust_hue(tile, rng.uniform(-0.05, 0.05)).clamp(0, 1)
        return tile.mul(2).sub(1).permute(1, 0, 2, 3).to(video.dtype)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        video = sample.get("video")

        # Optional families (geometric -> light) draw from their own RNG streams and
        # run before the photometric RA ops, so the legacy RA draws below are untouched.
        pre = self._apply_families(video, index)

        rng = random.Random(_mix_seed(self.seed, self.epoch, int(index)))
        if rng.random() >= self.probability:
            if pre is None:
                return sample
            augmented = pre
        elif self.layout == "robotwin_head_256x352" and self.name != "ra_photo":
            augmented = self._apply_head_photo(video, rng)
        elif self.name.startswith("openvla_"):
            augmented = self._apply_openvla(video, rng)
        else:
            augmented = self._apply_ra_photo(video if pre is None else pre, rng)

        result = dict(sample)
        result["video"] = augmented.clamp(-1.0, 1.0).contiguous()
        return result
