# Modified by Open-HumanScale contributors: removed internal development references.
"""Pure temporal-resampling helpers for native-speed LeRobot episodes.

The on-disk episode remains the source of truth.  A temporal-resampling spec defines
an optional *virtual* episode timeline used by the loader:

``identity``
    The virtual and native timelines are identical.
``endpoint``
    ``M = round(N * factor)`` and virtual frame ``v`` maps to
    ``u = v * (N - 1) / (M - 1)``.  This matches the endpoint-preserving slow-motion
    used by EgoDex/EgoVerse conversion.
``node_locked``
    ``M = round(N * factor)`` and ``u = min(v / factor, N - 1)``.  Native nodes stay
    locked at multiples of ``factor`` and the final node is held at the tail.  This
    matches the existing 4x vector interpolation and tail convention; video
    still follows the nearest-copy policy described next.

Video uses nearest-node sampling with round-half-up semantics
(``floor(u + 0.5)``), never Python/NumPy bankers rounding.  Continuous EEF/camera
vectors use linear interpolation except for real-first (wxyz) quaternion blocks,
which use normalized shortest-path SLERP.

This module deliberately has no dataset-loader state.  Both lazy and non-lazy loader
paths, as well as normalization-statistics code, can share these functions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch


TEMPORAL_RESAMPLING_SPEC_VERSION = 1
TEMPORAL_RESAMPLING_PROVENANCE_VERSION = 1

IDENTITY = "identity"
ENDPOINT = "endpoint"
NODE_LOCKED = "node_locked"
TemporalMapping = Literal["identity", "endpoint", "node_locked"]
SUPPORTED_TEMPORAL_MAPPINGS = frozenset({IDENTITY, ENDPOINT, NODE_LOCKED})


@dataclass(frozen=True)
class TemporalResamplingSpec:
    """Validated temporal-resampling configuration.

    Config dictionaries use exactly the keys ``version``, ``mapping``, and
    ``factor``.  ``version`` may be omitted and defaults to the current version.
    ``identity`` may omit ``factor`` (canonical value 1.0); slowdown mappings must
    explicitly provide a finite factor greater than one.
    """

    version: int
    mapping: TemporalMapping
    factor: float

    @property
    def is_identity(self) -> bool:
        return self.mapping == IDENTITY


@dataclass(frozen=True)
class TemporalIndexMap:
    """Mapping from one or more virtual episode-local indices to native rows."""

    native_length: int
    virtual_length: int
    virtual_indices: np.ndarray
    native_coordinates: np.ndarray
    lower_indices: np.ndarray
    upper_indices: np.ndarray
    alpha: np.ndarray
    nearest_indices: np.ndarray


@dataclass(frozen=True)
class ContinuousFeatureLayout:
    """Last-dimension layout for a temporally interpolated vector feature."""

    dimension: int
    quaternion_slices: tuple[tuple[int, int], ...]


EEF16_QUATERNION_SLICES = ((3, 7), (11, 15))
CAMERA7_QUATERNION_SLICES = ((3, 7),)

# Only these physical streams are interpolated.  Discrete metadata such as task and
# episode indices are copied by the loader; timestamps/frame indices are regenerated
# on the virtual 30-fps timeline; video is selected through ``nearest_indices``.
CONTINUOUS_FEATURE_LAYOUTS: Mapping[str, ContinuousFeatureLayout] = MappingProxyType(
    {
        "observation.state": ContinuousFeatureLayout(16, EEF16_QUATERNION_SLICES),
        "action": ContinuousFeatureLayout(16, EEF16_QUATERNION_SLICES),
        "observation.cam_extrinsic": ContinuousFeatureLayout(
            7, CAMERA7_QUATERNION_SLICES
        ),
    }
)


SpecLike = TemporalResamplingSpec | Mapping[str, Any] | None
ArrayLike = np.ndarray | torch.Tensor


def _parse_integer(value: Any, *, name: str) -> int:
    result = int(value)
    return result


def parse_temporal_resampling_spec(spec: SpecLike) -> TemporalResamplingSpec:
    """Parse a temporal-resampling spec.

    ``None`` is the explicit no-op default and canonicalizes to identity.
    """

    if spec is None:
        return TemporalResamplingSpec(
            version=TEMPORAL_RESAMPLING_SPEC_VERSION,
            mapping=IDENTITY,
            factor=1.0,
        )

    if isinstance(spec, TemporalResamplingSpec):
        raw: Mapping[str, Any] = {
            "version": spec.version,
            "mapping": spec.mapping,
            "factor": spec.factor,
        }
    elif isinstance(spec, Mapping):
        raw = spec

    version = raw.get("version", TEMPORAL_RESAMPLING_SPEC_VERSION)
    version = int(version)

    mapping = raw.get("mapping")

    if mapping == IDENTITY:
        factor_value = raw.get("factor", 1.0)
    else:
        factor_value = raw["factor"]
    factor = float(factor_value)

    return TemporalResamplingSpec(version=version, mapping=mapping, factor=factor)


def canonicalize_temporal_resampling_spec(spec: SpecLike) -> dict[str, Any]:
    """Return the stable JSON-safe representation used by provenance hashing."""

    value = parse_temporal_resampling_spec(spec)
    return {
        "version": value.version,
        "mapping": value.mapping,
        "factor": value.factor,
    }


def temporal_resampling_spec_digest(spec: SpecLike) -> str:
    """SHA256 of the canonical spec JSON (sorted keys, compact separators)."""

    encoded = json.dumps(
        canonicalize_temporal_resampling_spec(spec),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_identity_temporal_resampling(spec: SpecLike) -> bool:
    """Return whether ``spec`` leaves the native timeline unchanged."""

    return parse_temporal_resampling_spec(spec).is_identity


def virtual_episode_length(native_length: int, spec: SpecLike) -> int:
    """Return virtual length ``M`` for a native episode length ``N``.

    Empty episodes stay empty.  For both slowdown mappings and ``N > 0``, this is
    exactly ``int(round(N * factor))``; no hidden minimum or endpoint correction is
    applied.  A singleton therefore expands to repeated samples when rounding gives
    ``M > 1``.
    """

    native_length = _parse_integer(native_length, name="native_length")
    value = parse_temporal_resampling_spec(spec)
    return _virtual_episode_length_validated(native_length, value)


def _virtual_episode_length_validated(
    native_length: int,
    spec: TemporalResamplingSpec,
) -> int:
    if native_length == 0 or spec.is_identity:
        return native_length
    return int(round(native_length * spec.factor))


def _normalize_virtual_indices(
    virtual_indices: Any,
    *,
    virtual_length: int,
) -> np.ndarray:
    values = np.asarray(virtual_indices)
    values = values.astype(np.int64, copy=False)
    return values


def build_temporal_index_map(
    native_length: int,
    spec: SpecLike,
    virtual_indices: Any | None = None,
) -> TemporalIndexMap:
    """Build interpolation and nearest-video indices for an episode-local timeline.

    When ``virtual_indices`` is omitted, the complete virtual episode is mapped.
    Supplied indices may have any shape; every returned array has that same shape.
    """

    native_length = _parse_integer(native_length, name="native_length")
    value = parse_temporal_resampling_spec(spec)
    virtual_length = virtual_episode_length(native_length, value)
    if virtual_indices is None:
        indices = np.arange(virtual_length, dtype=np.int64)
    else:
        indices = _normalize_virtual_indices(
            virtual_indices, virtual_length=virtual_length
        )

    if indices.size == 0:
        empty_float = np.empty(indices.shape, dtype=np.float64)
        empty_int = np.empty(indices.shape, dtype=np.int64)
        return TemporalIndexMap(
            native_length=native_length,
            virtual_length=virtual_length,
            virtual_indices=indices,
            native_coordinates=empty_float,
            lower_indices=empty_int,
            upper_indices=empty_int.copy(),
            alpha=empty_float.copy(),
            nearest_indices=empty_int.copy(),
        )

    # Non-empty virtual indices imply a non-empty native episode by construction.
    if native_length == 1:
        coordinates = np.zeros(indices.shape, dtype=np.float64)
    elif value.mapping == IDENTITY:
        coordinates = indices.astype(np.float64)
    elif value.mapping == ENDPOINT:
        if virtual_length <= 1:
            coordinates = np.zeros(indices.shape, dtype=np.float64)
        else:
            coordinates = (
                indices.astype(np.float64)
                * float(native_length - 1)
                / float(virtual_length - 1)
            )
    elif value.mapping == NODE_LOCKED:
        coordinates = np.minimum(
            indices.astype(np.float64) / value.factor,
            float(native_length - 1),
        )

    coordinates = np.clip(coordinates, 0.0, float(native_length - 1))
    lower = np.floor(coordinates).astype(np.int64)
    upper = np.minimum(lower + 1, native_length - 1)
    alpha = coordinates - lower.astype(np.float64)
    # Exact half steps select the later physical frame; np.rint/round are forbidden.
    nearest = np.floor(coordinates + 0.5).astype(np.int64)
    nearest = np.clip(nearest, 0, native_length - 1)

    return TemporalIndexMap(
        native_length=native_length,
        virtual_length=virtual_length,
        virtual_indices=indices,
        native_coordinates=coordinates,
        lower_indices=lower,
        upper_indices=upper,
        alpha=alpha,
        nearest_indices=nearest,
    )


def continuous_feature_layout(feature_key: str) -> ContinuousFeatureLayout | None:
    """Return interpolation layout, or ``None`` for a discrete/non-vector feature."""

    return CONTINUOUS_FEATURE_LAYOUTS.get(feature_key)


def is_temporally_continuous_feature(feature_key: str) -> bool:
    """Whether ``feature_key`` must use lerp/SLERP on the virtual timeline."""

    return feature_key in CONTINUOUS_FEATURE_LAYOUTS


def _prepare_interpolation_inputs(
    lower: ArrayLike,
    upper: ArrayLike,
    alpha: Any,
    *,
    expected_dimension: int,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:

    prefix = tuple(lower.shape[:-1])
    if isinstance(lower, torch.Tensor):
        alpha_value = torch.as_tensor(alpha, dtype=lower.dtype, device=lower.device)
        alpha_value = torch.broadcast_to(alpha_value, prefix)
    else:
        alpha_value = np.asarray(alpha, dtype=lower.dtype)
        alpha_value = np.broadcast_to(alpha_value, prefix)
    return lower, upper, alpha_value


def _slerp_wxyz_numpy(
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    q0 = lower.astype(np.float64, copy=False)
    q1 = upper.astype(np.float64, copy=False)
    n0 = np.linalg.norm(q0, axis=-1, keepdims=True)
    n1 = np.linalg.norm(q1, axis=-1, keepdims=True)
    q0n = q0 / n0
    q1n = q1 / n1
    dot = np.sum(q0n * q1n, axis=-1, keepdims=True)
    q1n = np.where(dot < 0.0, -q1n, q1n)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    a = alpha.astype(np.float64, copy=False)[..., None]
    near = np.abs(sin_theta) < 1e-6
    safe_sin = np.where(near, 1.0, sin_theta)
    spherical = (
        np.sin((1.0 - a) * theta) / safe_sin * q0n
        + np.sin(a * theta) / safe_sin * q1n
    )
    linear = (1.0 - a) * q0n + a * q1n
    result = np.where(near, linear, spherical)
    result /= np.clip(np.linalg.norm(result, axis=-1, keepdims=True), 1e-12, None)
    # Preserve exact native rows at node coordinates (alpha=0), including their sign.
    result = np.where(a == 0.0, q0, result)
    return result.astype(lower.dtype, copy=False)


def _slerp_wxyz_torch(
    lower: torch.Tensor,
    upper: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    work_dtype = torch.float64 if lower.dtype == torch.float64 else torch.float32
    q0 = lower.to(dtype=work_dtype)
    q1 = upper.to(dtype=work_dtype)
    n0 = torch.linalg.vector_norm(q0, dim=-1, keepdim=True)
    n1 = torch.linalg.vector_norm(q1, dim=-1, keepdim=True)
    q0n = q0 / n0
    q1n = q1 / n1
    dot = torch.sum(q0n * q1n, dim=-1, keepdim=True)
    q1n = torch.where(dot < 0.0, -q1n, q1n)
    dot = torch.clamp(torch.abs(dot), 0.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    a = alpha.to(dtype=work_dtype).unsqueeze(-1)
    near = torch.abs(sin_theta) < 1e-6
    safe_sin = torch.where(near, torch.ones_like(sin_theta), sin_theta)
    spherical = (
        torch.sin((1.0 - a) * theta) / safe_sin * q0n
        + torch.sin(a * theta) / safe_sin * q1n
    )
    linear = (1.0 - a) * q0n + a * q1n
    result = torch.where(near, linear, spherical)
    result = result / torch.clamp(
        torch.linalg.vector_norm(result, dim=-1, keepdim=True), min=1e-12
    )
    result = torch.where(a == 0.0, q0, result)
    return result.to(dtype=lower.dtype)


def interpolate_continuous_vectors(
    lower: ArrayLike,
    upper: ArrayLike,
    alpha: Any,
    *,
    quaternion_slices: Sequence[tuple[int, int]] = (),
) -> ArrayLike:
    """Lerp vector pairs and shortest-path-SLERP selected wxyz blocks.

    ``lower``/``upper`` have shape ``(..., D)`` and ``alpha`` broadcasts to ``...``.
    All non-quaternion columns are linearly interpolated.  This generic helper is
    useful for tests or future continuous vector layouts; production ego fields
    should normally go through :func:`interpolate_continuous_feature`.
    """
    dimension = int(lower.shape[-1]) if lower.ndim else 0
    normalized_slices: list[tuple[int, int]] = []
    occupied: set[int] = set()
    for item in quaternion_slices:
        start, end = item
        start, end = int(start), int(end)
        columns = set(range(start, end))
        occupied |= columns
        normalized_slices.append((start, end))

    lower, upper, alpha_value = _prepare_interpolation_inputs(
        lower, upper, alpha, expected_dimension=dimension
    )
    if isinstance(lower, torch.Tensor):
        a = alpha_value.unsqueeze(-1)
        result = lower + (upper - lower) * a
        for start, end in normalized_slices:
            result[..., start:end] = _slerp_wxyz_torch(
                lower[..., start:end], upper[..., start:end], alpha_value
            )
        result = torch.where(a == 0.0, lower, result)
    else:
        a = alpha_value[..., None]
        result = lower + (upper - lower) * a
        for start, end in normalized_slices:
            result[..., start:end] = _slerp_wxyz_numpy(
                lower[..., start:end], upper[..., start:end], alpha_value
            )
        result = np.where(a == 0.0, lower, result).astype(lower.dtype, copy=False)
    return result


def interpolate_continuous_feature(
    feature_key: str,
    lower: ArrayLike,
    upper: ArrayLike,
    alpha: Any,
) -> ArrayLike:
    """Interpolate one registered EEF16/camera7 continuous feature."""

    layout = continuous_feature_layout(feature_key)
    return interpolate_continuous_vectors(
        lower,
        upper,
        alpha,
        quaternion_slices=layout.quaternion_slices,
    )


def _update_length_sequence_digest(digest: Any, length: int) -> None:
    digest.update(str(length).encode("ascii"))
    digest.update(b"\n")


def temporal_resampling_provenance(
    spec: SpecLike,
    *,
    native_episode_lengths: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Build JSON-safe algorithm/spec provenance and optional corpus counts.

    Length-sequence hashes are order-sensitive and stream over the iterable, so this
    remains bounded-memory for million-episode corpora.
    """

    value = parse_temporal_resampling_spec(spec)
    result: dict[str, Any] = {
        "version": TEMPORAL_RESAMPLING_PROVENANCE_VERSION,
        "spec": canonicalize_temporal_resampling_spec(value),
        "spec_sha256": temporal_resampling_spec_digest(value),
        "video_sampling": "nearest_floor_u_plus_0p5",
        "continuous_interpolation": {
            "linear": "lerp",
            "quaternion": "shortest_path_slerp",
            "quaternion_order": "wxyz",
            "features": {
                key: {
                    "dimension": layout.dimension,
                    "quaternion_slices": [list(item) for item in layout.quaternion_slices],
                }
                for key, layout in CONTINUOUS_FEATURE_LAYOUTS.items()
            },
        },
    }
    if native_episode_lengths is None:
        return result

    native_lengths_digest = hashlib.sha256()
    virtual_lengths_digest = hashlib.sha256()
    episodes = 0
    native_frames = 0
    virtual_frames = 0
    for index, raw_length in enumerate(native_episode_lengths):
        length = _parse_integer(
            raw_length, name=f"native_episode_lengths[{index}]"
        )
        virtual = _virtual_episode_length_validated(length, value)
        _update_length_sequence_digest(native_lengths_digest, length)
        _update_length_sequence_digest(virtual_lengths_digest, virtual)
        episodes += 1
        native_frames += length
        virtual_frames += virtual
    result["corpus"] = {
        "episodes": episodes,
        "native_frames": native_frames,
        "virtual_frames": virtual_frames,
        "native_episode_lengths_sha256": native_lengths_digest.hexdigest(),
        "virtual_episode_lengths_sha256": virtual_lengths_digest.hexdigest(),
    }
    return result
