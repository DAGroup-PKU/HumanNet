# Modified by Open-HumanScale contributors: removed internal development references.
"""Logical episode-segment manifests shared by training and statistics.

Large ego recordings often contain several temporally annotated substeps.  Rewriting
or duplicating their parquet/video payloads is unnecessary: a segment manifest can
declare half-open frame ranges that the loader treats as independent episodes.  The
same parser is used by the training loader, text-cache precomputation, and sampled
normalization statistics so caption and boundary semantics cannot drift.
"""

from __future__ import annotations
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1
CODEBOOK_SCHEMA_VERSION = 2
CODEBOOK_AXES = frozenset(
    {
        "visual_background",
        "background_layout_geometry",
        "background_appearance_illumination",
        "background_illumination_luminance",
        "background_palette_chroma",
        "background_contrast",
        "background_bright_area_fraction",
        "background_cool_chroma_fraction",
        "background_appearance_coarse",
        "background_appearance_fine_residual",
        "lighting_level",
        "lighting_temperature",
        "dominant_background_palette",
        "workspace_surface_type",
        "workspace_surface_color",
        "workspace_surface_pattern",
        "support_surface_base",
        "workspace_overlay",
        "fixed_fixtures",
        "coarse_layout_geometry",
        "persistent_clutter_level",
        "persistent_fixture_tokens",
        "workspace_surface_material",
        "persistent_layout_clutter",
        "semantic_place",
        "skill",
        "object",
        "viewpoint",
        "capture_domain",
    }
)
VALID_SOURCE_HAND_TAGS = {"left", "right", "both"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NAMESPACED_ID_RE = re.compile(r"^[a-z][a-z0-9_]*:[^:\s|]+:[^:\s|].*$")


def _require_nonempty_string(value: Any, field: str) -> str:
    return value.strip()


def _parse_codebook_registry(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_registry = value.get("codebooks")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_axis, raw_entry in raw_registry.items():
        axis = str(raw_axis).strip()
        entry = dict(raw_entry)
        for field in ("method", "version", "fit_population", "fit_group_field"):
            entry[field] = _require_nonempty_string(
                entry.get(field), f"codebooks.{axis}.{field}"
            )
        raw_ids = entry.get("ids", [])
        ids = [_require_nonempty_string(item, f"codebooks.{axis}.ids") for item in raw_ids]
        entry["ids"] = ids
        normalized[axis] = entry
    return normalized


def _parse_segment_codebook_axes(
    raw_axes: Any,
    registry: Mapping[str, Mapping[str, Any]],
    *,
    location: str,
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for raw_axis, raw_record in raw_axes.items():
        axis = str(raw_axis).strip()
        record = dict(raw_record)
        for field in ("method", "version"):
            record[field] = _require_nonempty_string(
                record.get(field), f"{location}.codebook_axes.{axis}.{field}"
            )
        confidence = record.get("confidence")
        confidence = float(confidence)
        record["confidence"] = confidence
        normalized[axis] = record
    return normalized


def load_episode_segment_manifest(
    manifest: str | Path | None,
    *,
    include_document: bool = False,
) -> (
    tuple[Path | None, dict[str, dict[str, Any]]]
    | tuple[Path | None, dict[str, dict[str, Any]], dict[str, Any] | None]
):
    """Load a legacy or annotation-only logical-segment manifest.

    Segment ranges are always half-open on the *materialized LeRobot timeline*:
    ``[start_frame, end_frame)``.  Extra segment fields are retained verbatim because
    object/environment studies consume metadata such as ``environment_id``,
    ``scene_desc``, and ``objects``.  ``source_hand_tag`` may explicitly override the
    physical episode tag for a logical segment (for example, a per-hand payload
    reused as a verified paired ``both``-arm sample).

    A mixed corpus may additionally list ``passthrough_roots``. Those roots use their
    physical episodes/tasks unchanged while roots in ``roots`` use logical segments.
    The normalized mapping contains both kinds and marks passthrough entries with
    ``passthrough=true`` and ``segments=None`` so every downstream consumer can enforce
    the same exact configured-root inventory.
    """

    if manifest is None or str(manifest).strip().lower() in {"", "none", "null"}:
        if include_document:
            return None, {}, None
        return None, {}

    manifest_path = Path(str(manifest)).expanduser().resolve()

    value = json.loads(manifest_path.read_text(encoding="utf-8"))

    has_v2_fields = any(
        key in value
        for key in ("codebook_schema_version", "codebooks", "codebook_annotation_provenance")
    )
    if has_v2_fields:
        codebook_registry = _parse_codebook_registry(value)
        value["codebooks"] = codebook_registry
    else:
        codebook_registry = {}

    roots = value.get("roots")

    normalized: dict[str, dict[str, Any]] = {}
    observed_segments = 0
    observed_frames = 0
    for raw_root, raw_entry in roots.items():
        raw_segments = raw_entry.get("segments")

        segments: list[dict[str, Any]] = []
        for position, raw_segment in enumerate(raw_segments):
            episode_index = int(raw_segment.get("episode_index", -1))
            start_frame = int(raw_segment.get("start_frame", -1))
            end_frame = int(raw_segment.get("end_frame", -1))
            segment_index = int(raw_segment.get("segment_index", position))
            task = str(raw_segment.get("task", "")).strip()
            raw_source_hand_tag = raw_segment.get("source_hand_tag")
            source_hand_tag = None
            if raw_source_hand_tag is not None:
                source_hand_tag = str(raw_source_hand_tag).strip().lower()
            raw_codebook_axes = raw_segment.get("codebook_axes")
            if raw_codebook_axes is not None:
                normalized_axes = _parse_segment_codebook_axes(
                    raw_codebook_axes,
                    codebook_registry,
                    location=(
                        f"roots[{raw_root!r}].segments[{position}]"
                    ),
                )
            else:
                normalized_axes = None

            segment = dict(raw_segment)
            segment.update(
                {
                    "episode_index": episode_index,
                    "segment_index": segment_index,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "task": task,
                }
            )
            if source_hand_tag is not None:
                segment["source_hand_tag"] = source_hand_tag
            if normalized_axes is not None:
                segment["codebook_axes"] = normalized_axes
            segments.append(segment)
            observed_segments += 1
            observed_frames += end_frame - start_frame
        root_frames = sum(
            int(segment["end_frame"]) - int(segment["start_frame"])
            for segment in segments
        )

        root = str(Path(str(raw_root)).expanduser().resolve())
        entry = dict(raw_entry)
        entry["segments"] = segments
        entry["total_segments"] = len(segments)
        entry["total_segment_frames"] = root_frames
        entry["passthrough"] = False
        normalized[root] = entry

    raw_passthrough_roots = value.get("passthrough_roots", [])
    seen_passthrough: set[str] = set()
    for raw_root in raw_passthrough_roots:
        root = str(Path(str(raw_root)).expanduser().resolve())
        seen_passthrough.add(root)
        normalized[root] = {
            "passthrough": True,
            "segments": None,
            "total_segments": None,
            "total_segment_frames": None,
        }

    if include_document:
        return manifest_path, normalized, value
    return manifest_path, normalized
