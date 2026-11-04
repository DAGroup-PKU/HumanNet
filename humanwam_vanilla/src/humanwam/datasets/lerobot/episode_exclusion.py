"""Shared episode-exclusion manifest helpers.

The joint ego corpus is split across hundreds of self-contained LeRobot roots.  A
curation pass therefore records bad episodes as a non-destructive, per-root denylist
instead of rewriting the source trees.  Both training and standalone statistics must
interpret that denylist identically; keeping the parser here avoids subtle corpus
drift between those two paths.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Optional


MANIFEST_VERSION = 1


def load_episode_exclusion_manifest(
    manifest: Optional[str | Path],
) -> tuple[Optional[Path], dict[str, dict[str, Any]]]:
    """Load and normalize a version-1 per-root episode denylist.

    The returned mapping is keyed by resolved absolute root paths.  Unknown fields are
    intentionally tolerated so a producer can attach provenance without requiring a
    synchronized loader update; the filtering contract itself remains the two fields
    normalized below.
    """

    if manifest is None or str(manifest).strip().lower() in {"", "none", "null"}:
        return None, {}

    manifest_path = Path(str(manifest)).expanduser().resolve()

    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    roots = value.get("roots")

    normalized: dict[str, dict[str, Any]] = {}
    for root, entry in roots.items():
        raw_indices = entry.get("excluded_episode_indices", [])
        indices = [int(index) for index in raw_indices]

        excluded_frames = entry.get("excluded_frames")
        if excluded_frames is not None:
            excluded_frames = int(excluded_frames)

        normalized_root = str(Path(str(root)).expanduser().resolve())
        normalized[normalized_root] = {
            "excluded_episode_indices": frozenset(indices),
            "excluded_frames": excluded_frames,
        }

    return manifest_path, normalized
