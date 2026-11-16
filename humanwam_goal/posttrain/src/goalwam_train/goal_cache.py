"""RoboTwin 2.0 training goal cache (`robotwin_train_goal_cache_v1`).

    python -m goalwam_train.goal_cache verify --root /path/to/train_goal_cache
"""

import argparse
import hashlib
import io
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

SCHEMA = "robotwin_train_goal_cache_v1"
IMAGE_SIZE = [224, 288]  # [H, W]
EPISODE_KEYS = frozenset({"task", "episode_index", "instruction", "goal_image", "sha256"})


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _local(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Goal cache path escapes root: {relative}")
    return path


def _validate_goal_png(raw, path):
    with Image.open(io.BytesIO(raw)) as image:
        if image.mode != "RGB" or image.size != (IMAGE_SIZE[1], IMAGE_SIZE[0]):
            raise ValueError(f"{path}: expected 288x224 RGB PNG, got {image.size} {image.mode}")
        pixels = np.asarray(image, dtype=np.uint8)
    # 240x320 source camera letterboxed into 224x288 leaves 4-pixel top/bottom black bars.
    outside = pixels.copy()
    outside[4:220, :] = 0
    if outside.any():
        raise ValueError(f"{path}: expected black top/bottom letterbox bars")
    return pixels


class TrainGoalCache:
    """Episode-paired 224x288 head-camera goals for RoboTwin `clean_50` post-training."""

    def __init__(self, root, manifest_sha256=None):
        self.root = Path(root).resolve()
        complete_path = self.root / "COMPLETE.json"
        if not complete_path.is_file():
            raise FileNotFoundError(f"Incomplete training goal cache (missing COMPLETE.json): {self.root}")
        marker = json.loads(complete_path.read_text())
        if marker.get("schema") != SCHEMA or marker.get("complete") is not True:
            raise ValueError(f"Unsupported or incomplete training goal cache: {self.root}")
        expected_sha = manifest_sha256 or marker.get("manifest_sha256")
        manifest_raw = (self.root / "manifest.json").read_bytes()
        if sha256_bytes(manifest_raw) != expected_sha:
            raise ValueError("Training goal cache manifest SHA-256 mismatch")
        self.manifest_sha256 = expected_sha
        manifest = json.loads(manifest_raw)
        if manifest.get("schema") != SCHEMA or manifest.get("image_size") != IMAGE_SIZE:
            raise ValueError("Unsupported training goal cache manifest schema or image_size")
        self.tasks = frozenset(manifest["tasks"])
        self.entries = {}
        for row in manifest["episodes"]:
            key = (row["task"], int(row["episode_index"]))
            if key in self.entries:
                raise ValueError(f"Duplicate episode in training goal cache: {key}")
            _local(self.root, row["goal_image"])
            self.entries[key] = row
        if len(self.entries) != int(marker["episodes"]):
            raise ValueError("Training goal cache episode count disagrees with COMPLETE.json")
        self._cache = OrderedDict()

    def resolve_task(self, dataset_path, episode):
        """Resolve canonical RoboTwin task name for a dataset directory."""
        if "annotation" in episode and episode["annotation"]:
            ann = json.loads(episode["annotation"])
            task = (ann.get("source") or {}).get("task")
            if task in self.tasks:
                return task
        dataset_path = Path(dataset_path)
        for candidate in (dataset_path.name, dataset_path.parent.name):
            if candidate in self.tasks:
                return candidate
            for prefix in self.tasks:
                if candidate.startswith(prefix + "_"):
                    return prefix
        raise ValueError(f"Cannot match dataset {dataset_path} to a RoboTwin task in {self.root}")

    def has(self, task, episode_index, instruction):
        row = self.entries.get((task, int(episode_index)))
        if row is None:
            return False
        if row["instruction"] != instruction:
            raise ValueError(
                f"Training goal cache instruction mismatch for {task}/episode_{episode_index:06d}"
            )
        return True

    def read(self, task, episode_index, instruction):
        """Return verified HWC uint8 RGB array [224, 288, 3]."""
        key = (task, int(episode_index))
        row = self.entries.get(key)
        if row is None:
            raise KeyError(f"Missing pre-generated training goal for {key}")
        if row["instruction"] != instruction:
            raise ValueError(f"Instruction mismatch for {key}: {instruction!r} != {row['instruction']!r}")
        if key not in self._cache:
            path = _local(self.root, row["goal_image"])
            raw = path.read_bytes()
            if sha256_bytes(raw) != row["sha256"]:
                raise ValueError(f"Training goal PNG checksum mismatch: {path}")
            with Image.open(io.BytesIO(raw)) as image:
                pixels = np.asarray(image, dtype=np.uint8).copy()
            if pixels.shape != (IMAGE_SIZE[0], IMAGE_SIZE[1], 3):
                raise ValueError(f"Unexpected training goal dimensions at {path}: {pixels.shape}")
            self._cache[key] = pixels
            while len(self._cache) > 32:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]


def verify_goal_cache(root, manifest_sha256=None):
    cache = TrainGoalCache(root, manifest_sha256=manifest_sha256)
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    if set(manifest) != {"schema", "image_size", "tasks", "episodes"}:
        raise ValueError(f"Unexpected top-level keys in manifest.json: {sorted(manifest)}")
    for row in manifest["episodes"]:
        if set(row) != EPISODE_KEYS or row["goal_image"] != f"{row['task']}/episode_{int(row['episode_index']):06d}.png":
            raise ValueError(f"Non-canonical episode entry in manifest.json: {row}")
    extra_files = {
        str(p.relative_to(cache.root))
        for p in cache.root.rglob("*")
        if p.is_file()
    } - {"manifest.json", "COMPLETE.json", *(r["goal_image"] for r in cache.entries.values())}
    if extra_files:
        raise ValueError(f"Unexpected extra files in goal cache: {sorted(extra_files)[:5]}")
    for row in cache.entries.values():
        path = _local(cache.root, row["goal_image"])
        raw = path.read_bytes()
        if sha256_bytes(raw) != row["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        _validate_goal_png(raw, path)
    return dict(
        schema=SCHEMA,
        tasks=len(cache.tasks),
        episodes=len(cache.entries),
        manifest_sha256=cache.manifest_sha256,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify", help="Verify every image, checksum and entry in a training goal cache")
    v.add_argument("--root", type=Path, required=True)
    v.add_argument("--manifest-sha256", default=None)
    args = parser.parse_args(argv)
    print(json.dumps(verify_goal_cache(args.root, args.manifest_sha256), indent=2), flush=True)


if __name__ == "__main__":
    main()
