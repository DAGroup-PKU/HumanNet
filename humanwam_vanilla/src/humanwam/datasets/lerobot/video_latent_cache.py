# Modified by Open-HumanScale contributors: removed unused release runtime code.
"""Read-only BF16 mmap cache for deterministic frozen video-VAE latents."""

from __future__ import annotations

import bisect
import json
import os
from pathlib import Path

import numpy as np
import torch


SCHEMA = "fastwam_video_latent_cache_v1"
STORAGE_DTYPE = np.dtype("<u2")


def storage_to_bf16_tensor(value: np.ndarray) -> torch.Tensor:
    """Copy a mmap slice and reinterpret its uint16 payload as BF16."""
    storage = np.array(value, dtype=STORAGE_DTYPE, copy=True)
    return torch.from_numpy(storage).view(torch.bfloat16)


class VideoLatentCache:
    """Lazily mmap contiguous sample-index shards produced by the frozen VAE."""

    def __init__(
        self,
        cache_dir: str | os.PathLike,
    ) -> None:
        self.root = Path(cache_dir)
        manifest_path = self.root / "CACHE_MANIFEST.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.samples = int(self.manifest["samples"])
        self.video_frames = int(self.manifest["video_frames"])
        self.latent_shape = tuple(int(value) for value in self.manifest["latent_shape"])

        self.shards = tuple(self.manifest["shards"])
        self._stops = []
        for shard in self.shards:
            self._stops.append(int(shard["stop"]))
        self._buffers: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return self.samples

    def _buffer(self, shard_index: int) -> np.memmap:
        buffer = self._buffers.get(shard_index)
        if buffer is None:
            shard = self.shards[shard_index]
            start, stop = int(shard["start"]), int(shard["stop"])
            buffer = np.memmap(
                self.root / shard["file"],
                dtype=STORAGE_DTYPE,
                mode="r",
                shape=(stop - start, *self.latent_shape),
            )
            self._buffers[shard_index] = buffer
        return buffer

    def lookup(self, sample_index: int) -> torch.Tensor:
        index = int(sample_index)
        shard_index = bisect.bisect_right(self._stops, index)
        shard = self.shards[shard_index]
        local_index = index - int(shard["start"])
        return storage_to_bf16_tensor(self._buffer(shard_index)[local_index])
