"""Bounded, process-shared read-through staging for immutable merged frames.

Fixed slots avoid a directory walk/global eviction lock on every fill. Readers
hold the slot lock until bytes have been copied, so replacement cannot unlink a
live mmap or race an open. DataLoader workers provide ordered batch prefetch.
This class does not claim that a caller-provided directory is an NVMe device.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np


class LocalFrameStage:
    def __init__(self, directory, *, max_bytes, frame_bytes, block_frames=8, stride=4):
        self.directory = Path(directory)
        self.frame_bytes = int(frame_bytes)
        self.block_frames = int(block_frames)
        self.stride = int(stride)
        # Include metadata and filesystem block-rounding in the capacity bound.
        self.slot_bytes = ((self.frame_bytes * self.block_frames + 4095) // 4096) * 4096 + 8192
        self.slots = int(max_bytes) // self.slot_bytes
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = self.source_bytes = 0

    def _slot(self, key):
        return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little") % self.slots

    def _read_block(self, source, episode, length, start):
        count = min(self.block_frames, (length - start + self.stride - 1) // self.stride)
        key = f"{episode}:{length}:{start}:{count}:{self.stride}"
        slot = self._slot(key)
        stem = self.directory / f"slot_{slot:08d}"
        expected = count * self.frame_bytes
        # Lock files are stable inodes: never delete or atomically replace them.
        with stem.with_suffix(".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            meta = stem.with_suffix(".json")
            path = stem.with_suffix(".bin")
            try:
                record = json.loads(meta.read_text())
                if record["key"] == key and path.stat().st_size == expected:
                    payload = path.read_bytes()
                    self.hits += 1
                    return payload
            except (FileNotFoundError, ValueError, KeyError):
                pass
            # A killed/failed fill has no valid metadata and is rebuilt on retry.
            meta.unlink(missing_ok=True)
            pieces = []
            with open(source, "rb", buffering=0) as src:
                for frame in range(start, start + count * self.stride, self.stride):
                    offset = frame * self.frame_bytes
                    part = bytearray()
                    while len(part) < self.frame_bytes:
                        chunk = os.pread(src.fileno(), self.frame_bytes - len(part), offset + len(part))
                        if not chunk:
                            raise EOFError(f"truncated merged frame source: {source}, frame={frame}")
                        part.extend(chunk)
                    pieces.append(part)
            payload = b"".join(pieces)
            # Disposable local cache: no fsync and no extra temporary payload.
            # Truncation/replacement is protected through the complete read.
            # O_TRUNC on an existing ext4 file can trigger auto_da_alloc and a
            # synchronous writeback on close, even without fsync. Overwrite
            # under the slot lock, then trim the tail without truncating to 0.
            fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
            try:
                written = 0
                while written < len(payload):
                    count_written = os.pwrite(fd, memoryview(payload)[written:], written)
                    if count_written <= 0:
                        raise OSError("short local stage write")
                    written += count_written
                os.ftruncate(fd, len(payload))
            finally:
                os.close(fd)
            meta.write_text(json.dumps({"key": key}))
            self.misses += 1
            self.source_bytes += len(payload)
            return payload

    def lookup(self, source, episode, length, indices, *, dtype, frame_shape):
        indices = np.asarray(indices, dtype=np.int64)
        span = self.block_frames * self.stride
        starts = (indices // span) * span + indices % self.stride
        result = np.empty((len(indices), *frame_shape), dtype=dtype)
        for start in np.unique(starts):
            payload = self._read_block(source, episode, length, int(start))
            frames = np.frombuffer(payload, dtype=dtype).reshape(-1, *frame_shape)
            mask = starts == start
            result[mask] = frames[(indices[mask] - start) // self.stride]
        return result
