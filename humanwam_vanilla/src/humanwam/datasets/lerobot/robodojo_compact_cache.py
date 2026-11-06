"""Verified full-corpus uint8 compact canvas cache. No online decode/resize."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


class RoboDojoCompactCache:
    SEPARATE_THREE_VIEWS = False
    PREMERGED_T = True
    SOURCE_RESIZE_SIGNATURE = True

    def __init__(self, inner_dataset, out_hw, cache_dir, *, camera_keys, transform_signature, **kwargs):
        self.dir = Path(cache_dir)
        raw = (self.dir / 'CACHE_MANIFEST.json').read_bytes()
        m = self.manifest = json.loads(raw)
        complete = json.loads((self.dir / 'COMPLETE.json').read_text())
        assert complete['status'] == 'PASS' and complete['episodes'] == 3500
        assert complete['manifest_sha256'] == hashlib.sha256(raw).hexdigest()
        assert m['dtype'] == 'uint8' and m['shape'] == [3, 384, 352]
        assert m['source_root'] == str(Path(inner_dataset.root).resolve())
        assert m['transform_signature'] == transform_signature
        assert m['source_out_hw'] == list(out_hw)
        self.camera_keys = list(camera_keys)
        assert self.camera_keys == ['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist']
        self.episodes = list(inner_dataset.episodes) if inner_dataset.episodes is not None else list(range(inner_dataset.meta.total_episodes))
        self.episodes = [int(x) for x in self.episodes]
        self.ep_lengths = {ep: int(inner_dataset.episode_data_index['to'][i] - inner_dataset.episode_data_index['from'][i]) for i, ep in enumerate(self.episodes)}
        assert self.episodes == m['episodes']
        assert {str(k): v for k, v in self.ep_lengths.items()} == m['episode_lengths']
        self.fps = float(inner_dataset.meta.fps)
        assert self.fps == m['fps']
        self._fds = {}

    def _path(self, ep):
        return self.dir / f'episode_{int(ep):06d}.u8'

    def missing_episodes(self):
        return [ep for ep in self.episodes if not self._path(ep).is_file() or self._path(ep).stat().st_size != self.ep_lengths[ep]*3*384*352]

    def is_complete(self):
        return not self.missing_episodes()

    def validate_training_storage(self):
        assert self.dir.is_dir()

    def lookup_window(self, query_timestamps, ep_idx, frame_positions=None):
        ep = int(ep_idx)
        ts = query_timestamps[self.camera_keys[0]]
        indices = np.clip(np.round(np.asarray(ts, dtype=np.float64)*self.fps), 0, self.ep_lengths[ep]-1).astype(np.int64)
        if frame_positions is not None:
            indices = indices[np.asarray(frame_positions, dtype=np.int64)]
        for key in self.camera_keys[1:]:
            assert np.array_equal(query_timestamps[key], ts), 'Camera timestamps differ'
        key = (os.getpid(), ep)
        fd = self._fds.pop(key, None)
        if fd is None:
            fd = os.open(self._path(ep), os.O_RDONLY)
            while len(self._fds) >= 64:
                old = next(iter(self._fds))
                previous = self._fds.pop(old)
                if old[0] == key[0]:
                    os.close(previous)
        self._fds[key] = fd
        size = 3*384*352
        frames = {}
        for index in np.unique(indices):
            data = os.pread(fd, size, int(index)*size)
            if len(data) != size:
                raise EOFError(f'Truncated compact cache: episode={ep} frame={index}')
            frames[int(index)] = np.frombuffer(data, dtype=np.uint8).reshape(3, 384, 352)
        return torch.from_numpy(np.stack([frames[int(i)] for i in indices]))
