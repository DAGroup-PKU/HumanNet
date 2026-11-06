"""OpenWAM RoboDojo ColorJitter, whole canvas, temporal factors shared.

Pinned reference: OpenWAM 3dfc9a7d9230f0e1382e6b22639278f1a86e5eb5.
PIL byte semantics are preserved; uint8 IPC avoids repeated float conversion.
"""
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from .video_augmentation import _mix_seed


class OpenWAMCompactAugmentation(Dataset):
    def __init__(self, dataset, *, seed=42, **kwargs):
        assert kwargs == {'name': 'openwam_robodojo_compact', 'probability': 1.0,
                          'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2,
                          'hue': 0.0, 'sync_time': True, 'sync_cameras': True}, kwargs
        self.dataset = dataset
        self.seed, self.epoch = int(seed), 0

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        if name == 'dataset':
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        nested = getattr(self.dataset, 'set_epoch', None)
        if callable(nested):
            nested(epoch)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        video = sample['video']
        assert video.dtype == torch.uint8 and video.ndim == 4 and video.shape[-2:] == (384, 352)
        rng = random.Random(_mix_seed(self.seed, self.epoch, int(index)))
        factors = [1.0 + rng.uniform(-0.2, 0.2) for _ in range(3)]
        frames = video.permute(1, 2, 3, 0).numpy()
        output = []
        for frame in frames:
            image = Image.fromarray(frame)
            for operation, factor in zip((ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color), factors):
                image = operation(image).enhance(factor)
            output.append(torch.from_numpy(np.array(image, copy=True)))
        sample['video'] = torch.stack(output).permute(3, 0, 1, 2).contiguous()
        return sample
