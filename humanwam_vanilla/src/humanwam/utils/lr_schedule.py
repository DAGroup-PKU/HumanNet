"""Learning-rate schedules with an explicit, persistent decay floor."""
import math

from torch.optim.lr_scheduler import LRScheduler


class FloorCosineLR(LRScheduler):
    """Follow the existing 1%-endpoint cosine, then hold at an absolute floor.

    Warmup is deliberately owned by the caller, so its low initial learning
    rates are not clamped. Clamp progress as well as rate to prevent a rebound
    after T_max, including when a checkpoint is resumed beyond that horizon.
    """

    def __init__(self, optimizer, T_max, floor_lr, endpoint_ratio=0.01, last_epoch=-1):
        self.T_max = int(T_max)
        self.floor_lr = float(floor_lr)
        self.endpoint_ratio = float(endpoint_ratio)
        super().__init__(optimizer, last_epoch=last_epoch)

    def _get_closed_form_lr(self):
        progress = min(max(self.last_epoch / self.T_max, 0.0), 1.0)
        factor = self.endpoint_ratio + (1 - self.endpoint_ratio) * (1 + math.cos(math.pi * progress)) / 2
        return [max(self.floor_lr, base * factor) for base in self.base_lrs]

    def get_lr(self):
        return self._get_closed_form_lr()
