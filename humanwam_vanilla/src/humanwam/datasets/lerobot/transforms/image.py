import torch
import torch.nn as nn
import torchvision.transforms as TF


class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        # uint8 [0,255] -> float [0,1]; float inputs (e.g. the float video cache)
        # pass through unchanged to avoid a redundant uint8<->float round-trip.
        if x.dtype == torch.uint8:
            return x.to(torch.float32) / 255.0
        return x

class Pad(nn.Module):
    def __init__(self, padding, fill=0, padding_mode='constant'):
        super().__init__()
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode
        self.pad = TF.Pad(padding=tuple(padding), fill=fill, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor):
        return self.pad(x)
