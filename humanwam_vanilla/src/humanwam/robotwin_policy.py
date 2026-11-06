"""RoboTwin observation/action adapter for the HumanWAM-Vanilla W0 checkpoint.

The image conversion and chunk seed sequence follow the paper evaluation entry.
Simulator execution lives in a separate process so its dependencies stay isolated.
"""
import io

import av
import numpy as np
import torch
from hydra.utils import instantiate
from torchvision.transforms import functional as TF

from humanwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from humanwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json


def prepare_images(cameras, *, device, dtype):
    """Resize three RGB views, form the T canvas, and match its training codec."""
    views = []
    for name in ('head_camera', 'left_camera', 'right_camera'):
        rgb = np.asarray(cameras[name])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f'{name} must be uint8 HWC RGB, got {rgb.dtype} {rgb.shape}')
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255)
        image = TF.resize(image, [256, 352], antialias=True)
        views.append(TF.normalize(image, mean=0.5, std=0.5))
    # Keep the historical bf16 rounding before the second wrist resize.
    views = torch.stack(views).to(device=device, dtype=dtype).float()
    wrists = [torch.nn.functional.interpolate(v.unsqueeze(0), size=(128, 176),
               mode='bilinear', align_corners=False, antialias=True)[0] for v in views[1:]]
    canvas = torch.cat([views[0], torch.cat(wrists, dim=-1)], dim=-2).to(dtype=dtype)
    pixels = ((canvas.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    pixels = pixels.permute(1, 2, 0).contiguous().cpu().numpy()
    buffer = io.BytesIO()
    with av.open(buffer, 'w', format='mp4') as container:
        stream = container.add_stream('libx264', rate=50)
        stream.width, stream.height, stream.pix_fmt = 352, 384, 'yuv444p'
        stream.options = dict(crf='10', g='8', bf='0', preset='medium', threads='1')
        frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
        frame.pts = 0
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    buffer.seek(0)
    with av.open(buffer) as container:
        decoded = next(container.decode(video=0)).to_ndarray(format='rgb24')
    return torch.from_numpy(decoded).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(
        device=device, dtype=dtype)


class RoboTwinPolicy:
    def __init__(self, cfg, checkpoint, stats, *, device='cuda:0', seed=0, steps=10):
        self.device = device
        self.dtype = torch.bfloat16
        self.seed, self.steps, self.chunk_index = int(seed), int(steps), 0
        cfg.model.load_text_encoder = True
        self.model = instantiate(cfg.model, model_dtype=self.dtype, device=device)
        self.model.load_checkpoint(str(checkpoint))
        self.model.eval()
        if self.model.video_vae_layout != 'robotwin_packed_multiview':
            raise ValueError('This adapter implements the W0 packed three-view protocol')
        self.processor = instantiate(cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats)))

    def reset(self):
        self.chunk_index = 0

    def prepare_images(self, cameras):
        return prepare_images(cameras, device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def predict(self, cameras, raw_state, instruction):
        state = torch.as_tensor(raw_state, dtype=torch.float32).reshape(1, -1)
        if state.shape != (1, 16) or not torch.isfinite(state).all():
            raise ValueError('Expected a finite world-frame EEF16 state')
        batch = self.processor.action_state_transform({'state': {'default': state}})
        anchor = batch['state']['default'].clone()
        proprio = self.processor.normalizer.normalizers['state']['default'].forward(anchor)
        image = self.prepare_images(cameras)
        result = self.model.infer_action(
            prompt=DEFAULT_PROMPT.format(task=str(instruction)), input_image=image,
            proprio=proprio.to(device=self.device, dtype=self.dtype),
            action_horizon=32, num_video_frames=9, num_inference_steps=self.steps,
            sigma_shift=None, seed=self.seed + self.chunk_index, rand_device='cpu',
            text_cfg_scale=1.0, tiled=False)
        action = self.processor.normalizer.normalizers['action']['default'].backward(
            result['action'].float().unsqueeze(0))
        batch = {'action': {'default': action}, 'state': {'default': anchor}}
        for transform in reversed(self.processor.action_state_transforms or []):
            batch = transform.backward(batch)
        output = batch['action']['default'][0].cpu().numpy().astype(np.float32)
        if output.shape != (32, 16) or not np.isfinite(output).all():
            raise ValueError('Expected finite world-frame actions with shape (32, 16)')
        output[:, [7, 15]] = np.clip(output[:, [7, 15]], 0, 1)
        self.chunk_index += 1
        return output
