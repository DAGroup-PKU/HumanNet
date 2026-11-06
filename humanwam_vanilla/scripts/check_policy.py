#!/usr/bin/env python3
"""Load a full post-training checkpoint and predict actions from a real sample.

This checks offline inference, not simulator success rate.
"""
import argparse
import json
from pathlib import Path

import torch
from hydra.utils import instantiate

from humanwam.config import load_config
from humanwam.runtime import build_datasets
from humanwam.utils import misc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/robotwin.yaml')
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args()
    assert args.steps > 0
    cfg = load_config(args.config, args.paths, args.overrides)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    work = args.output.resolve().with_suffix('')
    work.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(work))
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device='cuda:0')
    payload = torch.load(cfg.resume, map_location='cpu', weights_only=True, mmap=True)
    expected = model.mot.state_dict()
    assert set(expected) == set(payload['mot']), 'Checkpoint and MoT parameter names differ'
    assert all(expected[k].shape == v.shape for k, v in payload['mot'].items())
    if model.proprio_encoder is not None:
        assert set(model.proprio_encoder.state_dict()) == set(payload['proprio_encoder'])
    checkpoint_step = payload.get('step')
    del payload, expected
    model.load_checkpoint(cfg.resume)
    model.eval()
    dataset, _ = build_datasets(cfg.data)
    sample = dataset[0]
    video = sample['video']
    image = video[:, 0]
    if image.dtype == torch.uint8:
        image = image.float().div(127.5).sub(1)
    proprio = sample.get('proprio')
    if proprio is not None and proprio.ndim == 2:
        proprio = proprio[0]
    result = model.infer_action(
        prompt=None, input_image=image, proprio=proprio,
        context=sample['context'].unsqueeze(0),
        context_mask=sample['context_mask'].unsqueeze(0),
        action_horizon=sample['action'].shape[0], num_video_frames=video.shape[1],
        num_inference_steps=args.steps, seed=42)
    action = result['action']
    torch.cuda.synchronize()
    assert tuple(action.shape[-2:]) == tuple(sample['action'].shape)
    assert torch.isfinite(action).all()
    report = dict(passed=True, checkpoint_step=checkpoint_step,
                  checkpoint_parameter_names_and_shapes_match=True,
                  inference_steps=args.steps, action_shape=list(action.shape),
                  action_min=float(action.min()), action_max=float(action.max()),
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                  simulator_policy_rollout_tested=False)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
