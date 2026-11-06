#!/usr/bin/env python3
"""Local stdin/stdout policy worker, launched by eval_robotwin.py."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()
    response_stream = sys.stdout

    def respond(payload):
        response_stream.write(json.dumps(payload) + '\n')
        response_stream.flush()

    # Library initialization and compiler diagnostics must not corrupt the protocol.
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        import humanwam
        from humanwam.config import load_config
        from humanwam.robotwin_policy import RoboTwinPolicy
        from omegaconf import OmegaConf
        torch.set_num_threads(1)
        cfg = load_config(args.config)
        policy = RoboTwinPolicy(cfg, args.checkpoint, args.stats,
                               seed=args.seed, steps=args.steps)
        def digest(path):
            result = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    result.update(block)
            return result.hexdigest()
        source = hashlib.sha256()
        package = Path(humanwam.__file__).parent
        for path in sorted(package.rglob('*.py')):
            source.update(str(path.relative_to(package)).encode() + b'\0')
            source.update(path.read_bytes())
        source.update(Path(__file__).read_bytes())
        respond(dict(ready=True, action_horizon=32, replan_steps=24, steps=args.steps,
                     checkpoint_sha256=digest(args.checkpoint), stats_sha256=digest(args.stats),
                     source_sha256=source.hexdigest(),
                     gpu=dict(name=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0)),
                     model_config=OmegaConf.to_container(cfg.model, resolve=True),
                     processor_config=OmegaConf.to_container(cfg.data.train.processor, resolve=True),
                     python=sys.version, prefix=sys.prefix,
                     versions={name: importlib.metadata.version(name) for name in ('torch','av','torchvision')}))
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('reset'):
                policy.reset()
                respond(dict(reset=True))
                continue
            with np.load(request['observation'], allow_pickle=False) as arrays:
                cameras = {key: arrays[key] for key in ('head_camera', 'left_camera', 'right_camera')}
                actions = policy.predict(cameras, arrays['state'], request['instruction'])
            respond(dict(actions=actions.tolist(), chunk_index=policy.chunk_index))


if __name__ == '__main__':
    main()
