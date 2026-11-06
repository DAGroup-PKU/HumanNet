#!/usr/bin/env python3
"""Launch pretraining, mid-training or post-training from a stage configuration."""
import argparse
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from humanwam.config import configure_launch, load_config, set_training_checkpoint, training_config_gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/robotwin.yaml')
    parser.add_argument('--paths', type=Path, help='Local path bindings, shaped like configs/paths.example.yaml')
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument('--init-checkpoint', type=Path,
                            help='Previous-stage weights FILE; start a new optimizer and step counter')
    checkpoint.add_argument('--resume', type=Path,
                            help='Training-state DIRECTORY; continue the same interrupted run')
    parser.add_argument('--print-config', action='store_true', help='Print the resolved configuration without loading models')
    parser.add_argument('overrides', nargs='*', help='Explicit OmegaConf key=value overrides')
    args = parser.parse_args()
    if args.init_checkpoint is not None or args.resume is not None:
        if any(v.split('=', 1)[0] in {'resume', 'paths.resume', 'resume_step'} for v in args.overrides):
            parser.error('Do not combine checkpoint flags with resume/resume_step overrides')
    try:
        cfg = load_config(args.config, args.paths, args.overrides)
        set_training_checkpoint(cfg, init_checkpoint=args.init_checkpoint, resume=args.resume)
        gaps = training_config_gaps(cfg)
        if gaps:
            parser.error('Configuration is incomplete; fill these entries before launch:\n  '
                         + '\n  '.join(gaps))
        OmegaConf.to_container(cfg, resolve=True)
    except (OSError, ValueError, OmegaConfBaseException) as exc:
        parser.error(str(exc))
    launch = configure_launch(cfg)
    report = {'launch': launch, 'config': OmegaConf.to_container(cfg, resolve=True)}
    if launch['rank'] == 0:
        print(json.dumps(report, indent=2), flush=True)
    if args.print_config:
        return
    from humanwam.runtime import run_training
    run_training(cfg)


if __name__ == '__main__':
    main()
