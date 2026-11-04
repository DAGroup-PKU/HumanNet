#!/usr/bin/env python3
"""Launch RoboTwin fine-tuning; --print-config prints settings without training."""
import argparse
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from humanwam.config import configure_launch, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/robotwin.yaml')
    parser.add_argument('--paths', type=Path, help='Local path bindings, shaped like configs/paths.example.yaml')
    parser.add_argument('--print-config', action='store_true', help='Print the resolved configuration without loading models')
    parser.add_argument('overrides', nargs='*', help='Explicit OmegaConf key=value overrides')
    args = parser.parse_args()
    cfg = load_config(args.config, args.paths, args.overrides)
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
