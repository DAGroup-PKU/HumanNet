#!/usr/bin/env python3
"""Serve the W0 RoboDojo policy to an existing XPolicyLab WebSocket client.

Simulator/asset installation is separate; see docs/post_training.md.
"""
import argparse
import asyncio
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robodojo-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/robodojo.yaml')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=19276)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()
    xpolicy = args.robodojo_root.resolve() / 'XPolicyLab'
    if not (xpolicy / 'client_server/ws/model_server.py').is_file():
        parser.error('Expected the XPolicyLab WebSocket server under --robodojo-root')
    if args.steps < 1:
        parser.error('--steps must be positive')
    sys.path.insert(0, str(xpolicy))
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    import torch
    from humanwam.config import load_config
    from humanwam.robodojo_policy import RoboDojoPolicy, RoboDojoDeployment

    cfg = load_config(args.config, overrides=[
        'model.mot_compile_blocks=false', 'model.vae_compile_encoder=false'])
    torch.set_num_threads(1)
    policy = RoboDojoPolicy(cfg, args.checkpoint, args.stats, seed=args.seed, steps=args.steps)
    server = PolicyServer(RoboDojoDeployment(policy), PolicyServerConfig(
        host=args.host, port=args.port, ws_ping_timeout_s=None))
    print(f'Policy ready: ws://{args.host}:{args.port}', flush=True)
    asyncio.run(server.serve_forever())


if __name__ == '__main__':
    main()
