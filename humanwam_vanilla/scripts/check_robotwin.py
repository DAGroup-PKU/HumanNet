#!/usr/bin/env python3
"""Load a RoboTwin task, render observations and exercise its action/reset API.

This is a simulator smoke test, not a learned-policy benchmark.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robotwin-root', type=Path, required=True)
    parser.add_argument('--task', default='beat_block_hammer')
    parser.add_argument('--task-config', default='demo_clean')
    parser.add_argument('--expert', action='store_true', help='Also run the task expert planner')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.robotwin_root.resolve(), args.output.resolve()
    icd = Path('/etc/vulkan/icd.d/nvidia_icd.json')
    if icd.is_file():
        os.environ.setdefault('VK_ICD_FILENAMES', str(icd))
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    import numpy as np
    import yaml

    os.chdir(root)
    sys.path.insert(0, str(root))
    config = yaml.safe_load((root / 'task_config' / (args.task_config + '.yml')).read_text())
    embodiments = yaml.safe_load((root / 'task_config/_embodiment_config.yml').read_text())
    assert len(config['embodiment']) == 1, 'This smoke test uses one dual-arm embodiment'
    embodiment = config['embodiment'][0]
    robot_path = (root / embodiments[embodiment]['file_path']).resolve()
    robot_config = yaml.safe_load((robot_path / 'config.yml').read_text())
    config.update(task_name=args.task, task_config=args.task_config,
                  left_robot_file=str(robot_path), right_robot_file=str(robot_path),
                  left_embodiment_config=robot_config, right_embodiment_config=robot_config,
                  dual_arm_embodied=True, embodiment_name=embodiment,
                  save_data=False, need_plan=True, eval_mode=True, render_freq=0,
                  eval_video_log=False, eval_video_save_dir=None)
    module = importlib.import_module('envs.' + args.task)
    task = getattr(module, args.task)()
    report = dict(passed=False, task=args.task, config=args.task_config,
                  learned_policy_tested=False, expert_requested=args.expert, resets=[])
    try:
        for seed in (0, 1):
            task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **config)
            observations = task.get_obs()['observation']
            shapes = {name: list(payload['rgb'].shape) for name, payload in observations.items()
                      if 'rgb' in payload}
            assert len(shapes) >= 3, shapes
            assert all(np.isfinite(p['rgb']).all() for p in observations.values() if 'rgb' in p)
            if seed == 0:
                for _ in range(2):
                    action = np.asarray(task.robot.get_left_arm_jointState()
                                        + task.robot.get_right_arm_jointState(), dtype=np.float64)
                    assert action.shape == (14,) and np.isfinite(action).all()
                    task.take_action(action, action_type='qpos')
                    observations = task.get_obs()['observation']
                    assert all(np.isfinite(p['rgb']).all() for p in observations.values() if 'rgb' in p)
                report['action_calls'] = int(task.take_action_cnt)
            elif args.expert:
                task.play_once()
                report['expert_plan_success'] = bool(task.plan_success)
                report['expert_task_success'] = bool(task.check_success())
                assert report['expert_plan_success'] and report['expert_task_success']
            report['resets'].append(dict(seed=seed, camera_shapes=shapes))
            task.close_env(clear_cache=True)
        report['passed'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
