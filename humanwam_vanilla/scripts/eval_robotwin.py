#!/usr/bin/env python3
"""Evaluate HumanWAM-Vanilla with separate policy and RoboTwin Python environments.

Run this script with the simulation environment's Python. A local subprocess uses
--policy-python to load the model; no network service or training data is needed.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time

from robotwin_protocol import SeedRetries
from robotwin_replay import load_replay, verify_replay_policy


class PolicyProcess:
    def __init__(self, command, log):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=log, text=True, bufsize=1)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            self.ready = self.receive(timeout=600)
            if not self.ready.get('ready'):
                raise RuntimeError(f'Policy failed to initialize: {self.ready}')
        except BaseException:
            self.close()
            raise

    def receive(self, timeout=300):
        if not self.selector.select(timeout):
            raise TimeoutError('Policy response timed out; inspect policy.log')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'Policy worker exited ({self.process.poll()}); inspect policy.log')
        return json.loads(line)

    def request(self, payload):
        self.process.stdin.write(json.dumps(payload) + '\n')
        self.process.stdin.flush()
        return self.receive()

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.selector.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robotwin-root', type=Path, required=True)
    parser.add_argument('--policy-python', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/robotwin.yaml')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--stats', type=Path, required=True)
    task_selection = parser.add_mutually_exclusive_group()
    task_selection.add_argument('--tasks', nargs='+')
    task_selection.add_argument('--task-file', type=Path, help='One task name per line; # comments allowed')
    parser.add_argument('--phase', choices=['clean', 'random'], default='clean')
    parser.add_argument('--episodes', type=int, default=7, help='Episodes in this shard (paper: 7,7,6,6,6,6,6,6)')
    parser.add_argument('--seed', type=int, default=0, help='Paper shard index, 0 through 7')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--max-chunks', type=int, help='Diagnostic truncation; excluded from benchmark success rate')
    parser.add_argument('--replay-from', type=Path,
                        help='Diagnostic receipt replay; inherits phase, shard, denoising steps and instructions')
    parser.add_argument('--replay-task', help='One task from --replay-from')
    parser.add_argument('--replay-seeds', nargs='+', type=int, help='Completed scene seeds to replay, in this order')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    replay = None
    if args.replay_from:
        if not args.replay_task or args.tasks or args.task_file:
            parser.error('--replay-from requires --replay-task and cannot be combined with --tasks/--task-file')
        try:
            replay = load_replay(args.replay_from, args.replay_task, args.replay_seeds)
        except (ValueError, KeyError, OSError) as exc:
            parser.error(str(exc))
        args.tasks = [replay['task']]
        args.phase, args.seed, args.steps = replay['phase'], replay['shard'], replay['steps']
        args.episodes = len(replay['episodes'])
    elif args.replay_task or args.replay_seeds:
        parser.error('--replay-task/--replay-seeds require --replay-from')
    if args.task_file:
        args.tasks = [line.split('#', 1)[0].strip() for line in args.task_file.read_text().splitlines()]
        args.tasks = [name for name in args.tasks if name]
    elif args.tasks is None:
        args.tasks = ['beat_block_hammer']
    if not args.tasks or len(set(args.tasks)) != len(args.tasks) or not all(name.isidentifier() for name in args.tasks):
        parser.error('Specify a nonempty list of unique task identifiers')
    if args.episodes < 1 or args.steps < 1 or (args.max_chunks is not None and args.max_chunks < 1):
        parser.error('episodes, steps, and max-chunks must be positive')
    if not 0 <= args.seed <= 7:
        parser.error('seed must be a paper shard index from 0 through 7')
    root, output = args.robotwin_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Resolving a venv's python symlink would accidentally run the system Python.
    command = [str(args.policy_python.absolute()), '-u', str(Path(__file__).with_name('robotwin_policy_worker.py')),
               '--config', str(args.config.resolve()), '--checkpoint', str(args.checkpoint.resolve()),
               '--stats', str(args.stats.resolve()), '--seed', str(args.seed), '--steps', str(args.steps)]
    icd = Path('/etc/vulkan/icd.d/nvidia_icd.json')
    if icd.is_file():
        os.environ.setdefault('VK_ICD_FILENAMES', str(icd))
    os.environ.setdefault('ROBOTWIN_RENDER_DEVICE', 'cuda:0')
    os.environ.setdefault('ROBOTWIN_RT_DENOISER', 'oidn')
    import numpy as np
    import yaml
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / 'description/utils')]
    from generate_episode_instructions import generate_episode_descriptions
    from envs.utils.create_actor import UnStableError

    phase_name = 'demo_clean' if args.phase == 'clean' else 'demo_randomized'
    config = yaml.safe_load((root / 'task_config' / (phase_name + '.yml')).read_text())
    embodiments = yaml.safe_load((root / 'task_config/_embodiment_config.yml').read_text())
    if len(config['embodiment']) != 1:
        raise ValueError('Expected one dual-arm embodiment')
    embodiment = config['embodiment'][0]
    robot_path = (root / embodiments[embodiment]['file_path']).resolve()
    robot_cfg = yaml.safe_load((robot_path / 'config.yml').read_text())
    config.update(task_config=phase_name, left_robot_file=str(robot_path), right_robot_file=str(robot_path),
                  left_embodiment_config=robot_cfg, right_embodiment_config=robot_cfg,
                  dual_arm_embodied=True, embodiment_name=embodiment, save_data=False,
                  need_plan=True, eval_mode=True, render_freq=0, eval_video_log=False,
                  eval_video_save_dir=None)
    cameras = yaml.safe_load((root / 'task_config/_camera_config.yml').read_text())
    head = cameras[config['camera']['head_camera_type']]
    config.update(head_camera_h=head['h'], head_camera_w=head['w'])
    report = dict(status='running', phase=args.phase, instruction_type='unseen', seed=args.seed,
                  episodes_per_task=args.episodes, steps=args.steps, replan_steps=24,
                  max_chunks=args.max_chunks, robotwin_root=str(root),
                  denoiser=os.environ['ROBOTWIN_RT_DENOISER'], tasks=[],
                  instruction_sampling='Historical Python random state is unseeded; actual instructions are recorded per episode',
                  benchmark_parity_verified=False)
    if replay:
        report['replay'] = replay
        report['instruction_sampling'] = 'Exact instruction strings from the replay source receipt'
    report['simulator'] = dict(python=sys.version, prefix=sys.prefix, versions={})
    source = hashlib.sha256()
    for directory in ('envs', 'task_config', 'description'):
        for path in sorted((root / directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.json', '.yml', '.yaml') and 'curobo' not in path.relative_to(root).parts:
                source.update(str(path.relative_to(root)).encode() + b'\0')
                source.update(path.read_bytes())
    source.update(Path(__file__).read_bytes())
    source.update(Path(__file__).with_name('robotwin_protocol.py').read_bytes())
    source.update(Path(__file__).with_name('robotwin_replay.py').read_bytes())
    report['simulator']['source_sha256'] = source.hexdigest()
    for package in ('sapien', 'mplib', 'torch', 'warp-lang', 'numpy', 'scipy', 'nvidia-curobo'):
        report['simulator']['versions'][package] = importlib.metadata.version(package)

    def save():
        pending = output / 'result.json.tmp'
        pending.write_text(json.dumps(report, indent=2) + '\n')
        pending.replace(output / 'result.json')

    save()
    policy = None
    try:
        if os.environ['ROBOTWIN_RT_DENOISER'] == 'oidn':
            from check_sim_environment import check_oidn_device
            report['oidn_device'] = check_oidn_device()
        with (output / 'policy.log').open('w') as log, tempfile.TemporaryDirectory(prefix='humanwam-observation-') as temp:
            policy = PolicyProcess(command, log)
            report['policy'] = policy.ready
            if replay:
                verify_replay_policy(replay, policy.ready)
            report['task_config'] = config
            save()
            for task_name in args.tasks:
                config['task_name'] = task_name
                task = getattr(importlib.import_module('envs.' + task_name), task_name)()
                task_report = dict(task=task_name, episodes=[], skipped_seeds=[])
                report['tasks'].append(task_report)
                retries = SeedRetries(args.episodes, UnStableError)
                seed = 100000 * (1 + args.seed)
                try:
                    while len(task_report['episodes']) < args.episodes:
                        now_id = len(task_report['episodes'])
                        if replay:
                            selected = replay['episodes'][now_id]
                            seed, now_id = selected['seed'], selected['original_episode_index']
                        task_report.update(stage='expert_check', candidate_seed=seed)
                        save()
                        try:
                            task.setup_demo(now_ep_num=now_id, seed=seed, is_test=True, **config)
                            info = task.play_once()
                            task.close_env()
                            valid = bool(task.plan_success and task.check_success())
                        except Exception as exc:
                            if replay:
                                raise RuntimeError(f'Replay expert check failed at fixed seed {seed}; no seed substitution') from exc
                            if not retries.retry(exc, 'expert'):
                                raise
                            task.close_env()
                            task_report['skipped_seeds'].append(dict(seed=seed, stage='expert', error=f'{type(exc).__name__}: {exc}'))
                            seed += 1
                            save()
                            continue
                        if not valid:
                            if replay:
                                raise RuntimeError(f'Replay expert rejected fixed seed {seed}; no seed substitution')
                            task_report['skipped_seeds'].append(dict(seed=seed, error='expert_failed'))
                            seed += 1
                            save()
                            continue
                        try:
                            task.setup_demo(now_ep_num=now_id, seed=seed, is_test=True, **config)
                        except Exception as exc:
                            if replay:
                                raise RuntimeError(f'Replay policy setup failed at fixed seed {seed}; no seed substitution') from exc
                            if not retries.retry(exc, 'policy_setup'):
                                raise
                            task.close_env()
                            task_report['skipped_seeds'].append(dict(seed=seed, stage='policy_setup', error=f'{type(exc).__name__}: {exc}'))
                            seed += 1
                            save()
                            continue
                        if replay and int(task.step_lim) != selected['step_limit']:
                            raise RuntimeError('Replay task horizon differs from its source episode')
                        descriptions = generate_episode_descriptions(task_name, [info['info']],
                            replay['description_count'] if replay else args.episodes)
                        instruction = str(np.random.choice(descriptions[0]['unseen']))
                        if replay:
                            instruction = selected['instruction']
                        task.set_instruction(instruction=instruction)
                        policy.request(dict(reset=True))
                        episode = dict(seed=seed, instruction=instruction, chunks=0, action_calls=0,
                                       success=False, truncated=False)
                        report['active_episode'] = episode
                        task_report['stage'] = 'policy_rollout'
                        save()
                        start = time.monotonic()
                        while task.take_action_cnt < task.step_lim and not task.eval_success:
                            observation = task.get_obs()
                            eef = observation['endpose']
                            state = np.concatenate([np.asarray(eef['left_endpose']).reshape(-1),
                                np.asarray(eef['left_gripper']).reshape(-1),
                                np.asarray(eef['right_endpose']).reshape(-1),
                                np.asarray(eef['right_gripper']).reshape(-1)])
                            arrays = {k: observation['observation'][k]['rgb'] for k in
                                      ('head_camera', 'left_camera', 'right_camera')}
                            episode['camera_shapes'] = {k: list(v.shape) for k, v in arrays.items()}
                            path = Path(temp) / 'observation.npz'
                            np.savez(path, state=state, **arrays)
                            reply = policy.request(dict(observation=str(path), instruction=instruction))
                            actions = np.asarray(reply['actions'], dtype=np.float32)
                            if actions.shape != (32, 16) or not np.isfinite(actions).all():
                                raise ValueError('Invalid policy actions')
                            if replay and episode['chunks'] == 0:
                                artifact = output / f'{task_name}-seed{seed}-first-chunk.npz'
                                np.savez(artifact, state=state, actions=actions, **arrays)
                                episode['first_chunk_artifact'] = artifact.name
                                episode['first_chunk_sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest()
                            for action in actions[:24]:
                                if task.eval_success or task.take_action_cnt >= task.step_lim:
                                    break
                                task.take_action(action, action_type='ee')
                            episode['chunks'] += 1
                            episode['action_calls'] = int(task.take_action_cnt)
                            save()
                            print(f'{task_name} seed={seed} chunk={episode["chunks"]} actions={task.take_action_cnt} success={task.eval_success}', flush=True)
                            if args.max_chunks and episode['chunks'] >= args.max_chunks:
                                episode['truncated'] = not bool(task.eval_success) and task.take_action_cnt < task.step_lim
                                break
                        episode.update(success=bool(task.eval_success), seconds=time.monotonic()-start,
                                       step_limit=int(task.step_lim))
                        task_report['episodes'].append(episode)
                        report.pop('active_episode', None)
                        task.close_env(clear_cache=((now_id + 2) % config['clear_cache_freq'] == 0))
                        seed += 1
                        save()
                    task_report['stage'] = 'complete'
                    task_report.pop('candidate_seed', None)
                finally:
                    task.close_env(clear_cache=True)
            report['status'] = 'diagnostic_complete' if args.max_chunks or replay else 'complete'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if policy is not None:
            policy.close()
        save()


if __name__ == '__main__':
    main()
