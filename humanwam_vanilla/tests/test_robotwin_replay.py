import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('robotwin_replay', SCRIPTS / 'robotwin_replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def source_receipt(tmp_path):
    policy = dict(ready=True, checkpoint_sha256='weights', stats_sha256='stats',
                  model_config={'test': 'model'}, processor_config={'test': 'processor'},
                  action_horizon=32, replan_steps=24, steps=10)
    source = dict(phase='random', seed=0, instruction_type='unseen', max_chunks=None,
                  steps=10, replan_steps=24, episodes_per_task=7, policy=policy,
                  tasks=[dict(task='task_a', episodes=[dict(seed=100000+i, success=True,
                      truncated=False, instruction=f'original instruction {i}',
                      action_calls=2, step_limit=4) for i in range(2)])])
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(source))
    return path, source


def test_replay_selects_exact_instruction_and_original_episode_index(tmp_path):
    path, source = source_receipt(tmp_path)
    plan = replay.load_replay(path, 'task_a', [100001])
    assert plan['phase'] == 'random'
    assert plan['episodes'][0]['instruction'] == 'original instruction 1'
    assert plan['episodes'][0]['original_episode_index'] == 1
    assert plan['description_count'] == 7
    replay.verify_replay_policy(plan, source['policy'])
    with pytest.raises(ValueError, match='checkpoint_sha256'):
        replay.verify_replay_policy(plan, dict(source['policy'], checkpoint_sha256='other'))


@pytest.mark.parametrize('kind', ['truncated', 'missing_instruction', 'early_failure', 'duplicate_seed'])
def test_invalid_replay_evidence_is_rejected(tmp_path, kind):
    path, source = source_receipt(tmp_path)
    episode = source['tasks'][0]['episodes'][0]
    if kind == 'truncated':
        episode['truncated'] = True
    elif kind == 'missing_instruction':
        episode['instruction'] = ''
    elif kind == 'early_failure':
        episode['success'] = False
    else:
        source['tasks'][0]['episodes'][1]['seed'] = episode['seed']
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError):
        replay.load_replay(path, 'task_a')


@pytest.mark.parametrize('mode', ['normal', 'replay', 'reject'])
def test_evaluator_replay_uses_fixed_seed_and_never_substitutes(tmp_path, monkeypatch, mode):
    reject_expert = mode == 'reject'
    is_replay = mode != 'normal'
    generated = 'a new random instruction must not replace the original'
    expected_instruction = 'original instruction 1' if is_replay else generated
    path, source = source_receipt(tmp_path)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    module_spec = importlib.util.spec_from_file_location('eval_robotwin_replay_test', SCRIPTS / 'eval_robotwin.py')
    evaluator = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(evaluator)
    root = tmp_path / 'robotwin'
    (root / 'task_config').mkdir(parents=True)
    (root / 'robot').mkdir()
    (root / 'robot/config.yml').write_text('{}')
    (root / 'task_config/demo_randomized.yml').write_text(
        'embodiment: [robot]\ncamera: {head_camera_type: D435}\nclear_cache_freq: 5\n')
    (root / 'task_config/_embodiment_config.yml').write_text('robot: {file_path: robot}')
    (root / 'task_config/_camera_config.yml').write_text('D435: {h: 2, w: 2}')
    calls, instructions = [], []

    class Task:
        plan_success = True
        step_lim = 4

        def setup_demo(self, now_ep_num, seed, **kwargs):
            calls.append((now_ep_num, seed))
            self.take_action_cnt, self.eval_success = 0, False

        def play_once(self):
            return {'info': {}}

        def check_success(self):
            return not reject_expert

        def close_env(self, **kwargs):
            pass

        def set_instruction(self, instruction):
            instructions.append(instruction)

        def get_obs(self):
            return dict(endpose=dict(left_endpose=np.zeros(7), left_gripper=0,
                                     right_endpose=np.zeros(7), right_gripper=0),
                        observation={name: {'rgb': np.zeros((2, 2, 3), np.uint8)}
                                     for name in ('head_camera', 'left_camera', 'right_camera')})

        def take_action(self, action, **kwargs):
            self.take_action_cnt += 1
            self.eval_success = self.take_action_cnt == 2

    class Policy:
        def __init__(self, *args):
            self.ready = source['policy']

        def request(self, payload):
            if not payload.get('reset'):
                assert payload['instruction'] == expected_instruction
            return {'actions': np.zeros((32, 16)).tolist()}

        def close(self):
            pass

    for name, attrs in {
        'envs.task_a': {'task_a': Task},
        'envs.utils.create_actor': {'UnStableError': RuntimeError},
        'generate_episode_instructions': {'generate_episode_descriptions':
            lambda *args: [{'unseen': [generated]}]},
    }.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(evaluator, 'PolicyProcess', Policy)
    monkeypatch.setattr(evaluator.importlib.metadata, 'version', lambda name: 'test')
    monkeypatch.setenv('ROBOTWIN_RT_DENOISER', 'optix')
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'output'
    command = ['eval_robotwin.py', '--robotwin-root', str(root),
        '--policy-python', sys.executable, '--checkpoint', str(tmp_path / 'checkpoint'),
        '--stats', str(tmp_path / 'stats'), '--output', str(output)]
    if is_replay:
        command += ['--replay-from', str(path), '--replay-task', 'task_a', '--replay-seeds', '100001']
    else:
        command += ['--tasks', 'task_a', '--phase', 'random', '--episodes', '1']
    monkeypatch.setattr(sys, 'argv', command)
    original_path = list(sys.path)
    try:
        if reject_expert:
            with pytest.raises(RuntimeError, match='no seed substitution'):
                evaluator.main()
        else:
            evaluator.main()
    finally:
        sys.path[:] = original_path
    result = json.loads((output / 'result.json').read_text())
    if is_replay:
        assert result['replay']['source_sha256']
    if reject_expert:
        assert calls == [(1, 100001)]
        assert result['status'] == 'failed'
        assert not result['tasks'][0]['episodes']
    else:
        expected_call = (1, 100001) if is_replay else (0, 100000)
        assert calls == [expected_call, expected_call]
        assert instructions == [expected_instruction]
        assert result['status'] == ('diagnostic_complete' if is_replay else 'complete')
        episode = result['tasks'][0]['episodes'][0]
        assert episode['success'] and not episode['truncated']
        if is_replay:
            with np.load(output / episode['first_chunk_artifact']) as artifact:
                assert artifact['actions'].shape == (32, 16)
                assert artifact['head_camera'].shape == (2, 2, 3)
        else:
            assert 'first_chunk_artifact' not in episode
            assert 'replay' not in result
