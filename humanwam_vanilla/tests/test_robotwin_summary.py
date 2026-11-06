import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    'summarize_robotwin', Path(__file__).resolve().parents[1] / 'scripts/summarize_robotwin.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
summarize = _module.summarize


def receipt(tmp_path, phase='clean', shard=0, *, count=None, checkpoint='same', name=None):
    expected = _module.SHARD_EPISODES[shard]
    report = dict(status='complete', phase=phase, seed=shard, episodes_per_task=expected,
                  steps=10, replan_steps=24, instruction_type='unseen', max_chunks=None,
                  policy=dict(action_horizon=32, checkpoint_sha256=checkpoint, stats_sha256='stats'),
                  tasks=[dict(task='task_a', episodes=[dict(seed=100000*(1+shard)+i,
                      success=False, truncated=False, action_calls=400, step_limit=400,
                      instruction='test instruction') for i in range(expected if count is None else count)])])
    path = tmp_path / (name or f'{phase}-{shard}.json')
    path.write_text(json.dumps(report))
    return path


def test_complete_coverage_counts_failed_episodes(tmp_path):
    paths = [receipt(tmp_path, phase, shard) for phase in ('clean', 'random') for shard in range(8)]
    result = summarize(paths, ['task_a'])
    assert result['coverage_complete']
    assert result['complete_cells'] == 16
    assert result['phases']['clean']['episodes'] == 50
    assert result['phases']['random']['success_rate_pct'] == 0
    assert result['benchmark_parity_verified'] is False


def test_partial_coverage_never_produces_benchmark_percentage(tmp_path):
    result = summarize([receipt(tmp_path), receipt(tmp_path, shard=1, count=2)], ['task_a'])
    assert result['complete_cells'] == 1
    assert not result['coverage_complete']
    assert result['phases']['clean']['episodes'] == 7
    assert result['phases']['clean']['success_rate_pct'] is None
    assert len(result['excluded']) == 1


def test_duplicate_cells_and_mixed_weights_are_rejected(tmp_path):
    first = receipt(tmp_path)
    duplicate = receipt(tmp_path, name='duplicate.json')
    with pytest.raises(ValueError, match='Duplicate cell'):
        summarize([first, duplicate], ['task_a'])
    different = receipt(tmp_path, shard=1, checkpoint='different')
    with pytest.raises(ValueError, match='mix checkpoints'):
        summarize([first, different], ['task_a'])


def test_early_stop_cannot_count_as_full_failure(tmp_path):
    path = receipt(tmp_path)
    report = json.loads(path.read_text())
    report['tasks'][0]['episodes'][0]['action_calls'] = 24
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='before the task horizon'):
        summarize([path], ['task_a'])
    report['max_chunks'] = 1
    path.write_text(json.dumps(report))
    result = summarize([path], ['task_a'])
    assert result['complete_cells'] == 0
    assert result['excluded'][0]['reason'] == 'truncated diagnostic'


@pytest.mark.parametrize('status', ['running', 'failed', 'diagnostic_complete'])
def test_replay_never_counts_towards_benchmark_even_with_full_episodes(tmp_path, status):
    path = receipt(tmp_path)
    report = json.loads(path.read_text())
    report.update(status=status, replay={'source_sha256': 'source'})
    path.write_text(json.dumps(report))
    result = summarize([path], ['task_a'])
    assert result['complete_cells'] == 0
    assert result['excluded'][0]['reason'] == 'fixed-instruction diagnostic replay'


@pytest.mark.parametrize('kind', ['renderer', 'simulator', 'source'])
def test_runtime_or_simulator_drift_cannot_be_aggregated(tmp_path, kind):
    first, second = receipt(tmp_path), receipt(tmp_path, shard=1)
    report = json.loads(second.read_text())
    if kind == 'renderer':
        report['denoiser'] = 'optix'
    elif kind == 'simulator':
        report['task_config'] = {'domain_randomization': {'random_background': True}}
    else:
        report['policy']['source_sha256'] = 'changed'
    second.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='mix'):
        summarize([first, second], ['task_a'])
