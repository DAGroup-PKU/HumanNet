"""Validate a fixed-instruction diagnostic replay from an evaluator receipt."""
import hashlib
import json
from pathlib import Path


POLICY_KEYS = ('checkpoint_sha256', 'stats_sha256', 'model_config', 'processor_config',
               'action_horizon', 'replan_steps', 'steps')


def load_replay(path, task, seeds=None):
    path = Path(path).resolve()
    raw = path.read_bytes()
    source = json.loads(raw)
    if (source.get('phase') not in ('clean', 'random')
            or type(source.get('seed')) is not int or not 0 <= source['seed'] <= 7
            or source.get('instruction_type') != 'unseen'
            or source.get('max_chunks') is not None or source.get('replay')):
        raise ValueError('Replay requires an original, untruncated unseen-instruction receipt')
    policy = source.get('policy', {})
    if not policy.get('ready') or any(key not in policy for key in POLICY_KEYS):
        raise ValueError('Replay source is missing policy provenance')
    if policy['action_horizon'] != 32 or source.get('replan_steps') != 24:
        raise ValueError('Replay requires the 32-action / 24-step execution protocol')
    if source.get('steps') != policy['steps'] or source.get('replan_steps') != policy['replan_steps']:
        raise ValueError('Replay source has inconsistent policy settings')
    matches = [row for row in source.get('tasks', []) if row['task'] == task]
    if len(matches) != 1 or not task.isidentifier():
        raise ValueError('Replay task must identify exactly one source task')
    episodes = matches[0]['episodes']
    by_seed = {}
    for index, episode in enumerate(episodes):
        seed = episode.get('seed')
        if type(seed) is not int or seed < 100000 * (1 + source['seed']) or seed in by_seed:
            raise ValueError('Replay source has invalid or duplicate scene seeds')
        by_seed[seed] = dict(episode, original_episode_index=index)
    selected = list(by_seed) if seeds is None else seeds
    if not selected or len(set(selected)) != len(selected) or any(seed not in by_seed for seed in selected):
        raise ValueError('Select distinct, completed scene seeds from the source task')
    chosen = [by_seed[seed] for seed in selected]
    for episode in chosen:
        if (episode.get('truncated') is not False or type(episode.get('success')) is not bool
                or not isinstance(episode.get('instruction'), str) or not episode['instruction'].strip()
                or episode.get('action_calls', 0) <= 0 or episode.get('step_limit', 0) <= 0):
            raise ValueError('Replay source episode is incomplete')
        if not episode['success'] and episode['action_calls'] < episode['step_limit']:
            raise ValueError('Replay source failure stopped before the task horizon')
    return dict(source_path=str(path), source_sha256=hashlib.sha256(raw).hexdigest(),
                task=task, phase=source['phase'], shard=source['seed'], steps=source['steps'],
                description_count=source['episodes_per_task'], episodes=chosen,
                expected_policy={key: policy[key] for key in POLICY_KEYS},
                mode='fixed-instruction diagnostic; excluded from benchmark aggregation')


def verify_replay_policy(replay, ready):
    changed = [key for key, value in replay['expected_policy'].items() if ready.get(key) != value]
    if changed:
        raise ValueError('Replay policy does not match its source: ' + ', '.join(changed))
