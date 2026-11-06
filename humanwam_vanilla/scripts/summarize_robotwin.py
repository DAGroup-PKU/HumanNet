#!/usr/bin/env python3
"""Collect RoboTwin's 50-task, eight-shard results without counting partial cells."""
import argparse
import json
from pathlib import Path


SHARD_EPISODES = (7, 7, 6, 6, 6, 6, 6, 6)


def summarize(paths, task_names):
    cells, excluded, fingerprints = {}, [], set()
    phase_settings = {'clean': set(), 'random': set()}
    for path in sorted(set(Path(p).resolve() for p in paths)):
        report = json.loads(path.read_text())
        if report.get('replay'):
            excluded.append(dict(path=str(path), reason='fixed-instruction diagnostic replay'))
            continue
        if report.get('max_chunks') is not None or report.get('status') == 'diagnostic_complete':
            excluded.append(dict(path=str(path), reason='truncated diagnostic'))
            continue
        shard, phase = report.get('seed'), report.get('phase')
        if type(shard) is not int or not 0 <= shard < 8 or phase not in ('clean', 'random'):
            raise ValueError(f'Invalid phase/shard: {path}')
        expected = SHARD_EPISODES[shard]
        if (report.get('episodes_per_task') != expected or report.get('steps') != 10
                or report.get('replan_steps') != 24 or report.get('instruction_type') != 'unseen'):
            raise ValueError(f'Non-paper evaluation protocol: {path}')
        policy = report.get('policy')
        if not policy:
            excluded.append(dict(path=str(path), reason='policy not initialized'))
            continue
        if policy.get('action_horizon') != 32:
            raise ValueError(f'Unexpected action horizon: {path}')
        fingerprint = json.dumps(dict(
            checkpoint=policy['checkpoint_sha256'], statistics=policy['stats_sha256'],
            model=policy.get('model_config'), processor=policy.get('processor_config'),
            policy_versions=policy.get('versions'), policy_source=policy.get('source_sha256'),
            gpu=policy.get('gpu'),
            simulator_versions=report.get('simulator', {}).get('versions'),
            simulator_source=report.get('simulator', {}).get('source_sha256'),
            denoiser=report.get('denoiser'),
            oidn_libraries=sorted(Path(p).name for p in report.get('oidn_device', {}).get('loaded_libraries', [])),
        ), sort_keys=True)
        included = False
        for task in report['tasks']:
            name, episodes = task['task'], task['episodes']
            if name not in task_names:
                raise ValueError(f'Unexpected task {name}: {path}')
            if len(episodes) < expected:
                excluded.append(dict(path=str(path), task=name, reason='incomplete task shard',
                                     completed_episodes=len(episodes), expected_episodes=expected))
                continue
            if len(episodes) != expected:
                raise ValueError(f'Too many episodes for {name}: {path}')
            seeds = [e['seed'] for e in episodes]
            if len(set(seeds)) != expected or min(seeds) < 100000 * (1 + shard):
                raise ValueError(f'Invalid or repeated scene seeds for {name}: {path}')
            for episode in episodes:
                if (episode.get('truncated') is not False or type(episode.get('success')) is not bool
                        or not episode.get('instruction') or episode.get('action_calls', 0) <= 0
                        or episode.get('step_limit', 0) <= 0):
                    raise ValueError(f'Incomplete episode evidence for {name}: {path}')
                if not episode['success'] and episode['action_calls'] < episode['step_limit']:
                    raise ValueError(f'Failed episode stopped before the task horizon: {path}')
            key = (phase, name, shard)
            if key in cells:
                raise ValueError(f'Duplicate cell {key}: {path} and {cells[key]["path"]}')
            cells[key] = dict(path=str(path), successes=sum(e['success'] for e in episodes),
                              episodes=expected, scene_seeds=seeds)
            included = True
        if included:
            fingerprints.add(fingerprint)
            simulation = dict(report.get('task_config', {}))
            for key in ('task_name', 'left_robot_file', 'right_robot_file'):
                simulation.pop(key, None)
            phase_settings[phase].add(json.dumps(simulation, sort_keys=True))
    if len(fingerprints) > 1:
        raise ValueError('Results mix checkpoints, statistics, model settings, source, or runtime versions')
    if any(len(settings) > 1 for settings in phase_settings.values()):
        raise ValueError('Results mix simulator settings within a phase')
    phases, missing = {}, []
    for phase in ('clean', 'random'):
        rows = []
        for name in task_names:
            present = []
            for shard in range(8):
                cell = cells.get((phase, name, shard))
                if cell is None:
                    missing.append(dict(phase=phase, task=name, shard=shard))
                else:
                    present.append(cell)
            successes = sum(c['successes'] for c in present)
            episodes = sum(c['episodes'] for c in present)
            rows.append(dict(task=name, successes=successes, episodes=episodes, expected_episodes=50,
                             success_rate_pct=100 * successes / episodes if episodes == 50 else None))
        successes = sum(r['successes'] for r in rows)
        episodes = sum(r['episodes'] for r in rows)
        expected = 50 * len(task_names)
        phases[phase] = dict(successes=successes, episodes=episodes, expected_episodes=expected,
                             success_rate_pct=100 * successes / episodes if episodes == expected else None,
                             tasks=rows)
    return dict(coverage_complete=not missing, benchmark_parity_verified=False,
                complete_cells=len(cells), expected_cells=16 * len(task_names), phases=phases,
                accepted_cells=[dict(phase=phase, task=task, shard=shard, **cell)
                                for (phase, task, shard), cell in sorted(cells.items())],
                missing_cells=missing, excluded=excluded,
                configuration=json.loads(next(iter(fingerprints))) if fingerprints else None,
                note='Coverage establishes episode accounting, not numerical parity with the historical renderer or instructions.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='+', type=Path, help='result.json files or directories to search')
    parser.add_argument('--task-file', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'evaluation/robotwin/tasks-50.txt')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    names = [line.split('#', 1)[0].strip() for line in args.task_file.read_text().splitlines()]
    names = [name for name in names if name]
    if not names or len(set(names)) != len(names):
        parser.error('Task list must contain unique task names')
    paths = []
    for path in args.results:
        if not path.exists():
            parser.error(f'Result path does not exist: {path}')
        paths.extend(path.rglob('result.json') if path.is_dir() else [path])
    if not paths:
        parser.error('No result.json files found')
    if args.output.resolve() in {path.resolve() for path in paths}:
        parser.error('Output must not overwrite an input result')
    result = summarize(paths, names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(coverage_complete=result['coverage_complete'],
                         complete_cells=result['complete_cells'], expected_cells=result['expected_cells'],
                         phases={k: {key: value for key, value in v.items() if key != 'tasks'}
                                 for k, v in result['phases'].items()}), indent=2))


if __name__ == '__main__':
    main()
