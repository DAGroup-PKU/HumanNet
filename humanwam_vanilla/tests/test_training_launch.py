import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest

from humanwam.config import configure_launch, load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch_env(monkeypatch):
    for key in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT',
                'NNODES', 'NODE_RANK', 'ACCELERATE_USE_DEEPSPEED',
                'ACCELERATE_USE_FSDP', 'ACCELERATE_USE_CPU'):
        monkeypatch.delenv(key, raising=False)
    # Track variables configure_launch writes so they are restored after tests.
    monkeypatch.setenv('FASTWAM_MANUAL_FSDP', '0')
    monkeypatch.setenv('FSDP_SHARDING', 'SHARD_GRAD_OP')
    return monkeypatch


def test_single_gpu_resolves_sampler_and_batch(launch_env):
    cfg = load_config(ROOT / 'configs/robotwin.yaml', overrides=[
        'batch_size=2', 'gradient_accumulation_steps=3'])
    report = configure_launch(cfg)
    assert report['strategy'] == 'single'
    assert report['effective_batch_size'] == 6
    assert cfg.dataloader_rank_aware_strict is False
    assert os.environ['FASTWAM_MANUAL_FSDP'] == '0'
    assert configure_launch(cfg) == report


def test_torchrun_resolves_fsdp_and_preserves_full_corpus(launch_env):
    for key, value in dict(WORLD_SIZE='4', RANK='2', LOCAL_RANK='2',
                           MASTER_ADDR='localhost', MASTER_PORT='29500').items():
        launch_env.setenv(key, value)
    cfg = load_config(ROOT / 'configs/robotwin.yaml')
    report = configure_launch(cfg)
    assert report['strategy'] == 'manual_fsdp'
    assert report['effective_batch_size'] == 256
    assert cfg.dataloader_rank_aware_strict is True
    assert cfg.data.train.val_set_proportion == 0.0
    assert cfg.model.action_dit_config.separate_ego_action_tokenizer is True
    assert os.environ['FASTWAM_MANUAL_FSDP'] == '1'




def test_device_binding_uses_local_rank(launch_env):
    from humanwam.runtime import _resolve_train_device
    import torch

    launch_env.setattr(torch.cuda, 'is_available', lambda: True)
    launch_env.setattr(torch.cuda, 'device_count', lambda: 2)
    selected = []
    launch_env.setattr(torch.cuda, 'set_device', selected.append)
    launch_env.setenv('LOCAL_RANK', '1')
    assert _resolve_train_device() == 'cuda:1'
    assert selected == [1]


def test_actual_two_process_launcher_config_check(launch_env, tmp_path):
    """Launch real CPU workers through torchrun; never load a model or use GPUs."""
    cfg = load_config(ROOT / 'configs/robotwin.yaml')
    bindings = {}
    for key in cfg.paths:
        path = tmp_path / key
        if key in ('resume', 'data__train__pretrained_norm_stats',
                   'data__train__object_conditioned_instruction_manifest'):
            path.write_text('{}')
        else:
            path.mkdir()
            if key.startswith('data__train__dataset_dirs__'):
                (path / 'meta').mkdir()
                (path / 'meta/info.json').write_text(json.dumps({'total_episodes': 2}))
        bindings[key] = str(path)
    paths_file = tmp_path / 'paths.yaml'
    OmegaConf.save(OmegaConf.create(bindings), paths_file)
    env = dict(os.environ, NPROC_PER_NODE='2', PYTHON_BIN=sys.executable,
               CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', HF_HUB_OFFLINE='1')
    result = subprocess.run(
        ['bash', str(ROOT / 'scripts/train.sh'), '--paths', str(paths_file),
         '--print-config', 'batch_size=2', 'gradient_accumulation_steps=3'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert len(report['config']['data']['train']['dataset_dirs']) == 50
    assert report['config']['data']['train']['val_set_proportion'] == 0.0
    assert report['launch']['world_size'] == 2
    assert report['launch']['strategy'] == 'manual_fsdp'
    assert report['launch']['effective_batch_size'] == 12
    assert not any((tmp_path / 'output_dir').iterdir())
