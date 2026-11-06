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


@pytest.mark.parametrize('processes', [1, 2])
def test_actual_launcher_config_check(launch_env, tmp_path, processes):
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
    env = dict(os.environ, NPROC_PER_NODE=str(processes), PYTHON_BIN=sys.executable,
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
    assert report['launch']['world_size'] == processes
    assert report['launch']['strategy'] == ('manual_fsdp' if processes > 1 else 'single')
    assert report['launch']['effective_batch_size'] == 6 * processes
    assert not any((tmp_path / 'output_dir').iterdir())


def test_stage_initialization_resets_step_without_changing_recipe(tmp_path):
    from humanwam.config import set_training_checkpoint
    weights = tmp_path / 'previous stage.pt'
    weights.touch()
    cfg = load_config(ROOT / 'configs/robodojo.yaml', overrides=['resume_step=45000'])
    set_training_checkpoint(cfg, init_checkpoint=weights)
    assert cfg.resume == cfg.paths.resume == str(weights.resolve())
    assert cfg.resume_step == 0
    assert cfg.max_steps == 60000
    assert cfg.data.train.val_set_proportion == 0.0
    with pytest.raises(ValueError, match='weights file'):
        set_training_checkpoint(cfg, init_checkpoint=tmp_path)
    with pytest.raises(ValueError, match='training-state directory'):
        set_training_checkpoint(cfg, resume=weights)
    state_dir = tmp_path / 'checkpoint_state'
    state_dir.mkdir()
    set_training_checkpoint(cfg, resume=state_dir)
    assert cfg.resume == str(state_dir.resolve())
    with pytest.raises(ValueError, match='only one'):
        set_training_checkpoint(cfg, init_checkpoint=weights, resume=state_dir)


def test_main_stage_recipes_keep_lineage_and_sampler():
    from humanwam.config import training_config_gaps
    pretrain = load_config(ROOT / 'configs/pretrain.yaml')
    assert pretrain.experiment.stage == 'pretrain'
    assert pretrain.max_steps == 100000
    assert pretrain.data.wds.draws_per_episode == 8
    assert pretrain.data.wds.sampling_reference_anchors == 135.3679531470708
    midtrain = load_config(ROOT / 'configs/midtrain.yaml')
    assert midtrain.experiment.pretrain_step == 40000
    assert midtrain.max_steps == 10000
    assert midtrain.model.latent_tshape_future_wrist_loss_weight == 0.0
    for cfg in (pretrain, midtrain):
        assert cfg.data.train.val_set_proportion == 0.0
        assert 'data.train._target_' in training_config_gaps(cfg)
        assert 'data.train.processor' in training_config_gaps(cfg)


@pytest.mark.parametrize('stage', ['pretrain', 'midtrain'])
def test_unfinished_stage_fails_before_model_loading(launch_env, tmp_path, stage):
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/train.py'), '--config',
         str(ROOT / 'configs' / f'{stage}.yaml')],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1'))
    assert result.returncode == 2
    assert 'Configuration is incomplete' in result.stderr
    assert 'data.train._target_' in result.stderr
    assert 'Traceback' not in result.stderr
    assert not (tmp_path / 'runs').exists()


@pytest.mark.parametrize('flag', ['--init-checkpoint', '--resume'])
def test_actual_launcher_checkpoint_flags(launch_env, tmp_path, flag):
    cfg = load_config(ROOT / 'configs/robodojo.yaml')
    # Config-only test: no dataset/GPU/model access, but use a real file/directory.
    cfg.paths = {key: str(tmp_path / key) for key in cfg.paths}
    cfg.resume_step = 123
    config_file = tmp_path / 'stage config.yaml'
    OmegaConf.save(cfg, config_file)
    checkpoint = tmp_path / 'checkpoint with spaces'
    if flag == '--init-checkpoint':
        checkpoint.touch()
    else:
        checkpoint.mkdir()
    env = dict(os.environ, NPROC_PER_NODE='1', PYTHON_BIN=sys.executable,
               CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1')
    result = subprocess.run(
        ['bash', str(ROOT / 'scripts/train.sh'), '--config', str(config_file),
         flag, str(checkpoint), '--print-config'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    config = json.loads(result.stdout)['config']
    assert config['resume'] == config['paths']['resume'] == str(checkpoint)
    assert config['resume_step'] == 0
    assert config['max_steps'] == 60000


def test_unused_binding_catalog_does_not_block_a_resolved_recipe():
    from humanwam.config import training_config_gaps
    cfg = OmegaConf.create({'resume': None, 'data': {'train': {'val_set_proportion': 0.0}},
                           'paths': {'unused': '???'}, 'corpus_indices': {'other_series': '???'}})
    assert training_config_gaps(cfg) == []
    assert OmegaConf.to_container(cfg, resolve=True)['resume'] is None
    cfg.data.train.dataset_dirs = '${paths.unused}'
    from omegaconf.errors import InterpolationToMissingValueError
    with pytest.raises(InterpolationToMissingValueError):
        OmegaConf.to_container(cfg, resolve=True)


def test_checkpoint_override_conflicts_are_rejected(launch_env, tmp_path):
    weights = tmp_path / 'weights.pt'
    weights.touch()
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/train.py'),
                             '--init-checkpoint', str(weights), 'resume_step=100'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert 'Do not combine checkpoint flags' in result.stderr
