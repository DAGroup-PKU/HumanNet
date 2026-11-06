from pathlib import Path

import pytest
from omegaconf import OmegaConf
from humanwam.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    cfg = load_config(ROOT / 'configs/robotwin.yaml')
    # Config-only fixtures do not claim these input resources exist.
    cfg.paths = {k: f'/example/{k}' for k in cfg.paths}
    return cfg


def test_portable_training_defaults(config):
    assert config.data.train.val_set_proportion == 0.0
    assert config.model.action_dit_config.separate_ego_action_tokenizer is True
    assert len(config.data.train.dataset_dirs) == 50
    assert config.max_steps == 24000


def test_local_bindings_and_command_overrides(tmp_path):
    paths = tmp_path / 'paths.yaml'
    OmegaConf.save(OmegaConf.create({'output_dir': './custom-run', 'resume': './policy.pt'}), paths)
    cfg = load_config(ROOT / 'configs/robotwin.yaml', paths, ['batch_size=2', 'learning_rate=0.0002'])
    assert cfg.output_dir == './custom-run'
    assert cfg.resume == './policy.pt'
    assert cfg.batch_size == 2
    assert cfg.learning_rate == 0.0002
