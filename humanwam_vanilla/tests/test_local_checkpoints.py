"""Local checkpoint persistence after removing remote publishing backends."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from humanwam.trainer import Wan22Trainer
from humanwam.utils.checkpoints import prune_training_states


class TinyPolicy(torch.nn.Linear):
    def save_checkpoint(self, path, optimizer, step):
        torch.save({'model': self.state_dict(), 'step': step}, path)


def make_trainer(tmp_path, *, fail_state_save=False):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = TinyPolicy(2, 2)
    trainer._is_deepspeed = False
    trainer.lora_enabled = False
    trainer.global_step = 1000
    trainer.epoch = 2
    trainer.batch_in_epoch = 3
    trainer.weights_dir = str(tmp_path / 'weights')
    trainer.state_dir = str(tmp_path / 'state')
    Path(trainer.weights_dir).mkdir()
    Path(trainer.state_dir).mkdir()
    trainer.checkpoint_state_keep_last = 1

    def save_state(output_dir):
        assert (Path(trainer.weights_dir) / 'step_001000.pt').is_file()
        if fail_state_save:
            raise OSError('state write failed')
        torch.save({'optimizer': {}}, Path(output_dir) / 'optimizer.pt')

    trainer.accelerator = SimpleNamespace(
        is_main_process=True, wait_for_everyone=lambda: None, save_state=save_state,
    )
    return trainer


def test_training_saves_readable_local_weights_and_state(tmp_path):
    trainer = make_trainer(tmp_path)
    old_state = Path(trainer.state_dir) / 'step_000500'
    old_state.mkdir()
    (old_state / 'trainer_state.json').write_text('{}')
    old_weight = Path(trainer.weights_dir) / 'step_000500.pt'
    old_weight.write_bytes(b'previous weights')

    result = trainer.save_checkpoint()
    checkpoint = torch.load(result['weights_path'], weights_only=True)
    assert checkpoint['step'] == 1000
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(checkpoint['model'][key], value)
    state = Path(result['state_path'])
    assert (state / 'optimizer.pt').is_file()
    assert json.loads((state / 'trainer_state.json').read_text()) == {
        'global_step': 1000, 'epoch': 2, 'batch_in_epoch': 3,
    }
    assert not old_state.exists()
    assert old_weight.read_bytes() == b'previous weights'


def test_failed_state_save_preserves_previous_state_and_new_weights(tmp_path):
    trainer = make_trainer(tmp_path, fail_state_save=True)
    previous = Path(trainer.state_dir) / 'step_000500'
    previous.mkdir()
    (previous / 'trainer_state.json').write_text('{}')
    with pytest.raises(OSError, match='state write failed'):
        trainer.save_checkpoint()
    assert previous.is_dir()
    assert (Path(trainer.weights_dir) / 'step_001000.pt').is_file()
    assert not (Path(trainer.state_dir) / 'step_001000/trainer_state.json').exists()


def test_retention_ignores_incomplete_states_and_unrelated_directories(tmp_path):
    for name in ['step_000100', 'step_000200', 'step_000300', 'notes']:
        (tmp_path / name).mkdir()
    for name in ['step_000100', 'step_000200', 'notes']:
        (tmp_path / name / 'trainer_state.json').write_text('{}')
    assert prune_training_states(tmp_path, 1) == [tmp_path / 'step_000100']
    assert all((tmp_path / name).is_dir() for name in ['step_000200', 'step_000300', 'notes'])
