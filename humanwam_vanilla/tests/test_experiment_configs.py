from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from humanwam.config import load_config
from humanwam.datasets.lerobot.transforms.pose_frame import (
    DualArmRelativePoseTransform, DualArmWristBasisTransform,
)
from humanwam.datasets.lerobot.utils.rotation import quaternion_to_matrix

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


def test_recipe_composition_paths_overrides_and_missing_values(tmp_path, monkeypatch):
    (tmp_path / 'common.yaml').write_text(
        'nested: {keep: 1, replace: 2}\nitems: [1, 2]\n'
        'output_dir: ${paths.output_dir}\npaths: {output_dir: old}\n')
    (tmp_path / 'second.yaml').write_text('nested: {keep: 4, other: 5}\n')
    folder = tmp_path / 'series'
    folder.mkdir()
    recipe = folder / 'run.yaml'
    recipe.write_text('_base_: [../common.yaml, ../second.yaml]\n'
                      'nested: {replace: "???"}\nitems: [3]\n')
    paths = tmp_path / 'local.yaml'
    paths.write_text('output_dir: new\n')
    monkeypatch.chdir('/')
    cfg = load_config(recipe, paths, ['nested.other=7'])
    assert cfg.nested.keep == 4
    assert cfg.nested.other == 7
    assert OmegaConf.is_missing(cfg.nested, 'replace')
    assert cfg['items'] == [3]
    assert cfg.output_dir == 'new'
    assert '_base_' not in cfg
    assert load_config(tmp_path / 'common.yaml').nested.replace == 2


def test_flat_config_loading_is_unchanged():
    assert load_config(CONFIGS / 'robotwin.yaml') == OmegaConf.load(CONFIGS / 'robotwin.yaml')


@pytest.mark.parametrize('family,hours', [
    ('ia', [100, 500, 1000, 5000, 10000]),
    ('idm', [100, 500, 1000, 5000, 20500, 50000]),
])
def test_scaling_schedule_batches_and_corpus_identity(family, hours):
    batches = {100: 512, 500: 1024, 1000: 2048, 5000: 2048,
               10000: 4096, 20500: 4096, 50000: 8192}
    for h in hours:
        cfg = load_config(CONFIGS / f'scaling_{family}/pretrain_{h}h.yaml')
        assert cfg.batch_size * 8 * cfg.gradient_accumulation_steps == batches[h]
        assert cfg.experiment.global_batch_size == batches[h]
        assert cfg.max_steps == 100000
        assert cfg.warmup_ratio == .05
        assert cfg.save_every == 10000
        assert cfg.data.train.val_set_proportion == 0
        assert cfg.model.action_dit_config.separate_ego_action_tokenizer is False
        assert OmegaConf.is_missing(cfg.data.train, '_target_')
        assert cfg.corpus.nominal_hours == h
        if h == 10000:
            assert cfg.corpus.selection_file is None
        else:
            assert cfg.corpus.mapping_status == 'complete'
            assert cfg.corpus.selection_file == f'subsets/{h}h.parquet'
            assert cfg.corpus.config_name == f'{h}h'
        if h == 50000:
            assert cfg.data.wds.draws_per_episode == 8
            assert cfg.data.wds.sampling_reference_anchors == 135.3679531470708


@pytest.mark.parametrize('family', ['ia', 'idm'])
def test_exploration_posttraining_does_not_inherit_main_result(family):
    cfg = load_config(CONFIGS / f'scaling_{family}/posttrain.yaml')
    assert cfg.max_steps == 12000
    assert cfg.batch_size * 8 * cfg.gradient_accumulation_steps == 512
    assert cfg.data.train.val_set_proportion == 0
    assert cfg.model.action_dit_config.separate_ego_action_tokenizer is True
    assert cfg.model.video_vae_layout == 'pixel'
    assert [t._target_ for t in cfg.data.train.processor.train_transforms] == [
        'humanwam.datasets.lerobot.transforms.image.ToTensor', 'torchvision.transforms.Resize']
    transforms = cfg.data.train.processor.action_state_transforms
    assert all('.ego_alignment.' not in t._target_ and '.rotation.' not in t._target_
               for t in transforms)
    assert cfg.checkpoint_steps == [9000, 10000, 11000, 12000]


def test_matched_diversity_uses_distinct_indices_and_historical_schedules():
    cfg = load_config(CONFIGS / 'diversity/pretrain_matched_100h.yaml')
    assert cfg.corpus.family == 'matched-diversity'
    assert cfg.corpus.selection_file is None
    assert cfg.model.loss.component_weights.rotation == .5
    series = OmegaConf.load(CONFIGS / 'diversity/series.yaml')
    clean100 = next(r for r in series.selected_checkpoints if r.hours == 100 and r.setting == 'clean')
    schedule = load_config(CONFIGS / 'diversity' / clean100.posttrain_config)
    assert clean100.posttrain_step == 12000
    assert schedule.max_steps == 15000


@pytest.mark.parametrize('initialization', ['no_ego', '500h', '5000h'])
def test_low_demo_configs_require_actual_episode_selection(initialization):
    for count in [5, 10, 25]:
        cfg = load_config(CONFIGS / f'demonstration_efficiency/posttrain_{initialization}_{count}demos.yaml')
        assert cfg.experiment.demonstrations_per_task == count
        assert OmegaConf.is_missing(cfg.paths, 'robotwin_episode_exclusion_manifest')
        cfg.paths.robotwin_episode_exclusion_manifest = './episodes.json'
        assert cfg.data.train.episode_exclusion_manifest == './episodes.json'
        assert cfg.data.train.val_set_proportion == 0
        if initialization == 'no_ego':
            assert cfg.resume is None


def test_action_supervision_does_not_leak_into_posttraining():
    for rate in [0, 50, 100]:
        cfg = load_config(CONFIGS / f'action_supervision/pretrain_{rate:03d}percent.yaml')
        assert cfg.model.loss.lambda_action == (0 if rate == 0 else 1)
        if rate == 50:
            assert OmegaConf.is_missing(cfg.experiment, 'supervised_episode_selection')
    cfg = load_config(CONFIGS / 'action_supervision/posttrain.yaml')
    assert cfg.model.loss.lambda_action == 1


def test_unfinished_ablations_cannot_silently_use_parent_models_or_data():
    cfg = load_config(CONFIGS / 'attention/pretrain_mutual_relative.yaml')
    assert OmegaConf.is_missing(cfg.model, 'model_class')
    cfg = load_config(CONFIGS / 'multiview/posttrain_idm_conditioning.yaml')
    assert OmegaConf.is_missing(cfg.data, 'train')
    assert OmegaConf.is_missing(cfg.model, 'video_vae_layout')
    cfg = load_config(CONFIGS / 'multiview/midtrain_idm_conditioning.yaml')
    assert OmegaConf.is_missing(cfg, 'max_steps')
    assert OmegaConf.is_missing(cfg, 'gradient_accumulation_steps')
    cfg = load_config(CONFIGS / 'action_initialization/pretrain_interpolated.yaml')
    cfg.paths.model__action_dit_pretrained_path = './action-init.pt'
    assert cfg.model.action_dit_pretrained_path == './action-init.pt'


@pytest.mark.parametrize('batched', [False, True])
def test_relative_pose_uses_camera_axes_and_roundtrips(batched):
    torch.manual_seed(42)
    shape = (2, 5, 2, 8) if batched else (5, 2, 8)
    pose = torch.randn(shape)
    pose[..., 3:7] = torch.nn.functional.normalize(pose[..., 3:7], dim=-1)
    states = torch.randn(shape)
    states[..., 3:7] = torch.nn.functional.normalize(states[..., 3:7], dim=-1)
    actions, state = pose.flatten(-2), states.flatten(-2)
    batch = {'action': {'default': actions.clone()}, 'state': {'default': state.clone()}}
    transform = DualArmRelativePoseTransform({'action': ['default']})
    transformed = transform.forward(batch)['action']['default'].reshape(shape)
    anchor = states[..., :1, :, :]
    torch.testing.assert_close(transformed[..., :3], pose[..., :3] - anchor[..., :3])
    torch.testing.assert_close(quaternion_to_matrix(transformed[..., 3:7]),
        quaternion_to_matrix(pose[..., 3:7]) @ quaternion_to_matrix(anchor[..., 3:7]).transpose(-1, -2))
    restored = transform.backward(batch)['action']['default'].reshape(shape)
    torch.testing.assert_close(restored[..., :3], pose[..., :3])
    torch.testing.assert_close(restored[..., 7:], pose[..., 7:])
    torch.testing.assert_close(quaternion_to_matrix(restored[..., 3:7]),
                               quaternion_to_matrix(pose[..., 3:7]))
    torch.testing.assert_close(batch['state']['default'], state)


def test_idm_basis_matches_right_multiplication_without_origin_shift():
    cfg = load_config(CONFIGS / 'scaling_idm/posttrain.yaml')
    settings = OmegaConf.to_container(cfg.data.train.processor.action_state_transforms[1])
    settings.pop('_target_')
    transform = DualArmWristBasisTransform(**settings)
    torch.manual_seed(4)
    poses = torch.randn(7, 2, 8)
    poses[..., 3:7] = torch.nn.functional.normalize(poses[..., 3:7], dim=-1)
    batch = {key: {'default': poses.flatten(-2).clone()} for key in ['action', 'state']}
    out = transform.forward(batch)['action']['default'].reshape_as(poses)
    bases = torch.tensor([settings['basis_left'], settings['basis_right']])
    torch.testing.assert_close(out[..., :3], poses[..., :3])
    torch.testing.assert_close(quaternion_to_matrix(out[..., 3:7]),
                               quaternion_to_matrix(poses[..., 3:7]) @ bases)
    restored = transform.backward(batch)['action']['default'].reshape_as(poses)
    torch.testing.assert_close(quaternion_to_matrix(restored[..., 3:7]),
                               quaternion_to_matrix(poses[..., 3:7]))
