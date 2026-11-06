import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from humanwam.robodojo_policy import prepare_images
from humanwam.datasets.lerobot.robot_video_dataset import _validate_norm_stats_provenance
from humanwam.datasets.lerobot.utils.rotation import quaternion_to_matrix
from humanwam.datasets.openwam_compact_augmentation import OpenWAMCompactAugmentation


def test_dojo_camera_layout_and_byte_quantization():
    cameras = {name: np.full((480, 640, 3), value, np.uint8)
               for name, value in zip(('head_camera', 'left_camera', 'right_camera'), (51, 102, 153))}
    image = prepare_images(cameras, device='cpu', dtype=torch.float32)[0]
    assert image.shape == (3, 384, 352)
    for row, col, value in ((100, 100, 51), (320, 80, 102), (320, 270, 153)):
        torch.testing.assert_close(image[:, row, col], torch.full((3,), value / 127.5 - 1))
    cameras['head_camera'] = cameras['head_camera'].astype(np.float32)
    with pytest.raises(ValueError, match='uint8'):
        prepare_images(cameras, device='cpu', dtype=torch.float32)


def test_dojo_world_camera_wrist_roundtrip():
    cfg = OmegaConf.load(Path(__file__).parents[1] / 'configs/robodojo.yaml')
    transforms = [instantiate(t) for t in cfg.data.train.processor.action_state_transforms]
    state = torch.tensor([[0.1, 0.2, 0.9, 1., 0., 0., 0., .3,
                           .4, .5, 1.1, -1., 0., 0., 0., .8]])
    batch = {'state': {'default': state.clone()}}
    for transform in transforms:
        batch = transform.forward(batch)
    transformed = batch['state']['default'].clone()
    # Same input rotations but different wrist definitions must produce different rotations.
    assert not torch.allclose(quaternion_to_matrix(transformed[:, 3:7]),
                              quaternion_to_matrix(transformed[:, 11:15]))
    for transform in reversed(transforms):
        batch = transform.backward(batch)
    restored = batch['state']['default']
    torch.testing.assert_close(restored[:, [0,1,2,7,8,9,10,15]], state[:, [0,1,2,7,8,9,10,15]])
    for start in (3, 11):
        torch.testing.assert_close(quaternion_to_matrix(restored[:, start:start+4]),
                                  quaternion_to_matrix(state[:, start:start+4]), atol=1e-6, rtol=1e-6)


def test_augmentation_shares_factors_over_time_and_changes_epoch():
    rng = np.random.default_rng(4)
    frame = torch.from_numpy(rng.integers(0, 256, (3, 1, 384, 352), dtype=np.uint8))
    video = frame.repeat(1, 3, 1, 1)
    wrapper = OpenWAMCompactAugmentation([{'video': video}], name='openwam_robodojo_compact',
        probability=1.0, brightness=.2, contrast=.2, saturation=.2, hue=0., sync_time=True, sync_cameras=True)
    first = wrapper[0]['video']
    assert first.dtype == torch.uint8
    assert torch.equal(first[:, 0], first[:, 1])
    assert torch.equal(first, wrapper[0]['video'])
    wrapper.set_epoch(1)
    assert not torch.equal(first, wrapper[0]['video'])
    assert torch.equal(video, frame.repeat(1, 3, 1, 1))


def test_provenance_accepts_namespace_rename_but_rejects_holdout_and_changed_transform(tmp_path):
    root = str(tmp_path.resolve())
    transform = {'_target_': 'fastwam.test.Transform', 'cam2world': [[0.8660254037844386]]}
    stats = {'num_episodes': 3500, 'num_transition': 1859602, 'metadata': {
        'corpus': {'dataset_dirs': [root], 'dataset_dirs_sha256': hashlib.sha256(root.encode()).hexdigest()},
        'resolved_shape_meta': {}, 'action_state_transforms': [transform], 'norm_stats_weighting': 'loss'}}
    current = copy.deepcopy(transform); current['_target_'] = 'humanwam.test.Transform'
    kwargs = dict(stats_path=tmp_path / 'stats.json', dataset_dirs=[root], episode_exclusion_manifest=None,
                  shape_meta={}, action_state_transforms=[current], norm_stats_weighting='loss',
                  expected_episodes=3500, expected_frames=1859602)
    _validate_norm_stats_provenance(stats, **kwargs)
    bad = copy.deepcopy(stats); bad['num_episodes'] = 3465
    with pytest.raises(ValueError, match='num_episodes'):
        _validate_norm_stats_provenance(bad, **kwargs)
    bad = copy.deepcopy(stats); bad['metadata']['action_state_transforms'][0]['cam2world'][0][0] = .866
    with pytest.raises(ValueError, match='action_state_transforms'):
        _validate_norm_stats_provenance(bad, **kwargs)


def test_cache_signature_matches_frozen_source_after_namespace_rename():
    from torchvision.transforms import Resize
    from humanwam.datasets.lerobot.transforms.image import ToTensor
    from humanwam.datasets.lerobot.merged_video_cache import image_transforms_signature
    cameras = ['observation.images.cam_high', 'observation.images.cam_left_wrist',
               'observation.images.cam_right_wrist']
    transforms = [ToTensor(), Resize([256, 352], antialias=True)]
    expected = '893dc52001fe49a195e085fd62ad836388e175c2c72c42fef5d5ac915f096583'
    assert image_transforms_signature(transforms, cameras, historical_namespace=True) == expected
    transforms[1] = Resize([256, 352], antialias=False)
    assert image_transforms_signature(transforms, cameras, historical_namespace=True) != expected


def test_deployment_preserves_pose_gripper_order_and_replan_horizon():
    from humanwam.robodojo_policy import RoboDojoDeployment

    class Policy:
        def predict(self, cameras, state, instruction):
            np.testing.assert_array_equal(state, np.arange(16, dtype=np.float32))
            assert list(cameras) == ['head_camera', 'left_camera', 'right_camera']
            assert instruction == 'stack blocks'
            return np.arange(32 * 16, dtype=np.float32).reshape(32, 16)

        def reset(self):
            pass

    adapter = RoboDojoDeployment(Policy())
    with pytest.raises(ValueError, match='update_obs'):
        adapter.get_action()
    state = {k: v for k, v in zip(adapter.STATE_KEYS,
             (np.arange(7), np.array([7]), np.arange(8, 15), np.array([15])))}
    obs = {'state': state, 'instruction': ['stack blocks'],
           'vision': {k: {'color': np.zeros((2, 2, 3), np.uint8)} for k in
                      ('cam_head', 'cam_left_wrist', 'cam_right_wrist')}}
    adapter.update_obs(obs)
    actions = adapter.get_action()
    assert len(actions) == 24
    np.testing.assert_array_equal(actions[1]['right_ee_pose'], np.arange(24, 31))
    np.testing.assert_array_equal(actions[1]['left_ee_joint_state'], [23])
    obs['state']['left_ee_pose'] = np.zeros(6)  # Joint-space dimensions must not pass.
    with pytest.raises(ValueError, match='link6'):
        adapter.get_action()
    adapter.reset()
    with pytest.raises(ValueError, match='update_obs'):
        adapter.get_action()
