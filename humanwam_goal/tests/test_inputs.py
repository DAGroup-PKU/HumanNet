import json

import numpy as np
import pytest
import torch

from goalwam.backends.native.model import patchify, unpatchify
from goalwam.processing.images import goalwam_canvas, select_goalwam_history
from goalwam.processing.observation import ObservationProcessor, ProcessorConfig
from goalwam.processing.robot import RobotProcessor


def observation():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    return {
        "instruction": "Move to the goal.",
        "images": {"head": image},
        "goal_images": {"head": image},
        "observation_timestamp": 3.0,
        "history": [],
        "normalized_state": [0.0] * 49,
        "domain_id": 0,
    }


def test_patch_channel_order():
    value = torch.arange(8).reshape(1, 2, 1, 2, 2)
    packed = patchify(value, 2)
    assert packed.flatten().tolist() == [0, 4, 1, 5, 2, 6, 3, 7]
    assert torch.equal(unpatchify(packed, value.shape, 2), value)
    odd = torch.arange(30).reshape(1, 2, 1, 3, 5)
    assert torch.equal(unpatchify(patchify(odd, 2), odd.shape, 2), odd)


def test_prefix_and_invalid_channels():
    source = observation()
    source["normalized_prefix"] = np.ones((3, 49), dtype=np.float32)
    result = ObservationProcessor(ProcessorConfig(camera_sizes=((32, 32),)))(source)
    assert result["action_noisy_mask"][0].tolist() == [False] * 4 + [True] * 45
    assert not result["action"][..., 42:].any()
    assert torch.equal(result["action"][0, 1:4, :42], torch.ones(3, 42))


def test_prefix_limit():
    source = observation()
    source["normalized_prefix"] = np.ones((8, 49), dtype=np.float32)
    with pytest.raises(ValueError, match="prefix"):
        ObservationProcessor(ProcessorConfig(camera_sizes=((32, 32),)))(source)


def test_history_is_causal():
    captures = [{"timestamp": 1.0}, {"timestamp": 2.49}, {"timestamp": 2.51}]
    selected = select_goalwam_history(captures, 3.0, (0.5, 2.0), 0.1)
    assert [capture["timestamp"] for capture, _ in selected] == [2.49, 1.0]
    with pytest.raises(ValueError, match="earlier"):
        select_goalwam_history([{"timestamp": 3.0}], 3.0, (0.5,), 0.1)


def test_missing_wrists_keep_canvas():
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    canvas = goalwam_canvas({"head": image}, cell_size=(32, 32), cameras=("head", "left", "right"))
    assert canvas.shape == (3, 64, 32)
    assert canvas[:, 8:40].eq(255).all()
    assert canvas[:, 40:].eq(0).all()


def robot_config():
    mask = [False] * 23 + [True] * 26
    stats = {"mode": "bounds", "low": [-1.0] * 49, "high": [1.0] * 49, "clip": 3}
    return {
        "domain_id": -1,
        "state_coordinate": "head_camera",
        "action_coordinate": "head_camera",
        "gripper_space": "gap_m",
        "open_aperture_m": 0.08,
        "aperture_reference_width_m": 0.1,
        "joint_channels": False,
        "state_valid_mask": mask,
        "action_valid_mask": mask,
        "normalization": {"state": stats, "action": stats},
    }


def test_physical_rtc_uses_current_anchor(tmp_path):
    path = tmp_path / "robot.json"
    path.write_text(json.dumps(robot_config()))
    processor = RobotProcessor(path)
    source = observation()
    source.pop("normalized_state")
    state = np.zeros(49, dtype=np.float32)
    state[[32, 40, 48]] = 1
    state[26] = 0.2
    state[33] = state[41] = 50
    commands = np.repeat(state[None], 4, axis=0)
    commands[:, 26] += 0.1
    source.update(state=state, prev_chunk=commands, s=2)
    prepared, context = processor.preprocess(source, 7)
    assert prepared["normalized_prefix"].shape == (2, 49)
    output = processor.postprocess(prepared["normalized_prefix"], (context[0], context[1][:2]))
    np.testing.assert_allclose(output["actions"][:, 26], 0.3, atol=1e-6)
    assert not output["actions"][:, :23].any()
    source["state"] = state.copy()
    source["state"][26] += 0.05
    changed, _ = processor.preprocess(source, 7)
    assert torch.all(changed["normalized_prefix"][:, 26] < prepared["normalized_prefix"][:, 26])


def test_physical_masks_cannot_override_checkpoint(tmp_path):
    path = tmp_path / "robot.json"
    path.write_text(json.dumps(robot_config()))
    processor = RobotProcessor(path)
    source = observation()
    source.pop("normalized_state")
    source.update(state=[0.0] * 49, action_valid_mask=[True] * 49)
    with pytest.raises(ValueError, match="disagrees"):
        processor.preprocess(source, 7)
