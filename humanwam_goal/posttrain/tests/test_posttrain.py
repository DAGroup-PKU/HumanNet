"""Unit tests for `goalwam_train`: conversion, normalization, goal cache, and training windows."""

import io
import json
import random
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from goalwam.processing.robot import RobotProcessor
from PIL import Image

from goalwam_train.convert import convert_archive, decode_hdf5_jpeg
from goalwam_train.data import Episodes, VideoReader, model_state_and_actions, sample_recorded_history
from goalwam_train.goal_cache import IMAGE_SIZE, SCHEMA, TrainGoalCache, sha256_bytes, verify_goal_cache
from goalwam_train.normalization import ARM_EEF, GRIPPERS, compute_robot_config, read_shared_eef_bounds


def _jpeg_bytes(rgb, *, xpl_marker=False):
    buf = io.BytesIO()
    if xpl_marker:
        Image.fromarray(rgb).save(buf, format="JPEG", quality=98, comment=b"XPL-RGB1")
    else:
        # Simulate RoboTwin's legacy `cv2.imencode('.jpg', rgb)` (stores BGR in JPEG).
        Image.fromarray(rgb[..., ::-1]).save(buf, format="JPEG", quality=98)
    return buf.getvalue()


def _make_raw_zip(zip_path, *, lengths=(10, 60)):
    rng = np.random.default_rng(7)
    cam2world = np.eye(4, dtype=np.float64)
    cam2world[:3, 3] = [0.0, -0.4, 1.2]
    with zipfile.ZipFile(zip_path, "w") as zf:
        for ep_idx, n in enumerate(lengths):
            raw = io.BytesIO()
            with h5py.File(raw, "w") as h5:
                h5.create_dataset("observation/head_camera/cam2world_gl", data=np.repeat(cam2world[None], n, 0))
                h5.create_dataset(
                    "joint_action/left_arm",
                    data=np.linspace(-0.5, 0.5 + 0.1 * ep_idx, n * 6, dtype=np.float32).reshape(n, 6),
                )
                h5.create_dataset(
                    "joint_action/right_arm",
                    data=np.linspace(0.2, 1.2, n * 6, dtype=np.float32).reshape(n, 6),
                )
                grip = np.linspace(0.1, 0.9, n, dtype=np.float32)
                for key in (
                    "joint_action/left_gripper",
                    "joint_action/right_gripper",
                    "endpose/left_gripper",
                    "endpose/right_gripper",
                ):
                    h5.create_dataset(key, data=grip)
                pose_l = np.tile(np.array([0.1, 0.2, 0.8, 1.0, 0.0, 0.0, 0.0], dtype=np.float64), (n, 1))
                pose_r = np.tile(np.array([-0.1, 0.2, 0.8, 1.0, 0.0, 0.0, 0.0], dtype=np.float64), (n, 1))
                pose_l[:, 0] += np.linspace(0.0, 0.05, n)
                pose_r[:, 1] -= np.linspace(0.0, 0.04, n)
                h5.create_dataset("endpose/left_endpose", data=pose_l)
                h5.create_dataset("endpose/right_endpose", data=pose_r)
                for cam in ("head_camera", "left_camera", "right_camera"):
                    frames = [
                        _jpeg_bytes(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
                        for _ in range(n)
                    ]
                    h5.create_dataset(f"observation/{cam}/rgb", data=np.asarray(frames, dtype="S"))
            zf.writestr(f"demo/data/episode{ep_idx}.hdf5", raw.getvalue())
            zf.writestr(
                f"demo/instructions/episode{ep_idx}.json",
                json.dumps({"seen": [f"Move the block episode {ep_idx}."]}),
            )


def _make_pretrained_checkpoint(root):
    root.mkdir(parents=True, exist_ok=True)
    state_mask = np.zeros(49, dtype=bool)
    state_mask[26:49] = True
    action_mask = np.zeros(49, dtype=bool)
    action_mask[26:42] = True
    s_low, s_high = [-5.0] * 49, [5.0] * 49
    a_low, a_high = [-2.5] * 49, [2.5] * 49
    for idx in GRIPPERS:
        s_low[idx], s_high[idx] = 0.0, 0.12
        a_low[idx], a_high[idx] = 0.0, 0.12
    robot = {
        "domain_id": -1,
        "state_coordinate": "head_camera",
        "action_coordinate": "head_camera",
        "gripper_space": "gap_m",
        "open_aperture_m": 0.1,
        "aperture_reference_width_m": 0.12,
        "joint_channels": False,
        "state_valid_mask": state_mask.tolist(),
        "action_valid_mask": action_mask.tolist(),
        "normalization": {
            "state": {"mode": "bounds", "low": s_low, "high": s_high},
            "action": {"mode": "bounds", "low": a_low, "high": a_high},
        },
        "instruction_prefix": "",
    }
    (root / "robot.json").write_text(json.dumps(robot))
    # Ensure `goalwam` RobotProcessor accepts the zero-shot pretrained robot.json contract.
    RobotProcessor(root / "robot.json")
    return root


def test_jpeg_decode_legacy_and_xpl():
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[:, :, 0] = 220  # bright red
    rgb[:, :, 2] = 20   # dark blue
    legacy = decode_hdf5_jpeg(_jpeg_bytes(rgb, xpl_marker=False))
    marked = decode_hdf5_jpeg(_jpeg_bytes(rgb, xpl_marker=True))
    assert legacy[:, :, 0].mean() > 200 and legacy[:, :, 2].mean() < 40
    assert marked[:, :, 0].mean() > 200 and marked[:, :, 2].mean() < 40


def test_convert_and_clean_normalization(tmp_path):
    raw_zip = tmp_path / "raw/adjust_bottle/aloha-agilex_clean_2.zip"
    raw_zip.parent.mkdir(parents=True)
    _make_raw_zip(raw_zip, lengths=(10, 60))

    clean_dir = tmp_path / "lerobot/adjust_bottle/aloha-agilex_clean_2"
    res = convert_archive(("adjust_bottle", raw_zip, clean_dir, False))
    assert res["status"] == "converted" and res["episodes"] == 2 and res["frames"] == 68

    # Place a randomized sibling directory to ensure discovery ignores it.
    rand_dir = tmp_path / "lerobot/adjust_bottle/aloha-agilex_randomized_2"
    convert_archive(("adjust_bottle", raw_zip, rand_dir, False))
    ep_index = Episodes(tmp_path / "lerobot")
    assert len(ep_index.datasets) == 1 and len(ep_index.episodes) == 2
    reader = VideoReader(clean_dir / "videos/observation.images.cam_high/chunk-000/file-000.mp4", 30.0)
    decoded_frames = reader.read([0, 8, 9, 67])
    assert len(decoded_frames) == 4 and decoded_frames[0].shape == (240, 320, 3)
    reader.container.close()

    pretrained = _make_pretrained_checkpoint(tmp_path / "pretrained")
    robot = compute_robot_config(tmp_path / "lerobot", pretrained, horizon=48, workers=1)
    s_low = np.asarray(robot["normalization"]["state"]["low"])
    s_high = np.asarray(robot["normalization"]["state"]["high"])
    a_low = np.asarray(robot["normalization"]["action"]["low"])
    a_high = np.asarray(robot["normalization"]["action"]["high"])

    # Shared EEF bounds come strictly from the pretrained checkpoint, never from RoboTwin data.
    assert np.allclose(s_low[ARM_EEF], -5.0) and np.allclose(s_high[ARM_EEF], 5.0)
    assert np.allclose(a_low[ARM_EEF], -2.5) and np.allclose(a_high[ARM_EEF], 2.5)
    # Grippers are fixed to [0, 1] command openness.
    assert np.allclose(s_low[GRIPPERS], 0.0) and np.allclose(s_high[GRIPPERS], 1.0)
    assert np.allclose(a_low[GRIPPERS], 0.0) and np.allclose(a_high[GRIPPERS], 1.0)
    # Joint bounds come from clean data (including the 9-frame short episode clamped to end).
    assert s_high[0] > s_low[0] and a_high[0] > a_low[0]

    robot_path = tmp_path / "robot.json"
    robot_path.write_text(json.dumps(robot))
    RobotProcessor(robot_path)
    eef_rt = read_shared_eef_bounds(robot_path)
    assert np.allclose(eef_rt["state"][0][ARM_EEF], -5.0)


def test_tail_action_masking_and_history_sampling():
    state = np.zeros(49, dtype=np.float32)
    state[48] = 1.0
    actions = np.ones((48, 49), dtype=np.float32)
    valid_time = np.arange(48) < 5
    _, s_mask, rel, rel_mask = model_state_and_actions(state, actions, valid_time)
    assert not s_mask[42:].any()
    assert rel_mask[:5, 0].all() and not rel_mask[5:].any()
    assert torch.all(rel[5:] == 0)

    seen_empty, seen_nonempty = False, False
    rng = random.Random(123)
    for _ in range(100):
        rows, ages = sample_recorded_history(90, 30.0, (0.5, 2.0), 0.1, rng)
        assert len(rows) == len(ages)
        assert all(0 <= r < 90 for r in rows)
        seen_empty |= len(rows) == 0
        seen_nonempty |= len(rows) == 2
    assert seen_empty and seen_nonempty


def test_goal_cache_verification(tmp_path):
    cache_dir = tmp_path / "train_goal_cache"
    (cache_dir / "adjust_bottle").mkdir(parents=True)
    pixels = np.zeros((224, 288, 3), dtype=np.uint8)
    pixels[4:220, :, :] = 128
    png_buf = io.BytesIO()
    Image.fromarray(pixels).save(png_buf, format="PNG")
    png_bytes = png_buf.getvalue()
    (cache_dir / "adjust_bottle/episode_000000.png").write_bytes(png_bytes)

    manifest = {
        "schema": SCHEMA,
        "image_size": IMAGE_SIZE,
        "tasks": ["adjust_bottle"],
        "episodes": [
            {
                "task": "adjust_bottle",
                "episode_index": 0,
                "instruction": "Pick up the bottle.",
                "goal_image": "adjust_bottle/episode_000000.png",
                "sha256": sha256_bytes(png_bytes),
            }
        ],
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    manifest_sha = sha256_bytes(manifest_bytes)
    (cache_dir / "manifest.json").write_bytes(manifest_bytes)
    (cache_dir / "COMPLETE.json").write_text(
        json.dumps(
            {"schema": SCHEMA, "complete": True, "tasks": 1, "episodes": 1, "manifest_sha256": manifest_sha}
        )
    )

    verified = verify_goal_cache(cache_dir, manifest_sha)
    assert verified["manifest_sha256"] == manifest_sha and verified["episodes"] == 1

    cache = TrainGoalCache(cache_dir)
    assert cache.resolve_task(Path("/datasets/adjust_bottle/aloha-agilex_clean_50"), {}) == "adjust_bottle"
    loaded = cache.read("adjust_bottle", 0, "Pick up the bottle.")
    assert np.array_equal(loaded, pixels)
    with pytest.raises(ValueError, match="Instruction mismatch"):
        cache.read("adjust_bottle", 0, "Wrong instruction.")
