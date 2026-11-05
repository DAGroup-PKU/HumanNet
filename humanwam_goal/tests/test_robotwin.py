"""CPU tests of the RoboTwin layer: bridge, goal cache, session and runtime check."""

import hashlib
import json
import shutil

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from goalwam.processing.observation import ProcessorConfig
from goalwam.processing.robot import RobotProcessor
from goalwam.robotwin import (
    ACTION_MASK,
    RPC_PROTOCOL,
    STATE_MASK,
    OfficialGoalCache,
    RoboTwinSession,
    cublaslt_runtime,
    encode_observation,
    joint_commands,
    native_observation,
)
from goalwam.robotwin.goal_cache import CACHE_SCHEMA, array_metadata, digest, identity_hash
from goalwam.robotwin.goal_cache import main as cache_main

INSTRUCTION = "Pick up the object"
PROFILE = "schema: humanwam_goal.robotwin.v1\nprofile: test\naction_type: qpos\nreplan_steps: 32\ngeneration_seed: 9000\n"


def wire_observation(step=0, rgb_offset=0, grip=(0.25, 0.75)):
    """One XPolicyLab v1.0 RoboTwin observation with a rotated head camera."""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.4, 0.7]).as_matrix()
    matrix[:3, 3] = [0.5, -0.2, 1.3]
    vision = {}
    for index, key in enumerate(("cam_head", "cam_left_wrist", "cam_right_wrist")):
        rgb = np.full((24, 32, 3), [30 + index + rgb_offset, 60, 90], dtype=np.uint8)
        vision[key] = dict(
            color=rgb, extrinsics_matrix=(np.diag([1, -1, -1, 1]) @ np.linalg.inv(matrix))[:3]
        )
    state = {}
    for index, side in enumerate(("left", "right")):
        state[f"{side}_arm_joint_state"] = np.arange(6, dtype=np.float32) * (0.1 - 0.2 * index)
        state[f"{side}_ee_joint_state"] = np.array([grip[index]])
        state[f"{side}_ee_pose"] = np.r_[[0.2, 0.1, 0.8], [1, 0, 0, 0]]
    return dict(
        data_format_version="v1.0",
        env_idx=0,
        vision=vision,
        state=state,
        instruction=INSTRUCTION,
        additional_info=dict(frequency=30, capture_step=step),
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def publish(root, manifest):
    index = b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode()
        for row in manifest["cases"]
    )
    (root / "index.jsonl").write_bytes(index)
    manifest["index_sha256"] = digest(index)
    write_json(root / "manifest.json", manifest)
    checksum = digest((root / "manifest.json").read_bytes())
    write_json(
        root / "COMPLETE.json",
        dict(
            schema=CACHE_SCHEMA,
            complete=True,
            cases=10000,
            manifest_sha256=checksum,
            index_sha256=digest(index),
        ),
    )
    return checksum


@pytest.fixture(scope="module")
def cache_template(tmp_path_factory):
    """A complete 10,000-row cache whose scene files exist for demo_clean/task0 only."""
    root = tmp_path_factory.mktemp("goal-cache")
    initial = encode_observation(
        native_observation(wire_observation()), gripper_representation="closedness_100"
    )
    (root / "initial").mkdir()
    np.savez_compressed(
        root / "initial/initial_state.npz",
        state=initial["state"],
        cam2world_gl=initial["cam2world_gl"],
    )
    rgb_files, rgb_hashes = {}, {}
    for camera, rgb in initial["images"].items():
        Image.fromarray(rgb).save(root / f"initial/initial_{camera}.png")
        rgb_files[camera] = f"initial/initial_{camera}.png"
        rgb_hashes[camera] = digest((root / rgb_files[camera]).read_bytes())
    Image.fromarray(np.full((224, 288, 3), 123, dtype=np.uint8)).save(root / "goal.png")
    tasks, configs = [f"task{i}" for i in range(50)], ["demo_clean", "demo_randomized"]
    protocol = dict(
        tasks=tasks,
        task_configs=configs,
        test_num=100,
        expected_total_cases=10000,
        expert_check=True,
    )
    write_json(root / "protocol.json", protocol)
    initial_hashes = {
        "initial_state.npz": digest((root / "initial/initial_state.npz").read_bytes()),
        **{f"initial_{c}.png": h for c, h in rgb_hashes.items()},
    }
    rows = []
    for config in configs:
        for task in tasks:
            contract_file = f"contracts/{config}/{task}.json"
            write_json(root / contract_file, dict(protocol=protocol, task=task, task_config=config))
            for seed in range(100000, 100100):
                row = dict(
                    task_config=config,
                    task=task,
                    seed=seed,
                    official_instruction=INSTRUCTION,
                    initial_head_sha256=rgb_hashes["head"],
                    case_id=f"{config}/{task}/{seed}",
                    official_instruction_sha256=digest(INSTRUCTION.encode()),
                    origin="generated",
                    goal_file="goal.png",
                    goal_sha256=digest((root / "goal.png").read_bytes()),
                    camera_boxes={"head": [0, 0, 224, 288]},
                    scene_case_json=f"scenes/{config}/{task}/{seed}.json",
                    source_contract_file=contract_file,
                    source_contract_sha256=digest((root / contract_file).read_bytes()),
                    initial_state_file="initial/initial_state.npz",
                    initial_state_sha256=initial_hashes["initial_state.npz"],
                    initial_rgb_files=rgb_files,
                    initial_rgb_sha256=rgb_hashes,
                    initial_arrays={
                        k: array_metadata(initial[k]) for k in ("state", "cam2world_gl")
                    },
                    initial_rgb_arrays={
                        c: array_metadata(rgb) for c, rgb in initial["images"].items()
                    },
                )
                scene = dict(
                    id=row["case_id"],
                    task=task,
                    task_config=config,
                    seed=seed,
                    instruction=INSTRUCTION,
                    policy_executed=False,
                    initial_files_sha256=initial_hashes,
                )
                scene_raw = json.dumps(scene, ensure_ascii=False).encode()
                row["scene_case_json_sha256"] = digest(scene_raw)
                if config == "demo_clean" and task == "task0":
                    write_json(root / row["scene_case_json"], scene)
                row["official_case_identity_sha256"] = identity_hash(row)
                rows.append(row)
    publish(
        root,
        dict(
            schema=CACHE_SCHEMA,
            config=dict(img_size=[224, 288], enable_cameras=["head"]),
            camera_boxes={"head": [0, 0, 224, 288]},
            tasks=tasks,
            task_configs=configs,
            cases=rows,
            protocol_file="protocol.json",
            protocol_sha256=digest((root / "protocol.json").read_bytes()),
            index_file="index.jsonl",
        ),
    )
    (root / "profile.yaml").write_text(PROFILE)
    return root


@pytest.fixture
def cache(cache_template, tmp_path):
    root = tmp_path / "cache"
    shutil.copytree(cache_template, root)
    return OfficialGoalCache(
        root, digest((root / "manifest.json").read_bytes()), root / "profile.yaml"
    )


def metadata(cache, seed=100000, **change):
    base = dict(
        protocol=RPC_PROTOCOL,
        expected_cache_manifest_sha256=cache.sha256,
        task_config="demo_clean",
        task_name="task0",
        seed=seed,
        instruction=INSTRUCTION,
        action_type="joint",
    )
    return {**base, **change}


class FakeModel:
    """The real robot pre/post-processing around a zero-action model."""

    def __init__(self, tmp_path, gripper_space="command_openness", state_mask=STATE_MASK):
        stats = dict(mode="bounds", low=[-1.0] * 49, high=[1.0] * 49)
        for dim in (7, 15, 33, 41):
            stats["low"][dim], stats["high"][dim] = 0.0, 1.0
        robot = dict(
            domain_id=16,
            state_coordinate="head_camera",
            action_coordinate="head_camera",
            gripper_space=gripper_space,
            joint_channels=True,
            state_valid_mask=np.asarray(state_mask).tolist(),
            action_valid_mask=ACTION_MASK.tolist(),
            normalization=dict(state=stats, action=dict(stats)),
        )
        if gripper_space != "command_openness":
            robot["open_aperture_m"] = 0.1
        (tmp_path / "robot.json").write_text(json.dumps(robot))
        self.robot = RobotProcessor(tmp_path / "robot.json")
        self.config = ProcessorConfig(camera_sizes=((224, 288),), prefix_length=0)
        self.calls = []

    def predict(self, observation):
        self.calls.append(observation)
        source, context = self.robot.preprocess(observation, 0)
        return self.robot.postprocess(torch.zeros(48, 49), context)


def test_bridge_encodes_head_camera_state_in_both_gripper_units():
    native = native_observation(wire_observation())
    openness = encode_observation(native)
    closedness = encode_observation(native, gripper_representation="closedness_100")
    np.testing.assert_allclose(openness["state"][[7, 15, 33, 41]], [0.25, 0.75, 0.25, 0.75])
    np.testing.assert_allclose(closedness["state"][[7, 15, 33, 41]], [75, 25, 75, 25])
    np.testing.assert_array_equal(
        openness["state"][42:49], [0, 0, 0, 0, 0, 0, 1]
    )  # identity head pose
    for quat in (slice(29, 33), slice(37, 41)):
        assert abs(np.linalg.norm(openness["state"][quat]) - 1) < 1e-6
    assert not openness["state"][~STATE_MASK].any()
    with pytest.raises(ValueError):
        encode_observation(native, gripper_representation="gap_m")


def test_joint_commands_select_joints_and_clip_openness():
    absolute = np.arange(48 * 49, dtype=np.float32).reshape(48, 49) / 1000
    absolute[:, 7], absolute[:, 15] = -0.5, 1.5
    valid = np.broadcast_to(ACTION_MASK, (48, 49)).copy()
    commands = joint_commands(absolute, valid)
    assert commands.shape == (48, 14)
    np.testing.assert_array_equal(commands[:, :6], absolute[:, 0:6])
    np.testing.assert_array_equal(commands[:, 7:13], absolute[:, 8:14])
    assert (commands[:, 6] == 0).all() and (commands[:, 13] == 1).all()
    valid[:, 3] = False
    with pytest.raises(ValueError, match="joint supervision"):
        joint_commands(absolute, valid)


def test_cached_case_acknowledges_the_client_binding_and_serves_32_steps(cache, tmp_path):
    model = FakeModel(tmp_path)
    session = RoboTwinSession(model, tmp_path / "session", cache)
    ack = session.prepare_case(metadata(cache))
    assert (
        ack["protocol"] == RPC_PROTOCOL
        and ack["status"] == "prepared"
        and ack["initial_verified"] is False
    )
    assert (
        ack["case_id"] == "demo_clean/task0/100000" and ack["cache_manifest_sha256"] == cache.sha256
    )
    assert ack["instruction_sha256"] == hashlib.sha256(INSTRUCTION.encode()).hexdigest()
    session.reset()
    ready = session.update_obs(wire_observation())
    assert ready["status"] == "ready" and ready["initial_verified"] is True
    first = session.get_action()
    assert first.shape == (32, 14)
    # Zero normalized deltas hold the anchor (up to the 1e-6 range guard of bounds normalization).
    np.testing.assert_allclose(
        first[:, :6], np.broadcast_to(np.arange(6) * 0.1, (32, 6)), atol=1e-5
    )
    session.update_obs(wire_observation(step=32, rgb_offset=5))
    session.get_action()
    seeds = [call["seed"] for call in model.calls]
    assert seeds == [9000, 9001]
    assert (model.calls[0]["goal_images"]["head"] == 123).all()
    assert [
        c["timestamp"] for c in model.calls[1]["history"]
    ] == []  # step 0 is 32 steps (1.07 s) back
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "session/events.jsonl").read_text().splitlines()
    ]
    assert events.count("action") == 2 and "initial_scene_check" in events


@pytest.mark.parametrize(
    "change,match",
    [
        (dict(seed=999999), "cache miss"),
        (dict(instruction="Pick up the object "), "exact official"),
        (dict(expected_cache_manifest_sha256="0" * 64), "manifest mismatch"),
        (dict(protocol="other"), "native RPC"),
        (dict(action_type="ee"), "control mode"),
    ],
)
def test_cache_miss_or_mismatch_never_serves_a_goal(cache, tmp_path, change, match):
    session = RoboTwinSession(FakeModel(tmp_path), tmp_path / "session", cache)
    with pytest.raises(ValueError, match=match):
        session.prepare_case(metadata(cache, **change))
    assert session.goal_case is None


@pytest.mark.parametrize(
    "observation,match",
    [
        (wire_observation(rgb_offset=1), "Initial RGB differs"),
        (wire_observation(grip=(0.3, 0.75)), "Not equal to tolerance"),
        (wire_observation(step=3), "step zero"),
    ],
)
def test_initial_scene_must_equal_the_cached_observation(cache, tmp_path, observation, match):
    session = RoboTwinSession(FakeModel(tmp_path), tmp_path / "session", cache)
    session.prepare_case(metadata(cache))
    session.reset()
    with pytest.raises((ValueError, AssertionError), match=match):
        session.update_obs(observation)
    assert session.goal_case is None and not session.initial_checked
    with pytest.raises(ValueError):
        session.get_action()


def test_instruction_and_checkpoint_mismatches_are_rejected(cache, tmp_path):
    session = RoboTwinSession(FakeModel(tmp_path), tmp_path / "session", cache)
    session.prepare_case(metadata(cache))
    session.reset()
    with pytest.raises(ValueError, match="instruction"):
        session.update_obs(dict(wire_observation(), instruction="Another task"))
    with pytest.raises(ValueError, match="command-openness"):
        RoboTwinSession(FakeModel(tmp_path, gripper_space="gap_m"), tmp_path / "other", cache)
    mask = STATE_MASK.copy()
    mask[23:26] = True
    with pytest.raises(ValueError, match="masks"):
        RoboTwinSession(FakeModel(tmp_path, state_mask=mask), tmp_path / "other", cache)


def test_observation_plan_requests_the_half_and_two_second_history(cache, tmp_path):
    session = RoboTwinSession(FakeModel(tmp_path), tmp_path / "session", cache)
    plan = session.get_observation_plan(dict(frequency=30, step_limit=200))
    queries = {q["step"]: q["history_steps"] for q in plan["queries"]}
    assert list(queries) == list(range(0, 200, 32))
    assert queries[0] == [] and queries[32] == [17] and queries[64] == [4, 49]


def test_scheduled_captures_replace_dense_updates(cache, tmp_path):
    model = FakeModel(tmp_path)
    session = RoboTwinSession(model, tmp_path / "session", cache)
    session.prepare_case(metadata(cache))
    session.reset()
    plan = session.get_observation_plan(dict(frequency=30, step_limit=100))
    for query in plan["queries"]:
        steps = query["history_steps"] + [query["step"]]
        frames = [wire_observation(step=s, rgb_offset=0 if s == 0 else s % 7) for s in steps]
        assert session.get_action(dict(captures=frames)).shape == (32, 14)
    assert [len(call["history"]) for call in model.calls] == [0, 1, 2, 2]
    with pytest.raises(ValueError, match="No planned query"):
        session.get_action(dict(captures=[wire_observation(step=128)]))
    fresh = RoboTwinSession(FakeModel(tmp_path), tmp_path / "fresh", cache)
    fresh.prepare_case(metadata(cache))
    fresh.reset()
    fresh.get_observation_plan(dict(frequency=30, step_limit=100))
    with pytest.raises(ValueError, match="planned query"):
        fresh.get_action(dict(captures=[wire_observation(step=5)]))


def test_debug_session_uses_the_current_head_image_as_goal(tmp_path):
    model = FakeModel(tmp_path)
    session = RoboTwinSession(model, tmp_path / "session")
    session.reset()
    assert session.update_obs(wire_observation(step=0, rgb_offset=9)) is None
    assert session.get_action().shape == (32, 14)
    np.testing.assert_array_equal(
        model.calls[0]["goal_images"]["head"], model.calls[0]["images"]["head"]
    )
    with pytest.raises(ValueError, match="goal cache"):
        session.validate_cases({})


def test_cublaslt_check_requires_exactly_the_pinned_library(tmp_path):
    library = tmp_path / "libcublasLt.so.12"
    library.write_bytes(b"not a real library")
    maps = tmp_path / "maps"
    maps.write_text("")
    with pytest.raises(RuntimeError, match="exactly one"):
        cublaslt_runtime(maps=maps)
    maps.write_text(f"7f00-7f01 r-xp 00000000 00:00 0                          {library}\n")
    with pytest.raises(RuntimeError, match="mismatch"):
        cublaslt_runtime(maps=maps)


def test_cli_exports_the_expected_population_once(cache_template, tmp_path, capsys):
    target = tmp_path / "cases.json"
    sha = digest((cache_template / "manifest.json").read_bytes())
    argv = [
        "--root",
        str(cache_template),
        "--manifest-sha256",
        sha,
        "--profile",
        str(cache_template / "profile.yaml"),
        "--export-cases",
        str(target),
    ]
    cache_main(argv)
    cases = json.loads(target.read_text())
    assert len(cases) == 10000 and cases[0]["id"] == "official-demo_clean-task0-100000"
    assert json.loads(capsys.readouterr().out)["cases"] == 10000
    with pytest.raises(FileExistsError):
        cache_main(argv)
