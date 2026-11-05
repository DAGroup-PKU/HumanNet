"""The released RoboTwin 2.0 goal cache: one reviewed goal per official evaluation case.

Each of the 10,000 official cases (50 tasks x 100 expert-accepted seeds x
demo_clean/demo_randomized) has one 224x288 head-camera goal generated from the
initial head image and the instruction only, plus that initial observation.
The pinned manifest SHA-256 binds every row and artifact; artifact bytes are
hashed on every use. A cache miss or a different initial scene fails: it never
selects, skips or replaces a seed.
"""

import argparse
import hashlib
import io
import json
import math
import os
import stat
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .bridge import ACTION_MASK, STATE_MASK

CACHE_SCHEMA = "robotwin_official_goal_cache_v1"
RPC_PROTOCOL = "goalwam.official_native.v1"
PROFILE_SCHEMA = "humanwam_goal.robotwin.v1"
CAMERAS = ("head", "left", "right")
IDENTITY_FIELDS = ("task_config", "task", "seed", "official_instruction", "initial_head_sha256")
INITIAL_FILES = {"initial_state.npz", *(f"initial_{camera}.png" for camera in CAMERAS)}
GOAL_IMAGE_SIZE = [224, 288]
PROFILE_FIELDS = {"schema", "profile", "action_type", "replan_steps", "generation_seed"}
STAT_WORKERS = 32


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_digest(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def identity_hash(row):
    identity = {key: row[key] for key in IDENTITY_FIELDS}
    return digest(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def array_metadata(array):
    array = np.ascontiguousarray(array)
    return dict(dtype=str(array.dtype), shape=list(array.shape), sha256=digest(array.tobytes()))


def cases_digest(cases):
    return digest(
        json.dumps(cases, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    )


def case_id(task_config, task, seed):
    """Filename-safe case ID for one cache row."""
    return f"official-{task_config}-{task}-{seed}"


def fingerprint(path):
    info = os.stat(path)
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def load_profile(path):
    """The run profile: control mode, executed prefix and generation seed."""
    import yaml

    raw = Path(path).read_bytes()
    config = yaml.safe_load(raw)
    require(
        isinstance(config, dict) and set(config) == PROFILE_FIELDS,
        f"RoboTwin profile must contain exactly {sorted(PROFILE_FIELDS)}",
    )
    require(config["schema"] == PROFILE_SCHEMA, "Unsupported RoboTwin profile schema")
    require(
        isinstance(config["profile"], str) and config["profile"].strip(),
        "RoboTwin profile needs a name",
    )
    require(config["action_type"] == "qpos", "This release serves joint (qpos) control")
    require(config["replan_steps"] == 32, "The released results execute a 32-step prefix")
    require(
        type(config["generation_seed"]) is int and config["generation_seed"] >= 0,
        "generation_seed must be a nonnegative integer",
    )
    return config, digest(raw)


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


class OfficialGoalCache:
    """One self-contained cache for both task configurations.

    Paths are checked lexically at startup and resolved when read, so a symlink
    cannot escape the export. Control files are re-hashed when their stat
    identity changes; case artifacts are hashed whenever a case is prepared.
    """

    def __init__(self, root, manifest_sha256, profile):
        self.root = Path(root).resolve()
        self._root_prefix = str(self.root) + os.sep
        require(valid_digest(manifest_sha256), "The goal cache requires a pinned manifest SHA-256")
        self.sha256 = manifest_sha256
        self.profile_path = Path(profile).resolve()
        self.config, self.profile_sha256 = load_profile(self.profile_path)
        self.action_type = "joint"
        self.marker_raw = (self.root / "COMPLETE.json").read_bytes()
        self.marker = json.loads(self.marker_raw)
        self.scope = self.marker.get("scope", "full")
        require(
            self.marker.get("schema") == CACHE_SCHEMA
            and self.marker.get("complete") is True
            and self.scope in ("full", "partial")
            and self.marker.get("manifest_sha256") == self.sha256
            and (self.scope == "partial" or self.marker.get("cases") == 10000),
            "The goal cache is not a complete pinned 10000-case export",
        )
        self.manifest = json.loads(self._read("manifest.json", self.sha256))
        manifest = self.manifest
        require(
            manifest.get("schema") == CACHE_SCHEMA and manifest.get("scope", "full") == self.scope,
            "Unsupported goal cache schema or scope",
        )
        require(
            manifest.get("config") == dict(img_size=GOAL_IMAGE_SIZE, enable_cameras=["head"])
            and manifest.get("camera_boxes") == {"head": [0, 0, 224, 288]},
            "Unsupported goal image layout",
        )
        require(
            manifest.get("task_configs") == ["demo_clean", "demo_randomized"],
            "Goal cache population mismatch",
        )
        tasks = manifest["tasks"]
        require(
            len(tasks) == len(set(tasks)) == 50 and all(self._component(task) for task in tasks),
            "The goal cache requires 50 unique task names",
        )
        require(
            manifest.get("index_file") == "index.jsonl"
            and manifest.get("index_sha256") == self.marker.get("index_sha256"),
            "Goal cache index identity mismatch",
        )
        self.index_sha256 = manifest["index_sha256"]
        self._read("index.jsonl", self.index_sha256)
        self.protocol = json.loads(
            self._read(manifest["protocol_file"], manifest["protocol_sha256"])
        )
        require(
            self.protocol.get("tasks") == tasks
            and self.protocol.get("task_configs") == manifest["task_configs"]
            and self.protocol.get("test_num") == 100
            and self.protocol.get("expected_total_cases") == 10000
            and self.protocol.get("expert_check") is True,
            "Goal cache protocol mismatch",
        )
        self.cases = {}
        for row in manifest["cases"]:
            self._validate_row(row)
            key = row["task_config"], row["task"], row["seed"]
            require(key not in self.cases, "Duplicate goal cache identity")
            self.cases[key] = row
        self.groups = Counter((c, t) for c, t, _ in self.cases)
        require(
            len(self.cases) == self.marker.get("cases")
            and all(count == 100 for count in self.groups.values()),
            "The goal cache must contain exactly 100 cases in each included group",
        )
        if self.scope == "full":
            require(
                len(self.cases) == 10000
                and self.groups
                == Counter({(c, t): 100 for c in manifest["task_configs"] for t in tasks}),
                "The goal cache must contain exactly 100 cases in each of 100 groups",
            )
        else:
            require(
                0 < len(self.groups) < 100 and self.marker.get("groups") == len(self.groups),
                "A partial goal cache must state its included groups",
            )
        expected = {
            "COMPLETE.json": digest(self.marker_raw),
            "manifest.json": self.sha256,
            "index.jsonl": self.index_sha256,
            manifest["protocol_file"]: manifest["protocol_sha256"],
        }
        self._pinned = {self._path(name): [checksum, None] for name, checksum in expected.items()}
        self._pinned[self.profile_path] = [self.profile_sha256, None]
        self.verify_unchanged(full=True)

    @staticmethod
    def _component(value):
        return isinstance(value, str) and value not in ("", ".", "..") and Path(value).name == value

    def _path(self, relative):
        require(
            isinstance(relative, str) and relative and not os.path.isabs(relative),
            "Goal cache artifacts must use relative paths",
        )
        path = os.path.normpath(os.path.join(str(self.root), relative))
        require(path.startswith(self._root_prefix), "Goal cache artifact escapes the export")
        return Path(path)

    def _read(self, relative, expected):
        require(valid_digest(expected), "Goal cache artifact is missing a SHA-256 digest")
        path = self._path(relative)
        require(
            os.path.realpath(path).startswith(self._root_prefix),
            "Goal cache artifact escapes the export",
        )
        raw = path.read_bytes()
        require(digest(raw) == expected, "Goal cache artifact checksum mismatch: " + relative)
        return raw

    def _validate_row(self, row):
        config, task, seed = row["task_config"], row["task"], row["seed"]
        require(
            config in self.manifest["task_configs"]
            and task in self.manifest["tasks"]
            and type(seed) is int
            and seed >= 100000,
            "Invalid goal cache identity",
        )
        require(row.get("case_id") == f"{config}/{task}/{seed}", "Goal cache case id mismatch")
        instruction = row["official_instruction"]
        require(
            isinstance(instruction, str)
            and instruction.strip()
            and instruction != task
            and digest(instruction.encode("utf-8")) == row.get("official_instruction_sha256"),
            "Official instruction checksum mismatch",
        )
        require(
            identity_hash(row) == row.get("official_case_identity_sha256"),
            "Case identity checksum mismatch",
        )
        require(
            row.get("origin") in ("reused_verified", "generated", "reused_reviewed"),
            "Unknown goal origin",
        )
        if row["origin"] == "reused_verified":
            require(
                row.get("generation_need_class") == "reuse_verified"
                and row.get("reuse_decision") == "reusable",
                "Only verified reusable inputs may use an earlier goal",
            )
        if row["origin"] == "reused_reviewed":
            require(
                row.get("reuse_decision") == "reviewed_near_identical",
                "A reviewed earlier goal must record its near-identical scene decision",
            )
        status = row.get("goal_review_status", "pass")
        require(
            status == "pass"
            or (
                status == "provisional_user_authorized"
                and row.get("user_authorized_approximation") is True
            ),
            "Goal lacks an accepted review status",
        )
        require(
            row.get("camera_boxes") == {"head": [0, 0, 224, 288]}, "Goals must be head-only images"
        )
        for key in ("initial_rgb_files", "initial_rgb_sha256", "initial_rgb_arrays"):
            require(
                set(row[key]) == set(CAMERAS), "The goal cache requires all three initial cameras"
            )
        require(
            set(row["initial_arrays"]) == {"state", "cam2world_gl"},
            "Unsupported initial state archive",
        )
        require(
            row["initial_head_sha256"] == row["initial_rgb_sha256"]["head"],
            "Initial head image identity mismatch",
        )
        for path_key, hash_key in (
            ("goal_file", "goal_sha256"),
            ("scene_case_json", "scene_case_json_sha256"),
            ("source_contract_file", "source_contract_sha256"),
            ("initial_state_file", "initial_state_sha256"),
        ):
            self._path(row[path_key])
            require(valid_digest(row[hash_key]), "Missing goal cache artifact checksum")
        for camera in CAMERAS:
            self._path(row["initial_rgb_files"][camera])
            require(valid_digest(row["initial_rgb_sha256"][camera]), "Missing initial RGB checksum")

    @staticmethod
    def _artifacts(row):
        return [
            row["goal_file"],
            row["scene_case_json"],
            row["source_contract_file"],
            row["initial_state_file"],
            *(row["initial_rgb_files"][camera] for camera in CAMERAS),
        ]

    def verify_unchanged(self, full=False):
        for path, pinned in self._pinned.items():
            current = fingerprint(path)
            if full or current != pinned[1]:
                require(
                    digest(path.read_bytes()) == pinned[0],
                    "Goal cache artifact checksum mismatch: " + path.name
                    if path != self.profile_path
                    else "RoboTwin profile changed",
                )
                pinned[1] = current

    def validate_model(self, model):
        """The loaded checkpoint must consume these goals the way the released results did."""
        c, robot = model.config, model.robot
        require(
            robot is not None,
            "RoboTwin serving needs a checkpoint with robot.json (physical inputs)",
        )
        require(
            [list(size) for size in c.camera_sizes] == [GOAL_IMAGE_SIZE]
            and c.goal_head_only
            and list(c.cameras) == list(CAMERAS),
            "The checkpoint must use one 224x288 cell, head/left/right cameras and head-only goals",
        )
        require(c.prefix_length == 0, "Released RoboTwin serving uses no real-time-chunking prefix")
        require(
            robot.gripper_space == "command_openness" and robot.config.get("joint_channels", True),
            "RoboTwin joint serving needs command-openness grippers and trained joint channels",
        )
        require(
            np.array_equal(robot.state_mask.numpy(), STATE_MASK)
            and np.array_equal(robot.action_mask.numpy(), ACTION_MASK),
            "The checkpoint's validity masks differ from the Aloha-AgileX bridge",
        )

    def case(self, task_config, task, seed, instruction):
        require(type(seed) is int, "Official seed must be an integer")
        row = self.cases.get((task_config, task, seed))
        require(row is not None, "Goal cache miss; a native seed must not be skipped or replaced")
        require(
            instruction == row["official_instruction"],
            "Instruction differs from the exact official input",
        )
        return row

    def expected_cases(self, groups=None):
        """The expected native accepted population, as a case list."""
        rows = sorted(self.cases.values(), key=lambda r: (r["task_config"], r["task"], r["seed"]))
        if groups is not None:
            groups = set(groups)
            require(groups <= set(self.groups), "Unknown task/config group")
            rows = [r for r in rows if (r["task_config"], r["task"]) in groups]
        return [
            dict(
                id=case_id(r["task_config"], r["task"], r["seed"]),
                task=r["task"],
                task_config=r["task_config"],
                episode_seed=r["seed"],
                instruction=r["official_instruction"],
                expert_check=True,
            )
            for r in rows
        ]

    def _load_initial(self, row):
        scene = json.loads(self._read(row["scene_case_json"], row["scene_case_json_sha256"]))
        require(
            scene.get("id") == row["case_id"]
            and scene.get("task") == row["task"]
            and scene.get("task_config") == row["task_config"]
            and scene.get("seed") == row["seed"]
            and scene.get("instruction") == row["official_instruction"]
            and scene.get("policy_executed") is False,
            "Source scene identity mismatch",
        )
        expected = {
            "initial_state.npz": row["initial_state_sha256"],
            **{f"initial_{camera}.png": row["initial_rgb_sha256"][camera] for camera in CAMERAS},
        }
        require(
            scene.get("initial_files_sha256") == expected and set(expected) == INITIAL_FILES,
            "Source scene initial artifact identity mismatch",
        )
        contract = json.loads(
            self._read(row["source_contract_file"], row["source_contract_sha256"])
        )
        require(
            contract.get("protocol") == self.protocol
            and contract.get("task") == row["task"]
            and contract.get("task_config") == row["task_config"],
            "Source scene contract differs",
        )
        state_raw = self._read(row["initial_state_file"], row["initial_state_sha256"])
        with np.load(io.BytesIO(state_raw), allow_pickle=False) as archive:
            require(
                set(archive.files) == {"state", "cam2world_gl"},
                "Only the initial state and head camera may be loaded",
            )
            initial = {key: archive[key].copy() for key in archive.files}
        for key, shape in (("state", (49,)), ("cam2world_gl", (4, 4))):
            array = initial[key]
            require(
                array.shape == shape
                and np.issubdtype(array.dtype, np.floating)
                and np.isfinite(array).all()
                and array_metadata(array) == row["initial_arrays"][key],
                "Initial state/camera checksum mismatch",
            )
        initial["images"] = {}
        for camera in CAMERAS:
            raw = self._read(row["initial_rgb_files"][camera], row["initial_rgb_sha256"][camera])
            with Image.open(io.BytesIO(raw)) as image:
                require(image.mode == "RGB", "Initial image is not RGB")
                array = np.array(image)
            require(
                array.ndim == 3
                and array.shape[2] == 3
                and array.dtype == np.uint8
                and array_metadata(array) == row["initial_rgb_arrays"][camera],
                "Initial RGB checksum mismatch",
            )
            initial["images"][camera] = array
        return initial

    def _load(self, row):
        initial = self._load_initial(row)
        raw = self._read(row["goal_file"], row["goal_sha256"])
        with Image.open(io.BytesIO(raw)) as image:
            require(
                image.mode == "RGB" and image.size == (288, 224),
                "Goals must be 224x288 RGB head cells",
            )
            head = np.array(image)
        provenance = dict(
            goal_source="official_initial_only_cache",
            goal_origin=row["origin"],
            generated_goal_png_sha256=row["goal_sha256"],
            goal_head_sha256=digest(head.tobytes()),
            official_cache_manifest_sha256=self.sha256,
            official_case_identity_sha256=row["official_case_identity_sha256"],
            profile_sha256=self.profile_sha256,
            profile=self.config["profile"],
            generation_instruction=row.get("generation_instruction"),
            policy_instruction=row["official_instruction"],
            goal_review_status=row.get("goal_review_status", "pass"),
            cache_scope=self.scope,
        )
        return dict(
            reference=dict(initial=initial, instruction=row["official_instruction"]),
            goal=head,
            provenance=provenance,
            config=dict(self.config, task_config=row["task_config"]),
            policy_instruction=row["official_instruction"],
            official_case=row,
        )

    def prepare(self, metadata):
        require(
            metadata.get("protocol") == RPC_PROTOCOL,
            "The official case requires the native RPC protocol",
        )
        require(
            metadata.get("expected_cache_manifest_sha256") == self.sha256,
            "Client/server goal cache manifest mismatch",
        )
        require(
            metadata.get("action_type") == self.action_type,
            "Client control mode differs from the profile",
        )
        self.verify_unchanged()
        row = self.case(
            metadata["task_config"],
            metadata["task_name"],
            metadata["seed"],
            metadata["instruction"],
        )
        return self._load(row)

    def acknowledgement(self, prepared, operation, initial_verified):
        row = prepared["official_case"]
        return dict(
            protocol=RPC_PROTOCOL,
            operation=operation,
            status="ready" if initial_verified else "prepared",
            cache_manifest_sha256=self.sha256,
            case_id=row["case_id"],
            instruction_sha256=row["official_instruction_sha256"],
            initial_verified=initial_verified,
            cache_entry_sha256=digest(
                json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ),
        )

    def validate_cases(self, request, action_type, replan_steps):
        """Read-only workload preflight: identities, complete groups and artifact availability."""
        require(
            request.get("protocol") == RPC_PROTOCOL
            and request.get("expected_cache_manifest_sha256") == self.sha256,
            "Preflight protocol/cache mismatch",
        )
        require(
            action_type == request.get("action_type") == self.action_type
            and replan_steps == self.config["replan_steps"],
            "Preflight control configuration mismatch",
        )
        self.verify_unchanged(full=True)
        cases = request.get("cases")
        require(isinstance(cases, list) and cases, "Preflight requires a nonempty case list")
        identities, ids, groups, paths = set(), set(), Counter(), set()
        for case in cases:
            require(
                isinstance(case, dict) and case.get("expert_check") is True,
                "Preflight cases must keep the native expert check",
            )
            key = case["task_config"], case["task"], case["episode_seed"]
            row = self.case(*key, case["instruction"])
            require(case.get("id") == case_id(*key), "Preflight case id differs from its identity")
            require(key not in identities and case["id"] not in ids, "Duplicate preflight case")
            identities.add(key)
            ids.add(case["id"])
            groups[key[:2]] += 1
            paths.update(self._path(relative) for relative in self._artifacts(row))
        require(
            all(count == self.groups[group] for group, count in groups.items()),
            "Preflight must list every cached case of each selected task/config group",
        )

        def available(path):
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode):
                return os.path.realpath(path).startswith(self._root_prefix) and stat.S_ISREG(
                    os.stat(path).st_mode
                )
            return stat.S_ISREG(mode)

        with ThreadPoolExecutor(max_workers=STAT_WORKERS) as pool:
            missing = [str(p) for p, ok in zip(paths, pool.map(available, paths)) if not ok]
        require(
            not missing,
            "Goal cache artifact is not a regular file inside the export: "
            + ", ".join(missing[:3]),
        )
        return dict(
            protocol=RPC_PROTOCOL,
            operation="validate_cases",
            status="ready",
            accepted=True,
            cache_manifest_sha256=self.sha256,
            case_count=len(cases),
            cases_sha256=cases_digest(cases),
            groups=len(groups),
            artifacts_checked=len(paths),
            action_type=action_type,
            replan_steps=replan_steps,
            profile_sha256=self.profile_sha256,
            profile=self.config["profile"],
        )

    def verify_artifacts(self, workers=16):
        """Offline full check: hash and decode every case, as prepare would."""
        rows = list(self.cases.values())

        def check(row):
            try:
                self._load(row)
                return None
            except Exception as error:  # Collect every failing case for review.
                return dict(case_id=row["case_id"], error=f"{type(error).__name__}: {error}")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            failures = [item for item in pool.map(check, rows) if item is not None]
        return dict(cases=len(rows), failures=failures, passed=not failures)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect the RoboTwin goal cache without loading a model"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--export-cases", type=Path, help="Write the expected case list (new file)")
    parser.add_argument(
        "--verify-artifacts", action="store_true", help="Hash and decode every cached case"
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    goals = OfficialGoalCache(args.root, args.manifest_sha256, args.profile)
    report = dict(
        root=str(goals.root),
        manifest_sha256=goals.sha256,
        cases=len(goals.cases),
        profile=goals.config["profile"],
        profile_sha256=goals.profile_sha256,
    )
    if args.export_cases:
        cases = goals.expected_cases()
        with args.export_cases.open("x", encoding="utf-8") as stream:
            json.dump(cases, stream, indent=1, ensure_ascii=False)
            stream.write("\n")
        report.update(exported=str(args.export_cases), cases_sha256=cases_digest(cases))
    if args.verify_artifacts:
        report["verification"] = goals.verify_artifacts(args.workers)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.verify_artifacts and not report["verification"]["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
