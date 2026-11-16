"""Compute RoboTwin 2.0 `robot.json` normalization from `clean_50` episodes and pretrained EEF bounds.

    python -m goalwam_train.normalization \\
        --data /path/to/robotwin_clean_50 \\
        --pretrained /path/to/HumanWAM-Goal \\
        --output /path/to/robot.json
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from goalwam.processing.geometry import (
    propagate_relative_action_validity,
    propagate_state_arm_eef_validity,
    to_relative_action,
    transform_state_arm_eef_coordinate,
    zero_chassis_state,
)

DOMAIN_ID = 16
HORIZON = 48
GRIPPERS = [7, 15, 33, 41]
ARM_EEF = [*range(26, 33), *range(34, 41)]
ALL_EEF = [*ARM_EEF, *range(42, 49)]
STATE_MASK = np.zeros(49, dtype=bool)
STATE_MASK[np.r_[0:6, 7, 8:14, 15, 26:49]] = True
ACTION_MASK = STATE_MASK.copy()
ACTION_MASK[42:] = False


def read_shared_eef_bounds(pretrained):
    """Read shared EEF `low`/`high` bounds from `<pretrained>/robot.json` (or a `robot.json` file)."""
    pretrained = Path(pretrained)
    robot_file = pretrained if pretrained.is_file() else pretrained / "robot.json"
    if not robot_file.is_file():
        raise FileNotFoundError(f"Missing pretrained robot.json: {robot_file}")
    norm = json.loads(robot_file.read_text())["normalization"]
    return {
        name: (
            np.asarray(norm[name]["low"], dtype=np.float64),
            np.asarray(norm[name]["high"], dtype=np.float64),
        )
        for name in ("state", "action")
    }


def discover_clean_datasets(root):
    """Find sorted RoboTwin `aloha_agilex` LeRobot v3.0 clean datasets below `root`."""
    root = Path(root)
    found = [
        Path(d).parent
        for d, _, files in os.walk(root, followlinks=True)
        if Path(d).name == "meta" and "info.json" in files
    ]
    datasets = []
    for path in sorted(found):
        info = json.loads((path / "meta/info.json").read_text())
        if info.get("codebase_version") != "v3.0" or info.get("robot_type") != "aloha_agilex":
            continue
        conv = info.get("robotwin_conversion") or {}
        if "randomized" in path.name or "randomized" in Path(conv.get("source_archive", "")).name:
            continue
        grip = info.get("gripper_representation")
        if grip not in (None, "command_openness"):
            raise ValueError(f"{path}: expected command_openness grippers, got {grip!r}")
        datasets.append((path, info))
    if not datasets:
        raise ValueError(f"No LeRobot v3.0 RoboTwin Aloha-AgileX clean datasets found under {root}")
    return datasets


def _dataset_arrays(job):
    """Return model-coordinate states and furthest-offset relative actions for one dataset."""
    path, horizon = job
    path = Path(path)
    parquet_files = sorted(path.glob("data/chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {path}/data/chunk-*/")

    ep_parts, fr_parts, st_parts, ac_parts = [], [], [], []
    for pf in parquet_files:
        table = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state", "action"])
        if table.num_rows:
            ep_parts.append(table["episode_index"].to_numpy())
            fr_parts.append(table["frame_index"].to_numpy())
            st_parts.append(np.stack(table["observation.state"].to_numpy(zero_copy_only=False)))
            ac_parts.append(np.stack(table["action"].to_numpy(zero_copy_only=False)))

    episodes = np.concatenate(ep_parts)
    frames = np.concatenate(fr_parts)
    states_all = np.concatenate(st_parts).astype(np.float32, copy=False)
    actions_all = np.concatenate(ac_parts).astype(np.float32, copy=False)
    order = np.lexsort((frames, episodes))
    episodes, states_all, actions_all = episodes[order], states_all[order], actions_all[order]

    state_mask_t = torch.from_numpy(STATE_MASK)
    action_mask_t = torch.from_numpy(ACTION_MASK)
    model_state_mask = propagate_state_arm_eef_validity(state_mask_t, "head_camera")

    out_states, out_rel, out_rel_mask = [], [], []
    _, starts = np.unique(episodes, return_index=True)
    ends = np.r_[starts[1:], len(episodes)]
    for start, end in zip(starts, ends, strict=True):
        st = torch.from_numpy(states_all[start:end]).masked_fill(~state_mask_t, 0)
        ac = torch.from_numpy(actions_all[start:end]).masked_fill(~action_mask_t, 0)
        n = len(st)
        furthest = torch.minimum(torch.arange(n) + (horizon - 1), torch.tensor(n - 1))
        ac_tk = ac[furthest]
        am_tk = action_mask_t.expand(n, 49)
        rel = to_relative_action(st, ac_tk)
        rel_mask = propagate_relative_action_validity(am_tk, state_mask_t)
        ms = zero_chassis_state(transform_state_arm_eef_coordinate(st, "head_camera"))
        out_states.append(ms.masked_fill(~model_state_mask, 0).numpy())
        out_rel.append(rel.masked_fill(~rel_mask, 0).numpy())
        out_rel_mask.append(rel_mask.numpy())

    return (
        np.concatenate(out_states, axis=0),
        np.broadcast_to(model_state_mask.numpy(), (len(episodes), 49)).copy(),
        np.concatenate(out_rel, axis=0),
        np.concatenate(out_rel_mask, axis=0),
    )


def _quantiles(values, valid):
    low = np.zeros(49, dtype=np.float64)
    high = np.zeros(49, dtype=np.float64)
    for dim in range(49):
        col = values[valid[:, dim], dim].astype(np.float64)
        if len(col):
            low[dim], high[dim] = np.quantile(col, [0.01, 0.99])
    return low, high


def compute_robot_config(data_root, pretrained, *, horizon=HORIZON, workers=None):
    """Compute the full `robot.json` dict for RoboTwin 2.0 Aloha-AgileX."""
    datasets = discover_clean_datasets(data_root)
    eef_bounds = read_shared_eef_bounds(pretrained)
    jobs = [(str(path), horizon) for path, _ in datasets]
    workers = min(len(jobs), workers if workers is not None else min(8, os.cpu_count() or 1))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_dataset_arrays, jobs))
    else:
        parts = [_dataset_arrays(job) for job in jobs]

    states = np.concatenate([p[0] for p in parts], axis=0)
    state_valid = np.concatenate([p[1] for p in parts], axis=0)
    rel = np.concatenate([p[2] for p in parts], axis=0)
    rel_valid = np.concatenate([p[3] for p in parts], axis=0)

    state_low, state_high = _quantiles(states, state_valid)
    action_low, action_high = _quantiles(rel, rel_valid)

    # Inherit the pretrained model's shared EEF bounds; fix grippers to [0, 1] openness.
    ref_s_low, ref_s_high = eef_bounds["state"]
    ref_a_low, ref_a_high = eef_bounds["action"]
    state_low[ARM_EEF], state_high[ARM_EEF] = ref_s_low[ARM_EEF], ref_s_high[ARM_EEF]
    action_low[ALL_EEF], action_high[ALL_EEF] = ref_a_low[ALL_EEF], ref_a_high[ALL_EEF]
    state_low[GRIPPERS], state_high[GRIPPERS] = 0.0, 1.0
    action_low[GRIPPERS], action_high[GRIPPERS] = 0.0, 1.0

    normalization = {
        "state": {
            "mode": "bounds",
            "low": np.asarray(state_low, dtype=np.float32).tolist(),
            "high": np.asarray(state_high, dtype=np.float32).tolist(),
        },
        "action": {
            "mode": "bounds",
            "low": np.asarray(action_low, dtype=np.float32).tolist(),
            "high": np.asarray(action_high, dtype=np.float32).tolist(),
        },
    }
    return {
        "domain_id": DOMAIN_ID,
        "state_coordinate": "head_camera",
        "action_coordinate": "head_camera",
        "gripper_space": "command_openness",
        "joint_channels": True,
        "state_valid_mask": STATE_MASK.tolist(),
        "action_valid_mask": ACTION_MASK.tolist(),
        "normalization": normalization,
        "instruction_prefix": "",
    }


def write_robot_json(data_root, pretrained, path, *, horizon=HORIZON, workers=None):
    robot = compute_robot_config(data_root, pretrained, horizon=horizon, workers=workers)
    text = json.dumps(robot, indent=2) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Directory of RoboTwin clean_50 LeRobot v3 datasets")
    parser.add_argument("--pretrained", type=Path, required=True, help="Pretrained HumanWAM-Goal bundle or checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output robot.json path")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    digest = write_robot_json(args.data, args.pretrained, args.output, horizon=args.horizon, workers=args.workers)
    print(json.dumps({"output": str(args.output), "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
