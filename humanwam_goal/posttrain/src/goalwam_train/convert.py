"""Convert official RoboTwin 2.0 `aloha-agilex_clean_50.zip` archives to LeRobot v3.0 datasets.

    python -m goalwam_train.convert \\
        --raw-root /path/to/robotwin_raw \\
        --output /path/to/robotwin_clean_50 \\
        --workers 8
"""

import argparse
import io
import json
import os
import re
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from goalwam.robotwin.bridge import ACTION_MASK, STATE_MASK, camera_transform, world_pose_to_canonical
from PIL import Image

FPS = 30
CAMERA_SOURCES = {
    "observation.images.cam_high": "observation/head_camera/rgb",
    "observation.images.cam_left_wrist": "observation/left_camera/rgb",
    "observation.images.cam_right_wrist": "observation/right_camera/rgb",
}
FEATURE_NAMES = [
    *(f"left_arm_{i}" for i in range(7)),
    "left_gripper",
    *(f"right_arm_{i}" for i in range(7)),
    "right_gripper",
    *(f"torso_{i}" for i in range(4)),
    *(f"head_{i}" for i in range(3)),
    *(f"chassis_{i}" for i in range(3)),
    *(f"left_pose_{i}" for i in range(7)),
    "left_gripper_pose",
    *(f"right_pose_{i}" for i in range(7)),
    "right_gripper_pose",
    *(f"head_pose_{i}" for i in range(7)),
]
EPISODE_RE = re.compile(r"^(?:.+/)?data/episode_?([0-9]+)\.hdf5$")
INSTRUCTION_RE = re.compile(r"^(?:.+/)?instructions?/episode_?([0-9]+)\.json$")


def decode_hdf5_jpeg(raw):
    """Decode RoboTwin HDF5 camera bytes to HWC uint8 RGB.

    RoboTwin passed RGB arrays to `cv2.imencode`, which stores swapped R/B channels
    unless marked with `XPL-RGB1`. PIL reads standard JPEG channels, so legacy buffers
    swap once to recover exact original RGB.
    """
    data = bytes(np.asarray(raw).tobytes()) if not isinstance(raw, (bytes, bytearray)) else bytes(raw)
    with Image.open(io.BytesIO(data)) as img:
        comment = img.info.get("comment")
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    if comment != b"XPL-RGB1":
        rgb = rgb[..., ::-1]
    return np.ascontiguousarray(rgb)


def _openness(values, label):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(arr).all() or np.any(arr < -1e-4) or np.any(arr > 1.0001):
        raise ValueError(f"{label}: gripper openness must lie in [0, 1]")
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def build_episode_vectors(h5):
    """Build 49-D head-camera NWU `command_openness` configurations `[N, 49]`."""
    matrices = np.asarray(h5["observation/head_camera/cam2world_gl"], dtype=np.float64)
    n = len(matrices)
    if n < 2 or matrices.shape != (n, 4, 4):
        raise ValueError(f"Expected >=2 samples and (N,4,4) cam2world_gl, got {matrices.shape}")
    if not np.allclose(matrices, matrices[0], atol=1e-6, rtol=0.0):
        raise ValueError("Head camera moves within episode; static camera required")
    cam2world = matrices[0]
    camera_transform(cam2world)

    left_arm = np.asarray(h5["joint_action/left_arm"], dtype=np.float32)
    right_arm = np.asarray(h5["joint_action/right_arm"], dtype=np.float32)
    if left_arm.shape != (n, 6) or right_arm.shape != (n, 6):
        raise ValueError(f"Expected (N,6) aloha-agilex arm joints, got {left_arm.shape}, {right_arm.shape}")
    if not (np.isfinite(left_arm).all() and np.isfinite(right_arm).all()):
        raise ValueError("Non-finite arm joint values")

    vectors = np.zeros((n, 49), dtype=np.float32)
    vectors[:, 0:6] = left_arm
    vectors[:, 7] = _openness(h5["joint_action/left_gripper"], "left_joint_gripper")
    vectors[:, 8:14] = right_arm
    vectors[:, 15] = _openness(h5["joint_action/right_gripper"], "right_joint_gripper")
    left_eef = np.asarray(h5["endpose/left_endpose"], dtype=np.float64)
    right_eef = np.asarray(h5["endpose/right_endpose"], dtype=np.float64)
    for i in range(n):
        vectors[i, 26:33] = world_pose_to_canonical(left_eef[i], cam2world)
        vectors[i, 34:41] = world_pose_to_canonical(right_eef[i], cam2world)
    vectors[:, 33] = _openness(h5["endpose/left_gripper"], "left_eef_gripper")
    vectors[:, 41] = _openness(h5["endpose/right_gripper"], "right_eef_gripper")
    vectors[:, 48] = 1.0
    return vectors


def _select_instruction(zf, member, task, episode_id, vectors):
    fallback = task.replace("_", " ")
    if member is None:
        return fallback
    payload = json.loads(zf.read(member))
    candidates = [str(x).strip() for x in (payload.get("seen") or []) if str(x).strip()]
    left_span = np.ptp(np.c_[vectors[:, :6], vectors[:, 7:8] * 100.0, vectors[:, 26:29]], axis=0).max()
    right_span = np.ptp(np.c_[vectors[:, 8:14], vectors[:, 15:16] * 100.0, vectors[:, 34:37]], axis=0).max()
    active = ({("left" if left_span > 1e-4 else None), ("right" if right_span > 1e-4 else None)} - {None})
    if active == {"left"}:
        candidates = [c for c in candidates if re.search(r"\bright arm\b", c, re.I) is None]
    elif active == {"right"}:
        candidates = [c for c in candidates if re.search(r"\bleft arm\b", c, re.I) is None]
    return candidates[episode_id % len(candidates)] if candidates else fallback


class _VideoSink:
    def __init__(self, path, height, width, fps):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "18", "preset": "fast"}

    def write(self, rgb):
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self, *, flush=True):
        if flush:
            for packet in self.stream.encode():
                self.container.mux(packet)
        self.container.close()


def convert_archive(job):
    """Convert one `<task>/aloha-agilex_clean_50.zip` archive to LeRobot v3.0."""
    import h5py

    task, zip_path, target_dir, overwrite = job
    zip_path, target_dir = Path(zip_path), Path(target_dir)
    if target_dir.exists():
        if not overwrite and (target_dir / "meta/info.json").is_file():
            return {"task": task, "status": "skipped", "episodes": 0, "frames": 0}
        if not overwrite:
            raise FileExistsError(f"Incomplete target exists: {target_dir} (pass --overwrite)")
        shutil.rmtree(target_dir)

    staging = target_dir.with_name(f".{target_dir.name}.tmp-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    sinks = {}
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            ep_members, ins_members = {}, {}
            for name in zf.namelist():
                if m := EPISODE_RE.fullmatch(name):
                    ep_members[int(m.group(1))] = name
                elif m := INSTRUCTION_RE.fullmatch(name):
                    ins_members[int(m.group(1))] = name
            if not ep_members:
                raise ValueError(f"No data/episode*.hdf5 found in {zip_path}")

            data_rows, ep_rows = [], []
            task_to_idx = {}
            resolution = None
            global_idx = 0

            for out_ep_idx, src_ep_idx in enumerate(sorted(ep_members)):
                with h5py.File(io.BytesIO(zf.read(ep_members[src_ep_idx])), "r") as h5:
                    vectors = build_episode_vectors(h5)
                    states = vectors[:-1]
                    actions = vectors[1:].copy()
                    actions[:, 42:49] = 0.0
                    length = len(states)
                    instruction = _select_instruction(
                        zf, ins_members.get(src_ep_idx), task, src_ep_idx, vectors
                    )
                    task_idx = task_to_idx.setdefault(instruction, len(task_to_idx))

                    for cam_key, h5_key in CAMERA_SOURCES.items():
                        ds = h5[h5_key]
                        if len(ds) != len(vectors):
                            raise ValueError(f"{zip_path}:{h5_key} length mismatch")
                        for i in range(length):
                            rgb = decode_hdf5_jpeg(ds[i])
                            if resolution is None:
                                resolution = (int(rgb.shape[0]), int(rgb.shape[1]))
                            elif resolution != (int(rgb.shape[0]), int(rgb.shape[1])):
                                raise ValueError(f"{zip_path}: inconsistent camera resolution")
                            if cam_key not in sinks:
                                vpath = staging / f"videos/{cam_key}/chunk-000/file-000.mp4"
                                sinks[cam_key] = _VideoSink(vpath, resolution[0], resolution[1], FPS)
                            sinks[cam_key].write(rgb)

                    start_idx = global_idx
                    for f_idx in range(length):
                        data_rows.append(
                            {
                                "observation.state": states[f_idx].tolist(),
                                "action": actions[f_idx].tolist(),
                                "timestamp": np.float32(f_idx / FPS),
                                "frame_index": f_idx,
                                "episode_index": out_ep_idx,
                                "index": global_idx,
                                "task_index": task_idx,
                            }
                        )
                        global_idx += 1

                    ep_row = {
                        "episode_index": out_ep_idx,
                        "tasks": [instruction],
                        "length": length,
                        "data/chunk_index": 0,
                        "data/file_index": 0,
                        "dataset_from_index": start_idx,
                        "dataset_to_index": global_idx,
                    }
                    for cam_key in CAMERA_SOURCES:
                        ep_row[f"videos/{cam_key}/chunk_index"] = 0
                        ep_row[f"videos/{cam_key}/file_index"] = 0
                        ep_row[f"videos/{cam_key}/from_timestamp"] = start_idx / FPS
                        ep_row[f"videos/{cam_key}/to_timestamp"] = global_idx / FPS
                    ep_rows.append(ep_row)

        for sink in sinks.values():
            sink.close()
        sinks.clear()

        vec_type = pa.list_(pa.float32(), 49)
        data_schema = pa.schema(
            [
                ("observation.state", vec_type),
                ("action", vec_type),
                ("timestamp", pa.float32()),
                ("frame_index", pa.int64()),
                ("episode_index", pa.int64()),
                ("index", pa.int64()),
                ("task_index", pa.int64()),
            ]
        )
        (staging / "data/chunk-000").mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(data_rows, schema=data_schema), staging / "data/chunk-000/file-000.parquet")

        (staging / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(ep_rows), staging / "meta/episodes/chunk-000/file-000.parquet")

        tasks_table = pa.Table.from_pylist(
            [{"task_index": idx, "task": text} for text, idx in sorted(task_to_idx.items(), key=lambda x: x[1])]
        )
        pq.write_table(tasks_table, staging / "meta/tasks.parquet")

        features = {
            name: {"dtype": "float32", "shape": [49], "names": FEATURE_NAMES}
            for name in ("observation.state", "action")
        }
        for cam_key in CAMERA_SOURCES:
            features[cam_key] = {
                "dtype": "video",
                "shape": [resolution[0], resolution[1], 3],
                "names": ["height", "width", "channels"],
            }
        info = {
            "codebase_version": "v3.0",
            "robot_type": "aloha_agilex",
            "total_episodes": len(ep_rows),
            "total_frames": global_idx,
            "total_tasks": len(task_to_idx),
            "chunks_size": 1000,
            "fps": FPS,
            "splits": {"train": f"0:{len(ep_rows)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features,
            "state_dim_mask": STATE_MASK.tolist(),
            "action_dim_mask": ACTION_MASK.tolist(),
            "gripper_representation": "command_openness",
            "gripper_unit": "dimensionless",
            "robotwin_conversion": {
                "task": task,
                "variant": "clean",
                "source_archive": zip_path.name,
                "world_frame": "head_camera_nwu",
            },
        }
        (staging / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(target_dir)
        return {"task": task, "status": "converted", "episodes": len(ep_rows), "frames": global_idx}
    except BaseException:
        for sink in sinks.values():
            sink.close(flush=False)
        shutil.rmtree(staging, ignore_errors=True)
        raise


def discover_clean_archives(raw_root, tasks=None):
    raw_root = Path(raw_root)
    found = []
    for path in sorted(raw_root.glob("*/aloha-agilex_clean_*.zip")):
        task = path.parent.name
        if tasks is None or task in tasks:
            found.append((task, path))
    if not found:
        raise FileNotFoundError(f"No <task>/aloha-agilex_clean_*.zip archives found under {raw_root}")
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory of <task>/aloha-agilex_clean_50.zip")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for LeRobot v3.0 datasets")
    parser.add_argument("--tasks", nargs="+", default=None, help="Optional subset of task names")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    wanted = set(args.tasks) if args.tasks else None
    archives = discover_clean_archives(args.raw_root, wanted)
    jobs = [(task, str(zpath), str(args.output / task / zpath.stem), args.overwrite) for task, zpath in archives]
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            results = list(pool.map(convert_archive, jobs))
    else:
        results = [convert_archive(job) for job in jobs]
    print(
        json.dumps(
            {
                "datasets": len(results),
                "converted": sum(r["status"] == "converted" for r in results),
                "skipped": sum(r["status"] == "skipped" for r in results),
                "episodes": sum(r["episodes"] for r in results),
                "frames": sum(r["frames"] for r in results),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
