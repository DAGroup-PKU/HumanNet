from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENCODER = "vitl"
DEFAULT_CHECKPOINT = (
    SCRIPT_DIR / "tools" / "video_depth_anything" / f"metric_video_depth_anything_{DEFAULT_ENCODER}.pth"
)

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def init_distributed() -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    initialized_here = False

    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
        initialized_here = True

    return rank, local_rank, world_size, initialized_here


def get_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        device_id = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        return torch.device("cuda", device_id)
    return torch.device("cpu")


def make_even(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    new_height = height if height % 2 == 0 else height + 1
    new_width = width if width % 2 == 0 else width + 1
    if new_height == height and new_width == width:
        return frame
    return np.pad(
        frame,
        ((0, new_height - height), (0, new_width - width), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def resolve_video_path(video_path: str, csv_dir: Path, video_root: Path | None) -> Path:
    path = Path(video_path).expanduser()
    if path.is_absolute():
        return path
    base_dir = video_root if video_root is not None else csv_dir
    return (base_dir / path).resolve()


def load_manifest(csv_path: Path, video_root: Path | None, rank: int, world_size: int) -> list[Path]:
    csv_dir = csv_path.resolve().parent
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "video_path" not in (reader.fieldnames or []):
            raise ValueError("The manifest must contain a 'video_path' column.")
        paths = [
            resolve_video_path(row["video_path"], csv_dir=csv_dir, video_root=video_root)
            for row in reader
        ]
    return [path for index, path in enumerate(paths) if index % world_size == rank]


def load_model(checkpoint: Path, encoder: str, device: torch.device) -> Any:
    from tools.video_depth_anything.video_depth import VideoDepthAnything

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Depth checkpoint not found: {checkpoint}. "
            "Download it and pass --checkpoint, or place it at the default relative path."
        )

    model = VideoDepthAnything(**MODEL_CONFIGS[encoder])
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def process_video(
    video_path: Path,
    output_dir: Path,
    model: Any,
    device: torch.device,
    max_len: int,
    target_fps: int,
    max_res: int,
    input_size: int,
    fp32: bool,
    save_vis: bool,
    save_source: bool,
    skip_existing: bool,
) -> None:
    from tools.video_depth_anything.utils.dc_utils import read_video_frames, save_video

    if not video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    depth_path = output_dir / f"{stem}_depth.npz"
    if skip_existing and depth_path.exists():
        print(f"[skip] Existing depth condition: {depth_path}")
        return

    frames, fps = read_video_frames(str(video_path), max_len, target_fps, max_res)
    frames = np.stack([make_even(frame) for frame in frames], axis=0)

    with torch.inference_mode():
        depths, fps = model.infer_video_depth(
            frames,
            fps,
            input_size=input_size,
            device=device,
            fp32=fp32,
        )

    np.savez_compressed(
        depth_path,
        depth=depths.astype(np.float32),
        fps=np.float32(fps),
        source_video=str(video_path),
        frame_shape=np.asarray(frames.shape[1:3], dtype=np.int32),
    )

    if save_source:
        save_video(frames, output_dir / f"{stem}_src.mp4", fps=fps)
    if save_vis:
        save_video(depths, output_dir / f"{stem}_depth_vis.mp4", fps=fps, is_depths=True, grayscale=False)


def main(args: argparse.Namespace) -> None:
    rank, local_rank, world_size, initialized_here = init_distributed()
    device = get_device(local_rank)

    csv_path = Path(args.csv_path).expanduser().resolve()
    video_root = Path(args.video_root).expanduser().resolve() if args.video_root else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()

    model = load_model(checkpoint=checkpoint, encoder=args.encoder, device=device)
    video_paths = load_manifest(csv_path, video_root=video_root, rank=rank, world_size=world_size)

    for video_path in video_paths:
        try:
            print(f"[Rank {rank}] Processing depth: {video_path}")
            process_video(
                video_path=video_path,
                output_dir=output_dir,
                model=model,
                device=device,
                max_len=args.max_len,
                target_fps=args.target_fps,
                max_res=args.max_res,
                input_size=args.input_size,
                fp32=args.fp32,
                save_vis=not args.no_vis,
                save_source=args.save_source,
                skip_existing=args.skip_existing,
            )
            print(f"[Rank {rank}] Finished depth: {video_path}")
        except Exception as exc:
            print(f"[Rank {rank}] Error processing depth for {video_path}: {exc}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if initialized_here:
        torch.distributed.destroy_process_group()
    if rank == 0:
        print("=== Depth inference finished ===")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate depth conditions from raw videos with Video Depth Anything."
    )
    parser.add_argument("--csv_path", required=True, help="CSV manifest with a video_path column.")
    parser.add_argument(
        "--video_root",
        default=None,
        help="Optional root for relative video_path entries. Defaults to the manifest directory.",
    )
    parser.add_argument("--output_dir", default="./outputs/depth", help="Output directory.")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Depth checkpoint path.")
    parser.add_argument("--max_res", type=int, default=1280, help="Resize so the longest side is at most this.")
    parser.add_argument("--max_len", type=int, default=-1, help="Maximum frames to process; -1 means all.")
    parser.add_argument("--target_fps", type=int, default=-1, help="Target FPS; -1 preserves source FPS.")
    parser.add_argument("--input_size", type=int, default=512, help="Depth model input size.")
    parser.add_argument("--fp32", action="store_true", help="Disable autocast and run depth inference in fp32.")
    parser.add_argument("--no_vis", action="store_true", help="Do not save depth visualization videos.")
    parser.add_argument("--save_source", action="store_true", help="Also save the resized source video.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip videos with existing depth output.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
