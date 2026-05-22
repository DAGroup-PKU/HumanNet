from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COTRACKER_REPO = SCRIPT_DIR / "tools" / "facebook_co-tracker"


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


def load_cotracker(model_dir: Path, device: torch.device):
    if not model_dir.exists():
        raise FileNotFoundError(f"CoTracker repo not found: {model_dir}")
    model = torch.hub.load(str(model_dir), "cotracker3_offline", source="local")
    return model.to(device).eval()


def video_numpy_to_tensor(frames: np.ndarray) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape [T,H,W,3], got {frames.shape}")

    # CoTracker expects float video tensors in pixel space, matching the official demo:
    # [B, T, C, H, W], RGB, range [0, 255].
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float().unsqueeze(0)


def run_cotracker_on_video(
    cotracker,
    video_tensor: torch.Tensor,
    grid_size: int,
    grid_query_frame: int,
    backward_tracking: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        video_tracks, video_visibility = cotracker(
            video_tensor,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
            backward_tracking=backward_tracking,
        )
    return video_tracks, video_visibility


def load_visualizer(model_dir: Path):
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from cotracker.utils.visualizer import Visualizer

    return Visualizer


def visualize_tracks(
    frames: np.ndarray,
    video_tracks: torch.Tensor,
    video_visibility: torch.Tensor,
    save_dir: Path,
    filename: str,
    fps: int,
    model_dir: Path,
) -> None:
    from tools.video_depth_anything.utils.dc_utils import read_video_frames

    Visualizer = load_visualizer(model_dir)
    video = video_numpy_to_tensor(frames)
    vis = Visualizer(
        save_dir=str(save_dir),
        pad_value=120,
        linewidth=3,
        mode="rainbow",
        tracks_leave_trace=-1,
        fps=fps,
    )
    vis.visualize(
        video=video,
        tracks=video_tracks,
        visibility=video_visibility,
        query_frame=0,
        filename=filename,
    )


def process_video(
    video_path: Path,
    output_dir: Path,
    cotracker,
    device: torch.device,
    model_dir: Path,
    max_len: int,
    target_fps: int,
    max_res: int,
    grid_size: int,
    grid_query_frame: int,
    backward_tracking: bool,
    visualize: bool,
    skip_existing: bool,
) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    output_path = output_dir / f"{stem}_tracks.pt"
    if skip_existing and output_path.exists():
        print(f"[skip] Existing track condition: {output_path}")
        return

    frames, fps = read_video_frames(str(video_path), max_len, target_fps, max_res)
    frames = np.asarray(frames)
    video_tensor = video_numpy_to_tensor(frames).to(device)

    video_tracks, video_visibility = run_cotracker_on_video(
        cotracker=cotracker,
        video_tensor=video_tensor,
        grid_size=grid_size,
        grid_query_frame=grid_query_frame,
        backward_tracking=backward_tracking,
    )

    torch.save(
        {
            "video_tracks": video_tracks.cpu(),
            "video_visibility": video_visibility.cpu(),
            "fps": fps,
            "source_video": str(video_path),
            "frame_shape": tuple(frames.shape[1:3]),
            "grid_size": grid_size,
            "grid_query_frame": grid_query_frame,
            "coordinate_format": "xy_pixels",
        },
        output_path,
    )

    if visualize:
        visualize_tracks(
            frames=frames,
            video_tracks=video_tracks.cpu(),
            video_visibility=video_visibility.cpu(),
            save_dir=output_dir,
            filename=f"{stem}_tracks_vis",
            fps=int(fps) if fps > 0 else 30,
            model_dir=model_dir,
        )


def main(args: argparse.Namespace) -> None:
    rank, local_rank, world_size, initialized_here = init_distributed()
    device = get_device(local_rank)

    csv_path = Path(args.csv_path).expanduser().resolve()
    video_root = Path(args.video_root).expanduser().resolve() if args.video_root else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()

    video_paths = load_manifest(csv_path, video_root=video_root, rank=rank, world_size=world_size)
    cotracker = load_cotracker(model_dir=model_dir, device=device)

    for video_path in video_paths:
        try:
            print(f"[Rank {rank}] Processing tracks: {video_path}")
            process_video(
                video_path=video_path,
                output_dir=output_dir,
                cotracker=cotracker,
                device=device,
                model_dir=model_dir,
                max_len=args.max_len,
                target_fps=args.target_fps,
                max_res=args.max_res,
                grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame,
                backward_tracking=args.backward_tracking,
                visualize=args.visualize,
                skip_existing=args.skip_existing,
            )
            print(f"[Rank {rank}] Finished tracks: {video_path}")
        except Exception as exc:
            print(f"[Rank {rank}] Error processing tracks for {video_path}: {exc}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if initialized_here:
        torch.distributed.destroy_process_group()
    if rank == 0:
        print("=== Track inference finished ===")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CoTracker trajectory conditions from raw videos."
    )
    parser.add_argument("--csv_path", required=True, help="CSV manifest with a video_path column.")
    parser.add_argument(
        "--video_root",
        default=None,
        help="Optional root for relative video_path entries. Defaults to the manifest directory.",
    )
    parser.add_argument("--output_dir", default="./outputs/tracks", help="Output directory.")
    parser.add_argument("--model_dir", default=str(DEFAULT_COTRACKER_REPO), help="Local CoTracker repo path.")
    parser.add_argument("--max_res", type=int, default=1280, help="Resize so the longest side is at most this.")
    parser.add_argument("--max_len", type=int, default=-1, help="Maximum frames to process; -1 means all.")
    parser.add_argument("--target_fps", type=int, default=-1, help="Target FPS; -1 preserves source FPS.")
    parser.add_argument("--grid_size", type=int, default=25, help="CoTracker query grid size.")
    parser.add_argument("--grid_query_frame", type=int, default=0, help="Frame index used to seed grid queries.")
    parser.add_argument("--backward_tracking", action="store_true", help="Track backward and forward in time.")
    parser.add_argument("--visualize", action="store_true", help="Also save track visualization videos.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip videos with existing track output.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
