# RoVid-X Condition Extraction

This directory contains example scripts for extracting reusable conditions from raw robot videos:

- **tracks**: CoTracker trajectories saved as `.pt`
- **depth**: Video Depth Anything depth maps saved as `.npz`

The intended flow is:

```text
raw video -> trajectory condition + depth condition -> conditioned video generation
```

This matches the RoVid-X data pipeline goal described in [Rethinking Video Generation Model for the Embodied World](https://arxiv.org/abs/2601.15282): retain physical structure and motion cues as conditions for robot-oriented video generation.

## Directory Layout

```text
src/dataset/rovid-x/
├── robot.csv                              # Small example manifest with relative video paths
├── infer_track_cotracker_multi_gpu.py     # Raw video -> CoTracker trajectories
├── infer_depth_multi_gpu.py               # Raw video -> depth maps
└── tools/
    ├── facebook_co-tracker/               # Local CoTracker source checkout
    └── video_depth_anything/              # Local Video Depth Anything source checkout
```

## Manifest Format

`robot.csv` is a minimal example. Keep paths relative where possible:

```csv
caption_short_list,video_path,frames,task
Example raw first-person robot video for condition extraction.,videos/example_000001.mp4,300,example_task
```

By default, relative `video_path` entries are resolved against the CSV file directory. Use `--video_root` if the videos live elsewhere:

```bash
--video_root /path/to/raw/videos
```

Do not commit private cluster paths, user home paths, or unpublished manifest files.

## Weights

Place model weights at the relative paths below, or pass explicit paths with CLI flags:

```text
tools/facebook_co-tracker/weights/scaled_offline.pth
tools/video_depth_anything/metric_video_depth_anything_vitl.pth
```

The CoTracker weight can be downloaded from the upstream CoTracker release:

```bash
mkdir -p tools/facebook_co-tracker/weights
wget -O tools/facebook_co-tracker/weights/scaled_offline.pth \
  https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
```

Download the Video Depth Anything checkpoint from its official release and place it at:

```text
tools/video_depth_anything/metric_video_depth_anything_vitl.pth
```

Large checkpoint files are intentionally ignored by Git.

## Run Track Extraction

Single process:

```bash
python infer_track_cotracker_multi_gpu.py \
  --csv_path robot.csv \
  --output_dir outputs/tracks \
  --visualize
```

Multi-GPU:

```bash
torchrun --nproc_per_node=4 infer_track_cotracker_multi_gpu.py \
  --csv_path robot.csv \
  --output_dir outputs/tracks \
  --visualize
```

Each output `.pt` file contains:

- `video_tracks`: CoTracker trajectories, shape `[1, T, N, 2]`, in `xy` pixel coordinates
- `video_visibility`: visibility mask, shape `[1, T, N]`
- `fps`, `source_video`, `frame_shape`, `grid_size`, `grid_query_frame`

## Run Depth Extraction

Single process:

```bash
python infer_depth_multi_gpu.py \
  --csv_path robot.csv \
  --output_dir outputs/depth
```

Multi-GPU:

```bash
torchrun --nproc_per_node=4 infer_depth_multi_gpu.py \
  --csv_path robot.csv \
  --output_dir outputs/depth
```

Each output `.npz` file contains:

- `depth`: depth maps, shape `[T, H, W]`, stored as `float32`
- `fps`, `source_video`, `frame_shape`

By default the depth script also writes a colorized `*_depth_vis.mp4` for quick inspection. Use `--no_vis` to skip visualization.

## Notes

- CoTracker expects RGB video tensors in pixel space `[0, 255]`; the script preserves this convention.
- Depth inference stores numeric depth maps separately from visualization videos so the maps can be used as conditioning signals.
- Both scripts support `--skip_existing` for resumable batch processing.
