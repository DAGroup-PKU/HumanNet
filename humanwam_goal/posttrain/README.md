# HumanWAM-Goal Post-Training

Post-train [HumanWAM-Goal](../README.md) from the pretrained safetensors checkpoint
(`DAGroup-PKU/HumanWAM-Goal-Base`) on RoboTwin 2.0 `demo_clean` demonstrations (`clean_50`,
50 episodes × 50 tasks). Both the pretrained checkpoint and saved post-training checkpoints
share the same safetensors layout and load directly with `goalwam.GoalWAM` and the
[RoboTwin 2.0 evaluation](../README.md#robotwin-20-evaluation).

## Environment Setup

Post-training reuses the native `humanwam_goal` environment:

```bash
cd humanwam_goal
bash scripts/setup_environment.sh /path/to/new/environment
source /path/to/new/environment/bin/activate
pip install -e "posttrain[convert]"
```

## Data Preparation

### 1. RoboTwin `clean_50` LeRobot v3.0 Datasets

`--data` expects LeRobot v3.0 `aloha_agilex` `clean_50` datasets at 30 FPS
(`observation.images.cam_high`, `cam_left_wrist`, `cam_right_wrist` and 49-D
`observation.state` / `action` with `command_openness` grippers).

To convert official RoboTwin 2.0 `<task>/aloha-agilex_clean_50.zip` archives to this layout:

```bash
python -m goalwam_train.convert \
  --raw-root /path/to/robotwin_raw \
  --output /path/to/robotwin_clean_50 \
  --workers 8
```

### 2. Pre-Generated Training Goal Cache (`--goal-cache`)

`train_goal_cache.tar` (published in
[`DAGroup-PKU/HumanWAM-Goal-RoboTwin`](https://huggingface.co/DAGroup-PKU/HumanWAM-Goal-RoboTwin))
provides one 224×288 head-camera goal image per `clean_50` training episode (2,500 PNGs
across 50 tasks, generated from the initial head image and instruction only, `manifest_sha256`:
`dd61e73b8950f28b4ba67d5be375b2489b05a14d9967edf44d9878bca5560ceb`):

```bash
hf download DAGroup-PKU/HumanWAM-Goal-RoboTwin train_goal_cache.tar --local-dir downloads
tar -xf downloads/train_goal_cache.tar -C downloads   # -> downloads/train_goal_cache/
python -m goalwam_train.goal_cache verify --root downloads/train_goal_cache
```

When `--goal-cache` is supplied, training samples the pre-generated goal on 80% of visits
and the episode's recorded terminal frame on 20%. When omitted, training uses the recorded
terminal frame (100%).

## Normalization (`robot.json`)

State and relative-action normalization uses unclipped 1st/99th percentile bounds
(`x_norm = 2 * (x - low) / (high - low) - 1`):

| Dimensions | Channels | Rule |
| --- | --- | --- |
| `0–5`, `8–13` | Left / right 6-DoF arm joints | `q01` / `q99` over `clean_50` (2,500 episodes, 549,787 frames) |
| `26–32`, `34–40` | Left / right arm EEF pose (`head_camera` NWU) | Copied from `--pretrained`'s shared EEF bounds (`robot.json`) |
| `7`, `15`, `33`, `41` | Joint & EEF grippers (`command_openness`) | Fixed `[0.0, 1.0]` (`0` closed, `1` open) |
| `42–48` | Static head-camera pose (state only; masked in model I/O) | Identity pose `[0, 0, 0, 0, 0, 0, 1]` |
| `6`, `14`, `16–25` | Unused arm/torso/head/chassis slots | Masked (`0.0`) |

Relative-action quantiles use the furthest target of each 48-step chunk,
`action[min(t + 47, length - 1)]` anchored at `state[t]`, so shorter-horizon near-zero
deltas do not narrow the action scale. Compute and verify `robot.json` standalone:

```bash
python -m goalwam_train.normalization \
  --data /path/to/robotwin_clean_50 \
  --pretrained /path/to/HumanWAM-Goal-Base \
  --output /path/to/outputs/robot.json
```

On the 50-task `clean_50` split with `HumanWAM-Goal-Base`, this produces `robot.json` with
SHA-256 `a9b20372b33f3a2618f763bf95f0b4b8da9225e3d038798b59075f1310b5e4c8` (matching
`HumanWAM-Goal-RoboTwin`).

## Training

```bash
torchrun --nproc-per-node 8 -m goalwam_train.train \
  --pretrained /path/to/HumanWAM-Goal-Base \
  --data /path/to/robotwin_clean_50 \
  --goal-cache /path/to/train_goal_cache \
  --output /path/to/outputs
```

- **Schedule & batching:** 36,000 steps (`--steps`), global `--batch-size 256`,
  `--micro-batch-size 16` per GPU with gradient accumulation, FSDP2 sharding (frozen BF16
  Reasoner, FP32 trainable master weights, `5×` LR on action/goal bridges, Power-EMA).
- **Conditioning:** includes episode tail windows (1/5/9/13-frame causal video prefixes and
  masked beyond-end actions), 10% instruction CFG dropout, and causal observation-history
  dropout/jitter (`0.5 s` and `2.0 s`).
- **Checkpoints:** writes `robot.json` first, then saves `checkpoint-XXXXXX/` every
  `--save-every` (2,000) steps directly in `goalwam` safetensors format:

```python
from goalwam import GoalWAM

model = GoalWAM("/path/to/outputs/checkpoint-036000")
```

## License

See the project README for the [license](../../README.md#license).
