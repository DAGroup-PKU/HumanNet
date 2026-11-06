# HumanWAM-Vanilla Configurations

This page maps the report's experiments to their training configurations.
See the [main README](../README.md#humanwam-vanilla) for installation, checkpoints,
training commands, and evaluation.

## Experiment Configurations

Each exploration series has its own directory under `configs/`.

| Series | Settings | Config files |
| --- | --- | --- |
| [IA scaling](configs/scaling_ia/) | Isolated attention, relative actions. 100 / 500 / 1,000 / 5,000 / 10,000 hours | `pretrain_<hours>h.yaml`, `posttrain.yaml`, `posttrain_no_ego.yaml` |
| [IDM scaling](configs/scaling_idm/) | IDM attention, absolute actions. 100 / 500 / 1,000 / 5,000 / 20,500 / 50,000 hours | `pretrain_<hours>h.yaml`, `posttrain.yaml`, `posttrain_no_ego.yaml` |
| [Task and scene diversity](configs/diversity/) | Matched diversity at 100 / 500 / 1,000 hours. Scaled diversity and the shared 5,000-hour setting reuse IA scaling configs | `pretrain_matched_<hours>h.yaml`, `posttrain_12k.yaml`, `posttrain_15k.yaml`, `series.yaml` |
| [Demonstration efficiency](configs/demonstration_efficiency/) | No-Ego / 500h / 5,000h initialization × 5 / 10 / 25 / 50 demonstrations per task | `posttrain_<initialization>_<count>demos.yaml` |
| [Human action supervision](configs/action_supervision/) | 0% / 50% / 100% human action supervision during pretraining | `pretrain_<percent>percent.yaml`, `posttrain.yaml` |
| [Multi-view adaptation](configs/multiview/) | Head only, wrist conditioning, or wrist prediction. Direct post-training or EgoPro mid-training followed by post-training | `midtrain_idm_<view>.yaml`, `posttrain_<attention>_<view>.yaml`, `posttrain_idm_<view>_midtrained.yaml` |
| [Attention and action representation](configs/attention/) | IA, Mutual, IDM, and Optional-IDM with the relative/absolute action settings provided in this directory | `pretrain_<attention>_<representation>.yaml`, `posttrain_<attention>_<representation>.yaml` |
| [Action-expert initialization](configs/action_initialization/) | Interpolated video weights or random initialization | `pretrain_interpolated.yaml`, `pretrain_random.yaml`, and matching `posttrain_*.yaml` |

For example, [scaling_ia/pretrain_1000h.yaml](configs/scaling_ia/pretrain_1000h.yaml)
selects 1,000-hour IA pretraining, while
[scaling_ia/posttrain.yaml](configs/scaling_ia/posttrain.yaml) defines the shared
head-only RoboTwin post-training recipe for that series.

Exploration post-training uses **12k steps without augmentation**, except for
the historical diversity runs listed in [diversity/series.yaml](configs/diversity/series.yaml),
which include 15k schedules. The [experiment index](configs/experiments.yaml)
records evaluation settings and checkpoint selection.

## Main Model Configurations

| Config | Setting |
| --- | --- |
| [scaling_idm/pretrain_50000h.yaml](configs/scaling_idm/pretrain_50000h.yaml) | 50K-hour IDM pretraining. `pretrain.yaml` inherits this recipe |
| [midtrain.yaml](configs/midtrain.yaml) | EgoPro W0 multi-view conditioning, 10k steps |
| [robotwin.yaml](configs/robotwin.yaml) | W0 multi-view RoboTwin post-training with augmentation, 24k steps |
| [robodojo.yaml](configs/robodojo.yaml) | W0 multi-view RoboDojo post-training with OpenWAM-style augmentation, 60k steps |

The `multiview/` configs describe the exploration ablations.
`midtrain.yaml` and `robotwin.yaml` define the final model's W0 training route.

## Config Structure

- `pretrain_*`, `midtrain_*`, and `posttrain_*` identify the training stage.
- [`_base_/`](configs/_base_/) contains shared model, data, and training settings.
  `_base_` paths are relative to the config file. Values in the selected config
  override inherited values, and command-line overrides apply last.
- `paths` contains local data, checkpoint, statistics, cache, and output paths.
- [`corpus_indices.yaml`](configs/corpus_indices.yaml) maps corpus sizes to the
  [published subset tables](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K/tree/main/subsets).

Use the matching dataset selection and normalization statistics for each setting.
Demonstration-efficiency experiments use the specified per-task subset, and the
50% supervision experiment uses a fixed episode mask. Keep the configured
learning-rate schedule when selecting an intermediate checkpoint.

The published `subsets/<hours>h.parquet` tables cover 100 / 500 / 1,000 / 5,000 /
10,000 / 20,500 / 50,000 hours. Each row selects a LeRobot part, episode, and
frame interval, with its task text and hand-supervision tag. The historical
100h–5,000h selections are complete. Video/action payload upload is ongoing.

The published 10,000h selection is distinct from the report's historical IA
10kh setting, whose binding remains pending. Matched-diversity, aligned 1,000h,
and EgoPro multi-view selections are also pending.

🚧 Pretraining loaders, some ablation processors, and exploration
evaluation are still being prepared. Unfinished bindings remain `???`.
Source-mix alignment and real-robot pretraining are outside this configuration index.
