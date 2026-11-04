<div align="center">

<p><img src="assets/branding/wordmark-v3.svg" width="580" alt="Open-HumanScale"></p>

### An Open Recipe for Scaling Human and Robot Experience<br>for Embodied Foundation Models

[![arXiv](https://img.shields.io/badge/arXiv-Coming_soon-B31B1B?style=flat&logo=arxiv&logoColor=white)](#citation)
[![Project Page](https://img.shields.io/badge/%F0%9F%8C%90-Project_Page-4285F4?style=flat)](https://jiankai-ss.github.io/open-humanscale/)
[![Hugging Face Open-HumanScale-50k](https://img.shields.io/badge/Hugging_Face-Open--HumanScale--50k_%C2%B7_Coming_soon-FFD21E?style=flat&logo=huggingface)](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanScale-50k)
[![Hugging Face Models](https://img.shields.io/badge/Hugging_Face-Model_Collection-FFD21E?style=flat&logo=huggingface)](https://huggingface.co/collections/DAGroup-PKU/open-humanscale)
[![ModelScope Corpus](https://img.shields.io/badge/ModelScope-Corpus_%C2%B7_Coming_soon-624AFF?style=flat&logo=modelscope&logoColor=white)](https://modelscope.cn/datasets/DAGroup-PKU/EgoCorpus-50kH-LeRobot-v3)

</div>

![Open-HumanScale overview: 50k-hour egocentric corpus, pure and hybrid scaling, RoboTwin results, and zero-shot real-robot generalization](assets/teaser.png)

**Open-HumanScale** shows how scaling human experience improves robot
**generalization, zero-shot execution, and post-training performance**.
Scaling egocentric pretraining from **100 to 50,000 hours** raises Scene-OOD
success from **49.1% to 73.2%**. Adding human data to a fixed robot corpus
improves zero-shot success from **11.48% to 13.94% on RoboTwin** and from
**20.0% to 30.5% across 20 real-world tasks**.

> 🏆 **[RoboTwin 2.0 Benchmark — Rank #1](https://robotwin-platform.github.io/leaderboard).**
> **HumanWAM-Goal** achieves **88.65% average success**, with **92.70% Clean / 84.60% Random**.

https://github.com/user-attachments/assets/469f1f1b-cbe2-4f89-aaaf-6cbdb3a2e50e

## 🔥 News

- **[2026/11/05]** We release the **Open-HumanScale technical report**!
  [Model weights](https://huggingface.co/collections/DAGroup-PKU/open-humanscale)
  and the [Open-HumanScale-50k corpus](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanScale-50k)
  are being uploaded. We are actively organizing the code and documentation.

## Table of Contents

- [Open-HumanScale-50k Corpus](#open-humanscale-50k-corpus)
- [HumanWAM-Vanilla](#humanwam-vanilla)
  - [Weight Download](#vanilla-weight-download)
  - [Environment Setup & Quick Start](#vanilla-quick-start)
  - [Training](#vanilla-training)
  - [Inference](#vanilla-inference)
  - [Reproduction Guide](#vanilla-reproduction-guide)
- [HumanWAM-Goal](#humanwam-goal)
  - [Weight Download](#goal-weight-download)
  - [Environment Setup & Quick Start](#goal-quick-start)
  - [Inference](#goal-inference)
- [Release Plan](#release-plan)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Open-HumanScale-50k Corpus

**Open-HumanScale-50k: 50,000 hours · 35.9M clips · Video, hand motion, and language.**

Curated from HumanNet, EgoSuite-Open100K, EgoDex, Gen-HumanEgo, and EgoVerse,
with hand-visibility, motion-stability, and activity filtering, followed by
cross-source alignment to a unified bimanual action representation.

![Construction of Open-HumanScale-50k: five data sources, data curation, and cross-source action alignment](assets/data_overview.png)

*Construction of Open-HumanScale-50k. (a) Five data sources and their retained
hours. (b) Curation by hand visibility, motion stability, and manipulation
activity. (c) Alignment of camera frames, wrist axes, and gripper openness.*

- **Corpus:** [Hugging Face](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanScale-50k).
- **Mirror:** [ModelScope](https://modelscope.cn/datasets/DAGroup-PKU/EgoCorpus-50kH-LeRobot-v3) — not uploaded yet.
- **Format:** LeRobot v3, with compressed MP4 video and language annotations.

The Hugging Face upload is in progress. Download instructions are being prepared.

## HumanWAM-Vanilla

<a id="vanilla-weight-download"></a>
### Weight Download

Pretrained, RoboTwin post-trained, scaling, and ablation checkpoints are being
uploaded to **one model repository**, organized under separate paths and linked
from the [Open-HumanScale Collection](https://huggingface.co/collections/DAGroup-PKU/open-humanscale).

| Checkpoint | Clean2Clean | Clean2Random | Average | Status |
| --- | ---: | ---: | ---: | --- |
| HumanWAM-Vanilla · RoboTwin post-trained | **92.08%** | **68.88%** | **80.48%** | [Uploading](https://huggingface.co/collections/DAGroup-PKU/open-humanscale) |
| HumanWAM-Vanilla · egocentric pretrained | — | — | — | [Uploading](https://huggingface.co/collections/DAGroup-PKU/open-humanscale) |

Results are reported in the technical report; both RoboTwin settings use the
same post-trained checkpoint. Direct download paths are being prepared.

Scaling human pretraining from 100 to 50,000 hours improves Scene-OOD success
from **49.1% to 73.2%** in the controlled IDM study:

![Egocentric scaling under inverse-dynamics attention](assets/ego_scaling_idm.png)

<a id="vanilla-quick-start"></a>
### Environment Setup & Quick Start

🚧 Work in progress.

<a id="vanilla-training"></a>
### Training

The portable launcher runs directly on your own CUDA server. Checkpoint downloads
and data/cache preparation are still being organized; the commands below require
a compatible full RoboTwin policy, task data, normalization statistics, text
embeddings, and merged-video caches.

After installing the environment, start from the repository root:

```bash
cd humanwam_vanilla
cp configs/paths.example.yaml configs/paths.local.yaml
# Fill in your local resource paths, then inspect the resolved configuration.
python scripts/train.py --paths configs/paths.local.yaml --print-config

# Single GPU (subject to available GPU memory).
NPROC_PER_NODE=1 bash scripts/train.sh --paths configs/paths.local.yaml batch_size=1

# Four GPUs on one server; uses the existing whole-model FSDP implementation.
NPROC_PER_NODE=4 bash scripts/train.sh --paths configs/paths.local.yaml \
  batch_size=1 gradient_accumulation_steps=16
```

`batch_size` is **per GPU**. Effective batch size is
`batch_size × total GPUs × gradient_accumulation_steps`; adjust it for your
hardware and experiment. These examples illustrate launching, not paper-result
reproduction. Add `--print-config` to the launcher to display its resolved settings
without training.

For multiple servers, run the same command on each server with `NNODES`, a unique
`NODE_RANK` (starting at 0), and the same `MASTER_ADDR` and `MASTER_PORT` set.
All servers must have the environment and required files at the configured paths;
use a shared output directory for distributed checkpoint saving. The launcher
uses [PyTorch torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html).

<a id="vanilla-inference"></a>
### Inference

🚧 Work in progress.

<a id="vanilla-reproduction-guide"></a>
### Reproduction Guide

🚧 Work in progress.

## HumanWAM-Goal

<a id="goal-weight-download"></a>
### Weight Download

Hybrid pretrained / zero-shot checkpoints and RoboTwin post-trained weights are
being uploaded under Goal-specific paths in the same model repository, linked
from the [Open-HumanScale Collection](https://huggingface.co/collections/DAGroup-PKU/open-humanscale).

| Checkpoint | Clean2Clean | Clean2Random | Average | Status |
| --- | ---: | ---: | ---: | --- |
| HumanWAM-Goal · RoboTwin post-trained | **92.70%** | **84.60%** | **88.65%** | [Uploading](https://huggingface.co/collections/DAGroup-PKU/open-humanscale) |

Results are reported in the technical report under the official 10,000-episode
protocol. Direct download paths are being prepared.

Without target-task parameter updates, HumanWAM-Goal achieves **13.94% mean
zero-shot success on RoboTwin** and **30.5% across 20 real-world tasks**.

<p align="center">
  <img src="assets/zero_shot_robotwin_success.png" width="49%" alt="RoboTwin zero-shot success under hybrid data scaling">
  <img src="assets/zero_shot_robotwin_coverage.png" width="49%" alt="RoboTwin zero-shot task coverage under hybrid data scaling">
</p>

<a id="goal-quick-start"></a>
### Environment Setup & Quick Start

🚧 Work in progress.

<a id="goal-inference"></a>
### Inference

🚧 Work in progress.

## Release Plan

- [ ] Open-HumanScale-50k corpus — uploading to Hugging Face; ModelScope pending.
- [ ] Pretrained and RoboTwin post-trained weights — uploading.
- [ ] HumanWAM-Vanilla code, training, inference, and reproduction guide — in progress.
- [ ] HumanWAM-Goal code, environment setup, and inference — in progress.

## License

Open-HumanScale is released under the [Apache License 2.0](Apache-2.0.txt).
Existing component notices and license terms remain applicable. Corpus and
checkpoint terms will be specified in their release cards.

## Citation

If you find Open-HumanScale useful, please cite our technical report.
Citation details are coming soon.

## Acknowledgements

We thank [FastWAM](https://github.com/yuantianyuan01/FastWAM) and
[ViGAR](https://github.com/DAGroup-PKU/ViGAR) for their open-source contributions.
