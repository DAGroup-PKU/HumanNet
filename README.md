<div align="center">

<p><img src="assets/branding/wordmark-v3.svg" width="580" alt="Open-HumanWAM"></p>

### An Open Recipe for Scaling Human and Robot Experience<br>for Embodied Foundation Models

[![arXiv](https://img.shields.io/badge/arXiv-2611.06817-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2611.06817)
[![Project Page](https://img.shields.io/badge/%F0%9F%8C%90-Project_Page-4285F4?style=flat)](https://dagroup-pku.github.io/open-humanwam/)
[![Hugging Face Open-HumanWAM-50K](https://img.shields.io/badge/Hugging_Face-Open--HumanWAM--50K_%C2%B7_Coming_soon-FFD21E?style=flat&logo=huggingface)](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K)
[![Hugging Face Models](https://img.shields.io/badge/Hugging_Face-Model_Collection-FFD21E?style=flat&logo=huggingface)](https://huggingface.co/collections/DAGroup-PKU/open-humanwam)
[![ModelScope Corpus](https://img.shields.io/badge/ModelScope-Corpus_%C2%B7_Coming_soon-624AFF?style=flat&logo=modelscope&logoColor=white)](https://modelscope.cn/datasets/DAGroup-PKU/EgoCorpus-50kH-LeRobot-v3)

</div>

![Open-HumanWAM overview: 50k-hour egocentric corpus, pure and hybrid scaling, RoboTwin results, and zero-shot real-robot generalization](assets/teaser.png)

**Open-HumanWAM** shows how scaling human experience improves robot
**generalization, zero-shot execution, and post-training performance**.
Scaling egocentric pretraining from **100 to 50,000 hours** raises Scene-OOD
success from **49.1% to 73.2%**. Adding human data to a fixed robot corpus
improves zero-shot success from **11.48% to 13.94% on RoboTwin** and from
**20.0% to 30.5% across 20 real-world tasks**.

> 🏆 **[RoboTwin 2.0 Benchmark — Rank #1](https://robotwin-platform.github.io/leaderboard).**
> **HumanWAM-Goal** achieves **88.65% average success**, with **92.70% Clean / 84.60% Random**.

https://github.com/user-attachments/assets/469f1f1b-cbe2-4f89-aaaf-6cbdb3a2e50e

## 🔥 News

- **[2026/11/05]** We release the **[Open-HumanWAM technical report](https://arxiv.org/abs/2611.06817)**!
  [Model weights](https://huggingface.co/collections/DAGroup-PKU/open-humanwam)
  and the [Open-HumanWAM-50K corpus](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K)
  are being uploaded. We are actively organizing the code and documentation.

## Table of Contents

- [Open-HumanWAM-50K Corpus](#open-humanwam-50k-corpus)
- [HumanWAM-Vanilla](#humanwam-vanilla)
  - [Model Checkpoints](#vanilla-weight-download)
  - [Environment Setup](#vanilla-environment-setup)
  - [Training](#vanilla-training)
  - [Evaluation](#vanilla-evaluation)
  - [Reproduction Guide](#vanilla-reproduction-guide)
- [HumanWAM-Goal](#humanwam-goal)
  - [Weight Download](#goal-weight-download)
  - [Environment Setup & Quick Start](#goal-quick-start)
  - [Inference](#goal-inference)
- [Release Plan](#release-plan)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Open-HumanWAM-50K Corpus

**Open-HumanWAM-50K: 50,000 hours · 35.9M clips · Video, hand motion, and language.**

Curated from HumanNet, EgoSuite-Open100K, EgoDex, Gen-HumanEgo, and EgoVerse,
with hand-visibility, motion-stability, and activity filtering, followed by
cross-source alignment to a unified bimanual action representation.

![Construction of Open-HumanWAM-50K: five data sources, data curation, and cross-source action alignment](assets/data_overview.png)

- **Hugging Face:** [Open-HumanWAM-50K](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K) — uploading.
- **ModelScope:** [Open-HumanWAM-50K](https://modelscope.cn/datasets/DAGroup-PKU/EgoCorpus-50kH-LeRobot-v3) — coming soon.

The corpus uses LeRobot v3, with compressed MP4 video and language annotations.
Download instructions are being prepared.

## HumanWAM-Vanilla

<a id="vanilla-weight-download"></a>
### Model Checkpoints

We are releasing the **RoboTwin post-trained checkpoints for the egocentric
pretraining exploration studies** in our [technical report](https://arxiv.org/abs/2611.06817),
together with the final **HumanWAM-Vanilla** checkpoints from **50,000-hour
egocentric pretraining**, **mid-training**, and **RoboTwin / RoboDojo post-training**.

| Checkpoint | Training stage | Download / status |
| --- | --- | --- |
| Exploration series | Scaling and ablation studies | [Collection](https://huggingface.co/collections/DAGroup-PKU/open-humanwam) · uploading |
| HumanWAM-Vanilla-Base | 50kh pretraining **pre40k**; W0 mid-training **mid10k** | [Base](https://huggingface.co/DAGroup-PKU/HumanWAM-Vanilla-Base) · weights available |
| HumanWAM-Vanilla-RoboTwin | **post21k** · **92.08 / 68.88 / 80.48** | [RoboTwin](https://huggingface.co/DAGroup-PKU/HumanWAM-Vanilla-RoboTwin) · weights available |
| HumanWAM-Vanilla-RoboDojo | 50kh → W0 → OpenWAM-style augmentation → **post50k** | [RoboDojo](https://huggingface.co/DAGroup-PKU/HumanWAM-Vanilla-RoboDojo) · model card only; weights pending |

RoboTwin scores are Clean / Random / average success (%), from the archived
**50-episode-per-task** evaluation. The manuscript describes 100 episodes per task;
this protocol difference remains under review. Its checkpoint uses **unaligned
native wrists**. RoboDojo's manuscript scores (**14.17% success / 19.00 progress**)
are not yet bound to a released checkpoint; they are not a verified post50k result.

Download the available weights after [installing the environment](#vanilla-environment-setup):

```bash
# From humanwam_vanilla/, with the training environment activated.
huggingface-cli download DAGroup-PKU/HumanWAM-Vanilla-Base \
  pretrained/step_040000.pt midtrained/step_010000.pt --local-dir ../checkpoints/vanilla-base
huggingface-cli download DAGroup-PKU/HumanWAM-Vanilla-RoboTwin \
  model.pt config.yaml dataset_stats.json --local-dir ../checkpoints/vanilla-robotwin
```

The RoboTwin release includes its evaluation configuration and normalization
statistics. Training data and cache preparation are being organized.

Scaling human pretraining from 100 to 50,000 hours improves Scene-OOD success
from **49.1% to 73.2%** in the controlled IDM study:

![Egocentric scaling under inverse-dynamics attention](assets/ego_scaling_idm.png)

<a id="vanilla-environment-setup"></a>
### Environment Setup

**1. Install**

Clone the repository and install the training and RoboTwin simulation environments.

```bash
git clone --branch open-humanscale https://github.com/DAGroup-PKU/HumanNet.git
cd HumanNet/humanwam_vanilla
bash scripts/setup_environment.sh train
bash scripts/setup_environment.sh sim
```

**2. Prepare Wan and RoboTwin**

From `humanwam_vanilla/`, download the Wan components and set up RoboTwin with its simulation assets.

```bash
bash scripts/prepare_resources.sh wan
bash scripts/prepare_resources.sh robotwin
```

<a id="vanilla-training"></a>
### Training

Run from `humanwam_vanilla/`. Set the data, statistics, cache, and output
paths in the selected config.

#### Pretrain

Example: **50K-hour IDM pretraining**, using absolute camera-frame actions.

```bash
bash scripts/train.sh --config configs/scaling_idm/pretrain_50000h.yaml
```

Select other settings from the per-series [experiment configurations](humanwam_vanilla/README.md#experiment-configurations),
such as `scaling_ia/pretrain_1000h.yaml` or `scaling_idm/pretrain_5000h.yaml`.

#### Mid-train

Adapt the pretrained model to **W0 multi-view conditioning** with **10k steps
on EgoPro**. Head and wrist observations condition action prediction, while
future video prediction uses only the head view.

```bash
bash scripts/train.sh --config configs/midtrain.yaml \
  --init-checkpoint /path/to/pretrained.pt
```

#### Post-train

**RoboTwin exploration — 12k steps, without data augmentation.**

```bash
bash scripts/train.sh --config configs/scaling_ia/posttrain.yaml --init-checkpoint /path/to/pretrained.pt # Isolated attention, head-only
bash scripts/train.sh --config configs/scaling_idm/posttrain.yaml --init-checkpoint /path/to/pretrained.pt # IDM, head-only
bash scripts/train.sh --config configs/multiview/posttrain_idm_conditioning_midtrained.yaml --init-checkpoint /path/to/midtrained.pt # W0, multi-view
```

**RoboTwin final model — W0 multi-view + augmentation, 24k steps.**

```bash
bash scripts/train.sh --config configs/robotwin.yaml \
  --init-checkpoint ../checkpoints/vanilla-base/midtrained/step_010000.pt
```

**RoboDojo — W0 multi-view + augmentation.

```bash
bash scripts/train.sh --config configs/robodojo.yaml \
  --init-checkpoint ../checkpoints/vanilla-base/midtrained/step_010000.pt
```

Set `NPROC_PER_NODE` to your GPU count. To preserve the effective batch size,
adjust `batch_size × total GPUs × gradient_accumulation_steps` together.

<a id="vanilla-evaluation"></a>
### Evaluation

#### RoboTwin 2.0

Follow the [RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin)
for simulator installation and assets. Our [environment setup](#vanilla-environment-setup)
provides the pinned simulator checkout and policy dependencies.
For Blackwell GPUs, apply the [OIDN compatibility setup](docs/environment.md#blackwell-rendering).

Download the W0 policy, its configuration, and normalization statistics.
Run from `humanwam_vanilla/`:

```bash
../.venv/train/bin/huggingface-cli download DAGroup-PKU/HumanWAM-Vanilla-RoboTwin \
  model.pt config.yaml dataset_stats.json --local-dir ../checkpoints/vanilla-robotwin
```

Evaluate all 50 tasks with 50 episodes per task in each setting:

```bash
bash scripts/eval_robotwin.sh ../checkpoints/vanilla-robotwin clean ../runs/robotwin-clean
bash scripts/eval_robotwin.sh ../checkpoints/vanilla-robotwin random ../runs/robotwin-random
../.venv/train/bin/python scripts/summarize_robotwin.py ../runs/robotwin-{clean,random}/shard-* \
  --output ../runs/robotwin-summary.json
```

Each run saves `result.json` and `policy.log` per shard. These commands evaluate
**HumanWAM-Vanilla-RoboTwin (W0)** under the archived protocol.
Exploration checkpoints and Scene-OOD evaluation are being integrated.

#### RoboDojo

Follow [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) and
[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) for the Isaac Sim environment
and assets. RoboDojo weights and public evaluation commands are coming soon.

<a id="vanilla-reproduction-guide"></a>
### Reproduction Guide

We will provide a more detailed reproduction guide in a future update.

For the exploration studies, see the per-series
[HumanWAM-Vanilla configurations](humanwam_vanilla/README.md#experiment-configurations)
and [published corpus subsets](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K/tree/main/subsets).
Pretraining and ablation integration are in progress.


## HumanWAM-Goal

<a id="goal-weight-download"></a>
### Weight Download

The RoboTwin post-trained checkpoint and its goal cache are released as
[DAGroup-PKU/HumanWAM-Goal-RoboTwin](https://huggingface.co/DAGroup-PKU/HumanWAM-Goal-RoboTwin);
hybrid pretrained / zero-shot checkpoints are being prepared. All are linked from the
[Open-HumanWAM Collection](https://huggingface.co/collections/DAGroup-PKU/open-humanwam).

| Checkpoint | Clean2Clean | Clean2Random | Average | Status |
| --- | ---: | ---: | ---: | --- |
| HumanWAM-Goal · RoboTwin post-trained | **92.70%** | **84.60%** | **88.65%** | [HumanWAM-Goal-RoboTwin](https://huggingface.co/DAGroup-PKU/HumanWAM-Goal-RoboTwin) |

Results are reported in the technical report under the official 10,000-episode
protocol.

Without target-task parameter updates, HumanWAM-Goal achieves **13.94% mean
zero-shot success on RoboTwin** and **30.5% across 20 real-world tasks**.

<p align="center">
  <img src="assets/zero_shot_robotwin_success.png" width="49%" alt="RoboTwin zero-shot success under hybrid data scaling">
  <img src="assets/zero_shot_robotwin_coverage.png" width="49%" alt="RoboTwin zero-shot task coverage under hybrid data scaling">
</p>

<a id="goal-quick-start"></a>
### Environment Setup & Quick Start

```bash
cd humanwam_goal
bash scripts/setup_environment.sh /path/to/new/environment   # Python 3.13, PyTorch 2.10, CUDA 12.8
python examples/make_sample.py
goalwam infer --checkpoint "$GOALWAM_CHECKPOINT" --sample examples/sample/observation.json
```

See [HumanWAM-Goal setup](humanwam_goal/README.md#environment-setup) for the requirements
(Hopper GPU, GLIBC ≥ 2.34) and [Quick Start](humanwam_goal/README.md#quick-start).

<a id="goal-inference"></a>
### Inference

The Python API, HTTP server and observation format are documented in
[HumanWAM-Goal inference](humanwam_goal/README.md#inference). RoboTwin 2.0 evaluation
with the released goal cache, and bit-exact reproduction of the published actions, are in
[RoboTwin 2.0 Evaluation](humanwam_goal/README.md#robotwin-20-evaluation).

## Release Plan

- [ ] Open-HumanWAM-50K corpus — uploading to Hugging Face; ModelScope pending.
- [x] HumanWAM-Goal RoboTwin post-trained weights.
- [x] HumanWAM-Vanilla Base pre40k / W0 mid10k and RoboTwin post21k weights.
- [ ] Exploration-series and RoboDojo weights, training resources, and remaining statistics.
- [ ] HumanWAM-Vanilla code, training, inference, and reproduction guide — in progress.
- [ ] HumanWAM-Goal code, environment setup, and inference — in progress.

## License

Open-HumanWAM is released under the [Apache License 2.0](Apache-2.0.txt).
Existing component notices and license terms remain applicable. Corpus and
checkpoint terms will be specified in their release cards.

## Citation

If you find Open-HumanWAM useful, please cite our technical report.
Citation details are coming soon.

## Acknowledgements

We thank [FastWAM](https://github.com/yuantianyuan01/FastWAM) and
[ViGAR](https://github.com/DAGroup-PKU/ViGAR) for their open-source contributions.
