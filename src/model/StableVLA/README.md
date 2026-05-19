<div align="center">

# [ICML 2026🔥🔥🔥] StableVLA: Towards Robust Vision-Language-Action Models without Extra Data

[![Paper](https://img.shields.io/badge/Paper-ICML%202026-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.18287)
[![Project Page](https://img.shields.io/badge/Project-Page-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white)](https://baiyingzhuying.github.io/StableVLA/)
[![HuggingFace](https://img.shields.io/badge/🤗-Model%20Weights-fcd022?style=for-the-badge)](https://huggingface.co/beikui12345/stablevla)

**Yiyang Fu<sup>1</sup>, Chubin Zhang<sup>2,3</sup>, Shukai Gong<sup>1</sup>, Yufan Deng<sup>1</sup>, Kaiwei Sun<sup>4</sup>, Qiyang Min, Qibin Hou<sup>5</sup>, Yansong Tang<sup>2</sup>, Jianan Wang<sup>3</sup>, Daquan Zhou<sup>1†</sup>**

<sup>1</sup>Peking University &emsp; <sup>2</sup>Tsinghua University &emsp; <sup>3</sup>Astribot &emsp; <sup>4</sup>Nanjing University &emsp; <sup>5</sup>Nankai University

<sup>†</sup>Corresponding author

</div>

---

> **📝 Paper:** https://arxiv.org/abs/2605.18287
> **🌍 Project page:** https://dagroup-pku.github.io/StableVLA/  
> **🤗 HuggingFace:** https://huggingface.co/DAGroup-PKU/StableVLA  
> **GitHub:** https://github.com/DAGroup-PKU/HumanNet/tree/main/src/model/StableVLA

---

## :loudspeaker: News

- **[2026/05]** StableVLA is accepted to **ICML 2026**! 🎉
- **[2026/05]** We release model weights and training code.

---

## 🌟 Table of Contents

- [:rocket: Quick Start](#rocket-quick-start)
- [:pencil: Data Preparation](#pencil-data-preparation)
- [⚓ Model Weights](#anchor-model-weights)
- [:fire: Training](#fire-training)
- [:mechanical_arm: Evaluation](#mechanical_arm-evaluation)
- [🌈 Results](#rainbow-results)
- [📝 Citation](#pencil2-citation)
- [:heart: Acknowledgment](#heart-acknowledgment)

---

## :rocket: Quick Start

### Conda Environment

```bash
conda create -n stablevla python=3.10.16 -y
conda activate stablevla
```

### Install Dependencies

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# Install package
git clone https://github.com/DAGroup-PKU/HumanNet.git
cd HumanNet/src/model/StableVLA
pip install -e .

pip install packaging ninja
ninja --version; echo $?  # Should return exit code "0"

# Install Flash Attention 2
pip install "flash-attn==2.5.5" --no-build-isolation
# If you have trouble, try: pip cache remove flash_attn
# Or download the matching .whl from https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.5
```

---

## :pencil: Data Preparation

### LIBERO

Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO) and required packages:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
```

Download the [LIBERO RLDS datasets](https://huggingface.co/datasets/openvla/modified_libero_rlds) (~10GB total):

```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

> **Note:** Remove the `modified_` prefix from the downloaded folder name.

### CALVIN

```bash
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=$(pwd)/calvin
cd $CALVIN_ROOT && sh install.sh

# Download CALVIN ABC→D dataset
cd $CALVIN_ROOT/dataset && sh download_data.sh ABC
```

RLDS format (~50GB): [zhouhongyi/calvin_abc_rlds](https://huggingface.co/datasets/zhouhongyi/calvin_abc_rlds)

> If you get `AttributeError: 'NoneType' object has no attribute 'eglQueryString'`, run:
> ```bash
> sudo apt-get install libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev
> ```

### Dataset Structure

```
.
├── data
│   ├── libero
│   │   ├── libero_spatial_no_noops/  1.0.0/
│   │   ├── libero_object_no_noops/   1.0.0/
│   │   ├── libero_goal_no_noops/     1.0.0/
│   │   └── libero_10_no_noops/       1.0.0/
│   └── calvin_abc/                   1.0.0/
```

---

## ⚓ Model Weights

All pretrained weights are on 🤗 HuggingFace at [`beikui12345/stablevla`](https://huggingface.co/beikui12345/stablevla):

| Folder | Description |
|--------|-------------|
| `pretrained_models/` | Pretrained VLM backbone (Fused IB-Adapter projector) |
| `spatial/` | StableVLA checkpoint for LIBERO-Spatial |
| `object/` | StableVLA checkpoint for LIBERO-Object |
| `goal/` | StableVLA checkpoint for LIBERO-Goal |
| `long/` | StableVLA checkpoint for LIBERO-Long |

Download with:

```python
from huggingface_hub import snapshot_download

# Download everything
snapshot_download(repo_id="beikui12345/stablevla", local_dir="./hf_weights")

# Or download a specific task checkpoint
snapshot_download(repo_id="beikui12345/stablevla", local_dir="outputs/spatial",
                  allow_patterns="spatial/*")
```

Place the downloaded folders as follows:

```
.
├── pretrained_models/
│   └── stablevla-fusedfan-projector/   # ← from pretrained_models/ on HF
└── outputs/
    ├── spatial/            # ← from spatial/ on HF
    ├── object/
    ├── goal/
    └── long/
```

---

## :fire: Training

StableVLA replaces the standard MLP projector in [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) with the **Fused IB-Adapter** module.

### VLM Pretraining (optional — skip if using HF weights above)

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
    --model.type "prism-qwen25-extra-dinosiglip-224px+0_5b+fusedfan-projector" \
    --dataset.type "llava-lvis4v-lrv" \
    --dataset.dataset_root_dir "data/vlm" \
    --run_root_dir "pretrained_models/stablevla-fusedfan-projector" \
    --stage "finetune" \
    --wandb_project "stablevla" \
    --wandb_entity "YOUR_WANDB_ENTITY"
```

### LIBERO Fine-tuning

```bash
data_name=libero_spatial_no_noops
# Replace with: libero_object_no_noops | libero_goal_no_noops | libero_10_no_noops

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes 1 --nproc-per-node 4 \
    vla-scripts/finetune_fused.py \
    --vlm_path pretrained_models/stablevla-fusedfan-projector \
    --config_file_path pretrained_models/configs_fusedfan \
    --data_root_dir data/libero \
    --dataset_name $data_name \
    --run_root_dir outputs \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_lora True \
    --use_fz False \
    --use_minivlm True \
    --image_aug True \
    --num_steps_before_decay 150000 \
    --max_steps 150005 \
    --save_freq 5000 \
    --save_latest_checkpoint_only False \
    --merge_lora_during_training True \
    --batch_size 16 \
    --grad_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --lora_rank 64 \
    --use_pro_version True \
    --wandb_entity "YOUR_WANDB_ENTITY" \
    --wandb_project "$data_name" \
    --run_id_note "stablevla" \
    > logs/stablevla--${data_name}.log 2>&1 &
```

**GPU memory guide:**

| VRAM | Recommended config |
|------|--------------------|
| 10–12 GB (e.g. RTX 3080) | `--batch_size 1 --grad_accumulation_steps 16` |
| 24 GB (e.g. RTX 3090/4090) | `--batch_size 4 --grad_accumulation_steps 4` |
| 40–80 GB (e.g. A100, H100) | `--batch_size 16 --grad_accumulation_steps 1` |

### CALVIN Fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes 1 --nproc-per-node 4 \
    vla-scripts/finetune_fused.py \
    --vlm_path pretrained_models/stablevla-fusedfan-projector \
    --config_file_path pretrained_models/configs_fusedfan \
    --data_root_dir data \
    --dataset_name calvin_abc \
    --run_root_dir outputs \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_lora True \
    --use_fz False \
    --use_minivlm True \
    --image_aug True \
    --num_steps_before_decay 150000 \
    --max_steps 150005 \
    --save_freq 5000 \
    --save_latest_checkpoint_only False \
    --merge_lora_during_training True \
    --batch_size 16 \
    --grad_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --lora_rank 64 \
    --use_pro_version True \
    --wandb_entity "YOUR_WANDB_ENTITY" \
    --wandb_project "calvin" \
    --run_id_note "stablevla" \
    > logs/stablevla--calvin.log 2>&1 &
```

---

## :mechanical_arm: Evaluation

### LIBERO — Clean

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --use_proprio True \
    --num_images_in_input 2 \
    --use_film False \
    --pretrained_checkpoint outputs/spatial \
    --task_suite_name libero_spatial \
    --use_pro_version True \
    > eval_logs/spatial_clean.log 2>&1 &
```

Replace `spatial` / `libero_spatial` with `object`/`libero_object`, `goal`/`libero_goal`, or `long`/`libero_10`.

### LIBERO — With Visual Corruptions

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval_noise.py \
    --use_proprio True \
    --num_images_in_input 2 \
    --use_film False \
    --pretrained_checkpoint outputs/spatial \
    --task_suite_name libero_spatial \
    --use_pro_version True \
    --corruption_type impulse_noise \
    --corruption_severity 3 \
    > eval_logs/spatial_noise.log 2>&1 &
```

Supported `corruption_type`: `gaussian_noise`, `impulse_noise`, `defocus_blur`, `fog`, `brightness`, and [15 more from ImageNet-C](https://github.com/hendrycks/robustness). `corruption_severity`: 3, 4, or 5.

### CALVIN — Clean

```bash
CUDA_VISIBLE_DEVICES=0 python vla-scripts/evaluate_calvin.py \
    --pretrained_checkpoint outputs/calvin \
    > eval_logs/calvin_clean.log 2>&1 &
```

### CALVIN — With Visual Corruptions

```bash
CUDA_VISIBLE_DEVICES=0 python vla-scripts/evaluate_calvin_noise.py \
    --pretrained_checkpoint outputs/calvin \
    --noise_type impulse_noise \
    --noise_severity 3 \
    > eval_logs/calvin_noise.log 2>&1 &
```

---

## 🌈 Results

### LIBERO Benchmark (Success Rate %)

C = Clean, S3/S4/S5 = Severity 3/4/5. **Bold** = best, <u>underline</u> = second best.

<table>
  <tr>
    <td><strong>Training</strong></td>
    <td><strong>Method</strong></td>
    <td><strong>Params</strong></td>
    <td><strong>Spatial<br>C / S3 / S4 / S5</strong></td>
    <td><strong>Object<br>C / S3 / S4 / S5</strong></td>
    <td><strong>Goal<br>C / S3 / S4 / S5</strong></td>
    <td><strong>Long<br>C / S3 / S4 / S5</strong></td>
  </tr>
  <tr>
    <td rowspan="2">OpenX Pretrain</td>
    <td>OpenVLA</td><td>7B</td>
    <td>80.0 / 40.9 / 24.6 / 14.7</td>
    <td>69.6 / 18.2 / 10.4 / 2.7</td>
    <td>74.0 / 38.7 / 27.0 / 16.3</td>
    <td>55.5 / 20.5 / 12.4 / 7.0</td>
  </tr>
  <tr>
    <td>OpenVLA-OFT</td><td>7B</td>
    <td>92.6 / 89.3 / <u>84.0</u> / <u>72.1</u></td>
    <td>98.4 / 82.5 / 69.2 / 52.8</td>
    <td>96.8 / <b>94.5</b> / <u>84.6</u> / <u>70.3</u></td>
    <td><b>94.4</b> / <b>77.6</b> / 61.9 / 40.3</td>
  </tr>
  <tr>
    <td>OpenX+Web Co-train</td>
    <td>OpenPi–0.5</td><td>3B</td>
    <td><b>98.4</b> / 88.3 / 79.0 / 62.4</td>
    <td><b>99.4</b> / <b>97.1</b> / <b>88.4</b> / <b>76.4</b></td>
    <td>97.2 / 87.2 / 82.5 / 64.2</td>
    <td>92.0 / 76.1 / <b>65.6</b> / <b>47.7</b></td>
  </tr>
  <tr>
    <td rowspan="2">VLM Direct FT</td>
    <td>VLA-Adapter</td><td>0.5B</td>
    <td>96.0 / <u>93.7</u> / 83.3 / 58.5</td>
    <td>96.8 / 71.0 / 44.1 / 29.3</td>
    <td><u>97.4</u> / 79.5 / 64.7 / 47.3</td>
    <td><u>94.4</u> / 63.5 / 41.0 / 26.2</td>
  </tr>
  <tr>
    <td><b>StableVLA (Ours)</b></td><td><b>0.5B</b></td>
    <td><u>96.2</u> / <b>94.4</b> / <b>92.1</b> / <b>82.0</b></td>
    <td><u>98.8</u> / <u>92.4</u> / <u>83.6</u> / <u>70.2</u></td>
    <td><b>98.0</b> / <u>93.4</u> / <b>85.0</b> / <b>71.9</b></td>
    <td>93.6 / <u>76.3</u> / <u>62.4</u> / <u>45.3</u></td>
  </tr>
</table>

### CALVIN Benchmark (Avg. Completed Tasks, max 5)

<table>
  <tr>
    <td><strong>Method</strong></td>
    <td><strong>Params</strong></td>
    <td><strong>Clean</strong></td>
    <td><strong>Sev 3</strong></td>
    <td><strong>Sev 4</strong></td>
    <td><strong>Sev 5</strong></td>
  </tr>
  <tr><td>VLA-Adapter</td><td>0.5B</td><td>4.14</td><td>2.56</td><td>1.89</td><td>1.44</td></tr>
  <tr><td><b>StableVLA (Ours)</b></td><td><b>0.5B</b></td><td><b>4.17</b></td><td><b>2.77</b></td><td><b>2.11</b></td><td><b>1.51</b></td></tr>
</table>


## :heart: Acknowledgment

We thank [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), [MiniVLA](https://github.com/Stanford-ILIAD/openvla-mini), and [RoboDual](https://github.com/OpenDriveLab/RoboDual) for their open-sourced work.

---

## :pencil2: Citation

If you find StableVLA helpful, please cite our paper:

```bibtex
@inproceedings{fu2026stablevla,
  title     = {StableVLA: Towards Robust Vision-Language-Action Models without Extra Data},
  author    = {Fu, Yiyang and Zhang, Chubin and Gong, Shukai and Deng, Yufan and
               Sun, Kaiwei and Min, Qiyang and Hou, Qibin and Tang, Yansong and
               Wang, Jianan and Zhou, Daquan},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```

---
