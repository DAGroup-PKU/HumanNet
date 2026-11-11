<div align="center">

<p>
<img src="./assets/branding/humannet-header-shadow.svg" width="1000" alt="HumanNet — DAGroup, Peking University" />
</p>

### Human-Centric Video Learning and Embodied AI Resources

DAGroup & SimpleSilicon Innovation Team

<h3>
<a href="./docs/humandata.md">📚 Docs</a> |
<a href="https://dagroup-pku.github.io/HumanNet/">Project</a> |
<a href="https://github.com/DAGroup-PKU/HumanNet/tree/Open-HumanWAM">Open-HumanWAM</a> |
<a href="https://dagroup-pku.github.io/StableVLA/">StableVLA</a> |
<a href="#citation">Citation</a>
</h3>

[![HumanNet Ego](./assets/badges/humannet-ego.svg)](https://huggingface.co/datasets/DAGroup-PKU/HumanNet)
[![Open-HumanWAM-50K](./assets/badges/open-humanwam-50k.svg)](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K)
[![HumanNet Paper](./assets/badges/humannet-paper.svg)](https://arxiv.org/abs/2605.06747)
[![Open-HumanWAM Paper](./assets/badges/open-humanwam-paper.svg)](https://arxiv.org/abs/2611.06817)
[![RoVid-X Dataset](./assets/badges/rovid-x-data.svg)](https://huggingface.co/datasets/DAGroup-PKU/RoVid-X)
[![RoVid-X Paper](./assets/badges/rovid-x-paper.svg)](https://arxiv.org/abs/2601.15282)

<p>
<img src="./assets/branding/dataset-highlights.svg" width="1000" alt="200K hours HumanNet Ego · 50K hours Open-HumanWAM-50K · 3M video clips RoVid-X" />
</p>

</div>

## 🔥 News

* `[2026.11.11]` 📦 **HumanNet Ego is released!** We release **HumanNet Ego**, an approximately **200,000-hour egocentric subset of the one-million-hour HumanNet corpus**, for the research community. Download the data from **[Hugging Face](https://huggingface.co/datasets/DAGroup-PKU/HumanNet)** or **[ModelScope](https://www.modelscope.cn/datasets/DAGroup-PKU/HumanNet)**. HumanNet contributes approximately **40,700 hours** to the **[Open-HumanWAM-50K](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K)** corpus, which combines five sources for robot-learning pretraining.
* `[2026.11.05]` 🔥 We release **Open-HumanWAM: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models** ([arXiv](https://arxiv.org/abs/2611.06817)). Our **[Open-HumanWAM-50K](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K)** corpus contains **50,000 hours of egocentric human experience**, supporting systematic experiments that demonstrate **log-linear scaling of downstream robot success with human pretraining data**. 🏆 **HumanWAM-Goal** achieves **Rank #1 performance on the [RoboTwin 2.0 benchmark](https://robotwin-platform.github.io/leaderboard)**, with **88.65% average success** (**92.70% Clean / 84.60% Random**). Explore the report, corpus, models, and code in our **[Open-HumanWAM branch](https://github.com/DAGroup-PKU/HumanNet/tree/Open-HumanWAM)**.
* `[2026.05.22]` 🔥 We reprocessed a compliant embodied dataset on our university cluster. In this version, we removed non-compliant third-person source data and, after the full data pipeline, obtained a high-quality first-person robot video dataset with **3M video clips**, **720P resolution**, and **30 FPS**. To the best of our knowledge, **RoVid-X** is the largest robot video generation dataset to date for training physics-aware models. See [Dataset](https://huggingface.co/datasets/DAGroup-PKU/RoVid-X)  | [Resources](./src/dataset/rovid-x/).
* `[2026.05.18]` 🔥 We release **StableVLA**. Congratulations on its acceptance to **ICML 2026**! It is a vision-language-action model for robust robot policy learning. See [Docs](./docs/stablevla.md) | [Code](./src/model/StableVLA/) | [Project](https://dagroup-pku.github.io/StableVLA/) | [Paper](https://arxiv.org/abs/2605.18287) | [Checkpoint](https://huggingface.co/DAGroup-PKU/StableVLA/tree/main).
* `[2026.05.11]`🔥 The **HumanNet** technical report and project page have been released: [Paper](https://arxiv.org/abs/2605.06747) | [Project](https://dagroup-pku.github.io/HumanNet/).

## 📑 Todo List
- [x] Release the **HumanNet** technical report on arXiv. ✅
- [x] Release **StableVLA** model code and documentation. ✅
- [x] Release **RoVid-X** on Hugging Face. ✅
- [x] Release **HumanNet Ego**, an approximately **200,000-hour egocentric subset of the one-million-hour HumanNet corpus**, on Hugging Face and ModelScope. ✅
- [x] Release the **Open-HumanWAM-50K** curated corpus. ✅
- [x] Release **Open-HumanWAM** model checkpoints, training recipes, and evaluation code. ✅
- [ ] Release the full one-million-hour HumanNet corpus with metadata and annotations.


## 📣 Overview
![teaser](./assets/teaser.png)

This repository is maintained as a growing research hub for human-centric video data, embodied learning models, and validation code. It brings together **HumanNet**, **Open-HumanWAM**, **RoVid-X**, and **StableVLA**, with links to data, models, training recipes, evaluation protocols, and release notes.

The initial core release is **HumanNet**, a scalable infrastructure for fine-grained activity understanding, motion-aware video learning, and embodied pretraining. HumanNet pairs first-person and third-person footage with caption labels, motion annotations, and hand and body signals, organized by a multi-axis taxonomy and produced by a curation pipeline that treats human-centric filtering, viewpoint characterization, quality control, and privacy review as first-class design choices.


### HumanNet Ego and Open-HumanWAM-50K

The HumanNet report describes a **one-million-hour** corpus spanning first-person and third-person video. **HumanNet Ego is an approximately 200,000-hour egocentric subset released from this broader HumanNet corpus**, supporting research on data curation, action annotation, video learning, and embodied AI.

**Open-HumanWAM-50K** is a **curated, multi-source training corpus**, combining approximately **40,700 hours from HumanNet** with EgoSuite-Open100K, Gen-HumanEgo, EgoDex, and EgoVerse. It provides egocentric video, hand motion, and language in LeRobot v3 format, with cross-source alignment to a unified bimanual action representation.

| Resource | Scope | Access |
| --- | --- | --- |
| **HumanNet Ego** | Approximately 200K-hour egocentric subset of the 1M-hour HumanNet corpus | [Hugging Face](https://huggingface.co/datasets/DAGroup-PKU/HumanNet) · [ModelScope](https://www.modelscope.cn/datasets/DAGroup-PKU/HumanNet) |
| **Open-HumanWAM-50K** | 50K hours of curated, aligned data from five sources | [Hugging Face](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K) · [ModelScope](https://modelscope.cn/datasets/DAGroup-PKU/EgoCorpus-50kH-LeRobot-v3) |
| **Open-HumanWAM** | Model weights, pretraining, post-training, and evaluation | [Project](https://dagroup-pku.github.io/open-humanwam/) · [Code](https://github.com/DAGroup-PKU/HumanNet/tree/Open-HumanWAM) · [Models](https://huggingface.co/collections/DAGroup-PKU/open-humanwam) |

<a id="demo"></a>

## 🎥 Demo
https://github.com/user-attachments/assets/52eaa410-0ec4-4f89-81e8-d2ecf9bb351c


## 📚 Dataset Family

| Dataset | Status | Documentation | Resources |
|---|---|---|---|
| **HumanNet Ego** | Approximately 200K-hour Ego subset released | [Docs](./docs/humandata.md) | [Hugging Face](https://huggingface.co/datasets/DAGroup-PKU/HumanNet) / [ModelScope](https://www.modelscope.cn/datasets/DAGroup-PKU/HumanNet) |
| **Open-HumanWAM-50K** | 50K hours; upload complete | [Dataset card](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K) | [Hugging Face](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K) / [ModelScope](https://modelscope.cn/datasets/DAGroup-PKU/EgoCorpus-50kH-LeRobot-v3) |
| **RoVid-X** | Released on Hugging Face | [Dataset card](https://huggingface.co/datasets/DAGroup-PKU/RoVid-X) | [Hugging Face](https://huggingface.co/datasets/DAGroup-PKU/RoVid-X) / [src/dataset/rovid-x](./src/dataset/rovid-x/) |

## 🤖 Model Family

| Model | Status | Documentation | Code |
|---|---|---|---|
| **HumanWAM-Vanilla / HumanWAM-Goal** | Code and model releases | [Project](https://dagroup-pku.github.io/open-humanwam/) / [Model collection](https://huggingface.co/collections/DAGroup-PKU/open-humanwam) | [Open-HumanWAM branch](https://github.com/DAGroup-PKU/HumanNet/tree/Open-HumanWAM) |
| **StableVLA** | Code and docs available | [Docs](./docs/stablevla.md) | [src/model/StableVLA](./src/model/StableVLA/) |

## 🗂️ Repository Map

```text
HumanNet/
├── README.md                 # Repository entry point
├── docs/                     # Component-level documentation and release notes
│   ├── humandata.md          # HumanNet dataset documentation
│   └── stablevla.md          # StableVLA documentation
├── assets/                   # Figures used by the repository README
└── src/
    ├── dataset/
    │   ├── humandata/        # HumanNet dataset resources
    │   └── rovid-x/          # ROViD-X dataset resources
    └── model/
        └── StableVLA/        # StableVLA source code, training scripts, and model README
```




## 🔧 Usage

### Download HumanNet Ego

Install the Hugging Face CLI, then download the dataset to a local directory.
See the [dataset card](https://huggingface.co/datasets/DAGroup-PKU/HumanNet)
for access terms and release details.

```bash
pip install -U huggingface_hub
hf download DAGroup-PKU/HumanNet --repo-type dataset --local-dir ./data/HumanNet
```

HumanNet Ego is also available on [ModelScope](https://www.modelscope.cn/datasets/DAGroup-PKU/HumanNet).

### Use Open-HumanWAM

For corpus organization, subset indices, and access conditions, see the
[Open-HumanWAM-50K dataset card](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K).

For model weights, installation, training, and evaluation, follow the
[Open-HumanWAM branch](https://github.com/DAGroup-PKU/HumanNet/tree/Open-HumanWAM).

### Download RoVid-X

```bash
hf download DAGroup-PKU/RoVid-X --repo-type dataset --local-dir ./data/RoVid-X
```

## 🙏 Acknowledgement
We gratefully acknowledge **SimpleSilicon Innovation** for providing funding and resource support, and **Astribot** for providing real-robot platforms and deployment experiment support.

## 📧 Ethics Concerns
The videos referenced in this repository are sourced from public domains and intended solely to showcase the capabilities of this research. Human-centric video raises non-trivial privacy, consent, and dual-use concerns; any release will follow license review, redaction, restricted-content filtering, access controls where necessary, and clear documentation of what is included or excluded.

* The service is a research preview. Please contact us if you find any potential violations.

<a id="citation"></a>

## ✏️ Citation

If you find our work useful in your research, please consider giving a star :star: and citation :pencil:.

Open-HumanWAM was originally released as **Open-HumanScale** in
[arXiv v1](https://arxiv.org/abs/2611.06817v1). Both versions are listed below.

### BibTeX
```bibtex
@misc{ma2026openhumanwam,
  title={{Open-HumanWAM}: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models},
  author={Ma, Juncheng and Li, Jiaxin and Tu, Jiankai and others},
  year={2026},
  eprint={2611.06817},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2611.06817}
}

@misc{ma2026openhumanscaleopenrecipescaling,
  title={Open-HumanScale: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models},
  author={Juncheng Ma and Jiaxin Li and Jiankai Tu and Bo Liang and Guodong Lv and Yufan Deng and Junzhe Tang and Liangzhu Yang and Chen Bian and Zicong Cao and Baohao Zhu and Jianxin Bi and Tengfei Yan and Yinyuan Zhang and Kewei Zhang and Xinglong Chen and Kaiqi Chen and Duomin Wang and Xiaotian Tang and Liang Tan and Li Li and Yuan Xu and Haosen Yang and Yuqi Wang and Jingfei Xia and Songyan Huang and Zhiyang Dou and Zhen Dong and Enze Xie and Tat-Seng Chua and Daquan Zhou},
  year={2026},
  eprint={2611.06817v1},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2611.06817v1}
}

@article{deng2026humannet,
  title={HumanNet: Scaling Human-centric Video Learning to One Million Hours},
  author={Deng, Yufan and Zhou, Daquan},
  journal={arXiv preprint arXiv:2605.06747},
  year={2026}
}

@misc{fu2026stablevlarobustvisionlanguageactionmodels,
      title={StableVLA: Towards Robust Vision-Language-Action Models without Extra Data}, 
      author={Yiyang Fu and Chubin Zhang and Shukai Gong and Yufan Deng and Kaiwei Sun and Qiyang Min and Qibin Hou and Yansong Tang and Jianan Wang and Daquan Zhou},
      year={2026},
      eprint={2605.18287},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.18287}, 
}

```

