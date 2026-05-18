<div align="center">

## HumanNet: Human-Centric Video Learning and Embodied AI Resources
[![arXiv](https://img.shields.io/badge/Arxiv-2605.06747-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2605.06747)
[![Home Page](https://img.shields.io/badge/Project-<Website>-blue.svg)](https://dagroup-pku.github.io/HumanNet/)
[![Dataset](https://img.shields.io/badge/Dataset-HumanNet-brightgreen)](https://huggingface.co/datasets/DAGroup-PKU/HumanNet/)
[![License](https://img.shields.io/badge/License-Apache--2.0-lightgrey)](YOUR_LINK)

**DAGroup & SimpleSilicon Innovation Team**

Peking University

</div>

## 🔥 News
* `[2026.05.18]` 🔥 We release **StableVLA**. Congratulations on its acceptance to **ICML 2026**! It is a vision-language-action model for robust robot policy learning. See [docs](./docs/stablevla.md) and [code](./src/model/StableVLA/).
* `[Next Month]` 🔥 We are preparing the open-source release of the HumanNet corpus, the curation pipeline, and the post-training validation code. Stay tuned!
* `[2026.05.11]`🔥 The **HumanNet** technical report and project page have been released: [Paper](https://arxiv.org/abs/2605.06747) | [Project](https://dagroup-pku.github.io/HumanNet/).

## 📑 Todo List
- [x] Release the **HumanNet** technical report on arXiv. ✅
- [x] Release **StableVLA** model code and documentation. ✅
- [ ] Release a HumanNet preview subset on Hugging Face for early access.
- [ ] Release the full one-million-hour HumanNet corpus with metadata and annotations.
- [ ] Release the trained checkpoints initialized from HumanNet.


## 📣 Overview
![teaser](./assets/teaser.png)
This repository is maintained as a growing research hub for human-centric video data, embodied learning models, and validation code. It currently centers on **HumanNet**, a one-million-hour human-centric video corpus, and will also host related models, training recipes, evaluation protocols, and release notes.

The initial core release is **HumanNet**, a scalable infrastructure for fine-grained activity understanding, motion-aware video learning, and embodied pretraining. HumanNet pairs first-person and third-person footage with caption labels, motion annotations, and hand and body signals, organized by a multi-axis taxonomy and produced by a curation pipeline that treats human-centric filtering, viewpoint characterization, quality control, and privacy review as first-class design choices.


## 🎥 Demo
https://github.com/user-attachments/assets/52eaa410-0ec4-4f89-81e8-d2ecf9bb351c


## 📚 Dataset Family

| Dataset | Status | Documentation | Resources |
|---|---|---|---|
| **HumanNet** | Documentation available | [Docs](./docs/humandata.md) | [src/dataset/humandata](./src/dataset/humandata/) |
| **Rovid-X** | Placeholder available | Coming soon | [src/dataset/rovid-x](./src/dataset/rovid-x/) |

## 🤖 Model Family

| Model | Status | Documentation | Code |
|---|---|---|---|
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
*Coming soon.*

```bash
# Download a HumanNet subset (placeholder)
# if you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
# pip install -U "huggingface_hub[cli]"
huggingface-cli download DAGroup-PKU/HumanNet
```


## 📧 Ethics Concerns
The videos referenced in this repository are sourced from public domains and intended solely to showcase the capabilities of this research. Human-centric video raises non-trivial privacy, consent, and dual-use concerns; any release will follow license review, redaction, restricted-content filtering, access controls where necessary, and clear documentation of what is included or excluded.

* The service is a research preview. Please contact us if you find any potential violations.

## ✏️ Citation

If you find our work useful in your research, please consider giving a star :star: and citation :pencil:.

### BibTeX
```bibtex
@misc{deng2026humannetscalinghumancentricvideo,
      title={HumanNet: Scaling Human-centric Video Learning to One Million Hours}, 
      author={Yufan Deng and Daquan Zhou},
      year={2026},
      eprint={2605.06747},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.06747}, 
}
```
