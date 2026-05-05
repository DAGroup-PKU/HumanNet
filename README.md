<div align="center">

## HumanNet: Scaling Human-centric Video Learning to One Million Hours

[![arXiv](https://img.shields.io/badge/Arxiv-Coming%20Soon-b31b1b.svg?logo=arXiv)](https://github.com/DAGroup-PKU/HumanNet/)
[![Home Page](https://img.shields.io/badge/Project-<Website>-blue.svg)](https://dagroup-pku.github.io/HumanNet/)
[![Dataset](https://img.shields.io/badge/Dataset-HumanNet-brightgreen)](https://huggingface.co/datasets/DAGroup-PKU/HumanNet/)
[![License](https://img.shields.io/badge/License-Apache--2.0-lightgrey)](YOUR_LINK)

DAGroup &amp; SimpleSilicon Team

Peking University

</div>

## 📣 Overview
![teaser](./assets/teaser.png)
This repository is the official home of **HumanNet**, a one-million-hour human-centric video corpus designed as scalable infrastructure for fine-grained activity understanding, motion-aware video learning, and embodied pretraining. HumanNet pairs first-person and third-person footage with caption labels, motion annotations, and hand and body signals, organized by a multi-axis taxonomy and produced by a curation pipeline that treats human-centric filtering, viewpoint characterization, quality control, and privacy review as first-class design choices. Under a controlled vision-language-action post-training protocol, initializing from 1,000 hours of egocentric video drawn from HumanNet matches or modestly surpasses initializing from 100 hours of real-robot data and substantially closes the gap to a 20,000-hour real-robot baseline, indicating that egocentric human video is a scalable and cost-effective substitute when robot data is limited.

## 🔥 News
* `[Ongoing]` 🔥 We are preparing the open-source release of the HumanNet corpus, the curation pipeline, and the post-training validation code. Stay tuned!
* `[Coming Soon]` 🔥 The HumanNet technical report and project page will be released. The dataset will be available on [Hugging Face](https://huggingface.co/datasets/DAGroup-PKU/HumanNet/) once internal review is approved.

## 🎥 Demo
*Coming soon.*

## 📑 Todo List
- [ ] Release the HumanNet technical report on arXiv.
- [ ] Release the curation pipeline (acquisition, human-centric filtering, viewpoint characterization, segmentation, deduplication, quality control, privacy review, caption / motion annotation).
- [ ] Release a HumanNet preview subset on Hugging Face for early access.
- [ ] Release the full one-million-hour HumanNet corpus with metadata and annotations.
- [ ] Release the trained checkpoints initialized from HumanNet.

## ⚙️ Installation
*Coming soon.*

```bash
# 0. Clone the repo
git clone https://github.com/DAGroup-PKU/HumanNet.git
cd HumanNet

# 1. Environment (placeholder)
conda create -n humannet python=3.10
conda activate humannet
pip install -r requirements.txt
```

## 📦 Dataset
*Coming soon.*

![dataset samples](./assets/sample_result.png)

HumanNet provides:
- **Scale.** One million hours of human-centric video curated from controlled, semi-structured, community, web-scale, and domain-specific sources.
- **Viewpoint diversity.** First-person and third-person footage explicitly indexed and balanced.
- **Annotations.** Caption labels, motion descriptions, hand and body signals, and motion-centric representations.
- **Taxonomy.** Multi-axis organization over source type, viewpoint, task structure, environment, interaction style, motion category, and metadata availability.

## 🔧 Usage
*Coming soon.*

```bash
# Download a HumanNet subset (placeholder)
# if you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
# pip install -U "huggingface_hub[cli]"
huggingface-cli download DAGroup-PKU/HumanNet
```

## 📈 Validation of Egocentric Data

![validation loss](./assets/loss.png)

Under a fixed LingBot-VLA post-training regime (100 tasks, 20 episodes per task, 34 hours total), we vary only the pretraining source. The model initialized with **1,000 hours of egocentric video drawn from HumanNet** matches or modestly surpasses the model initialized with **100 hours of real-robot data**, and substantially closes the gap to a **20,000-hour real-robot baseline**, supporting the central claim of HumanNet that large-scale egocentric video is a scalable and cost-effective substitute when robot data is limited.

## 📧 Ethics Concerns
The videos referenced in this repository are sourced from public domains and intended solely to showcase the capabilities of this research. Human-centric video raises non-trivial privacy, consent, and dual-use concerns; any release will follow license review, redaction, restricted-content filtering, access controls where necessary, and clear documentation of what is included or excluded.

* The service is a research preview. Please contact us if you find any potential violations.

## ✏️ Citation

If you find our work useful in your research, please consider giving a star :star: and citation :pencil:.

### BibTeX
```bibtex
@misc{humannet2026,
  title        = {HumanNet: Scaling Human-centric Video Learning to One Million Hours},
  author       = {{DAGroup-PKU} and {SimpleSilicon Team}},
  year         = {2026},
  howpublished = {\url{https://github.com/DAGroup-PKU/HumanNet/}},
}
```
