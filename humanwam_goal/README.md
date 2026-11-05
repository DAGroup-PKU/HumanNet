# HumanWAM-Goal

The goal-conditioned HumanWAM model for [Open-HumanScale](../README.md),
supporting zero-shot robot execution through hybrid human–robot training and visual goals.

HumanWAM-Goal predicts a 48-step action chunk from a text instruction, a goal
image, camera observations with a short visual history, and robot state. On the
RoboTwin 2.0 benchmark it reaches **88.65% average success** (92.70% Clean /
84.60% Random) under the official 10,000-episode protocol.

🚧 This release covers inference and RoboTwin 2.0 evaluation. The post-training
code will be released separately.

## Getting Started

- [Weight Download](#weight-download) — RoboTwin post-trained checkpoint.
- [Environment Setup](#environment-setup) — one Python 3.13 environment.
- [Quick Start](#quick-start) — run the model on a sample observation.
- [Inference](#inference) — Python API, HTTP server and [input format](docs/inputs.md).
- [RoboTwin 2.0 Evaluation](#robotwin-20-evaluation) — released goals and the evaluation session.
- [Reproducing Published Results](#reproducing-published-results)

See the project README for the [release plan](../README.md#release-plan),
[license](../README.md#license), and [citation](../README.md#citation).

## Weight Download

| Checkpoint | Clean2Clean | Clean2Random | Average | Status |
| --- | ---: | ---: | ---: | --- |
| HumanWAM-Goal · RoboTwin post-trained | **92.70%** | **84.60%** | **88.65%** | [HumanWAM-Goal-RoboTwin](https://huggingface.co/DAGroup-PKU/HumanWAM-Goal-RoboTwin) |

[DAGroup-PKU/HumanWAM-Goal-RoboTwin](https://huggingface.co/DAGroup-PKU/HumanWAM-Goal-RoboTwin),
part of the [Open-HumanScale Collection](https://huggingface.co/collections/DAGroup-PKU/open-humanscale),
holds the checkpoint at its root (`transformer/`, `vae/`, `text_tokenizer/`,
`processor.json`, `sampling.json`, `robot.json` and `export.json`, which records
provenance and the SHA-256 of every file) and the RoboTwin goal cache as
`goal_cache.tar`. See [release/huggingface.json](release/huggingface.json) for the
digests.

```bash
hf download DAGroup-PKU/HumanWAM-Goal-RoboTwin --exclude goal_cache.tar \
  --local-dir checkpoints/HumanWAM-Goal-RoboTwin
export GOALWAM_CHECKPOINT=$PWD/checkpoints/HumanWAM-Goal-RoboTwin
```

## Environment Setup

Requirements: Python 3.13, PyTorch 2.10 with CUDA 12.8, a Hopper GPU (SM90), and
Linux with GLIBC ≥ 2.34.

```bash
cd humanwam_goal
bash scripts/setup_environment.sh /path/to/new/environment
source /path/to/new/environment/bin/activate
```

The script installs PyTorch 2.10.0+cu128, this package with its RoboTwin layer
(`.[robotwin,dev]`, exact pins), and the cuBLASLt library needed for
[bit-exact reproduction](#reproducing-published-results), then checks the
environment (`scripts/check_environment.py`).

GLIBC ≥ 2.34 is needed only by the prebuilt FlashAttention-3 extension
(`flash-attn-3-nv`). Its wheel installs on older systems but then fails to import
(`GLIBC_2.34 not found`); on such hosts, run inside a container with a newer
userland, for example Ubuntu 24.04.

## Quick Start

Generate sample inputs, then run the model on them:

```bash
python examples/make_sample.py
CUDA_VISIBLE_DEVICES=0 goalwam infer \
  --checkpoint "$GOALWAM_CHECKPOINT" \
  --backend native \
  --sample examples/sample/observation.json \
  --output outputs/actions.npz
```

## Inference

```python
from goalwam import GoalWAM

with GoalWAM(checkpoint, backend="native") as model:
    result = model.predict(observation)  # see docs/inputs.md
```

With physical state input, `predict` returns `actions` (48 absolute 49-D
commands), `relative_actions`, `action_valid_mask` and `normalized_actions`.
`goalwam serve --checkpoint "$GOALWAM_CHECKPOINT"` exposes the same call over HTTP
(`/healthz`, `/metadata`, `/infer`; images as base64). See [Input format](docs/inputs.md)
for every observation field, the 49-D robot layout and the gripper conventions.
[examples/infer.py](examples/infer.py) shows the Python API end to end.

### PhyAI acceleration

[PhyAI](https://github.com/mingti-org/phyai) accelerates HumanWAM-Goal on one GPU
or with tensor parallelism across 2 or 4 NVLink-connected GPUs on one host.
It uses the same checkpoint, observation format and Python API. TP=2/4 also
requires the CUDA toolkit (`nvcc`) and roughly the single-GPU weight memory per GPU.

To include PhyAI 0.2.0rc3 when creating the environment:

```bash
bash scripts/setup_environment.sh /path/to/new/environment --phyai
source /path/to/new/environment/bin/activate
export LD_PRELOAD="$VIRTUAL_ENV/runtime/cublaslt-12.8.3.14/libcublasLt.so.12"
```

For an existing environment created above, install the optional backend:

```bash
uv pip install -e '.[phyai]' \
  --extra-index-url https://mingti-org.github.io/phyai/simple/ \
  --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0 \
  --index-strategy unsafe-best-match
export LD_PRELOAD="$VIRTUAL_ENV/runtime/cublaslt-12.8.3.14/libcublasLt.so.12"
```

Run the sample from Quick Start with two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 goalwam infer \
  --checkpoint "$GOALWAM_CHECKPOINT" --backend phyai --tp 2 \
  --sample examples/sample/observation.json --output outputs/actions.npz
```

Use `CUDA_VISIBLE_DEVICES=0` with `--tp 1` for one GPU, or
`CUDA_VISIBLE_DEVICES=0,1,2,3` with `--tp 4` for four. PhyAI starts the workers; do not
wrap the command in `torchrun`. The same `--backend phyai --tp 2` options work
with `goalwam serve`. In Python, pass `backend="phyai", tp=2` to `GoalWAM`.

Both backends share Torch 2.10.0. Keep the cuBLASLt preload above when comparing
their actions or reproducing the published results.

## RoboTwin 2.0 Evaluation

`goalwam.robotwin` connects the model to RoboTwin 2.0 Aloha-AgileX evaluation
through [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab):

| Module | Role |
| --- | --- |
| [bridge.py](src/goalwam/robotwin/bridge.py) | XPolicyLab observation ↔ 49-D head-camera state; absolute commands → 14 joint targets (6 joints + gripper openness per arm) |
| [goal_cache.py](src/goalwam/robotwin/goal_cache.py) | The released goal cache: one reviewed head-camera goal per official case |
| [session.py](src/goalwam/robotwin/session.py) | One evaluation session: strict step-zero check, 0.5 s / 2 s visual history, a query every 32 steps |
| [runtime.py](src/goalwam/robotwin/runtime.py) | Verifies the cuBLASLt library loaded in the process |

The goal cache (`robotwin_official_goal_cache_v1`) holds one 224×288 goal for each
of the 10,000 official cases (50 tasks × 100 expert-accepted seeds × Clean and
Random). Each goal was generated from the case's **initial head-camera image and
instruction only**, and then visually reviewed. A case is served only when its
task configuration, task, seed and exact instruction are in the cache and the
step-zero RGB, state and head-camera pose equal the cached initial observation;
a cache miss stops the run instead of skipping or replacing the seed.

```python
from goalwam import GoalWAM
from goalwam.robotwin import OfficialGoalCache, RoboTwinSession

model = GoalWAM(checkpoint, backend="native")
goals = OfficialGoalCache(cache_root, cache_manifest_sha256, "configs/robotwin.yaml")
session = RoboTwinSession(
    model, output_dir, goals
)  # prepare_case / reset / update_obs / get_action
```

[configs/robotwin.yaml](configs/robotwin.yaml) is the evaluation profile: joint
control, 32 executed steps per query and seed `9000 + query`. Download the cache
and verify it without loading a model:

```bash
hf download DAGroup-PKU/HumanWAM-Goal-RoboTwin goal_cache.tar --local-dir downloads
tar -xf downloads/goal_cache.tar -C downloads   # -> downloads/goal_cache/
export GOAL_CACHE=$PWD/downloads/goal_cache
python -m goalwam.robotwin.goal_cache --root "$GOAL_CACHE" \
  --manifest-sha256 a80be2cec8fa6f42879ec9d27a113b6b3475e69ae0e9af25cd75f70dfe5cf6ad \
  --profile configs/robotwin.yaml --verify-artifacts
```

The XPolicyLab adapter (`policy/HumanWAM_Goal`) runs RoboTwin's own evaluator
unchanged and serves this session. The training data converter and post-training
scripts are not part of this release.

## Reproducing Published Results

The published RoboTwin 2.0 results were served with cuBLASLt 12.8.3.14. PyTorch
2.10 ships cuBLAS 12.8.4.1, which selects different BF16 GEMM kernels, so its
actions differ slightly. `scripts/setup_environment.sh` places the library from
the official `nvidia-cublas-cu12==12.8.3.14` wheel in the environment's
`runtime/cublaslt-12.8.3.14/`; preload it to reproduce the published actions bit
for bit, and `goalwam.robotwin.cublaslt_runtime()` confirms it is the one loaded:

```bash
LD_PRELOAD=/path/to/environment/runtime/cublaslt-12.8.3.14/libcublasLt.so.12 goalwam infer ...
```

With this runtime, the released inference code reproduces the actions of the
serving engine behind the published numbers exactly: 300 recorded requests across
all 50 tasks, and 70 queries of full evaluation sessions.

## Code

- [src/goalwam](src/goalwam/) — model, preprocessing and backends.
- [src/goalwam/robotwin](src/goalwam/robotwin/) — RoboTwin 2.0 evaluation layer.
- [configs](configs/) — RoboTwin evaluation profile.
- [scripts](scripts/) — environment setup and release checks.
- [release](release/) — Hugging Face layout and the provenance of the published result.
- [docs/inputs.md](docs/inputs.md) — observation format.

## License

The native transformer, rotary embedding and image encoder components adapt
NVIDIA Cosmos code under the OpenMDW-1.1 license ([LICENSE](LICENSE)); see
[NOTICE](NOTICE) and [licenses/](licenses/) for the Apache-2.0 and MIT components.
Checkpoint terms are specified in the model card.

## Acknowledgements

We thank Mingti Technology and [PhyAI](https://github.com/mingti-org/phyai) for
their support with model serving and deployment infrastructure.
