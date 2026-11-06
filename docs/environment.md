# HumanWAM-Vanilla environment setup

[Back to the README](../README.md#vanilla-environment-setup)

This guide covers the training/policy environment and the separate RoboTwin
simulator environment. Both were built in new Python environments and exercised
locally. HumanWAM-Goal setup is still being prepared.

RoboDojo uses a different Isaac Sim environment; `setup_environment.sh sim` installs
RoboTwin only. See the [RoboDojo validation and commands](post_training.md#evaluation).
Its existing simulator environment has passed local checks, but installation
from scratch and asset packaging are still pending.

## System prerequisites

| Component | Training / policy | RoboTwin simulator |
| --- | --- | --- |
| OS | Linux; validated on Ubuntu 22.04 x86_64 | Same |
| Python | 3.10; validated with 3.10.12 | Same, in a separate environment |
| Package installer | `uv` | `uv` |
| GPU runtime | NVIDIA GPU and driver supporting the installed CUDA 12.8 PyTorch build | Same; NVIDIA Vulkan support is also required for rendering |
| Video libraries | FFmpeg executable **and shared libraries**; validated with FFmpeg 4.4.2 | FFmpeg |
| Build tools | A C++ compiler / development tools for native dependencies | Git, C++ compiler, and CUDA toolkit providing `nvcc` |

The local machine used driver 580.126.20 and a Blackwell GPU with compute
capability 12.0. CuRobo was compiled with CUDA toolkit 12.9. These are the tested
versions, not an assertion of the minimum requirements. The validation GPU had
about 96 GiB of memory; a minimum GPU-memory requirement has not been measured.

The scripts install Python packages, not GPU drivers, Vulkan, FFmpeg, or the CUDA
toolkit. A PyTorch CUDA wheel does not provide the `nvcc` compiler. Ensure these
system components are available before installing the simulator. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) separately.

## Install the environments

```bash
git clone --branch open-humanscale https://github.com/DAGroup-PKU/HumanNet.git
cd HumanNet/humanwam_vanilla
bash scripts/setup_environment.sh train
bash scripts/setup_environment.sh sim
```

Default environments are `OpenHumanScale/.venv/train` and `.venv/sim`. Training
and the learned policy use the first; SAPIEN, CuRobo, and task execution use the
second. Their dependency versions differ, so do not install them into one venv.

The setup script:

1. Creates a new Python 3.10 environment with `uv` and refuses to overwrite an
   existing directory, including an incomplete installation.
2. Installs PyTorch 2.7.1+cu128 and torchvision 0.22.1+cu128, then the pinned
   direct dependencies in [pyproject.toml](../humanwam_vanilla/pyproject.toml) or
   [simulator requirements](../humanwam_vanilla/evaluation/robotwin/requirements.txt).
3. For simulation, detects the visible GPUs' compute capabilities and builds
   [the pinned CuRobo revision](../humanwam_vanilla/evaluation/robotwin/requirements.txt)
   with one compilation worker by default.
4. Runs dependency/import checks and `uv pip check`. The training check also
   encodes and decodes a small video; the simulator check loads the CUDA extensions
   and exercises CPU physics. Installation does **not** run GPU rendering or a model.
5. Saves `environment-check.json` and `installed-packages.txt` in the environment.
   Direct dependencies are pinned; the package snapshot records resolved transitive
   versions and is not a portable lockfile.

Overrides are optional:

```bash
# Fresh custom environment; useful after a failed installation.
PYTHON_BIN=/usr/bin/python3.10 bash scripts/setup_environment.sh train /path/to/new-train-env

# Explicit target architecture for a build host without a visible GPU.
# Use your target GPU's capability; 12.0 is the tested Blackwell setting.
TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=1 \
  bash scripts/setup_environment.sh sim /path/to/new-sim-env
```

A failed installation leaves its directory available for inspection. Choose a
new path for a retry. Limit the visible GPUs with `CUDA_VISIBLE_DEVICES` if you
only want to build for a particular device.

## Evaluation resources

The resource download commands are listed directly in the
[README](../README.md#vanilla-environment-setup).

### Blackwell rendering

The locally tested SAPIEN 3.0.0b1 installation bundled OIDN 2.0.1, whose CUDA
device initialization failed on the validation GPU. Install the tested OIDN 2.3.3
compatibility library once:

Run from `humanwam_vanilla/`:

```bash
mkdir -p ../.runtime/oidn
curl -fL https://github.com/RenderKit/oidn/releases/download/v2.3.3/oidn-2.3.3.x86_64.linux.tar.gz \
  -o ../.runtime/oidn/oidn.tar.gz
tar -xzf ../.runtime/oidn/oidn.tar.gz -C ../.runtime/oidn
export LD_PRELOAD="$(realpath ../.runtime/oidn/oidn-2.3.3.x86_64.linux/lib/libOpenImageDenoise.so.2.3.3)${LD_PRELOAD:+:$LD_PRELOAD}"
```

OIDN 2.3.3 differs from the paper's OIDN 2.0.1 renderer and does not establish
numerical parity with the archived evaluation.

## Training inputs and optional checks

Activate the training environment before following the
[training commands](../README.md#vanilla-training):

```bash
source ../.venv/train/bin/activate
cp configs/local/robotwin.example.yaml configs/local/robotwin.yaml
# Edit the local paths before continuing.
bash scripts/train.sh --config configs/local/robotwin.yaml \
  --init-checkpoint ../checkpoints/vanilla-base/midtrained/step_010000.pt --print-config
```

Training additionally needs the RoboTwin episode data, normalization statistics,
text embeddings, instruction manifests, and merged-video caches listed in that
config. These preparation/download steps are still being organized. Environment
installation alone does not prepare a training dataset. New training uses the
full intended corpus (`val_set_proportion: 0.0`) and keeps provenance validation
on; the tested 50-task configuration selected all 2,500 episodes.

Optional checks, from `humanwam_vanilla/`:

```bash
# Training imports and real encode/decode round trip; no GPU or weights.
../.venv/train/bin/python scripts/check_environment.py --output ../runs/train-environment.json

# Simulator imports / CUDA extension loading and CPU physics.
../.venv/sim/bin/python scripts/check_sim_environment.py --output ../runs/sim-environment.json

# Offline policy check with real sample data and configured local paths.
../.venv/train/bin/python scripts/check_policy.py --paths configs/paths.local.yaml \
  --steps 2 --output ../runs/policy-check.json
```

The two-step offline check verifies checkpoint loading and finite action output;
it does not measure benchmark success rates.

## Validation and limits

Local validation on 2026-11-05 established:

| Check | Outcome |
| --- | --- |
| Fresh training and simulator Python environments | Installed independently, without system-site-packages; dependency/import checks passed |
| Video I/O | PyAV encode and TorchCodec decode passed |
| Real training | One optimizer step completed; all 2,500 intended episodes selected, no validation holdout |
| GPU simulation | OIDN 2.3.3 rendering, camera observations, reset/action interface, and expert planning passed |
| Learned-policy demo | Full 21k-checkpoint episode completed successfully in 104 actions |

The fresh installs reused the host OS, GPU driver/toolkit and explicitly supplied
model/data/asset files; they did not provision a new operating system. Multi-GPU
training, HumanWAM-Goal, and full benchmark success rates are outside this
validation. The README's success-rate tables report the technical report's results.
The new environment uses newer GPU libraries than the archived evaluation;
see the [reproduction guide](../README.md#vanilla-reproduction-guide) for protocol
and comparison limits.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| Environment already exists | Use a fresh target path; setup deliberately preserves existing installations |
| TorchCodec cannot load FFmpeg | Check both `ffmpeg -version` and availability of FFmpeg shared libraries in the dynamic linker paths |
| `nvcc` missing / CuRobo build fails | Make the CUDA toolkit and C++ compiler available; inspect `TORCH_CUDA_ARCH_LIST`; keep `MAX_JOBS=1` when memory is constrained |
| No visible GPU during architecture detection | Set `CUDA_VISIBLE_DEVICES` correctly or specify the target `TORCH_CUDA_ARCH_LIST` for an offline build |
| Vulkan or camera initialization fails | Verify the host NVIDIA Vulkan driver and container graphics access; run the standalone render check |
| OIDN device initialization fails on Blackwell | Follow the Blackwell rendering commands above and set `LD_PRELOAD` before evaluation |
| Missing robot meshes / invalid robot paths | Check the pinned asset extraction layout and rerun path relocation on your own asset copy |
| Policy startup fails | Inspect `policy.log`; check checkpoint/statistics pairing, Wan files, offline flags, and available GPU memory |
| Output directory already exists | Choose a new `--output` path; existing rollout records are preserved |
