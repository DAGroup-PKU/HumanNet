#!/usr/bin/env bash
# Usage: bash scripts/setup_environment.sh train|sim [new-environment-directory]
# Prerequisites: uv, Python 3.10, FFmpeg shared libraries; sim also needs CUDA nvcc.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Usage: bash scripts/setup_environment.sh train|sim [new-environment-directory]
Default: <repository>/.venv/train or <repository>/.venv/sim
Requires uv, Python 3.10, FFmpeg and its shared libraries.
Simulation also requires Git, a C++ compiler, and a CUDA toolkit (nvcc).
PYTHON_BIN overrides python3.10; TORCH_CUDA_ARCH_LIST overrides GPU detection.
MAX_JOBS defaults to 1 for the CuRobo build. Existing environments are not changed.
HELP
  exit 0
fi
mode="${1:-}"
case "$mode" in train|sim) ;; *) echo 'Expected train or sim; use --help for usage.' >&2; exit 2 ;; esac
[[ $# -le 2 ]] || { echo 'Too many arguments; use --help.' >&2; exit 2; }
target="${2:-$root/../.venv/$mode}"
if [[ -e "$target" || -L "$target" ]]; then
  echo "Environment already exists: $target. Choose a new directory." >&2
  exit 2
fi
for dependency in uv ffmpeg; do
  command -v "$dependency" >/dev/null || { echo "Missing prerequisite: $dependency" >&2; exit 2; }
done
if [[ "$mode" == sim ]]; then
  for dependency in git nvcc c++; do
    command -v "$dependency" >/dev/null || { echo "Missing prerequisite: $dependency" >&2; exit 2; }
  done
fi
uv venv --python "${PYTHON_BIN:-python3.10}" --seed "$target"
python_bin="$(cd -- "$target" && pwd)/bin/python"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10"'
uv pip install --python "$python_bin" \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
if [[ "$mode" == train ]]; then
  uv pip install --python "$python_bin" -e "$root[dev]"
  "$python_bin" "$root/scripts/check_environment.py" --output "$target/environment-check.json"
else
  if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    TORCH_CUDA_ARCH_LIST="$("$python_bin" - <<'PYTHON'
import torch
if not torch.cuda.is_available():
    raise SystemExit("No visible CUDA GPU. Set TORCH_CUDA_ARCH_LIST explicitly for an offline build.")
architectures = sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
print(";".join(f"{major}.{minor}" for major, minor in architectures))
PYTHON
)"
  fi
  export TORCH_CUDA_ARCH_LIST
  echo "Building CuRobo for CUDA architectures: $TORCH_CUDA_ARCH_LIST"
  MAX_JOBS="${MAX_JOBS:-1}" uv pip install --python "$python_bin" \
    --no-build-isolation-package nvidia-curobo \
    -r "$root/evaluation/robotwin/requirements.txt"
  "$python_bin" "$root/scripts/check_sim_environment.py" --output "$target/environment-check.json"
fi
uv pip check --python "$python_bin"
uv pip freeze --python "$python_bin" > "$target/installed-packages.txt"
echo "Environment ready: $target"
echo "Checks: $target/environment-check.json; versions: $target/installed-packages.txt"
