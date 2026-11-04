#!/usr/bin/env bash
# Usage: bash scripts/setup_environment.sh train|sim /path/to/new/environment
# Prerequisites: uv, Python 3.10, FFmpeg shared libraries; sim also needs CUDA nvcc.
set -euo pipefail

mode="${1:?Specify train or sim}"
target="${2:?Specify a new environment directory}"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
case "$mode" in train|sim) ;; *) echo 'Expected train or sim' >&2; exit 2 ;; esac
if [[ -e "$target" ]]; then
  echo "Environment already exists: $target. Choose a new directory." >&2
  exit 2
fi
command -v uv >/dev/null
command -v ffmpeg >/dev/null
if [[ "$mode" == sim ]]; then
  command -v git >/dev/null
  command -v nvcc >/dev/null
  : "${TORCH_CUDA_ARCH_LIST:?Set the target GPU architecture, e.g. 12.0 for the tested GPU}"
fi
uv venv --python "${PYTHON_BIN:-python3.10}" --seed "$target"
python_bin="$(cd -- "$target" && pwd)/bin/python"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10"'
uv pip install --python "$python_bin" \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
if [[ "$mode" == train ]]; then
  uv pip install --python "$python_bin" -e "$root[dev]"
  "$python_bin" "$root/scripts/check_environment.py"
else
  uv pip install --python "$python_bin" -r "$root/evaluation/robotwin/requirements-sim.txt"
  MAX_JOBS="${MAX_JOBS:-1}" uv pip install --python "$python_bin" \
    --no-build-isolation -r "$root/evaluation/robotwin/requirements-cuda.txt"
  "$python_bin" "$root/scripts/check_sim_environment.py"
fi
uv pip check --python "$python_bin"
echo "Environment ready: $target"
