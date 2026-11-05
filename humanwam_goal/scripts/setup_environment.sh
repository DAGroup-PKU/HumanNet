#!/usr/bin/env bash
# Usage: bash scripts/setup_environment.sh /path/to/new/environment
# Prerequisites: uv, Linux with GLIBC >= 2.34 (FlashAttention-3), a CUDA 12.8 driver.
# Installs Python 3.13, torch 2.10.0+cu128, HumanWAM-Goal with its RoboTwin layer,
# and the cuBLASLt 12.8.3.14 library used to reproduce the published actions.
set -euo pipefail

target="${1:?Specify a new environment directory}"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -e "$target" ]]; then
  echo "Environment already exists: $target. Choose a new directory." >&2
  exit 2
fi
command -v uv >/dev/null
command -v unzip >/dev/null
uv venv --python "${PYTHON_BIN:-python3.13}" --seed "$target"
python_bin="$(cd -- "$target" && pwd)/bin/python"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 13), "Use Python 3.13"'
uv pip install --python "$python_bin" torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$python_bin" -e "$root[robotwin,dev]" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0 \
  --index-strategy unsafe-best-match

# PyTorch 2.10 depends on cuBLAS 12.8.4.1; the published results used cuBLASLt 12.8.3.14.
runtime="$(cd -- "$target" && pwd)/runtime/cublaslt-12.8.3.14"
mkdir -p "$runtime/wheel"
"$python_bin" -m pip download --no-deps --quiet 'nvidia-cublas-cu12==12.8.3.14' -d "$runtime/wheel"
unzip -q -o -j "$runtime"/wheel/nvidia_cublas_cu12-12.8.3.14-*.whl nvidia/cublas/lib/libcublasLt.so.12 -d "$runtime"
rm -rf "$runtime/wheel"
echo "4ef47ce60325811dfce57acbc2ced3000dbb6c50ce65f4ec34340b1897d6911a  $runtime/libcublasLt.so.12" | sha256sum --check --quiet -
chmod a-w "$runtime/libcublasLt.so.12"

"$python_bin" "$root/scripts/check_environment.py" --cublaslt "$runtime/libcublasLt.so.12"
uv pip check --python "$python_bin"
echo "Environment ready: $target"
echo "For bit-exact RoboTwin actions, run with LD_PRELOAD=$runtime/libcublasLt.so.12"
