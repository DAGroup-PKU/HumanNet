#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
hub="$root/.venv/train/bin/huggingface-cli"

case "${1:-}" in
  wan)
    models="${DIFFSYNTH_MODEL_BASE_PATH:-./checkpoints}"
    "$hub" download Wan-AI/Wan2.2-TI2V-5B \
      Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth \
      --local-dir "$models/Wan-AI/Wan2.2-TI2V-5B"
    "$hub" download Wan-AI/Wan2.1-T2V-1.3B \
      --include 'google/umt5-xxl/*' \
      --local-dir "$models/Wan-AI/Wan2.1-T2V-1.3B"
    ;;
  robotwin)
    robotwin="$root/RoboTwin"
    git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$robotwin"
    git -C "$robotwin" checkout 13c3c47ff4312dd62484bcd51be034af55c062d1
    git -C "$robotwin" apply "$root/humanwam_vanilla/evaluation/robotwin/render-device.patch"
    git -C "$robotwin" apply "$root/humanwam_vanilla/evaluation/robotwin/paper-render.patch"
    "$hub" download TianxingChen/RoboTwin2.0 \
      background_texture.zip embodiments.zip objects.zip \
      --repo-type dataset --revision 9dc9299c163db059931898a9f0852098a61155a1 \
      --local-dir "$robotwin/assets"
    for archive in background_texture embodiments objects; do
      unzip -q "$robotwin/assets/$archive.zip" -d "$robotwin/assets"
    done
    cd "$robotwin"
    "$root/.venv/sim/bin/python" script/update_embodiment_config_path.py
    ;;
  *)
    echo 'Usage: bash scripts/prepare_resources.sh {wan|robotwin}'
    exit 2
    ;;
esac
