#!/usr/bin/env bash
# Evaluate the released W0 policy on all 50 tasks, with 50 episodes per task.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo 'Usage: bash scripts/eval_robotwin.sh CHECKPOINT_DIRECTORY {clean|random} OUTPUT_DIRECTORY'
  exit 2
fi

bundle="$(realpath "$1")"
phase="$2"
output="$(realpath -m "$3")"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
policy_python="${POLICY_PYTHON:-$root/.venv/train/bin/python}"
sim_python="${SIM_PYTHON:-$root/.venv/sim/bin/python}"
robotwin="${ROBOTWIN_ROOT:-$root/RoboTwin}"
cd "$root/humanwam_vanilla"

for shard in {0..7}; do
  episodes=6
  if (( shard < 2 )); then episodes=7; fi
  "$sim_python" scripts/eval_robotwin.py \
    --robotwin-root "$robotwin" --policy-python "$policy_python" \
    --config "$bundle/config.yaml" --checkpoint "$bundle/model.pt" \
    --stats "$bundle/dataset_stats.json" \
    --task-file evaluation/robotwin/tasks-50.txt \
    --phase "$phase" --seed "$shard" --episodes "$episodes" \
    --output "$output/shard-$shard"
done
