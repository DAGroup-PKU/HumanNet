#!/usr/bin/env bash
# Usage: bash scripts/train.sh --config CONFIG [--init-checkpoint WEIGHTS | --resume STATE_DIR]
# Optional local bindings: --paths PATHS.yaml; overrides: batch_size=1, etc.
# Multi-node: run on each node with NNODES, NODE_RANK, MASTER_ADDR, and MASTER_PORT.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x "$script_dir/../../.venv/train/bin/python" ]]; then
  python_bin="$script_dir/../../.venv/train/bin/python"
else
  python_bin=python
fi
processes="${NPROC_PER_NODE:-1}"
nodes="${NNODES:-1}"
node_rank="${NODE_RANK:-0}"
if [[ ! "$processes" =~ ^[1-9][0-9]*$ || ! "$nodes" =~ ^[1-9][0-9]*$ ]]; then
  echo 'NPROC_PER_NODE and NNODES must be positive integers.' >&2
  exit 2
fi

if (( nodes == 1 && processes == 1 )); then
  exec "$python_bin" "$script_dir/train.py" "$@"
fi

launch=(--nnodes "$nodes" --nproc-per-node "$processes")
if (( nodes == 1 )); then
  launch+=(--standalone)
else
  launch+=(--node-rank "$node_rank" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT")
fi

exec "$python_bin" -m torch.distributed.run "${launch[@]}" "$script_dir/train.py" "$@"
