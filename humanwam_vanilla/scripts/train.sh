#!/usr/bin/env bash
# Local launch: NPROC_PER_NODE=4 bash scripts/train.sh --paths configs/paths.local.yaml
# Multi-node: run on each node with NNODES, NODE_RANK, MASTER_ADDR, and MASTER_PORT.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
processes="${NPROC_PER_NODE:-1}"
nodes="${NNODES:-1}"
node_rank="${NODE_RANK:-0}"

launch=(--nnodes "$nodes" --nproc-per-node "$processes")
if (( nodes == 1 )); then
  launch+=(--standalone)
else
  launch+=(--node-rank "$node_rank" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT")
fi

exec "$python_bin" -m torch.distributed.run "${launch[@]}" "$script_dir/train.py" "$@"
