#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG [GPUS_PER_NODE] [extra training arguments...]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config="$1"
gpus_per_node="${2:-${GPUS_PER_NODE:-${NUM_GPUS:-1}}}"
num_nodes="${NNODES:-${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}}"
node_rank="${NODE_RANK:-${SLURM_NODEID:-0}}"
torchrun_bin="${TORCHRUN_BIN:-torchrun}"

if [[ "$config" != /* ]]; then
  config="$repo_root/$config"
fi
if [[ ! -f "$config" ]]; then
  echo "Config not found: $config" >&2
  exit 2
fi
if ! [[ "$gpus_per_node" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPUS_PER_NODE must be a positive integer, got: $gpus_per_node" >&2
  exit 2
fi
if ! [[ "$num_nodes" =~ ^[1-9][0-9]*$ ]]; then
  echo "NNODES must be a positive integer, got: $num_nodes" >&2
  exit 2
fi
if ! [[ "$node_rank" =~ ^[0-9]+$ ]] || (( node_rank >= num_nodes )); then
  echo "NODE_RANK must be in [0, NNODES), got NODE_RANK=$node_rank NNODES=$num_nodes" >&2
  exit 2
fi

cd "$repo_root"
runtime_root="$repo_root/.runtime"
mkdir -p \
  "$runtime_root/wandb/cache" \
  "$runtime_root/wandb/config" \
  "$runtime_root/wandb/data" \
  "$runtime_root/huggingface" \
  "$runtime_root/torchelastic"
# The official 1.2B/500B run uploads metrics; other configs default to offline.
# An explicit WANDB_MODE always takes precedence.
wandb_default_mode=offline
if [[ "$config" == "$repo_root/configs/mixed_data/1p2b_arloop_r3_500b.yaml" ]]; then
  wandb_default_mode=online
fi
export WANDB_MODE="${WANDB_MODE:-$wandb_default_mode}"
export WANDB_ERROR_REPORTING=false
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_CACHE_DIR="$runtime_root/wandb/cache"
export WANDB_CONFIG_DIR="$runtime_root/wandb/config"
export WANDB_DATA_DIR="$runtime_root/wandb/data"
export HF_HOME="$runtime_root/huggingface"
export HF_DATASETS_CACHE="$runtime_root/huggingface/datasets"
export TRANSFORMERS_CACHE="$runtime_root/huggingface/transformers"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHELASTIC_ERROR_FILE="$runtime_root/torchelastic/error-node${node_rank}.json"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
fla_vendor="$repo_root/third_party/flash-linear-attention"
if [[ -d "$fla_vendor/fla" ]]; then
  export PYTHONPATH="$fla_vendor:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
fi
export PARCAE_DEVICES="$gpus_per_node"
export PARCAE_NUM_NODES="$num_nodes"

extra_args=()
if [[ $# -gt 2 ]]; then
  extra_args=("${@:3}")
fi

if (( num_nodes == 1 )); then
  rendezvous_args=(--standalone)
else
  if [[ -z "${MASTER_ADDR:-}" ]]; then
    echo "MASTER_ADDR is required when NNODES > 1" >&2
    exit 2
  fi
  master_port="${MASTER_PORT:-29500}"
  rendezvous_args=(
    --nnodes="$num_nodes"
    --node_rank="$node_rank"
    --master_addr="$MASTER_ADDR"
    --master_port="$master_port"
  )
fi

exec "$torchrun_bin" \
  "${rendezvous_args[@]}" \
  --nproc_per_node="$gpus_per_node" \
  scripts/train/train.py \
  --config "$config" \
  "${extra_args[@]}"
