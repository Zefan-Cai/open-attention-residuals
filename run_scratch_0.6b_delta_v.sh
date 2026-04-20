#!/bin/bash
# Train Qwen3-0.6B architecture from scratch: Delta-V AttnRes (V-stream decoupled)
# d=1024, L=28, 20k steps on FineWeb-Edu
# V gets independent depth routing from Q/K

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$SCRIPT_DIR/venv/bin/python3}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="residual"
export WANDB_ENTITY="wdlctc_abr"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_INTERPRET=1
export TRANSFORMERS_NO_FLASH_ATTN=1
export TORCH_COMPILE_DISABLE=1

NGPUS="${NGPUS:-8}"
GRAD_ACCUM=$((32 / (NGPUS * 2)))

$PYTHON -m torch.distributed.run \
    --nproc_per_node=$NGPUS \
    --rdzv_backend c10d \
    --rdzv_endpoint "localhost:0" \
    "$SCRIPT_DIR/train_scratch.py" \
    --mode        delta_v \
    --hidden_size 1024 \
    --num_layers  28 \
    --num_heads   16 \
    --num_kv_heads 8 \
    --intermediate_size 3072 \
    --num_blocks  8 \
    --seq_len     2048 \
    --steps       20000 \
    --batch_size  2 \
    --grad_accum  $GRAD_ACCUM \
    --lr          3e-4 \
    --lr_min      3e-5 \
    --warmup      1000 \
    --save_every  2000 \
    --log_every   10
