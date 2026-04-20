#!/bin/bash
# Train 100M model from scratch: Delta-V AttnRes (V-stream decoupled)
# d=512, L=12, 20k steps — quick validation experiment
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
# effective batch = NGPUS * batch_size * grad_accum = 8 * 4 * 2 = 64
# Matches existing 100M baseline/delta runs (train_scratch.py defaults)

$PYTHON -m torch.distributed.run \
    --nproc_per_node=$NGPUS \
    --rdzv_backend c10d \
    --rdzv_endpoint "localhost:0" \
    "$SCRIPT_DIR/train_scratch.py" \
    --mode        delta_v \
    --hidden_size 512 \
    --num_layers  12 \
    --num_heads   8 \
    --num_kv_heads 4 \
    --intermediate_size 1536 \
    --num_blocks  4 \
    --seq_len     2048 \
    --steps       20000 \
    --batch_size  4 \
    --grad_accum  2 \
    --lr          6e-4 \
    --lr_min      6e-5 \
    --warmup      1000 \
    --save_every  2000 \
    --log_every   10 \
    --run_name    "scratch-delta_v-d512-L12-20k"
