#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/transfer/masked_pretrain_t2t_align.yaml}"
OUT="${2:-outputs/transfer/masked_pretrain_t2t_align}"

python -m src.train.train_align \
  --config "$CONFIG" \
  --output_dir "$OUT" \
  --max_train_samples 0 \
  --max_val_samples 0
