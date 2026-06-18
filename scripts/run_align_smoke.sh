#!/usr/bin/env bash
set -euo pipefail

python -m src.train.train_align \
  --config configs/day2_align.yaml \
  --max_train_samples 512 \
  --max_val_samples 128 \
  --epochs 2 \
  --output_dir outputs/day2_align_smoke
