#!/usr/bin/env bash
set -euo pipefail

python -m src.train.train_align \
  --config configs/day2_align.yaml \
  --output_dir outputs/day2_align
