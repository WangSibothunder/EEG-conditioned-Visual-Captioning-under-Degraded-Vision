#!/usr/bin/env bash
set -euo pipefail

python -m src.eval.metrics \
  --input_dir "${1:-outputs/sanity_real}" \
  --csv outputs/sanity_real/metrics.csv \
  --md outputs/sanity_real/metrics.md \
  --require_corruptions clean blur occlusion noise lowres \
  --require_modes vision_only real_eeg shuffled_eeg random_eeg eeg_only
