#!/usr/bin/env bash
set -euo pipefail

PRED_DIR="${1:-outputs/day3/sanity_real}"

python -m src.eval.metrics \
  --pred_dir "${PRED_DIR}" \
  --out "${PRED_DIR}/metrics.md" \
  --require_modes vision_only real_eeg shuffled_eeg random_eeg eeg_only \
  --require_corruptions clean blur occlusion noise lowres

python -m src.eval.gate_analysis \
  --pred_dir "${PRED_DIR}" \
  --out "${PRED_DIR}/gate_analysis.md" \
  --sample_out "${PRED_DIR}/sample_predictions.jsonl" \
  --sample_limit 10
