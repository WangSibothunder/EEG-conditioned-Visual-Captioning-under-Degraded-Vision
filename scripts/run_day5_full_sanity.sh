#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/day5_sanity/logs

CAPTION_CKPT="${CAPTION_CKPT:-outputs/day5_fusion/F1_real_eeg/checkpoints/best.pt}"
EEG_CKPT="${EEG_CKPT:-outputs/day4_alignment/best_overall.pt}"
MANIFEST="${MANIFEST:-data/thought2text/test_human_caption.jsonl}"

python -m src.eval.sanity_check \
  --manifest "${MANIFEST}" \
  --max_samples -1 \
  --caption_ckpt "${CAPTION_CKPT}" \
  --eeg_ckpt "${EEG_CKPT}" \
  --modes vision_only real_eeg shuffled_eeg random_eeg eeg_only \
  --corruptions clean blur occlusion noise lowres \
  --use_degraded_clip_cache true \
  --degraded_cache_dir data/thought2text/cache/degraded_test \
  --out outputs/day5_sanity

python -m src.eval.metrics \
  --pred_dir outputs/day5_sanity \
  --csv outputs/day5_sanity/FULL_SANITY_METRICS.csv \
  --out outputs/day5_sanity/FULL_SANITY_METRICS.md \
  --require_modes vision_only real_eeg shuffled_eeg random_eeg eeg_only \
  --require_corruptions clean blur occlusion noise lowres

python -m src.eval.gate_analysis \
  --pred_dir outputs/day5_sanity \
  --out outputs/day5_sanity/gate_analysis.md \
  --sample_out outputs/day5_sanity/sample_predictions.jsonl \
  --sample_limit 30

python scripts/make_day5_sanity_artifacts.py \
  --root outputs/day5_sanity \
  --out outputs/day5_sanity/qualitative_examples.md
