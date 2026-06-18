#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/day4_alignment/logs

run_variant() {
  local name="$1"
  local config="$2"
  local out_dir="$3"
  echo "=== ${name} $(date -Iseconds) ==="
  python -m src.train.train_align \
    --config "${config}" \
    --output_dir "${out_dir}"
  python -m src.eval.retrieval \
    --manifest data/thought2text/test.jsonl \
    --clip_cache data/thought2text/cache/clip_test.npy \
    --clip_index data/thought2text/cache/clip_index_test.json \
    --eeg_ckpt "${out_dir}/checkpoints/best.pt" \
    --out "${out_dir}/alignment_metrics.json"
  echo "=== ${name} done $(date -Iseconds) ==="
}

run_variant "B_contrastive_seed42" "configs/day4_align_B_contrastive.yaml" "outputs/day4_alignment/B_contrastive_seed42"
run_variant "C_simdistill_seed42" "configs/day4_align_C_simdistill.yaml" "outputs/day4_alignment/C_simdistill_seed42"
run_variant "D_full_seed42" "configs/day4_align_D_full.yaml" "outputs/day4_alignment/D_full_seed42"
