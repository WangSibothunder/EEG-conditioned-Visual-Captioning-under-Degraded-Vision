#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/day5_fusion/logs

ALIGN_CKPT="${ALIGN_CKPT:-outputs/day4_alignment/best_overall.pt}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-data/thought2text/train_human_caption.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-data/thought2text/val_human_caption.jsonl}"

if [[ ! -f "${ALIGN_CKPT}" ]]; then
  echo "Missing alignment checkpoint: ${ALIGN_CKPT}" >&2
  exit 2
fi

run_fusion() {
  local name="$1"
  local mode="$2"
  local out_dir="outputs/day5_fusion/${name}"
  echo "=== ${name} mode=${mode} $(date -Iseconds) ==="
  python -m src.train.train_fusion \
    --train_manifest "${TRAIN_MANIFEST}" \
    --val_manifest "${VAL_MANIFEST}" \
    --root data/thought2text \
    --clip_train_cache data/thought2text/cache/clip_train.npy \
    --clip_val_cache data/thought2text/cache/clip_val.npy \
    --eeg_ckpt "${ALIGN_CKPT}" \
    --llm Qwen/Qwen2.5-1.5B-Instruct \
    --freeze_llm true \
    --freeze_eeg_encoder true \
    --epochs 10 \
    --batch_size 4 \
    --grad_accum_steps 8 \
    --bf16 true \
    --train_mode "${mode}" \
    --output_dir "${out_dir}"
}

run_fusion "F0_vision_only" "vision_only"
run_fusion "F1_real_eeg" "real_eeg"
run_fusion "F2_random_encoder_control" "random_eeg"
run_fusion "F3_shuffled_training_control" "shuffled_eeg"
