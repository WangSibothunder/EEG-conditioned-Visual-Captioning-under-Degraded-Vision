#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/cloud/cloud-ssd1/eeg_vision_caption_data/THINGS-EEG2}"
CACHE_DIR="${CACHE_DIR:-/cloud/cloud-ssd1/eeg_vision_caption_data/THINGS-EEG2/derived/raw_window_cache_full}"
CONFIG="${CONFIG:-configs/pretrain/masked_eeg_things_eeg2_raw_windows.yaml}"

python scripts/build_things_eeg2_raw_window_cache.py \
  --root "$ROOT" \
  --out_dir "$CACHE_DIR" \
  --window_size "${WINDOW_SIZE:-250}" \
  --stride "${STRIDE:-500}" \
  --channels "${CHANNELS:-64}" \
  --max_train_windows "${MAX_TRAIN_WINDOWS:-0}" \
  --max_val_windows "${MAX_VAL_WINDOWS:-0}"

python -m src.train.train_masked_eeg_pretrain --config "$CONFIG"
