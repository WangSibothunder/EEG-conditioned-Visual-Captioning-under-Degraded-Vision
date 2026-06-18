#!/usr/bin/env bash
set -euo pipefail

python -m src.train.train_semantic_fusion \
  --train_manifest data/thought2text/train_human_caption.jsonl \
  --val_manifest data/thought2text/val_human_caption.jsonl \
  --train_cache data/thought2text/cache/clip_train.npy \
  --val_cache data/thought2text/cache/clip_val.npy \
  --train_index data/thought2text/cache/clip_index_train.json \
  --val_index data/thought2text/cache/clip_index_val.json \
  --eeg_checkpoint outputs/day4_alignment/best_overall.pt \
  --output_dir outputs/semantic_caption \
  --epochs 3 \
  --batch_size 64
