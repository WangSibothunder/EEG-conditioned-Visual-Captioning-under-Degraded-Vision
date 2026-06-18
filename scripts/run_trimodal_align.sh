#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/trimodal/full_masked_pretrained.yaml}
OUT_DIR=${2:-}

TEXT_ARGS=(
  --data_dir data/thought2text
  --out data/thought2text/cache/text_embeddings.npy
  --index_out data/thought2text/cache/text_index.json
  --report outputs/trimodal/text_embedding_report.md
  --batch_size 64
)

if [[ "${REBUILD_TEXT_CACHE:-0}" == "1" ]]; then
  TEXT_ARGS+=(--overwrite --require_real_model)
fi

python scripts/build_trimodal_text_embeddings.py "${TEXT_ARGS[@]}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  python -m src.train.train_trimodal_align \
    --config "$CONFIG" \
    --out_dir "${OUT_DIR:-outputs/trimodal/smoke}" \
    --epochs 2 \
    --batch_size 32 \
    --max_train_samples 256 \
    --max_val_samples 256 \
    --log_every 1
else
  if [[ -n "$OUT_DIR" ]]; then
    python -m src.train.train_trimodal_align --config "$CONFIG" --out_dir "$OUT_DIR"
  else
    python -m src.train.train_trimodal_align --config "$CONFIG"
  fi
fi
