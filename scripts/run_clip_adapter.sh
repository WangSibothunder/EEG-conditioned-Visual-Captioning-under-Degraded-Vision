#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/clip_adapter/thought2text_adapter.yaml"
EXTRA_ARGS=()

if [[ $# -gt 0 && "${1}" != --* ]]; then
  CONFIG="$1"
  shift
fi

if [[ $# -gt 0 && "${1}" != --* ]]; then
  EXTRA_ARGS+=(--output_dir "$1")
  shift
fi

python -m src.train.train_clip_adapter \
  --config "$CONFIG" \
  "${EXTRA_ARGS[@]}" \
  "$@"
