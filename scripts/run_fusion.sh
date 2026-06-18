#!/usr/bin/env bash
set -euo pipefail

TERMINAL_LOG="${FEISHU_TERMINAL_LOG:-outputs/monitor/terminal.log}"
mkdir -p "$(dirname "$TERMINAL_LOG")"

python scripts/make_dummy_data.py --config configs/debug.yaml --num-train 8 --num-val 4 2>&1 | tee -a "$TERMINAL_LOG"
python -m src.train.train_fusion --config configs/debug.yaml 2>&1 | tee -a "$TERMINAL_LOG"
