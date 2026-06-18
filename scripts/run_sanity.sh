#!/usr/bin/env bash
set -euo pipefail

TERMINAL_LOG="${FEISHU_TERMINAL_LOG:-outputs/monitor/terminal.log}"
mkdir -p "$(dirname "$TERMINAL_LOG")"

python -m src.eval.sanity_check --config configs/debug.yaml --checkpoint outputs/debug/fusion/checkpoint_last.pt 2>&1 | tee -a "$TERMINAL_LOG"
