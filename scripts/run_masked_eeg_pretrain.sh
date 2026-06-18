#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/pretrain/masked_eeg_thought2text_heavy.yaml}"
shift || true
python -m src.train.train_masked_eeg_pretrain --config "$CONFIG"
