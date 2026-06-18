#!/usr/bin/env bash
set -euo pipefail

python -m src.data.check_split_leakage \
  --train data/thought2text/train.jsonl \
  --val data/thought2text/val.jsonl \
  --test data/thought2text/test.jsonl \
  --out outputs/day2/split_leakage_report.md
