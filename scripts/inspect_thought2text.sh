#!/usr/bin/env bash
set -euo pipefail

python -m src.data.inspect_thought2text --root data/thought2text --out outputs/thought2text_inspection.md
