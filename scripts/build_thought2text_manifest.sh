#!/usr/bin/env bash
set -euo pipefail

python -m src.data.build_thought2text_manifest --root data/thought2text
