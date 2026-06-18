#!/usr/bin/env bash
set -euo pipefail

python -m src.data.link_thought2text_images "$@"
