#!/usr/bin/env bash
set -euo pipefail

python -m pip install --quiet gdown
python -m gdown --folder 'https://drive.google.com/drive/folders/1XqV6MMl28iYXkQBMEFHfEXllGmCbqpOu' -O data/raw/thought2text
