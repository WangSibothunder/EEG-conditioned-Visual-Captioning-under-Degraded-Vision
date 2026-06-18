#!/usr/bin/env bash
set -euo pipefail

python scripts/build_text_prototypes.py \
  --data_root data/thought2text \
  --output outputs/semantic_caption/prototypes.pt \
  --report outputs/semantic_caption/prototype_bank.md \
  --splits train val \
  --clip_prefix clip

python -m src.eval.constrained_caption_eval \
  --prototype_bank outputs/semantic_caption/prototypes.pt \
  --manifest data/thought2text/test_human_caption.jsonl \
  --cache_dir data/thought2text/cache \
  --output_dir outputs/semantic_caption \
  --corruptions clean blur occlusion noise lowres \
  --modes vision_only \
  --device auto
