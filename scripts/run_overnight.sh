#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/overnight/logs

run_or_report() {
  local report="$1"
  shift
  if ! "$@"; then
    {
      echo "# Overnight Stage Error"
      echo
      echo "- Command: \`$*\`"
      echo "- Time: \`$(date -Iseconds)\`"
    } > "$report"
    return 1
  fi
}

{
  date
  nvidia-smi || true
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
  bash scripts/run_smoke.sh || true

  python -m src.data.inspect_thought2text --root data/thought2text --out outputs/overnight/thought2text_inspection.md
  bash scripts/build_thought2text_manifest.sh
  python scripts/make_manifest_report.py --manifest data/thought2text/train.jsonl --root data/thought2text --out outputs/overnight/manifest_report.md

  run_or_report outputs/overnight/clip_cache_error.md python scripts/precompute_vision.py \
    --manifest data/thought2text/train.jsonl \
    --image_root data/thought2text \
    --out data/thought2text/cache/clip_train.npy \
    --index_out data/thought2text/cache/clip_index_train.json \
    --report outputs/overnight/clip_cache_report.md

  python scripts/precompute_vision.py \
    --manifest data/thought2text/val.jsonl \
    --image_root data/thought2text \
    --out data/thought2text/cache/clip_val.npy \
    --index_out data/thought2text/cache/clip_index_val.json \
    --report outputs/overnight/clip_cache_report_val.md

  python scripts/precompute_vision.py \
    --manifest data/thought2text/test.jsonl \
    --image_root data/thought2text \
    --out data/thought2text/cache/clip_test.npy \
    --index_out data/thought2text/cache/clip_index_test.json \
    --report outputs/overnight/clip_cache_report_test.md

  python -m src.train.train_align \
    --config configs/overnight_align.yaml \
    --max_train_samples 512 \
    --max_val_samples 128 \
    --epochs 2 \
    --output_dir outputs/overnight/align_smoke

  python -m src.train.train_align \
    --config configs/overnight_align.yaml \
    --output_dir outputs/overnight/align_strong

  python -m src.eval.retrieval \
    --manifest data/thought2text/test.jsonl \
    --clip_cache data/thought2text/cache/clip_test.npy \
    --eeg_ckpt outputs/overnight/align_strong/checkpoints/best.pt \
    --out outputs/overnight/align_strong/retrieval_metrics.json

  python -m src.train.train_fusion \
    --train_manifest data/thought2text/train.jsonl \
    --val_manifest data/thought2text/val.jsonl \
    --root data/thought2text \
    --clip_train_cache data/thought2text/cache/clip_train.npy \
    --clip_val_cache data/thought2text/cache/clip_val.npy \
    --eeg_ckpt outputs/overnight/align_strong/checkpoints/best.pt \
    --llm Qwen/Qwen2.5-1.5B-Instruct \
    --freeze_llm true \
    --freeze_eeg_encoder true \
    --epochs 5 \
    --batch_size 4 \
    --grad_accum_steps 8 \
    --bf16 true \
    --output_dir outputs/overnight/fusion_qwen15 || true

  python -m src.eval.sanity_check \
    --manifest data/thought2text/test.jsonl \
    --max_samples 128 \
    --caption_ckpt outputs/overnight/fusion_qwen15/checkpoints/best.pt \
    --eeg_ckpt outputs/overnight/align_strong/checkpoints/best.pt \
    --modes vision_only real_eeg shuffled_eeg random_eeg \
    --corruptions clean blur occlusion \
    --out outputs/overnight/sanity_mini || true

  python -m src.eval.metrics \
    --pred_dir outputs/overnight/sanity_mini \
    --out outputs/overnight/sanity_mini/metrics.md || true

  python scripts/make_overnight_report.py \
    --root outputs/overnight \
    --out outputs/overnight/OVERNIGHT_REPORT.md

  date
} 2>&1 | tee outputs/overnight/logs/overnight.log
