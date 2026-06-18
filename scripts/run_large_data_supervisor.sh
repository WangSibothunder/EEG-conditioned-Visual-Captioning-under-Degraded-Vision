#!/usr/bin/env bash
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."

zip_ok() {
  local zip_path="$1"
  python - "$zip_path" <<'PY' >/dev/null 2>&1
from pathlib import Path
from zipfile import ZipFile
import sys

p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(1)
with ZipFile(p) as zf:
    zf.infolist()
PY
}

json_marker_ok() {
  local marker_path="$1"
  python - "$marker_path" <<'PY' >/dev/null 2>&1
from pathlib import Path
import json
import sys

p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(1)
payload = json.loads(p.read_text())
if payload.get("status") != "complete":
    raise SystemExit(1)
PY
}

extraction_markers_ok() {
  local extracted_dir="$1"
  json_marker_ok "$extracted_dir/.extract_complete.json" \
    && json_marker_ok "$extracted_dir/.nested_extract_complete.json"
}

ensure_tmux_session() {
  local name="$1"
  local script="$2"
  local done_path="${3:-}"
  local extracted_dir="${4:-}"
  if [ -n "$extracted_dir" ] && extraction_markers_ok "$extracted_dir"; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $name already complete: extracted markers present at $extracted_dir"
  elif [ -n "$done_path" ] && zip_ok "$done_path"; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $name already complete: $done_path"
  elif tmux has-session -t "$name" 2>/dev/null; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $name already running"
  else
    tmux new-session -d -s "$name" "$script"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] started $name"
  fi
}

if json_marker_ok /cloud/cloud-ssd1/eeg_vision_caption_data/THINGS-EEG2/.verification_complete.json; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] things_eeg2_dl already complete"
else
  ensure_tmux_session things_eeg2_dl /workspace/scripts/run_things_eeg2_download_loop.sh
fi
ensure_tmux_session \
  eeg_imagenet_dl \
  /workspace/scripts/run_eeg_imagenet_kaggle_download_loop.sh \
  /cloud/cloud-ssd1/eeg_vision_caption_data/EEG-ImageNet/eeg-imagenet.zip \
  /cloud/cloud-ssd1/eeg_vision_caption_data/EEG-ImageNet/extracted
ensure_tmux_session \
  imagenet_dl \
  /workspace/scripts/run_imagenet_kaggle_download_loop.sh \
  /cloud/cloud-ssd1/eeg_vision_caption_data/ImageNet/kaggle_cls_loc/imagenet-object-localization-challenge.zip \
  /cloud/cloud-ssd1/eeg_vision_caption_data/ImageNet/kaggle_cls_loc/extracted
ensure_tmux_session large_data_postprocess /workspace/scripts/run_large_data_postprocess_loop.sh
ensure_tmux_session large_data_progress_report /workspace/scripts/run_large_data_progress_report_loop.sh

tmux ls
