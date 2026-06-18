#!/usr/bin/env bash
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/setup_large_data_env.sh
set +e

DEST="$EEG_CAPTION_DATA_ROOT/THINGS-EEG2"
CACHE="$EEG_CAPTION_DATA_ROOT/model_cache/huggingface/hub"
LOG="$EEG_CAPTION_DATA_ROOT/logs/things_eeg2_tmux.log"
STATUS="$EEG_CAPTION_DATA_ROOT/logs/things_eeg2_python_status.jsonl"

mkdir -p "$DEST" "$CACHE" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

for attempt in $(seq 1 100); do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] things-eeg2 attempt $attempt"
  python scripts/download_hf_repo_files.py \
    --repo gasparyanartur/things-eeg2 \
    --repo-type dataset \
    --local-dir "$DEST" \
    --cache-dir "$CACHE" \
    --status-jsonl "$STATUS" \
    --sleep-seconds 0.2 \
    --list-retries 8 \
    --download-retries 5 \
    --retry-sleep-seconds 30
  code=$?
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] things-eeg2 exit $code"
  if [ "$code" -eq 0 ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] things-eeg2 download complete"
    exit 0
  fi
  wait_s=$((attempt * 60))
  if [ "$wait_s" -gt 900 ]; then
    wait_s=900
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] retrying in ${wait_s}s"
  sleep "$wait_s"
done

exit 1
