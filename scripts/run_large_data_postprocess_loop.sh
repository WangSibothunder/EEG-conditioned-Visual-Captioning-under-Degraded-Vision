#!/usr/bin/env bash
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/setup_large_data_env.sh
set +e

LOG="$EEG_CAPTION_DATA_ROOT/logs/large_data_postprocess_tmux.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

export POSTPROCESS_POLL_SECONDS="${POSTPROCESS_POLL_SECONDS:-300}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] large data postprocess loop start"
python scripts/postprocess_large_data.py
code=$?
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] large data postprocess loop exit $code"
exit "$code"
