#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/setup_large_data_env.sh

LOG="$EEG_CAPTION_DATA_ROOT/logs/large_data_progress_report_tmux.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] large data progress report loop start"
while true; do
  python scripts/update_large_data_progress_report.py || true
  sleep "${PROGRESS_REPORT_POLL_SECONDS:-300}"
done
