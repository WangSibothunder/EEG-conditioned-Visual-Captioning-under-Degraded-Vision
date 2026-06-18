#!/usr/bin/env bash
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/setup_large_data_env.sh
set +e

LOG="$EEG_CAPTION_DATA_ROOT/logs/large_data_supervisor_tmux.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

while true; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] supervisor tick"
  bash scripts/run_large_data_supervisor.sh
  sleep "${SUPERVISOR_POLL_SECONDS:-300}"
done
