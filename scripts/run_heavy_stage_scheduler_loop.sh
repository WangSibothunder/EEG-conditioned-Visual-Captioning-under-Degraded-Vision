#!/usr/bin/env bash
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."
set +e

LOG="outputs/heavy_stage/heavy_stage_scheduler_loop.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] heavy stage scheduler loop start"
while true; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] scheduler tick"
  python scripts/gpu_monitor.py
  python scripts/heavy_stage_scheduler.py --launch-when-idle --max-launches "${HEAVY_STAGE_MAX_LAUNCHES:-8}"
  python scripts/materialize_heavy_stage_reports.py --outputs-root outputs
  python scripts/update_large_data_progress_report.py
  sleep "${HEAVY_STAGE_SCHEDULER_POLL_SECONDS:-300}"
done
