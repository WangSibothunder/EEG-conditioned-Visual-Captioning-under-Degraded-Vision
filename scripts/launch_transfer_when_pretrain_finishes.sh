#!/usr/bin/env bash
set -euo pipefail

PRETRAIN_PID="${1:?pretrain pid required}"
CONFIG="${2:-configs/transfer/masked_pretrain_t2t_align.yaml}"
OUT="${3:-outputs/transfer/masked_pretrain_t2t_align}"
POLL_SECONDS="${4:-60}"
QUEUE_PATH="${HEAVY_STAGE_QUEUE_PATH:-configs/heavy_stage_queue.yaml}"
JOB_ID="${HEAVY_STAGE_QUEUE_JOB_ID:-TRANSFER_EEG_IMAGENET_PRETRAIN_TO_THOUGHT2TEXT}"

mkdir -p "$OUT"
LOG="$OUT/transfer_launcher.log"

mark_failed() {
  python scripts/update_heavy_stage_queue_status.py --queue "$QUEUE_PATH" --job-id "$JOB_ID" --status failed || true
}
trap mark_failed ERR

{
  echo "waiting_for_pretrain_pid=$PRETRAIN_PID"
  python scripts/update_heavy_stage_queue_status.py --queue "$QUEUE_PATH" --job-id "$JOB_ID" --status waiting || true
  while kill -0 "$PRETRAIN_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
  echo "pretrain_finished_at=$(date -u +%FT%TZ)"
  python scripts/update_heavy_stage_queue_status.py --queue "$QUEUE_PATH" --job-id "$JOB_ID" --status running || true
  bash scripts/run_masked_pretrain_transfer.sh "$CONFIG" "$OUT"
  python scripts/update_heavy_stage_queue_status.py --queue "$QUEUE_PATH" --job-id "$JOB_ID" --status completed || true
} >>"$LOG" 2>&1
