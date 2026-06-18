#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

INTERVAL_SECONDS="${FEISHU_REPORT_INTERVAL_SECONDS:-3600}"
TERMINAL_LOG="${FEISHU_TERMINAL_LOG:-outputs/monitor/terminal.log}"

mkdir -p "$(dirname "$TERMINAL_LOG")"
touch "$TERMINAL_LOG"

echo "飞书小时汇报已启动: interval=${INTERVAL_SECONDS}s terminal_log=${TERMINAL_LOG}"

while true; do
  python -m src.utils.feishu_report --terminal-log "$TERMINAL_LOG" || true
  sleep "$INTERVAL_SECONDS"
done
