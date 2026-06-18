#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PID_FILE="outputs/monitor/feishu_reporter.pid"
LOG_FILE="outputs/monitor/feishu_reporter.log"

mkdir -p outputs/monitor

if [ -f "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE")"
  if ps -p "$OLD_PID" >/dev/null 2>&1; then
    echo "飞书小时汇报进程已在运行: $OLD_PID"
    exit 0
  fi
fi

setsid bash scripts/run_hourly_feishu_report.sh >> "$LOG_FILE" 2>&1 < /dev/null &
PID="$!"
echo "$PID" > "$PID_FILE"
sleep 1

if ps -p "$PID" >/dev/null 2>&1; then
  echo "已启动飞书小时汇报进程: $PID"
else
  echo "飞书小时汇报进程启动后已退出，请检查 $LOG_FILE" >&2
  exit 1
fi
