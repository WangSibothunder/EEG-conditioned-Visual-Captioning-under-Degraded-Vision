#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PID_FILE="outputs/monitor/feishu_reporter.pid"
if [ ! -f "$PID_FILE" ]; then
  echo "未找到飞书小时汇报进程 PID 文件: $PID_FILE"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill "$PID" 2>/dev/null; then
  rm -f "$PID_FILE"
  echo "已停止飞书小时汇报进程: $PID"
else
  rm -f "$PID_FILE"
  echo "飞书小时汇报进程不存在或已退出: $PID"
fi
