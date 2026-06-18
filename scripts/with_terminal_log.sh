#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TERMINAL_LOG="${FEISHU_TERMINAL_LOG:-outputs/monitor/terminal.log}"
mkdir -p "$(dirname "$TERMINAL_LOG")"

if [ "$#" -eq 0 ]; then
  echo "用法: bash scripts/with_terminal_log.sh <command> [args...]" >&2
  exit 2
fi

"$@" 2>&1 | tee -a "$TERMINAL_LOG"
