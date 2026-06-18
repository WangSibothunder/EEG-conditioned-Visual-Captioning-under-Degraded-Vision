#!/usr/bin/env bash
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/setup_large_data_env.sh
set +e

DEST="$EEG_CAPTION_DATA_ROOT/EEG-ImageNet"
ZIP="$DEST/eeg-imagenet.zip"
EXTRACTED="$DEST/extracted"
LOG="$EEG_CAPTION_DATA_ROOT/logs/eeg_imagenet_kaggle_tmux.log"
mkdir -p "$DEST" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

python - "$EXTRACTED" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
for name in [".extract_complete.json", ".nested_extract_complete.json"]:
    path = root / name
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise SystemExit(1)
print(f"existing extraction markers ok: {root}")
PY
if [ "$?" -eq 0 ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] eeg-imagenet extraction already complete"
  exit 0
fi

python - "$ZIP" <<'PY'
from pathlib import Path
from zipfile import ZipFile, BadZipFile
import sys

p = Path(sys.argv[1])
try:
    with ZipFile(p) as zf:
        print(f"existing zip central directory ok: {p} entries={len(zf.infolist())}")
except (FileNotFoundError, BadZipFile):
    raise SystemExit(1)
PY
if [ "$?" -eq 0 ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] eeg-imagenet download already complete"
  exit 0
fi

for attempt in $(seq 1 100); do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] eeg-imagenet attempt $attempt"
  kaggle datasets download zhannalucky/eeg-imagenet -p "$DEST"
  code=$?
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] eeg-imagenet download exit $code"
  if [ "$code" -eq 0 ]; then
    python - "$ZIP" <<'PY'
from pathlib import Path
from zipfile import ZipFile, BadZipFile
import sys

p = Path(sys.argv[1])
try:
    with ZipFile(p) as zf:
        print(f"zip central directory ok: {p} entries={len(zf.infolist())}")
except (FileNotFoundError, BadZipFile) as exc:
    print(f"zip central directory check failed: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
PY
    verify_code=$?
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] eeg-imagenet central-dir verify exit $verify_code"
    if [ "$verify_code" -eq 0 ]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] eeg-imagenet download complete"
      exit 0
    fi
  fi
  wait_s=$((attempt * 60))
  if [ "$wait_s" -gt 1200 ]; then
    wait_s=1200
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] retrying eeg-imagenet in ${wait_s}s"
  sleep "$wait_s"
done

exit 1
