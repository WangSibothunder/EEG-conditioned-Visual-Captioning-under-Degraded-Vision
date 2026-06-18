from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def run(command: list[str]) -> tuple[int, str]:
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout


def parse_files(dataset: str) -> list[tuple[str, int]]:
    code, output = run(["kaggle", "datasets", "files", dataset])
    if code != 0:
        raise RuntimeError(output)
    rows: list[tuple[str, int]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("name ") or line.startswith("---") or line.startswith("Next Page Token"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        size_text = parts[1]
        if not size_text.isdigit():
            continue
        rows.append((name, int(size_text)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kaggle dataset files one by one with retry logs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--status-jsonl", required=True)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=30.0)
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    status_path = Path(args.status_jsonl)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    files = parse_files(args.dataset)
    print(json.dumps({"event": "file_list", "count": len(files), "bytes": sum(size for _, size in files)}), flush=True)

    for index, (name, expected_size) in enumerate(files, start=1):
        local = dest / name
        if local.exists() and local.stat().st_size == expected_size:
            record = {
                "event": "skip_existing",
                "index": index,
                "count": len(files),
                "file": name,
                "expected_size": expected_size,
                "actual_size": local.stat().st_size,
            }
            print(json.dumps(record), flush=True)
            status_path.open("a", encoding="utf-8").write(json.dumps(record) + "\n")
            continue

        for attempt in range(1, args.max_attempts + 1):
            start = time.time()
            code, output = run([
                "kaggle",
                "datasets",
                "download",
                args.dataset,
                "--file",
                name,
                "--path",
                str(dest),
                "--force",
            ])
            actual_size = local.stat().st_size if local.exists() else 0
            ok = code == 0 and actual_size == expected_size
            record = {
                "event": "downloaded" if ok else "retry",
                "index": index,
                "count": len(files),
                "file": name,
                "attempt": attempt,
                "expected_size": expected_size,
                "actual_size": actual_size,
                "exit_code": code,
                "seconds": round(time.time() - start, 3),
                "output_tail": output[-1000:],
            }
            print(json.dumps(record), flush=True)
            status_path.open("a", encoding="utf-8").write(json.dumps(record) + "\n")
            if ok:
                break
            time.sleep(args.sleep_seconds * attempt)
        else:
            raise SystemExit(f"failed to download {name}")


if __name__ == "__main__":
    main()
