from __future__ import annotations

import argparse
from pathlib import Path


def update_queue_status(queue_path: str | Path, job_id: str, new_status: str) -> bool:
    path = Path(queue_path)
    if not path.exists():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    updated: list[str] = []
    in_target = False
    changed = False
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if stripped.startswith("- id: "):
            in_target = stripped == f"- id: {job_id}"
            updated.append(line)
            continue
        if in_target and stripped.startswith("status:"):
            indent_prefix = line[: len(line) - len(line.lstrip())]
            updated.append(f"{indent_prefix}status: {new_status}")
            changed = True
            in_target = False
            continue
        updated.append(line)
    if changed:
        path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Update a single job status in the heavy-stage queue YAML.")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--status", required=True)
    args = parser.parse_args()
    changed = update_queue_status(args.queue, args.job_id, args.status)
    print(f"changed={changed}")


if __name__ == "__main__":
    main()
