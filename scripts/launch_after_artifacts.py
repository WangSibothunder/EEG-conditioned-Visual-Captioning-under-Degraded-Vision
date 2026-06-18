from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def all_artifacts_ready(paths: list[Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths)


def build_shell_command(command: str) -> list[str]:
    return ["bash", "-lc", command]


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def update_queue(queue: Path, job_id: str, status: str, log: Path) -> None:
    if not job_id:
        return
    command = [
        "python",
        "scripts/update_heavy_stage_queue_status.py",
        "--queue",
        str(queue),
        "--job-id",
        job_id,
        "--status",
        status,
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    append_log(log, f"{utc_now()} queue_update job_id={job_id} status={status} rc={result.returncode} stdout={result.stdout.strip()} stderr={result.stderr.strip()}")


def wait_and_launch(
    artifacts: list[Path],
    command: str,
    log: Path,
    *,
    poll_seconds: int = 300,
    max_wait_seconds: int = -1,
    queue: Path = Path("configs/heavy_stage_queue.yaml"),
    job_id: str = "",
) -> int:
    append_log(log, f"{utc_now()} waiting artifacts={','.join(str(item) for item in artifacts)}")
    update_queue(queue, job_id, "waiting", log)
    started = time.monotonic()
    while not all_artifacts_ready(artifacts):
        if max_wait_seconds >= 0 and time.monotonic() - started > max_wait_seconds:
            append_log(log, f"{utc_now()} timeout")
            return 2
        time.sleep(poll_seconds)
    append_log(log, f"{utc_now()} ready")
    update_queue(queue, job_id, "running", log)
    append_log(log, f"{utc_now()} launch command={command}")
    result = subprocess.run(build_shell_command(command))
    update_queue(queue, job_id, "completed" if result.returncode == 0 else "failed", log)
    append_log(log, f"{utc_now()} finished rc={result.returncode}")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a shell command after required artifacts exist.")
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-wait-seconds", type=int, default=-1)
    parser.add_argument("--queue", default="configs/heavy_stage_queue.yaml")
    parser.add_argument("--job-id", default="")
    args = parser.parse_args()
    raise SystemExit(
        wait_and_launch(
            [Path(item) for item in args.artifact],
            args.command,
            Path(args.log),
            poll_seconds=args.poll_seconds,
            max_wait_seconds=args.max_wait_seconds,
            queue=Path(args.queue),
            job_id=args.job_id,
        )
    )


if __name__ == "__main__":
    main()
