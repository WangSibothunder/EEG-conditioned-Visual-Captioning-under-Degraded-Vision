from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def manifests_ready(paths: list[Path]) -> bool:
    return bool(paths) and all(path.exists() and path.stat().st_size > 0 for path in paths)


def link_reports_ready(paths: list[Path]) -> bool:
    if not paths:
        return True
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            return False
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return False
        if (
            "Loader-ready status: `fully image-linked`" not in text
            and "Exact-linked subset status: `ready for paired training`" not in text
        ):
            return False
    return True


def build_queue_ready_command(queue: Path, job_id: str) -> list[str]:
    return [
        "python",
        "scripts/update_heavy_stage_queue_status.py",
        "--queue",
        str(queue),
        "--job-id",
        job_id,
        "--status",
        "queued",
    ]


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def wait_and_mark_queued(
    manifests: list[Path],
    *,
    reports: list[Path] | None = None,
    queue: Path,
    job_id: str,
    poll_seconds: int,
    max_wait_seconds: int,
    log: Path,
) -> int:
    report_paths = reports or []
    append_log(
        log,
        (
            f"{utc_now()} waiting manifests={','.join(str(path) for path in manifests)} "
            f"reports={','.join(str(path) for path in report_paths)}"
        ),
    )
    started = time.monotonic()
    while not (manifests_ready(manifests) and link_reports_ready(report_paths)):
        if max_wait_seconds >= 0 and time.monotonic() - started > max_wait_seconds:
            append_log(log, f"{utc_now()} timeout")
            return 2
        time.sleep(max(1, poll_seconds))

    command = build_queue_ready_command(queue, job_id)
    append_log(log, f"{utc_now()} ready command={' '.join(command)}")
    result = subprocess.run(command, text=True, capture_output=True)
    append_log(
        log,
        f"{utc_now()} queue_update rc={result.returncode} stdout={result.stdout.strip()} stderr={result.stderr.strip()}",
    )
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Mark a queue job as queued after required manifest artifacts exist.")
    parser.add_argument("--manifest", action="append", required=True, help="Required manifest path. Repeatable.")
    parser.add_argument("--report", action="append", default=[], help="Required fully image-linked report path. Repeatable.")
    parser.add_argument("--queue", default="configs/heavy_stage_queue.yaml")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-wait-seconds", type=int, default=-1)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    raise SystemExit(
        wait_and_mark_queued(
            [Path(path) for path in args.manifest],
            reports=[Path(path) for path in args.report],
            queue=Path(args.queue),
            job_id=args.job_id,
            poll_seconds=args.poll_seconds,
            max_wait_seconds=args.max_wait_seconds,
            log=Path(args.log),
        )
    )


if __name__ == "__main__":
    main()
