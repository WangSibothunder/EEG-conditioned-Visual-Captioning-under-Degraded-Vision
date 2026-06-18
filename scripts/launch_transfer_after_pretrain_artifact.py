from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pretrain_artifact_ready(checkpoint: Path, report: Path | None = None) -> bool:
    if not checkpoint.exists() or checkpoint.stat().st_size <= 0:
        return False
    if report is not None and not report.exists():
        return False
    return True


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def maybe_write_recovery_report(
    *,
    checkpoint: Path,
    report: Path | None,
    metrics: Path | None,
    training_pids: list[int],
    log: Path,
) -> bool:
    if report is None or report.exists():
        return False
    if not training_pids or any(pid_alive(pid) for pid in training_pids):
        return False
    if not checkpoint.exists() or checkpoint.stat().st_size <= 0:
        return False
    if metrics is None or not metrics.exists():
        return False
    try:
        payload = json.loads(metrics.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    history = payload.get("history") if isinstance(payload, dict) else None
    if not isinstance(history, list) or not history:
        return False
    best = min(
        (row for row in history if isinstance(row, dict)),
        key=lambda row: float(row.get("val_loss", float("inf"))),
        default={},
    )
    last = history[-1] if isinstance(history[-1], dict) else {}
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Recovered Masked EEG Pretraining Report",
        "",
        "This report was recovered by the transfer watcher after the monitored pretraining PID exited without writing the formal report.",
        "Use it only to unblock the configured transfer from an existing nonempty best checkpoint.",
        "",
        f"- Recovered UTC: `{utc_now()}`",
        f"- Checkpoint: `{checkpoint}`",
        f"- Metrics: `{metrics}`",
        f"- Last epoch in metrics: `{last.get('epoch', 'unknown')}`",
        f"- Best epoch in metrics: `{best.get('epoch', 'unknown')}`",
        f"- Best val loss in metrics: `{best.get('val_loss', payload.get('best_val_loss', 'unknown'))}`",
        "",
        "## Recovery Boundary",
        "",
        "This fallback is not used while the monitored training PID is still alive.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_log(log, f"{utc_now()} recovered_report report={report} checkpoint={checkpoint} metrics={metrics}")
    return True


def build_transfer_command(config: Path, output_dir: Path) -> list[str]:
    return [
        "bash",
        "scripts/run_masked_pretrain_transfer.sh",
        str(config),
        str(output_dir),
    ]


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def update_queue(queue: Path, job_id: str, status: str, log: Path) -> None:
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
    append_log(log, f"{utc_now()} queue_update status={status} rc={result.returncode} stdout={result.stdout.strip()} stderr={result.stderr.strip()}")


def wait_and_launch(
    checkpoint: Path,
    report: Path | None,
    config: Path,
    output_dir: Path,
    queue: Path,
    job_id: str,
    poll_seconds: int,
    max_wait_seconds: int,
    log: Path,
    training_pids: list[int] | None = None,
    metrics: Path | None = None,
) -> int:
    append_log(log, f"{utc_now()} waiting checkpoint={checkpoint} report={report or 'none'}")
    update_queue(queue, job_id, "waiting", log)
    started = time.monotonic()
    while not pretrain_artifact_ready(checkpoint, report):
        maybe_write_recovery_report(
            checkpoint=checkpoint,
            report=report,
            metrics=metrics,
            training_pids=training_pids or [],
            log=log,
        )
        if pretrain_artifact_ready(checkpoint, report):
            break
        if max_wait_seconds >= 0 and time.monotonic() - started > max_wait_seconds:
            append_log(log, f"{utc_now()} timeout")
            return 2
        time.sleep(poll_seconds)
    append_log(log, f"{utc_now()} ready")
    update_queue(queue, job_id, "running", log)
    command = build_transfer_command(config, output_dir)
    append_log(log, f"{utc_now()} launch command={' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode == 0:
        update_queue(queue, job_id, "completed", log)
    else:
        update_queue(queue, job_id, "failed", log)
    append_log(log, f"{utc_now()} finished rc={result.returncode}")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a Thought2Text transfer after a masked EEG pretrain artifact exists.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--queue", default="configs/heavy_stage_queue.yaml")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-wait-seconds", type=int, default=-1)
    parser.add_argument("--training-pid", action="append", type=int, default=[])
    parser.add_argument("--metrics", default="")
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    raise SystemExit(
        wait_and_launch(
            checkpoint=Path(args.checkpoint),
            report=Path(args.report) if args.report else None,
            config=Path(args.config),
            output_dir=Path(args.output_dir),
            queue=Path(args.queue),
            job_id=args.job_id,
            poll_seconds=args.poll_seconds,
            max_wait_seconds=args.max_wait_seconds,
            log=Path(args.log),
            training_pids=args.training_pid,
            metrics=Path(args.metrics) if args.metrics else None,
        )
    )


if __name__ == "__main__":
    main()
