from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


HEAVY_STAGE_DIR = Path("outputs/heavy_stage")
LIVE_STATUS_PATH = HEAVY_STAGE_DIR / "LIVE_STATUS.md"
GPU_REPORT_PATH = HEAVY_STAGE_DIR / "GPU_UTILIZATION_REPORT.md"
GPU_SAMPLES_PATH = HEAVY_STAGE_DIR / "gpu_samples.jsonl"


@dataclass(frozen=True)
class GPURecord:
    index: int
    name: str
    utilization_gpu_pct: int
    memory_used_mb: int
    memory_total_mb: int
    power_draw_w: float | None
    power_limit_w: float | None
    temperature_c: int | None

    @property
    def memory_used_gb(self) -> float:
        return self.memory_used_mb / 1024.0

    @property
    def memory_total_gb(self) -> float:
        return self.memory_total_mb / 1024.0


@dataclass(frozen=True)
class PythonJob:
    pid: int
    ppid: int
    elapsed: str
    cpu_pct: float
    mem_pct: float
    command: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_optional_float(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_optional_int(value: str) -> int | None:
    parsed = _parse_optional_float(value)
    return None if parsed is None else int(parsed)


def query_gpu_records() -> list[GPURecord]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    records: list[GPURecord] = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        row = next(csv.reader([line], skipinitialspace=True))
        if len(row) < 8:
            continue
        records.append(
            GPURecord(
                index=int(float(row[0])),
                name=row[1].strip(),
                utilization_gpu_pct=int(float(row[2])),
                memory_used_mb=int(float(row[3])),
                memory_total_mb=int(float(row[4])),
                power_draw_w=_parse_optional_float(row[5]),
                power_limit_w=_parse_optional_float(row[6]),
                temperature_c=_parse_optional_int(row[7]),
            )
        )
    return records


def query_active_python_jobs() -> list[PythonJob]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,ppid,etimes,pcpu,pmem,args"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    jobs: list[PythonJob] = []
    current_pid = str(__import__("os").getpid())
    for line in result.stdout.splitlines()[1:]:
        parts = line.strip().split(maxsplit=5)
        if len(parts) < 6:
            continue
        pid, ppid, etimes, pcpu, pmem, command = parts
        command_lower = command.lower()
        if "python" not in command_lower:
            continue
        if pid == current_pid:
            continue
        try:
            elapsed = _format_elapsed_seconds(int(etimes))
            jobs.append(
                PythonJob(
                    pid=int(pid),
                    ppid=int(ppid),
                    elapsed=elapsed,
                    cpu_pct=float(pcpu),
                    mem_pct=float(pmem),
                    command=command,
                )
            )
        except ValueError:
            continue
    return jobs


def _format_elapsed_seconds(seconds: int) -> str:
    hours, remainder = divmod(max(0, seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_snapshot() -> dict[str, object]:
    gpus = query_gpu_records()
    jobs = query_active_python_jobs()
    return {
        "timestamp_utc": utc_now_iso(),
        "gpus": [asdict(gpu) | {"memory_used_gb": gpu.memory_used_gb, "memory_total_gb": gpu.memory_total_gb} for gpu in gpus],
        "active_python_jobs": [asdict(job) for job in jobs],
    }


def append_snapshot(snapshot: dict[str, object]) -> None:
    HEAVY_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    with GPU_SAMPLES_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, sort_keys=True) + "\n")


def _load_recent_samples(limit: int = 24) -> list[dict[str, object]]:
    if not GPU_SAMPLES_PATH.exists():
        return []
    lines = GPU_SAMPLES_PATH.read_text(encoding="utf-8").splitlines()
    samples: list[dict[str, object]] = []
    for line in lines[-limit:]:
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return samples


def _primary_gpu(snapshot: dict[str, object]) -> dict[str, object] | None:
    gpus = snapshot.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        return None
    first = gpus[0]
    return first if isinstance(first, dict) else None


def _read_recent_lines(path: Path, max_lines: int = 2000) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-max_lines:]


def _infer_job_type(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    for key in ["pretrain", "trimodal", "transfer", "architectures", "clip_adapter", "final_semantic"]:
        if key in parts:
            return key
    return "unknown"


def _infer_dataset(job_name: str, path: Path) -> str:
    value = " ".join([job_name, *path.parts]).lower()
    if "things" in value:
        return "THINGS-EEG2"
    if "eeg_imagenet" in value or "imagenet" in value:
        return "EEG-ImageNet"
    if "thought2text" in value or "t2t" in value:
        return "Thought2Text"
    return "unknown"


def collect_training_statuses(outputs_root: Path = Path("outputs"), limit: int = 20) -> list[dict[str, object]]:
    statuses: list[dict[str, object]] = []
    if not outputs_root.exists():
        return statuses

    log_paths = sorted(
        [*outputs_root.rglob("train.stdout.log"), *outputs_root.rglob("train_log.jsonl")],
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    for log_path in log_paths[: max(limit * 3, limit)]:
        latest_step: dict[str, object] = {}
        latest_epoch: dict[str, object] = {}
        best_val: float | None = None
        peak_gpu: float | None = None

        for line in _read_recent_lines(log_path):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue

            if "step" in record:
                latest_step = record
            if "val_loss" in record:
                latest_epoch = record
                val_loss = _optional_float_from_object(record.get("val_loss"))
                if val_loss is not None:
                    best_val = val_loss if best_val is None else min(best_val, val_loss)
            elif "total" in record:
                train_total = _optional_float_from_object(record.get("total"))
                if train_total is not None:
                    best_val = train_total if best_val is None else min(best_val, train_total)
            gpu_mem = _optional_float_from_object(record.get("gpu_mem_peak_mb"))
            if gpu_mem is not None:
                peak_gpu = gpu_mem if peak_gpu is None else max(peak_gpu, gpu_mem)

        if not latest_step and not latest_epoch:
            continue

        job_name = log_path.parent.name
        epoch = latest_step.get("epoch", latest_epoch.get("epoch"))
        current_val = _optional_float_from_object(latest_epoch.get("val_loss"))
        if current_val is None:
            current_val = _optional_float_from_object(latest_step.get("total"))
        status = {
            "active_job": job_name,
            "job_type": _infer_job_type(log_path),
            "dataset": _infer_dataset(job_name, log_path),
            "epoch": epoch,
            "step": latest_step.get("step"),
            "batch_size": latest_step.get("batch_size"),
            "effective_batch_size": latest_step.get("effective_batch_size"),
            "step_time": latest_step.get("step_time", latest_epoch.get("avg_step_time")),
            "current_validation_metric": current_val,
            "best_validation_metric": best_val,
            "gpu_mem_peak_mb": peak_gpu,
            "log_path": str(log_path),
        }
        statuses.append(status)
        if len(statuses) >= limit:
            break

    return statuses


def write_live_status(snapshot: dict[str, object]) -> None:
    gpu = _primary_gpu(snapshot)
    jobs = snapshot.get("active_python_jobs", [])
    lines = [
        "# Heavy Stage Live Status",
        "",
        f"- Last update UTC: {snapshot['timestamp_utc']}",
    ]
    if gpu is None:
        lines.append("- GPU: unavailable (`nvidia-smi` failed or no GPU reported)")
    else:
        lines.extend(
            [
                f"- GPU: {gpu.get('index')} / {gpu.get('name')}",
                f"- Utilization: {gpu.get('utilization_gpu_pct')}%",
                f"- Memory: {float(gpu.get('memory_used_gb', 0.0)):.2f} / {float(gpu.get('memory_total_gb', 0.0)):.2f} GB",
                f"- Power: {_format_watts(gpu.get('power_draw_w'))} / {_format_watts(gpu.get('power_limit_w'))}",
                f"- Active python jobs: {len(jobs) if isinstance(jobs, list) else 0}",
            ]
        )
    lines.extend(["", "## Active Python Jobs", ""])
    if isinstance(jobs, list) and jobs:
        lines.append("| PID | PPID | Elapsed | CPU % | MEM % | Command |")
        lines.append("| --- | --- | --- | ---: | ---: | --- |")
        for job in jobs[:20]:
            if not isinstance(job, dict):
                continue
            lines.append(
                f"| {job.get('pid')} | {job.get('ppid')} | {job.get('elapsed')} | "
                f"{float(job.get('cpu_pct', 0.0)):.1f} | {float(job.get('mem_pct', 0.0)):.1f} | "
                f"`{_shorten(str(job.get('command', '')), 140)}` |"
            )
    else:
        lines.append("No active python jobs detected.")
    LIVE_STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_metric(value: object, precision: int = 4) -> str:
    parsed = _optional_float_from_object(value)
    return "n/a" if parsed is None else f"{parsed:.{precision}f}"


def write_gpu_report(snapshot: dict[str, object], outputs_root: Path = Path("outputs")) -> None:
    samples = _load_recent_samples()
    rows: list[tuple[str, float, int, float | None]] = []
    for sample in samples:
        gpu = _primary_gpu(sample)
        if gpu is None:
            continue
        rows.append(
            (
                str(sample.get("timestamp_utc", "")),
                float(gpu.get("memory_used_gb", 0.0)),
                int(gpu.get("utilization_gpu_pct", 0)),
                _optional_float_from_object(gpu.get("power_draw_w")),
            )
        )

    lines = [
        "# GPU Utilization Report",
        "",
        f"Last update UTC: {snapshot['timestamp_utc']}",
        "",
        "This report is generated by `scripts/gpu_monitor.py`. It records GPU utilization, memory, power, and active python jobs without launching or stopping training processes.",
        "",
        "## Recent Samples",
        "",
    ]
    if rows:
        avg_mem = sum(row[1] for row in rows) / len(rows)
        avg_util = sum(row[2] for row in rows) / len(rows)
        power_rows = [row[3] for row in rows if row[3] is not None]
        avg_power = sum(power_rows) / len(power_rows) if power_rows else None
        lines.extend(
            [
                f"- Samples summarized: {len(rows)}",
                f"- Average memory used: {avg_mem:.2f} GB",
                f"- Average GPU utilization: {avg_util:.1f}%",
                f"- Average power draw: {_format_watts(avg_power)}",
                "",
                "| Timestamp UTC | Memory GB | GPU Util % | Power W |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for timestamp, memory_gb, util_pct, power_w in rows[-12:]:
            lines.append(f"| {timestamp} | {memory_gb:.2f} | {util_pct} | {_format_watts(power_w)} |")
    else:
        lines.append("No valid GPU samples recorded yet.")
    lines.extend(["", "## Training Status", ""])
    statuses = collect_training_statuses(outputs_root)
    if statuses:
        lines.append(
            "| Active Job | Job Type | Dataset | Epoch | Step | Batch Size | Effective Batch | Step Time | Current Val | Best Val | Peak GPU MB | Log |"
        )
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for status in statuses:
            lines.append(
                f"| {status.get('active_job')} | {status.get('job_type')} | {status.get('dataset')} | "
                f"{status.get('epoch', 'n/a')} | {status.get('step', 'n/a')} | "
                f"{status.get('batch_size', 'n/a')} | {status.get('effective_batch_size', 'n/a')} | "
                f"{_format_metric(status.get('step_time'), 3)} | "
                f"{_format_metric(status.get('current_validation_metric'))} | "
                f"{_format_metric(status.get('best_validation_metric'))} | "
                f"{_format_metric(status.get('gpu_mem_peak_mb'), 1)} | "
                f"`{_shorten(str(status.get('log_path', '')), 80)}` |"
            )
    else:
        lines.append("No JSON training status logs found under the outputs root.")
    lines.extend(["", "## Current Jobs", ""])
    jobs = snapshot.get("active_python_jobs", [])
    if isinstance(jobs, list) and jobs:
        lines.append("| PID | Elapsed | CPU % | MEM % | Command |")
        lines.append("| --- | --- | ---: | ---: | --- |")
        for job in jobs[:20]:
            if isinstance(job, dict):
                lines.append(
                    f"| {job.get('pid')} | {job.get('elapsed')} | {float(job.get('cpu_pct', 0.0)):.1f} | "
                    f"{float(job.get('mem_pct', 0.0)):.1f} | `{_shorten(str(job.get('command', '')), 140)}` |"
                )
    else:
        lines.append("No active python jobs detected.")
    GPU_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GPU_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _optional_float_from_object(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_watts(value: object) -> str:
    parsed = _optional_float_from_object(value)
    return "N/A" if parsed is None else f"{parsed:.1f}"


def _shorten(value: str, limit: int) -> str:
    value = value.replace("|", "\\|")
    return value if len(value) <= limit else value[: limit - 3] + "..."


def record_once() -> dict[str, object]:
    snapshot = build_snapshot()
    append_snapshot(snapshot)
    write_live_status(snapshot)
    write_gpu_report(snapshot)
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Record heavy-stage GPU utilization and active python jobs.")
    parser.add_argument("--interval-seconds", type=int, default=300, help="Sampling interval for --watch mode.")
    parser.add_argument("--watch", action="store_true", help="Keep sampling forever. Default records one sample.")
    args = parser.parse_args()

    if args.watch:
        while True:
            record_once()
            time.sleep(max(1, args.interval_seconds))
    else:
        record_once()


if __name__ == "__main__":
    main()
