from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reconcile_heavy_stage_queue_status import reconcile_transfer_job_status
from scripts.update_heavy_stage_queue_status import update_queue_status


HEAVY_STAGE_DIR = Path("outputs/heavy_stage")
QUEUE_PATH = Path("configs/heavy_stage_queue.yaml")
BOARD_CSV_PATH = HEAVY_STAGE_DIR / "EXPERIMENT_BOARD.csv"
BOARD_MD_PATH = HEAVY_STAGE_DIR / "EXPERIMENT_BOARD.md"
IDLE_DIAGNOSIS_PATH = HEAVY_STAGE_DIR / "GPU_IDLE_DIAGNOSIS.md"
LIVE_STATUS_PATH = HEAVY_STAGE_DIR / "LIVE_STATUS.md"
SCHEDULER_STATE_PATH = HEAVY_STAGE_DIR / "scheduler_state.json"
AUTO_LAUNCH_LOG_PATH = HEAVY_STAGE_DIR / "auto_launches.jsonl"


@dataclass(frozen=True)
class QueueItem:
    id: str
    command: str
    status: str
    priority: int
    expected_output: str
    notes: str


@dataclass(frozen=True)
class GPUState:
    available: bool
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    utilization_gpu_pct: int = 0
    power_draw_w: float | None = None
    name: str = "unknown"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def query_gpu_state() -> GPUState:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return GPUState(available=False)
    first = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not first:
        return GPUState(available=False)
    row = next(csv.reader([first], skipinitialspace=True))
    if len(row) < 5:
        return GPUState(available=False)
    return GPUState(
        available=True,
        name=row[0].strip(),
        memory_used_gb=float(row[1]) / 1024.0,
        memory_total_gb=float(row[2]) / 1024.0,
        utilization_gpu_pct=int(float(row[3])),
        power_draw_w=_parse_optional_float(row[4]),
    )


def _parse_optional_float(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_timestamp_utc(value: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def count_active_python_jobs() -> int:
    return len(query_active_python_jobs())


def query_active_python_jobs() -> list[dict[str, str]]:
    try:
        result = subprocess.run(["ps", "-eo", "pid,ppid,etimes,pcpu,pmem,args"], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    current_pid = str(__import__("os").getpid())
    jobs: list[dict[str, str]] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.strip().split(maxsplit=5)
        if len(parts) < 6:
            continue
        pid, ppid, etimes, pcpu, pmem, command = parts
        if pid == current_pid or "python" not in command.lower():
            continue
        jobs.append(
            {
                "pid": pid,
                "ppid": ppid,
                "elapsed": _format_elapsed_seconds(int(etimes)) if etimes.isdigit() else etimes,
                "cpu_pct": pcpu,
                "mem_pct": pmem,
                "command": command,
            }
        )
    return jobs


def _format_elapsed_seconds(seconds: int) -> str:
    hours, remainder = divmod(max(0, seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_queue(path: Path = QUEUE_PATH) -> list[QueueItem]:
    if not path.exists():
        return []
    return _parse_queue_yaml(path.read_text(encoding="utf-8"))


def _parse_queue_yaml(text: str) -> list[QueueItem]:
    # Minimal YAML subset parser for this repo-owned queue file. Keeps runtime dependency-free.
    items: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    in_jobs = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "jobs:":
            in_jobs = True
            continue
        if not in_jobs:
            continue
        if stripped.startswith("- "):
            if current is not None:
                items.append(current)
            current = {}
            remainder = stripped[2:].strip()
            if remainder and ":" in remainder:
                key, value = remainder.split(":", 1)
                current[key.strip()] = _clean_yaml_scalar(value.strip())
            continue
        if current is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current[key.strip()] = _clean_yaml_scalar(value.strip())
    if current is not None:
        items.append(current)

    queue: list[QueueItem] = []
    for item in items:
        queue.append(
            QueueItem(
                id=item.get("id", "unknown"),
                command=item.get("command", ""),
                status=item.get("status", "queued"),
                priority=int(item.get("priority", "999") or 999),
                expected_output=item.get("expected_output", ""),
                notes=item.get("notes", ""),
            )
        )
    return sorted(queue, key=lambda entry: (entry.priority, entry.id))


def _clean_yaml_scalar(value: str) -> str:
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def infer_item_status(item: QueueItem) -> str:
    status = item.status.lower()
    if status in {"failed", "blocked", "skipped"}:
        return status
    if item.id in {"DATASET_EEG_IMAGENET_IMAGE_LINK", "EEG_IMAGENET_PAIRED_ALIGNMENT_READY_WATCHER"}:
        if _eeg_imagenet_link_reports_fully_ready():
            return "completed"
        if _eeg_imagenet_exact_subset_ready():
            return "partial_ready" if item.id == "DATASET_EEG_IMAGENET_IMAGE_LINK" else "completed"
        return status if status in {"running", "waiting", "queued"} else "waiting"
    if (
        status == "running"
        and _is_direct_gpu_training_command(item.command)
        and _has_active_matching_process(item.command)
    ):
        return "running"
    if item.expected_output and Path(item.expected_output).exists():
        return "completed"
    if status in {"completed", "running"}:
        return item.status.lower()
    return status or "queued"


def _has_active_matching_process(command: str) -> bool:
    needle = _normalize_command_for_process_match(command)
    if not needle:
        return False
    for job in query_active_python_jobs():
        active_command = _normalize_command_for_process_match(str(job.get("command", "")))
        if needle in active_command or active_command in needle:
            return True
    return False


def _normalize_command_for_process_match(command: str) -> str:
    tokens = command.split()
    filtered: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in {">", ">>", "2>", "2>>"}:
            skip_next = True
            continue
        if token.startswith((">", "2>")):
            continue
        if token == "2>&1":
            continue
        filtered.append(token)
    return " ".join(filtered)


def _eeg_imagenet_link_reports_fully_ready(project_root: Path | None = None) -> bool:
    if project_root is None:
        project_root = PROJECT_ROOT
    report_dir = project_root / "outputs" / "datasets"
    report_paths = [
        report_dir / "EEG_IMAGENET_IMAGE_LINK_TRAIN_REPORT.md",
        report_dir / "EEG_IMAGENET_IMAGE_LINK_VAL_REPORT.md",
        report_dir / "EEG_IMAGENET_IMAGE_LINK_TEST_REPORT.md",
    ]
    for report_path in report_paths:
        if not report_path.exists():
            return False
        text = report_path.read_text(encoding="utf-8")
        if "Loader-ready status: `fully image-linked`" not in text:
            return False
    return True


def _eeg_imagenet_exact_subset_ready(project_root: Path | None = None) -> bool:
    if project_root is None:
        project_root = PROJECT_ROOT
    report_dir = project_root / "outputs" / "datasets"
    manifest_dir = project_root / "data" / "EEG-ImageNet"
    report_paths = [
        report_dir / "EEG_IMAGENET_IMAGE_LINK_TRAIN_REPORT.md",
        report_dir / "EEG_IMAGENET_IMAGE_LINK_VAL_REPORT.md",
        report_dir / "EEG_IMAGENET_IMAGE_LINK_TEST_REPORT.md",
    ]
    manifest_paths = [
        manifest_dir / "train_image_exact.jsonl",
        manifest_dir / "val_image_exact.jsonl",
        manifest_dir / "test_image_exact.jsonl",
    ]
    if not all(path.exists() and path.stat().st_size > 0 for path in manifest_paths):
        return False
    for report_path in report_paths:
        if not report_path.exists():
            return False
        text = report_path.read_text(encoding="utf-8")
        if "Exact-linked subset status: `ready for paired training`" not in text:
            return False
    return True


def build_board_rows(queue: list[QueueItem]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in queue:
        rows.append(
            {
                "ID": item.id,
                "Priority": str(item.priority),
                "Status": infer_item_status(item),
                "Command": item.command,
                "Expected Output": item.expected_output,
                "Notes": item.notes,
            }
        )
    return rows


def sync_inferred_queue_statuses(queue_path: Path, queue: list[QueueItem], rows: list[dict[str, str]]) -> None:
    """Persist inferred terminal statuses so completed jobs do not remain running in YAML."""
    by_id = {row["ID"]: row for row in rows}
    mutable_statuses = {"queued", "waiting", "running", "partial_ready"}
    inferred_terminal = {"completed", "blocked", "skipped", "failed"}
    for item in queue:
        current_status = item.status.lower()
        inferred = by_id.get(item.id, {}).get("Status", "").lower()
        if current_status in mutable_statuses and inferred in inferred_terminal and inferred != current_status:
            update_queue_status(queue_path, item.id, inferred)


def write_board(rows: list[dict[str, str]]) -> None:
    HEAVY_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ID", "Priority", "Status", "Command", "Expected Output", "Notes"]
    with BOARD_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Heavy Stage Experiment Board",
        "",
        f"Last scheduler update UTC: {utc_now_iso()}",
        "",
        "| ID | Priority | Status | Command | Expected Output | Notes |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {ID} | {Priority} | {Status} | `{Command}` | `{Expected Output}` | {Notes} |".format(
                **{key: _escape_md(value) for key, value in row.items()}
            )
        )
    BOARD_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _escape_md(value: str) -> str:
    return str(value).replace("|", "\\|")


def select_next_queued(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if row["Status"] == "queued":
            return row
    return None


def _is_direct_gpu_training_command(command: str) -> bool:
    lowered = command.lower()
    gpu_markers = [
        "src.train.",
        "precompute_vision.py",
        "precompute_degraded_vision.py",
        "run_masked_eeg_pretrain",
        "run_trimodal_align",
        "run_align.sh",
    ]
    return any(marker in lowered for marker in gpu_markers)


def _is_heavy_gpu_training_command(command: str) -> bool:
    lowered = command.lower()
    heavy_markers = [
        "src.train.train_masked_eeg_pretrain",
        "src.train.train_trimodal_align",
        "src.train.train_align",
        "run_masked_eeg_pretrain",
        "run_trimodal_align",
        "run_align.sh",
    ]
    return any(marker in lowered for marker in heavy_markers)


def _is_safe_to_auto_launch(row: dict[str, str], gpu: GPUState | None) -> bool:
    if gpu is None or not gpu.available:
        return True
    memory_free_gb = gpu.memory_total_gb - gpu.memory_used_gb
    if memory_free_gb >= 16.0:
        return True
    return not _is_direct_gpu_training_command(row.get("Command", ""))


def _gpu_snapshot_is_underused(snapshot: dict[str, object]) -> bool:
    gpus = snapshot.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        return False
    first = gpus[0]
    if not isinstance(first, dict):
        return False
    try:
        memory_used_gb = float(first.get("memory_used_gb", 0.0))
        utilization_gpu_pct = int(float(first.get("utilization_gpu_pct", 0)))
    except (TypeError, ValueError):
        return False
    return memory_used_gb < 8.0 or utilization_gpu_pct < 30


def _load_recent_gpu_samples(path: Path = Path("outputs/heavy_stage/gpu_samples.jsonl"), window_minutes: int = 10) -> list[dict[str, object]]:
    if not path.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
    recent_samples: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            sample = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(sample, dict):
            continue
        timestamp_raw = sample.get("timestamp_utc")
        if not isinstance(timestamp_raw, str):
            continue
        timestamp = _parse_timestamp_utc(timestamp_raw)
        if timestamp is None or timestamp < cutoff:
            continue
        recent_samples.append(sample)
    return recent_samples


def should_write_idle_diagnosis(
    gpu: GPUState,
    samples: list[dict[str, object]],
    *,
    current_time: datetime | None = None,
    window_minutes: int = 10,
) -> tuple[bool, dict[str, int]]:
    if current_time is None:
        current_time = datetime.now(timezone.utc)
    cutoff = current_time - timedelta(minutes=window_minutes)
    window_samples: list[dict[str, object]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        timestamp_raw = sample.get("timestamp_utc")
        if not isinstance(timestamp_raw, str):
            continue
        timestamp = _parse_timestamp_utc(timestamp_raw)
        if timestamp is None or timestamp < cutoff:
            continue
        window_samples.append(sample)

    if not window_samples:
        return False, {"window_samples": 0, "underused_samples": 0}

    underused_samples = sum(1 for sample in window_samples if _gpu_snapshot_is_underused(sample))
    return underused_samples > len(window_samples) / 2.0, {
        "window_samples": len(window_samples),
        "underused_samples": underused_samples,
    }


def write_idle_diagnosis(gpu: GPUState, rows: list[dict[str, str]], active_python_jobs: int) -> bool:
    recent_samples = _load_recent_gpu_samples()
    is_idle, window_stats = should_write_idle_diagnosis(gpu, recent_samples)
    if not is_idle:
        return False
    next_item = select_next_queued(rows)
    lines = [
        "# GPU Idle Diagnosis",
        "",
        f"- Timestamp UTC: {utc_now_iso()}",
        f"- Recent sample window: last 10 minutes",
        f"- Recent samples considered: {window_stats['window_samples']}",
        f"- Underused samples: {window_stats['underused_samples']}",
        f"- GPU available: {gpu.available}",
        f"- GPU name: {gpu.name}",
        f"- Memory used: {gpu.memory_used_gb:.2f} / {gpu.memory_total_gb:.2f} GB",
        f"- GPU utilization: {gpu.utilization_gpu_pct}%",
        f"- Power draw: {_format_watts(gpu.power_draw_w)} W",
        f"- Active python jobs: {active_python_jobs}",
        "",
        "Diagnosis:",
        "- Heavy-stage policy considers the GPU underused when a recent sample shows memory below 8 GB or utilization below 30%.",
        "- The scheduler writes an idle diagnosis only when a majority of samples in the last 10 minutes are underused.",
        "- When invoked with `--launch-when-idle`, the scheduler attempts to launch the next queued job after writing this diagnosis.",
        "- The scheduler never kills running jobs; it only launches an already queued command when the idle policy triggers.",
        "",
        "Recommended next queued job:",
    ]
    if next_item is None:
        lines.append("- None. Queue has no `queued` items.")
    else:
        lines.extend(
            [
                f"- ID: {next_item['ID']}",
                f"- Command: `{next_item['Command']}`",
                f"- Expected output: `{next_item['Expected Output']}`",
            ]
        )
    IDLE_DIAGNOSIS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def launch_next_queued_job(
    rows: list[dict[str, str]],
    *,
    launch_enabled: bool,
    log_dir: Path = HEAVY_STAGE_DIR,
    queue_path: Path | None = None,
    max_launches: int = 1,
    gpu: GPUState | None = None,
) -> int:
    if not launch_enabled:
        return 0
    if max_launches < 1:
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    auto_log = log_dir / "auto_launches.jsonl"
    launched = 0
    launched_heavy_gpu_job = False
    for next_item in [row for row in rows if row["Status"] == "queued"]:
        if launched >= max_launches:
            break
        if not _is_safe_to_auto_launch(next_item, gpu):
            continue
        command = next_item.get("Command", "").strip()
        if not command:
            continue
        is_heavy_gpu_job = _is_heavy_gpu_training_command(command)
        if is_heavy_gpu_job and launched_heavy_gpu_job:
            continue
        stdout_log = log_dir / f"auto_launch_{next_item['ID']}.log"
        stdout_handle = stdout_log.open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                ["bash", "-lc", command],
                stdout=stdout_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            stdout_handle.close()
        raw_pid = getattr(process, "pid", None)
        payload = {
            "timestamp_utc": utc_now_iso(),
            "id": next_item["ID"],
            "command": command,
            "pid": raw_pid if isinstance(raw_pid, int) else str(raw_pid),
            "output_log": str(stdout_log),
            "expected_output": next_item.get("Expected Output", ""),
        }
        with auto_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        if queue_path is not None:
            update_queue_status(queue_path, next_item["ID"], "running")
        launched += 1
        launched_heavy_gpu_job = launched_heavy_gpu_job or is_heavy_gpu_job
    return launched


def _format_watts(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f}"


def write_live_scheduler_status(
    gpu: GPUState,
    rows: list[dict[str, str]],
    active_python_jobs: int,
    idle_written: bool,
    auto_launched: int | bool,
    active_jobs: list[dict[str, object]] | None = None,
    live_status_path: Path = LIVE_STATUS_PATH,
) -> None:
    next_item = select_next_queued(rows)
    lines = [
        "# Heavy Stage Live Status",
        "",
        f"- Last scheduler update UTC: {utc_now_iso()}",
        f"- GPU available: {gpu.available}",
        f"- GPU: {gpu.name}",
        f"- Memory: {gpu.memory_used_gb:.2f} / {gpu.memory_total_gb:.2f} GB",
        f"- Utilization: {gpu.utilization_gpu_pct}%",
        f"- Power draw: {_format_watts(gpu.power_draw_w)} W",
        f"- Active python jobs: {active_python_jobs}",
        f"- Idle diagnosis written: {idle_written}",
        f"- Auto-launched queued jobs: {int(auto_launched)}",
        "",
        "## Queue Summary",
        "",
    ]
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["Status"]] = status_counts.get(row["Status"], 0) + 1
    if status_counts:
        for status, count in sorted(status_counts.items()):
            lines.append(f"- {status}: {count}")
    else:
        lines.append("- No queue items found.")
    lines.extend(["", "## Next Queued Job", ""])
    if next_item is None:
        lines.append("No queued job available.")
    else:
        lines.append(f"- {next_item['ID']}: `{next_item['Command']}`")
    lines.extend(["", "## Active Python Jobs", ""])
    if active_jobs:
        lines.append("| PID | PPID | Elapsed | CPU % | MEM % | Command |")
        lines.append("| --- | --- | --- | ---: | ---: | --- |")
        for job in active_jobs[:20]:
            command = str(job.get("command", "")).replace("|", "\\|")
            if len(command) > 140:
                command = command[:137] + "..."
            lines.append(
                f"| {job.get('pid')} | {job.get('ppid')} | {job.get('elapsed')} | "
                f"{float(job.get('cpu_pct', 0.0)):.1f} | {float(job.get('mem_pct', 0.0)):.1f} | `{command}` |"
            )
    else:
        lines.append("No active python jobs detected.")
    live_status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scheduler_state(
    gpu: GPUState,
    rows: list[dict[str, str]],
    active_python_jobs: int,
    idle_written: bool,
    auto_launched: int | bool,
) -> None:
    payload: dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "gpu": {
            "available": gpu.available,
            "name": gpu.name,
            "memory_used_gb": gpu.memory_used_gb,
            "memory_total_gb": gpu.memory_total_gb,
            "utilization_gpu_pct": gpu.utilization_gpu_pct,
            "power_draw_w": gpu.power_draw_w,
        },
        "active_python_jobs": active_python_jobs,
        "idle_diagnosis_written": idle_written,
        "auto_launched": int(auto_launched),
        "queue_items": rows,
    }
    SCHEDULER_STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_scheduler(queue_path: Path = QUEUE_PATH, *, launch_when_idle: bool = False, max_launches: int = 1) -> None:
    reconcile_transfer_job_status(queue_path, Path("outputs/transfer/eeg_imagenet_pretrain_t2t_align"))
    queue = load_queue(queue_path)
    rows = build_board_rows(queue)
    sync_inferred_queue_statuses(queue_path, queue, rows)
    gpu = query_gpu_state()
    active_jobs = query_active_python_jobs()
    active_python_jobs = len(active_jobs)
    write_board(rows)
    idle_written = write_idle_diagnosis(gpu, rows, active_python_jobs)
    auto_launched = launch_next_queued_job(
        rows,
        launch_enabled=launch_when_idle and idle_written,
        queue_path=queue_path,
        max_launches=max_launches,
        gpu=gpu,
    )
    write_live_scheduler_status(gpu, rows, active_python_jobs, idle_written, auto_launched, active_jobs=active_jobs)
    write_scheduler_state(gpu, rows, active_python_jobs, idle_written, auto_launched)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit heavy-stage queue state without launching jobs.")
    parser.add_argument("--queue", type=Path, default=QUEUE_PATH, help="Queue YAML path.")
    parser.add_argument(
        "--launch-when-idle",
        action="store_true",
        help="If idle diagnosis triggers, launch the next queued command in the background.",
    )
    parser.add_argument(
        "--max-launches",
        type=int,
        default=1,
        help="Maximum queued jobs to launch during one idle scheduler tick.",
    )
    args = parser.parse_args()
    run_scheduler(args.queue, launch_when_idle=args.launch_when_idle, max_launches=args.max_launches)


if __name__ == "__main__":
    main()
