from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GPUStatus:
    index: int = 0
    name: str = "unknown"
    memory_used_mb: int = 0
    memory_total_mb: int = 0
    utilization_gpu: int = 0

    @property
    def memory_used_gb(self) -> float:
        return self.memory_used_mb / 1024.0


def query_gpu_status() -> GPUStatus | None:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    first = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not first:
        return None
    row = next(csv.reader([first], skipinitialspace=True))
    return GPUStatus(
        index=int(row[0]),
        name=row[1],
        memory_used_mb=int(float(row[2])),
        memory_total_mb=int(float(row[3])),
        utilization_gpu=int(float(row[4])),
    )


def _count_top_level_alignment_processes(rows: list[tuple[int, int, str]]) -> int:
    train_pids = {pid for pid, _ppid, args in rows if "python" in args and "src.train.train_align" in args}
    child_pids = {pid for pid, ppid, args in rows if ppid in train_pids and "python" in args and "src.train.train_align" in args}
    return len(train_pids - child_pids)


def count_running_alignment_processes() -> int:
    try:
        result = subprocess.run(["ps", "-eo", "pid,ppid,args"], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return 0
    rows: list[tuple[int, int, str]] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
        except ValueError:
            continue
    return _count_top_level_alignment_processes(rows)


def allowed_concurrency(memory_used_gb: float, gpu_util: int, max_concurrent: int = 8) -> int:
    max_concurrent = max(1, min(int(max_concurrent), 8))
    if memory_used_gb < 8.0 and gpu_util < 30:
        return max_concurrent
    if memory_used_gb < 12.0 and gpu_util < 40:
        return min(max_concurrent, 8)
    if memory_used_gb < 20.0 and gpu_util < 60:
        return min(max_concurrent, 6)
    if memory_used_gb < 32.0 and gpu_util < 75:
        return min(max_concurrent, 4)
    return 1


def queue_snapshot(max_concurrent: int = 8) -> dict[str, Any]:
    gpu = query_gpu_status()
    running = count_running_alignment_processes()
    if gpu is None:
        allowed = 1
        gpu_payload: dict[str, Any] = {"available": False}
    else:
        allowed = allowed_concurrency(gpu.memory_used_gb, gpu.utilization_gpu, max_concurrent=max_concurrent)
        gpu_payload = {
            "available": True,
            "index": gpu.index,
            "name": gpu.name,
            "memory_used_mb": gpu.memory_used_mb,
            "memory_total_mb": gpu.memory_total_mb,
            "memory_used_gb": gpu.memory_used_gb,
            "utilization_gpu": gpu.utilization_gpu,
        }
    return {
        "gpu": gpu_payload,
        "running_alignment_processes": running,
        "allowed_concurrency": allowed,
        "available_slots": max(0, allowed - running),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GPU alignment queue policy.")
    parser.add_argument("--max_concurrent", type=int, default=8)
    args = parser.parse_args()
    import json

    print(json.dumps(queue_snapshot(args.max_concurrent), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
