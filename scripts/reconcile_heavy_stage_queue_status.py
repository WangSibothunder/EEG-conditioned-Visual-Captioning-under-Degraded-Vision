from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.update_heavy_stage_queue_status import update_queue_status


DEFAULT_JOB_ID = "TRANSFER_EEG_IMAGENET_PRETRAIN_TO_THOUGHT2TEXT"


def reconcile_transfer_job_status(
    queue_path: str | Path,
    transfer_dir: str | Path,
    *,
    job_id: str = DEFAULT_JOB_ID,
) -> bool:
    queue = Path(queue_path)
    out_dir = Path(transfer_dir)
    checkpoint = out_dir / "checkpoints" / "best.pt"
    metrics = out_dir / "retrieval_metrics.json"
    if checkpoint.exists() and metrics.exists():
        return update_queue_status(queue, job_id, "completed")

    launcher_log = out_dir / "transfer_launcher.log"
    if launcher_log.exists() and "pretrain_finished_at=" in launcher_log.read_text(encoding="utf-8", errors="replace"):
        return update_queue_status(queue, job_id, "running")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile heavy-stage queue status from transfer job artifacts.")
    parser.add_argument("--queue", default="configs/heavy_stage_queue.yaml")
    parser.add_argument("--transfer-dir", default="outputs/transfer/eeg_imagenet_pretrain_t2t_align")
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    args = parser.parse_args()
    changed = reconcile_transfer_job_status(args.queue, args.transfer_dir, job_id=args.job_id)
    print(f"changed={changed}")


if __name__ == "__main__":
    main()
