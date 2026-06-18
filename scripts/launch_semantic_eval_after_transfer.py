from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_CHECKPOINT = Path("outputs/transfer/eeg_imagenet_pretrain_t2t_align/checkpoints/best.pt")
DEFAULT_READY_MARKER = Path("outputs/transfer/eeg_imagenet_pretrain_t2t_align/alignment_metrics.json")
DEFAULT_OUTPUT_DIR = Path("outputs/final_semantic/eeg_imagenet_transfer_eval")
DEFAULT_LOG = Path("outputs/final_semantic/eeg_imagenet_transfer_eval/launcher.log")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def wait_for_checkpoint(checkpoint: Path, *, poll_seconds: int, max_wait_seconds: int) -> bool:
    start = time.time()
    while True:
        if checkpoint.exists():
            return True
        if max_wait_seconds >= 0 and time.time() - start >= max_wait_seconds:
            return False
        time.sleep(max(0, poll_seconds))


def build_semantic_eval_command(
    *,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[str]:
    return [
        "bash",
        "-lc",
        " && ".join(
            [
                "python scripts/build_text_prototypes.py "
                "--data_root data/thought2text "
                "--output outputs/semantic_caption/prototypes.pt "
                "--report outputs/semantic_caption/prototype_bank.md "
                "--splits train val "
                "--clip_prefix clip",
                "python -m src.eval.constrained_caption_eval "
                "--prototype_bank outputs/semantic_caption/prototypes.pt "
                "--manifest data/thought2text/test_human_caption.jsonl "
                "--cache_dir data/thought2text/cache "
                f"--output_dir {output_dir} "
                "--corruptions clean blur occlusion noise lowres "
                "--modes vision_only real_eeg shuffled_eeg random_eeg eeg_only "
                f"--eeg_checkpoint {checkpoint} "
                "--batch_size 64 "
                "--device auto",
                f"python scripts/make_semantic_caption_report.py --metrics {output_dir / 'FULL_METRICS.csv'} --output_dir {output_dir}",
                f"python scripts/make_robustness_report.py --semantic_dir {output_dir} --out_dir outputs/robustness/eeg_imagenet_transfer_eval",
                f"python scripts/materialize_final_semantic_report.py --source-dir {output_dir} --robustness-dir outputs/robustness/eeg_imagenet_transfer_eval --out-dir outputs/final_semantic --primary-summary outputs/final_semantic/A2_SEMANTIC_FUSION_MULTISEED_SUMMARY.md",
            ]
        ),
    ]


def launch_eval(
    *,
    checkpoint: Path,
    ready_marker: Path,
    output_dir: Path,
    log_path: Path,
    poll_seconds: int,
    max_wait_seconds: int,
    background: bool,
) -> dict[str, object]:
    ready = wait_for_checkpoint(ready_marker, poll_seconds=poll_seconds, max_wait_seconds=max_wait_seconds)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not ready:
        payload = {
            "timestamp_utc": utc_now(),
            "status": "waiting_timed_out",
            "checkpoint": str(checkpoint),
            "ready_marker": str(ready_marker),
            "output_dir": str(output_dir),
        }
        log_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    if not checkpoint.exists():
        payload = {
            "timestamp_utc": utc_now(),
            "status": "checkpoint_missing_after_ready_marker",
            "checkpoint": str(checkpoint),
            "ready_marker": str(ready_marker),
            "output_dir": str(output_dir),
        }
        log_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    command = build_semantic_eval_command(checkpoint=checkpoint, output_dir=output_dir)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp_utc": utc_now(),
                    "status": "launching",
                    "checkpoint": str(checkpoint),
                    "ready_marker": str(ready_marker),
                    "command": command,
                }
            )
            + "\n"
        )
        if background:
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            payload = {
                "timestamp_utc": utc_now(),
                "status": "launched",
                "pid": process.pid,
                "checkpoint": str(checkpoint),
                "ready_marker": str(ready_marker),
                "output_dir": str(output_dir),
                "log": str(log_path),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            return payload
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return {
        "timestamp_utc": utc_now(),
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "checkpoint": str(checkpoint),
        "ready_marker": str(ready_marker),
        "output_dir": str(output_dir),
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch full constrained semantic eval after a transfer checkpoint appears.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--ready-marker",
        type=Path,
        default=DEFAULT_READY_MARKER,
        help="File that indicates the transfer run has finished. Defaults to alignment_metrics.json, not best.pt.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-wait-seconds", type=int, default=-1, help="Use -1 to wait indefinitely.")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    payload = launch_eval(
        checkpoint=args.checkpoint,
        ready_marker=args.ready_marker,
        output_dir=args.output_dir,
        log_path=args.log,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
        background=args.background,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
