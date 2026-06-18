from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_IMAGE_ROOT = Path("/cloud/cloud-ssd1/eeg_vision_caption_data/ImageNet/kaggle_cls_loc/extracted")
DEFAULT_RELATIVE_TO = Path("/workspace")
DEFAULT_REPORT_DIR = Path("outputs/datasets")
DEFAULT_LOG = Path("outputs/datasets/EEG_IMAGENET_IMAGE_LINK_WATCHER.log")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _marker_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "complete"


def _has_any_image(image_root: Path) -> bool:
    return any(image_root.rglob("*.JPEG")) or any(image_root.rglob("*.jpg")) or any(image_root.rglob("*.jpeg"))


def _has_kaggle_cls_loc_layout(image_root: Path) -> bool:
    cls_root = image_root / "ILSVRC" / "Data" / "CLS-LOC"
    train_root = cls_root / "train"
    val_root = cls_root / "val"
    if train_root.exists() and val_root.exists():
        return any(train_root.glob("*/*.JPEG")) and any(val_root.glob("*.JPEG"))
    return False


def image_root_ready(image_root: Path) -> bool:
    if not image_root.exists():
        return False
    # A partially extracted ImageNet tree can expose some JPEGs before all class
    # archives finish. Require postprocess completion markers before linking.
    return (
        _marker_complete(image_root / ".extract_complete.json")
        and _marker_complete(image_root / ".nested_extract_complete.json")
        and _has_any_image(image_root)
        and _has_kaggle_cls_loc_layout(image_root)
    )


def build_link_commands(
    *,
    image_root: Path,
    relative_to: Path,
    report_dir: Path,
) -> list[list[str]]:
    commands: list[list[str]] = []
    for split in ("train", "val", "test"):
        commands.append(
            [
                "python",
                "scripts/link_eeg_imagenet_images.py",
                "--manifest",
                f"data/EEG-ImageNet/{split}.jsonl",
                "--image-root",
                str(image_root),
                "--out",
                f"data/EEG-ImageNet/{split}_image_linked.jsonl",
                "--filtered-out",
                f"data/EEG-ImageNet/{split}_image_exact.jsonl",
                "--report",
                str(report_dir / f"EEG_IMAGENET_IMAGE_LINK_{split.upper()}_REPORT.md"),
                "--relative-to",
                str(relative_to),
            ]
        )
    return commands


def append_log(log_path: Path, payload: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def wait_and_link(
    *,
    image_root: Path,
    relative_to: Path,
    report_dir: Path,
    log_path: Path,
    poll_seconds: int,
    max_wait_seconds: int,
    run: bool,
) -> dict[str, object]:
    start = time.time()
    while True:
        if image_root_ready(image_root):
            break
        if max_wait_seconds >= 0 and time.time() - start >= max_wait_seconds:
            payload = {"timestamp_utc": utc_now(), "status": "waiting_timed_out", "image_root": str(image_root)}
            append_log(log_path, payload)
            return payload
        append_log(log_path, {"timestamp_utc": utc_now(), "status": "waiting", "image_root": str(image_root)})
        time.sleep(max(1, poll_seconds))

    commands = build_link_commands(image_root=image_root, relative_to=relative_to, report_dir=report_dir)
    append_log(log_path, {"timestamp_utc": utc_now(), "status": "image_root_ready", "commands": commands})
    if not run:
        return {"timestamp_utc": utc_now(), "status": "ready_dry_run", "commands": commands}

    results: list[dict[str, object]] = []
    for command in commands:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        results.append({"command": command, "returncode": result.returncode, "output_tail": result.stdout[-4000:]})
        append_log(log_path, {"timestamp_utc": utc_now(), "status": "command_finished", "result": results[-1]})
        if result.returncode != 0:
            return {"timestamp_utc": utc_now(), "status": "failed", "results": results}
    return {"timestamp_utc": utc_now(), "status": "completed", "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for ImageNet extraction, then link EEG-ImageNet manifests to real JPEG files.")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--relative-to", type=Path, default=DEFAULT_RELATIVE_TO)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-wait-seconds", type=int, default=-1)
    parser.add_argument("--run", action="store_true", help="Actually run link commands when ImageNet JPEGs are detected.")
    args = parser.parse_args()
    payload = wait_and_link(
        image_root=args.image_root,
        relative_to=args.relative_to,
        report_dir=args.report_dir,
        log_path=args.log,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
        run=args.run,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
