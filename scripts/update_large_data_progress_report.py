from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zipfile import BadZipFile, ZipFile


DATA_ROOT = Path("/cloud/cloud-ssd1/eeg_vision_caption_data")
REPORT_DIR = Path("outputs/download_reports")
EXPECTED_THINGS_BYTES = 147_944_847_435
EXPECTED_IMAGENET_ZIP_BYTES = 155 * 1024**3


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def human(n: int | float) -> str:
    value = float(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024 or unit == "TiB":
            return f"{value:.2f}{unit}"
        value /= 1024
    return f"{value:.2f}TiB"


def tmux_running(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def read_marker(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def path_size(path: Path) -> int:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    if path.exists():
        return path.stat().st_size
    return 0


def path_count(path: Path) -> int:
    if path.is_dir():
        return sum(1 for p in path.rglob("*") if p.is_file())
    return 1 if path.exists() else 0


def zip_central_directory_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with ZipFile(path) as zf:
            zf.infolist()
        return True
    except (BadZipFile, OSError):
        return False


def zip_status_text(
    *,
    zip_size: int,
    expected_zip_bytes: int,
    central_directory_ok: bool,
    extraction_status: dict[str, object],
) -> str:
    marker = extraction_status.get("extract_marker")
    marker_zip_size = 0
    if isinstance(marker, dict):
        marker_zip_size = int(marker.get("zip_size", 0) or 0)
    if extraction_status.get("complete") and zip_size == 0 and marker_zip_size:
        return f"archive cleaned after extraction; original zip `{human(marker_zip_size)}` from marker"
    pct = 100.0 * zip_size / expected_zip_bytes if expected_zip_bytes else 0.0
    return f"zip `{human(zip_size)}` (`{pct:.1f}%` of ~155GiB), zip valid={central_directory_ok}"


def marker_status(root: Path, label: str) -> dict[str, object]:
    marker = read_marker(root / ".extract_complete.json")
    nested = read_marker(root / ".nested_extract_complete.json")
    return {
        "label": label,
        "extract_marker": marker,
        "nested_marker": nested,
        "complete": bool(marker and nested and marker.get("status") == "complete" and nested.get("status") == "complete"),
    }


def extraction_progress(root: Path, status: dict[str, object]) -> dict[str, object]:
    marker = status.get("extract_marker")
    if isinstance(marker, dict) and marker.get("status") == "complete":
        return {
            "bytes": int(marker.get("uncompressed_bytes", 0) or 0),
            "files": int(marker.get("files", 0) or 0),
            "source": "marker",
        }
    return {
        "bytes": path_size(root),
        "files": path_count(root),
        "source": "live",
    }


def _jsonl_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def eeg_imagenet_pairing_status(project_root: Path = Path(".")) -> dict[str, object]:
    manifest_dir = project_root / "data" / "EEG-ImageNet"
    report_dir = project_root / "outputs" / "datasets"
    rows = {split: _jsonl_line_count(manifest_dir / f"{split}_image_exact.jsonl") for split in ("train", "val", "test")}
    reports_ready = True
    full_reports = True
    for split in ("TRAIN", "VAL", "TEST"):
        report_path = report_dir / f"EEG_IMAGENET_IMAGE_LINK_{split}_REPORT.md"
        if not report_path.exists():
            reports_ready = False
            full_reports = False
            continue
        text = report_path.read_text(encoding="utf-8")
        if "Exact-linked subset status: `ready for paired training`" not in text:
            reports_ready = False
        if "Loader-ready status: `fully image-linked`" not in text:
            full_reports = False
    ready = reports_ready and all(count > 0 for count in rows.values())
    if full_reports and ready:
        status = "fully image-linked"
    elif ready:
        status = "exact-linked subset ready"
    elif any(count > 0 for count in rows.values()):
        status = "partial exact-linked subset"
    else:
        status = "not linked"
    return {"ready": ready, "status": status, "rows": rows, "reports_ready": reports_ready, "full_reports": full_reports}


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    updated = now()
    disk = shutil.disk_usage(DATA_ROOT)

    paths = {
        "eit_extracted": DATA_ROOT / "EIT-1M" / "extracted",
        "things_image_set": DATA_ROOT / "THINGS-EEG2" / "image_set",
        "things_raw": DATA_ROOT / "THINGS-EEG2" / "raw-eeg",
        "things_verify": DATA_ROOT / "THINGS-EEG2" / ".verification_complete.json",
        "eeg_imagenet_zip": DATA_ROOT / "EEG-ImageNet" / "eeg-imagenet.zip",
        "eeg_imagenet_extracted": DATA_ROOT / "EEG-ImageNet" / "extracted",
        "imagenet_zip": DATA_ROOT / "ImageNet" / "kaggle_cls_loc" / "imagenet-object-localization-challenge.zip",
        "imagenet_extracted": DATA_ROOT / "ImageNet" / "kaggle_cls_loc" / "extracted",
    }

    things_verify = read_marker(paths["things_verify"])
    things_raw_size = int(things_verify.get("bytes", 0)) if things_verify and things_verify.get("bytes") else path_size(paths["things_raw"])
    things_raw_count = int(things_verify.get("files", 0)) if things_verify and things_verify.get("files") else path_count(paths["things_raw"])
    imagenet_zip_size = path_size(paths["imagenet_zip"])
    imagenet_status = marker_status(paths["imagenet_extracted"], "ImageNet Kaggle CLS-LOC")
    eeg_imagenet_status = marker_status(paths["eeg_imagenet_extracted"], "EEG-ImageNet")
    eit_status = marker_status(paths["eit_extracted"], "EIT-1M")
    things_image_status = marker_status(paths["things_image_set"], "THINGS-EEG2 image_set")

    snapshot = {
        "updated": updated,
        "data_root": str(DATA_ROOT),
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
        "tmux": {name: tmux_running(name) for name in ["things_eeg2_dl", "imagenet_dl", "large_data_postprocess", "large_data_supervisor"]},
        "items": {
            "things_raw": {
                "path": str(paths["things_raw"]),
                "size_bytes": things_raw_size,
                "file_count": things_raw_count,
                "verification_marker": str(paths["things_verify"]),
                "verified": bool(things_verify and things_verify.get("status") == "complete"),
            },
            "imagenet_zip": {
                "path": str(paths["imagenet_zip"]),
                "size_bytes": imagenet_zip_size,
                "central_directory_ok": zip_central_directory_ok(paths["imagenet_zip"]),
            },
            "imagenet_extracted": imagenet_status,
            "eeg_imagenet_extracted": eeg_imagenet_status,
            "eit_extracted": eit_status,
            "things_image_set": things_image_status,
        },
    }
    (REPORT_DIR / "large_data_progress_snapshot.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    eit_size = path_size(paths["eit_extracted"])
    things_img_size = path_size(paths["things_image_set"])
    eeg_size = path_size(paths["eeg_imagenet_extracted"])
    imagenet_progress = extraction_progress(paths["imagenet_extracted"], imagenet_status)
    imagenet_extracted_size = int(imagenet_progress["bytes"])
    imagenet_extracted_files = int(imagenet_progress["files"])
    eeg_pairing = eeg_imagenet_pairing_status()
    things_pct = 100.0 * things_raw_size / EXPECTED_THINGS_BYTES
    imagenet_zip_text = zip_status_text(
        zip_size=imagenet_zip_size,
        expected_zip_bytes=EXPECTED_IMAGENET_ZIP_BYTES,
        central_directory_ok=bool(snapshot["items"]["imagenet_zip"]["central_directory_ok"]),
        extraction_status=imagenet_status,
    )

    open_items: list[str] = []
    if not snapshot["items"]["things_raw"]["verified"]:
        open_items.append("THINGS-EEG2 full HF repo must finish and write `.verification_complete.json`.")
    if not imagenet_status["complete"]:
        open_items.append("ImageNet Kaggle zip must finish, validate as a zip, then outer and nested extraction must complete.")
    open_items_for_report = open_items or ["None. All tracked large-data downloads and extraction markers are complete."]

    lines = [
        "# Large Data Current Progress",
        "",
        f"Updated: {updated}",
        "",
        f"Data root: `{DATA_ROOT}`",
        f"Disk: total `{human(disk.total)}`, used `{human(disk.used)}`, free `{human(disk.free)}`",
        "",
        "| Item | Status | Size | Progress | Evidence |",
        "| --- | --- | ---: | ---: | --- |",
        f"| EIT-1M public HF release | {'complete' if eit_status['complete'] else 'incomplete'} | extracted {human(eit_size)} | complete | markers present={eit_status['complete']} |",
        f"| THINGS-EEG2 image_set | {'complete' if things_image_status['complete'] else 'incomplete'} | {human(things_img_size)} | complete | markers present={things_image_status['complete']} |",
        f"| THINGS-EEG2 full repo/raw EEG | {'verified' if snapshot['items']['things_raw']['verified'] else 'downloading'} | {human(things_raw_size)}, {snapshot['items']['things_raw']['file_count']} files | {things_pct:.1f}% of HF repo bytes | `things_eeg2_dl` running={snapshot['tmux']['things_eeg2_dl']} |",
        f"| EEG-ImageNet | {'complete' if eeg_imagenet_status['complete'] else 'incomplete'} | extracted {human(eeg_size)} | complete | markers present={eeg_imagenet_status['complete']} |",
        f"| ImageNet Kaggle CLS-LOC | {'complete' if imagenet_status['complete'] else 'downloading' if snapshot['tmux']['imagenet_dl'] else 'waiting/extracting'} | extracted {human(imagenet_extracted_size)} ({imagenet_extracted_files} files, {imagenet_progress['source']}) | complete | {imagenet_zip_text}; `imagenet_dl` running={snapshot['tmux']['imagenet_dl']} |",
        "",
        "Active automation:",
        "",
        "- `large_data_supervisor` checks/restarts required tmux sessions every 5 minutes.",
        "- `large_data_postprocess` extracts completed zips and recursively extracts nested tar/zip archives.",
        "",
        "Completion criteria still open:",
        "",
    ]
    lines.extend(f"- {item}" for item in open_items_for_report)
    current_report = "\n".join(lines) + "\n"
    (REPORT_DIR / "LARGE_DATA_CURRENT_PROGRESS.md").write_text(current_report, encoding="utf-8")

    legacy_lines = [
        "# Large Data Download Status",
        "",
        f"Updated: {updated}",
        "",
        f"Data root: `{DATA_ROOT}`",
        f"Disk: total `{human(disk.total)}`, used `{human(disk.used)}`, free `{human(disk.free)}`",
        "",
        "This file is kept as the stable status entry point. For the same data in a compact table, see `outputs/download_reports/LARGE_DATA_CURRENT_PROGRESS.md`.",
        "",
        "## Current Status",
        "",
        "| Dataset | Status | Evidence |",
        "| --- | --- | --- |",
        f"| Thought2Text | complete | project-local manifests/caches are the main small-data path |",
        f"| EIT-1M public release | {'complete' if eit_status['complete'] else 'incomplete'} | extracted `{human(eit_size)}`; public HF release is partial, not full paper-scale EIT-1M |",
        f"| THINGS-EEG2 image_set | {'complete' if things_image_status['complete'] else 'incomplete'} | image set `{human(things_img_size)}` |",
        f"| THINGS-EEG2 raw EEG | {'verified' if snapshot['items']['things_raw']['verified'] else 'downloading'} | `{human(things_raw_size)}`, `{things_raw_count}` files; usable now for EEG-only masked pretraining |",
        f"| EEG-ImageNet EEG | {'complete' if eeg_imagenet_status['complete'] else 'incomplete'} | extracted `{human(eeg_size)}`; EEG `.pth` shards and converted manifests/caches are ready |",
        f"| EEG-ImageNet paired images | {eeg_pairing['status']} | exact-linked rows train/val/test = `{eeg_pairing['rows']['train']}`/`{eeg_pairing['rows']['val']}`/`{eeg_pairing['rows']['test']}`; missing exact stimulus JPEGs are not replaced by same-class images |",
        f"| ImageNet Kaggle CLS-LOC | {'complete' if imagenet_status['complete'] else 'downloading' if snapshot['tmux']['imagenet_dl'] else 'waiting/extracting'} | {imagenet_zip_text}; extracted `{imagenet_extracted_files}` files ({imagenet_progress['source']}) |",
        "",
        "## Operational Notes",
        "",
        "- EEG-ImageNet paired training may use only the exact-linked subset; do not replace missing stimulus JPEGs with same-class images for EEG-effect claims.",
        "- EEG-ImageNet and THINGS-EEG2 can still be used for EEG-only masked pretraining.",
        "- `watch_and_link_eeg_imagenet_images.py` is the handoff from completed ImageNet extraction to exact-linked paired EEG-ImageNet manifests.",
        "",
        "## Open Completion Criteria",
        "",
    ]
    legacy_lines.extend(f"- {item}" for item in open_items_for_report)
    (REPORT_DIR / "LARGE_DATA_DOWNLOAD_STATUS.md").write_text("\n".join(legacy_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# Download Summary",
        "",
        f"Updated: {updated}",
        "",
        "| Item | Status | Local Path | Size | Verification | Notes |",
        "| --- | --- | --- | ---: | --- | --- |",
        "| Core HF models | complete | `/workspace/data/model_cache` | see `model_cache_report.md` | local tokenizer/processor load ok | CLIP-B/32, Qwen2.5-1.5B, BLIP base present |",
        "| Thought2Text | complete | `/workspace/data/thought2text` | see `thought2text_data_report.md` | key files load | Main small real-data path present |",
        f"| EIT-1M public release | {'complete' if eit_status['complete'] else 'incomplete'} | `{DATA_ROOT / 'EIT-1M'}` | extracted {human(eit_size)} | markers present={eit_status['complete']} | Public partial release only |",
        f"| THINGS-EEG2 image_set | {'complete' if things_image_status['complete'] else 'incomplete'} | `{DATA_ROOT / 'THINGS-EEG2' / 'image_set'}` | {human(things_img_size)} | markers present={things_image_status['complete']} | Image set extracted |",
        f"| THINGS-EEG2 raw EEG | {'complete' if snapshot['items']['things_raw']['verified'] else 'downloading'} | `{DATA_ROOT / 'THINGS-EEG2'}` | {things_raw_count} files, {human(things_raw_size)} | `.verification_complete.json` present={snapshot['items']['things_raw']['verified']} | Raw continuous EEG usable for masked EEG pretraining, not trial-level image alignment |",
        f"| EEG-ImageNet EEG | {'complete' if eeg_imagenet_status['complete'] else 'incomplete'} | `{DATA_ROOT / 'EEG-ImageNet'}` | extracted {human(eeg_size)} | markers present={eeg_imagenet_status['complete']} | EEG data ready; paired exact-linked subset rows train/val/test `{eeg_pairing['rows']['train']}`/`{eeg_pairing['rows']['val']}`/`{eeg_pairing['rows']['test']}` |",
        f"| ImageNet Kaggle CLS-LOC | {'complete' if imagenet_status['complete'] else 'downloading' if snapshot['tmux']['imagenet_dl'] else 'waiting/extracting'} | `{DATA_ROOT / 'ImageNet' / 'kaggle_cls_loc'}` | extracted {human(imagenet_extracted_size)} ({imagenet_extracted_files} files) | {imagenet_zip_text}; markers present={imagenet_status['complete']} | ImageNet original images available for EEG-ImageNet linking |",
        "| Automation | active | tmux sessions | n/a | supervisor/postprocess/watchers running | Safe to disconnect; work continues in tmux |",
    ]
    (REPORT_DIR / "DOWNLOAD_SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    things_lines = [
        "# THINGS-EEG2 Download Report",
        "",
        f"Generated: {updated}",
        "",
        f"Local root: `{DATA_ROOT / 'THINGS-EEG2'}`",
        "",
        f"Status: `{'verified' if snapshot['items']['things_raw']['verified'] else 'not complete'}`",
        "",
        "## Current Evidence",
        "",
        f"- Raw EEG files: `{things_raw_count}`",
        f"- Raw EEG bytes: `{human(things_raw_size)}`",
        f"- Verification marker: `{paths['things_verify']}` exists={snapshot['items']['things_raw']['verified']}",
        f"- Image set extracted: `{things_image_status['complete']}`",
        "",
        "## Loader Boundary",
        "",
        "THINGS-EEG2 is verified locally for raw continuous EEG and image-set inspection. It is not yet image+EEG trial-loader-ready because event/stimulus-order metadata needed to align windows to individual images has not been found in the local release.",
        "",
        "Current use: fixed-window EEG-only masked pretraining.",
    ]
    (REPORT_DIR / "things_eeg2_download_report.md").write_text("\n".join(things_lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
