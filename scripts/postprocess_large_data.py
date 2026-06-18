from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from zipfile import BadZipFile, ZipFile


DATA_ROOT = Path(os.environ.get("EEG_CAPTION_DATA_ROOT", "/cloud/cloud-ssd1/eeg_vision_caption_data"))
REPORT_DIR = Path("outputs/download_reports")
STATUS_JSON = REPORT_DIR / "large_data_postprocess_status.json"
STATUS_MD = REPORT_DIR / "LARGE_DATA_POSTPROCESS_STATUS.md"
THINGS_VERIFY_MARKER = DATA_ROOT / "THINGS-EEG2" / ".verification_complete.json"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tmux_exists(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def disk_free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def human(n: int | float) -> str:
    n = float(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if n < 1024 or unit == units[-1]:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def write_status(records: dict[str, dict[str, object]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated": now(),
        "data_root": str(DATA_ROOT),
        "free_bytes": disk_free_bytes(DATA_ROOT),
        "records": records,
    }
    STATUS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    lines = [
        "# Large Data Postprocess Status",
        "",
        f"Updated: {payload['updated']}",
        "",
        f"Data root: `{DATA_ROOT}`",
        f"Free space: `{human(payload['free_bytes'])}`",
        "",
        "| Item | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for item, record in records.items():
        detail = str(record.get("detail", "")).replace("\n", " ")
        lines.append(f"| {item} | {record.get('status', 'unknown')} | {detail} |")
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def marker_path(dest: Path) -> Path:
    return dest / ".extract_complete.json"


def nested_marker_path(dest: Path) -> Path:
    return dest / ".nested_extract_complete.json"


def marker_matches(dest: Path, zip_path: Path) -> bool:
    marker = marker_path(dest)
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        payload.get("zip_path") == str(zip_path)
        and payload.get("zip_size") == zip_path.stat().st_size
        and payload.get("status") == "complete"
    )


def nested_marker_complete(dest: Path) -> bool:
    marker = nested_marker_path(dest)
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "complete"


def extraction_fully_complete(dest: Path) -> bool:
    marker = marker_path(dest)
    if not marker.exists() or not nested_marker_complete(dest):
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "complete"


def safe_target(dest: Path, member_name: str) -> Path:
    target = (dest / member_name).resolve()
    root = dest.resolve()
    if target != root and root not in target.parents:
        raise RuntimeError(f"Unsafe zip member path: {member_name}")
    return target


def is_supported_archive(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".zip")
        or name.endswith(".tar")
        or name.endswith(".tar.gz")
        or name.endswith(".tgz")
    )


def nested_extract_dest(archive_path: Path) -> Path:
    name = archive_path.name
    lower = name.lower()
    if lower.endswith(".tar.gz"):
        stem = name[:-7]
    elif lower.endswith(".tgz"):
        stem = name[:-4]
    elif lower.endswith(".zip") or lower.endswith(".tar"):
        stem = archive_path.stem
    else:
        stem = archive_path.stem
    return archive_path.with_name(stem)


def nested_archive_is_complete(archive_path: Path, dest: Path) -> bool:
    marker = marker_path(dest)
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        payload.get("status") == "complete"
        and payload.get("archive_path") == str(archive_path)
        and payload.get("archive_size") == archive_path.stat().st_size
    )


def safe_tar_target(dest: Path, member_name: str) -> Path:
    # tar archives may contain absolute paths or ../ entries; reject those.
    return safe_target(dest, member_name)


def extract_tar(archive_path: Path, dest: Path, label: str) -> dict[str, object]:
    if not archive_path.exists():
        return {"status": "waiting", "detail": f"missing {archive_path}"}
    if nested_archive_is_complete(archive_path, dest):
        return {"status": "complete", "detail": f"already extracted to {dest}"}

    dest.mkdir(parents=True, exist_ok=True)
    start = time.time()
    extracted_files = 0
    extracted_bytes = 0
    last_report = time.time()
    try:
        with tarfile.open(archive_path) as tf:
            members = tf.getmembers()
            total_members = len(members)
            total_bytes = sum(member.size for member in members if member.isfile())
            for index, member in enumerate(members, start=1):
                target = safe_tar_target(dest, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.stat().st_size == member.size:
                    extracted_files += 1
                    extracted_bytes += member.size
                    continue
                source = tf.extractfile(member)
                if source is None:
                    continue
                with source, target.open("wb") as dst:
                    shutil.copyfileobj(source, dst, length=16 * 1024 * 1024)
                extracted_files += 1
                extracted_bytes += member.size
                if time.time() - last_report > 60:
                    print(
                        json.dumps(
                            {
                                "event": "nested_extract_progress",
                                "label": label,
                                "archive": str(archive_path),
                                "index": index,
                                "members": total_members,
                                "bytes": extracted_bytes,
                                "total_bytes": total_bytes,
                            },
                            ensure_ascii=True,
                        ),
                        flush=True,
                    )
                    last_report = time.time()

        payload = {
            "status": "complete",
            "label": label,
            "archive_path": str(archive_path),
            "archive_size": archive_path.stat().st_size,
            "dest": str(dest),
            "files": extracted_files,
            "uncompressed_bytes": extracted_bytes,
            "seconds": round(time.time() - start, 3),
            "completed_at": now(),
        }
        marker_path(dest).write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return {"status": "complete", "detail": f"{extracted_files} files, {human(extracted_bytes)} extracted to {dest}"}
    except tarfile.TarError as exc:
        return {"status": "error", "detail": f"bad tar {archive_path}: {exc}"}
    except Exception as exc:
        return {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}


def extract_archive(archive_path: Path, dest: Path, label: str) -> dict[str, object]:
    lower = archive_path.name.lower()
    if lower.endswith(".zip"):
        return extract_zip(archive_path, dest, label)
    if lower.endswith(".tar") or lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        return extract_tar(archive_path, dest, label)
    return {"status": "skipped", "detail": f"unsupported archive {archive_path}"}


def extract_nested_archives(root: Path, label: str, *, delete_after_extract: bool = True) -> dict[str, object]:
    if not root.exists():
        return {"status": "waiting", "detail": f"missing {root}"}
    if nested_marker_complete(root):
        return {"status": "complete", "detail": f"nested archives already processed under {root}"}

    processed = 0
    deleted = 0
    errors: list[str] = []
    passes = 0
    while True:
        archives = sorted(path for path in root.rglob("*") if path.is_file() and is_supported_archive(path))
        if not archives:
            break
        passes += 1
        if passes > 10000:
            return {"status": "error", "detail": f"too many nested extraction passes under {root}"}

        progress_this_pass = 0
        for archive_path in archives:
            dest = nested_extract_dest(archive_path)
            result = extract_archive(archive_path, dest, f"{label} nested {archive_path.name}")
            if result.get("status") != "complete":
                errors.append(f"{archive_path}: {result.get('detail')}")
                continue
            processed += 1
            progress_this_pass += 1
            if delete_after_extract and archive_path.exists():
                archive_path.unlink()
                deleted += 1
        if errors:
            break
        if progress_this_pass == 0:
            return {"status": "error", "detail": f"no progress while nested archives remain under {root}"}

    if errors:
        return {
            "status": "error",
            "detail": f"{processed}/{len(archives)} nested archives processed; first errors: {errors[:3]}",
        }

    nested_marker_path(root).write_text(
        json.dumps(
            {
                "status": "complete",
                "label": label,
                "archives": processed,
                "deleted_archives": deleted,
                "passes": passes,
                "completed_at": now(),
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "complete",
        "detail": f"{processed} nested archives processed across {passes} passes; {deleted} extracted archive files deleted",
    }


def extract_zip(zip_path: Path, dest: Path, label: str, *, skip_macos: bool = True) -> dict[str, object]:
    if extraction_fully_complete(dest):
        return {"status": "complete", "detail": f"already extracted to {dest}; archive no longer required"}
    if not zip_path.exists():
        return {"status": "waiting", "detail": f"missing {zip_path}"}
    if marker_matches(dest, zip_path):
        return {"status": "complete", "detail": f"already extracted to {dest}"}

    dest.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        with ZipFile(zip_path) as zf:
            infos = zf.infolist()
            total_members = len(infos)
            total_bytes = sum(info.file_size for info in infos)
            extracted_files = 0
            extracted_bytes = 0
            last_report = time.time()

            for index, info in enumerate(infos, start=1):
                name = info.filename
                if skip_macos and (name.startswith("__MACOSX/") or name.endswith(".DS_Store")):
                    continue
                target = safe_target(dest, name)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.stat().st_size == info.file_size:
                    extracted_files += 1
                    extracted_bytes += info.file_size
                    continue
                with zf.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                extracted_files += 1
                extracted_bytes += info.file_size
                if time.time() - last_report > 60:
                    print(
                        json.dumps(
                            {
                                "event": "extract_progress",
                                "label": label,
                                "index": index,
                                "members": total_members,
                                "bytes": extracted_bytes,
                                "total_bytes": total_bytes,
                            },
                            ensure_ascii=True,
                        ),
                        flush=True,
                    )
                    last_report = time.time()

        payload = {
            "status": "complete",
            "label": label,
            "zip_path": str(zip_path),
            "zip_size": zip_path.stat().st_size,
            "dest": str(dest),
            "files": extracted_files,
            "uncompressed_bytes": extracted_bytes,
            "seconds": round(time.time() - start, 3),
            "completed_at": now(),
        }
        marker_path(dest).write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return {"status": "complete", "detail": f"{extracted_files} files, {human(extracted_bytes)} extracted to {dest}"}
    except BadZipFile as exc:
        return {"status": "error", "detail": f"bad zip {zip_path}: {exc}"}
    except Exception as exc:  # Keep the monitor alive; the next loop may recover after download completes.
        return {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}


def verify_things_repo() -> dict[str, object]:
    if THINGS_VERIFY_MARKER.exists():
        try:
            payload = json.loads(THINGS_VERIFY_MARKER.read_text(encoding="utf-8"))
            if payload.get("status") == "complete":
                return {"status": "complete", "detail": str(payload.get("detail", "verified"))}
        except json.JSONDecodeError:
            pass

    if tmux_exists("things_eeg2_dl"):
        return {"status": "waiting", "detail": "things_eeg2_dl is still running"}

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        missing: list[str] = []
        wrong_size: list[str] = []
        total_files = 0
        total_bytes = 0
        for info in api.list_repo_tree("gasparyanartur/things-eeg2", repo_type="dataset", recursive=True, expand=True):
            path = getattr(info, "path", "")
            size = getattr(info, "size", None)
            if not path or size is None:
                continue
            total_files += 1
            total_bytes += int(size)
            local = DATA_ROOT / "THINGS-EEG2" / path
            if not local.exists():
                missing.append(path)
            elif local.stat().st_size != int(size):
                wrong_size.append(path)
        if missing or wrong_size:
            return {
                "status": "incomplete",
                "detail": f"{len(missing)} missing, {len(wrong_size)} wrong-size files out of {total_files}",
                "missing": missing[:20],
                "wrong_size": wrong_size[:20],
            }
        detail = f"{total_files} files, {human(total_bytes)} verified"
        THINGS_VERIFY_MARKER.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "detail": detail,
                    "files": total_files,
                    "bytes": total_bytes,
                    "completed_at": now(),
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {"status": "complete", "detail": detail}
    except Exception as exc:
        return {"status": "error", "detail": f"THINGS verification failed: {type(exc).__name__}: {exc}"}


def process_once() -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}

    records["EIT-1M public release"] = extract_zip(
        DATA_ROOT / "EIT-1M" / "Participant4_Session1_Visual_Textual.zip",
        DATA_ROOT / "EIT-1M" / "extracted",
        "EIT-1M",
    )
    if records["EIT-1M public release"]["status"] == "complete":
        records["EIT-1M nested archives"] = extract_nested_archives(
            DATA_ROOT / "EIT-1M" / "extracted",
            "EIT-1M",
        )
    else:
        records["EIT-1M nested archives"] = {"status": "waiting", "detail": "outer zip is not extracted yet"}

    things_img_train = extract_zip(
        DATA_ROOT / "THINGS-EEG2" / "imgs" / "training_images.zip",
        DATA_ROOT / "THINGS-EEG2" / "image_set",
        "THINGS-EEG2 training images",
    )
    things_img_test = extract_zip(
        DATA_ROOT / "THINGS-EEG2" / "imgs" / "test_images.zip",
        DATA_ROOT / "THINGS-EEG2" / "image_set",
        "THINGS-EEG2 test images",
    )
    records["THINGS-EEG2 image zips"] = {
        "status": "complete" if things_img_train["status"] == things_img_test["status"] == "complete" else "waiting",
        "detail": f"train={things_img_train['status']}; test={things_img_test['status']}",
    }
    if records["THINGS-EEG2 image zips"]["status"] == "complete":
        records["THINGS-EEG2 image nested archives"] = extract_nested_archives(
            DATA_ROOT / "THINGS-EEG2" / "image_set",
            "THINGS-EEG2 image set",
        )
    else:
        records["THINGS-EEG2 image nested archives"] = {"status": "waiting", "detail": "image zips are not extracted yet"}
    records["THINGS-EEG2 full file verification"] = verify_things_repo()

    eeg_imagenet_dest = DATA_ROOT / "EEG-ImageNet" / "extracted"
    if extraction_fully_complete(eeg_imagenet_dest):
        records["EEG-ImageNet extraction"] = {"status": "complete", "detail": f"already extracted to {eeg_imagenet_dest}; archive no longer required"}
        records["EEG-ImageNet nested archives"] = {"status": "complete", "detail": f"nested archives already processed under {eeg_imagenet_dest}"}
    elif tmux_exists("eeg_imagenet_dl"):
        records["EEG-ImageNet extraction"] = {"status": "waiting", "detail": "eeg_imagenet_dl is still running"}
        records["EEG-ImageNet nested archives"] = {"status": "waiting", "detail": "outer zip is not extracted yet"}
    else:
        records["EEG-ImageNet extraction"] = extract_zip(
            DATA_ROOT / "EEG-ImageNet" / "eeg-imagenet.zip",
            eeg_imagenet_dest,
            "EEG-ImageNet",
        )
        if records["EEG-ImageNet extraction"]["status"] == "complete":
            records["EEG-ImageNet nested archives"] = extract_nested_archives(
                eeg_imagenet_dest,
                "EEG-ImageNet",
            )
        else:
            records["EEG-ImageNet nested archives"] = {"status": "waiting", "detail": "outer zip is not extracted yet"}

    imagenet_dest = DATA_ROOT / "ImageNet" / "kaggle_cls_loc" / "extracted"
    if extraction_fully_complete(imagenet_dest):
        records["ImageNet Kaggle extraction"] = {"status": "complete", "detail": f"already extracted to {imagenet_dest}; archive no longer required"}
        records["ImageNet Kaggle nested archives"] = {"status": "complete", "detail": f"nested archives already processed under {imagenet_dest}"}
    elif tmux_exists("imagenet_dl"):
        records["ImageNet Kaggle extraction"] = {"status": "waiting", "detail": "imagenet_dl is still running"}
        records["ImageNet Kaggle nested archives"] = {"status": "waiting", "detail": "outer zip is not extracted yet"}
    else:
        records["ImageNet Kaggle extraction"] = extract_zip(
            DATA_ROOT / "ImageNet" / "kaggle_cls_loc" / "imagenet-object-localization-challenge.zip",
            imagenet_dest,
            "ImageNet Kaggle CLS-LOC",
        )
        if records["ImageNet Kaggle extraction"]["status"] == "complete":
            records["ImageNet Kaggle nested archives"] = extract_nested_archives(
                imagenet_dest,
                "ImageNet Kaggle CLS-LOC",
            )
        else:
            records["ImageNet Kaggle nested archives"] = {"status": "waiting", "detail": "outer zip is not extracted yet"}

    return records


def all_complete(records: dict[str, dict[str, object]]) -> bool:
    return all(record.get("status") == "complete" for record in records.values())


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    poll_seconds = int(os.environ.get("POSTPROCESS_POLL_SECONDS", "300"))
    while True:
        records = process_once()
        write_status(records)
        print(json.dumps({"event": "status", "updated": now(), "records": records}, ensure_ascii=True), flush=True)
        if all_complete(records):
            print(json.dumps({"event": "all_complete", "updated": now()}, ensure_ascii=True), flush=True)
            return
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
