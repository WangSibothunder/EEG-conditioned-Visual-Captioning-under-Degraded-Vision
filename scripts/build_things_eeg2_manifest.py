from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EEG_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".mat", ".fif", ".set"}
REQUIRED_MANIFEST_KEYS = {"image_id", "image_path", "eeg_path", "caption", "label"}


def _utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _scan(root: Path) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
    eeg_files = [
        path
        for path in files
        if path.suffix.lower() in EEG_SUFFIXES
        or any(token in path.name.lower() for token in ("eeg", "preprocessed", "epoch"))
    ]
    image_files = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    metadata_files = [
        path
        for path in files
        if path.suffix.lower() in {".json", ".jsonl", ".csv", ".tsv", ".txt", ".mat", ".pkl", ".pickle"}
    ]
    jsonl_manifests = [path for path in files if path.suffix.lower() == ".jsonl"]
    subjects = sorted(
        {
            part
            for path in files
            for part in path.parts
            if part.lower().startswith(("sub-", "subject", "subj"))
        }
    )
    return {
        "exists": root.exists(),
        "file_count": len(files),
        "total_size_bytes": sum(path.stat().st_size for path in files) if files else 0,
        "eeg_files": eeg_files,
        "image_files": image_files,
        "metadata_files": metadata_files,
        "jsonl_manifests": jsonl_manifests,
        "subjects": subjects,
        "examples": files[:40],
    }


def _validate_project_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                missing = REQUIRED_MANIFEST_KEYS.difference(row)
                if missing:
                    return None
                rows.append(row)
    except (OSError, json.JSONDecodeError):
        return None
    if not rows:
        return None
    return {
        "path": path,
        "rows": rows,
        "count": len(rows),
    }


def _find_project_manifest(scan: dict[str, Any]) -> dict[str, Any] | None:
    for path in sorted(scan["jsonl_manifests"]):
        manifest = _validate_project_manifest(path)
        if manifest is not None:
            return manifest
    return None


def _write_status(root: Path, out: Path, scan: dict[str, Any], manifest: dict[str, Any] | None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    eeg_files = scan["eeg_files"]
    image_files = scan["image_files"]
    metadata_files = scan["metadata_files"]
    subjects = scan["subjects"]
    loader_ready = manifest is not None
    trial_alignment_ready = loader_ready
    raw_continuous_eeg = any(path.name.startswith("raw_eeg_") for path in eeg_files)
    image_set_ready = any("image_set" in path.parts for path in image_files)
    lines = [
        "# THINGS-EEG2 Status",
        "",
        f"- Date: `{_utc_date()}`",
        f"- Data root: `{root}`",
        f"- Exists: `{scan['exists']}`",
        f"- File count: `{scan['file_count']}`",
        f"- Total file size bytes: `{scan['total_size_bytes']}`",
        f"- Available subjects: `{', '.join(subjects) if subjects else 'none detected'}`",
        f"- EEG-like files: `{len(eeg_files)}`",
        f"- Image files: `{len(image_files)}`",
        f"- Metadata/table files: `{len(metadata_files)}`",
        f"- Loader-ready: `{loader_ready}`",
        f"- Trial/image alignment ready: `{trial_alignment_ready}`",
        f"- Raw continuous EEG present: `{raw_continuous_eeg}`",
        f"- Extracted image set present: `{image_set_ready}`",
        f"- Project-ready manifest detected: `{manifest['path'] if manifest else 'no'}`",
        "",
        "## Availability",
        "",
    ]
    if manifest is not None:
        lines.append("Project-ready manifest detected. The current tree is loader-ready for the current project schema.")
    elif not scan["exists"]:
        lines.append("Blocked: dataset root is missing.")
    elif scan["file_count"] == 0:
        lines.append("Blocked: `data/THINGS-EEG2` exists but contains no files.")
    elif raw_continuous_eeg and image_set_ready:
        lines.append(
            "Partial local files exist: raw continuous EEG and extracted images are present, but trial/image alignment is not yet materialized."
        )
    elif not eeg_files or not image_files:
        lines.append("Blocked: the current tree does not expose both EEG-like files and image files.")
    else:
        lines.append("Partial local files exist. A real manifest still requires dataset-specific metadata mapping images, EEG epochs, captions/classes, subjects, and splits.")
    lines.extend(
        [
            "",
            "## Required Directory Structure",
            "",
            "Provide either a project-ready JSONL manifest or a convertible subset with this information:",
            "",
            "```text",
            "data/THINGS-EEG2/",
            "  images/...                         # stimulus images",
            "  eeg/...                            # preprocessed per-trial EEG windows or raw files plus events",
            "  metadata/...                       # image IDs, class labels/captions, subject IDs, split/event mapping",
            "  train.jsonl / val.jsonl            # optional project-compatible manifests",
            "```",
            "",
            "Expected project JSONL fields: `image_id`, `image_path`, `eeg_path`, `caption`, `label`, `subject_id`, `split`.",
            "",
            "## Readiness Interpretation",
            "",
            f"- Loader-ready: `{loader_ready}`",
            f"- Trial/image alignment ready: `{trial_alignment_ready}`",
            f"- Raw continuous EEG only: `{raw_continuous_eeg and not loader_ready}`",
            "",
            "## Example Files",
            "",
        ]
    )
    for path in scan["examples"]:
        lines.append(f"- `{_rel(path, root)}` ({path.stat().st_size} bytes)")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_smoke_report(root: Path, out: Path, scan: dict[str, Any], manifest: dict[str, Any] | None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    ready = manifest is not None
    files_present = bool(scan["exists"] and scan["eeg_files"] and scan["image_files"])
    raw_continuous_eeg = any(path.name.startswith("raw_eeg_") for path in scan["eeg_files"])
    lines = [
        "# THINGS-EEG2 Smoke Report",
        "",
        f"Date: {_utc_date()}",
        "",
        "## Result",
        "",
        f"- CPU inspection ran: `true`",
        f"- Dataset usable for current image+EEG manifest: `{ready}`",
        f"- Files found: `{scan['file_count']}`",
        f"- EEG-like files: `{len(scan['eeg_files'])}`",
        f"- Image files: `{len(scan['image_files'])}`",
        f"- Raw continuous EEG present: `{raw_continuous_eeg}`",
        "",
        "## Blocker",
        "",
    ]
    if ready:
        lines.append("A project-compatible JSONL manifest was detected.")
    elif files_present:
        lines.append("Files are present, but this skeleton did not infer trial-level alignment without dataset-specific metadata.")
    elif scan["exists"]:
        lines.append("The local THINGS-EEG2 directory is empty or lacks required EEG/image assets.")
    else:
        lines.append("The local THINGS-EEG2 directory is missing.")
    lines.extend(
        [
            "",
            "## Next CPU Smoke",
            "",
            "After placing a small converted subset, run a dataset loader smoke against the generated JSONL before any GPU training.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(root: Path, out_manifest: Path, status_out: Path, smoke_out: Path) -> int:
    scan = _scan(root)
    manifest = _find_project_manifest(scan)
    if manifest is not None:
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.write_text(manifest["path"].read_text(encoding="utf-8"), encoding="utf-8")
    _write_status(root, status_out, scan, manifest)
    _write_smoke_report(root, smoke_out, scan, manifest)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or report THINGS-EEG2 manifest readiness without guessing trial alignment.")
    parser.add_argument("--root", default="data/THINGS-EEG2")
    parser.add_argument("--out", default="data/THINGS-EEG2/small_manifest.jsonl")
    parser.add_argument("--status-out", default="outputs/datasets/THINGS_EEG2_STATUS.md")
    parser.add_argument("--smoke-out", default="outputs/datasets/things_eeg2_smoke_report.md")
    args = parser.parse_args()
    raise SystemExit(build(Path(args.root), Path(args.out), Path(args.status_out), Path(args.smoke_out)))


if __name__ == "__main__":
    main()
