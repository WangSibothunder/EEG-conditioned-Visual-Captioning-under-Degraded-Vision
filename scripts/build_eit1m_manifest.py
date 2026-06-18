from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {".vhdr", ".vmrk", ".json", ".jsonl", ".csv", ".tsv", ".txt", ".md"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _normalize_category(raw: str) -> str:
    return {"airplne": "airplane"}.get(raw, raw)


def _safe_read_text_member(archive: zipfile.ZipFile, name: str, max_bytes: int = 2_000_000) -> str:
    info = archive.getinfo(name)
    if info.file_size > max_bytes:
        return ""
    data = archive.read(name)
    return data.decode("utf-8", errors="replace")


def _parse_vmrk_positions(text: str, limit: int) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("Mk"):
            continue
        _, _, payload = line.partition("=")
        parts = [part.strip() for part in payload.split(",")]
        if len(parts) < 3 or parts[0] != "Stimulus":
            continue
        code = parts[1]
        try:
            position = int(parts[2])
        except ValueError:
            continue
        markers.append({"code": code, "position": position})
        if len(markers) >= limit:
            break
    return markers


def _category_from_member(member: str) -> str | None:
    name = Path(member).name
    if not name.endswith("_train.vhdr"):
        return None
    return name[: -len("_train.vhdr")]


def _inspect_zip(zip_path: Path, max_samples: int) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive:
        infos = {info.filename: info for info in archive.infolist()}
        members = list(infos)
        vhdr_members = sorted(name for name in members if name.endswith(".vhdr") and "__MACOSX/" not in name)
        vmrk_members = sorted(name for name in members if name.endswith(".vmrk") and "__MACOSX/" not in name)
        eeg_members = sorted(name for name in members if name.endswith(".eeg") and "__MACOSX/" not in name)
        image_members = [name for name in members if Path(name).suffix.lower() in IMAGE_SUFFIXES]
        sidecars = [
            name
            for name in members
            if Path(name).suffix.lower() in {".json", ".jsonl", ".csv", ".tsv", ".txt"}
            and "__MACOSX/" not in name
        ]
        rows: list[dict[str, Any]] = []
        category_rows: list[dict[str, Any]] = []
        per_category_limit = max(1, (max_samples + max(1, len(vhdr_members)) - 1) // max(1, len(vhdr_members)))
        for label_id, vhdr in enumerate(vhdr_members):
            raw = _category_from_member(vhdr)
            if raw is None:
                continue
            category = _normalize_category(raw)
            base = vhdr[: -len(".vhdr")]
            vmrk = f"{base}.vmrk"
            eeg = f"{base}.eeg"
            vmrk_text = _safe_read_text_member(archive, vmrk) if vmrk in infos else ""
            markers = _parse_vmrk_positions(vmrk_text, per_category_limit)
            category_rows.append(
                {
                    "label_id": label_id,
                    "category_raw": raw,
                    "category_normalized": category,
                    "vhdr_member": vhdr,
                    "vmrk_member": vmrk if vmrk in infos else None,
                    "eeg_member": eeg if eeg in infos else None,
                    "marker_count_sampled": len(markers),
                    "eeg_file_size_bytes": infos[eeg].file_size if eeg in infos else None,
                }
            )
            for local_idx, marker in enumerate(markers):
                if len(rows) >= max_samples:
                    break
                image_id = f"eit1m_p4_s1_{category}_{local_idx:05d}"
                rows.append(
                    {
                        "image_id": image_id,
                        "image_path": "",
                        "eeg_path": "",
                        "caption": f"a visual-textual stimulus from the {category} category",
                        "label": label_id,
                        "subject_id": "P4",
                        "split": "train" if len(rows) % 5 else "val",
                        "metadata": {
                            "dataset": "EIT-1M",
                            "manifest_kind": "zip_index_only",
                            "source_zip": str(zip_path),
                            "vhdr_member": vhdr,
                            "vmrk_member": vmrk if vmrk in infos else None,
                            "eeg_member": eeg if eeg in infos else None,
                            "stimulus_code": marker["code"],
                            "stimulus_position": marker["position"],
                            "category_raw": raw,
                            "category_normalized": category,
                            "blocker": "No image files, no true caption sidecar, and no extracted per-epoch EEG .npy files are available.",
                        },
                    }
                )
        return {
            "members": members,
            "vhdr_members": vhdr_members,
            "vmrk_members": vmrk_members,
            "eeg_members": eeg_members,
            "image_members": image_members,
            "sidecars": sidecars,
            "category_rows": category_rows,
            "rows": rows,
            "total_eeg_uncompressed_bytes": sum(infos[name].file_size for name in eeg_members),
            "total_eeg_compressed_bytes": sum(infos[name].compress_size for name in eeg_members),
        }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_status(root: Path, zip_path: Path | None, report: dict[str, Any] | None, out: Path, manifest: Path | None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
    zip_files = [path for path in files if path.suffix.lower() == ".zip"]
    lines = [
        "# EIT-1M Status",
        "",
        f"- Date: `{_utc_date()}`",
        f"- Data root: `{root}`",
        f"- Exists: `{root.exists()}`",
        f"- File count: `{len(files)}`",
        f"- Total file size bytes: `{sum(path.stat().st_size for path in files) if files else 0}`",
        f"- Zip files: `{len(zip_files)}`",
        f"- Selected zip: `{zip_path if zip_path else 'none'}`",
        f"- Small manifest generated: `{manifest if manifest else 'no'}`",
        "",
    ]
    if report is None:
        lines.extend(["## Availability", "", "Blocked: no local EIT-1M zip file was found."])
    else:
        lines.extend(
            [
                "## Zip Inventory",
                "",
                f"- Total members: `{len(report['members'])}`",
                f"- `.vhdr` files: `{len(report['vhdr_members'])}`",
                f"- `.vmrk` files: `{len(report['vmrk_members'])}`",
                f"- `.eeg` files: `{len(report['eeg_members'])}`",
                f"- Image members: `{len(report['image_members'])}`",
                f"- Text/table sidecars: `{len(report['sidecars'])}`",
                f"- Total `.eeg` uncompressed bytes: `{report['total_eeg_uncompressed_bytes']}`",
                f"- Total `.eeg` compressed bytes: `{report['total_eeg_compressed_bytes']}`",
                f"- Zip-index rows written: `{len(report['rows'])}`",
                "",
                "## Availability",
                "",
                "Partially available for CPU metadata smoke only. The archive exposes BrainVision EEG recordings and category-level filenames, but no image files, no true caption/text sidecar, and no extracted per-trial EEG windows.",
                "",
                "The generated JSONL is a zip-index manifest for conversion planning. It is not directly usable by the current image+EEG caption loader until EEG epochs and image paths are materialized.",
                "",
                "## Categories",
                "",
            ]
        )
        for row in report["category_rows"]:
            lines.append(
                f"- `{row['category_normalized']}`: sampled markers `{row['marker_count_sampled']}`, EEG member `{row['eeg_member']}`"
            )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_smoke_report(zip_path: Path | None, report: dict[str, Any] | None, out: Path, manifest: Path | None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# EIT-1M Smoke Report",
        "",
        f"Date: {_utc_date()}",
        "",
        "## Result",
        "",
        "- CPU zip inspection ran without full extraction.",
        "- No `.eeg` member was extracted.",
    ]
    if report is None:
        lines.append("- Result: blocked, no zip found.")
    else:
        lines.extend(
            [
                f"- Source zip: `{zip_path}`",
                f"- Zip members inspected: `{len(report['members'])}`",
                f"- Small zip-index manifest: `{manifest}`",
                f"- Rows written: `{len(report['rows'])}`",
                f"- Image members found: `{len(report['image_members'])}`",
                f"- Text/table sidecars found: `{len(report['sidecars'])}`",
                "",
                "## Blocker",
                "",
                "The current archive is not directly trainable for image+EEG captioning because it lacks stimulus images, true captions/text sidecars, and extracted epoch `.npy` files. Safe next conversion requires streaming selected byte ranges from `.eeg` files using `.vmrk` marker positions.",
            ]
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(root: Path, out_manifest: Path, status_out: Path, smoke_out: Path, max_samples: int) -> int:
    zip_files = sorted(root.glob("*.zip")) if root.exists() else []
    zip_path = zip_files[0] if zip_files else None
    report = _inspect_zip(zip_path, max_samples=max_samples) if zip_path else None
    manifest_path: Path | None = None
    if report is not None:
        _write_jsonl(out_manifest, report["rows"])
        manifest_path = out_manifest
    _write_status(root, zip_path, report, status_out, manifest_path)
    _write_smoke_report(zip_path, report, smoke_out, manifest_path)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small EIT-1M zip-index manifest without extracting the archive.")
    parser.add_argument("--root", default="data/EIT-1M")
    parser.add_argument("--out", default="data/EIT-1M/small_manifest.jsonl")
    parser.add_argument("--status-out", default="outputs/datasets/EIT1M_STATUS.md")
    parser.add_argument("--smoke-out", default="outputs/datasets/eit1m_smoke_report.md")
    parser.add_argument("--max-samples", type=int, default=512)
    args = parser.parse_args()
    raise SystemExit(
        build(
            root=Path(args.root),
            out_manifest=Path(args.out),
            status_out=Path(args.status_out),
            smoke_out=Path(args.smoke_out),
            max_samples=max(1, min(args.max_samples, 512)),
        )
    )


if __name__ == "__main__":
    main()
