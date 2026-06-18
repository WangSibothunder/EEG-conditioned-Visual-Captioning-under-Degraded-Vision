from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        obj = json.loads(raw_line)
        if not isinstance(obj, dict):
            raise ValueError(f"JSONL row in {path} is not an object")
        rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _wnid_from_image_path(image_path: str) -> str:
    parts = Path(image_path).parts
    if len(parts) >= 2 and parts[0] == "images":
        return parts[1]
    return Path(image_path).stem.split("_", 1)[0]


def _candidate_paths(image_root: Path, row: dict[str, Any], relative_to: Path | None) -> list[Path]:
    image_id = str(row.get("image_id", "")).strip()
    image_path = str(row.get("image_path", "")).strip()
    wnid = _wnid_from_image_path(image_path) if image_path else image_id.split("_", 1)[0]
    candidates = [
        image_root / "ILSVRC" / "Data" / "CLS-LOC" / "train" / wnid / f"{image_id}.JPEG",
        image_root / "ILSVRC" / "Data" / "CLS-LOC" / "val" / f"{image_id}.JPEG",
        image_root / "Data" / "CLS-LOC" / "train" / wnid / f"{image_id}.JPEG",
        image_root / "Data" / "CLS-LOC" / "val" / f"{image_id}.JPEG",
        image_root / "train" / wnid / f"{image_id}.JPEG",
        image_root / "val" / wnid / f"{image_id}.JPEG",
        image_root / "images" / wnid / f"{image_id}.JPEG",
        image_root / wnid / f"{image_id}.JPEG",
        image_root / f"{image_id}.JPEG",
    ]
    if relative_to is not None:
        candidates.extend(
            [
                relative_to / "data" / "EEG-ImageNet" / "images" / wnid / f"{image_id}.JPEG",
                relative_to / "data" / "images" / wnid / f"{image_id}.JPEG",
            ]
        )
    return candidates


def rewrite_manifest_image_paths(
    manifest_path: str | Path,
    *,
    image_root: str | Path,
    out: str | Path,
    report: str | Path,
    relative_to: str | Path | None = None,
    filtered_out: str | Path | None = None,
) -> dict[str, Any]:
    manifest = Path(manifest_path)
    image_root_path = Path(image_root)
    out_path = Path(out)
    report_path = Path(report)
    relative_to_path = Path(relative_to) if relative_to is not None else None
    filtered_out_path = Path(filtered_out) if filtered_out is not None else None

    rows = _read_jsonl(manifest)
    rewritten: list[dict[str, Any]] = []
    filtered_rows: list[dict[str, Any]] = []
    matched_rows = 0
    missing_rows = 0
    missing_examples: list[dict[str, str]] = []
    missing_unique_images: set[str] = set()
    wnid_hits: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    for row in rows:
        if not isinstance(row, dict):
            continue
        updated = dict(row)
        candidates = _candidate_paths(image_root_path, updated, relative_to_path)
        matched_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if matched_path is not None:
            matched_rows += 1
            wnid_hits[_wnid_from_image_path(str(updated.get("image_path", "")))] += 1
            if relative_to_path is not None:
                try:
                    updated["image_path"] = str(matched_path.relative_to(relative_to_path))
                except ValueError:
                    updated["image_path"] = str(matched_path)
            else:
                updated["image_path"] = str(matched_path)
            filtered_rows.append(dict(updated))
        else:
            missing_rows += 1
            missing_unique_images.add(str(updated.get("image_id", "")))
            if len(missing_examples) < 10:
                missing_examples.append(
                    {
                        "image_id": str(updated.get("image_id", "")),
                        "image_path": str(updated.get("image_path", "")),
                    }
                )
        split_counts[str(updated.get("split", "unknown"))] += 1
        rewritten.append(updated)

    _write_jsonl(out_path, rewritten)
    if filtered_out_path is not None:
        _write_jsonl(filtered_out_path, filtered_rows)
    loader_ready = "fully image-linked" if missing_rows == 0 and rewritten else "not fully image-linked"
    report_lines = [
        "# EEG-ImageNet Image Link Report",
        "",
        f"- Date: `{_utc_date()}`",
        f"- Manifest: `{manifest}`",
        f"- Image root: `{image_root_path}`",
        f"- Output manifest: `{out_path}`",
        f"- Total rows: `{len(rewritten)}`",
        f"- Matched rows: `{matched_rows}`",
        f"- Missing rows: `{missing_rows}`",
        f"- Missing unique image IDs: `{len(missing_unique_images)}`",
        f"- Loader-ready status: `{loader_ready}`",
        f"- Split counts: `{dict(split_counts)}`",
        f"- WNID hits: `{dict(wnid_hits)}`",
    ]
    if filtered_out_path is not None:
        report_lines.extend(
            [
                f"- Exact-linked filtered manifest: `{filtered_out_path}`",
                f"- Exact-linked filtered rows: `{len(filtered_rows)}`",
                f"- Exact-linked subset status: `{'ready for paired training' if filtered_rows else 'empty'}`",
            ]
        )
    report_lines.extend(["", "## Blockers", ""])
    if missing_rows:
        report_lines.append("- Some manifest rows still have no resolvable local image file.")
        for example in missing_examples:
            report_lines.append(f"- Missing: `{example['image_id']}` -> `{example['image_path']}`")
    else:
        report_lines.append("- No image-path blocker detected.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "manifest_path": str(manifest),
        "image_root": str(image_root_path),
        "output_manifest": str(out_path),
        "report_path": str(report_path),
        "total_rows": len(rewritten),
        "matched_rows": matched_rows,
        "missing_rows": missing_rows,
        "missing_unique_images": len(missing_unique_images),
        "filtered_rows": len(filtered_rows),
        "filtered_output_manifest": str(filtered_out_path) if filtered_out_path is not None else "",
        "loader_ready": missing_rows == 0 and bool(rewritten),
        "wnid_hits": dict(wnid_hits),
        "split_counts": dict(split_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite EEG-ImageNet image paths once ImageNet files are available locally.")
    parser.add_argument("--manifest", required=True, help="Input EEG-ImageNet JSONL manifest.")
    parser.add_argument("--image-root", required=True, help="Local ImageNet root to search for images.")
    parser.add_argument("--out", required=True, help="Output rewritten JSONL manifest.")
    parser.add_argument("--filtered-out", default=None, help="Optional output JSONL containing only exact-linked rows.")
    parser.add_argument("--report", required=True, help="Report markdown path.")
    parser.add_argument("--relative-to", default=None, help="Optional root used to keep output paths relative.")
    args = parser.parse_args()
    stats = rewrite_manifest_image_paths(
        args.manifest,
        image_root=args.image_root,
        out=args.out,
        report=args.report,
        relative_to=args.relative_to,
        filtered_out=args.filtered_out,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
