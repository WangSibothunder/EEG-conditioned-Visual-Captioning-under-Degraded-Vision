from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.imagenet_labels import caption_for_wnid


DEFAULT_SOURCES = [
    Path("/cloud/cloud-ssd1/eeg_vision_caption_data/EEG-ImageNet/extracted/EEG-ImageNet_1.pth"),
    Path("/cloud/cloud-ssd1/eeg_vision_caption_data/EEG-ImageNet/extracted/EEG-ImageNet_2.pth"),
]


def _utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _load_pth(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing EEG-ImageNet pth: {path}")
    obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(obj, dict) or not isinstance(obj.get("dataset"), list):
        raise ValueError(f"{path} must contain a dict with a `dataset` list")
    return obj


def _source_paths_from_root(root: str | Path | None) -> list[Path]:
    if root is None:
        return DEFAULT_SOURCES
    root_path = Path(root)
    if root_path.is_file():
        return [root_path]
    candidates = sorted(root_path.glob("EEG-ImageNet_*.pth"))
    if not candidates:
        candidates = sorted(root_path.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No EEG-ImageNet .pth files found under {root_path}")
    return candidates


def _iter_source_objects(source_paths: Iterable[Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    for source_path in source_paths:
        yield source_path, _load_pth(source_path)


def _inspect_sources(source_paths: list[Path]) -> dict[str, Any]:
    labels: set[str] = set()
    subjects: set[str] = set()
    eeg_shapes: Counter[str] = Counter()
    dataset_counts: dict[str, int] = {}
    for source_path, obj in _iter_source_objects(source_paths):
        dataset = obj["dataset"]
        dataset_counts[str(source_path)] = len(dataset)
        raw_labels = obj.get("labels")
        if isinstance(raw_labels, (list, tuple, set)):
            labels.update(str(label) for label in raw_labels)
        else:
            labels.update(str(row.get("label")) for row in dataset if row.get("label") is not None)
        for row in dataset:
            if not isinstance(row, dict):
                continue
            if row.get("label") is not None:
                labels.add(str(row["label"]))
            subjects.add(_subject_id(row.get("subject", "unknown")))
            eeg_tensor = row.get("eeg_data")
            if hasattr(eeg_tensor, "shape"):
                eeg_shapes[str(tuple(eeg_tensor.shape))] += 1
    return {
        "label_to_id": {label: idx for idx, label in enumerate(sorted(labels))},
        "subject_count": len(subjects),
        "eeg_shapes": dict(eeg_shapes),
        "dataset_counts": dataset_counts,
        "source_rows_total": sum(dataset_counts.values()),
    }


def _image_id(image_name: str) -> str:
    return Path(image_name).stem


def _logical_image_path(label_name: str, image_name: str) -> str:
    return str(Path("images") / label_name / image_name)


def _split_for_index(index: int) -> str:
    bucket = index % 10
    if bucket == 8:
        return "val"
    if bucket == 9:
        return "test"
    return "train"


def _subject_id(raw_subject: Any) -> str:
    try:
        value = int(raw_subject)
    except (TypeError, ValueError):
        if hasattr(raw_subject, "item"):
            value = int(raw_subject.item())
        else:
            return f"S{raw_subject}"
    return f"S{value:02d}"


def _actual_image_exists(out_dir: Path, image_root: Path | None, image_path: str) -> bool:
    if image_root is not None and (image_root / image_path).exists():
        return True
    return (out_dir / image_path).exists()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_report(path: Path, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    loader_ready = "fully loader-ready" if stats["actual_image_count"] == stats["sample_count"] else "not fully loader-ready"
    image_status = (
        "actual paths available"
        if stats["actual_image_count"] == stats["sample_count"]
        else "logical paths only; local image files were not found for every manifest row"
    )
    blockers = []
    if stats["sample_count"] == 0:
        blockers.append("No rows were converted from the `.pth` dataset lists.")
    if stats["actual_image_count"] < stats["sample_count"]:
        blockers.append(
            "Image files are not materialized at the manifest `image_path` locations, so standard loader training would fail unless `allow_missing_images` or real ImageNet files are supplied."
        )
    if not stats["eeg_files_written"]:
        blockers.append("No extracted EEG `.npy` files were written.")
    if not blockers:
        blockers.append("No conversion blocker detected for the sampled rows.")

    lines = [
        "# EEG-ImageNet Loader-Ready Report",
        "",
        f"- Date: `{_utc_date()}`",
        f"- Source files: `{', '.join(stats['source_files'])}`",
        f"- Manifest: `{stats['manifest_path']}`",
        f"- Max samples requested: `{stats['max_samples'] if stats['max_samples'] is not None else 'all'}`",
        f"- Sample count: `{stats['sample_count']}`",
        f"- Source dataset rows inspected: `{stats['source_rows_inspected']}`",
        f"- Source dataset counts: `{stats['source_dataset_counts']}`",
        f"- Rows visited for conversion: `{stats['visited_for_conversion']}`",
        f"- EEG files written: `{stats['eeg_files_written']}`",
        f"- EEG shapes: `{', '.join(f'{shape}: {count}' for shape, count in stats['eeg_shapes'].items()) if stats['eeg_shapes'] else 'none'}`",
        f"- Sampled EEG shapes: `{', '.join(f'{shape}: {count}' for shape, count in stats['sampled_eeg_shapes'].items()) if stats['sampled_eeg_shapes'] else 'none'}`",
        f"- Label count: `{stats['label_count']}`",
        f"- Subject count: `{stats['subject_count']}`",
        f"- Sampled subject count: `{stats['sampled_subject_count']}`",
        f"- Split counts: `{dict(stats['split_counts'])}`",
        f"- Actual image files found: `{stats['actual_image_count']}`",
        f"- Image path status: `{image_status}`",
        f"- Loader-ready status: `{loader_ready}`",
        "",
        "## Interpretation",
        "",
        "The generated JSONL follows the project schema and the EEG side is materialized as `.npy` files. Image paths are conservative ImageNet-style logical paths unless matching local image files are present.",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {blocker}" for blocker in blockers)
    lines.extend(
        [
            "",
            "## Example Rows",
            "",
            "| image_id | image_path | eeg_path | caption | subject_id | split |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in stats["examples"]:
        lines.append(
            f"| {row['image_id']} | {row['image_path']} | {row['eeg_path']} | {row['caption']} | {row['subject_id']} | {row['split']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    source_paths: list[str | Path] | None = None,
    *,
    out_dir: str | Path = "outputs/datasets/eeg_imagenet",
    out: str | Path | None = None,
    report_path: str | Path | None = None,
    max_samples: int | None = 2048,
    image_root: str | Path | None = None,
) -> dict[str, Any]:
    sources = [Path(path) for path in (source_paths or DEFAULT_SOURCES)]
    manifest_path = Path(out) if out is not None else Path(out_dir) / "small_manifest.jsonl"
    out_path = manifest_path.parent
    eeg_dir = out_path / "eeg"
    report = Path(report_path) if report_path is not None else out_path / "EEG_IMAGENET_READY_REPORT.md"
    image_root_path = Path(image_root) if image_root is not None else None

    source_stats = _inspect_sources(sources)
    label_to_id = source_stats["label_to_id"]
    total_rows = int(source_stats["source_rows_total"])
    if max_samples is None or max_samples >= total_rows:
        selected_indices: set[int] | None = None
    else:
        selected_indices = set(np.linspace(0, total_rows - 1, max_samples, dtype=np.int64).tolist())

    rows: list[dict[str, Any]] = []
    sampled_eeg_shapes: Counter[str] = Counter()
    sampled_subjects: set[str] = set()
    split_counts: Counter[str] = Counter()
    actual_image_count = 0
    visited_for_conversion = 0
    global_index = -1

    eeg_dir.mkdir(parents=True, exist_ok=True)
    for source_path, obj in _iter_source_objects(sources):
        for item in obj["dataset"]:
            global_index += 1
            if max_samples is not None and len(rows) >= max_samples:
                break
            if selected_indices is not None and global_index not in selected_indices:
                continue
            visited_for_conversion += 1
            if not isinstance(item, dict):
                raise ValueError(f"{source_path} dataset item {visited_for_conversion} is not a dict")
            label_name = str(item.get("label", "unknown"))
            image_name = str(item.get("image", f"{label_name}_{len(rows):06d}.JPEG"))
            image_path = _logical_image_path(label_name, image_name)
            eeg_tensor = item.get("eeg_data")
            if not isinstance(eeg_tensor, torch.Tensor):
                raise ValueError(f"{source_path} item {visited_for_conversion} missing tensor `eeg_data`")
            eeg_array = eeg_tensor.detach().cpu().float().numpy()
            if eeg_array.ndim != 2:
                raise ValueError(f"{source_path} item {visited_for_conversion} EEG must be 2D, got {eeg_array.shape}")
            eeg_rel = str(Path("eeg") / f"eeg_imagenet_{len(rows):06d}.npy")
            np.save(out_path / eeg_rel, eeg_array.astype(np.float32, copy=False))

            subject = _subject_id(item.get("subject", "unknown"))
            split = _split_for_index(len(rows))
            row = {
                "image_id": _image_id(image_name),
                "image_path": image_path,
                "eeg_path": eeg_rel,
                "caption": caption_for_wnid(label_name),
                "label": int(label_to_id.get(label_name, -1)),
                "subject_id": subject,
                "split": split,
            }
            rows.append(row)
            sampled_eeg_shapes[str(tuple(eeg_array.shape))] += 1
            sampled_subjects.add(subject)
            split_counts[split] += 1
            if _actual_image_exists(out_path, image_root_path, image_path):
                actual_image_count += 1
        if max_samples is not None and len(rows) >= max_samples:
            break

    _write_jsonl(manifest_path, rows)
    stats: dict[str, Any] = {
        "source_files": [str(path) for path in sources],
        "manifest_path": str(manifest_path),
        "max_samples": max_samples,
        "sample_count": len(rows),
        "source_rows_inspected": source_stats["source_rows_total"],
        "source_dataset_counts": source_stats["dataset_counts"],
        "visited_for_conversion": visited_for_conversion,
        "eeg_files_written": len(rows),
        "eeg_shapes": source_stats["eeg_shapes"],
        "sampled_eeg_shapes": dict(sampled_eeg_shapes),
        "label_count": len(label_to_id),
        "subject_count": source_stats["subject_count"],
        "sampled_subject_count": len(sampled_subjects),
        "split_counts": dict(split_counts),
        "actual_image_count": actual_image_count,
        "examples": rows[:10],
    }
    _write_report(report, stats)
    if report != out_path / "EEG_IMAGENET_READY_REPORT.md":
        _write_report(out_path / "EEG_IMAGENET_READY_REPORT.md", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a project-style JSONL manifest from extracted EEG-ImageNet .pth files.")
    parser.add_argument("--root", default=None, help="Directory containing extracted EEG-ImageNet .pth shards.")
    parser.add_argument("--source", action="append", default=None, help="Source EEG-ImageNet .pth file. Repeatable.")
    parser.add_argument("--out-dir", default="outputs/datasets/eeg_imagenet")
    parser.add_argument("--out", default=None, help="Exact manifest JSONL output path.")
    parser.add_argument("--report", default="outputs/datasets/EEG_IMAGENET_READY_REPORT.md")
    parser.add_argument("--max-samples", type=int, default=2048, help="Number of rows to convert. Use 0 for all rows.")
    parser.add_argument("--image-root", default=None, help="Optional root containing images/<wnid>/<file>.")
    args = parser.parse_args()

    max_samples = None if args.max_samples == 0 else args.max_samples
    sources = [Path(path) for path in args.source] if args.source else _source_paths_from_root(args.root)
    stats = build_manifest(
        sources,
        out_dir=args.out_dir,
        out=args.out,
        report_path=args.report,
        max_samples=max_samples,
        image_root=args.image_root,
    )
    print(json.dumps({key: value for key, value in stats.items() if key != "examples"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
