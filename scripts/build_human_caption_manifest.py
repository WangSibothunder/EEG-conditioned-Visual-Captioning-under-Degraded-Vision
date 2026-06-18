from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.imagenet_labels import caption_for_wnid, human_name_for_wnid, label_name_from_row


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def convert_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    converted_rows: list[dict[str, Any]] = []
    examples: list[dict[str, str]] = []
    unknown_wnids: set[str] = set()
    converted = 0

    for row in rows:
        wnid = label_name_from_row(row)
        human_name = human_name_for_wnid(wnid)
        new_row = dict(row)
        old_caption = str(row.get("caption", ""))
        new_row["caption"] = caption_for_wnid(wnid)
        new_row["human_label_name"] = human_name if human_name else wnid
        new_row["caption_source"] = "human_class" if human_name else "wnid_fallback"
        if human_name:
            converted += 1
        elif wnid:
            unknown_wnids.add(wnid)
        if len(examples) < 20:
            examples.append(
                {
                    "image_id": str(row.get("image_id", "")),
                    "wnid": str(wnid or ""),
                    "before": old_caption,
                    "after": str(new_row["caption"]),
                    "source": str(new_row["caption_source"]),
                }
            )
        converted_rows.append(new_row)

    stats: dict[str, Any] = {
        "rows": len(rows),
        "converted": converted,
        "unknown": len(unknown_wnids),
        "unknown_wnids": sorted(unknown_wnids),
        "examples": examples,
    }
    return converted_rows, stats


def write_report(path: str | Path, split_stats: dict[str, dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = sum(int(stats["rows"]) for stats in split_stats.values())
    total_converted = sum(int(stats["converted"]) for stats in split_stats.values())
    unknown = sorted({wnid for stats in split_stats.values() for wnid in stats["unknown_wnids"]})
    lines = [
        "# ImageNet Classname Manifest Report",
        "",
        f"- Total rows: `{total_rows}`",
        f"- Captions converted: `{total_converted}`",
        f"- Unknown wnids: `{len(unknown)}`",
        f"- Unknown wnid list: `{', '.join(unknown) if unknown else 'none'}`",
        "",
        "| Split | Rows | Converted | Unknown WNIDs |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split, stats in split_stats.items():
        lines.append(f"| {split} | {stats['rows']} | {stats['converted']} | {stats['unknown']} |")

    lines.extend(["", "## Examples", "", "| Split | Image ID | WNID | Before | After | Source |", "| --- | --- | --- | --- | --- | --- |"])
    remaining = 20
    for split, stats in split_stats.items():
        for example in stats["examples"]:
            if remaining <= 0:
                break
            lines.append(
                "| "
                + " | ".join(
                    [
                        split,
                        example["image_id"],
                        example["wnid"],
                        example["before"],
                        example["after"],
                        example["source"],
                    ]
                )
                + " |"
            )
            remaining -= 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_manifest(
    source: str | Path,
    target: str | Path,
    *,
    report_path: str | Path | None = None,
    split_name: str | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(source)
    converted_rows, stats = convert_rows(rows)
    _write_jsonl(target, converted_rows)
    if report_path is not None:
        write_report(report_path, {split_name or Path(source).stem: stats})
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Thought2Text manifests with human-readable ImageNet captions.")
    parser.add_argument("--root", default="data/thought2text")
    parser.add_argument("--report", default="outputs/day4_caption_targets/imagenet_classname_manifest_report.md")
    args = parser.parse_args()

    root = Path(args.root)
    split_stats: dict[str, dict[str, Any]] = {}
    for split in ["train", "val", "test"]:
        split_stats[split] = convert_manifest(
            root / f"{split}.jsonl",
            root / f"{split}_human_caption.jsonl",
            split_name=split,
        )
    write_report(args.report, split_stats)
    print(args.report)


if __name__ == "__main__":
    main()
