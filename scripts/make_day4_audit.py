from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _count_jsonl(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _manifest_summary(root: Path) -> dict[str, Any]:
    split_rows = {split: _read_jsonl(root / f"{split}.jsonl") for split in ["train", "val", "test"]}
    all_rows = [row for rows in split_rows.values() for row in rows]
    return {
        "split_sizes": {split: len(rows) for split, rows in split_rows.items()},
        "trials": len(all_rows),
        "unique_images": len({str(row.get("image_id")) for row in all_rows}),
        "classes": len({str(row.get("label_name", row.get("label"))) for row in all_rows}),
    }


def _last_jsonl(path: str | Path) -> dict[str, Any] | None:
    last = None
    for row in _read_jsonl(path):
        last = row
    return last


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _duplicate_debug(rows: list[dict[str, Any]]) -> dict[str, Any]:
    image_counts = Counter(str(row.get("image_id")) for row in rows)
    class_counts = Counter(str(row.get("label_name", row.get("label"))) for row in rows)
    duplicate_images = {image_id: count for image_id, count in image_counts.items() if count > 1}
    return {
        "trials": len(rows),
        "unique_images": len(image_counts),
        "duplicate_image_ids": len(duplicate_images),
        "max_trials_per_image": max(image_counts.values()) if image_counts else 0,
        "classes": len(class_counts),
        "largest_class_count": max(class_counts.values()) if class_counts else 0,
    }


def write_audit(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_summary(data_root)
    alignment = _read_json("outputs/day2_align/alignment_metrics.json") or _read_json("outputs/day2_align/retrieval_metrics.json") or {}
    model_metrics = alignment.get("model", {}) if isinstance(alignment, dict) else {}
    random_metrics = alignment.get("random", {}) if isinstance(alignment, dict) else {}
    day3_metrics = _read_csv("outputs/day3/sanity_real/metrics.csv")
    gate_text = Path("outputs/day3/sanity_real/gate_analysis.md")
    fusion_val = _last_jsonl("outputs/day3/fusion_qwen15/val_log.jsonl")
    fusion_train = _last_jsonl("outputs/day3/fusion_qwen15/train_log.jsonl")
    samples = _read_jsonl("outputs/day3/sanity_real/sample_predictions.jsonl", limit=10)

    lines = [
        "# Current Result Audit",
        "",
        "## Dataset Size",
        "",
        f"- EEG trials: `{manifest['trials']}`",
        f"- Unique images: `{manifest['unique_images']}`",
        f"- Classes: `{manifest['classes']}`",
        f"- Split sizes: `{manifest['split_sizes']}`",
        "",
        "## Current Alignment Metrics",
        "",
        f"- R@1: `{_fmt(model_metrics.get('r@1'))}`",
        f"- R@5: `{_fmt(model_metrics.get('r@5'))}`",
        f"- R@10: `{_fmt(model_metrics.get('r@10'))}`",
        f"- Random R@1: `{_fmt(random_metrics.get('r@1'))}`",
        f"- Random R@5: `{_fmt(random_metrics.get('r@5'))}`",
        f"- Random R@10: `{_fmt(random_metrics.get('r@10'))}`",
        "",
        "## Current Fusion Metrics",
        "",
        f"- Last train log: `{fusion_train}`",
        f"- Last validation log: `{fusion_val}`",
        "- Sample predictions from Day3 sanity:",
    ]
    for row in samples[:5]:
        lines.append(
            f"  - `{row.get('corruption')}/{row.get('mode')}/{row.get('image_id')}`: "
            f"{' '.join(str(row.get('prediction', '')).split())}"
        )

    lines.extend(
        [
            "",
            "## Current Gate Values",
            "",
            f"- Gate analysis source: `{gate_text}`",
        ]
    )
    if gate_text.exists():
        for line in gate_text.read_text(encoding="utf-8").splitlines()[:20]:
            if line.startswith("|") and "gate_mean" not in line:
                lines.append(line)

    lines.extend(
        [
            "",
            "## Current Failure Modes",
            "",
            "Current caption generation quality is weak and code-like.",
            "Current EEG benefit is preliminary because real EEG does not consistently beat shuffled EEG.",
            "Current gate behavior does not yet prove selective EEG usage.",
            "",
            "Day3 sanity metrics show mixed real-vs-control behavior; Day4/Day5 must improve captions and rerun controlled comparisons.",
        ]
    )
    (out_dir / "CURRENT_RESULT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    test_rows = _read_jsonl(data_root / "test.jsonl")
    dup = _duplicate_debug(test_rows)
    r1 = float(model_metrics.get("r@1", 0.0) or 0.0)
    r5 = float(model_metrics.get("r@5", 0.0) or 0.0)
    debug_lines = [
        "# Retrieval Evaluation Debug",
        "",
        "## Top-k Invariant",
        "",
        f"- R@1: `{_fmt(r1)}`",
        f"- R@5: `{_fmt(r5)}`",
        f"- R@5 >= R@1: `{r5 >= r1}`",
        "- The retrieval implementation ranks normalized EEG embeddings against normalized CLIP embeddings.",
        "- Duplicate positives are handled by matching `image_id`, so any repeated trial for the same image is counted as a valid positive.",
        "",
        "## Test Candidate Set",
        "",
        f"- Test trials: `{dup['trials']}`",
        f"- Unique image IDs: `{dup['unique_images']}`",
        f"- Duplicate image IDs: `{dup['duplicate_image_ids']}`",
        f"- Max trials per image: `{dup['max_trials_per_image']}`",
        f"- Classes: `{dup['classes']}`",
        f"- Largest class count: `{dup['largest_class_count']}`",
        "",
        "## Interpretation",
        "",
    ]
    if abs(r5 - r1) < 1e-12:
        debug_lines.append(
            "R@1 and R@5 are identical in this run. This is possible when first positives are mostly at rank 1 or beyond rank 5; unit tests cover top-k and duplicate-positive handling."
        )
    else:
        debug_lines.append("R@5 differs from R@1, consistent with standard top-k behavior.")
    debug_lines.append("Random baseline is computed over the same duplicate-aware candidate set.")
    (out_dir / "retrieval_eval_debug.md").write_text("\n".join(debug_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Day4 audit reports from current Day2/Day3 outputs.")
    parser.add_argument("--data_root", default="data/thought2text")
    parser.add_argument("--out_dir", default="outputs/day4_audit")
    args = parser.parse_args()
    write_audit(args)
    print(args.out_dir)


if __name__ == "__main__":
    main()
