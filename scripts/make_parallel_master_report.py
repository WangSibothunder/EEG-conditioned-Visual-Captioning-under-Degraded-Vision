from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(path: str | Path, limit: int | None = None) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if limit is not None and len(text) > limit:
        return text[:limit].rstrip() + "\n\n... truncated ...\n"
    return text


def _csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _alignment_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for board_path in [
        Path("outputs/alignment_search/EXPERIMENT_BOARD.csv"),
        Path("outputs/alignment_long/EXPERIMENT_BOARD.csv"),
        Path("outputs/day4_alignment/multiseed_summary.csv"),
        Path("outputs/day4_search/EXPERIMENT_BOARD.csv"),
        Path("outputs/day5_extra_alignment/EXPERIMENT_BOARD.csv"),
    ]:
        if not board_path.exists():
            continue
        for row in _csv_rows(board_path):
            if row.get("status") not in {"completed", "", None} and "day4_alignment" not in str(board_path):
                continue
            try:
                val_r5 = float(row.get("val_R@5") or row.get("val_r5") or row.get("r@5") or 0.0)
            except ValueError:
                val_r5 = 0.0
            try:
                class_acc = float(row.get("class_acc") or 0.0)
            except ValueError:
                class_acc = 0.0
            candidates.append(
                {
                    "source": str(board_path),
                    "experiment_id": row.get("experiment_id") or row.get("id") or "unknown",
                    "encoder": row.get("encoder_type") or row.get("encoder") or "",
                    "loss": row.get("loss_combo") or row.get("loss") or "",
                    "score": val_r5 + 0.5 * class_acc,
                    "val_r5": val_r5,
                    "class_acc": class_acc,
                    "test_r5": row.get("test_R@5") or row.get("test_r5") or "",
                }
            )
    return candidates


def _best_alignment(source_contains: str | None = None) -> dict[str, Any] | None:
    candidates = _alignment_candidates()
    if source_contains is not None:
        candidates = [row for row in candidates if source_contains in row["source"]]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row["score"])


def _status_line(path: str | Path) -> str:
    path = Path(path)
    return f"- `{path}`: {'exists' if path.exists() else 'missing'}"


def build_report(out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    current_best = _best_alignment("outputs/alignment_search")
    historical_best = _best_alignment()
    board_rows = _csv_rows("outputs/parallel_stage/EXPERIMENT_BOARD.csv")
    robustness = _read("outputs/robustness/summary_metrics.csv")
    semantic = _read("outputs/semantic_caption/SEMANTIC_CAPTION_REPORT.md", limit=5000)
    trimodal = _read("outputs/trimodal/TRIMODAL_STATUS.md", limit=5000)
    subject = _read("outputs/subject_adaptation/SUBJECT_ADAPTATION_REPORT.md", limit=5000)
    spectrogram = _read("outputs/spectrogram/SPECTROGRAM_BRANCH_REPORT.md", limit=5000)
    dataset_rec = _read("outputs/datasets/DATASET_RECOMMENDATION.md", limit=5000)
    lines = [
        "# Parallel Stage Master Report",
        "",
        f"- Updated: `{_now()}`",
        f"- Workstreams tracked: `{len(board_rows)}`",
        "",
        "## Artifact Checklist",
        "",
    ]
    for path in [
        "outputs/parallel_stage/EXPERIMENT_BOARD.csv",
        "outputs/parallel_stage/EXPERIMENT_BOARD.md",
        "outputs/semantic_caption/SEMANTIC_CAPTION_REPORT.md",
        "outputs/semantic_caption/FULL_METRICS.csv",
        "outputs/semantic_caption/predictions.jsonl",
        "outputs/semantic_caption/checkpoints/best.pt",
        "outputs/alignment_search/SEARCH_SUMMARY.md",
        "outputs/alignment_search/best_overall.pt",
        "outputs/datasets/THINGS_EEG2_STATUS.md",
        "outputs/datasets/EIT1M_STATUS.md",
        "outputs/trimodal/TRIMODAL_STATUS.md",
        "outputs/trimodal/checkpoints/best.pt",
        "outputs/subject_adaptation/SUBJECT_ADAPTATION_REPORT.md",
        "outputs/spectrogram/SPECTROGRAM_BRANCH_REPORT.md",
        "outputs/robustness/ROBUSTNESS_REPORT.md",
        "outputs/robustness/qualitative_examples.md",
    ]:
        lines.append(_status_line(path))
    lines.extend(
        [
            "",
            "## Evidence Levels",
            "",
            "- Full controlled semantic caption evaluation: `outputs/semantic_caption/FULL_METRICS.csv` covers all 1997 test samples for 5 conditions x 5 modes.",
            "- Semantic fusion classifier: `outputs/semantic_caption/full_semantic_fusion/` is the full-data 7970/1998 run; the older `semantic_fusion_2k/` run is retained only as smoke/history.",
            "- Tri-modal alignment: current result is a 64/64 one-epoch smoke run; use it as a pipeline check, not efficacy evidence.",
            "- Spectrogram branch: P1/P2 completed as alignment smoke/search candidates; long P2 continuation is tracked separately under `outputs/alignment_long/` when available.",
            "- Subject adaptation/L7: dataset structure strongly supports the hypothesis, but paired metric evidence is mixed; do not claim a clean win yet.",
        ]
    )
    lines.extend(["", "## Current Sweep Best Alignment", ""])
    if current_best:
        lines.extend(
            [
                f"- Source: `{current_best['source']}`",
                f"- Experiment: `{current_best['experiment_id']}`",
                f"- Encoder: `{current_best['encoder']}`",
                f"- Loss: `{current_best['loss']}`",
                f"- Val R@5: `{current_best['val_r5']:.6f}`",
                f"- Test R@5: `{current_best['test_r5']}`",
                f"- Class accuracy: `{current_best['class_acc']:.6f}`",
            ]
        )
    else:
        lines.append("- No current alignment-search board found yet.")
    lines.extend(["", "## Historical Best Alignment", ""])
    if historical_best:
        lines.extend(
            [
                f"- Source: `{historical_best['source']}`",
                f"- Experiment: `{historical_best['experiment_id']}`",
                f"- Encoder: `{historical_best['encoder']}`",
                f"- Loss: `{historical_best['loss']}`",
                f"- Val R@5: `{historical_best['val_r5']:.6f}`",
                f"- Test R@5: `{historical_best['test_r5']}`",
                f"- Class accuracy: `{historical_best['class_acc']:.6f}`",
            ]
        )
    else:
        lines.append("- No historical alignment board found yet.")
    lines.extend(
        [
            "",
            "## Robustness Snapshot",
            "",
            "```csv",
            robustness.strip() if robustness else "pending",
            "```",
            "",
            "## Semantic Captioning",
            "",
            semantic.strip() if semantic else "Pending Subagent A. Free-form Qwen is not used as the primary evidence.",
            "",
            "## Dataset Preparation",
            "",
            dataset_rec.strip() if dataset_rec else "Pending dataset recommendation.",
            "",
            "## Tri-Modal",
            "",
            trimodal.strip() if trimodal else "Pending Subagent D.",
            "",
            "## Subject Adaptation",
            "",
            subject.strip() if subject else "Pending subject adaptation report.",
            "",
            "## Spectrogram Branch",
            "",
            spectrogram.strip() if spectrogram else "Pending spectrogram branch report.",
            "",
            "## Scientific Claim Status",
            "",
            "Allowed claim remains conditional: correctly paired EEG can improve constrained semantic prediction under degraded visual conditions only where real EEG beats shuffled/random controls.",
            "Do not claim open-ended caption generation is solved.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build parallel stage master report.")
    parser.add_argument("--out", default="outputs/parallel_stage/MASTER_REPORT.md")
    args = parser.parse_args()
    build_report(args.out)


if __name__ == "__main__":
    main()
