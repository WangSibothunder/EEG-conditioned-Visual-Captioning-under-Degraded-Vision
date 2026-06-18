from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


SOURCE_DIR = Path("outputs/semantic_caption")
ROBUSTNESS_DIR = Path("outputs/robustness")
OUT_DIR = Path("outputs/final_semantic")


def _copy_if_exists(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return True


def _read_gap_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    return sum(values) / max(1, len(values))


def _claim_boundary(total: int, real_beats_controls: int, degraded_count: int, degraded_controls: int, real_beats_vision: int) -> list[str]:
    if total == 0:
        return [
            "Not supported: no complete constrained semantic control rows were available.",
            "",
            "Re-run the full mode grid before making any paired-EEG semantic claim.",
        ]
    if real_beats_controls == total:
        claim = "Supported: paired real EEG carries class-level semantic information beyond shuffled/random controls across all evaluated conditions."
    elif real_beats_controls > 0:
        claim = (
            f"Limited: paired real EEG beats shuffled/random controls in `{real_beats_controls}/{total}` conditions "
            f"and `{degraded_controls}/{degraded_count}` degraded conditions; do not claim broad paired-EEG benefit."
        )
    else:
        claim = "Not supported: paired real EEG does not beat shuffled/random controls in the constrained semantic table."

    if real_beats_vision == total:
        vision_claim = "Supported by this table: real EEG improves over the vision-only prototype baseline in every evaluated condition."
    elif real_beats_vision > 0:
        vision_claim = (
            f"Limited: real EEG improves over vision-only in `{real_beats_vision}/{total}` conditions; "
            "do not claim a broad vision-only improvement."
        )
    else:
        vision_claim = "Not supported by the current constrained semantic table: real EEG improves over the strong vision-only CLIP prototype baseline."
    return [claim, "", vision_claim]


def _read_primary_summary(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def materialize_report(source_dir: Path, robustness_dir: Path, out_dir: Path, primary_summary: Path | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = {
        "FULL_METRICS.csv": _copy_if_exists(source_dir / "FULL_METRICS.csv", out_dir / "FULL_METRICS.csv"),
        "FULL_METRICS.md": _copy_if_exists(source_dir / "FULL_METRICS.md", out_dir / "FULL_METRICS.md"),
        "SEMANTIC_GAP_METRICS.csv": _copy_if_exists(source_dir / "SEMANTIC_GAP_METRICS.csv", out_dir / "SEMANTIC_GAP_METRICS.csv"),
        "SEMANTIC_GAP_METRICS.md": _copy_if_exists(source_dir / "SEMANTIC_GAP_METRICS.md", out_dir / "SEMANTIC_GAP_METRICS.md"),
        "qualitative_examples.md": _copy_if_exists(source_dir / "qualitative_examples.md", out_dir / "QUALITATIVE_EXAMPLES.md"),
        "ROBUSTNESS_REPORT.md": _copy_if_exists(robustness_dir / "ROBUSTNESS_REPORT.md", out_dir / "ROBUSTNESS_REPORT.md"),
    }
    rows = _read_gap_rows(out_dir / "SEMANTIC_GAP_METRICS.csv")
    real_beats_controls = sum(1 for row in rows if str(row.get("real_beats_controls", "")).lower() == "true")
    real_beats_vision = sum(1 for row in rows if str(row.get("real_beats_vision", "")).lower() == "true")
    total = len(rows)
    degraded = [row for row in rows if row.get("corruption") != "clean"]
    degraded_controls = sum(1 for row in degraded if str(row.get("real_beats_controls", "")).lower() == "true")
    mean_real_minus_vision = _mean(rows, "real_minus_vision") if rows else 0.0
    mean_real_minus_shuffled = _mean(rows, "real_minus_shuffled") if rows else 0.0
    mean_real_minus_random = _mean(rows, "real_minus_random") if rows else 0.0
    mean_top5_minus_vision = _mean(rows, "real_top5_minus_vision") if rows else 0.0
    claim_lines = _claim_boundary(
        total,
        real_beats_controls,
        len(degraded),
        degraded_controls,
        real_beats_vision,
    )
    if "eeg_imagenet_transfer_eval" in str(source_dir):
        next_experiment = (
            "This report already uses the EEG-ImageNet transfer semantic evaluation. "
            "Next, prefer THINGS raw-window transfer or paired EEG-ImageNet image training after ImageNet extraction completes."
        )
    else:
        next_experiment = (
            "If this report used the pre-transfer constrained-caption outputs, re-run it after the EEG-ImageNet pretraining transfer checkpoint is evaluated."
        )
    primary_text = _read_primary_summary(primary_summary)

    lines = [
        "# Full Robust Semantic Report",
        "",
        "This report materializes the constrained semantic-caption evaluation under the heavy-stage output path.",
        "It is not a free-form Qwen captioning claim.",
        "",
    ]
    if primary_text:
        lines.extend(
            [
                "## Primary Evidence",
                "",
                primary_text,
                "",
                "## Transfer Evaluation Kept As Secondary Evidence",
                "",
                "The transfer-evaluation table below is retained as a negative/limited result. It should not override the stronger A2 constrained-semantic evidence above.",
                "",
            ]
        )
    lines.extend(
        [
        f"- Source semantic directory: `{source_dir}`",
        f"- Source robustness directory: `{robustness_dir}`",
        "",
        "## Summary",
        "",
        f"- Conditions evaluated: `{total}`",
        f"- Real EEG beats shuffled/random controls: `{real_beats_controls}/{total}`",
        f"- Real EEG beats shuffled/random controls on degraded conditions: `{degraded_controls}/{len(degraded)}`",
        f"- Real EEG beats vision-only: `{real_beats_vision}/{total}`",
        f"- Mean real - vision top1 accuracy: `{mean_real_minus_vision:.6f}`",
        f"- Mean real - shuffled top1 accuracy: `{mean_real_minus_shuffled:.6f}`",
        f"- Mean real - random top1 accuracy: `{mean_real_minus_random:.6f}`",
        f"- Mean real - vision top5 accuracy: `{mean_top5_minus_vision:.6f}`",
        "",
        "## Claim Boundary",
        "",
        *claim_lines,
        "",
        "## Artifacts",
            "",
        ]
    )
    for name, ok in copied.items():
        status = "available" if ok else "missing"
        lines.append(f"- `{out_dir / name}`: `{status}`")
    lines.extend(
        [
            "",
            "## Next Required Experiment",
            "",
            next_experiment,
        ]
    )
    (out_dir / "FULL_ROBUST_SEMANTIC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize constrained semantic caption outputs into final report files.")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--robustness-dir", type=Path, default=ROBUSTNESS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--primary-summary", type=Path, default=None)
    args = parser.parse_args()
    materialize_report(args.source_dir, args.robustness_dir, args.out_dir, args.primary_summary)


if __name__ == "__main__":
    main()
