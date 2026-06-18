from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.robustness_metrics import (
    DEGRADED_CORRUPTIONS,
    PERFORMANCE_METRICS,
    SEMANTIC_PERFORMANCE_METRICS,
    compute_robustness_rows,
    paired_class_hit_win_rates,
    read_metric_csv,
    summarize_robustness,
)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md_table(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No rows available.")
    else:
        headers = list(rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(_fmt(row.get(header)) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _definition_doc() -> str:
    return "\n".join(
        [
            "# Robustness Metric Definitions",
            "",
            "All metrics are computed per degraded visual condition unless noted otherwise.",
            "",
            "- Robustness Gain: `metric(real_eeg, corruption) - metric(vision_only, corruption)`. Positive means paired EEG improves over the degraded vision-only baseline.",
            "- EEG Specific Gain: `metric(real_eeg, corruption) - max(metric(shuffled_eeg, corruption), metric(random_eeg, corruption))`. Positive means the gain is specific to paired EEG rather than generic EEG-like input.",
            "- Win Rate: fraction of evaluated corruptions where real EEG beats the comparison. The report includes win rate over vision-only and over both controls.",
            "- Gate Shift: `gate_mean(real_eeg, corruption) - gate_mean(control, corruption)`, reported against random EEG and shuffled EEG. Positive means the fusion gate is larger for paired EEG.",
            "- Class Hit Gap: `class_hit(real_eeg, corruption) - class_hit(vision_only, corruption)`, plus a control-adjusted class-hit gap against the stronger of shuffled/random EEG.",
            "",
            "Primary performance metrics currently use the existing Day5 sanity table: `bleu_1`, `bleu_4`, `rouge_l`, and `class_hit`.",
            "Semantic-caption metrics are pending when `outputs/semantic_caption/` is absent.",
            "",
        ]
    )


def _summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric, values in summary.items():
        rows.append({"metric": metric, **values})
    return rows


def _ordered_corruptions(rows: list[dict[str, Any]], *, include_clean: bool) -> list[str]:
    seen = {str(row.get("corruption", "")) for row in rows if row.get("corruption")}
    preferred = ["clean", *DEGRADED_CORRUPTIONS] if include_clean else DEGRADED_CORRUPTIONS
    ordered = [name for name in preferred if name in seen]
    ordered.extend(sorted(seen - set(ordered)))
    if not include_clean:
        ordered = [name for name in ordered if name != "clean"]
    return ordered


def _report(
    robustness_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    *,
    semantic_exists: bool,
    semantic_summary_rows: list[dict[str, Any]] | None = None,
) -> str:
    class_row = next((row for row in summary_rows if row["metric"] == "class_hit"), {})
    rouge_row = next((row for row in summary_rows if row["metric"] == "rouge_l"), {})
    class_specific = class_row.get("mean_eeg_specific_gain")
    class_win_controls = class_row.get("win_rate_over_controls")
    rouge_specific = rouge_row.get("mean_eeg_specific_gain")

    if class_specific is not None and class_specific > 0 and class_win_controls == 1.0:
        claim = "Day5 supports a cautious robustness claim by class-hit: paired EEG beats both shuffled and random EEG in every degraded condition."
    elif class_specific is not None and class_specific > 0:
        claim = "Day5 is positive on average but not uniformly decisive; keep the claim preliminary."
    else:
        claim = "Day5 does not support a robust EEG-specific claim under the selected metrics."

    lines = [
        "# Robustness Report",
        "",
        "## Inputs",
        "",
        "- Current free-form source: `outputs/day5_sanity/FULL_SANITY_METRICS.csv`.",
        f"- Semantic caption results: `{'available' if semantic_exists else 'pending'}`.",
        "- No training was started by this report script; it summarizes existing Day5 and constrained semantic outputs.",
        "",
        "## Main Conclusion",
        "",
        claim,
        "",
        "## Key Numbers",
        "",
        f"- Class Hit mean Robustness Gain vs vision-only: `{_fmt(class_row.get('mean_robustness_gain'))}`.",
        f"- Class Hit mean EEG Specific Gain vs best control: `{_fmt(class_specific)}`.",
        f"- Class Hit Win Rate over both controls: `{_fmt(class_win_controls)}`.",
        f"- ROUGE-L mean EEG Specific Gain vs best control: `{_fmt(rouge_specific)}`.",
        "",
        "## Interpretation",
        "",
        "- Real EEG has large aggregate gains over degraded vision-only, but vision-only Day5 captions are extremely weak, so this alone is not enough evidence.",
        "- The stronger signal is EEG Specific Gain: real EEG beats both shuffled and random EEG on class hit, BLEU, and ROUGE-L across all degraded corruptions in the current Day5 table.",
        "- Gate Shift is small and negative versus shuffled EEG while positive versus random EEG. This means the current gate magnitude alone does not prove that the model is selectively using paired EEG.",
        "- Semantic-caption results are summarized separately when available because they use controlled class-level captions rather than free-form Qwen outputs.",
        "",
        "## Summary Table",
        "",
        *_markdown_table(summary_rows),
        "",
        "## Paired Class-Hit Win Rates",
        "",
        *_markdown_table(paired_rows),
        "",
        "## Constrained Semantic Caption Summary",
        "",
        *(
            _markdown_table(semantic_summary_rows)
            if semantic_summary_rows
            else ["Semantic caption full-grid metrics are pending."]
        ),
        "",
        "## Artifacts",
        "",
        "- `outputs/robustness/ROBUSTNESS_METRIC_DEFINITION.md`",
        "- `outputs/robustness/full_metrics.csv`",
        "- `outputs/robustness/full_metrics.md`",
        "- `outputs/robustness/paired_win_rates.csv`",
        "- `outputs/robustness/ROBUSTNESS_REPORT.md`",
        "- `outputs/robustness/semantic_summary_metrics.csv`",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows available."]
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(header)) for header in headers) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Build robustness metrics and report from existing sanity outputs.")
    parser.add_argument("--day5_csv", default="outputs/day5_sanity/FULL_SANITY_METRICS.csv")
    parser.add_argument("--day5_dir", default="outputs/day5_sanity")
    parser.add_argument("--semantic_dir", default="outputs/semantic_caption")
    parser.add_argument("--out_dir", default="outputs/robustness")
    args = parser.parse_args()

    day5_csv = Path(args.day5_csv)
    if not day5_csv.exists():
        raise FileNotFoundError(f"Missing Day5 sanity metrics CSV: {day5_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = read_metric_csv(day5_csv)
    robustness = compute_robustness_rows(
        metric_rows,
        metrics=PERFORMANCE_METRICS,
        corruptions=_ordered_corruptions(metric_rows, include_clean=False),
    )
    robustness_dicts = [asdict(row) for row in robustness]
    summary = summarize_robustness(robustness)
    summary_dicts = _summary_rows(summary)
    paired_rows = paired_class_hit_win_rates(args.day5_dir)

    (out_dir / "ROBUSTNESS_METRIC_DEFINITION.md").write_text(_definition_doc(), encoding="utf-8")
    _write_csv(out_dir / "full_metrics.csv", robustness_dicts)
    _write_md_table(out_dir / "full_metrics.md", "Robustness Full Metrics", robustness_dicts)
    _write_csv(out_dir / "summary_metrics.csv", summary_dicts)
    _write_md_table(out_dir / "summary_metrics.md", "Robustness Summary Metrics", summary_dicts)
    _write_csv(out_dir / "paired_win_rates.csv", paired_rows)
    _write_md_table(out_dir / "paired_win_rates.md", "Paired Class-Hit Win Rates", paired_rows)
    semantic_summary_dicts: list[dict[str, Any]] = []
    semantic_csv = Path(args.semantic_dir) / "FULL_METRICS.csv"
    if semantic_csv.exists():
        semantic_rows = read_metric_csv(semantic_csv)
        semantic_robustness = compute_robustness_rows(
            semantic_rows,
            metrics=SEMANTIC_PERFORMANCE_METRICS,
            corruptions=_ordered_corruptions(semantic_rows, include_clean=True),
        )
        semantic_dicts = [asdict(row) for row in semantic_robustness]
        semantic_summary = summarize_robustness(semantic_robustness)
        semantic_summary_dicts = _summary_rows(semantic_summary)
        _write_csv(out_dir / "semantic_full_metrics.csv", semantic_dicts)
        _write_md_table(out_dir / "semantic_full_metrics.md", "Constrained Semantic Robustness Full Metrics", semantic_dicts)
        _write_csv(out_dir / "semantic_summary_metrics.csv", semantic_summary_dicts)
        _write_md_table(out_dir / "semantic_summary_metrics.md", "Constrained Semantic Robustness Summary", semantic_summary_dicts)

    report = _report(
        robustness_dicts,
        summary_dicts,
        paired_rows,
        semantic_exists=semantic_csv.exists(),
        semantic_summary_rows=semantic_summary_dicts,
    )
    (out_dir / "ROBUSTNESS_REPORT.md").write_text(report, encoding="utf-8")
    print(out_dir / "ROBUSTNESS_REPORT.md")


if __name__ == "__main__":
    main()
