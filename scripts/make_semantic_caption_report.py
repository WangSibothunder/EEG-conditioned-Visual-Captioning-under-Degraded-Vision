from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CORRUPTIONS = ["clean", "blur", "occlusion", "noise", "lowres"]
MODES = ["vision_only", "real_eeg", "shuffled_eeg", "random_eeg", "eeg_only"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0.0)
    except ValueError:
        return 0.0


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
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


def _md_table(rows: list[dict[str, Any]]) -> list[str]:
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


def _read_jsonl(path: Path, limit: int = 3) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def _boolish(value: Any) -> bool:
    return str(value).lower() == "true"


def _ordered_corruptions(rows: list[dict[str, str]]) -> list[str]:
    seen = {str(row.get("corruption", "")) for row in rows if row.get("corruption")}
    ordered = [name for name in CORRUPTIONS if name in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def _control_claim_lines(gap_rows: list[dict[str, Any]], degraded: list[dict[str, Any]]) -> tuple[list[str], str]:
    total = len(gap_rows)
    control_wins = sum(1 for row in gap_rows if _boolish(row["real_beats_controls"]))
    degraded_wins = sum(1 for row in degraded if _boolish(row["real_beats_controls"]))
    top5_control_wins = sum(
        1
        for row in gap_rows
        if float(row["real_top5_minus_shuffled"]) > 0 and float(row["real_top5_minus_random"]) > 0
    )
    vision_wins = sum(1 for row in gap_rows if _boolish(row["real_beats_vision"]))

    if total == 0:
        return (
            [
                "- No complete real/shuffled/random/vision/eeg-only condition rows were available, so no EEG-specific caption claim can be made.",
                "- Re-run the constrained semantic evaluation with the full mode grid before interpreting EEG benefit.",
            ],
            "Not supported: missing complete control rows for paired EEG, shuffled EEG, random EEG, and vision-only.",
        )

    if control_wins == total:
        top5_note = (
            " Top-5 control wins are also uniform."
            if top5_control_wins == total
            else f" Top-5 control wins are `{top5_control_wins}/{total}`, so top-5 evidence should be reported separately."
        )
        interpretation = [
            "- Paired real EEG beats both shuffled and random EEG controls in every evaluated condition."
            + top5_note,
            "- The defensible claim is control-specific semantic signal, not overall improvement over a strong clean/degraded CLIP prototype baseline.",
        ]
        claim = "Supported: correctly paired EEG carries semantic information beyond shuffled/random EEG controls across all evaluated conditions."
    elif control_wins > 0:
        interpretation = [
            f"- Limited: paired real EEG beats both shuffled and random EEG controls in `{control_wins}/{total}` conditions and `{degraded_wins}/{len(degraded)}` degraded conditions.",
            "- This is mixed evidence, so report the exact gaps instead of claiming a broad EEG-specific caption benefit.",
        ]
        claim = (
            f"Limited: correctly paired EEG beats shuffled/random controls in `{control_wins}/{total}` conditions; "
            "do not claim broad paired-EEG semantic benefit from this table."
        )
    else:
        interpretation = [
            "- Not supported: paired real EEG does not beat both shuffled and random EEG controls in any evaluated condition.",
            "- This table should be treated as a negative or failed transfer result for paired-EEG semantic captioning.",
        ]
        claim = "Not supported: correctly paired EEG does not beat shuffled/random EEG controls in this table."

    if vision_wins == total:
        interpretation.append("- Real EEG also beats vision-only in every evaluated condition.")
    elif vision_wins > 0:
        interpretation.append(
            f"- Real EEG beats vision-only in `{vision_wins}/{total}` conditions; this is not enough for a broad vision-improvement claim."
        )
    else:
        interpretation.append(
            "- Real EEG does not beat vision-only in this prototype setup, so no vision-improvement claim is supported."
        )
    return interpretation, claim


def build_report(metrics_path: Path, output_dir: Path) -> None:
    rows = _read_csv(metrics_path)
    by_key = {(row["corruption"], row["mode"]): row for row in rows}
    corruptions = _ordered_corruptions(rows)
    gap_rows: list[dict[str, Any]] = []
    for corruption in corruptions:
        real = by_key.get((corruption, "real_eeg"))
        vision = by_key.get((corruption, "vision_only"))
        shuffled = by_key.get((corruption, "shuffled_eeg"))
        random = by_key.get((corruption, "random_eeg"))
        eeg_only = by_key.get((corruption, "eeg_only"))
        if not all([real, vision, shuffled, random, eeg_only]):
            continue
        real_acc = _float(real, "accuracy")
        real_top5 = _float(real, "top5_accuracy")
        vision_acc = _float(vision, "accuracy")
        vision_top5 = _float(vision, "top5_accuracy")
        shuffled_acc = _float(shuffled, "accuracy")
        shuffled_top5 = _float(shuffled, "top5_accuracy")
        random_acc = _float(random, "accuracy")
        random_top5 = _float(random, "top5_accuracy")
        eeg_only_acc = _float(eeg_only, "accuracy")
        eeg_only_top5 = _float(eeg_only, "top5_accuracy")
        gap_rows.append(
            {
                "corruption": corruption,
                "vision_only_acc": vision_acc,
                "vision_only_top5": vision_top5,
                "real_eeg_acc": real_acc,
                "real_eeg_top5": real_top5,
                "shuffled_eeg_acc": shuffled_acc,
                "shuffled_eeg_top5": shuffled_top5,
                "random_eeg_acc": random_acc,
                "random_eeg_top5": random_top5,
                "eeg_only_acc": eeg_only_acc,
                "eeg_only_top5": eeg_only_top5,
                "real_minus_vision": real_acc - vision_acc,
                "real_minus_shuffled": real_acc - shuffled_acc,
                "real_minus_random": real_acc - random_acc,
                "real_top5_minus_vision": real_top5 - vision_top5,
                "real_top5_minus_shuffled": real_top5 - shuffled_top5,
                "real_top5_minus_random": real_top5 - random_top5,
                "real_beats_controls": real_acc > shuffled_acc and real_acc > random_acc,
                "real_beats_vision": real_acc > vision_acc,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "SEMANTIC_GAP_METRICS.csv", gap_rows)
    (output_dir / "SEMANTIC_GAP_METRICS.md").write_text(
        "# Semantic Caption Gap Metrics\n\n" + "\n".join(_md_table(gap_rows)) + "\n",
        encoding="utf-8",
    )

    degraded = [row for row in gap_rows if row["corruption"] != "clean"]
    real_control_win_rate = (
        sum(1 for row in gap_rows if row["real_beats_controls"]) / len(gap_rows) if gap_rows else 0.0
    )
    degraded_control_win_rate = (
        sum(1 for row in degraded if row["real_beats_controls"]) / len(degraded) if degraded else 0.0
    )
    mean_real_minus_vision = sum(float(row["real_minus_vision"]) for row in gap_rows) / max(1, len(gap_rows))
    mean_real_minus_shuffled = sum(float(row["real_minus_shuffled"]) for row in gap_rows) / max(1, len(gap_rows))
    mean_real_minus_random = sum(float(row["real_minus_random"]) for row in gap_rows) / max(1, len(gap_rows))
    mean_real_top5_minus_vision = sum(float(row["real_top5_minus_vision"]) for row in gap_rows) / max(1, len(gap_rows))
    mean_real_top5_minus_shuffled = sum(float(row["real_top5_minus_shuffled"]) for row in gap_rows) / max(1, len(gap_rows))
    mean_real_top5_minus_random = sum(float(row["real_top5_minus_random"]) for row in gap_rows) / max(1, len(gap_rows))
    best_degraded = max(degraded, key=lambda row: row["real_eeg_acc"], default=None)
    worst_degraded = min(degraded, key=lambda row: row["real_eeg_acc"], default=None)
    interpretation_lines, claim_rule = _control_claim_lines(gap_rows, degraded)

    examples: list[str] = ["# Semantic Caption Qualitative Examples", ""]
    example_corruptions = [name for name in corruptions if name != "clean"][:2]
    for corruption in example_corruptions:
        examples.append(f"## {corruption}")
        examples.append("")
        for mode in ["vision_only", "real_eeg", "shuffled_eeg", "random_eeg"]:
            path = output_dir / f"{corruption}_{mode}.jsonl"
            records = _read_jsonl(path, limit=2)
            examples.append(f"### {mode}")
            examples.append("")
            for record in records:
                examples.append(
                    f"- image `{record.get('image_id')}` true `{record.get('human_label_name')}` "
                    f"pred `{record.get('pred_class_name')}` caption `{record.get('prediction')}`"
                )
            if not records:
                examples.append("- No examples available.")
            examples.append("")
    (output_dir / "qualitative_examples.md").write_text("\n".join(examples), encoding="utf-8")

    semantic_fusion_report = output_dir / "semantic_fusion_2k" / "semantic_fusion_train_report.md"
    smoke_text = semantic_fusion_report.read_text(encoding="utf-8") if semantic_fusion_report.exists() else "Not run."
    lines = [
        "# Controlled Semantic Caption Report",
        "",
        "## Decision",
        "",
        "Use constrained class-level captions as the primary captioning evidence for this small 40-class EEG dataset.",
        "The generated caption is deterministic: `predicted_class -> a photo of a {class_name}`.",
        "Free-form Qwen generation remains a qualitative failure case, not the main metric.",
        "",
        "## Full Grid Summary",
        "",
        f"- Conditions evaluated: `{len(corruptions)}`",
        f"- Modes evaluated: `{', '.join(MODES)}`",
        f"- Test samples per condition/mode: `{rows[0].get('count', 'NA') if rows else 'NA'}`",
        f"- Real EEG beats shuffled/random controls: `{sum(1 for row in gap_rows if row['real_beats_controls'])}/{len(gap_rows)}`",
        f"- Real EEG beats shuffled/random controls on degraded only: `{sum(1 for row in degraded if row['real_beats_controls'])}/{len(degraded)}`",
        f"- Mean real - vision accuracy: `{mean_real_minus_vision:.6f}`",
        f"- Mean real - shuffled accuracy: `{mean_real_minus_shuffled:.6f}`",
        f"- Mean real - random accuracy: `{mean_real_minus_random:.6f}`",
        f"- Mean real - vision top5: `{mean_real_top5_minus_vision:.6f}`",
        f"- Mean real - shuffled top5: `{mean_real_top5_minus_shuffled:.6f}`",
        f"- Mean real - random top5: `{mean_real_top5_minus_random:.6f}`",
        f"- Best degraded real EEG accuracy: `{best_degraded['corruption'] if best_degraded else 'NA'} = {best_degraded['real_eeg_acc']:.6f}`" if best_degraded else "- Best degraded real EEG accuracy: `NA`",
        f"- Worst degraded real EEG accuracy: `{worst_degraded['corruption'] if worst_degraded else 'NA'} = {worst_degraded['real_eeg_acc']:.6f}`" if worst_degraded else "- Worst degraded real EEG accuracy: `NA`",
        "",
        "## Main Interpretation",
        "",
        "- The constrained decoder produces stable class-level captions and removes URL/code-like free-form failures.",
        "- Vision-only CLIP prototype classification is very strong, so paired EEG does not beat vision-only in this simple additive prototype setup.",
        *interpretation_lines,
        "",
        "## Gap Table",
        "",
        *_md_table(gap_rows),
        "",
        "## Semantic Fusion Classifier",
        "",
        smoke_text.strip(),
        "",
        "## Artifacts",
        "",
        "- `outputs/semantic_caption/prototypes.pt`",
        "- `outputs/semantic_caption/FULL_METRICS.csv`",
        "- `outputs/semantic_caption/FULL_METRICS.md`",
        "- `outputs/semantic_caption/SEMANTIC_GAP_METRICS.csv`",
        "- `outputs/semantic_caption/SEMANTIC_GAP_METRICS.md`",
        "- `outputs/semantic_caption/qualitative_examples.md`",
        "",
        "## Claim Rule",
        "",
        claim_rule,
        "Not supported by this table: real EEG improves over a strong vision-only CLIP prototype classifier.",
        "",
        f"Control win rates: all `{real_control_win_rate:.3f}`, degraded `{degraded_control_win_rate:.3f}`.",
    ]
    (output_dir / "SEMANTIC_CAPTION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir / "SEMANTIC_CAPTION_REPORT.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize controlled semantic caption metrics.")
    parser.add_argument("--metrics", default="outputs/semantic_caption/FULL_METRICS.csv")
    parser.add_argument("--output_dir", default="outputs/semantic_caption")
    args = parser.parse_args()
    build_report(Path(args.metrics), Path(args.output_dir))


if __name__ == "__main__":
    main()
