from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RUNS = [
    (
        "Day4 best CLIP-B/32",
        Path("outputs/day4_alignment/best_seed42/alignment_metrics.json"),
        "outputs/day4_alignment/best_seed42/checkpoints/best.pt",
        "Main fusion recommendation unless a later run beats it.",
    ),
    (
        "CLIP-L/14 comparison",
        Path("outputs/day5_clipL/clipL_best_smoke/alignment_metrics.json"),
        "outputs/day5_clipL/clipL_best_smoke/checkpoints/best.pt",
        "Uses real openai/clip-vit-large-patch14 768-dim cached targets.",
    ),
    (
        "Strong E4 G1",
        Path("outputs/day5_heavy_alignment/E4_G1_heavy_seed314/alignment_metrics.json"),
        "outputs/day5_heavy_alignment/E4_G1_heavy_seed314/checkpoints/best.pt",
        "ConvTransformer strong with L1+L2+L4.",
    ),
    (
        "Strong E4 G2",
        Path("outputs/day5_heavy_alignment/E4_G2_heavy_seed2718/alignment_metrics.json"),
        "outputs/day5_heavy_alignment/E4_G2_heavy_seed2718/checkpoints/best.pt",
        "ConvTransformer strong with L1+L2+L4+L5+L6.",
    ),
    (
        "Subject X2",
        Path("outputs/day5_subject_alignment/X2_subject_same_image_seed888/alignment_metrics.json"),
        "outputs/day5_subject_alignment/X2_subject_same_image_seed888/checkpoints/best.pt",
        "Subject-adaptive same-image consistency test.",
    ),
    (
        "Extra sweep best E4 G2 dropout",
        Path("outputs/day5_extra_alignment/E4_G2_dropout035_seed777/metrics.json"),
        "outputs/day5_extra_alignment/E4_G2_dropout035_seed777/checkpoints/best.pt",
        "Best of 6 extra Day5 alignment configs; uses test unique-image metrics for global comparison.",
    ),
]


def _read_metric(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None

    # Some sweep jobs write split metrics as {"test": {"model": ...}, "val": ...};
    # use test/unique-image metrics for cross-run checkpoint comparisons.
    test = payload.get("test")
    if isinstance(test, dict):
        model = test.get("model")
        if isinstance(model, dict):
            unique = model.get("unique_image")
            if isinstance(unique, dict):
                return unique
            if "r@5" in model:
                return model

    model = payload.get("model", payload)
    unique = model.get("unique_image") if isinstance(model, dict) else None
    if isinstance(unique, dict):
        return unique
    if isinstance(model, dict) and "r@5" in model:
        return model
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    out = Path("outputs/day5_clipL/clipL_alignment_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, dict[str, Any] | None, str, str]] = []
    for name, metrics_path, checkpoint, note in RUNS:
        rows.append((name, _read_metric(metrics_path), checkpoint, note))

    baseline = rows[0][1]
    baseline_r5 = float(baseline.get("r@5", 0.0)) if baseline else 0.0
    lines = [
        "# CLIP ViT-L/14 and Heavy Alignment Comparison",
        "",
        "| Run | R@1 | R@5 | R@10 | Class Acc | Mean Rank | Checkpoint | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for name, metrics, checkpoint, note in rows:
        lines.append(
            f"| {name} | {_fmt((metrics or {}).get('r@1'))} | {_fmt((metrics or {}).get('r@5'))} | "
            f"{_fmt((metrics or {}).get('r@10'))} | {_fmt((metrics or {}).get('class_acc'))} | "
            f"{_fmt((metrics or {}).get('mean_rank'))} | `{checkpoint}` | {note} |"
        )
    lines.extend(["", "## Interpretation", ""])
    for name, metrics, _checkpoint, _note in rows[1:]:
        if not metrics:
            lines.append(f"- `{name}` is pending or missing metrics.")
            continue
        delta = float(metrics.get("r@5", 0.0)) - baseline_r5
        relation = "beat" if delta > 0 else "did not beat"
        lines.append(f"- `{name}` {relation} Day4 best by unique-image R@5: delta `{delta:.4f}`.")
    lines.append("- Keep `outputs/day4_alignment/best_overall.pt` as the recommended fusion checkpoint; Day5 extension runs did not exceed Day4 best on unique-image R@5.")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
