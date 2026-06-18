from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from statistics import mean, pstdev
from typing import Any


VARIANTS = [
    ("A_mse_ce_seed42", "A", "MSE + Class CE"),
    ("B_contrastive_seed42", "B", "InfoNCE + Cosine + Class CE"),
    ("C_simdistill_seed42", "C", "InfoNCE + Cosine + Class CE + Similarity Distillation"),
    ("D_full_seed42", "D", "InfoNCE + Cosine + Class CE + Similarity Distillation + Aug Consistency + Prototype Alignment"),
]


def _read_json(path: str | Path) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _best_epoch(run_dir: Path) -> int | str:
    history = _read_json(run_dir / "history.json")
    metrics = _read_json(run_dir / "alignment_metrics.json")
    if not isinstance(history, list) or not history:
        return "NA"
    best = max(history, key=lambda row: float(row.get("metrics", {}).get("r@5", -1)))
    return int(best.get("epoch", 0))


def write_ablation_report(root: str | Path) -> tuple[str | None, dict[str, Any] | None]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    best_name: str | None = None
    best_payload: dict[str, Any] | None = None
    best_r5 = -1.0
    for dirname, variant, terms in VARIANTS:
        run_dir = root / dirname
        payload = _read_json(run_dir / "alignment_metrics.json") or _read_json(run_dir / "retrieval_metrics.json")
        if not isinstance(payload, dict):
            rows.append({"variant": variant, "terms": terms, "missing": True})
            continue
        model = payload.get("model", {})
        random = payload.get("random", {})
        r5 = float(model.get("r@5", 0.0) or 0.0)
        if r5 > best_r5 and (run_dir / "checkpoints" / "best.pt").exists():
            best_r5 = r5
            best_name = dirname
            best_payload = payload
        rows.append(
            {
                "variant": variant,
                "terms": terms,
                "r@1": model.get("r@1"),
                "r@5": model.get("r@5"),
                "r@10": model.get("r@10"),
                "class_acc": model.get("class_acc"),
                "mean_rank": model.get("mean_rank"),
                "random_r@5": random.get("r@5"),
                "best_epoch": _best_epoch(run_dir),
                "missing": False,
            }
        )

    lines = [
        "# Alignment Ablation Report",
        "",
        "| Variant | Loss Terms | R@1 | R@5 | R@10 | Class Acc | Mean Rank | Random R@5 | Best Epoch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row.get("missing"):
            lines.append(f"| {row['variant']} | {row['terms']} | NA | NA | NA | NA | NA | NA | NA |")
        else:
            lines.append(
                f"| {row['variant']} | {row['terms']} | {_fmt(row.get('r@1'))} | {_fmt(row.get('r@5'))} | "
                f"{_fmt(row.get('r@10'))} | {_fmt(row.get('class_acc'))} | {_fmt(row.get('mean_rank'), 2)} | "
                f"{_fmt(row.get('random_r@5'))} | {row.get('best_epoch')} |"
            )
    lines.extend(["", "## Best Variant", ""])
    lines.append(f"- Best seed42 run by test R@5: `{best_name or 'not available'}`")
    (root / "ALIGNMENT_ABLATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return best_name, best_payload


def write_multiseed_summary(root: str | Path, best_prefix: str = "best") -> None:
    root = Path(root)
    seed_dirs = sorted(path for path in root.glob(f"{best_prefix}_seed*") if path.is_dir())
    rows: list[dict[str, Any]] = []
    for run_dir in seed_dirs:
        payload = _read_json(run_dir / "alignment_metrics.json") or _read_json(run_dir / "retrieval_metrics.json")
        if not isinstance(payload, dict):
            continue
        model = payload.get("model", {})
        rows.append(
            {
                "seed": run_dir.name.replace(f"{best_prefix}_seed", ""),
                "r@1": model.get("r@1"),
                "r@5": model.get("r@5"),
                "r@10": model.get("r@10"),
                "class_acc": model.get("class_acc"),
                "mean_rank": model.get("mean_rank"),
                "best_epoch": _best_epoch(run_dir),
                "run_dir": run_dir,
            }
        )

    lines = [
        "# Multi-seed Alignment Summary",
        "",
        "| Seed | R@1 | R@5 | R@10 | Class Acc | Mean Rank | Best Epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {_fmt(row.get('r@1'))} | {_fmt(row.get('r@5'))} | {_fmt(row.get('r@10'))} | "
            f"{_fmt(row.get('class_acc'))} | {_fmt(row.get('mean_rank'), 2)} | {row.get('best_epoch')} |"
        )
    if rows:
        for key in ["r@1", "r@5", "r@10", "class_acc", "mean_rank"]:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            if values:
                lines.append(f"- {key} mean +/- std: `{mean(values):.4f} +/- {pstdev(values):.4f}`")
        best_row = max(rows, key=lambda row: float(row.get("r@5", 0.0) or 0.0))
        source = Path(best_row["run_dir"]) / "checkpoints" / "best.pt"
        if source.exists():
            shutil.copy2(source, root / "best_overall.pt")
            lines.append(f"- Best overall checkpoint: `{root / 'best_overall.pt'}`")
    else:
        lines.append("- No multi-seed runs found yet.")
    (root / "multiseed_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Day4 alignment ablation and multiseed reports.")
    parser.add_argument("--root", default="outputs/day4_alignment")
    args = parser.parse_args()
    write_ablation_report(args.root)
    write_multiseed_summary(args.root)
    print(args.root)


if __name__ == "__main__":
    main()
