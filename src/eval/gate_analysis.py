from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from src.eval.metrics import iter_prediction_jsonl_files


def load_prediction_records(pred_dir: str | Path) -> list[dict[str, Any]]:
    pred_dir = Path(pred_dir)
    records: list[dict[str, Any]] = []
    for path in iter_prediction_jsonl_files(pred_dir):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_gates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        corruption = str(record.get("corruption", "unknown"))
        mode = str(record.get("mode", "unknown"))
        groups[(corruption, mode)].append(record)

    rows: list[dict[str, Any]] = []
    for (corruption, mode), group in sorted(groups.items()):
        gate_values = [
            gate_value
            for record in group
            if (gate_value := _float_or_none(record.get("gate_mean"))) is not None
        ]
        rows.append(
            {
                "corruption": corruption,
                "mode": mode,
                "count": len(group),
                "gate_count": len(gate_values),
                "gate_mean": sum(gate_values) / len(gate_values) if gate_values else None,
            }
        )
    return rows


def write_gate_report(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Gate Analysis", ""]
    if rows:
        headers = ["corruption", "mode", "count", "gate_count", "gate_mean"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            values = []
            for header in headers:
                value = row[header]
                if value is None:
                    values.append("NA")
                elif isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
    else:
        lines.append("No prediction records found.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sample_predictions(path: str | Path, records: list[dict[str, Any]], limit: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = records[: max(0, limit)]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in selected),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize gated fusion values from sanity JSONL outputs.")
    parser.add_argument("--pred_dir", default=None)
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--out", default="outputs/day3/sanity_real/gate_analysis.md")
    parser.add_argument("--sample_out", default="outputs/day3/sanity_real/sample_predictions.jsonl")
    parser.add_argument("--sample_limit", type=int, default=10)
    args = parser.parse_args()

    pred_dir = args.pred_dir or args.input_dir
    if pred_dir is None:
        parser.error("one of --pred_dir or --input_dir is required")

    records = load_prediction_records(pred_dir)
    write_gate_report(args.out, summarize_gates(records))
    write_sample_predictions(args.sample_out, records, args.sample_limit)


if __name__ == "__main__":
    main()
