from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else 0.0


def _by_condition(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["corruption"]), str(row["mode"])): row for row in rows}


def compare(no_gate_csv: Path, gated_csv: Path, out: Path) -> None:
    no_gate = _by_condition(_read_csv(no_gate_csv))
    gated = _by_condition(_read_csv(gated_csv))
    corruptions = sorted({corruption for corruption, _mode in gated})
    rows: list[dict[str, Any]] = []
    for corruption in corruptions:
        no_real = no_gate.get((corruption, "real_eeg"), {})
        no_vision = no_gate.get((corruption, "vision_only"), {})
        gated_real = gated.get((corruption, "real_eeg"), {})
        gated_vision = gated.get((corruption, "vision_only"), {})
        gated_shuffled = gated.get((corruption, "shuffled_eeg"), {})
        gated_random = gated.get((corruption, "random_eeg"), {})
        rows.append(
            {
                "corruption": corruption,
                "no_gate_real_acc": _float(no_real, "accuracy"),
                "gated_real_acc": _float(gated_real, "accuracy"),
                "gated_minus_no_gate_real": _float(gated_real, "accuracy") - _float(no_real, "accuracy"),
                "no_gate_real_minus_vision": _float(no_real, "accuracy") - _float(no_vision, "accuracy"),
                "gated_real_minus_vision": _float(gated_real, "accuracy") - _float(gated_vision, "accuracy"),
                "gated_real_minus_shuffled": _float(gated_real, "accuracy") - _float(gated_shuffled, "accuracy"),
                "gated_real_minus_random": _float(gated_real, "accuracy") - _float(gated_random, "accuracy"),
                "gate_real": _float(gated_real, "gate_mean"),
                "gate_vision_only": _float(gated_vision, "gate_mean"),
                "gate_shuffled": _float(gated_shuffled, "gate_mean"),
                "gate_random": _float(gated_random, "gate_mean"),
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else []
    lines = ["# A2 Reliability Gate Ablation", ""]
    lines.append(f"- No-gate metrics: `{no_gate_csv}`")
    lines.append(f"- Gated metrics: `{gated_csv}`")
    lines.append("")
    if rows:
        real_wins_controls = sum(
            1
            for row in rows
            if row["gated_real_minus_shuffled"] > 0.0 and row["gated_real_minus_random"] > 0.0
        )
        real_wins_vision = sum(1 for row in rows if row["gated_real_minus_vision"] > 0.0)
        gated_beats_no_gate = sum(1 for row in rows if row["gated_minus_no_gate_real"] > 0.0)
        lines.extend(
            [
                f"- Gated real EEG beats shuffled/random: `{real_wins_controls}/{len(rows)}` conditions.",
                f"- Gated real EEG beats vision-only: `{real_wins_vision}/{len(rows)}` conditions.",
                f"- Gated real EEG beats no-gate real EEG: `{gated_beats_no_gate}/{len(rows)}` conditions.",
                "- Mechanism note: gate_mean is reported for audit, but a higher real-EEG gate under stronger degradation is required before claiming a learned reliability gate.",
                "",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
        )
        for row in rows:
            lines.append("| " + " | ".join(f"{row[h]:.4f}" if isinstance(row[h], float) else str(row[h]) for h in headers) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare semantic fusion reliability-gate and no-gate metrics.")
    parser.add_argument("--no_gate_csv", required=True)
    parser.add_argument("--gated_csv", required=True)
    parser.add_argument("--out", default="outputs/final_semantic/A2_GATE_VS_NOGATE_REPORT.md")
    args = parser.parse_args()
    compare(Path(args.no_gate_csv), Path(args.gated_csv), Path(args.out))


if __name__ == "__main__":
    main()
