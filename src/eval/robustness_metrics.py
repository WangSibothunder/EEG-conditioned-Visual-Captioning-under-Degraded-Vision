from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any


PERFORMANCE_METRICS = ["bleu_1", "bleu_4", "rouge_l", "class_hit"]
SEMANTIC_PERFORMANCE_METRICS = ["accuracy", "top5_accuracy", "bleu_1", "bleu_4", "rouge_l"]
REQUIRED_MODES = ["vision_only", "real_eeg", "shuffled_eeg", "random_eeg"]
DEGRADED_CORRUPTIONS = ["blur", "occlusion", "noise", "lowres"]


@dataclass(frozen=True)
class RobustnessRow:
    corruption: str
    metric: str
    vision_only: float | None
    real_eeg: float | None
    shuffled_eeg: float | None
    random_eeg: float | None
    robustness_gain: float | None
    eeg_specific_gain: float | None
    win_real_over_vision: bool | None
    win_real_over_controls: bool | None
    gate_shift_random: float | None
    gate_shift_shuffled: float | None
    class_hit_gap: float | None
    class_hit_specific_gap: float | None


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_metric_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def index_metric_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        corruption = str(row.get("corruption", ""))
        mode = str(row.get("mode", ""))
        if corruption and mode:
            indexed[corruption][mode] = row
    return dict(indexed)


def value(indexed: dict[str, dict[str, dict[str, Any]]], corruption: str, mode: str, metric: str) -> float | None:
    return parse_float(indexed.get(corruption, {}).get(mode, {}).get(metric))


def delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def gt(left: float | None, right: float | None) -> bool | None:
    if left is None or right is None:
        return None
    return left > right


def compute_robustness_rows(
    metric_rows: list[dict[str, Any]],
    *,
    metrics: list[str] | None = None,
    corruptions: list[str] | None = None,
) -> list[RobustnessRow]:
    indexed = index_metric_rows(metric_rows)
    metrics = metrics or PERFORMANCE_METRICS
    corruptions = corruptions or sorted(c for c in indexed if c != "clean")

    rows: list[RobustnessRow] = []
    for corruption in corruptions:
        for metric in metrics:
            vision = value(indexed, corruption, "vision_only", metric)
            real = value(indexed, corruption, "real_eeg", metric)
            shuffled = value(indexed, corruption, "shuffled_eeg", metric)
            random = value(indexed, corruption, "random_eeg", metric)
            control_values = [v for v in [shuffled, random] if v is not None]
            best_control = max(control_values) if control_values else None

            class_real = value(indexed, corruption, "real_eeg", "class_hit")
            class_vision = value(indexed, corruption, "vision_only", "class_hit")
            class_controls = [
                v
                for v in [
                    value(indexed, corruption, "shuffled_eeg", "class_hit"),
                    value(indexed, corruption, "random_eeg", "class_hit"),
                ]
                if v is not None
            ]
            class_best_control = max(class_controls) if class_controls else None

            rows.append(
                RobustnessRow(
                    corruption=corruption,
                    metric=metric,
                    vision_only=vision,
                    real_eeg=real,
                    shuffled_eeg=shuffled,
                    random_eeg=random,
                    robustness_gain=delta(real, vision),
                    eeg_specific_gain=delta(real, best_control),
                    win_real_over_vision=gt(real, vision),
                    win_real_over_controls=(
                        None
                        if real is None or shuffled is None or random is None
                        else real > shuffled and real > random
                    ),
                    gate_shift_random=delta(
                        value(indexed, corruption, "real_eeg", "gate_mean"),
                        value(indexed, corruption, "random_eeg", "gate_mean"),
                    ),
                    gate_shift_shuffled=delta(
                        value(indexed, corruption, "real_eeg", "gate_mean"),
                        value(indexed, corruption, "shuffled_eeg", "gate_mean"),
                    ),
                    class_hit_gap=delta(class_real, class_vision),
                    class_hit_specific_gap=delta(class_real, class_best_control),
                )
            )
    return rows


def summarize_robustness(rows: list[RobustnessRow]) -> dict[str, Any]:
    by_metric: dict[str, list[RobustnessRow]] = defaultdict(list)
    for row in rows:
        by_metric[row.metric].append(row)

    summary: dict[str, Any] = {}
    for metric, metric_rows in sorted(by_metric.items()):
        valid = [row for row in metric_rows if row.eeg_specific_gain is not None]
        control_wins = [row for row in metric_rows if row.win_real_over_controls is not None]
        vision_wins = [row for row in metric_rows if row.win_real_over_vision is not None]
        summary[metric] = {
            "mean_robustness_gain": _mean(row.robustness_gain for row in valid),
            "mean_eeg_specific_gain": _mean(row.eeg_specific_gain for row in valid),
            "win_rate_over_controls": _rate(row.win_real_over_controls for row in control_wins),
            "win_rate_over_vision": _rate(row.win_real_over_vision for row in vision_wins),
            "n_conditions": len(metric_rows),
        }
    return summary


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def paired_class_hit_win_rates(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    outputs: list[dict[str, Any]] = []
    for corruption in DEGRADED_CORRUPTIONS:
        mode_records: dict[str, dict[str, dict[str, Any]]] = {}
        for mode in REQUIRED_MODES:
            path = root / f"{corruption}_{mode}.jsonl"
            if not path.exists():
                continue
            rows = read_jsonl(path)
            mode_records[mode] = {str(row.get("image_id")): row for row in rows}
        if not all(mode in mode_records for mode in REQUIRED_MODES):
            outputs.append({"corruption": corruption, "status": "missing_jsonl"})
            continue

        image_ids = set.intersection(*(set(records) for records in mode_records.values()))
        real_vs_vision = []
        real_vs_best_control = []
        for image_id in image_ids:
            hits = {mode: _record_class_hit(mode_records[mode][image_id]) for mode in REQUIRED_MODES}
            if any(hit is None for hit in hits.values()):
                continue
            real_vs_vision.append(float(hits["real_eeg"] > hits["vision_only"]))
            best_control = max(hits["shuffled_eeg"], hits["random_eeg"])
            real_vs_best_control.append(float(hits["real_eeg"] > best_control))

        outputs.append(
            {
                "corruption": corruption,
                "status": "ok",
                "paired_count": len(real_vs_vision),
                "class_hit_win_rate_vs_vision": _mean(real_vs_vision),
                "class_hit_win_rate_vs_best_control": _mean(real_vs_best_control),
            }
        )
    return outputs


def _record_class_hit(row: dict[str, Any]) -> float | None:
    class_name = str(row.get("human_label_name", "")).lower().strip()
    prediction = str(row.get("prediction", "")).lower()
    if not class_name:
        return None
    aliases = [class_name, *(part.strip() for part in class_name.split(",") if part.strip())]
    return 1.0 if any(alias and alias in prediction for alias in aliases) else 0.0


def _mean(values: Any) -> float | None:
    materialized = [float(v) for v in values if v is not None]
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


def _rate(values: Any) -> float | None:
    materialized = [v for v in values if v is not None]
    if not materialized:
        return None
    return sum(1.0 for v in materialized if v) / len(materialized)
