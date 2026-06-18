from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUNS = [
    ("F0_vision_only", "vision_only"),
    ("F1_real_eeg", "image + aligned real EEG gated fusion"),
    ("F2_random_encoder_control", "image + random EEG control"),
    ("F3_shuffled_training_control", "image + shuffled EEG training control"),
]


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # 其他 agent 可能正在写 JSONL；忽略未写完的尾行，报告为当前可读状态。
                    continue
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _last_reset_segment(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    if not rows:
        return rows
    start = 0
    previous: int | None = None
    for index, row in enumerate(rows):
        value = row.get(key)
        try:
            current = int(value)
        except (TypeError, ValueError):
            continue
        if previous is not None and current <= previous:
            start = index
        previous = current
    return rows[start:]


def _number(row: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _loss_summary(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float | int | None]:
    if not rows:
        return {"start": None, "last": None, "best": None, "best_epoch": None}
    values: list[tuple[int | None, float]] = []
    for row in rows:
        value = _number(row, keys)
        if value is None:
            continue
        epoch_value = row.get("epoch")
        try:
            epoch = int(epoch_value) if epoch_value is not None else None
        except (TypeError, ValueError):
            epoch = None
        values.append((epoch, value))
    if not values:
        return {"start": None, "last": None, "best": None, "best_epoch": None}
    best_epoch, best = min(values, key=lambda item: item[1])
    return {
        "start": values[0][1],
        "last": values[-1][1],
        "best": best,
        "best_epoch": best_epoch,
    }


def _sample_files(run_dir: Path) -> list[Path]:
    files: list[Path] = []
    legacy = run_dir / "samples.jsonl"
    if legacy.exists():
        files.append(legacy)
    sample_dir = run_dir / "samples"
    if sample_dir.exists():
        files.extend(sorted(sample_dir.glob("*.jsonl")))
    return files


def _sample_summary(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, float | int | None]]:
    rows: list[dict[str, Any]] = []
    for path in _sample_files(run_dir):
        rows.extend(_read_jsonl(path))
    predictions = [" ".join(str(row.get("prediction", "")).split()) for row in rows]
    nonempty = [prediction for prediction in predictions if prediction]
    lengths = [len(prediction.split()) for prediction in nonempty]
    avg_len = sum(lengths) / len(lengths) if lengths else None
    return rows, {
        "count": len(rows),
        "avg_pred_len": avg_len,
        "distinct_predictions": len(set(nonempty)),
    }


def _checkpoint_paths(run_dir: Path) -> list[Path]:
    candidates = [
        run_dir / "checkpoints" / "best.pt",
        run_dir / "checkpoints" / "last.pt",
        run_dir / "checkpoint_best.pt",
        run_dir / "checkpoint_last.pt",
    ]
    patterns = ["*.pt", "*.pth", "*.safetensors"]
    for checkpoint_dir in [run_dir / "checkpoints", run_dir]:
        if checkpoint_dir.exists():
            for pattern in patterns:
                candidates.extend(sorted(checkpoint_dir.glob(pattern)))
    seen: set[Path] = set()
    existing: list[Path] = []
    for path in candidates:
        if path.exists() and path not in seen:
            seen.add(path)
            existing.append(path)
    return existing


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_path_list(paths: list[Path]) -> str:
    if not paths:
        return "missing"
    return "<br>".join(f"`{path}`" for path in paths)


def _max_epoch(rows: list[dict[str, Any]]) -> int | None:
    epochs: list[int] = []
    for row in rows:
        value = row.get("epoch")
        try:
            epochs.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(epochs) if epochs else None


def _status(
    run_dir: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    sample_count: int,
    checkpoints: list[Path],
    expected_epochs: int,
) -> str:
    if not run_dir.exists():
        return "pending: run directory missing"
    missing: list[str] = []
    if not train_rows:
        missing.append("train_log")
    if not val_rows:
        missing.append("val_log")
    if sample_count == 0:
        missing.append("samples")
    if not checkpoints:
        missing.append("checkpoint")
    if missing:
        return "partial: missing " + ", ".join(missing)
    latest_epoch = _max_epoch(val_rows)
    if expected_epochs > 0 and (latest_epoch is None or latest_epoch < expected_epochs):
        return f"running/partial: val epoch {latest_epoch or 0}/{expected_epochs}"
    return "complete"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Day5 fusion comparison report.")
    parser.add_argument("--root", default="outputs/day5_fusion")
    parser.add_argument("--out", default="outputs/day5_fusion/FUSION_COMPARISON_REPORT.md")
    parser.add_argument("--expected_epochs", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root)
    lines = [
        "# Fusion Comparison Report",
        "",
        "- Caption target type used: `human_class_caption`",
        "- LLM: frozen `Qwen/Qwen2.5-1.5B-Instruct`",
        "- EEG encoder: frozen Day4 best alignment encoder when available",
        "",
        "| Run | Variant | Status | Train Loss Start | Train Loss Last | Best Val Loss | Last Val Loss | Best Epoch | Samples | Avg Pred Len | Distinct Preds | Checkpoints |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for dirname, variant in RUNS:
        run_dir = root / dirname
        train_rows = _last_reset_segment(_read_jsonl(run_dir / "train_log.jsonl"), "step")
        val_rows = _last_reset_segment(_read_jsonl(run_dir / "val_log.jsonl"), "epoch")
        train = _loss_summary(train_rows, ["train_loss", "loss"])
        val = _loss_summary(val_rows, ["val_loss", "loss"])
        samples, sample_stats = _sample_summary(run_dir)
        checkpoints = _checkpoint_paths(run_dir)
        status = _status(run_dir, train_rows, val_rows, int(sample_stats["count"] or 0), checkpoints, args.expected_epochs)
        lines.append(
            f"| {dirname} | {variant} | {status} | "
            f"{_fmt(train['start'])} | {_fmt(train['last'])} | {_fmt(val['best'])} | {_fmt(val['last'])} | "
            f"{_fmt(val['best_epoch'], digits=0)} | {_fmt(sample_stats['count'], digits=0)} | "
            f"{_fmt(sample_stats['avg_pred_len'], digits=2)} | {_fmt(sample_stats['distinct_predictions'], digits=0)} | "
            f"{_fmt_path_list(checkpoints)} |"
        )
    lines.extend(["", "## Generated Examples", ""])
    for dirname, _ in RUNS:
        samples, _ = _sample_summary(root / dirname)
        samples = samples[:3]
        if not samples:
            lines.append(f"- `{dirname}`: pending samples")
            continue
        for row in samples:
            prediction = " ".join(str(row.get("prediction", "")).split())
            lines.append(f"- `{dirname}/{row.get('image_id', '')}`: {prediction} (ref: {row.get('reference', '')})")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Average gate values are measured in the downstream sanity outputs, not during this training report.",
            "- Overfitting is indicated by low train loss with poor sample text quality or rising validation loss.",
            "- Do not claim EEG benefit until real EEG beats shuffled/random in full sanity metrics.",
        ]
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
