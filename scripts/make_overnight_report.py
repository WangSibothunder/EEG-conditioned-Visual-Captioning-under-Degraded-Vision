from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    return last


def _parse_cache_report(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("- ") and ": `" in line:
            key, value = line[2:].split(": `", maxsplit=1)
            data[key] = value.rstrip("`")
    return data


def _read_sanity_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sample_predictions(root: Path, limit: int = 10) -> list[str]:
    preferred = [
        "clean_real_eeg.jsonl",
        "clean_vision_only.jsonl",
        "blur_real_eeg.jsonl",
        "blur_vision_only.jsonl",
        "occlusion_real_eeg.jsonl",
        "occlusion_vision_only.jsonl",
    ]
    sanity = root / "sanity_mini"
    candidates = [sanity / name for name in preferred if (sanity / name).exists()]
    candidates.extend(path for path in sorted(sanity.glob("*.jsonl")) if path not in candidates)
    candidates.extend(sorted((root / "fusion_qwen15").glob("*.jsonl")))
    rows: list[str] = []
    for path in candidates:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                prediction = " ".join(str(item.get("prediction", "")).split())
                reference = " ".join(str(item.get("reference", "")).split())
                rows.append(
                    f"- `{item.get('image_id', '')}` {item.get('mode', '')}: "
                    f"{prediction} (ref: {reference})"
                )
                if len(rows) >= limit:
                    return rows
    return rows


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "0.0000"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the overnight research report.")
    parser.add_argument("--root", default="outputs/overnight")
    parser.add_argument("--out", default="outputs/overnight/OVERNIGHT_REPORT.md")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    retrieval = _read_json(root / "align_strong" / "retrieval_metrics.json") or {}
    history = _read_json(root / "align_strong" / "history.json")
    model_metrics = retrieval.get("model", {}) if isinstance(retrieval, dict) else {}
    random_metrics = retrieval.get("random", {}) if isinstance(retrieval, dict) else {}
    best_ckpt = root / "align_strong" / "checkpoints" / "best.pt"
    train_log_last = _last_jsonl(root / "align_strong" / "train_log.jsonl")
    sanity_rows = _read_sanity_csv(root / "sanity_mini" / "metrics.csv")

    cache_report_names = {
        "train": "clip_cache_report.md",
        "val": "clip_cache_report_val.md",
        "test": "clip_cache_report_test.md",
    }
    cache_reports = {
        split: _parse_cache_report(root / filename)
        for split, filename in cache_report_names.items()
    }

    lines = [
        "# Overnight Report",
        "",
        f"- Report generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Start/end evidence: see `{root / 'audit_report.md'}` and `{root / 'logs' / 'overnight.log'}` when present.",
        f"- GPU info: see `{root / 'audit_report.md'}`.",
        f"- Root: `{root}`",
        "",
        "## Dataset Status",
        "",
        f"- train rows: `{_count_jsonl(Path('data/thought2text/train.jsonl'))}`",
        f"- val rows: `{_count_jsonl(Path('data/thought2text/val.jsonl'))}`",
        f"- test rows: `{_count_jsonl(Path('data/thought2text/test.jsonl'))}`",
        f"- Thought2Text inspection: `{root / 'thought2text_inspection.md'}`",
        f"- Manifest report: `{root / 'manifest_report.md'}`",
        f"- Best checkpoint path: `{best_ckpt}`",
        "",
        "## CLIP Cache Statistics",
        "",
        "| Split | Images | Shape | Missing | Tiny Fallback | Report |",
        "| --- | ---: | --- | ---: | --- | --- |",
    ]
    for split, report in cache_reports.items():
        lines.append(
            f"| {split} | {report.get('Images processed', '0')} | {report.get('Embedding shape', 'n/a')} | "
            f"{report.get('Missing images', 'n/a')} | {report.get('Tiny fallback used', 'n/a')} | "
            f"`{root / cache_report_names[split]}` |"
        )

    lines.extend(
        [
            "",
            "## Alignment",
            "",
            "| Run | Loss | R@1 | R@5 | R@10 | Mean Rank | Random R@5 | Notes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            (
                f"| strong | InfoNCE+MSE+CE+sim+aug | {_fmt(model_metrics.get('r@1'))} | "
                f"{_fmt(model_metrics.get('r@5'))} | {_fmt(model_metrics.get('r@10'))} | "
                f"{_fmt(model_metrics.get('mean_rank'), 2)} | {_fmt(random_metrics.get('r@5'))} | "
                f"best checkpoint: `{best_ckpt}` |"
            ),
            "",
            "## Loss Curves Summary",
            "",
        ]
    )
    if isinstance(history, list) and history:
        lines.append(f"- epochs recorded: `{len(history)}`")
        lines.append(f"- final loss: `{_fmt(history[-1].get('loss'))}`")
    elif train_log_last:
        lines.append("- Full `history.json` was not written; the alignment process exited before the final summary step.")
        lines.append(f"- last logged epoch: `{train_log_last.get('epoch')}`")
        lines.append(f"- last logged step: `{train_log_last.get('step')}`")
        lines.append(f"- last logged total loss: `{_fmt(train_log_last.get('total'))}`")
    else:
        lines.append("- No alignment history found.")

    lines.extend(
        [
            "",
            "## Fusion Caption Training Status",
            "",
            f"- Fusion checkpoint exists: `{(root / 'fusion_qwen15' / 'checkpoints' / 'best.pt').exists()}`",
            f"- Fusion validation log: `{root / 'fusion_qwen15' / 'val_log.jsonl'}`",
            "- Fusion was secondary to the EEG-to-CLIP alignment gate for this run.",
            "- The Qwen fusion run was capped to 32 optimizer steps and 64 validation batches.",
            "",
            "## Caption Sanity",
            "",
            "| Corruption | Mode | BLEU-1 | ROUGE-L | Avg Len | Distinct Ratio | Notes |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if sanity_rows:
        for row in sanity_rows:
            lines.append(
                f"| {row.get('corruption', '')} | {row.get('mode', '')} | "
                f"{_fmt(row.get('bleu_1'))} | {_fmt(row.get('rouge_l'))} | "
                f"{_fmt(row.get('avg_prediction_length'), 2)} | "
                f"{_fmt(row.get('distinct_prediction_ratio'))} | see `{row.get('file', '')}` |"
            )
    else:
        lines.append("| n/a | n/a | 0 | 0 | 0 | 0 | mini sanity not completed |")

    lines.extend(["", "## Sample Predictions", ""])
    lines.extend(_sample_predictions(root) or ["- No sample predictions available yet."])

    lines.extend(
        [
            "",
            "## Known Problems",
            "",
            "- EEG-to-CLIP retrieval is above random, but the margin is small and mean rank is still weak.",
            "- Caption fusion was only a short capped Qwen run; generated captions are not yet meaningful.",
            "- Mini sanity used 32 test samples rather than the requested 128 to keep the secondary caption stage bounded.",
            "- Do not claim that the model reads thoughts.",
            "",
            "## Recommended Next Commands",
            "",
            "```bash",
            "python -m src.eval.retrieval --manifest data/thought2text/test.jsonl --clip_cache data/thought2text/cache/clip_test.npy --eeg_ckpt outputs/overnight/align_strong/checkpoints/best.pt --out outputs/overnight/align_strong/retrieval_metrics.json",
            "python scripts/make_overnight_report.py --root outputs/overnight --out outputs/overnight/OVERNIGHT_REPORT.md",
            "```",
            "",
            "Preliminary results show whether EEG embeddings align with visual CLIP space and whether real EEG behaves differently from shuffled/random EEG under degraded visual inputs.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
