from __future__ import annotations

from collections import Counter
import argparse
import csv
import json
from pathlib import Path


IGNORED_JSONL_FILENAMES = {
    "sample_predictions.jsonl",
}


def iter_prediction_jsonl_files(input_dir: str | Path) -> list[Path]:
    input_dir = Path(input_dir)
    paths: list[Path] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        if path.name in IGNORED_JSONL_FILENAMES:
            continue
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            paths.append(path)
            continue
        first = records[0]
        if first.get("corruption") is None or first.get("mode") is None:
            continue
        paths.append(path)
    return paths


def token_f1(reference: str, prediction: str) -> float:
    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum((ref_counts & pred_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def bleu_n(reference: str, prediction: str, n: int) -> float:
    ref = reference.lower().split()
    pred = prediction.lower().split()
    if len(pred) < n or not ref:
        return 0.0
    ref_counts = ngram_counts(ref, n)
    pred_counts = ngram_counts(pred, n)
    overlap = sum((ref_counts & pred_counts).values())
    total = sum(pred_counts.values())
    if total == 0:
        return 0.0
    precision = overlap / total
    brevity = min(1.0, len(pred) / max(1, len(ref)))
    return precision * brevity


def rouge_l(reference: str, prediction: str) -> float:
    ref = reference.lower().split()
    pred = prediction.lower().split()
    if not ref or not pred:
        return 0.0
    dp = [[0] * (len(pred) + 1) for _ in range(len(ref) + 1)]
    for i, ref_token in enumerate(ref, start=1):
        for j, pred_token in enumerate(pred, start=1):
            if ref_token == pred_token:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _class_name_from_record(row: dict) -> str | None:
    value = row.get("human_label_name")
    if value:
        return str(value).lower()
    reference = str(row.get("reference", "")).lower()
    marker = "a photo of "
    if marker in reference:
        return reference.split(marker, maxsplit=1)[1].strip()
    return None


def class_hit(row: dict) -> float | None:
    class_name = _class_name_from_record(row)
    if not class_name:
        return None
    prediction = str(row.get("prediction", "")).lower()
    aliases = [class_name]
    aliases.extend(part.strip() for part in class_name.split(",") if part.strip())
    return 1.0 if any(alias and alias in prediction for alias in aliases) else 0.0


def summarize_records(records: list[dict]) -> dict[str, float]:
    if not records:
        return {
            "count": 0.0,
            "bleu_1": 0.0,
            "bleu_4": 0.0,
            "rouge_l": 0.0,
            "avg_prediction_length": 0.0,
            "distinct_prediction_ratio": 0.0,
            "gate_mean": None,
            "class_hit": None,
        }
    predictions = [str(row.get("prediction", "")) for row in records]
    references = [str(row.get("reference", "")) for row in records]
    gate_values = [
        float(row["gate_mean"])
        for row in records
        if row.get("gate_mean") is not None
    ]
    class_hits = [hit for row in records if (hit := class_hit(row)) is not None]
    return {
        "count": float(len(records)),
        "bleu_1": sum(bleu_n(r, p, 1) for r, p in zip(references, predictions, strict=False)) / len(records),
        "bleu_4": sum(bleu_n(r, p, 4) for r, p in zip(references, predictions, strict=False)) / len(records),
        "rouge_l": sum(rouge_l(r, p) for r, p in zip(references, predictions, strict=False)) / len(records),
        "avg_prediction_length": sum(len(p.split()) for p in predictions) / len(records),
        "distinct_prediction_ratio": len(set(predictions)) / len(records),
        "gate_mean": sum(gate_values) / len(gate_values) if gate_values else None,
        "class_hit": sum(class_hits) / len(class_hits) if class_hits else None,
    }


def evaluate_directory(
    input_dir: str | Path,
    csv_path: str | Path,
    md_path: str | Path,
    *,
    require_modes: list[str] | None = None,
    require_corruptions: list[str] | None = None,
) -> None:
    input_dir = Path(input_dir)
    rows = []
    for path in iter_prediction_jsonl_files(input_dir):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        summary = summarize_records(records)
        first = records[0] if records else {}
        rows.append(
            {
                "file": path.name,
                "corruption": first.get("corruption", path.stem.split("_")[0]),
                "mode": first.get("mode", "_".join(path.stem.split("_")[1:])),
                **summary,
            }
        )

    if require_modes and require_corruptions:
        seen = {(str(row["corruption"]), str(row["mode"])) for row in rows}
        missing = [
            (corruption, mode)
            for corruption in require_corruptions
            for mode in require_modes
            if (corruption, mode) not in seen
        ]
        if missing:
            examples = ", ".join(f"{corruption}_{mode}.jsonl" for corruption, mode in missing[:10])
            raise FileNotFoundError(
                f"Missing {len(missing)} required sanity outputs in {input_dir}. "
                f"Examples: {examples}. Run src.eval.sanity_check first."
            )

    csv_path = Path(csv_path)
    md_path = Path(md_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    lines = ["# Sanity Metrics", ""]
    if rows:
        headers = list(rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    "NA" if row[h] is None else f"{row[h]:.4f}" if isinstance(row[h], float) else str(row[h])
                    for h in headers
                )
                + " |"
            )
    else:
        lines.append("No JSONL prediction files found.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generation JSONL files.")
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--pred_dir", default=None)
    parser.add_argument("--csv", default="outputs/sanity_real/metrics.csv")
    parser.add_argument("--md", default="outputs/sanity_real/metrics.md")
    parser.add_argument("--out", default=None)
    parser.add_argument("--require_modes", nargs="*", default=None)
    parser.add_argument("--require_corruptions", nargs="*", default=None)
    args = parser.parse_args()
    input_dir = args.input_dir or args.pred_dir
    if input_dir is None:
        parser.error("one of --input_dir or --pred_dir is required")
    md_path = args.out or args.md
    csv_path = args.csv
    if args.out and args.csv == "outputs/sanity_real/metrics.csv":
        csv_path = str(Path(args.out).with_suffix(".csv"))
    try:
        evaluate_directory(
            input_dir,
            csv_path,
            md_path,
            require_modes=args.require_modes,
            require_corruptions=args.require_corruptions,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
