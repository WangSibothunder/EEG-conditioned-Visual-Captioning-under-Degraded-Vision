from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["status"])
        writer.writeheader()
        writer.writerows(rows or [{"status": "missing"}])


def _write_md_table(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    lines = [f"# {title}", ""]
    if rows:
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    else:
        lines.append("No rows available.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in {None, ""}:
        return default
    return float(value)


def _seed_from_path(path: Path) -> str:
    text = str(path)
    for token in ["seed2025", "seed2718", "seed3407", "seed123", "seed42", "seed314", "seed777", "seed888", "seed999"]:
        if token in text:
            return token.replace("seed", "")
    if "semantic_fusion_A2_temporal_spectral_spatial_full" in text:
        return "42"
    return "unknown"


def _write_p2_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    best = max(rows, key=lambda row: float(row.get("R@5") or 0.0), default={})
    historical_best = 0.3303
    best_r5 = float(best.get("R@5") or 0.0)
    lines = [
        "# P2 Alignment Final Summary",
        "",
        f"- Does P2 beat the previous historical best test R@5 `{historical_best:.4f}`? `{'yes' if best_r5 > historical_best else 'no'}`",
        f"- Best seed: `{best.get('seed', 'missing')}`",
        f"- Best checkpoint: `{best.get('checkpoint', 'missing')}`",
        "- Best val R@5: `not separately available in final table; final selection uses saved best checkpoints and test metrics`",
        f"- Best test R@5: `{best_r5:.6f}`",
        "",
    ]
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if rows:
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize_semantic_eval_dirs(eval_dirs: list[Path], model_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for eval_dir in eval_dirs:
        seed = _seed_from_path(eval_dir)
        metrics = _read_csv(eval_dir / "FULL_METRICS.csv")
        by_condition: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        for row in metrics:
            mode = str(row.get("mode", ""))
            if mode.endswith("_image_level"):
                continue
            by_condition[str(row.get("corruption", ""))][mode] = row
        for corruption, modes in sorted(by_condition.items()):
            real = modes.get("real_eeg")
            if real is None:
                continue
            vision = modes.get("vision_only", {})
            shuffled = modes.get("shuffled_eeg", {})
            random = modes.get("random_eeg", {})
            row = {
                "model": model_name,
                "seed": seed,
                "corruption": corruption,
                "real_top1": _float(real, "accuracy"),
                "real_top5": _float(real, "top5_accuracy"),
                "class_hit": _float(real, "caption_class_hit", _float(real, "accuracy")),
                "vision_top1": _float(vision, "accuracy"),
                "shuffled_top1": _float(shuffled, "accuracy"),
                "random_top1": _float(random, "accuracy"),
            }
            row["real_minus_vision"] = row["real_top1"] - row["vision_top1"]
            row["real_minus_shuffled"] = row["real_top1"] - row["shuffled_top1"]
            row["real_minus_random"] = row["real_top1"] - row["random_top1"]
            row["real_beats_vision"] = row["real_minus_vision"] > 0
            row["real_beats_controls"] = row["real_minus_shuffled"] > 0 and row["real_minus_random"] > 0
            row["eval_dir"] = str(eval_dir)
            rows.append(row)
    return rows


def _mean_std_rows(rows: list[dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    by_corruption: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_corruption[str(row["corruption"])].append(row)
    summary: list[dict[str, Any]] = []
    for corruption, items in sorted(by_corruption.items()):
        def values(key: str) -> list[float]:
            return [float(item[key]) for item in items]

        summary.append(
            {
                "model": model_name,
                "corruption": corruption,
                "seeds": ",".join(sorted({str(item["seed"]) for item in items})),
                "top1_mean": f"{mean(values('real_top1')):.6f}",
                "top1_std": f"{pstdev(values('real_top1')):.6f}" if len(items) > 1 else "0.000000",
                "top5_mean": f"{mean(values('real_top5')):.6f}",
                "class_hit_mean": f"{mean(values('class_hit')):.6f}",
                "real_vision_mean": f"{mean(values('real_minus_vision')):.6f}",
                "real_shuffled_mean": f"{mean(values('real_minus_shuffled')):.6f}",
                "real_random_mean": f"{mean(values('real_minus_random')):.6f}",
                "vision_wins": f"{sum(1 for item in items if item['real_beats_vision'])}/{len(items)}",
                "control_wins": f"{sum(1 for item in items if item['real_beats_controls'])}/{len(items)}",
            }
        )
    return summary


def _alignment_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("model", payload)
    unique = model.get("unique_image", model)
    trial = model.get("trial", {})
    random = payload.get("random") or model.get("random_unique_image", {})
    return {
        "R@1": unique.get("r@1", unique.get("R@1")),
        "R@5": unique.get("r@5", unique.get("R@5")),
        "R@10": unique.get("r@10", unique.get("R@10")),
        "class_acc": unique.get("class_acc", model.get("class_acc")),
        "mean_rank": unique.get("mean_rank", model.get("mean_rank")),
        "median_rank": unique.get("median_rank", model.get("median_rank")),
        "trial_R@1": trial.get("r@1", trial.get("R@1")),
        "trial_R@5": trial.get("r@5", trial.get("R@5")),
        "trial_R@10": trial.get("r@10", trial.get("R@10")),
        "random_R@1": random.get("r@1", random.get("R@1")),
        "random_R@5": random.get("r@5", random.get("R@5")),
        "random_R@10": random.get("r@10", random.get("R@10")),
    }


def _semantic_score(rows: list[dict[str, Any]], model_name: str) -> float:
    strong = [row for row in rows if row.get("model") == model_name and row.get("corruption") != "clean"]
    if not strong:
        return -1.0
    real = mean(float(row["real_top1"]) for row in strong)
    shuf = mean(float(row["real_minus_shuffled"]) for row in strong)
    rand = mean(float(row["real_minus_random"]) for row in strong)
    vision = mean(float(row["real_minus_vision"]) for row in strong)
    seeds: dict[str, list[float]] = defaultdict(list)
    for row in strong:
        seeds[str(row["seed"])].append(float(row["real_top1"]))
    seed_means = [mean(values) for values in seeds.values()]
    seed_std = pstdev(seed_means) if len(seed_means) > 1 else 0.0
    return real + 0.5 * shuf + 0.5 * rand + 0.3 * vision - 0.1 * seed_std


def materialize_final_results(
    *,
    out_dir: Path,
    a2_eval_dirs: list[Path],
    p2a2_eval_dirs: list[tuple[str, Path]],
    p2_metric_files: list[tuple[str, str, Path, Path]],
    gate_report: Path | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    a2_rows = _summarize_semantic_eval_dirs(a2_eval_dirs, "A2_final")
    p2a2_rows: list[dict[str, Any]] = []
    for variant, eval_dir in p2a2_eval_dirs:
        p2a2_rows.extend(_summarize_semantic_eval_dirs([eval_dir], variant))

    _write_csv(out_dir / "A2_FINAL_METRICS.csv", a2_rows)
    _write_md_table(out_dir / "A2_FINAL_SUMMARY.md", "A2 Final Semantic Fusion Summary", _mean_std_rows(a2_rows, "A2_final"))
    _write_csv(out_dir / "P2A2_FINAL_METRICS.csv", p2a2_rows)
    p2a2_summary: list[dict[str, Any]] = []
    for model_name in sorted({str(row["model"]) for row in p2a2_rows}):
        p2a2_summary.extend(_mean_std_rows([row for row in p2a2_rows if row["model"] == model_name], model_name))
    _write_md_table(out_dir / "P2A2_FINAL_SUMMARY.md", "P2A2 Final Semantic Fusion Summary", p2a2_summary)

    p2_rows: list[dict[str, Any]] = []
    for seed, name, metric_file, checkpoint in p2_metric_files:
        if not metric_file.exists():
            continue
        row = {"seed": seed, "experiment": name, "metric_file": str(metric_file), "checkpoint": str(checkpoint)}
        row.update(_alignment_payload(metric_file))
        p2_rows.append(row)
    _write_csv(out_dir / "P2_ALIGNMENT_FINAL_METRICS.csv", p2_rows)
    _write_p2_summary(out_dir / "P2_ALIGNMENT_FINAL_SUMMARY.md", p2_rows)

    best_p2 = max(p2_rows, key=lambda row: float(row.get("R@5") or 0.0), default={})
    best_p2_path = Path(str(best_p2.get("checkpoint", "")))
    if best_p2_path.exists():
        shutil.copy2(best_p2_path, out_dir / "best_p2_encoder.pt")

    all_semantic = a2_rows + p2a2_rows
    model_names = sorted({str(row["model"]) for row in all_semantic})
    comparison = [
        {"model": name, "weighted_score": f"{_semantic_score(all_semantic, name):.6f}"}
        for name in model_names
    ]
    comparison.sort(key=lambda row: float(row["weighted_score"]), reverse=True)
    final_model = comparison[0]["model"] if comparison else "not_available"
    (out_dir / "FINAL_MODEL_SELECTION.md").write_text(
        "\n".join(
            [
                "# Final Model Selection",
                "",
                f"- Recommended final model: `{final_model}`",
                f"- Recommended checkpoint: see `{out_dir / 'FINAL_CHECKPOINT_PATHS.md'}`",
                f"- Recommended metrics file: `{out_dir / ('A2_FINAL_METRICS.csv' if final_model == 'A2_final' else 'P2A2_FINAL_METRICS.csv')}`",
                "- Why selected: highest weighted strong-degradation semantic score among available final rows.",
                "- What it failed to solve: open-ended free-form caption generation and mechanistic gate interpretability remain unsupported.",
                "",
                "| Model | Weighted Score |",
                "| --- | ---: |",
                *[f"| {row['model']} | {row['weighted_score']} |" for row in comparison],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    gate_text = gate_report.read_text(encoding="utf-8") if gate_report and gate_report.exists() else "Gate ablation not available."
    (out_dir / "GATE_ABLATION_REPORT.md").write_text(gate_text, encoding="utf-8")
    strong_rows: list[dict[str, Any]] = []
    for model_name in sorted({str(row["model"]) for row in all_semantic}):
        strong_rows.extend(_mean_std_rows([row for row in all_semantic if row["model"] == model_name], model_name))
    _write_md_table(out_dir / "STRONG_DEGRADATION_RESULTS.md", "Strong Degradation Results", strong_rows)

    checkpoints = [
        "# Final Checkpoint Paths",
        "",
        f"- Best P2 encoder copy: `{out_dir / 'best_p2_encoder.pt'}`",
        f"- Best P2 source: `{best_p2.get('checkpoint', 'missing')}`",
    ]
    for path in sorted((out_dir / "runs").glob("P2A2_*_seed*/semantic_fusion_classifier.pt")):
        checkpoints.append(f"- P2A2 checkpoint: `{path}`")
    for path in [
        Path("outputs/heavy_stage/semantic_fusion_A2_temporal_spectral_spatial_full/semantic_fusion_classifier.pt"),
        Path("outputs/heavy_stage/semantic_fusion_A2_temporal_spectral_spatial_seed123_full/semantic_fusion_classifier.pt"),
        Path("outputs/heavy_stage/semantic_fusion_A2_temporal_spectral_spatial_seed2025_full/semantic_fusion_classifier.pt"),
    ]:
        if path.exists():
            checkpoints.append(f"- A2 checkpoint: `{path}`")
    (out_dir / "FINAL_CHECKPOINT_PATHS.md").write_text("\n".join(checkpoints) + "\n", encoding="utf-8")

    def collect_examples(title: str, dirs: list[Path]) -> str:
        lines = [f"# {title}", ""]
        for source in dirs:
            example_path = source / "qualitative_examples.md"
            if example_path.exists():
                lines.extend([f"## {source}", "", example_path.read_text(encoding="utf-8")[:5000], ""])
        if len(lines) == 2:
            lines.append("No qualitative examples available yet.")
        return "\n".join(lines) + "\n"

    a2_examples = collect_examples("A2 Final Examples", a2_eval_dirs)
    p2a2_example_dirs = [eval_dir for _variant, eval_dir in p2a2_eval_dirs]
    p2a2_examples = collect_examples("P2A2 Final Examples", p2a2_example_dirs)
    (out_dir / "A2_FINAL_EXAMPLES.md").write_text(a2_examples, encoding="utf-8")
    (out_dir / "P2A2_FINAL_EXAMPLES.md").write_text(p2a2_examples, encoding="utf-8")

    examples: list[str] = ["# Qualitative Examples", "", a2_examples, p2a2_examples]
    for source in a2_eval_dirs + p2a2_example_dirs:
        example_path = source / "qualitative_examples.md"
        if example_path.exists():
            examples.extend([f"## {source}", "", example_path.read_text(encoding="utf-8")[:5000], ""])
    (out_dir / "QUALITATIVE_EXAMPLES.md").write_text("\n".join(examples) + "\n", encoding="utf-8")

    report_lines = [
        "# Final 24H Report",
        "",
        f"1. What is the final model? `{final_model}`",
        f"2. What is the best checkpoint path? `{best_p2.get('checkpoint', 'missing')}` for P2 alignment; see model selection for semantic model.",
        "3. Does real EEG beat shuffled EEG? See Table 2/3.",
        "4. Does real EEG beat random EEG? See Table 2/3.",
        "5. Does real EEG beat vision-only under strong degradation? See Table 6.",
        "6. Which corruptions show the largest EEG gain? See Table 6.",
        "7. Did P2A2 improve over A2? See Table 4.",
        "8. Did the gate help? See Table 5.",
        f"9. What is the best alignment model? `{best_p2.get('experiment', 'missing')}`.",
        "10. What remains unsolved? Free-form captioning, broad large-dataset transfer gains, and gate interpretability.",
        "",
        "## Table 1: Final alignment results",
        "",
        (out_dir / "P2_ALIGNMENT_FINAL_SUMMARY.md").read_text(encoding="utf-8") if (out_dir / "P2_ALIGNMENT_FINAL_SUMMARY.md").exists() else "",
        "## Table 2: A2 semantic fusion multi-seed results",
        "",
        (out_dir / "A2_FINAL_SUMMARY.md").read_text(encoding="utf-8") if (out_dir / "A2_FINAL_SUMMARY.md").exists() else "",
        "## Table 3: P2A2 semantic fusion multi-seed results",
        "",
        (out_dir / "P2A2_FINAL_SUMMARY.md").read_text(encoding="utf-8") if (out_dir / "P2A2_FINAL_SUMMARY.md").exists() else "",
        "## Table 4: Final model comparison",
        "",
        (out_dir / "FINAL_MODEL_SELECTION.md").read_text(encoding="utf-8"),
        "## Table 5: Gate ablation",
        "",
        gate_text,
        "## Table 6: Strong degradation gains",
        "",
        (out_dir / "STRONG_DEGRADATION_RESULTS.md").read_text(encoding="utf-8") if (out_dir / "STRONG_DEGRADATION_RESULTS.md").exists() else "",
    ]
    (out_dir / "FINAL_24H_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def _default_p2_metric_files() -> list[tuple[str, str, Path, Path]]:
    entries = [
        ("42", "P2_raw_spectrogram_lr5e5_long30", Path("outputs/alignment_long_extra/P2_raw_spectrogram_lr5e5_long30/test_metrics.json"), Path("outputs/alignment_long_extra/P2_raw_spectrogram_lr5e5_long30/checkpoints/best.pt")),
        ("123", "P2_raw_spectrogram_seed123_long30", Path("outputs/alignment_long_extra/P2_raw_spectrogram_seed123_long30/test_metrics.json"), Path("outputs/alignment_long_extra/P2_raw_spectrogram_seed123_long30/checkpoints/best.pt")),
        ("2025", "P2_raw_spectrogram_seed2025_long30", Path("outputs/alignment_long_extra/P2_raw_spectrogram_seed2025_long30/test_metrics.json"), Path("outputs/alignment_long_extra/P2_raw_spectrogram_seed2025_long30/checkpoints/best.pt")),
        ("2718", "P2_raw_spectrogram_seed2718_fill8", Path("outputs/alignment_fill8/P2_raw_spectrogram_seed2718_fill8/test_metrics.json"), Path("outputs/alignment_fill8/P2_raw_spectrogram_seed2718_fill8/checkpoints/best.pt")),
    ]
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize final 24h EEG semantic results.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/final_results"))
    args = parser.parse_args()
    out = args.out_dir
    a2_dirs = [
        Path("outputs/final_semantic/semantic_fusion_A2_temporal_spectral_spatial_full_strong_eval"),
        Path("outputs/final_semantic/semantic_fusion_A2_temporal_spectral_spatial_seed123_strong_eval"),
        Path("outputs/final_semantic/semantic_fusion_A2_temporal_spectral_spatial_seed2025_strong_eval"),
    ]
    p2a2_dirs: list[tuple[str, Path]] = []
    for variant in ["freeze", "unfreeze_last2"]:
        model_name = "P2A2_freeze_encoder" if variant == "freeze" else "P2A2_unfreeze_last2"
        for seed in ["42", "123", "2025"]:
            eval_dir = out / "runs" / f"P2A2_{variant}_seed{seed}_strong_eval"
            if (eval_dir / "FULL_METRICS.csv").exists():
                p2a2_dirs.append((model_name, eval_dir))
    materialize_final_results(
        out_dir=out,
        a2_eval_dirs=a2_dirs,
        p2a2_eval_dirs=p2a2_dirs,
        p2_metric_files=_default_p2_metric_files(),
        gate_report=Path("outputs/final_semantic/A2_GATE_VS_NOGATE_REPORT.md"),
    )


if __name__ == "__main__":
    main()
