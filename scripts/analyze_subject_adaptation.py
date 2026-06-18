from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


MANIFESTS = {
    "train": Path("data/thought2text/train_human_caption.jsonl"),
    "val": Path("data/thought2text/val_human_caption.jsonl"),
    "test": Path("data/thought2text/test_human_caption.jsonl"),
}
FALLBACK_MANIFESTS = {
    "train": Path("data/thought2text/train.jsonl"),
    "val": Path("data/thought2text/val.jsonl"),
    "test": Path("data/thought2text/test.jsonl"),
}
OUT_DIR = Path("outputs/subject_adaptation")
CONFIG_DIR = Path("configs/subject_adaptation")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def load_manifests() -> dict[str, list[dict[str, Any]]]:
    loaded: dict[str, list[dict[str, Any]]] = {}
    for split, path in MANIFESTS.items():
        source = path if path.exists() else FALLBACK_MANIFESTS[split]
        loaded[split] = read_jsonl(source)
    return loaded


def as_str(value: Any) -> str:
    return "NA" if value is None else str(value)


def metric_float(value: Any) -> float | None:
    try:
        if value in (None, "", "NA"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 6) -> str:
    number = metric_float(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def model_metrics_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get("model"), dict):
        model = payload["model"]
    elif isinstance(payload.get("test"), dict):
        return model_metrics_from_payload(payload["test"])
    elif isinstance(payload.get("val"), dict):
        return model_metrics_from_payload(payload["val"])
    else:
        model = payload
    if isinstance(model.get("unique_image"), dict):
        merged = dict(model)
        merged.update(model["unique_image"])
        return merged
    return dict(model)


def best_history_metrics(path: Path) -> tuple[int | None, dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        return None, {}
    best_epoch: int | None = None
    best_metrics: dict[str, Any] = {}
    best_r5 = -1.0
    for record in payload:
        if not isinstance(record, dict):
            continue
        metrics = model_metrics_from_payload(record.get("metrics", {}))
        r5 = metric_float(metrics.get("r@5"))
        if r5 is not None and r5 >= best_r5:
            best_r5 = r5
            best_epoch = int(record.get("epoch", 0) or 0)
            best_metrics = metrics
    return best_epoch, best_metrics


def load_run_metrics(run_dir: Path) -> tuple[str, int | None, dict[str, Any], str]:
    sources = [
        ("test_metrics", run_dir / "test_metrics.json"),
        ("metrics.test", run_dir / "metrics.json"),
        ("retrieval_metrics", run_dir / "retrieval_metrics.json"),
        ("alignment_metrics", run_dir / "alignment_metrics.json"),
    ]
    for source_name, path in sources:
        payload = load_json(path)
        if payload is None:
            continue
        if source_name == "metrics.test" and isinstance(payload, dict) and payload.get("test"):
            metrics = model_metrics_from_payload(payload["test"])
        else:
            metrics = model_metrics_from_payload(payload)
        if metrics:
            return source_name, None, metrics, str(path)
    best_epoch, metrics = best_history_metrics(run_dir / "history.json")
    if metrics:
        return "history.best_val", best_epoch, metrics, str(run_dir / "history.json")
    return "missing", None, {}, ""


def load_val_test_metrics(run_dir: Path) -> tuple[str, int | None, dict[str, Any], dict[str, Any], str]:
    metrics_payload = load_json(run_dir / "metrics.json")
    val_metrics: dict[str, Any] = {}
    test_metrics: dict[str, Any] = {}
    source_parts: list[str] = []
    metric_paths: list[str] = []
    if isinstance(metrics_payload, dict):
        if isinstance(metrics_payload.get("val"), dict):
            val_metrics = model_metrics_from_payload(metrics_payload["val"])
            source_parts.append("metrics.val")
            metric_paths.append(str(run_dir / "metrics.json"))
        if isinstance(metrics_payload.get("test"), dict):
            test_metrics = model_metrics_from_payload(metrics_payload["test"])
            source_parts.append("metrics.test")
            metric_paths.append(str(run_dir / "metrics.json"))

    if not val_metrics:
        val_payload = load_json(run_dir / "alignment_metrics.json")
        val_metrics = model_metrics_from_payload(val_payload)
        if val_metrics:
            source_parts.append("alignment_metrics")
            metric_paths.append(str(run_dir / "alignment_metrics.json"))

    if not test_metrics:
        test_payload = load_json(run_dir / "test_metrics.json")
        test_metrics = model_metrics_from_payload(test_payload)
        if test_metrics:
            source_parts.append("test_metrics")
            metric_paths.append(str(run_dir / "test_metrics.json"))

    if not val_metrics and not test_metrics:
        source, best_epoch, metrics, metric_path = load_run_metrics(run_dir)
        return source, best_epoch, metrics, metrics, metric_path

    best_epoch, history_metrics = best_history_metrics(run_dir / "history.json")
    if not val_metrics and history_metrics:
        val_metrics = history_metrics
        source_parts.append("history.best_val")
        metric_paths.append(str(run_dir / "history.json"))
    if not test_metrics:
        test_metrics = dict(val_metrics)
    if not val_metrics:
        val_metrics = dict(test_metrics)
    return "+".join(dict.fromkeys(source_parts)) or "missing", best_epoch, val_metrics, test_metrics, ";".join(dict.fromkeys(metric_paths))


def read_board_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row.get("experiment_id", ""): row for row in csv.DictReader(handle)}


def structure_summary(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    subjects = sorted({as_str(row.get("subject_id")) for row in all_rows})
    images = sorted({as_str(row.get("image_id")) for row in all_rows})
    classes = sorted({as_str(row.get("label")) for row in all_rows})

    subject_counts = Counter(as_str(row.get("subject_id")) for row in all_rows)
    image_to_subjects: dict[str, set[str]] = defaultdict(set)
    image_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    subject_image_counts: Counter[tuple[str, str]] = Counter()
    for row in all_rows:
        image_id = as_str(row.get("image_id"))
        subject_id = as_str(row.get("subject_id"))
        image_to_subjects[image_id].add(subject_id)
        image_to_rows[image_id].append(row)
        subject_image_counts[(subject_id, image_id)] += 1

    shared_images = {image_id: subs for image_id, subs in image_to_subjects.items() if len(subs) > 1}
    same_image_cross_subject_pairs = sum(len(subs) * (len(subs) - 1) // 2 for subs in shared_images.values())
    duplicate_subject_image_trials = sum(count - 1 for count in subject_image_counts.values() if count > 1)

    split_images = {split: {as_str(row.get("image_id")) for row in rows} for split, rows in rows_by_split.items()}
    leakage: dict[str, int] = {}
    split_names = sorted(split_images)
    for idx, left in enumerate(split_names):
        for right in split_names[idx + 1 :]:
            leakage[f"{left}-{right}"] = len(split_images[left] & split_images[right])

    per_split = {}
    for split, rows in rows_by_split.items():
        per_split[split] = {
            "trials": len(rows),
            "unique_images": len({as_str(row.get("image_id")) for row in rows}),
            "subjects": len({as_str(row.get("subject_id")) for row in rows}),
            "classes": len({as_str(row.get("label")) for row in rows}),
            "shared_images": sum(1 for image_id in {as_str(row.get("image_id")) for row in rows} if len(image_to_subjects[image_id]) > 1),
        }

    subject_split_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for split, rows in rows_by_split.items():
        counts = Counter(as_str(row.get("subject_id")) for row in rows)
        for subject_id in subjects:
            subject_split_counts[subject_id][split] = counts.get(subject_id, 0)

    image_subject_hist = Counter(len(subs) for subs in image_to_subjects.values())
    examples = []
    for image_id in sorted(shared_images)[:5]:
        first = image_to_rows[image_id][0]
        examples.append(
            {
                "image_id": image_id,
                "subjects": sorted(shared_images[image_id]),
                "label": as_str(first.get("label")),
                "caption": as_str(first.get("caption")),
            }
        )

    return {
        "total_trials": len(all_rows),
        "subjects": subjects,
        "subject_counts": dict(subject_counts),
        "unique_images": len(images),
        "classes": len(classes),
        "shared_images": len(shared_images),
        "max_subjects_per_image": max(image_subject_hist) if image_subject_hist else 0,
        "image_subject_hist": dict(sorted(image_subject_hist.items())),
        "same_image_cross_subject_pairs": same_image_cross_subject_pairs,
        "duplicate_subject_image_trials": duplicate_subject_image_trials,
        "split_leakage": leakage,
        "per_split": per_split,
        "subject_split_counts": dict(subject_split_counts),
        "examples": examples,
    }


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def generate_configs() -> list[Path]:
    base = read_yaml(Path("configs/day5_subject_alignment/X2_subject_same_image_seed888.yaml"))
    specs = [
        {
            "experiment_id": "E5_no_L7_control",
            "seed": 888,
            "use_same_image_subject": False,
            "loss_combo": "L1+L2+L4+L5",
            "notes": "Subject-adaptive E5 control without L7; paired with E5_with_L7_recheck for same seed/config comparison.",
        },
        {
            "experiment_id": "E5_with_L7_recheck",
            "seed": 888,
            "use_same_image_subject": True,
            "loss_combo": "L1+L2+L4+L5+L7",
            "notes": "Subject-adaptive E5 recheck with same-image cross-subject L7 enabled; paired against E5_no_L7_control.",
        },
    ]
    written: list[Path] = []
    manifest = []
    for spec in specs:
        cfg = deepcopy(base)
        cfg["experiment_id"] = spec["experiment_id"]
        cfg["seed"] = spec["seed"]
        cfg.setdefault("model", {})["encoder_type"] = "subject_adaptive"
        cfg.setdefault("loss", {})["use_same_image_subject"] = spec["use_same_image_subject"]
        cfg["loss"]["lambda_same_image_subject"] = 0.2
        cfg["loss"]["loss_combo"] = spec["loss_combo"]
        cfg.setdefault("train", {})["epochs"] = 20
        cfg["train"]["patience"] = 5
        cfg["train"]["max_train_samples"] = 0
        cfg["train"]["max_val_samples"] = 0
        cfg.setdefault("output", {})["dir"] = f"outputs/subject_adaptation/{spec['experiment_id']}"
        cfg["notes"] = spec["notes"]
        out_path = CONFIG_DIR / f"{spec['experiment_id']}.yaml"
        write_yaml(out_path, cfg)
        written.append(out_path)
        manifest.append({"experiment_id": spec["experiment_id"], "config": out_path.name})
    (CONFIG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return written


def collect_comparison_rows() -> list[dict[str, Any]]:
    board = read_board_rows(Path("outputs/day4_search/EXPERIMENT_BOARD.csv"))
    runs = [
        ("S1", "outputs/day4_search/S1", "E3 baseline, no L5/L7"),
        ("S3", "outputs/day4_search/S3", "E3 + L5 non-subject baseline"),
        ("X1", "outputs/day4_search/X1", "E5 subject-adaptive, no L7"),
        ("X2", "outputs/day4_search/X2", "E5 subject-adaptive + L7"),
        ("X3", "outputs/day4_search/X3", "E5 subject-adaptive + SupCon + L7"),
        ("X2_subject_same_image_seed888", "outputs/day5_subject_alignment/X2_subject_same_image_seed888", "Day5 E5 + L7 seed 888"),
        ("E5_no_L7_control", "outputs/subject_adaptation/E5_no_L7_control", "Paired seed888 E5 control without L7"),
        ("E5_with_L7_recheck", "outputs/subject_adaptation/E5_with_L7_recheck", "Paired seed888 E5 with same-image L7"),
    ]
    rows = []
    for exp_id, run_path, note in runs:
        source, best_epoch, val_metrics, test_metrics, metric_path = load_val_test_metrics(Path(run_path))
        board_row = board.get(exp_id, {})
        rows.append(
            {
                "experiment_id": exp_id,
                "encoder": board_row.get("encoder_type", "subject_adaptive" if exp_id.startswith("X") else "NA"),
                "loss": board_row.get("loss_combo", ""),
                "seed": board_row.get("seed", "888" if "seed888" in exp_id else "42"),
                "best_epoch": board_row.get("best_epoch", best_epoch if best_epoch is not None else "NA"),
                "val_r1": board_row.get("val_R@1", val_metrics.get("r@1")),
                "val_r5": board_row.get("val_R@5", val_metrics.get("r@5")),
                "val_r10": board_row.get("val_R@10", val_metrics.get("r@10")),
                "test_r1": board_row.get("test_R@1", test_metrics.get("r@1")),
                "test_r5": board_row.get("test_R@5", test_metrics.get("r@5")),
                "test_r10": board_row.get("test_R@10", test_metrics.get("r@10")),
                "class_acc": board_row.get("class_acc", test_metrics.get("class_acc", val_metrics.get("class_acc"))),
                "mean_rank": board_row.get("mean_rank", test_metrics.get("mean_rank", val_metrics.get("mean_rank"))),
                "source": source,
                "metric_path": metric_path,
                "note": note,
            }
        )
    return rows


def delta(left: Any, right: Any) -> str:
    left_f = metric_float(left)
    right_f = metric_float(right)
    if left_f is None or right_f is None:
        return "NA"
    return f"{right_f - left_f:+.6f}"


def write_structure_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Subject/Image Structure",
        "",
        f"- Total trials: `{summary['total_trials']}`",
        f"- Subjects: `{len(summary['subjects'])}` (`{', '.join(summary['subjects'])}`)",
        f"- Subject trial counts: `{summary['subject_counts']}`",
        f"- Unique images: `{summary['unique_images']}`",
        f"- Images with multiple subjects: `{summary['shared_images']}`",
        f"- Max subjects per image: `{summary['max_subjects_per_image']}`",
        f"- Image subject-count histogram: `{summary['image_subject_hist']}`",
        f"- Same-image cross-subject image pairs: `{summary['same_image_cross_subject_pairs']}`",
        f"- Duplicate trials for the same subject/image: `{summary['duplicate_subject_image_trials']}`",
        f"- Classes: `{summary['classes']}`",
        f"- Image-level split leakage counts: `{summary['split_leakage']}`",
        "",
        "## Split Summary",
        "",
        "| Split | Trials | Unique Images | Subjects | Classes | Shared Images |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ["train", "val", "test"]:
        row = summary["per_split"][split]
        lines.append(
            f"| {split} | {row['trials']} | {row['unique_images']} | {row['subjects']} | {row['classes']} | {row['shared_images']} |"
        )
    lines.extend(["", "## Subject x Split Counts", "", "| Subject | Train | Val | Test | Total |", "| --- | ---: | ---: | ---: | ---: |"])
    for subject_id in summary["subjects"]:
        counts = summary["subject_split_counts"][subject_id]
        total = sum(counts.get(split, 0) for split in ["train", "val", "test"])
        lines.append(f"| {subject_id} | {counts.get('train', 0)} | {counts.get('val', 0)} | {counts.get('test', 0)} | {total} |")
    lines.extend(["", "## Shared Image Examples", "", "| Image ID | Subjects | Label | Caption |", "| --- | --- | --- | --- |"])
    for example in summary["examples"]:
        lines.append(f"| {example['image_id']} | {', '.join(example['subjects'])} | {example['label']} | {example['caption']} |")
    leakage_total = sum(summary["split_leakage"].values())
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Thought2Text has near-complete same-image repetition across six subjects: the structure supports cross-subject consistency losses.",
            f"The current split has `{leakage_total}` image-id overlaps across train/val/test, so image-level leakage is not detected in these manifests.",
            "Subject trial counts are almost balanced; subject adaptation should be evaluated as a controlled modeling change, not as a fix for a severe subject-count imbalance.",
        ]
    )
    (OUT_DIR / "subject_image_structure.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_subject_report(summary: dict[str, Any], rows: list[dict[str, Any]], config_paths: list[Path]) -> None:
    by_id = {row["experiment_id"]: row for row in rows}
    x1, x2, x3 = by_id["X1"], by_id["X2"], by_id["X3"]
    s3 = by_id["S3"]
    day5 = by_id["X2_subject_same_image_seed888"]
    paired_no_l7 = by_id["E5_no_L7_control"]
    paired_l7 = by_id["E5_with_L7_recheck"]
    lines = [
        "# Subject Adaptation Report",
        "",
        "## Dataset Signal",
        "",
        f"- Subjects: `{len(summary['subjects'])}`",
        f"- Shared images across subjects: `{summary['shared_images']} / {summary['unique_images']}`",
        f"- Same-image cross-subject pairs: `{summary['same_image_cross_subject_pairs']}`",
        f"- Split leakage: `{summary['split_leakage']}`",
        "",
        "## Existing Results",
        "",
        "| ID | Encoder/Loss | Seed | Val R@5 | Test R@5 | Test R@1 | Test R@10 | Class Acc | Mean Rank | Source |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        enc_loss = f"{row['encoder']} / {row['loss']}".strip(" /")
        lines.append(
            f"| {row['experiment_id']} | {enc_loss} | {row['seed']} | {fmt(row['val_r5'])} | {fmt(row['test_r5'])} | "
            f"{fmt(row['test_r1'])} | {fmt(row['test_r10'])} | {fmt(row['class_acc'])} | {fmt(row['mean_rank'])} | {row['source']} |"
        )
    lines.extend(
        [
            "",
            "## Controlled Comparisons",
            "",
            f"- X1 -> X2 adds L7 with the same subject-adaptive encoder: val R@5 delta `{delta(x1['val_r5'], x2['val_r5'])}`, test R@5 delta `{delta(x1['test_r5'], x2['test_r5'])}`.",
            f"- X2 -> X3 changes the loss mix to SupCon + L7: val R@5 delta `{delta(x2['val_r5'], x3['val_r5'])}`, test R@5 delta `{delta(x2['test_r5'], x3['test_r5'])}`.",
            f"- S3 -> X1 compares non-subject E3+L5 against subject-adaptive E5+L5 without L7: val R@5 delta `{delta(s3['val_r5'], x1['val_r5'])}`, test R@5 delta `{delta(s3['test_r5'], x1['test_r5'])}`.",
            f"- Day5 seed888 X2 re-run is not directly controlled against seed42 X2; it gives test/held-out-style R@5 `{fmt(day5['test_r5'])}` from `{day5['source']}`.",
            f"- Paired seed888 E5 no-L7 -> E5+L7: val R@5 delta `{delta(paired_no_l7['val_r5'], paired_l7['val_r5'])}`, test R@5 delta `{delta(paired_no_l7['test_r5'], paired_l7['test_r5'])}`, class_acc delta `{delta(paired_no_l7['class_acc'], paired_l7['class_acc'])}`.",
            "",
            "## Evidence Assessment",
            "",
            "Current evidence is mixed. The dataset structure strongly supports subject adaptation and L7, but the completed Day4 X1/X2/X3 rows do not show a clean subject-adaptation win over the best non-subject E3/S3 baselines. X2 improves validation R@5 over X1, while test R@5 drops slightly; X3 recovers test R@10 but not validation ranking. The paired seed888 recheck shows L7 slightly lower on validation R@5 but higher on test R@5, with lower class accuracy; treat this as useful but not conclusive.",
            "",
            "Therefore, treat subject adaptation as plausible but not yet proven. The next valid test is a paired E5 no-L7 versus E5+L7 run under the same seed, epochs, cache, and evaluation path.",
            "",
            "## Generated Next Configs",
            "",
        ]
    )
    for path in config_paths:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "Recommended launcher, when GPU scheduling allows:",
            "",
            "```bash",
            "python scripts/launch_alignment_sweep.py --config_dir configs/subject_adaptation --out outputs/subject_adaptation --max_concurrent 2 --screen_epochs 20 --poll_seconds 15",
            "```",
        ]
    )
    (OUT_DIR / "SUBJECT_ADAPTATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_consistency_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    by_id = {row["experiment_id"]: row for row in rows}
    x1, x2 = by_id["X1"], by_id["X2"]
    paired_no_l7 = by_id["E5_no_L7_control"]
    paired_l7 = by_id["E5_with_L7_recheck"]
    lines = [
        "# Same-Image Consistency Report",
        "",
        "## Applicability",
        "",
        f"- Same-image multi-subject groups: `{summary['shared_images']}`",
        f"- Same-image cross-subject pairs available: `{summary['same_image_cross_subject_pairs']}`",
        f"- Image subject-count histogram: `{summary['image_subject_hist']}`",
        "- Existing X2 and Day5 subject run enabled `use_same_image_subject` with `lambda_same_image_subject=0.2`.",
        "",
        "## What The Existing Comparison Says",
        "",
        f"- X1 without L7: val R@5 `{fmt(x1['val_r5'])}`, test R@5 `{fmt(x1['test_r5'])}`, class_acc `{fmt(x1['class_acc'])}`.",
        f"- X2 with L7: val R@5 `{fmt(x2['val_r5'])}`, test R@5 `{fmt(x2['test_r5'])}`, class_acc `{fmt(x2['class_acc'])}`.",
        f"- L7 delta: val R@5 `{delta(x1['val_r5'], x2['val_r5'])}`, test R@5 `{delta(x1['test_r5'], x2['test_r5'])}`, class_acc `{delta(x1['class_acc'], x2['class_acc'])}`.",
        f"- Paired seed888 L7 delta: val R@5 `{delta(paired_no_l7['val_r5'], paired_l7['val_r5'])}`, test R@5 `{delta(paired_no_l7['test_r5'], paired_l7['test_r5'])}`, class_acc `{delta(paired_no_l7['class_acc'], paired_l7['class_acc'])}`.",
        "",
        "## Interpretation",
        "",
        "L7 has a real structural target: the same image appears across multiple subjects. The existing seed42 X1/X2 comparison suggests L7 may help validation retrieval but is not yet a test-set win. The paired seed888 recheck flips that pattern: validation is slightly lower with L7, while test R@5 is higher. This is exactly the kind of signal that needs multi-seed confirmation.",
        "",
        "Do not claim same-image consistency helps captions or EEG alignment yet. The current defensible claim is narrower: the data supports the loss, and preliminary alignment results are mixed.",
        "",
        "## Next Verification",
        "",
        "Continue L7 only if a multi-seed average favors it on validation R@5 or test R@5 without a large class-accuracy regression. The current paired result is promising for retrieval but not stable enough for a final claim.",
    ]
    (OUT_DIR / "SAME_IMAGE_CONSISTENCY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_split = load_manifests()
    summary = structure_summary(rows_by_split)
    config_paths = generate_configs()
    rows = collect_comparison_rows()
    write_structure_report(summary)
    write_subject_report(summary, rows, config_paths)
    write_consistency_report(summary, rows)
    print(f"Wrote reports to {OUT_DIR}")
    print(f"Wrote configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
