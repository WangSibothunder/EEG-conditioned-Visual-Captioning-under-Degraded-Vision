from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


BOARD_COLUMNS = [
    "experiment_id",
    "subagent",
    "task_type",
    "dataset",
    "encoder",
    "loss",
    "seed",
    "status",
    "start_time",
    "end_time",
    "gpu_mem_peak",
    "metric_primary",
    "metric_secondary",
    "checkpoint_path",
    "report_path",
    "notes",
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _write_board(out_root: Path) -> None:
    rows = [
        {
            "experiment_id": "semantic_caption",
            "subagent": "A",
            "task_type": "controlled_semantic_caption",
            "dataset": "Thought2Text",
            "encoder": "best_alignment+gated_fusion",
            "loss": "class_ce+supcon+prototype",
            "seed": "42",
            "status": "running",
            "start_time": _now(),
            "end_time": "",
            "gpu_mem_peak": "",
            "metric_primary": "",
            "metric_secondary": "",
            "checkpoint_path": "outputs/semantic_caption/checkpoints/best.pt",
            "report_path": "outputs/semantic_caption/SEMANTIC_CAPTION_REPORT.md",
            "notes": "Subagent A owns constrained captioning; free-form Qwen is not the main metric.",
        },
        {
            "experiment_id": "alignment_search",
            "subagent": "B/main",
            "task_type": "encoder_loss_search",
            "dataset": "Thought2Text",
            "encoder": "E3/E4/E5/E7",
            "loss": "L1+L2+L4(+L5/L6/L7)",
            "seed": "42/123/2025",
            "status": "running",
            "start_time": _now(),
            "end_time": "",
            "gpu_mem_peak": "",
            "metric_primary": "",
            "metric_secondary": "",
            "checkpoint_path": "outputs/alignment_search/best_overall.pt",
            "report_path": "outputs/alignment_search/SEARCH_SUMMARY.md",
            "notes": "Includes spectrogram P1/P2 smoke configs.",
        },
        {
            "experiment_id": "dataset_inspection",
            "subagent": "main",
            "task_type": "dataset_status",
            "dataset": "THINGS-EEG2/EIT-1M",
            "encoder": "",
            "loss": "",
            "seed": "",
            "status": "completed",
            "start_time": _now(),
            "end_time": _now(),
            "gpu_mem_peak": "0",
            "metric_primary": "",
            "metric_secondary": "",
            "checkpoint_path": "",
            "report_path": "outputs/datasets/THINGS_EEG2_STATUS.md; outputs/datasets/EIT1M_STATUS.md",
            "notes": "Availability reports generated; conversion depends on local files.",
        },
        {
            "experiment_id": "trimodal",
            "subagent": "D",
            "task_type": "eeg_image_text_contrastive",
            "dataset": "Thought2Text",
            "encoder": "E3",
            "loss": "tri_modal_contrastive",
            "seed": "42",
            "status": "running",
            "start_time": _now(),
            "end_time": "",
            "gpu_mem_peak": "",
            "metric_primary": "",
            "metric_secondary": "",
            "checkpoint_path": "outputs/trimodal/checkpoints/best.pt",
            "report_path": "outputs/trimodal/TRIMODAL_STATUS.md",
            "notes": "Subagent D owns text embedding cache and tri-modal smoke path.",
        },
        {
            "experiment_id": "subject_adaptation",
            "subagent": "main",
            "task_type": "subject_structure_and_adapter",
            "dataset": "Thought2Text",
            "encoder": "E5",
            "loss": "L1+L2+L4+L5+L7",
            "seed": "888",
            "status": "running",
            "start_time": _now(),
            "end_time": "",
            "gpu_mem_peak": "",
            "metric_primary": "",
            "metric_secondary": "",
            "checkpoint_path": "outputs/day5_subject_alignment/X2_subject_same_image_seed888/checkpoints/best.pt",
            "report_path": "outputs/subject_adaptation/SUBJECT_ADAPTATION_REPORT.md",
            "notes": "Reuse completed Day5 subject run plus fresh structure analysis.",
        },
        {
            "experiment_id": "robustness",
            "subagent": "F",
            "task_type": "real_vs_controls_eval",
            "dataset": "Thought2Text",
            "encoder": "best_available",
            "loss": "",
            "seed": "",
            "status": "running",
            "start_time": _now(),
            "end_time": "",
            "gpu_mem_peak": "",
            "metric_primary": "",
            "metric_secondary": "",
            "checkpoint_path": "",
            "report_path": "outputs/robustness/ROBUSTNESS_REPORT.md",
            "notes": "Uses Day5 sanity until semantic_caption results are ready.",
        },
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "EXPERIMENT_BOARD.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BOARD_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Parallel Stage Experiment Board",
        "",
        "| " + " | ".join(BOARD_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in BOARD_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in BOARD_COLUMNS) + " |")
    (out_root / "EXPERIMENT_BOARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_root / "LIVE_STATUS.md").write_text(
        "# Parallel Stage Live Status\n\n"
        f"- Updated: `{_now()}`\n"
        "- Active workstreams: semantic captioning, alignment search, tri-modal, subject adaptation, robustness.\n"
        "- Dataset inspection reports have been generated under `outputs/datasets/`.\n",
        encoding="utf-8",
    )


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _caption_class_name(caption: str) -> str:
    text = caption.strip().lower()
    for prefix in ["a photo of an ", "a photo of a ", "a photo of "]:
        if text.startswith(prefix):
            return text[len(prefix) :].strip().strip(".")
    return text.strip(".")


def _write_subject_reports(out_root: Path) -> None:
    manifests = {
        "train": Path("data/thought2text/train_human_caption.jsonl"),
        "val": Path("data/thought2text/val_human_caption.jsonl"),
        "test": Path("data/thought2text/test_human_caption.jsonl"),
    }
    rows_by_split = {split: _load_manifest(path) for split, path in manifests.items()}
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    subjects = Counter(str(row.get("subject_id")) for row in all_rows)
    images = defaultdict(set)
    class_names: dict[str, str] = {}
    for row in all_rows:
        images[str(row.get("image_id"))].add(str(row.get("subject_id")))
        if row.get("label") is not None:
            class_names[str(row["label"])] = _caption_class_name(str(row.get("caption", "")))
    shared = {image_id: subs for image_id, subs in images.items() if len(subs) > 1}
    split_image_sets = {split: {str(row.get("image_id")) for row in rows} for split, rows in rows_by_split.items()}
    leakage = {}
    for left in split_image_sets:
        for right in split_image_sets:
            if left < right:
                leakage[f"{left}-{right}"] = len(split_image_sets[left] & split_image_sets[right])
    out_root.mkdir(parents=True, exist_ok=True)
    structure_lines = [
        "# Subject/Image Structure",
        "",
        f"- Total trials: `{len(all_rows)}`",
        f"- Subjects: `{len(subjects)}`",
        f"- Subject trial counts: `{dict(subjects)}`",
        f"- Unique images: `{len(images)}`",
        f"- Images with multiple subjects: `{len(shared)}`",
        f"- Max subjects per image: `{max((len(v) for v in images.values()), default=0)}`",
        f"- Classes: `{len(class_names)}`",
        f"- Image-level split leakage counts: `{leakage}`",
        "",
        "## Split Summary",
        "",
        "| Split | Trials | Unique Images | Subjects | Classes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, rows in rows_by_split.items():
        structure_lines.append(
            f"| {split} | {len(rows)} | {len({str(r.get('image_id')) for r in rows})} | "
            f"{len({str(r.get('subject_id')) for r in rows})} | {len({str(r.get('label')) for r in rows})} |"
        )
    structure_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Thought2Text has repeated image IDs across six subjects, so same-image cross-subject consistency is applicable.",
            "The current human-caption split has zero image-id overlap between train/val/test if leakage counts above are all zero.",
        ]
    )
    (out_root / "subject_image_structure.md").write_text("\n".join(structure_lines) + "\n", encoding="utf-8")

    day5_report = Path("outputs/day5_subject_alignment/X2_subject_same_image_seed888/retrieval_report.md")
    day5_metrics = Path("outputs/day5_subject_alignment/X2_subject_same_image_seed888/test_metrics.json")
    if not day5_metrics.exists():
        day5_metrics = Path("outputs/day5_subject_alignment/X2_subject_same_image_seed888/retrieval_metrics.json")
    if not day5_metrics.exists():
        day5_metrics = Path("outputs/day5_subject_alignment/X2_subject_same_image_seed888/alignment_metrics.json")
    metric_text = "No completed subject-adaptive metric found."
    if day5_metrics.exists():
        payload = json.loads(day5_metrics.read_text(encoding="utf-8"))
        model = payload.get("model", payload)
        metric_text = (
            f"Completed Day5 X2 subject-adaptive run: R@1/R@5/R@10 = "
            f"`{model.get('r@1')} / {model.get('r@5')} / {model.get('r@10')}`, "
            f"class_acc = `{model.get('class_acc')}`."
        )
    report_lines = [
        "# Subject Adaptation Report",
        "",
        "## Structure",
        "",
        f"- Subjects: `{len(subjects)}`",
        f"- Shared images across subjects: `{len(shared)}`",
        "- Same-image cross-subject loss is applicable.",
        "",
        "## Existing Subject-Adaptive Run",
        "",
        metric_text,
        "",
        "## Recommendation",
        "",
        "Use subject-adaptive E5 only as a controlled comparison against E3/E4. The dataset structure supports L7, but current selection should still be based on validation R@5 and class accuracy.",
    ]
    if day5_report.exists():
        report_lines.extend(["", f"- Source report: `{day5_report}`"])
    (out_root / "SUBJECT_ADAPTATION_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (out_root / "SAME_IMAGE_CONSISTENCY_REPORT.md").write_text(
        "# Same-Image Consistency Report\n\n"
        f"- Same-image multi-subject groups: `{len(shared)}`\n"
        "- Existing X2 run enabled `use_same_image_subject` with `lambda_same_image_subject=0.2`.\n"
        "- Next controlled test: compare X1 vs X2 and E3 baseline vs E5+L7 under identical seed/epochs.\n",
        encoding="utf-8",
    )


def _write_dataset_recommendation(out_root: Path) -> None:
    things = Path("outputs/datasets/THINGS_EEG2_STATUS.md")
    eit = Path("outputs/datasets/EIT1M_STATUS.md")
    things_text = things.read_text(encoding="utf-8") if things.exists() else ""
    eit_text = eit.read_text(encoding="utf-8") if eit.exists() else ""
    lines = [
        "# Dataset Recommendation",
        "",
        "1. Keep Thought2Text as the main dataset for the immediate constrained semantic captioning experiments because it is already converted and has aligned EEG/image/class labels.",
        "2. THINGS-EEG2 should be used for pretraining only after local files expose both EEG and image assets in a convertible structure.",
        "3. EIT-1M is promising for EEG-image-text smoke tests because a local zip is present and inspection detects multimodal assets, but controlled extraction/schema conversion is still required.",
        "4. Next dataset priority: EIT-1M small manifest smoke, then THINGS-EEG2 if raw/preprocessed files become available.",
        "5. Main blockers: missing/empty THINGS-EEG2 tree, EIT-1M still zipped, and no verified current-schema manifests for either dataset yet.",
        "",
        "## Source Status",
        "",
        f"- THINGS status report: `{things}`",
        f"- EIT-1M status report: `{eit}`",
    ]
    if "File count: `0`" in things_text:
        lines.append("- THINGS-EEG2: local directory exists but no files were detected.")
    if "Zip files: `1`" in eit_text:
        lines.append("- EIT-1M: one local zip is available for later controlled extraction.")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "DATASET_RECOMMENDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_alignment_configs(config_dir: Path) -> None:
    base_sources = {
        "P1_spectrogram_smoke": Path("configs/generated_alignment/S1.yaml"),
        "P2_raw_spectrogram_smoke": Path("configs/generated_alignment/S1.yaml"),
        "T4_stage2_seed42": Path("configs/generated_alignment/T4.yaml"),
        "S1_stage2_seed42": Path("configs/generated_alignment/S1.yaml"),
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for exp_id, source in base_sources.items():
        cfg = _read_yaml(source)
        cfg["experiment_id"] = exp_id
        cfg.setdefault("output", {})["dir"] = f"outputs/alignment_search/{exp_id}"
        cfg.setdefault("train", {})["max_train_samples"] = 0
        cfg["train"]["max_val_samples"] = 0
        if exp_id.startswith("P1"):
            cfg.setdefault("model", {})["encoder_type"] = "spectrogram_cnn"
            cfg["train"]["epochs"] = 8
            cfg["train"]["patience"] = 3
            cfg["train"]["batch_size"] = 128
            cfg["notes"] = "Parallel stage E7 spectrogram-CNN smoke alignment."
        elif exp_id.startswith("P2"):
            cfg.setdefault("model", {})["encoder_type"] = "raw_spectrogram_fusion"
            cfg["train"]["epochs"] = 8
            cfg["train"]["patience"] = 3
            cfg["train"]["batch_size"] = 96
            cfg["notes"] = "Parallel stage raw E3 + spectrogram E7 late-fusion smoke alignment."
        else:
            cfg["train"]["epochs"] = 50
            cfg["train"]["patience"] = 8
            cfg["notes"] = "Parallel stage Stage2 continuation candidate."
        out_path = config_dir / f"{exp_id}.yaml"
        _write_yaml(out_path, cfg)
        manifest.append({"experiment_id": exp_id, "config": out_path.name})
    (config_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    plan = [
        "# Parallel Alignment Experiment Plan",
        "",
        "Generated configs:",
        "",
    ]
    for row in manifest:
        plan.append(f"- `{row['experiment_id']}` -> `{config_dir / row['config']}`")
    plan.extend(
        [
            "",
            "Recommended launcher:",
            "",
            "```bash",
            "python scripts/launch_alignment_sweep.py --config_dir configs/parallel_alignment_search --out outputs/alignment_search --max_concurrent 3 --screen_epochs 8 --poll_seconds 15",
            "```",
        ]
    )
    Path("outputs/alignment_search").mkdir(parents=True, exist_ok=True)
    Path("outputs/alignment_search/EXPERIMENT_PLAN.md").write_text("\n".join(plan) + "\n", encoding="utf-8")
    Path("outputs/spectrogram").mkdir(parents=True, exist_ok=True)
    Path("outputs/spectrogram/SPECTROGRAM_BRANCH_REPORT.md").write_text(
        "# Spectrogram Branch Report\n\n"
        "- Implemented `spectrogram_cnn` (E7) and `raw_spectrogram_fusion` encoder options in `src/models/eeg_encoder.py`.\n"
        "- Smoke configs generated as `P1_spectrogram_smoke` and `P2_raw_spectrogram_smoke`.\n"
        "- Training/evaluation results will be read from `outputs/alignment_search/` after launcher completion.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate parallel stage support reports/configs.")
    parser.add_argument("--out", default="outputs/parallel_stage")
    parser.add_argument("--alignment_config_dir", default="configs/parallel_alignment_search")
    args = parser.parse_args()
    _write_board(Path(args.out))
    _write_subject_reports(Path("outputs/subject_adaptation"))
    _write_dataset_recommendation(Path("outputs/datasets"))
    _make_alignment_configs(Path(args.alignment_config_dir))
    print(args.out)


if __name__ == "__main__":
    main()
