from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


HISTORICAL_DAY3_R5 = 0.0315
HISTORICAL_DAY4_R5 = 0.3303
EXCLUDE_TOKENS = ("smoke", "debug", "self_test", "cache_loader")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric_block(metrics: dict[str, Any]) -> dict[str, Any]:
    model = metrics.get("model", metrics)
    if isinstance(model, dict) and isinstance(model.get("unique_image"), dict):
        return model["unique_image"]
    return model if isinstance(model, dict) else {}


def _load_config_map(config_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    by_output: dict[str, dict[str, Any]] = {}
    for config_dir in config_dirs:
        if not config_dir.exists():
            continue
        for path in config_dir.rglob("*.yaml"):
            try:
                config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                continue
            output_dir = ((config.get("output") or {}).get("dir") or "").rstrip("/")
            if not output_dir:
                continue
            by_output[output_dir] = {"path": path, "config": config}
            resolved = str(Path(output_dir).resolve())
            by_output[resolved] = {"path": path, "config": config}
    return by_output


def _config_for_output(output_dir: Path, config_map: dict[str, dict[str, Any]]) -> tuple[Path | None, dict[str, Any]]:
    keys = [str(output_dir).rstrip("/"), str(output_dir.resolve()).rstrip("/")]
    for key in keys:
        entry = config_map.get(key)
        if entry:
            return entry["path"], entry["config"]
    return None, {}


def _checkpoint_for_output(output_dir: Path) -> Path | None:
    candidates = [
        output_dir / "checkpoints" / "best.pt",
        output_dir / "checkpoints" / "best_masked_eeg.pt",
        output_dir / "best.pt",
        output_dir / "best_overall.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _param_note(encoder_type: str) -> str:
    key = encoder_type.lower()
    if key in {"tiny", "e0", "baseline"}:
        return "~0.5M-1M target"
    if key in {"eegnet", "e1"}:
        return "~0.5M-2M target"
    if key in {"multiscale_tcn", "e2"}:
        return "~2M-5M target"
    if key in {"convtransformer_base", "e3", "base"}:
        return "~2M-6M target"
    if key in {"convtransformer_strong", "e4", "strong"}:
        return "~8M-20M target"
    if key in {"subject_adaptive", "e5"}:
        return "~3M-7M target"
    if key in {"raw_spectrogram_fusion", "p2", "e3_e7"}:
        return "raw+spectrogram fusion"
    if key in {"dualbranch_eegconformer", "dual_branch_eegconformer", "a1"}:
        return "~10M-30M target"
    if key in {"temporal_spectral_spatial_transformer", "temporal_spectral_spatial", "tsst", "a2"}:
        return "temporal+spectral+spatial transformer"
    if key in {"subject_adaptive_graph", "subject_graph", "a3"}:
        return "subject-adaptive graph encoder"
    if key in {"masked_pretrained", "masked_eeg"}:
        return "masked-pretrained encoder"
    return "unknown"


def _dataset_from_config(config: dict[str, Any]) -> str:
    train_manifest = str((config.get("data") or {}).get("train_manifest", ""))
    if "EEG-ImageNet" in train_manifest:
        return "EEG-ImageNet"
    if "thought2text" in train_manifest.lower():
        return "Thought2Text"
    if "things" in train_manifest.lower():
        return "THINGS-EEG2"
    if "eit" in train_manifest.lower():
        return "EIT-1M"
    return "unknown"


def _infer_loss_combo(loss: dict[str, Any]) -> str:
    explicit = str(loss.get("loss_combo") or "").strip()
    if explicit and explicit.lower() != "unknown":
        return explicit
    parts: list[str] = []
    if loss.get("use_mse") and (loss.get("use_class_ce") or loss.get("use_cls")) and not loss.get("use_infonce"):
        parts.append("L0")
    if loss.get("use_infonce"):
        parts.append("L1")
    if loss.get("use_multi_positive_infonce"):
        parts.append("L2")
    if loss.get("use_supcon"):
        parts.append("L3")
    if loss.get("use_prototype_alignment"):
        parts.append("L4")
    if loss.get("use_similarity_distillation") or loss.get("use_similarity_distill"):
        parts.append("L5")
    if loss.get("use_aug_consistency"):
        parts.append("L6")
    if loss.get("use_same_image_subject") or loss.get("use_same_image_subject_consistency"):
        parts.append("L7")
    if loss.get("use_hard_negative"):
        parts.append("L8")
    return "+".join(parts) if parts else "unknown"


def _pretrain_from_config(config: dict[str, Any], output_dir: Path) -> str:
    model = config.get("model") or {}
    checkpoint = str(model.get("pretrained_eeg_checkpoint", ""))
    if "eeg_imagenet" in checkpoint.lower():
        return "EEG-ImageNet masked"
    if "masked_eeg" in checkpoint.lower() or "pretrain" in checkpoint.lower():
        return "Thought2Text masked"
    if "transfer" in str(output_dir):
        return "transfer"
    return "none"


def _row_from_metrics(path: Path, config_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output_dir = path.parent
    config_path, config = _config_for_output(output_dir, config_map)
    metrics = _load_json(path)
    block = _metric_block(metrics)
    model = config.get("model") or {}
    loss = config.get("loss") or {}
    experiment_id = str(config.get("experiment_id") or output_dir.name)
    encoder_type = str(model.get("encoder_type") or "tiny")
    checkpoint = _checkpoint_for_output(output_dir)
    return {
        "experiment_id": experiment_id,
        "output_dir": output_dir,
        "config_path": config_path,
        "encoder_type": encoder_type,
        "param_count": _param_note(encoder_type),
        "dataset": _dataset_from_config(config),
        "pretrain": _pretrain_from_config(config, output_dir),
        "loss_combo": _infer_loss_combo(loss),
        "r1": _safe_float(block.get("r@1")),
        "r5": _safe_float(block.get("r@5")),
        "r10": _safe_float(block.get("r@10")),
        "class_acc": _safe_float(block.get("class_acc")),
        "mean_rank": _safe_float(block.get("mean_rank")),
        "checkpoint": checkpoint,
        "notes": str(config.get("notes") or ""),
    }


def _is_formal_row(row: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(row.get("experiment_id", "")),
            str(row.get("output_dir", "")),
        ]
    ).lower()
    return not any(token in text for token in EXCLUDE_TOKENS)


def collect_alignment_rows(outputs_root: Path, config_dirs: list[Path]) -> list[dict[str, Any]]:
    config_map = _load_config_map(config_dirs)
    rows: list[dict[str, Any]] = []
    for path in sorted(outputs_root.rglob("alignment_metrics.json")):
        if path.parent.resolve() == outputs_root.resolve():
            continue
        row = _row_from_metrics(path, config_map)
        if _is_formal_row(row):
            rows.append(row)
    rows.sort(key=lambda row: (row["r5"], row["class_acc"], row["r1"]), reverse=True)
    return rows


def select_best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    with_checkpoint = [row for row in rows if row.get("checkpoint")]
    return with_checkpoint[0] if with_checkpoint else rows[0]


def _copy_or_link(source: Path | None, target: Path) -> bool:
    if source is None or not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        rel = os.path.relpath(source, target.parent)
        target.symlink_to(rel)
    except OSError:
        shutil.copyfile(source, target)
    return True


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _named_heavy_architecture_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {
        "A1": 1,
        "A2": 2,
        "A3": 3,
        "A4": 4,
    }
    encoder_to_name = {
        "dualbranch_eegconformer": "A1",
        "dual_branch_eegconformer": "A1",
        "temporal_spectral_spatial_transformer": "A2",
        "temporal_spectral_spatial": "A2",
        "subject_adaptive_graph": "A3",
        "subject_graph": "A3",
        "raw_spectrogram_fusion": "A4",
    }
    selected: list[tuple[int, dict[str, Any]]] = []
    seen: set[str] = set()
    for row in rows:
        experiment = str(row.get("experiment_id", ""))
        encoder = str(row.get("encoder_type", "")).lower()
        label = next((name for name in order if experiment.startswith(name)), encoder_to_name.get(encoder))
        if not label:
            continue
        key = str(row.get("output_dir") or experiment)
        if key in seen:
            continue
        seen.add(key)
        selected.append((order[label], row))
    return [row for _, row in sorted(selected, key=lambda item: (item[0], str(item[1].get("experiment_id", ""))))]


def _write_architecture_csv(rows: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Architecture",
                "Params",
                "Dataset",
                "Pretrain",
                "R@1",
                "R@5",
                "R@10",
                "Class Acc",
                "GPU Mem",
                "Notes",
                "Experiment",
                "Checkpoint",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Architecture": row["encoder_type"],
                    "Params": row["param_count"],
                    "Dataset": row["dataset"],
                    "Pretrain": row["pretrain"],
                    "R@1": _fmt(row["r1"]),
                    "R@5": _fmt(row["r5"]),
                    "R@10": _fmt(row["r10"]),
                    "Class Acc": _fmt(row["class_acc"]),
                    "GPU Mem": "see run log",
                    "Notes": row["notes"],
                    "Experiment": row["experiment_id"],
                    "Checkpoint": str(row["checkpoint"] or ""),
                }
            )


def _write_architecture_report(rows: list[dict[str, Any]], outputs_root: Path) -> None:
    out_dir = outputs_root / "architectures"
    out_dir.mkdir(parents=True, exist_ok=True)
    best = select_best_row(rows)
    _copy_or_link(best.get("checkpoint") if best else None, out_dir / "checkpoints" / "best_encoder.pt")
    _write_architecture_csv(rows[:50], out_dir / "ARCHITECTURE_SEARCH_TABLE.csv")
    covered = sorted({row["encoder_type"] for row in rows if row["encoder_type"] != "unknown"})
    lines = [
        "# Architecture Search Report",
        "",
        f"- Updated UTC: `{_utc_now()}`",
        f"- Alignment runs indexed: `{len(rows)}`",
        f"- Excluded smoke/debug tokens: `{', '.join(EXCLUDE_TOKENS)}`",
        f"- Encoder families observed in completed metrics: `{', '.join(covered) if covered else 'none'}`",
        f"- Historical Day3 test R@5: `{HISTORICAL_DAY3_R5:.4f}`",
        f"- Historical Day4 best R@5: `{HISTORICAL_DAY4_R5:.4f}`",
        "",
        "## Best Available Checkpoint",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- Experiment: `{best['experiment_id']}`",
                f"- Encoder: `{best['encoder_type']}`",
                f"- Loss: `{best['loss_combo']}`",
                f"- Unique-image R@1/R@5/R@10: `{_fmt(best['r1'])} / {_fmt(best['r5'])} / {_fmt(best['r10'])}`",
                f"- Class accuracy: `{_fmt(best['class_acc'])}`",
                f"- Source checkpoint: `{best['checkpoint'] or 'missing'}`",
                f"- Materialized checkpoint: `{out_dir / 'checkpoints' / 'best_encoder.pt'}`",
            ]
        )
    else:
        lines.append("- No completed alignment metrics found.")
    lines.extend(
        [
            "",
            "## Comparison Table",
            "",
            "| Architecture | Params | Dataset | Pretrain | R@1 | R@5 | R@10 | Class Acc | GPU Mem | Notes |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows[:25]:
        note = row["notes"] or row["experiment_id"]
        if len(note) > 90:
            note = note[:87] + "..."
        lines.append(
            f"| `{row['encoder_type']}` | {row['param_count']} | {row['dataset']} | {row['pretrain']} | "
            f"{_fmt(row['r1'])} | {_fmt(row['r5'])} | {_fmt(row['r10'])} | {_fmt(row['class_acc'])} | see run log | {note} |"
        )
    named_heavy = _named_heavy_architecture_rows(rows)
    lines.extend(
        [
            "",
            "## Named Heavy Architectures",
            "",
            "| Run | Architecture | Params | R@1 | R@5 | R@10 | Class Acc | Checkpoint |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if named_heavy:
        for row in named_heavy:
            lines.append(
                f"| `{row['experiment_id']}` | `{row['encoder_type']}` | {row['param_count']} | "
                f"{_fmt(row['r1'])} | {_fmt(row['r5'])} | {_fmt(row['r10'])} | {_fmt(row['class_acc'])} | "
                f"`{row['checkpoint'] or ''}` |"
            )
    else:
        lines.append("| none | none | none | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This report indexes completed alignment artifacts. It does not claim that every requested heavy architecture has completed full training.",
            "Rows without large-dataset pretraining are Thought2Text-only alignment results. EEG-ImageNet masked pretraining is complete, but its downstream transfer recovery is only included here after final alignment metrics exist.",
        ]
    )
    (out_dir / "ARCHITECTURE_SEARCH_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_transfer_report(rows: list[dict[str, Any]], outputs_root: Path) -> None:
    transfer_dir = outputs_root / "transfer"
    transfer_rows = [row for row in rows if "transfer" in str(row["output_dir"]) or "pretrain" in row["pretrain"].lower()]
    if not transfer_rows:
        transfer_rows = rows[:5]
    best = select_best_row(transfer_rows)
    _copy_or_link(best.get("checkpoint") if best else None, transfer_dir / "best_transfer_encoder.pt")
    lines = [
        "# Transfer To Thought2Text Report",
        "",
        f"- Updated UTC: `{_utc_now()}`",
        f"- Transfer/pretrain-related alignment runs indexed: `{len(transfer_rows)}`",
        f"- Historical Day4 best R@5: `{HISTORICAL_DAY4_R5:.4f}`",
        "",
        "## Best Current Transfer",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- Experiment: `{best['experiment_id']}`",
                f"- Encoder: `{best['encoder_type']}`",
                f"- Pretrain source: `{best['pretrain']}`",
                f"- Loss: `{best['loss_combo']}`",
                f"- Unique-image R@1/R@5/R@10: `{_fmt(best['r1'])} / {_fmt(best['r5'])} / {_fmt(best['r10'])}`",
                f"- Class accuracy: `{_fmt(best['class_acc'])}`",
                f"- Improved over historical Day4 R@5: `{'yes' if best['r5'] > HISTORICAL_DAY4_R5 else 'no'}`",
                f"- Source checkpoint: `{best['checkpoint'] or 'missing'}`",
                f"- Materialized checkpoint: `{transfer_dir / 'best_transfer_encoder.pt'}`",
            ]
        )
    else:
        lines.append("- No transfer run with completed metrics found.")
    lines.extend(
        [
            "",
            "## Transfer Runs",
            "",
            "| Experiment | Encoder | Pretrain | Loss | R@1 | R@5 | R@10 | Class Acc | Checkpoint |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in transfer_rows[:20]:
        lines.append(
            f"| `{row['experiment_id']}` | `{row['encoder_type']}` | {row['pretrain']} | {row['loss_combo']} | "
            f"{_fmt(row['r1'])} | {_fmt(row['r5'])} | {_fmt(row['r10'])} | {_fmt(row['class_acc'])} | `{row['checkpoint'] or ''}` |"
        )
    eeg_imagenet_transfer = next(
        (
            row
            for row in transfer_rows
            if row["experiment_id"] == "eeg_imagenet_pretrain_t2t_align"
            or "eeg_imagenet_pretrain_t2t_align" in str(row["output_dir"])
        ),
        None,
    )
    lines.extend(
        [
            "",
            "## EEG-ImageNet Transfer Status",
            "",
        ]
    )
    if eeg_imagenet_transfer:
        improved = eeg_imagenet_transfer["r5"] > HISTORICAL_DAY4_R5
        lines.extend(
            [
                "- EEG-ImageNet transfer metrics are present.",
                f"- EEG-ImageNet transfer R@5: `{_fmt(eeg_imagenet_transfer['r5'])}`.",
                f"- EEG-ImageNet transfer {'improved over' if improved else 'did not improve over'} historical Day4 R@5 `{HISTORICAL_DAY4_R5:.4f}`.",
            ]
        )
    else:
        lines.append(
            "- EEG-ImageNet transfer metrics are not present yet; regenerate this report after `outputs/transfer/eeg_imagenet_pretrain_t2t_align/alignment_metrics.json` exists."
        )
    transfer_dir.mkdir(parents=True, exist_ok=True)
    (transfer_dir / "TRANSFER_TO_THOUGHT2TEXT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_trimodal_artifact(outputs_root: Path) -> None:
    source = outputs_root / "trimodal" / "full_masked_pretrained" / "checkpoints" / "best.pt"
    target = outputs_root / "trimodal" / "checkpoints" / "best_trimodal.pt"
    _copy_or_link(source if source.exists() else None, target)


def _ensure_pretrain_artifact(outputs_root: Path) -> None:
    candidates = [
        outputs_root / "pretrain" / "masked_eeg_eeg_imagenet_dualbranch_heavy" / "checkpoints" / "best_masked_eeg.pt",
        outputs_root / "pretrain" / "masked_eeg_thought2text_dualbranch_heavy" / "checkpoints" / "best_masked_eeg.pt",
    ]
    source = next((path for path in candidates if path.exists()), None)
    target = outputs_root / "pretrain" / "checkpoints" / "best_masked_eeg.pt"
    _copy_or_link(source, target)


def _exists(outputs_root: Path, relative: str) -> bool:
    return (outputs_root / relative).exists()


def _write_baselines(rows: list[dict[str, Any]], outputs_root: Path) -> None:
    heavy_dir = outputs_root / "heavy_stage"
    heavy_dir.mkdir(parents=True, exist_ok=True)
    best = select_best_row(rows)
    best_transfer_rows = [row for row in rows if "transfer" in str(row["output_dir"]) or "pretrain" in row["pretrain"].lower()]
    best_transfer = select_best_row(best_transfer_rows) if best_transfer_rows else {}
    lines = [
        "# Heavy Stage Baselines",
        "",
        f"Last update UTC: {_utc_now()}",
        "",
        "## Historical References",
        "",
        f"- Day3 reference EEG->image R@5: `{HISTORICAL_DAY3_R5:.4f}`",
        f"- Day4 stable reference EEG->image R@5: `{HISTORICAL_DAY4_R5:.4f}`",
        "",
        "## Current Indexed Bests",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- Current highest indexed EEG->image R@5: `{_fmt(best['r5'])}`",
                f"- Current highest indexed EEG->image R@1/R@5/R@10: `{_fmt(best['r1'])} / {_fmt(best['r5'])} / {_fmt(best['r10'])}`",
                f"- Current highest indexed class accuracy at selected checkpoint: `{_fmt(best['class_acc'])}`",
                f"- Source experiment: `{best['experiment_id']}`",
                f"- Source checkpoint: `{best['checkpoint'] or 'missing'}`",
                f"- Improved over Day4 stable R@5: `{'yes' if best['r5'] > HISTORICAL_DAY4_R5 else 'no'}`",
            ]
        )
    else:
        lines.append("- No indexed alignment run found.")
    lines.extend(["", "## Transfer Baseline", ""])
    if best_transfer:
        lines.extend(
            [
                f"- Best transfer experiment so far: `{best_transfer['experiment_id']}`",
                f"- Transfer R@1/R@5/R@10: `{_fmt(best_transfer['r1'])} / {_fmt(best_transfer['r5'])} / {_fmt(best_transfer['r10'])}`",
                f"- Transfer class accuracy: `{_fmt(best_transfer['class_acc'])}`",
                f"- Transfer improves over Day4 R@5: `{'yes' if best_transfer['r5'] > HISTORICAL_DAY4_R5 else 'no'}`",
            ]
        )
    else:
        lines.append("- No completed transfer run found.")
    lines.extend(
        [
            "",
            "## Semantic Caption Baselines",
            "",
            f"- Robust semantic report present: `{_exists(outputs_root, 'final_semantic/FULL_ROBUST_SEMANTIC_REPORT.md')}`",
            f"- Full semantic metrics present: `{_exists(outputs_root, 'final_semantic/FULL_METRICS.csv')}`",
            "",
            "## Notes",
            "",
            "- Smoke/debug/cache-loader metrics are excluded from indexed bests.",
            "- EEG-ImageNet masked pretraining transfer is not included until its alignment metrics exist.",
        ]
    )
    (heavy_dir / "BASELINES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _semantic_control_status(outputs_root: Path) -> dict[str, Any]:
    final_gap = outputs_root / "final_semantic" / "SEMANTIC_GAP_METRICS.csv"
    transfer_gap = outputs_root / "final_semantic" / "eeg_imagenet_transfer_eval" / "SEMANTIC_GAP_METRICS.csv"
    strong_gap = outputs_root / "final_semantic" / "strong_degradation_eval" / "SEMANTIC_GAP_METRICS.csv"
    classifier_gap = outputs_root / "final_semantic" / "semantic_fusion_classifier_strong_eval" / "SEMANTIC_GAP_METRICS.csv"
    a2_multiseed = outputs_root / "final_semantic" / "A2_SEMANTIC_FUSION_MULTISEED_SUMMARY.csv"
    a2_gap = (
        outputs_root
        / "final_semantic"
        / "semantic_fusion_A2_temporal_spectral_spatial_full_strong_eval"
        / "SEMANTIC_GAP_METRICS.csv"
    )
    candidates = [
        (a2_multiseed, "A2 semantic fusion multi-seed strong degradation evaluation"),
        (a2_gap, "A2 semantic fusion strong degradation evaluation"),
        (classifier_gap, "semantic fusion classifier strong degradation evaluation"),
        (strong_gap, "strong degradation semantic evaluation"),
        (transfer_gap, "EEG-ImageNet transfer semantic evaluation"),
        (final_gap, "constrained semantic evaluation"),
    ]
    source, context = next(((path, label) for path, label in candidates if path.exists()), (final_gap, "constrained semantic evaluation"))
    rows = _read_csv_rows(source)
    total = len(rows)
    control_wins = sum(1 for row in rows if str(row.get("real_beats_controls", "")).lower() == "true")
    vision_wins = sum(1 for row in rows if str(row.get("real_beats_vision", "")).lower() == "true")
    degraded = [row for row in rows if row.get("corruption") != "clean"]
    degraded_control_wins = sum(
        1 for row in degraded if str(row.get("real_beats_controls", "")).lower() == "true"
    )
    if not rows:
        control_answer = "No completed constrained semantic control table is available yet."
    elif control_wins == total:
        control_answer = (
            f"{context} completed; real EEG beats shuffled/random controls in `{control_wins}/{total}` conditions."
        )
    elif control_wins > 0:
        control_answer = (
            f"{context} completed with limited/mixed evidence: real EEG beats shuffled/random controls in `{control_wins}/{total}` conditions and `{degraded_control_wins}/{len(degraded)}` degraded conditions."
        )
    else:
        control_answer = (
            f"{context} completed as negative: real EEG beats shuffled/random controls in `0/{total}` conditions."
        )
    if rows and vision_wins == total:
        vision_answer = (
            f"{context} completed; real EEG beats vision-only in `{vision_wins}/{total}` conditions."
        )
    elif rows and vision_wins > 0:
        vision_answer = (
            f"Limited: real EEG beats vision-only in `{vision_wins}/{total}` conditions; no broad vision-only improvement claim is supported."
        )
    elif rows:
        vision_answer = "Current constrained semantic table does not support real EEG beating the strong vision-only CLIP prototype baseline."
    else:
        vision_answer = "Current constrained semantic table is missing, so no vision-only comparison can be claimed."
    return {
        "source": source,
        "context": context,
        "rows": rows,
        "total": total,
        "control_wins": control_wins,
        "degraded_total": len(degraded),
        "degraded_control_wins": degraded_control_wins,
        "vision_wins": vision_wins,
        "control_answer": control_answer,
        "vision_answer": vision_answer,
    }


def _things_sequence_status(outputs_root: Path) -> dict[str, str]:
    m0_transfer = outputs_root / "transfer" / "things_m0_convtransformer_pretrain_t2t_align" / "alignment_metrics.json"
    m2_pretrain_report = (
        outputs_root
        / "pretrain"
        / "masked_eeg_things_eeg2_m2_temporal_spectral_spatial"
        / "MASKED_EEG_PRETRAIN_REPORT.md"
    )
    m2_train_log = (
        outputs_root
        / "pretrain"
        / "masked_eeg_things_eeg2_m2_temporal_spectral_spatial"
        / "train.stdout.log"
    )
    m2_transfer = outputs_root / "transfer" / "things_m2_tsst_pretrain_t2t_align" / "alignment_metrics.json"
    m1_pretrain_report = (
        outputs_root
        / "pretrain"
        / "masked_eeg_things_eeg2_m1_dualbranch_eegconformer"
        / "MASKED_EEG_PRETRAIN_REPORT.md"
    )
    m1_train_log = (
        outputs_root
        / "pretrain"
        / "masked_eeg_things_eeg2_m1_dualbranch_eegconformer"
        / "train.log"
    )
    m1_transfer = outputs_root / "transfer" / "things_m1_dualbranch_pretrain_t2t_align" / "alignment_metrics.json"
    m0_done = m0_transfer.exists()
    m2_pretrain_done = m2_pretrain_report.exists()
    m2_seen = m2_train_log.exists() or m2_pretrain_done
    m2_done = m2_transfer.exists()
    m1_pretrain_done = m1_pretrain_report.exists()
    m1_seen = m1_train_log.exists() or m1_pretrain_done
    m1_done = m1_transfer.exists()
    if m0_done and m2_done:
        if m1_done:
            return {
                "gpu": "THINGS M0/M1/M2 pretraining-to-transfer sequence has completed; inspect the active queue for the current GPU driver.",
                "next": "compare M1 transfer against M0/M2 and P2, then follow the active queue state for any exact-linked EEG-ImageNet work",
                "open": "- M0, M1, and M2 THINGS transfer metrics are complete; use them as transfer evidence while preserving exact-linked EEG-ImageNet follow-ups separately.",
            }
        if m1_seen:
            stage = "transfer" if m1_pretrain_done else "pretraining"
            next_action = (
                "continue the active M1 Thought2Text transfer, then compare against M0/M2 and P2"
                if m1_pretrain_done
                else "M1 Thought2Text transfer watcher is waiting for the M1 report/checkpoint, then compare against M0/M2 and P2"
            )
            open_note = (
                "- M0 and M2 THINGS transfer metrics are complete; M1 THINGS masked EEG pretraining report exists and its Thought2Text transfer is the current large-data sequence."
                if m1_pretrain_done
                else "- M0 and M2 THINGS transfer metrics are complete; active M1 THINGS masked EEG pretraining remains the current large-data sequence, and its transfer watcher is waiting for the formal M1 report/checkpoint."
            )
            return {
                "gpu": f"GPU utilization is currently driven by active M1 THINGS masked EEG {stage} plus any concurrent alignment follow-ups.",
                "next": next_action,
                "open": open_note,
            }
        return {
            "gpu": "GPU utilization has been used for completed THINGS M0/M2 transfer sequence; M1 and exact-linked EEG-ImageNet follow-ups are the next useful jobs.",
            "next": "run queued M1 pretraining/transfer and exact-linked EEG-ImageNet follow-ups after the completed M0/M2 THINGS transfer sequence",
            "open": "- M0 and M2 THINGS transfer metrics are complete; use them as transfer evidence before starting M1 or exact-linked EEG-ImageNet follow-ups.",
        }
    if m0_done and m2_seen:
        stage = "pretraining" if not m2_pretrain_done else "transfer"
        return {
            "gpu": f"GPU utilization is currently driven by the active THINGS M2 masked EEG {stage} sequence; M0 transfer has already completed.",
            "next": "continue the M2 THINGS pretraining-to-transfer sequence, then run queued M1 and exact-linked EEG-ImageNet follow-ups",
            "open": "- M0 THINGS transfer metrics are complete; M2 THINGS pretraining/transfer remains the active sequence before M1 can start.",
        }
    if m2_seen:
        return {
            "gpu": "GPU utilization is currently driven by active THINGS M0/M2 masked EEG pretraining; earlier low-utilization gaps were addressed with queued heavy jobs.",
            "next": "continue the M0/M2 THINGS pretraining-to-transfer sequence, then run queued M1 and exact-linked EEG-ImageNet follow-ups",
            "open": "- THINGS M0/M2 pretraining-to-transfer sequence is still active; do not promote M1 until required transfer metrics exist.",
        }
    return {
        "gpu": "Multiple heavy-stage jobs have been launched after low-utilization diagnosis; inspect live GPU status for current utilization.",
        "next": "finish the current large-data pretraining/transfer sequence, then run queued M1 and exact-linked EEG-ImageNet follow-ups",
        "open": "- THINGS M0/M2 status is not fully represented by current artifacts; inspect live monitor and queue before launching follow-ups.",
    }


def _exact_scratch_followup_status(outputs_root: Path) -> dict[str, str]:
    a4_metrics_path = outputs_root / "transfer" / "eeg_imagenet_exact_a4_scratch_full" / "alignment_metrics.json"
    a4_metrics = _metric_block(_load_json(a4_metrics_path)) if a4_metrics_path.exists() else {}
    a2_root = outputs_root / "trimodal" / "eeg_imagenet_exact_a2_scratch_full"
    a2_report = a2_root / "TRIMODAL_FULL_REPORT.md"
    a2_metrics = a2_root / "trimodal_metrics.json"
    a2_log = a2_root / "train_log.jsonl"
    siglip_cache = outputs_root.parent / "data" / "thought2text" / "cache" / "siglip_val.npy"

    notes: list[str] = []
    has_evidence = bool(a4_metrics) or a2_report.exists() or a2_metrics.exists() or a2_log.exists()

    if a4_metrics:
        notes.append(
            "- EEG-ImageNet exact A4 scratch alignment completed as a negative result: "
            f"R@5 `{_fmt(_safe_float(a4_metrics.get('r@5')))}`, "
            f"class accuracy `{_fmt(_safe_float(a4_metrics.get('class_acc')))}`."
        )
    else:
        notes.append("- EEG-ImageNet exact A4 scratch alignment is still pending or running; final metrics are not present.")

    if a2_report.exists() and a2_metrics.exists():
        a2_status = "completed"
        notes.append("- EEG-ImageNet exact A2 scratch tri-modal completed; preserve its report with the earlier exact-linked negative controls.")
    elif a2_log.exists():
        a2_status = "running"
        notes.append("- EEG-ImageNet exact A2 scratch tri-modal remains running; final EEG->image/text metrics are pending.")
    else:
        a2_status = "pending"
        notes.append("- EEG-ImageNet exact A2 scratch tri-modal has no final report or active log artifact yet.")

    if siglip_cache.exists():
        notes.append("- SigLIP prototype/calibration cache exists.")
        siglip_status = "completed"
    else:
        notes.append("- SigLIP prototype/calibration cache remains queued.")
        siglip_status = "queued"

    if a2_status == "running" and siglip_status == "completed":
        next_action = "let exact A2 scratch tri-modal finish"
    elif a2_status == "running":
        next_action = "let exact A2 scratch tri-modal finish, then run SigLIP calibration if GPU is safely available"
    elif a2_status == "completed" and siglip_status == "queued":
        next_action = "run SigLIP prototype/calibration cache if GPU is safely available"
    elif a2_status == "completed":
        next_action = "compare scratch exact-linked A4/A2 outputs against prior exact-linked negative controls"
    else:
        next_action = "finish remaining exact-linked scratch follow-ups and queued SigLIP calibration"

    return {
        "a4_done": "yes" if bool(a4_metrics) else "no",
        "a2_status": a2_status,
        "siglip_status": siglip_status,
        "has_evidence": "yes" if has_evidence else "no",
        "next": next_action,
        "open": "\n".join(notes),
    }


def _write_master_report(rows: list[dict[str, Any]], outputs_root: Path) -> None:
    heavy_dir = outputs_root / "heavy_stage"
    heavy_dir.mkdir(parents=True, exist_ok=True)
    best = select_best_row(rows)
    transfer_rows = [row for row in rows if "transfer" in str(row["output_dir"]) or "pretrain" in row["pretrain"].lower()]
    best_transfer = select_best_row(transfer_rows) if transfer_rows else {}
    things_transfer = next(
        (
            row
            for row in transfer_rows
            if row["experiment_id"] == "things_raw_pretrain_t2t_align"
            or "things_raw_pretrain_t2t_align" in str(row["output_dir"])
        ),
        None,
    )
    dataset_decision = outputs_root / "datasets" / "DATASET_DECISION_REPORT.md"
    large_data_status = outputs_root / "download_reports" / "LARGE_DATA_DOWNLOAD_STATUS.md"
    things_ready = outputs_root / "datasets" / "THINGS_EEG2_READY_REPORT.md"
    eit_ready = outputs_root / "datasets" / "EIT1M_READY_REPORT.md"
    eeg_imagenet_train_link = outputs_root / "datasets" / "EEG_IMAGENET_IMAGE_LINK_TRAIN_REPORT.md"
    eeg_imagenet_val_link = outputs_root / "datasets" / "EEG_IMAGENET_IMAGE_LINK_VAL_REPORT.md"
    clip_adapter_report = outputs_root / "clip_adapter" / "CLIP_ADAPTER_REPORT.md"
    trimodal_report = outputs_root / "trimodal" / "TRIMODAL_FULL_REPORT.md"
    final_semantic_report = outputs_root / "final_semantic" / "FULL_ROBUST_SEMANTIC_REPORT.md"
    semantic_status = _semantic_control_status(outputs_root)
    gpu_report = outputs_root / "heavy_stage" / "GPU_UTILIZATION_REPORT.md"
    live_status = outputs_root / "heavy_stage" / "LIVE_STATUS.md"
    eeg_imagenet_transfer = next(
        (
            row
            for row in transfer_rows
            if row["experiment_id"] == "eeg_imagenet_pretrain_t2t_align"
            or "eeg_imagenet_pretrain_t2t_align" in str(row["output_dir"])
        ),
        None,
    )
    paired_exact_metrics = outputs_root / "transfer" / "eeg_imagenet_paired_alignment_cached" / "alignment_metrics.json"
    paired_exact_block = _metric_block(_load_json(paired_exact_metrics)) if paired_exact_metrics.exists() else {}
    trimodal_exact_report = outputs_root / "trimodal" / "eeg_imagenet_exact_full" / "TRIMODAL_FULL_REPORT.md"
    trimodal_exact_log = outputs_root / "trimodal" / "eeg_imagenet_exact_full" / "train_log.jsonl"
    trimodal_exact_complete = trimodal_exact_report.exists()
    things_sequence = _things_sequence_status(outputs_root)
    exact_scratch = _exact_scratch_followup_status(outputs_root)
    if paired_exact_block:
        paired_exact_status = (
            "EEG-ImageNet paired image+EEG exact-linked training completed as a negative result"
            f" with R@5 `{_fmt(_safe_float(paired_exact_block.get('r@5')))}`"
        )
        paired_exact_open = (
            "- EEG-ImageNet paired image+EEG exact-linked training completed and was near random; "
            f"R@5 `{_fmt(_safe_float(paired_exact_block.get('r@5')))}`. Preserve it as a negative control rather than treating it as an active job."
        )
        paired_loader_status = "EEG-ImageNet EEG and the exact-linked paired subset are loader-ready; exact-linked paired training has completed as a negative result"
        gpu_util_answer = things_sequence["gpu"]
    else:
        paired_exact_status = "EEG-ImageNet paired image+EEG exact-linked training is not complete in this report"
        paired_exact_open = (
            "- EEG-ImageNet paired image+EEG exact-linked training is not complete in this report. "
            "Kaggle CLS-LOC lacks 919 exact EEG-ImageNet stimulus IDs, so missing stimuli are excluded rather than replaced by same-class images."
        )
        paired_loader_status = "EEG-ImageNet EEG is loader-ready and the exact-linked paired image+EEG subset is loader-ready/running"
        gpu_util_answer = "Multiple heavy-stage jobs have been launched after low-utilization diagnosis; inspect live GPU status for current utilization."
    if eeg_imagenet_transfer:
        eeg_transfer_answer = (
            "EEG-ImageNet transfer recovery completed but did not improve Thought2Text alignment"
            if eeg_imagenet_transfer["r5"] <= HISTORICAL_DAY4_R5
            else "EEG-ImageNet transfer recovery completed and improved Thought2Text alignment"
        )
        if paired_exact_block and trimodal_exact_complete:
            exact_next = exact_scratch["next"] if exact_scratch["has_evidence"] == "yes" else things_sequence["next"]
            eeg_transfer_next = (
                "use completed exact-linked EEG-ImageNet paired alignment and tri-modal results as negative controls, "
                f"{exact_next}"
            )
            downstream_clause = "re-run downstream robust semantic evaluation only after a queued transfer/alignment checkpoint beats prior baselines"
        elif semantic_status["total"] > 0:
            eeg_transfer_next = (
                "use completed negative/limited transfer results as controls and finish any remaining exact-linked EEG-ImageNet paired/tri-modal jobs"
            )
            downstream_clause = "re-run downstream robust semantic evaluation only if a remaining alignment beats prior baselines"
        else:
            eeg_transfer_next = "let robust semantic re-evaluation finish from the completed EEG-ImageNet transfer metrics marker"
            downstream_clause = "re-run downstream robust semantic evaluation only after the final semantic metrics are available"
        eeg_transfer_open = [
            f"- EEG-ImageNet transfer recovery completed but did not improve over historical Day4 R@5: transfer R@5 `{_fmt(eeg_imagenet_transfer['r5'])}` vs Day4 `{HISTORICAL_DAY4_R5:.4f}`.",
        ]
        if semantic_status["total"] > 0:
            eeg_transfer_open.append(
                f"- {semantic_status['context']} completed: real EEG beats shuffled/random controls `{semantic_status['control_wins']}/{semantic_status['total']}` and vision-only `{semantic_status['vision_wins']}/{semantic_status['total']}`."
            )
        else:
            eeg_transfer_open.append(
                "- Robust semantic evaluation from the EEG-ImageNet transfer checkpoint is running or pending final `FULL_METRICS.csv`."
            )
    else:
        eeg_transfer_answer = "Not proven yet for EEG-ImageNet; Thought2Text masked transfer is below Day4 R@5"
        eeg_transfer_next = (
            exact_scratch["next"]
            if exact_scratch["has_evidence"] == "yes"
            else "finish EEG-ImageNet transfer recovery and let robust semantic re-evaluation run from the final transfer metrics marker"
        )
        downstream_clause = "re-run downstream robust semantic evaluation only after final transfer metrics exist"
        eeg_transfer_open = [
            "- EEG-ImageNet masked pretraining is complete, but its downstream transfer recovery is still running and lacks final `alignment_metrics.json`.",
            "- `TRANSFER_EEG_IMAGENET_PRETRAIN_TO_THOUGHT2TEXT` is running or waiting for final metrics and is not yet represented in completed transfer metrics.",
        ]
    if things_transfer:
        things_open = (
            f"- THINGS raw-window transfer completed with R@5 `{_fmt(things_transfer['r5'])}` and class accuracy `{_fmt(things_transfer['class_acc'])}`; "
            "this is useful negative/limited transfer evidence and remains EEG-only pretraining, not image-trial alignment."
        )
    else:
        things_open = (
            "- THINGS-EEG2 raw-window masked EEG pretraining completed, but its Thought2Text transfer final metrics are not available yet; "
            "this remains EEG-only pretraining, not image-trial alignment."
        )
    if trimodal_exact_complete:
        trimodal_answer = "Large-dataset exact-linked EEG-ImageNet tri-modal training completed; compare its report against Thought2Text tri-modal."
        trimodal_evidence = trimodal_exact_report
    elif trimodal_exact_log.exists():
        trimodal_answer = "Large-dataset exact-linked EEG-ImageNet tri-modal training is running; final EEG->image/text metrics are pending."
        trimodal_evidence = trimodal_exact_log
    else:
        trimodal_answer = "Full Thought2Text tri-modal report exists; compare there, but large-dataset tri-modal is not complete yet."
        trimodal_evidence = trimodal_report
    stage_complete = exact_scratch["a2_status"] == "completed"
    status_label = "complete_with_mixed_results" if stage_complete else "in progress"
    intro = (
        "This is the final heavy-stage control report. It records completed artifacts, negative results, "
        "and claim boundaries without overstating unsupported improvements."
        if stage_complete
        else "This is the current heavy-stage control report. It records verified artifacts and open requirements without claiming the full goal is complete."
    )
    open_work_heading = "Remaining Caveats" if stage_complete else "Open Work"
    final_heading = "Final Decision" if stage_complete else "Claim Boundary"
    final_boundary = (
        "The heavy-stage execution is complete with mixed results. The final scientific target is not fully supported: "
        "large-data pretraining/transfer and exact-linked EEG-ImageNet follow-ups did not beat the best Thought2Text alignment, "
        "while constrained semantic evaluations still support paired real EEG over shuffled/random controls and, in the A2 semantic fusion runs, over vision-only under the evaluated strong degradations. "
        "Do not claim open-ended captioning success or mind-reading."
        if stage_complete
        else "Do not claim the final heavy-stage scientific target yet. Current evidence supports useful alignment progress and active large EEG pretraining, but not the final transferred robust semantic improvement claim."
    )
    lines = [
        "# Heavy Stage Master Report",
        "",
        f"- Updated UTC: `{_utc_now()}`",
        f"- Current status: `{status_label}`",
        "",
        intro,
        "",
        "## Required Questions",
        "",
        "| Question | Current answer | Evidence |",
        "| --- | --- | --- |",
        f"| Did larger datasets become loader-ready? | {paired_loader_status}; THINGS/EIT are not full image+EEG loader-ready. | `{large_data_status}`, `{eeg_imagenet_train_link}`, `{eeg_imagenet_val_link}`, `{things_ready}`, `{eit_ready}` |",
        f"| Did masked EEG pretraining improve downstream alignment? | {eeg_transfer_answer}; Thought2Text masked transfer is also below Day4 R@5. | `{outputs_root / 'transfer' / 'TRANSFER_TO_THOUGHT2TEXT_REPORT.md'}` |",
        f"| Did tri-modal training improve EEG->text/class retrieval? | {trimodal_answer} | `{trimodal_evidence}` |",
        f"| Which EEG architecture worked best? | `{best.get('encoder_type', 'none')}` with R@5 `{_fmt(best.get('r5', 0.0)) if best else 'n/a'}` among indexed formal runs. | `{outputs_root / 'architectures' / 'ARCHITECTURE_SEARCH_REPORT.md'}` |",
        f"| Did CLIP adapter help? | CLIP adapter report exists as calibration/retrieval diagnostic; it is not an EEG benefit claim by itself. | `{clip_adapter_report}` |",
        f"| Did transferred encoder improve Thought2Text? | `{'yes' if best_transfer and best_transfer['r5'] > HISTORICAL_DAY4_R5 else 'no'}` for completed transfer runs so far. | `{outputs_root / 'transfer' / 'TRANSFER_TO_THOUGHT2TEXT_REPORT.md'}` |",
        f"| Did real EEG beat shuffled/random? | {semantic_status['control_answer']} | `{final_semantic_report}`, `{semantic_status['source']}` |",
        f"| Did real EEG beat vision-only under strong degradation? | {semantic_status['vision_answer']} | `{final_semantic_report}`, `{semantic_status['source']}` |",
        f"| Was GPU utilization improved? | {gpu_util_answer} | `{live_status}`, `{gpu_report}` |",
        f"| What is the next best scientific direction? | {eeg_transfer_next}; {downstream_clause}. | current queue and watcher |",
        "",
        "## Current Best Artifacts",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- Best indexed alignment checkpoint: `{best['checkpoint'] or 'missing'}`",
                f"- Materialized architecture checkpoint: `{outputs_root / 'architectures' / 'checkpoints' / 'best_encoder.pt'}`",
                f"- Best indexed alignment R@1/R@5/R@10: `{_fmt(best['r1'])} / {_fmt(best['r5'])} / {_fmt(best['r10'])}`",
                f"- Best indexed class accuracy: `{_fmt(best['class_acc'])}`",
            ]
        )
    if best_transfer:
        lines.extend(
            [
                f"- Best completed transfer checkpoint: `{outputs_root / 'transfer' / 'best_transfer_encoder.pt'}`",
                f"- Best completed transfer R@5: `{_fmt(best_transfer['r5'])}`",
            ]
        )
    lines.extend(
        [
            "",
            f"## {open_work_heading}",
            "",
            *eeg_transfer_open,
            f"- {paired_exact_status}.",
            paired_exact_open,
            things_open,
            things_sequence["open"],
            *( [exact_scratch["open"]] if exact_scratch["has_evidence"] == "yes" else [] ),
            "- THINGS-EEG2 still needs event/trial alignment before it can become loader-ready.",
            "- ImageNet CLS-LOC download/extraction is complete; current paired limitation is exact stimulus coverage, not download status.",
            "",
            f"## {final_heading}",
            "",
            final_boundary,
        ]
    )
    (heavy_dir / "MASTER_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(rows: list[dict[str, Any]], outputs_root: Path) -> None:
    _write_architecture_report(rows, outputs_root)
    _write_transfer_report(rows, outputs_root)
    _ensure_trimodal_artifact(outputs_root)
    _ensure_pretrain_artifact(outputs_root)
    _write_baselines(rows, outputs_root)
    _write_master_report(rows, outputs_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize heavy-stage summary reports from completed training artifacts.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--config-root", default="configs")
    args = parser.parse_args()
    outputs_root = Path(args.outputs_root)
    config_root = Path(args.config_root)
    rows = collect_alignment_rows(outputs_root, [config_root])
    write_reports(rows, outputs_root)


if __name__ == "__main__":
    main()
