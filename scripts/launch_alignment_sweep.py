from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gpu_queue import allowed_concurrency, count_running_alignment_processes, query_gpu_status
from src.models.alignment_model import EEGCLIPAlignmentModel
from src.utils.config import load_config


EXPERIMENT_BOARD_COLUMNS = [
    "experiment_id",
    "encoder_type",
    "param_count",
    "loss_combo",
    "seed",
    "status",
    "best_epoch",
    "val_R@1",
    "val_R@5",
    "val_R@10",
    "test_R@1",
    "test_R@5",
    "test_R@10",
    "class_acc",
    "mean_rank",
    "gpu_mem_peak",
    "time_minutes",
    "notes",
]


@dataclass
class RunningJob:
    experiment_id: str
    config_path: Path
    out_dir: Path
    process: subprocess.Popen
    log_handle: Any
    start_time: float
    gpu_mem_peak: int = 0
    retry_count: int = 0
    notes: str = ""


@dataclass
class BoardRow:
    experiment_id: str
    encoder_type: str
    param_count: int
    loss_combo: str
    seed: int
    status: str
    best_epoch: str = "NA"
    val_R1: str = "NA"
    val_R5: str = "NA"
    val_R10: str = "NA"
    test_R1: str = "NA"
    test_R5: str = "NA"
    test_R10: str = "NA"
    class_acc: str = "NA"
    mean_rank: str = "NA"
    gpu_mem_peak: str = "NA"
    time_minutes: str = "NA"
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "encoder_type": self.encoder_type,
            "param_count": self.param_count,
            "loss_combo": self.loss_combo,
            "seed": self.seed,
            "status": self.status,
            "best_epoch": self.best_epoch,
            "val_R@1": self.val_R1,
            "val_R@5": self.val_R5,
            "val_R@10": self.val_R10,
            "test_R@1": self.test_R1,
            "test_R@5": self.test_R5,
            "test_R@10": self.test_R10,
            "class_acc": self.class_acc,
            "mean_rank": self.mean_rank,
            "gpu_mem_peak": self.gpu_mem_peak,
            "time_minutes": self.time_minutes,
            "notes": self.notes,
        }


def _read_json(path: str | Path) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml
    except ModuleNotFoundError:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _num_classes(manifest: str | Path) -> int | None:
    labels: list[int] = []
    path = Path(manifest)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("label") is not None:
                    labels.append(int(row["label"]))
    return max(labels) + 1 if labels else None


def estimate_param_count(config: dict[str, Any]) -> int:
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    num_classes = _num_classes(config.get("data", {}).get("train_manifest", "")) if bool(loss_cfg.get("use_cls", True)) else None
    model = EEGCLIPAlignmentModel(
        eeg_channels=int(model_cfg.get("eeg_channels", 64)),
        eeg_timesteps=int(model_cfg.get("eeg_time_steps", 250)),
        eeg_dim=int(model_cfg.get("eeg_embed_dim", 512)),
        clip_dim=int(model_cfg.get("clip_embed_dim", 512)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        transformer_layers=int(model_cfg.get("transformer_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_classes=num_classes,
        encoder_type=str(model_cfg.get("encoder_type", "tiny")),
    )
    with torch.no_grad():
        dummy = torch.zeros(
            1,
            int(model_cfg.get("eeg_channels", 64)),
            int(model_cfg.get("eeg_time_steps", 250)),
        )
        model(dummy)
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _best_epoch(out_dir: Path) -> str:
    history = _read_json(out_dir / "history.json")
    if not isinstance(history, list) or not history:
        return "NA"
    best = max(history, key=lambda row: float(row.get("metrics", {}).get("r@5", -1.0)))
    return str(best.get("epoch", "NA"))


def load_completed_row(config_path: str | Path, out_dir: str | Path, status: str | None = None) -> BoardRow:
    config = load_config(config_path)
    out_dir = Path(out_dir)
    val_payload = _read_json(out_dir / "alignment_metrics.json") or {}
    test_payload = _read_json(out_dir / "test_metrics.json") or {}
    metrics_payload = _read_json(out_dir / "metrics.json") or {}
    if metrics_payload:
        val_payload = {"model": metrics_payload.get("val", {}).get("model", metrics_payload.get("val", {}))}
        test_payload = {"model": metrics_payload.get("test", {}).get("model", metrics_payload.get("test", {}))}
    val_model = val_payload.get("model", {}) if isinstance(val_payload, dict) else {}
    test_model = test_payload.get("model", {}) if isinstance(test_payload, dict) else {}
    try:
        param_count = estimate_param_count(config)
    except Exception:
        param_count = 0
    return BoardRow(
        experiment_id=str(config.get("experiment_id", out_dir.name)),
        encoder_type=str(config.get("model", {}).get("encoder_type", "tiny")),
        param_count=param_count,
        loss_combo=str(config.get("loss", {}).get("loss_combo", "")),
        seed=int(config.get("seed", 42)),
        status=status or ("completed" if (out_dir / "metrics.json").exists() else "pending"),
        best_epoch=_best_epoch(out_dir),
        val_R1=_fmt_metric(val_model.get("r@1")),
        val_R5=_fmt_metric(val_model.get("r@5")),
        val_R10=_fmt_metric(val_model.get("r@10")),
        test_R1=_fmt_metric(test_model.get("r@1")),
        test_R5=_fmt_metric(test_model.get("r@5")),
        test_R10=_fmt_metric(test_model.get("r@10")),
        class_acc=_fmt_metric(test_model.get("class_acc", val_model.get("class_acc"))),
        mean_rank=_fmt_metric(test_model.get("mean_rank", val_model.get("mean_rank"))),
        notes=str(config.get("notes", "")),
    )


def build_initial_board_rows(config_paths: list[Path], out_root: str | Path) -> dict[str, BoardRow]:
    out_root = Path(out_root)
    rows: dict[str, BoardRow] = {}
    for config_path in config_paths:
        config = load_config(config_path)
        experiment_id = str(config.get("experiment_id", config_path.stem))
        out_dir = out_root / experiment_id
        status = "completed" if (out_dir / "metrics.json").exists() else "queued"
        rows[experiment_id] = load_completed_row(config_path, out_dir, status=status)
    return rows


def _write_table(path: Path, rows: list[BoardRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_BOARD_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    md_path = path.with_suffix(".md")
    lines = [
        "# Experiment Board",
        "",
        "| " + " | ".join(EXPERIMENT_BOARD_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in EXPERIMENT_BOARD_COLUMNS) + " |",
    ]
    for row in rows:
        data = row.as_dict()
        lines.append("| " + " | ".join(str(data[column]) for column in EXPERIMENT_BOARD_COLUMNS) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rows_from_board(path: str | Path) -> list[BoardRow]:
    rows: list[BoardRow] = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                BoardRow(
                    experiment_id=raw["experiment_id"],
                    encoder_type=raw["encoder_type"],
                    param_count=int(float(raw.get("param_count", 0) or 0)),
                    loss_combo=raw["loss_combo"],
                    seed=int(float(raw.get("seed", 42) or 42)),
                    status=raw["status"],
                    best_epoch=raw.get("best_epoch", "NA"),
                    val_R1=raw.get("val_R@1", "NA"),
                    val_R5=raw.get("val_R@5", "NA"),
                    val_R10=raw.get("val_R@10", "NA"),
                    test_R1=raw.get("test_R@1", "NA"),
                    test_R5=raw.get("test_R@5", "NA"),
                    test_R10=raw.get("test_R@10", "NA"),
                    class_acc=raw.get("class_acc", "NA"),
                    mean_rank=raw.get("mean_rank", "NA"),
                    gpu_mem_peak=raw.get("gpu_mem_peak", "NA"),
                    time_minutes=raw.get("time_minutes", "NA"),
                    notes=raw.get("notes", ""),
                )
            )
    return rows


def _score(row: BoardRow) -> float:
    def f(value: str) -> float:
        try:
            return float(value)
        except ValueError:
            return 0.0

    return f(row.val_R5) + 0.5 * f(row.class_acc)


def _metric_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _comparison_note(
    label: str,
    baseline: BoardRow | None,
    candidate: BoardRow | None,
    feature_name: str,
) -> str:
    if baseline is None or candidate is None:
        return f"{label}: not evaluated; missing controlled rows."
    base_val = _metric_float(baseline.val_R5)
    cand_val = _metric_float(candidate.val_R5)
    base_test = _metric_float(baseline.test_R5)
    cand_test = _metric_float(candidate.test_R5)
    if base_val is None or cand_val is None:
        return f"{label}: not evaluated; missing validation R@5."
    delta_val = cand_val - base_val
    delta_test = (cand_test - base_test) if base_test is not None and cand_test is not None else None
    if delta_val > 1e-9:
        if delta_test is not None and delta_test < -1e-9:
            return (
                f"{label}: mixed; {feature_name} improved validation R@5 "
                f"({baseline.experiment_id} {base_val:.6f} -> {candidate.experiment_id} {cand_val:.6f}) "
                f"but did not improve test R@5 ({base_test:.6f} -> {cand_test:.6f})."
            )
        return (
            f"{label}: helped validation R@5 "
            f"({baseline.experiment_id} {base_val:.6f} -> {candidate.experiment_id} {cand_val:.6f})."
        )
    if delta_val < -1e-9:
        return (
            f"{label}: did not help; {feature_name} reduced validation R@5 "
            f"({baseline.experiment_id} {base_val:.6f} -> {candidate.experiment_id} {cand_val:.6f})."
        )
    if delta_test is not None and delta_test < -1e-9:
        return (
            f"{label}: did not clearly help; validation R@5 tied "
            f"({base_val:.6f}) but test R@5 dropped ({base_test:.6f} -> {cand_test:.6f})."
        )
    return f"{label}: tied on validation R@5 ({base_val:.6f}); no clear improvement."


def controlled_comparison_notes(rows: list[BoardRow]) -> dict[str, str]:
    by_id = {row.experiment_id: row for row in rows if row.status == "completed"}
    same_image = _comparison_note("Subject/same-image consistency", by_id.get("X1"), by_id.get("X2"), "adding L7")
    if same_image.endswith("missing controlled rows."):
        same_image = (
            "Subject/same-image consistency: partially evaluated; `X2_stage2` completed in this sweep, "
            "and paired E5 no-L7 vs E5+L7 recheck is summarized in "
            "`outputs/subject_adaptation/SAME_IMAGE_CONSISTENCY_REPORT.md`."
        )

    similarity = _comparison_note("Similarity distillation", by_id.get("S1"), by_id.get("S3"), "adding L5")
    if similarity.endswith("missing controlled rows."):
        similarity = _comparison_note(
            "Similarity distillation",
            by_id.get("S1_stage2") or by_id.get("S1_stage2_seed42"),
            by_id.get("S3_stage2"),
            "adding L5",
        )

    augmentation = _comparison_note("Augmentation consistency", by_id.get("S3"), by_id.get("S4"), "adding L6")
    if augmentation.endswith("missing controlled rows."):
        augmentation = (
            "Augmentation consistency: not directly controlled in this continuation sweep; "
            "`G2_*` includes L6 but no matched E4 no-L6 pair was run here."
        )
    return {
        "same_image_subject": same_image,
        "similarity": similarity,
        "augmentation": augmentation,
    }


def _write_ranking(out_root: Path, rows: list[BoardRow]) -> None:
    completed = [row for row in rows if row.status == "completed"]
    ranked = sorted(completed, key=_score, reverse=True)
    csv_path = out_root / "RANKING.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "score", *EXPERIMENT_BOARD_COLUMNS])
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            writer.writerow({"rank": rank, "score": f"{_score(row):.6f}", **row.as_dict()})
    lines = ["# Alignment Search Ranking", "", "| Rank | Score | ID | Encoder | Loss | Val R@5 | Class Acc | Test R@5 |", "| ---: | ---: | --- | --- | --- | ---: | ---: | ---: |"]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {_score(row):.6f} | {row.experiment_id} | {row.encoder_type} | {row.loss_combo} | "
            f"{row.val_R5} | {row.class_acc} | {row.test_R5} |"
        )
    (out_root / "RANKING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_search_summary(out_root: Path, rows: list[BoardRow]) -> None:
    completed = [row for row in rows if row.status == "completed"]
    best = max(completed, key=_score) if completed else None
    tiny_best = max((row for row in completed if row.encoder_type == "tiny"), key=_score, default=None)
    base_strong_best = max((row for row in completed if row.encoder_type != "tiny"), key=_score, default=None)
    comparison_notes = controlled_comparison_notes(rows)
    lines = [
        "# Day4 Alignment Search Summary",
        "",
        f"1. Experiments generated: `{len(rows)}`",
        f"2. Experiments completed: `{len(completed)}`",
        f"3. Best encoder: `{best.encoder_type if best else 'NA'}`",
        f"4. Best loss combination: `{best.loss_combo if best else 'NA'}`",
        f"5. Best checkpoint path: `{out_root / (best.experiment_id if best else 'NA') / 'checkpoints' / 'best.pt'}`",
        f"6. Best R@1/R@5/R@10: `{best.test_R1 if best else 'NA'} / {best.test_R5 if best else 'NA'} / {best.test_R10 if best else 'NA'}`",
        f"7. Best class accuracy: `{best.class_acc if best else 'NA'}`",
        f"8. Base/Strong beat Tiny: `{bool(base_strong_best and (not tiny_best or _score(base_strong_best) > _score(tiny_best)))}`",
        f"9. Subject/same-image consistency helped: `{comparison_notes['same_image_subject']}`",
        f"10. Similarity distillation helped: `{comparison_notes['similarity']}`",
        f"11. Augmentation consistency helped: `{comparison_notes['augmentation']}`",
        f"12. Recommended checkpoint for fusion: `{out_root / (best.experiment_id if best else 'NA') / 'checkpoints' / 'best.pt'}`",
        "",
        "Current Day3 reference test R@5 is `0.0315`; use the ranking table to verify whether the best completed run beats it.",
    ]
    (out_root / "SEARCH_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_search_artifacts(out_root: str | Path) -> None:
    out_root = Path(out_root)
    rows = _rows_from_board(out_root / "EXPERIMENT_BOARD.csv")
    _write_ranking(out_root, rows)
    _write_search_summary(out_root, rows)
    completed = [row for row in rows if row.status == "completed"]
    top = sorted(completed, key=_score, reverse=True)[:4]
    (out_root / "TOP_CANDIDATES.md").write_text(
        "# Top Candidates\n\n"
        + "\n".join(
            f"- `{row.experiment_id}` score `{_score(row):.6f}` checkpoint `{out_root / row.experiment_id / 'checkpoints' / 'best.pt'}`"
            for row in top
        )
        + ("\n" if top else "- No completed candidates yet.\n"),
        encoding="utf-8",
    )
    (out_root / "MULTISEED_FINAL.md").write_text(
        "# Multi-seed Final\n\nPending Stage 2 multi-seed runs after screening review.\n",
        encoding="utf-8",
    )
    if top:
        source = out_root / top[0].experiment_id / "checkpoints" / "best.pt"
        if source.exists():
            shutil.copy2(source, out_root / "best_overall.pt")


def _write_live_status(out_root: Path, rows: list[BoardRow], running: list[RunningJob]) -> None:
    gpu = query_gpu_status()
    lines = ["# Live Alignment Sweep Status", ""]
    if gpu:
        lines.append(f"- GPU: `{gpu.name}` mem `{gpu.memory_used_mb}/{gpu.memory_total_mb} MiB` util `{gpu.utilization_gpu}%`")
    lines.append(f"- Running jobs: `{len(running)}`")
    for job in running:
        lines.append(f"  - `{job.experiment_id}` pid `{job.process.pid}` elapsed `{(time.time() - job.start_time) / 60:.1f} min`")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    lines.append(f"- Status counts: `{counts}`")
    (out_root / "LIVE_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _merge_metrics(out_dir: Path) -> None:
    val_payload = _read_json(out_dir / "alignment_metrics.json") or {}
    test_payload = _read_json(out_dir / "test_metrics.json") or {}
    metrics = {
        "val": val_payload,
        "test": test_payload,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_job_report(out_dir: Path, row: BoardRow) -> None:
    lines = [
        f"# Alignment Experiment {row.experiment_id}",
        "",
        f"- Encoder: `{row.encoder_type}`",
        f"- Loss combo: `{row.loss_combo}`",
        f"- Status: `{row.status}`",
        f"- Best epoch: `{row.best_epoch}`",
        f"- Val R@1/R@5/R@10: `{row.val_R1} / {row.val_R5} / {row.val_R10}`",
        f"- Test R@1/R@5/R@10: `{row.test_R1} / {row.test_R5} / {row.test_R10}`",
        f"- Class accuracy: `{row.class_acc}`",
        f"- Mean rank: `{row.mean_rank}`",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finish_success(config_path: Path, out_dir: Path, gpu_mem_peak: int, start_time: float) -> BoardRow:
    config = load_config(config_path)
    data = config["data"]
    ckpt = out_dir / "checkpoints" / "best.pt"
    test_out = out_dir / "test_metrics.json"
    if ckpt.exists() and not test_out.exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.eval.retrieval",
                "--manifest",
                str(data["test_manifest"]),
                "--clip_cache",
                str(data["clip_test_cache"]),
                "--clip_index",
                str(data.get("clip_index_test", "")),
                "--eeg_ckpt",
                str(ckpt),
                "--out",
                str(test_out),
            ],
            check=True,
        )
    _merge_metrics(out_dir)
    row = load_completed_row(config_path, out_dir, status="completed")
    row.gpu_mem_peak = str(gpu_mem_peak)
    row.time_minutes = f"{(time.time() - start_time) / 60.0:.2f}"
    _write_job_report(out_dir, row)
    return row


def _start_job(config_path: Path, out_root: Path, screen_epochs: int | None) -> RunningJob:
    config = load_config(config_path)
    out_dir = out_root / str(config.get("experiment_id", config_path.stem))
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(out_dir / "config.yaml", config)
    train_cfg = config.get("train", {})
    command = [
        sys.executable,
        "-m",
        "src.train.train_align",
        "--config",
        str(config_path),
        "--output_dir",
        str(out_dir),
        "--max_train_samples",
        str(int(train_cfg.get("max_train_samples", 0))),
        "--max_val_samples",
        str(int(train_cfg.get("max_val_samples", 0))),
    ]
    if screen_epochs is not None:
        command.extend(["--epochs", str(screen_epochs)])
    log_handle = (out_dir / "train.log").open("a", encoding="utf-8")
    log_handle.write("$ " + " ".join(command) + "\n")
    log_handle.flush()
    process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    return RunningJob(
        experiment_id=str(config.get("experiment_id", config_path.stem)),
        config_path=config_path,
        out_dir=out_dir,
        process=process,
        log_handle=log_handle,
        start_time=time.time(),
    )


def ordered_config_paths(config_dir: str | Path) -> list[Path]:
    config_dir = Path(config_dir)
    manifest_path = config_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ordered: list[Path] = []
        for row in manifest:
            path = config_dir / str(row.get("config", f"{row['experiment_id']}.yaml"))
            if path.exists():
                ordered.append(path)
        remaining = [path for path in sorted(config_dir.glob("*.yaml")) if path not in set(ordered)]
        return ordered + remaining
    priority = {"T": 0, "S": 1, "X": 2, "G": 3, "H": 4}
    return sorted(config_dir.glob("*.yaml"), key=lambda path: (priority.get(path.stem[:1], 99), path.stem))


def available_internal_launch_slots(allowed_total: int, internal_running: int, external_running: int) -> int:
    allowed_total = max(1, min(int(allowed_total), 8))
    occupied = max(0, int(internal_running)) + max(0, int(external_running))
    return max(0, allowed_total - occupied)


def run_sweep(config_dir: str | Path, out_root: str | Path, max_concurrent: int = 8, screen_epochs: int | None = 20, poll_seconds: int = 20) -> None:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    pending = ordered_config_paths(config_dir)
    running: list[RunningJob] = []
    retries: dict[str, int] = {}
    rows_by_id: dict[str, BoardRow] = build_initial_board_rows(pending, out_root)
    _write_table(out_root / "EXPERIMENT_BOARD.csv", [rows_by_id[key] for key in sorted(rows_by_id)])
    _write_ranking(out_root, [rows_by_id[key] for key in sorted(rows_by_id)])
    _write_search_summary(out_root, [rows_by_id[key] for key in sorted(rows_by_id)])

    filtered_pending: list[Path] = []
    for config_path in pending:
        config = load_config(config_path)
        out_dir = out_root / str(config.get("experiment_id", config_path.stem))
        if (out_dir / "metrics.json").exists():
            rows_by_id[str(config.get("experiment_id", config_path.stem))] = load_completed_row(config_path, out_dir, status="completed")
        else:
            filtered_pending.append(config_path)
    pending = filtered_pending

    while pending or running:
        gpu = query_gpu_status()
        if gpu is None:
            allowed = 1
        else:
            allowed = allowed_concurrency(gpu.memory_used_gb, gpu.utilization_gpu, max_concurrent=max_concurrent)
        total_alignment = count_running_alignment_processes()
        external_running = max(0, total_alignment - len(running))
        launch_slots = available_internal_launch_slots(allowed, len(running), external_running)
        while pending and launch_slots > 0:
            job = _start_job(pending.pop(0), out_root, screen_epochs)
            rows_by_id[job.experiment_id] = load_completed_row(job.config_path, job.out_dir, status="running")
            running.append(job)
            launch_slots -= 1

        time.sleep(max(1, poll_seconds))
        next_running: list[RunningJob] = []
        for job in running:
            gpu_now = query_gpu_status()
            if gpu_now is not None:
                job.gpu_mem_peak = max(job.gpu_mem_peak, gpu_now.memory_used_mb)
            return_code = job.process.poll()
            if return_code is None:
                rows_by_id[job.experiment_id] = load_completed_row(job.config_path, job.out_dir, status="running")
                next_running.append(job)
                continue
            job.log_handle.close()
            if return_code == 0:
                try:
                    rows_by_id[job.experiment_id] = _finish_success(job.config_path, job.out_dir, job.gpu_mem_peak, job.start_time)
                except Exception as exc:
                    rows_by_id[job.experiment_id] = load_completed_row(job.config_path, job.out_dir, status="failed")
                    rows_by_id[job.experiment_id].notes = f"postprocess failed: {exc}"
            else:
                retry_count = retries.get(job.experiment_id, 0)
                if retry_count < 1:
                    retries[job.experiment_id] = retry_count + 1
                    pending.append(job.config_path)
                    rows_by_id[job.experiment_id] = load_completed_row(job.config_path, job.out_dir, status="retry_queued")
                else:
                    rows_by_id[job.experiment_id] = load_completed_row(job.config_path, job.out_dir, status="failed")

        running = next_running
        rows = [rows_by_id[key] for key in sorted(rows_by_id)]
        _write_table(out_root / "EXPERIMENT_BOARD.csv", rows)
        _write_ranking(out_root, rows)
        _write_search_summary(out_root, rows)
        _write_live_status(out_root, rows, running)

    rows = [rows_by_id[key] for key in sorted(rows_by_id)]
    _write_table(out_root / "EXPERIMENT_BOARD.csv", rows)
    _write_ranking(out_root, rows)
    _write_search_summary(out_root, rows)
    (out_root / "successive_halving_report.md").write_text(
        "# Successive Halving Report\n\nStage 1 screening completed for all launched candidates. Continue top 50% to 50 epochs after reviewing `RANKING.md`.\n",
        encoding="utf-8",
    )
    refresh_search_artifacts(out_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Day4 alignment sweep with GPU-aware queueing.")
    parser.add_argument("--config_dir", default="configs/generated_alignment")
    parser.add_argument("--out", default="outputs/day4_search")
    parser.add_argument("--max_concurrent", type=int, default=8)
    parser.add_argument("--screen_epochs", type=int, default=20)
    parser.add_argument("--poll_seconds", type=int, default=20)
    parser.add_argument("--refresh_only", action="store_true")
    args = parser.parse_args()
    if args.refresh_only:
        refresh_search_artifacts(args.out)
        return
    run_sweep(
        config_dir=args.config_dir,
        out_root=args.out,
        max_concurrent=args.max_concurrent,
        screen_epochs=args.screen_epochs,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    main()
