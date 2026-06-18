from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_DIR = ROOT / "configs" / "generated_alignment"
TARGET_CONFIG_DIR = ROOT / "configs" / "parallel_alignment_search"
OUTPUT_ROOT = Path("outputs/alignment_search")
PLAN_PATH = ROOT / OUTPUT_ROOT / "EXPERIMENT_PLAN.md"
SUMMARY_PATH = ROOT / OUTPUT_ROOT / "SEARCH_SUMMARY.md"
E7_IMPLEMENTED = False


RUNNABLE_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "source_id": "T4",
        "experiment_id": "T4_stage2",
        "seed": 42,
        "reason": "Day4 rank 1; strongest simple E3 + L1 baseline for 50-epoch continuation.",
    },
    {
        "source_id": "S1",
        "experiment_id": "S1_stage2",
        "seed": 42,
        "reason": "Day4 rank 2; best structured-loss E3 candidate and best Day4 test R@5 among top validation runs.",
    },
    {
        "source_id": "S3",
        "experiment_id": "S3_stage2",
        "seed": 42,
        "reason": "Day4 rank 3; similarity-distillation ablation with tied validation R@5 versus S1.",
    },
    {
        "source_id": "X2",
        "experiment_id": "X2_stage2",
        "seed": 42,
        "reason": "Day4 rank 4; best subject-adaptive/same-image consistency candidate.",
    },
    {
        "source_id": "G2",
        "experiment_id": "G2_recheck",
        "seed": 42,
        "reason": "Strong encoder anomaly check; low Day4 validation R@5 but high test R@5/R@10.",
    },
    {
        "source_id": "T4",
        "experiment_id": "T4_stage2_seed123",
        "seed": 123,
        "reason": "Multi-seed robustness check for the Day4 winner.",
    },
    {
        "source_id": "S1",
        "experiment_id": "S1_stage2_seed123",
        "seed": 123,
        "reason": "Multi-seed robustness check for the strongest structured-loss E3 candidate.",
    },
    {
        "source_id": "T4",
        "experiment_id": "T4_stage2_seed2025",
        "seed": 2025,
        "reason": "Second alternate seed for the Day4 winner.",
    },
    {
        "source_id": "S1",
        "experiment_id": "S1_stage2_seed2025",
        "seed": 2025,
        "reason": "Second alternate seed for the strongest structured-loss E3 candidate.",
    },
    {
        "source_id": "T4",
        "experiment_id": "T4_stage2_seed42",
        "seed": 42,
        "reason": "Seed-explicit alias for T4_stage2 kept for launch-board compatibility with parallel agents.",
    },
    {
        "source_id": "S1",
        "experiment_id": "S1_stage2_seed42",
        "seed": 42,
        "reason": "Seed-explicit alias for S1_stage2 kept for launch-board compatibility with parallel agents.",
    },
    {
        "source_id": "G2",
        "experiment_id": "G2_dropout035_recheck",
        "seed": 42,
        "model_updates": {"dropout": 0.35},
        "reason": "Day5 extra scan winner used G2 with dropout 0.35; rerun full data/50 epochs at seed 42.",
    },
    {
        "source_id": "S1",
        "experiment_id": "S1_dropout03_recheck",
        "seed": 42,
        "model_updates": {"dropout": 0.30},
        "reason": "Day5 extra scan second-best used S1 with dropout 0.30; rerun full data/50 epochs at seed 42.",
    },
]


BLOCKED_PLACEHOLDER_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "source_id": "S3",
        "experiment_id": "P1_spectrogram_smoke",
        "seed": 42,
        "requested_encoder_type": "spectrogram_cnn",
        "model_updates": {"encoder_type": "convtransformer_base", "requested_encoder_type": "spectrogram_cnn"},
        "loss_updates": {
            "use_multi_positive_infonce": False,
            "use_prototype_alignment": True,
            "use_similarity_distillation": True,
            "use_similarity_distill": True,
            "loss_combo": "L1+L4+L5",
        },
        "train_updates": {"batch_size": 1, "epochs": 1, "patience": 1, "max_train_samples": 1, "max_val_samples": 1},
        "blocked_reason": "E7 spectrogram_cnn is not implemented; this is a launch-compatible blocked placeholder using E3 fallback only.",
        "reason": "P1 requested E7 + L1 + L4 + L5, but E7 is blocked in the current encoder factory.",
    },
    {
        "source_id": "S3",
        "experiment_id": "P2_raw_spectrogram_smoke",
        "seed": 42,
        "requested_encoder_type": "raw_e3_plus_spectrogram_e7",
        "model_updates": {"encoder_type": "convtransformer_base", "requested_encoder_type": "raw_e3_plus_spectrogram_e7"},
        "loss_updates": {
            "use_multi_positive_infonce": False,
            "use_prototype_alignment": True,
            "use_similarity_distillation": True,
            "use_similarity_distill": True,
            "loss_combo": "L1+L4+L5",
        },
        "train_updates": {"batch_size": 1, "epochs": 1, "patience": 1, "max_train_samples": 1, "max_val_samples": 1},
        "blocked_reason": "Raw+spectrogram late fusion is not implemented; this is a launch-compatible blocked placeholder using E3 fallback only.",
        "reason": "P2 requested raw E3 + spectrogram E7 late fusion, but alignment_model has no dual-branch fusion path yet.",
    },
]


BLOCKED_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "experiment_id": "P1",
        "encoder": "E7 spectrogram_cnn",
        "loss_combo": "L1+L4+L5",
        "blocked_by": "build_eeg_encoder does not implement E7/spectrogram_cnn.",
    },
    {
        "experiment_id": "P2",
        "encoder": "raw E3 + E7 late fusion",
        "loss_combo": "L1+L4+L5",
        "blocked_by": "alignment_model accepts one raw EEG encoder; no raw+spectrogram fusion path exists.",
    },
]


DAY4_METRICS = {
    "T4": {"rank": 1, "val_r5": 0.291291, "test_r5": 0.294294, "class_acc": 0.248326},
    "S1": {"rank": 2, "val_r5": 0.288288, "test_r5": 0.306306, "class_acc": 0.251744},
    "S3": {"rank": 3, "val_r5": 0.288288, "test_r5": 0.294294, "class_acc": 0.232878},
    "X2": {"rank": 4, "val_r5": 0.282282, "test_r5": 0.288288, "class_acc": 0.220507},
    "G2": {"rank": 12, "val_r5": 0.032032, "test_r5": 0.324324, "class_acc": 0.281206},
}


DAY5_NOTES = {
    "G2_dropout035_recheck": "Day5 extra alignment rank 1: E4_G2_dropout035_seed777 val R@5 0.261261, test R@5 0.183183.",
    "S1_dropout03_recheck": "Day5 extra alignment rank 2: E3_S1_dropout03_seed777 val R@5 0.204204, test R@5 0.171171.",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing source config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def build_config(spec: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(_load_yaml(SOURCE_CONFIG_DIR / f"{spec['source_id']}.yaml"))
    experiment_id = str(spec["experiment_id"])
    config["experiment_id"] = experiment_id
    config["seed"] = int(spec["seed"])
    config.setdefault("train", {})
    config["train"]["epochs"] = 50
    config["train"]["patience"] = 8
    config["train"]["max_train_samples"] = 0
    config["train"]["max_val_samples"] = 0
    if spec.get("train_updates"):
        _deep_update(config["train"], copy.deepcopy(spec["train_updates"]))
    if spec.get("model_updates"):
        _deep_update(config.setdefault("model", {}), copy.deepcopy(spec["model_updates"]))
    if spec.get("loss_updates"):
        _deep_update(config.setdefault("loss", {}), copy.deepcopy(spec["loss_updates"]))
    config.setdefault("output", {})
    config["output"]["dir"] = str(OUTPUT_ROOT / experiment_id)
    extra = f" {DAY5_NOTES[experiment_id]}" if experiment_id in DAY5_NOTES else ""
    config["notes"] = f"Parallel alignment search from {spec['source_id']}: {spec['reason']}{extra}"
    if spec.get("blocked_reason"):
        config["blocked"] = True
        config["blocked_reason"] = str(spec["blocked_reason"])
        config["notes"] = f"BLOCKED PLACEHOLDER: {spec['blocked_reason']} {config['notes']}"
    return config


def _config_table_rows(configs: list[dict[str, Any]]) -> list[str]:
    rows = []
    for config in configs:
        metrics = DAY4_METRICS.get(str(config["experiment_id"]).split("_")[0], {})
        rows.append(
            "| {id} | {seed} | {encoder} | {loss} | {epochs} | {patience} | {out} | {rank} | {val} |".format(
                id=config["experiment_id"],
                seed=config["seed"],
                encoder=config["model"].get("encoder_type", ""),
                loss=config["loss"].get("loss_combo", ""),
                epochs=config["train"].get("epochs"),
                patience=config["train"].get("patience"),
                out=config["output"]["dir"],
                rank=metrics.get("rank", "Day5"),
                val=metrics.get("val_r5", "see Day5 note"),
            )
        )
    return rows


def write_plan(configs: list[dict[str, Any]]) -> None:
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Parallel Alignment Search Plan",
        "",
        "## Scope",
        "",
        "This queue continues the Day4/Day5 encoder-loss search without launching GPU training. Runnable configs are compatible with `scripts/launch_alignment_sweep.py`: each YAML keeps the existing top-level `experiment_id`, `seed`, `device`, `data`, `model`, `loss`, `train`, `output`, and `notes` fields.",
        "",
        "All runnable configs use `train.epochs=50`, `train.patience=8`, full train/val data (`max_*_samples=0`), and outputs under `outputs/alignment_search/<experiment_id>`.",
        "",
        "## Runnable Queue",
        "",
        "| ID | Seed | Encoder | Loss | Epochs | Patience | Output | Prior rank | Prior val R@5 |",
        "| --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: |",
        *_config_table_rows(configs),
        "",
        "## Blocked Spectrogram Candidates",
        "",
        "| ID | Encoder | Loss | Status | Blocker |",
        "| --- | --- | --- | --- | --- |",
    ]
    for spec in BLOCKED_EXPERIMENTS:
        lines.append(
            f"| {spec['experiment_id']} | {spec['encoder']} | {spec['loss_combo']} | blocked | {spec['blocked_by']} |"
        )
    lines.extend(
        [
            "",
        "P1/P2 are emitted as blocked-compatible smoke placeholders because `launch_alignment_sweep.py` launches every `*.yaml` it finds. They use an E3 fallback encoder only to keep the queue parser safe; their metrics must not be interpreted as E7/spectrogram results.",
            "",
            "## Recommended Launcher Command",
            "",
            "```bash",
            "python scripts/launch_alignment_sweep.py \\",
            "  --config_dir configs/parallel_alignment_search \\",
            "  --out outputs/alignment_search \\",
            "  --max_concurrent 2 \\",
            "  --screen_epochs 50 \\",
            "  --poll_seconds 20",
            "```",
            "",
            "For a cautious first pass, launch only the seed-42 primary candidates by temporarily selecting `T4_stage2.yaml`, `S1_stage2.yaml`, `S3_stage2.yaml`, `X2_stage2.yaml`, `G2_recheck.yaml`, `G2_dropout035_recheck.yaml`, and `S1_dropout03_recheck.yaml` into a separate config directory.",
            "",
            "## Selection Rule",
            "",
            "Rank by validation R@5 first, then class accuracy and test R@5/R@10 after validation-based selection. Treat `G2_recheck` and `G2_dropout035_recheck` as stability checks because Day4/Day5 strong-encoder behavior was split-sensitive.",
        ]
    )
    PLAN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pending_summary(configs: list[dict[str, Any]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Alignment Search Summary",
        "",
        "Status: `pending`",
        "",
        f"- Runnable configs generated: `{len(configs)}`",
        f"- Blocked spectrogram candidates: `{len(BLOCKED_EXPERIMENTS)}`",
        "- GPU training launched by this script: `no`",
        "- E7 blocker: `yes`",
        "",
        "## Pending Queue",
        "",
        "| ID | Seed | Encoder | Loss | Status |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for config in configs:
        lines.append(
            f"| {config['experiment_id']} | {config['seed']} | {config['model'].get('encoder_type', '')} | {config['loss'].get('loss_combo', '')} | queued |"
        )
    lines.extend(["", "## Blocked", ""])
    for spec in BLOCKED_EXPERIMENTS:
        lines.append(f"- `{spec['experiment_id']}`: {spec['blocked_by']}")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate() -> list[dict[str, Any]]:
    TARGET_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    configs = [build_config(spec) for spec in [*RUNNABLE_EXPERIMENTS, *BLOCKED_PLACEHOLDER_EXPERIMENTS]]
    for config in configs:
        _write_yaml(TARGET_CONFIG_DIR / f"{config['experiment_id']}.yaml", config)
    write_plan(configs)
    write_pending_summary(configs)
    return configs


def main() -> None:
    configs = generate()
    print(f"generated={len(configs)}")
    print(f"config_dir={TARGET_CONFIG_DIR.relative_to(ROOT)}")
    print(f"plan={PLAN_PATH.relative_to(ROOT)}")
    print(f"summary={SUMMARY_PATH.relative_to(ROOT)}")
    print(f"e7_blocked={not E7_IMPLEMENTED}")


if __name__ == "__main__":
    main()
