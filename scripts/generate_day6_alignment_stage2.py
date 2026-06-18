from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_DIR = ROOT / "configs" / "generated_alignment"
TARGET_CONFIG_DIR = ROOT / "configs" / "day6_alignment_stage2"
TARGET_OUTPUT_ROOT = Path("outputs/day6_alignment_stage2")
TARGET_PLAN = ROOT / TARGET_OUTPUT_ROOT / "STAGE2_PLAN.md"


STAGE2_EXPERIMENTS = [
    {
        "source_id": "T4",
        "experiment_id": "T4_stage2",
        "seed": 42,
        "reason": "Day4 top-ranked candidate; simplest winning loss combo L1 with convtransformer_base.",
    },
    {
        "source_id": "S1",
        "experiment_id": "S1_stage2",
        "seed": 42,
        "reason": "Second-ranked candidate; tests whether multi-positive/prototype terms remain useful at longer training.",
    },
    {
        "source_id": "S3",
        "experiment_id": "S3_stage2",
        "seed": 42,
        "reason": "Third-ranked candidate; rechecks similarity distillation after Day4 showed mixed validation/test behavior.",
    },
    {
        "source_id": "X2",
        "experiment_id": "X2_stage2",
        "seed": 42,
        "reason": "Best subject-adaptive candidate; keeps the L7 same-image/subject consistency ablation alive.",
    },
    {
        "source_id": "G2",
        "experiment_id": "G2_recheck",
        "seed": 42,
        "reason": "Strong encoder anomaly check; weak validation R@5 but high Day4 test R@5/R@10 warrants a controlled rerun.",
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
        "reason": "Multi-seed robustness check for the strongest structured-loss candidate.",
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
        "reason": "Second alternate seed for the strongest structured-loss candidate.",
    },
]


DAY4_METRICS = {
    "T4": {"rank": 1, "val_r5": 0.291291, "test_r5": 0.294294, "class_acc": 0.248326},
    "S1": {"rank": 2, "val_r5": 0.288288, "test_r5": 0.306306, "class_acc": 0.251744},
    "S3": {"rank": 3, "val_r5": 0.288288, "test_r5": 0.294294, "class_acc": 0.232878},
    "X2": {"rank": 4, "val_r5": 0.282282, "test_r5": 0.288288, "class_acc": 0.220507},
    "G2": {"rank": 12, "val_r5": 0.032032, "test_r5": 0.324324, "class_acc": 0.281206},
}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing source config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _stage2_config(spec: dict[str, Any]) -> dict[str, Any]:
    source_path = SOURCE_CONFIG_DIR / f"{spec['source_id']}.yaml"
    config = copy.deepcopy(_load_yaml(source_path))
    experiment_id = str(spec["experiment_id"])

    config["experiment_id"] = experiment_id
    config["seed"] = int(spec["seed"])
    config.setdefault("train", {})
    config["train"]["epochs"] = 50
    config["train"]["patience"] = 8
    config.setdefault("output", {})
    config["output"]["dir"] = str(TARGET_OUTPUT_ROOT / experiment_id)
    config["notes"] = (
        f"Day6 Stage2 from {spec['source_id']}: {spec['reason']} "
        "Original encoder and loss settings are preserved."
    )
    return config


def _plan_lines(written: list[Path]) -> list[str]:
    lines = [
        "# Day6 Alignment Stage2 Plan",
        "",
        "## Scope",
        "",
        "Generate longer alignment reruns from Day4 top candidates without launching GPU training.",
        "All Stage2 configs set `train.epochs=50`, `train.patience=8`, preserve the source encoder/loss fields, and write outputs under `outputs/day6_alignment_stage2/<experiment_id>`.",
        "",
        "## Candidate Rationale",
        "",
        "| Stage2 ID | Source | Seed | Day4 rank | Day4 val R@5 | Day4 test R@5 | Why run |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for spec in STAGE2_EXPERIMENTS:
        metrics = DAY4_METRICS[spec["source_id"]]
        lines.append(
            "| {experiment_id} | {source_id} | {seed} | {rank} | {val_r5:.6f} | {test_r5:.6f} | {reason} |".format(
                experiment_id=spec["experiment_id"],
                source_id=spec["source_id"],
                seed=int(spec["seed"]),
                rank=int(metrics["rank"]),
                val_r5=float(metrics["val_r5"]),
                test_r5=float(metrics["test_r5"]),
                reason=spec["reason"],
            )
        )
    lines.extend(
        [
            "",
            "## Expected Metrics",
            "",
            "- Primary selection metric remains validation `R@5`, with test `R@5/R@10` used after validation-based selection.",
            "- A useful Stage2 run should roughly match or improve the Day4 top validation band: `val R@5 ~= 0.28-0.30`.",
            "- `T4_stage2` is the default fusion-checkpoint candidate unless `S1_stage2` or a multi-seed S1/T4 run improves validation R@5 without a large test drop.",
            "- `G2_recheck` is not a default winner candidate; it is included to verify whether the Day4 strong-encoder test spike was stable or a split/early-stop artifact.",
            "",
            "## Recommended Launcher Commands",
            "",
            "Generate or refresh configs:",
            "",
            "```bash",
            "python scripts/generate_day6_alignment_stage2.py",
            "```",
            "",
            "Launch the full Stage2 queue when GPU scheduling permits:",
            "",
            "```bash",
            "python scripts/launch_alignment_sweep.py \\",
            "  --config_dir configs/day6_alignment_stage2 \\",
            "  --out outputs/day6_alignment_stage2 \\",
            "  --max_concurrent 2 \\",
            "  --screen_epochs 50 \\",
            "  --poll_seconds 20",
            "```",
            "",
            "Launch only the primary seed-42 configs by temporarily moving or selecting the five seed-42 YAMLs manually. Do not launch the multi-seed configs until the seed-42 reruns are reviewed.",
            "",
            "Direct single-run command example:",
            "",
            "```bash",
            "python -m src.train.train_align \\",
            "  --config configs/day6_alignment_stage2/T4_stage2.yaml \\",
            "  --output_dir outputs/day6_alignment_stage2/T4_stage2",
            "```",
            "",
            "## Generated Files",
            "",
        ]
    )
    for path in written:
        lines.append(f"- `{path.relative_to(ROOT)}`")
    return lines


def main() -> None:
    TARGET_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_PLAN.parent.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for spec in STAGE2_EXPERIMENTS:
        config = _stage2_config(spec)
        path = TARGET_CONFIG_DIR / f"{spec['experiment_id']}.yaml"
        _write_yaml(path, config)
        written.append(path)

    TARGET_PLAN.write_text("\n".join(_plan_lines(written)) + "\n", encoding="utf-8")
    print(f"Wrote {len(written)} Stage2 configs to {TARGET_CONFIG_DIR.relative_to(ROOT)}")
    print(f"Wrote plan to {TARGET_PLAN.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
