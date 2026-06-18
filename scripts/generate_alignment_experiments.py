from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import deep_update, load_config


DEFAULT_SWEEP = Path("configs/sweeps/alignment_search.yaml")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml
    except ModuleNotFoundError:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _has_repeated_image(manifest: str | Path) -> bool:
    seen: set[str] = set()
    path = Path(manifest)
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            image_id = str(json.loads(line).get("image_id", ""))
            if image_id in seen:
                return True
            seen.add(image_id)
    return False


def build_experiments(sweep_config: dict[str, Any], *, output_root: str | Path | None = None) -> list[dict[str, Any]]:
    output_root = Path(output_root or sweep_config.get("output", {}).get("root", "outputs/day4_search"))
    repeated_image = _has_repeated_image(sweep_config["data"]["train_manifest"])
    experiments: list[dict[str, Any]] = []
    for candidate in sweep_config.get("candidates", []):
        if bool(candidate.get("requires_repeated_image", False)) and not repeated_image:
            continue
        experiment_id = str(candidate["experiment_id"])
        config: dict[str, Any] = {
            "experiment_id": experiment_id,
            "seed": int(candidate.get("seed", sweep_config.get("seed", 42))),
            "device": sweep_config.get("device", "auto"),
            "data": copy.deepcopy(sweep_config["data"]),
            "model": copy.deepcopy(sweep_config["model"]),
            "loss": copy.deepcopy(sweep_config.get("loss_defaults", {})),
            "train": copy.deepcopy(sweep_config["train"]),
            "output": {"dir": str(output_root / experiment_id)},
            "notes": str(candidate.get("notes", "")),
        }
        config["model"]["encoder_type"] = str(candidate["encoder_type"])
        config["loss"]["loss_combo"] = str(candidate.get("loss_combo", ""))
        deep_update(config["loss"], copy.deepcopy(candidate.get("loss", {})))
        if "train" in candidate:
            deep_update(config["train"], copy.deepcopy(candidate["train"]))
        if "model" in candidate:
            deep_update(config["model"], copy.deepcopy(candidate["model"]))
        experiments.append(config)
    return experiments


def write_plan(path: str | Path, experiments: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Day4 Alignment Search Experiment Plan",
        "",
        f"Generated experiments: `{len(experiments)}`",
        "",
        "| ID | Encoder | Loss Combo | Epochs | Output |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for config in experiments:
        lines.append(
            f"| {config['experiment_id']} | {config['model']['encoder_type']} | "
            f"{config['loss'].get('loss_combo', '')} | {int(config['train'].get('epochs', 0))} | "
            f"{config['output']['dir']} |"
        )
    lines.extend(
        [
            "",
            "Stage 1 uses short screening runs. Stage 2 should select top candidates by validation R@5, class accuracy, and stability.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_experiment_configs(
    out_dir: str | Path = "configs/generated_alignment",
    plan_path: str | Path = "outputs/day4_search/EXPERIMENT_PLAN.md",
    sweep_path: str | Path = DEFAULT_SWEEP,
) -> list[dict[str, Any]]:
    sweep_config = load_config(sweep_path)
    experiments = build_experiments(sweep_config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for config in experiments:
        _write_yaml(out_dir / f"{config['experiment_id']}.yaml", config)
    manifest = [{"experiment_id": config["experiment_id"], "config": f"{config['experiment_id']}.yaml"} for config in experiments]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_plan(plan_path, experiments)
    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Day4 alignment search configs.")
    parser.add_argument("--sweep", default=str(DEFAULT_SWEEP))
    parser.add_argument("--out", default="configs/generated_alignment")
    parser.add_argument("--plan", default="outputs/day4_search/EXPERIMENT_PLAN.md")
    args = parser.parse_args()
    experiments = generate_experiment_configs(out_dir=args.out, plan_path=args.plan, sweep_path=args.sweep)
    print(f"generated={len(experiments)} out={args.out} plan={args.plan}")


if __name__ == "__main__":
    main()
