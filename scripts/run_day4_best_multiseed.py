from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config


VARIANT_CONFIGS = {
    "A_mse_ce_seed42": "configs/day4_align_A_mse_ce.yaml",
    "B_contrastive_seed42": "configs/day4_align_B_contrastive.yaml",
    "C_simdistill_seed42": "configs/day4_align_C_simdistill.yaml",
    "D_full_seed42": "configs/day4_align_D_full.yaml",
}


def _read_r5(path: Path) -> float:
    if not path.exists():
        return -1.0
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload.get("model", {}).get("r@5", -1.0))


def select_best_variant(root: Path) -> tuple[str, str]:
    candidates: list[tuple[float, str, str]] = []
    for dirname, config in VARIANT_CONFIGS.items():
        candidates.append((_read_r5(root / dirname / "alignment_metrics.json"), dirname, config))
    best_r5, best_dir, best_config = max(candidates, key=lambda item: item[0])
    if best_r5 < 0:
        raise SystemExit("No completed seed42 alignment variants found.")
    return best_dir, best_config


def write_seed_config(source_config: str, target_config: Path, seed: int, out_dir: str) -> None:
    cfg = load_config(source_config)
    cfg["seed"] = seed
    cfg.setdefault("output", {})["dir"] = out_dir
    try:
        import yaml
    except ModuleNotFoundError:
        target_config.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return
    target_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day4 multi-seed alignment for the best seed42 ablation variant.")
    parser.add_argument("--root", default="outputs/day4_alignment")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2025])
    args = parser.parse_args()

    root = Path(args.root)
    best_dir, best_config = select_best_variant(root)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "best_variant.txt").write_text(f"{best_dir}\n{best_config}\n", encoding="utf-8")
    for seed in args.seeds:
        out_dir = root / f"best_seed{seed}"
        cfg_path = Path("configs") / f"day4_align_best_seed{seed}.yaml"
        write_seed_config(best_config, cfg_path, seed, str(out_dir))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.train.train_align",
                "--config",
                str(cfg_path),
                "--output_dir",
                str(out_dir),
            ],
            check=True,
        )
    subprocess.run([sys.executable, "scripts/make_day4_alignment_reports.py", "--root", str(root)], check=True)


if __name__ == "__main__":
    main()
