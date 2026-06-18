from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.launch_alignment_sweep import (  # noqa: E402
    _write_ranking,
    _write_search_summary,
    _write_table,
    load_completed_row,
)


def refresh(out_root: Path) -> None:
    rows = []
    for exp_dir in sorted(path for path in out_root.iterdir() if path.is_dir()):
        config = exp_dir / "config.yaml"
        if not config.exists():
            continue
        if (exp_dir / "metrics.json").exists():
            status = "completed"
        elif (exp_dir / "train.log").exists():
            status = "running"
        else:
            status = "queued"
        rows.append(load_completed_row(config, exp_dir, status=status))
    _write_table(out_root / "EXPERIMENT_BOARD.csv", rows)
    _write_ranking(out_root, rows)
    _write_search_summary(out_root, rows)
    print(out_root / "EXPERIMENT_BOARD.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh alignment board from per-run saved configs.")
    parser.add_argument("--out", default="outputs/alignment_search")
    args = parser.parse_args()
    refresh(Path(args.out))


if __name__ == "__main__":
    main()
