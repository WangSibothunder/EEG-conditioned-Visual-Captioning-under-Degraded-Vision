from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.make_robustness_report import main as robustness_main


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["corruption", "mode", "accuracy", "top5_accuracy", "bleu_1", "bleu_4", "rouge_l", "class_hit"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _condition_rows(corruption: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for mode, value in {
        "vision_only": 0.2,
        "real_eeg": 0.4,
        "shuffled_eeg": 0.1,
        "random_eeg": 0.05,
    }.items():
        rows.append(
            {
                "corruption": corruption,
                "mode": mode,
                "accuracy": str(value),
                "top5_accuracy": str(value),
                "bleu_1": str(value),
                "bleu_4": str(value),
                "rouge_l": str(value),
                "class_hit": str(value),
            }
        )
    return rows


class RobustnessReportTests(unittest.TestCase):
    def test_semantic_robustness_uses_corruptions_from_semantic_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            day5 = root / "day5.csv"
            semantic_dir = root / "semantic"
            out_dir = root / "robustness"
            rows = _condition_rows("strong_blur")
            _write_rows(day5, rows)
            _write_rows(semantic_dir / "FULL_METRICS.csv", rows)

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "make_robustness_report.py",
                    "--day5_csv",
                    str(day5),
                    "--day5_dir",
                    str(root / "missing_jsonl"),
                    "--semantic_dir",
                    str(semantic_dir),
                    "--out_dir",
                    str(out_dir),
                ]
                robustness_main()
            finally:
                sys.argv = old_argv

            semantic_full = (out_dir / "semantic_full_metrics.csv").read_text(encoding="utf-8")

        self.assertIn("strong_blur", semantic_full)


if __name__ == "__main__":
    unittest.main()
