from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.make_semantic_caption_report import build_report


FIELDNAMES = ["corruption", "mode", "count", "accuracy", "top5_accuracy"]


def _write_metric_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _condition_rows(corruption: str, real: float, shuffled: float, random: float, vision: float = 0.9) -> list[dict[str, str]]:
    values = {
        "vision_only": vision,
        "real_eeg": real,
        "shuffled_eeg": shuffled,
        "random_eeg": random,
        "eeg_only": 0.02,
    }
    return [
        {
            "corruption": corruption,
            "mode": mode,
            "count": "10",
            "accuracy": str(acc),
            "top5_accuracy": str(acc),
        }
        for mode, acc in values.items()
    ]


class SemanticCaptionReportTests(unittest.TestCase):
    def test_partial_control_wins_are_reported_as_limited_not_supported(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "FULL_METRICS.csv"
            rows: list[dict[str, str]] = []
            rows.extend(_condition_rows("clean", real=0.50, shuffled=0.40, random=0.30))
            rows.extend(_condition_rows("blur", real=0.50, shuffled=0.51, random=0.30))
            rows.extend(_condition_rows("occlusion", real=0.50, shuffled=0.40, random=0.51))
            rows.extend(_condition_rows("noise", real=0.50, shuffled=0.51, random=0.52))
            rows.extend(_condition_rows("lowres", real=0.50, shuffled=0.50, random=0.49))
            _write_metric_rows(metrics, rows)

            build_report(metrics, root)

            report = (root / "SEMANTIC_CAPTION_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("Real EEG beats shuffled/random controls: `1/5`", report)
            self.assertIn("Limited:", report)
            self.assertNotIn("every clean/degraded condition", report)
            self.assertNotIn("Supported: correctly paired EEG carries semantic information beyond shuffled/random EEG controls.", report)

    def test_report_derives_strong_corruptions_from_metrics_csv(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "FULL_METRICS.csv"
            rows: list[dict[str, str]] = []
            rows.extend(_condition_rows("clean", real=0.50, shuffled=0.40, random=0.30))
            rows.extend(_condition_rows("strong_blur", real=0.55, shuffled=0.30, random=0.20, vision=0.40))
            rows.extend(_condition_rows("mixed", real=0.20, shuffled=0.25, random=0.10, vision=0.30))
            _write_metric_rows(metrics, rows)

            build_report(metrics, root)

            report = (root / "SEMANTIC_CAPTION_REPORT.md").read_text(encoding="utf-8")
            gaps = (root / "SEMANTIC_GAP_METRICS.csv").read_text(encoding="utf-8")
            self.assertIn("Conditions evaluated: `3`", report)
            self.assertIn("Real EEG beats shuffled/random controls: `2/3`", report)
            self.assertIn("strong_blur", gaps)
            self.assertIn("mixed", gaps)


if __name__ == "__main__":
    unittest.main()
