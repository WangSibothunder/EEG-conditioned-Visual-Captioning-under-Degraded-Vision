from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class OvernightReportTests(unittest.TestCase):
    def test_make_overnight_report_summarizes_metrics_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "overnight"
            align_dir = root / "align_strong"
            align_dir.mkdir(parents=True)
            (align_dir / "retrieval_metrics.json").write_text(
                json.dumps(
                    {
                        "model": {"r@1": 0.1, "r@5": 0.3, "r@10": 0.5, "mean_rank": 12.0},
                        "random": {"r@5": 0.05},
                    }
                ),
                encoding="utf-8",
            )
            (align_dir / "checkpoints").mkdir()
            (align_dir / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
            sanity = root / "sanity_mini"
            sanity.mkdir()
            (sanity / "metrics.md").write_text(
                "| file | corruption | mode | bleu_1 | rouge_l | avg_prediction_length | distinct_prediction_ratio |\n"
                "| --- | --- | --- | --- | --- | --- | --- |\n"
                "| clean_real_eeg.jsonl | clean | real_eeg | 0.2 | 0.3 | 4.0 | 0.5 |\n",
                encoding="utf-8",
            )
            out = root / "OVERNIGHT_REPORT.md"

            result = subprocess.run(
                [sys.executable, "scripts/make_overnight_report.py", "--root", str(root), "--out", str(out)],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            text = out.read_text(encoding="utf-8")
            self.assertIn("| Run | Loss | R@1 | R@5 | R@10 | Mean Rank | Random R@5 | Notes |", text)
            self.assertIn("| Corruption | Mode | BLEU-1 | ROUGE-L | Avg Len | Distinct Ratio | Notes |", text)
            self.assertIn("align_strong/checkpoints/best.pt", text)


if __name__ == "__main__":
    unittest.main()
