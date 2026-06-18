from __future__ import annotations

import csv
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]


def _write_gap_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "corruption",
                "real_beats_controls",
                "real_beats_vision",
                "real_minus_vision",
                "real_minus_shuffled",
                "real_minus_random",
                "real_top5_minus_vision",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "corruption": "clean",
                "real_beats_controls": "true",
                "real_beats_vision": "false",
                "real_minus_vision": "-0.1",
                "real_minus_shuffled": "0.2",
                "real_minus_random": "0.3",
                "real_top5_minus_vision": "-0.05",
            }
        )
        writer.writerow(
            {
                "corruption": "strong_blur",
                "real_beats_controls": "true",
                "real_beats_vision": "true",
                "real_minus_vision": "0.04",
                "real_minus_shuffled": "0.15",
                "real_minus_random": "0.22",
                "real_top5_minus_vision": "0.03",
            }
        )


class FinalSemanticMaterializerTests(unittest.TestCase):
    def test_cli_materializes_custom_transfer_eval_dirs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "transfer_eval"
            robustness = root / "robustness_transfer_eval"
            out = root / "final_semantic"
            source.mkdir()
            robustness.mkdir()
            (source / "FULL_METRICS.csv").write_text("corruption,mode,top1_acc\nclean,real_eeg,0.5\n", encoding="utf-8")
            (source / "FULL_METRICS.md").write_text("# Full Metrics\n", encoding="utf-8")
            _write_gap_metrics(source / "SEMANTIC_GAP_METRICS.csv")
            (source / "SEMANTIC_GAP_METRICS.md").write_text("# Gap Metrics\n", encoding="utf-8")
            (source / "qualitative_examples.md").write_text("# Examples\n", encoding="utf-8")
            (robustness / "ROBUSTNESS_REPORT.md").write_text("# Robustness\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "materialize_final_semantic_report.py"),
                    "--source-dir",
                    str(source),
                    "--robustness-dir",
                    str(robustness),
                    "--out-dir",
                    str(out),
                ],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((out / "FULL_METRICS.csv").read_text(encoding="utf-8"), (source / "FULL_METRICS.csv").read_text(encoding="utf-8"))
            self.assertTrue((out / "FULL_ROBUST_SEMANTIC_REPORT.md").exists())
            report = (out / "FULL_ROBUST_SEMANTIC_REPORT.md").read_text(encoding="utf-8")
            self.assertIn(f"Source semantic directory: `{source}`", report)
            self.assertIn(f"Source robustness directory: `{robustness}`", report)
            self.assertIn("Real EEG beats shuffled/random controls: `2/2`", report)
            self.assertIn("Real EEG beats vision-only: `1/2`", report)

    def test_partial_control_wins_are_not_materialized_as_supported_claim(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "transfer_eval"
            robustness = root / "robustness_transfer_eval"
            out = root / "final_semantic"
            source.mkdir()
            robustness.mkdir()
            (source / "FULL_METRICS.csv").write_text("corruption,mode,top1_acc\nclean,real_eeg,0.5\n", encoding="utf-8")
            (source / "FULL_METRICS.md").write_text("# Full Metrics\n", encoding="utf-8")
            with (source / "SEMANTIC_GAP_METRICS.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "corruption",
                        "real_beats_controls",
                        "real_beats_vision",
                        "real_minus_vision",
                        "real_minus_shuffled",
                        "real_minus_random",
                        "real_top5_minus_vision",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "corruption": "clean",
                        "real_beats_controls": "true",
                        "real_beats_vision": "false",
                        "real_minus_vision": "-0.1",
                        "real_minus_shuffled": "0.1",
                        "real_minus_random": "0.2",
                        "real_top5_minus_vision": "-0.1",
                    }
                )
                writer.writerow(
                    {
                        "corruption": "strong_blur",
                        "real_beats_controls": "false",
                        "real_beats_vision": "false",
                        "real_minus_vision": "-0.2",
                        "real_minus_shuffled": "-0.01",
                        "real_minus_random": "0.02",
                        "real_top5_minus_vision": "-0.1",
                    }
                )
            (source / "SEMANTIC_GAP_METRICS.md").write_text("# Gap Metrics\n", encoding="utf-8")
            (robustness / "ROBUSTNESS_REPORT.md").write_text("# Robustness\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "materialize_final_semantic_report.py"),
                    "--source-dir",
                    str(source),
                    "--robustness-dir",
                    str(robustness),
                    "--out-dir",
                    str(out),
                ],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = (out / "FULL_ROBUST_SEMANTIC_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("Real EEG beats shuffled/random controls: `1/2`", report)
            self.assertIn("Limited:", report)
            self.assertNotIn("Supported: paired real EEG carries class-level semantic information beyond shuffled/random controls across clean and degraded conditions.", report)

    def test_report_can_surface_primary_a2_summary_alongside_transfer_result(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "transfer_eval"
            robustness = root / "robustness_transfer_eval"
            out = root / "final_semantic"
            primary = root / "A2_SUMMARY.md"
            source.mkdir()
            robustness.mkdir()
            (source / "FULL_METRICS.csv").write_text("corruption,mode,top1_acc\nclean,real_eeg,0.5\n", encoding="utf-8")
            (source / "FULL_METRICS.md").write_text("# Full Metrics\n", encoding="utf-8")
            _write_gap_metrics(source / "SEMANTIC_GAP_METRICS.csv")
            (source / "SEMANTIC_GAP_METRICS.md").write_text("# Gap Metrics\n", encoding="utf-8")
            (robustness / "ROBUSTNESS_REPORT.md").write_text("# Robustness\n", encoding="utf-8")
            primary.write_text(
                "\n".join(
                    [
                        "# A2 Semantic Fusion Multi-seed Strong Eval Summary",
                        "",
                        "- Real EEG beats shuffled/random controls in `18/18` seed-condition pairs.",
                        "- Real EEG beats vision-only in `18/18` seed-condition pairs.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "materialize_final_semantic_report.py"),
                    "--source-dir",
                    str(source),
                    "--robustness-dir",
                    str(robustness),
                    "--out-dir",
                    str(out),
                    "--primary-summary",
                    str(primary),
                ],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = (out / "FULL_ROBUST_SEMANTIC_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("## Primary Evidence", report)
            self.assertIn("A2 Semantic Fusion Multi-seed Strong Eval Summary", report)
            self.assertIn("18/18", report)
            self.assertIn("## Transfer Evaluation Kept As Secondary Evidence", report)


if __name__ == "__main__":
    unittest.main()
