from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.materialize_final_24h_results import materialize_final_results


def _write_semantic_metrics(path: Path, real: float, vision: float, shuffled: float, random: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["corruption", "mode", "count", "accuracy", "top5_accuracy", "caption_class_hit"],
        )
        writer.writeheader()
        for corruption in ["clean", "mixed"]:
            for mode, acc in [
                ("vision_only", vision),
                ("real_eeg", real),
                ("shuffled_eeg", shuffled),
                ("random_eeg", random),
                ("eeg_only", shuffled),
            ]:
                writer.writerow(
                    {
                        "corruption": corruption,
                        "mode": mode,
                        "count": 10,
                        "accuracy": acc,
                        "top5_accuracy": min(1.0, acc + 0.2),
                        "caption_class_hit": acc,
                    }
                )


def _write_alignment_metrics(path: Path, r5: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {
            "unique_image": {
                "r@1": 0.1,
                "r@5": r5,
                "r@10": 0.5,
                "class_acc": 0.3,
                "mean_rank": 12.0,
                "median_rank": 5.0,
            },
            "trial": {"r@1": 0.01, "r@5": 0.02, "r@10": 0.03},
        },
        "random": {"r@1": 0.003, "r@5": 0.015, "r@10": 0.03},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class Final24hMaterializerTests(unittest.TestCase):
    def test_materialize_final_results_writes_required_core_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_semantic_metrics(root / "a2_seed42" / "FULL_METRICS.csv", 0.8, 0.5, 0.2, 0.1)
            _write_semantic_metrics(root / "p2a2_seed42" / "FULL_METRICS.csv", 0.7, 0.5, 0.25, 0.1)
            _write_alignment_metrics(root / "p2_seed42" / "test_metrics.json", 0.4)
            out = root / "final_results"

            materialize_final_results(
                out_dir=out,
                a2_eval_dirs=[root / "a2_seed42"],
                p2a2_eval_dirs=[("P2A2_freeze_encoder", root / "p2a2_seed42")],
                p2_metric_files=[("seed42", "P2_seed42", root / "p2_seed42" / "test_metrics.json", root / "p2_seed42" / "checkpoints" / "best.pt")],
                gate_report=None,
            )

            for name in [
                "A2_FINAL_METRICS.csv",
                "A2_FINAL_SUMMARY.md",
                "A2_FINAL_EXAMPLES.md",
                "P2_ALIGNMENT_FINAL_METRICS.csv",
                "P2_ALIGNMENT_FINAL_SUMMARY.md",
                "P2A2_FINAL_METRICS.csv",
                "P2A2_FINAL_SUMMARY.md",
                "P2A2_FINAL_EXAMPLES.md",
                "FINAL_MODEL_SELECTION.md",
                "FINAL_24H_REPORT.md",
            ]:
                self.assertTrue((out / name).exists(), name)
            selection = (out / "FINAL_MODEL_SELECTION.md").read_text(encoding="utf-8")
            self.assertIn("Recommended final model", selection)
            self.assertIn("A2_final", selection)


if __name__ == "__main__":
    unittest.main()
