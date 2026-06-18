from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.launch_alignment_sweep import BoardRow, controlled_comparison_notes, refresh_search_artifacts


class AlignmentSearchRefreshTests(unittest.TestCase):
    def test_controlled_comparison_notes_use_screening_rows(self) -> None:
        rows = [
            BoardRow("S1", "convtransformer_base", 1, "L1+L2+L4", 42, "completed", val_R5="0.30", test_R5="0.30", class_acc="0.20"),
            BoardRow("S3", "convtransformer_base", 1, "L1+L2+L4+L5", 42, "completed", val_R5="0.20", test_R5="0.20", class_acc="0.20"),
            BoardRow("S4", "convtransformer_base", 1, "L1+L2+L4+L5+L6", 42, "completed", val_R5="0.10", test_R5="0.10", class_acc="0.20"),
            BoardRow("X1", "subject_adaptive", 1, "L1+L2+L4+L5", 42, "completed", val_R5="0.30", test_R5="0.30", class_acc="0.20"),
            BoardRow("X2", "subject_adaptive", 1, "L1+L2+L4+L5+L7", 42, "completed", val_R5="0.25", test_R5="0.25", class_acc="0.20"),
        ]

        notes = controlled_comparison_notes(rows)

        self.assertIn("did not help", notes["same_image_subject"])
        self.assertIn("did not help", notes["similarity"])
        self.assertIn("did not help", notes["augmentation"])

    def test_refresh_search_artifacts_writes_top_and_summary_from_existing_board(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "A" / "checkpoints").mkdir(parents=True)
            (root / "A" / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
            board = root / "EXPERIMENT_BOARD.csv"
            with board.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "experiment_id",
                        "encoder_type",
                        "param_count",
                        "loss_combo",
                        "seed",
                        "status",
                        "best_epoch",
                        "val_R@1",
                        "val_R@5",
                        "val_R@10",
                        "test_R@1",
                        "test_R@5",
                        "test_R@10",
                        "class_acc",
                        "mean_rank",
                        "gpu_mem_peak",
                        "time_minutes",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "experiment_id": "A",
                        "encoder_type": "convtransformer_base",
                        "param_count": "1",
                        "loss_combo": "L1",
                        "seed": "42",
                        "status": "completed",
                        "best_epoch": "2",
                        "val_R@1": "0.1",
                        "val_R@5": "0.2",
                        "val_R@10": "0.3",
                        "test_R@1": "0.1",
                        "test_R@5": "0.2",
                        "test_R@10": "0.3",
                        "class_acc": "0.4",
                        "mean_rank": "5",
                        "gpu_mem_peak": "100",
                        "time_minutes": "1",
                        "notes": "",
                    }
                )

            refresh_search_artifacts(root)

            self.assertTrue((root / "TOP_CANDIDATES.md").exists())
            self.assertTrue((root / "MULTISEED_FINAL.md").exists())
            self.assertTrue((root / "best_overall.pt").exists())


if __name__ == "__main__":
    unittest.main()
