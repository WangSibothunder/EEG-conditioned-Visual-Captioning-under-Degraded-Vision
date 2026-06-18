from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class HeavyStageQueueReconcileTests(unittest.TestCase):
    def test_reconcile_marks_transfer_running_from_launcher_log(self) -> None:
        from scripts.reconcile_heavy_stage_queue_status import reconcile_transfer_job_status

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            queue = root / "queue.yaml"
            transfer_dir = root / "transfer"
            transfer_dir.mkdir()
            queue.write_text(
                "\n".join(
                    [
                        "jobs:",
                        "  - id: TRANSFER_EEG_IMAGENET_PRETRAIN_TO_THOUGHT2TEXT",
                        "    priority: 70",
                        "    status: waiting",
                        "    command: \"bash scripts/run_masked_pretrain_transfer.sh\"",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (transfer_dir / "transfer_launcher.log").write_text("pretrain_finished_at=2026-06-16T12:10:00Z\n", encoding="utf-8")

            changed = reconcile_transfer_job_status(queue, transfer_dir)

            self.assertTrue(changed)
            self.assertIn("status: running", queue.read_text(encoding="utf-8"))

    def test_reconcile_marks_transfer_completed_from_artifacts(self) -> None:
        from scripts.reconcile_heavy_stage_queue_status import reconcile_transfer_job_status

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            queue = root / "queue.yaml"
            transfer_dir = root / "transfer"
            (transfer_dir / "checkpoints").mkdir(parents=True)
            queue.write_text(
                "\n".join(
                    [
                        "jobs:",
                        "  - id: TRANSFER_EEG_IMAGENET_PRETRAIN_TO_THOUGHT2TEXT",
                        "    priority: 70",
                        "    status: running",
                        "    command: \"bash scripts/run_masked_pretrain_transfer.sh\"",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (transfer_dir / "checkpoints" / "best.pt").write_bytes(b"fake")
            (transfer_dir / "retrieval_metrics.json").write_text("{}", encoding="utf-8")

            changed = reconcile_transfer_job_status(queue, transfer_dir)

            self.assertTrue(changed)
            self.assertIn("status: completed", queue.read_text(encoding="utf-8"))

    def test_reconcile_is_noop_without_signals(self) -> None:
        from scripts.reconcile_heavy_stage_queue_status import reconcile_transfer_job_status

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            queue = root / "queue.yaml"
            transfer_dir = root / "transfer"
            transfer_dir.mkdir()
            original = "\n".join(
                [
                    "jobs:",
                    "  - id: TRANSFER_EEG_IMAGENET_PRETRAIN_TO_THOUGHT2TEXT",
                    "    priority: 70",
                    "    status: waiting",
                    "",
                ]
            )
            queue.write_text(original, encoding="utf-8")

            changed = reconcile_transfer_job_status(queue, transfer_dir)

            self.assertFalse(changed)
            self.assertEqual(queue.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
