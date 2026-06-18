from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class LaunchTransferAfterPretrainArtifactTests(unittest.TestCase):
    def test_ready_requires_checkpoint_and_optional_report(self) -> None:
        from scripts.launch_transfer_after_pretrain_artifact import pretrain_artifact_ready

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "checkpoints" / "best_masked_eeg.pt"
            report = root / "MASKED_EEG_PRETRAIN_REPORT.md"

            self.assertFalse(pretrain_artifact_ready(checkpoint, report))
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            self.assertFalse(pretrain_artifact_ready(checkpoint, report))
            report.write_text("# report\n", encoding="utf-8")
            self.assertTrue(pretrain_artifact_ready(checkpoint, report))
            self.assertTrue(pretrain_artifact_ready(checkpoint, None))

    def test_build_transfer_command_uses_existing_transfer_runner(self) -> None:
        from scripts.launch_transfer_after_pretrain_artifact import build_transfer_command

        command = build_transfer_command(
            config=Path("configs/transfer/things_raw_pretrain_t2t_align.yaml"),
            output_dir=Path("outputs/transfer/things_raw_pretrain_t2t_align"),
        )

        self.assertEqual(command[0], "bash")
        self.assertEqual(command[1], "scripts/run_masked_pretrain_transfer.sh")
        self.assertIn("configs/transfer/things_raw_pretrain_t2t_align.yaml", command)
        self.assertIn("outputs/transfer/things_raw_pretrain_t2t_align", command)

    def test_recovery_report_is_not_written_while_training_pid_is_alive(self) -> None:
        from scripts.launch_transfer_after_pretrain_artifact import maybe_write_recovery_report

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report = root / "MASKED_EEG_PRETRAIN_REPORT.md"
            checkpoint = root / "checkpoints" / "best_masked_eeg.pt"
            metrics = root / "metrics.json"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            metrics.write_text('{"history": [{"epoch": 3, "val_loss": 2.0}], "best_val_loss": 2.0}', encoding="utf-8")

            with patch("scripts.launch_transfer_after_pretrain_artifact.pid_alive", return_value=True):
                wrote = maybe_write_recovery_report(
                    checkpoint=checkpoint,
                    report=report,
                    metrics=metrics,
                    training_pids=[123],
                    log=root / "launcher.log",
                )

            self.assertFalse(wrote)
            self.assertFalse(report.exists())

    def test_recovery_report_is_written_after_training_pid_exits(self) -> None:
        from scripts.launch_transfer_after_pretrain_artifact import maybe_write_recovery_report

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report = root / "MASKED_EEG_PRETRAIN_REPORT.md"
            checkpoint = root / "checkpoints" / "best_masked_eeg.pt"
            metrics = root / "metrics.json"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            metrics.write_text(
                '{"history": [{"epoch": 1, "val_loss": 3.0}, {"epoch": 2, "val_loss": 2.0}], "best_val_loss": 2.0}',
                encoding="utf-8",
            )

            with patch("scripts.launch_transfer_after_pretrain_artifact.pid_alive", return_value=False):
                wrote = maybe_write_recovery_report(
                    checkpoint=checkpoint,
                    report=report,
                    metrics=metrics,
                    training_pids=[123],
                    log=root / "launcher.log",
                )

            self.assertTrue(wrote)
            text = report.read_text(encoding="utf-8")
            self.assertIn("Recovered Masked EEG Pretraining Report", text)
            self.assertIn("best_masked_eeg.pt", text)


if __name__ == "__main__":
    unittest.main()
