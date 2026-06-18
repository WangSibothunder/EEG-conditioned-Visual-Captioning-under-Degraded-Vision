from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.gpu_monitor import collect_training_statuses, write_gpu_report


class GpuMonitorTrainingStatusTests(unittest.TestCase):
    def test_collect_training_statuses_reads_latest_step_and_best_val(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "outputs/pretrain/run_a"
            run_dir.mkdir(parents=True)
            (run_dir / "train.stdout.log").write_text(
                "\n".join(
                    [
                        '{"time": "2026-06-17T00:00:01Z", "epoch": 1, "step": 10, "batch_size": 128, "effective_batch_size": 128, "step_time": 0.5, "total": 3.0}',
                        '{"epoch": 1, "train_loss": 2.0, "val_loss": 1.5, "epoch_seconds": 12.0, "avg_step_time": 0.6, "gpu_mem_peak_mb": 2048.0}',
                        '{"time": "2026-06-17T00:01:01Z", "epoch": 2, "step": 20, "batch_size": 256, "effective_batch_size": 256, "step_time": 0.4, "total": 2.5}',
                        '{"epoch": 2, "train_loss": 1.8, "val_loss": 1.2, "epoch_seconds": 10.0, "avg_step_time": 0.5, "gpu_mem_peak_mb": 3072.0}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            statuses = collect_training_statuses(root / "outputs")

            self.assertEqual(len(statuses), 1)
            status = statuses[0]
            self.assertEqual(status["active_job"], "run_a")
            self.assertEqual(status["epoch"], 2)
            self.assertEqual(status["step"], 20)
            self.assertEqual(status["batch_size"], 256)
            self.assertEqual(status["current_validation_metric"], 1.2)
            self.assertEqual(status["best_validation_metric"], 1.2)
            self.assertEqual(status["gpu_mem_peak_mb"], 3072.0)

    def test_collect_training_statuses_reads_alignment_train_log_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "outputs/transfer/things_m0_transfer"
            run_dir.mkdir(parents=True)
            (run_dir / "train_log.jsonl").write_text(
                "\n".join(
                    [
                        '{"step": 10, "epoch": 1, "total": 10.25, "infonce": 5.5}',
                        '{"step": 20, "epoch": 2, "total": 9.75, "infonce": 5.1}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            statuses = collect_training_statuses(root / "outputs")

            self.assertEqual(len(statuses), 1)
            status = statuses[0]
            self.assertEqual(status["active_job"], "things_m0_transfer")
            self.assertEqual(status["job_type"], "transfer")
            self.assertEqual(status["dataset"], "THINGS-EEG2")
            self.assertEqual(status["epoch"], 2)
            self.assertEqual(status["step"], 20)
            self.assertEqual(status["current_validation_metric"], 9.75)
            self.assertEqual(status["best_validation_metric"], 9.75)
            self.assertTrue(str(status["log_path"]).endswith("train_log.jsonl"))

    def test_gpu_report_includes_training_status_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "outputs"
            run_dir = out_dir / "pretrain/run_b"
            run_dir.mkdir(parents=True)
            (run_dir / "train.stdout.log").write_text(
                '{"epoch": 3, "train_loss": 4.0, "val_loss": 2.0, "epoch_seconds": 20.0, "avg_step_time": 0.7, "gpu_mem_peak_mb": 4096.0}\n',
                encoding="utf-8",
            )
            report_path = out_dir / "heavy_stage/GPU_UTILIZATION_REPORT.md"
            snapshot = {
                "timestamp_utc": "2026-06-17T00:00:00Z",
                "gpus": [
                    {
                        "memory_used_gb": 12.0,
                        "utilization_gpu_pct": 90,
                        "power_draw_w": 300.0,
                    }
                ],
                "active_python_jobs": [],
            }

            with patch("scripts.gpu_monitor.GPU_REPORT_PATH", report_path), patch(
                "scripts.gpu_monitor.GPU_SAMPLES_PATH",
                out_dir / "heavy_stage/gpu_samples.jsonl",
            ):
                write_gpu_report(snapshot, outputs_root=out_dir)

            text = report_path.read_text(encoding="utf-8")
            self.assertIn("## Training Status", text)
            self.assertIn("run_b", text)
            self.assertIn("2.0000", text)


if __name__ == "__main__":
    unittest.main()
