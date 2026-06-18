from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.heavy_stage_scheduler import (
    GPUState,
    QueueItem,
    infer_item_status,
    launch_next_queued_job,
    should_write_idle_diagnosis,
    write_idle_diagnosis,
    write_live_scheduler_status,
)


class HeavyStageSchedulerTests(unittest.TestCase):
    def test_idle_diagnosis_requires_majority_underused_samples_in_recent_window(self) -> None:
        now = datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc)
        recent = [
            {"timestamp_utc": (now - timedelta(minutes=1)).isoformat(), "gpus": [{"memory_used_gb": 7.5, "utilization_gpu_pct": 70}]},
            {"timestamp_utc": (now - timedelta(minutes=2)).isoformat(), "gpus": [{"memory_used_gb": 9.0, "utilization_gpu_pct": 35}]},
            {"timestamp_utc": (now - timedelta(minutes=3)).isoformat(), "gpus": [{"memory_used_gb": 9.2, "utilization_gpu_pct": 35}]},
            {"timestamp_utc": (now - timedelta(minutes=11)).isoformat(), "gpus": [{"memory_used_gb": 1.0, "utilization_gpu_pct": 1}]},
        ]

        should_write, stats = should_write_idle_diagnosis(
            GPUState(available=True, memory_used_gb=9.5, memory_total_gb=24.0, utilization_gpu_pct=55),
            recent,
            current_time=now,
            window_minutes=10,
        )

        self.assertFalse(should_write)
        self.assertEqual(stats["window_samples"], 3)
        self.assertEqual(stats["underused_samples"], 1)

    def test_idle_diagnosis_triggers_when_majority_recent_samples_are_underused(self) -> None:
        now = datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc)
        recent = [
            {"timestamp_utc": (now - timedelta(minutes=1)).isoformat(), "gpus": [{"memory_used_gb": 7.5, "utilization_gpu_pct": 70}]},
            {"timestamp_utc": (now - timedelta(minutes=2)).isoformat(), "gpus": [{"memory_used_gb": 9.0, "utilization_gpu_pct": 25}]},
            {"timestamp_utc": (now - timedelta(minutes=3)).isoformat(), "gpus": [{"memory_used_gb": 7.2, "utilization_gpu_pct": 31}]},
            {"timestamp_utc": (now - timedelta(minutes=11)).isoformat(), "gpus": [{"memory_used_gb": 1.0, "utilization_gpu_pct": 1}]},
        ]

        should_write, stats = should_write_idle_diagnosis(
            GPUState(available=True, memory_used_gb=9.5, memory_total_gb=24.0, utilization_gpu_pct=55),
            recent,
            current_time=now,
            window_minutes=10,
        )

        self.assertTrue(should_write)
        self.assertEqual(stats["window_samples"], 3)
        self.assertEqual(stats["underused_samples"], 3)

    def test_launch_next_queued_job_is_noop_when_launch_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            rows = [
                {
                    "ID": "JOB_A",
                    "Priority": "1",
                    "Status": "queued",
                    "Command": "python train.py",
                    "Expected Output": "outputs/job_a/metrics.json",
                    "Notes": "",
                }
            ]

            launched = launch_next_queued_job(rows, launch_enabled=False, log_dir=log_dir)

            self.assertFalse(launched)
            self.assertFalse((log_dir / "auto_launches.jsonl").exists())

    def test_launch_next_queued_job_records_and_spawns_when_launch_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            rows = [
                {
                    "ID": "JOB_A",
                    "Priority": "1",
                    "Status": "queued",
                    "Command": "python train.py",
                    "Expected Output": "outputs/job_a/metrics.json",
                    "Notes": "",
                }
            ]

            with patch("scripts.heavy_stage_scheduler.subprocess.Popen") as popen:
                launched = launch_next_queued_job(rows, launch_enabled=True, log_dir=log_dir)

            self.assertTrue(launched)
            popen.assert_called_once()
            self.assertIn("python train.py", (log_dir / "auto_launches.jsonl").read_text(encoding="utf-8"))

    def test_launch_next_queued_job_updates_queue_status_when_queue_path_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            queue = root / "queue.yaml"
            queue.write_text(
                "\n".join(
                    [
                        "jobs:",
                        "  - id: JOB_A",
                        "    priority: 1",
                        "    status: queued",
                        "    command: \"python train.py\"",
                        "    expected_output: \"outputs/job_a/metrics.json\"",
                        "    notes: \"\"",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = [
                {
                    "ID": "JOB_A",
                    "Priority": "1",
                    "Status": "queued",
                    "Command": "python train.py",
                    "Expected Output": "outputs/job_a/metrics.json",
                    "Notes": "",
                }
            ]

            with patch("scripts.heavy_stage_scheduler.subprocess.Popen") as popen:
                launched = launch_next_queued_job(rows, launch_enabled=True, log_dir=root, queue_path=queue)

            self.assertTrue(launched)
            popen.assert_called_once()
            self.assertIn("status: running", queue.read_text(encoding="utf-8"))

    def test_launch_next_queued_job_can_launch_multiple_up_to_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            queue = root / "queue.yaml"
            queue.write_text(
                "\n".join(
                    [
                        "jobs:",
                        "  - id: JOB_A",
                        "    priority: 1",
                        "    status: queued",
                        "    command: \"python train_a.py\"",
                        "    expected_output: \"outputs/job_a/metrics.json\"",
                        "    notes: \"\"",
                        "  - id: JOB_B",
                        "    priority: 2",
                        "    status: queued",
                        "    command: \"python train_b.py\"",
                        "    expected_output: \"outputs/job_b/metrics.json\"",
                        "    notes: \"\"",
                        "  - id: JOB_C",
                        "    priority: 3",
                        "    status: queued",
                        "    command: \"python train_c.py\"",
                        "    expected_output: \"outputs/job_c/metrics.json\"",
                        "    notes: \"\"",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = [
                {
                    "ID": "JOB_A",
                    "Priority": "1",
                    "Status": "queued",
                    "Command": "python train_a.py",
                    "Expected Output": "outputs/job_a/metrics.json",
                    "Notes": "",
                },
                {
                    "ID": "JOB_B",
                    "Priority": "2",
                    "Status": "queued",
                    "Command": "python train_b.py",
                    "Expected Output": "outputs/job_b/metrics.json",
                    "Notes": "",
                },
                {
                    "ID": "JOB_C",
                    "Priority": "3",
                    "Status": "queued",
                    "Command": "python train_c.py",
                    "Expected Output": "outputs/job_c/metrics.json",
                    "Notes": "",
                },
            ]

            with patch("scripts.heavy_stage_scheduler.subprocess.Popen") as popen:
                launched = launch_next_queued_job(rows, launch_enabled=True, log_dir=root, queue_path=queue, max_launches=2)

            self.assertEqual(launched, 2)
            self.assertEqual(popen.call_count, 2)
            queue_text = queue.read_text(encoding="utf-8")
            self.assertEqual(queue_text.count("status: running"), 2)
            self.assertEqual(queue_text.count("status: queued"), 1)

    def test_launch_next_queued_job_limits_heavy_gpu_jobs_to_one_per_tick(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = [
                {
                    "ID": "HEAVY_ALIGN",
                    "Priority": "1",
                    "Status": "queued",
                    "Command": "python -m src.train.train_align --config heavy_align.yaml",
                    "Expected Output": "outputs/heavy_align/metrics.json",
                    "Notes": "",
                },
                {
                    "ID": "HEAVY_TRIMODAL",
                    "Priority": "2",
                    "Status": "queued",
                    "Command": "python -m src.train.train_trimodal_align --config heavy_trimodal.yaml",
                    "Expected Output": "outputs/heavy_trimodal/metrics.json",
                    "Notes": "",
                },
                {
                    "ID": "WATCHER",
                    "Priority": "3",
                    "Status": "queued",
                    "Command": "python scripts/launch_transfer_after_pretrain_artifact.py --checkpoint ckpt.pt",
                    "Expected Output": "outputs/watcher/metrics.json",
                    "Notes": "",
                },
            ]

            with patch("scripts.heavy_stage_scheduler.subprocess.Popen") as popen:
                launched = launch_next_queued_job(
                    rows,
                    launch_enabled=True,
                    log_dir=root,
                    max_launches=8,
                    gpu=GPUState(
                        available=True,
                        name="Test GPU",
                        memory_used_gb=1.0,
                        memory_total_gb=48.0,
                        utilization_gpu_pct=0,
                    ),
                )

            self.assertEqual(launched, 2)
            launched_commands = [call.args[0][-1] for call in popen.call_args_list]
            self.assertTrue(any("src.train.train_align" in command for command in launched_commands))
            self.assertTrue(any("launch_transfer_after_pretrain_artifact.py" in command for command in launched_commands))
            self.assertFalse(any("src.train.train_trimodal_align" in command for command in launched_commands))

    def test_launch_next_queued_job_skips_direct_gpu_training_when_memory_headroom_is_low(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = [
                {
                    "ID": "HEAVY_ALIGN",
                    "Priority": "1",
                    "Status": "queued",
                    "Command": "python -m src.train.train_align --config heavy.yaml",
                    "Expected Output": "outputs/heavy/metrics.json",
                    "Notes": "",
                },
                {
                    "ID": "WATCHER",
                    "Priority": "2",
                    "Status": "queued",
                    "Command": "python scripts/launch_transfer_after_pretrain_artifact.py --checkpoint ckpt.pt",
                    "Expected Output": "outputs/watcher/metrics.json",
                    "Notes": "",
                },
            ]

            with patch("scripts.heavy_stage_scheduler.subprocess.Popen") as popen:
                launched = launch_next_queued_job(
                    rows,
                    launch_enabled=True,
                    log_dir=root,
                    max_launches=8,
                    gpu=GPUState(
                        available=True,
                        name="Test GPU",
                        memory_used_gb=35.0,
                        memory_total_gb=48.0,
                        utilization_gpu_pct=20,
                    ),
                )

            self.assertEqual(launched, 1)
            popen.assert_called_once()
            launched_command = popen.call_args.args[0][-1]
            self.assertIn("launch_transfer_after_pretrain_artifact.py", launched_command)
            self.assertNotIn("src.train.train_align", launched_command)

    def test_idle_diagnosis_text_describes_auto_launch_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            samples_path = root / "gpu_samples.jsonl"
            diagnosis_path = root / "GPU_IDLE_DIAGNOSIS.md"
            now = datetime.now(timezone.utc)
            samples_path.write_text(
                "\n".join(
                    [
                        '{"timestamp_utc": "%s", "gpus": [{"memory_used_gb": 1.0, "utilization_gpu_pct": 0}]}'
                        % (now - timedelta(minutes=1)).isoformat(),
                        '{"timestamp_utc": "%s", "gpus": [{"memory_used_gb": 1.5, "utilization_gpu_pct": 5}]}'
                        % (now - timedelta(minutes=2)).isoformat(),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = [
                {
                    "ID": "JOB_A",
                    "Priority": "1",
                    "Status": "queued",
                    "Command": "python train.py",
                    "Expected Output": "outputs/job_a/metrics.json",
                    "Notes": "",
                }
            ]

            with patch("scripts.heavy_stage_scheduler.IDLE_DIAGNOSIS_PATH", diagnosis_path), patch(
                "scripts.heavy_stage_scheduler._load_recent_gpu_samples",
                return_value=[
                    {
                        "timestamp_utc": (now - timedelta(minutes=1)).isoformat(),
                        "gpus": [{"memory_used_gb": 1.0, "utilization_gpu_pct": 0}],
                    },
                    {
                        "timestamp_utc": (now - timedelta(minutes=2)).isoformat(),
                        "gpus": [{"memory_used_gb": 1.5, "utilization_gpu_pct": 5}],
                    },
                ],
            ):
                wrote = write_idle_diagnosis(
                    GPUState(
                        available=True,
                        name="Test GPU",
                        memory_used_gb=1.0,
                        memory_total_gb=48.0,
                        utilization_gpu_pct=0,
                    ),
                    rows,
                    active_python_jobs=0,
                )

            self.assertTrue(wrote)
            text = diagnosis_path.read_text(encoding="utf-8")
            self.assertIn("--launch-when-idle", text)
            self.assertNotIn("audit-only", text)

    def test_live_status_includes_queue_summary_and_active_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            live_status = Path(tmpdir) / "LIVE_STATUS.md"
            rows = [
                {
                    "ID": "JOB_A",
                    "Priority": "1",
                    "Status": "running",
                    "Command": "python train_a.py",
                    "Expected Output": "outputs/a/metrics.json",
                    "Notes": "",
                },
                {
                    "ID": "JOB_B",
                    "Priority": "2",
                    "Status": "queued",
                    "Command": "python train_b.py",
                    "Expected Output": "outputs/b/metrics.json",
                    "Notes": "",
                },
            ]
            active_jobs = [
                {
                    "pid": 123,
                    "ppid": 1,
                    "elapsed": "00:10:00",
                    "cpu_pct": 101.5,
                    "mem_pct": 2.5,
                    "command": "python -m src.train.train_align --config c.yaml",
                }
            ]

            write_live_scheduler_status(
                GPUState(
                    available=True,
                    name="Test GPU",
                    memory_used_gb=12.0,
                    memory_total_gb=24.0,
                    utilization_gpu_pct=75,
                    power_draw_w=150.0,
                ),
                rows,
                active_python_jobs=len(active_jobs),
                idle_written=False,
                auto_launched=False,
                active_jobs=active_jobs,
                live_status_path=live_status,
            )

            text = live_status.read_text(encoding="utf-8")
            self.assertIn("## Queue Summary", text)
            self.assertIn("- queued: 1", text)
            self.assertIn("## Active Python Jobs", text)
            self.assertIn("src.train.train_align", text)

    def test_infer_item_status_marks_running_job_completed_when_expected_output_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            expected = Path(tmpdir) / "metrics.json"
            expected.write_text("{}\n", encoding="utf-8")

            status = infer_item_status(
                QueueItem(
                    id="JOB_A",
                    command="python train.py",
                    status="running",
                    priority=1,
                    expected_output=str(expected),
                    notes="",
                )
            )

            self.assertEqual(status, "completed")

    def test_infer_item_status_keeps_active_training_process_running_even_if_artifact_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            expected = Path(tmpdir) / "TRIMODAL_FULL_REPORT.md"
            expected.write_text("# Report\n", encoding="utf-8")

            active_jobs = [
                {
                    "pid": "123",
                    "ppid": "1",
                    "elapsed": "00:10:00",
                    "cpu_pct": "95.0",
                    "mem_pct": "5.0",
                    "command": "python -m src.train.train_trimodal_align --config configs/trimodal/eeg_imagenet_exact_a2_scratch_full.yaml",
                }
            ]

            with patch("scripts.heavy_stage_scheduler.query_active_python_jobs", return_value=active_jobs):
                status = infer_item_status(
                    QueueItem(
                        id="TRIMODAL_EEG_IMAGENET_EXACT_A2_SCRATCH_FULL",
                        command="python -m src.train.train_trimodal_align --config configs/trimodal/eeg_imagenet_exact_a2_scratch_full.yaml",
                        status="running",
                        priority=136,
                        expected_output=str(expected),
                        notes="",
                    )
                )

            self.assertEqual(status, "running")

    def test_run_scheduler_writes_inferred_completed_status_back_to_queue(self) -> None:
        from scripts.heavy_stage_scheduler import run_scheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            expected = root / "outputs" / "trimodal" / "finished" / "TRIMODAL_FULL_REPORT.md"
            expected.parent.mkdir(parents=True)
            expected.write_text("# Report\n", encoding="utf-8")
            queue = root / "queue.yaml"
            queue.write_text(
                yaml.safe_dump(
                    {
                        "jobs": [
                            {
                                "id": "FINISHED_RUNNING_JOB",
                                "priority": 1,
                                "status": "running",
                                "command": "python -m src.train.train_trimodal_align --config configs/finished.yaml",
                                "expected_output": str(expected),
                                "notes": "",
                            }
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            with (
                patch("scripts.heavy_stage_scheduler.query_active_python_jobs", return_value=[]),
                patch(
                    "scripts.heavy_stage_scheduler.query_gpu_state",
                    return_value=GPUState(available=True, memory_used_gb=0.0, memory_total_gb=48.0, utilization_gpu_pct=0),
                ),
                patch("scripts.heavy_stage_scheduler.reconcile_transfer_job_status"),
                patch("scripts.heavy_stage_scheduler.write_board"),
                patch("scripts.heavy_stage_scheduler.write_live_scheduler_status"),
                patch("scripts.heavy_stage_scheduler.write_scheduler_state"),
                patch("scripts.heavy_stage_scheduler.write_idle_diagnosis", return_value=False),
            ):
                run_scheduler(queue_path=queue)

            queue_payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
            self.assertEqual(queue_payload["jobs"][0]["status"], "completed")

    def test_infer_item_status_requires_full_eeg_imagenet_link_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            expected = root / "data/EEG-ImageNet/train_image_linked.jsonl"
            expected.parent.mkdir(parents=True)
            expected.write_text("{}\n", encoding="utf-8")
            report_root = root / "outputs/datasets"
            report_root.mkdir(parents=True)
            (report_root / "EEG_IMAGENET_IMAGE_LINK_TRAIN_REPORT.md").write_text(
                "- Loader-ready status: `fully image-linked`\n",
                encoding="utf-8",
            )
            (report_root / "EEG_IMAGENET_IMAGE_LINK_VAL_REPORT.md").write_text(
                "- Loader-ready status: `not fully image-linked`\n",
                encoding="utf-8",
            )
            (report_root / "EEG_IMAGENET_IMAGE_LINK_TEST_REPORT.md").write_text(
                "- Loader-ready status: `fully image-linked`\n",
                encoding="utf-8",
            )

            with patch("scripts.heavy_stage_scheduler.PROJECT_ROOT", root):
                status = infer_item_status(
                    QueueItem(
                        id="DATASET_EEG_IMAGENET_IMAGE_LINK",
                        command="python scripts/watch_and_link_eeg_imagenet_images.py",
                        status="waiting",
                        priority=26,
                        expected_output=str(expected),
                        notes="",
                    )
                )

            self.assertEqual(status, "waiting")

    def test_infer_item_status_marks_eeg_imagenet_link_partial_ready_for_exact_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            expected = root / "data/EEG-ImageNet/train_image_exact.jsonl"
            expected.parent.mkdir(parents=True)
            for split in ("train", "val", "test"):
                (expected.parent / f"{split}_image_exact.jsonl").write_text("{}\n", encoding="utf-8")
            report_root = root / "outputs/datasets"
            report_root.mkdir(parents=True)
            for split in ("TRAIN", "VAL", "TEST"):
                (report_root / f"EEG_IMAGENET_IMAGE_LINK_{split}_REPORT.md").write_text(
                    "\n".join(
                        [
                            "- Loader-ready status: `not fully image-linked`",
                            "- Exact-linked subset status: `ready for paired training`",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )

            with patch("scripts.heavy_stage_scheduler.PROJECT_ROOT", root):
                status = infer_item_status(
                    QueueItem(
                        id="DATASET_EEG_IMAGENET_IMAGE_LINK",
                        command="python scripts/watch_and_link_eeg_imagenet_images.py",
                        status="waiting",
                        priority=26,
                        expected_output=str(expected),
                        notes="",
                    )
                )

            self.assertEqual(status, "partial_ready")

    def test_infer_item_status_requires_full_reports_for_paired_ready_watcher(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            expected = root / "data/EEG-ImageNet/train_image_linked.jsonl"
            expected.parent.mkdir(parents=True)
            expected.write_text("{}\n", encoding="utf-8")
            report_root = root / "outputs/datasets"
            report_root.mkdir(parents=True)
            (report_root / "EEG_IMAGENET_IMAGE_LINK_TRAIN_REPORT.md").write_text(
                "- Loader-ready status: `fully image-linked`\n",
                encoding="utf-8",
            )
            (report_root / "EEG_IMAGENET_IMAGE_LINK_VAL_REPORT.md").write_text(
                "- Loader-ready status: `not fully image-linked`\n",
                encoding="utf-8",
            )
            (report_root / "EEG_IMAGENET_IMAGE_LINK_TEST_REPORT.md").write_text(
                "- Loader-ready status: `fully image-linked`\n",
                encoding="utf-8",
            )

            with patch("scripts.heavy_stage_scheduler.PROJECT_ROOT", root):
                status = infer_item_status(
                    QueueItem(
                        id="EEG_IMAGENET_PAIRED_ALIGNMENT_READY_WATCHER",
                        command="python scripts/launch_after_manifest_artifact.py",
                        status="running",
                        priority=37,
                        expected_output=str(expected),
                        notes="",
                    )
                )

            self.assertEqual(status, "running")


if __name__ == "__main__":
    unittest.main()
