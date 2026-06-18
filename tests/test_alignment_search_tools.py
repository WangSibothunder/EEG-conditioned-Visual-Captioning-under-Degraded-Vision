from __future__ import annotations

import tempfile
import subprocess
import sys
import unittest
from pathlib import Path

from scripts.generate_alignment_experiments import generate_experiment_configs
from scripts.gpu_queue import _count_top_level_alignment_processes, allowed_concurrency
from scripts.launch_alignment_sweep import (
    EXPERIMENT_BOARD_COLUMNS,
    available_internal_launch_slots,
    build_initial_board_rows,
    estimate_param_count,
    ordered_config_paths,
)
from src.utils.config import load_config


class AlignmentSearchToolTests(unittest.TestCase):
    def test_generator_writes_required_day4_search_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "configs"
            plan_path = root / "EXPERIMENT_PLAN.md"

            experiments = generate_experiment_configs(out_dir=out_dir, plan_path=plan_path)

            experiment_ids = {experiment["experiment_id"] for experiment in experiments}
            for required in ["T0", "T1", "T2", "T3", "T4", "S1", "S2", "S3", "S4", "S5", "X1", "X2", "X3", "G1", "G2"]:
                self.assertIn(required, experiment_ids)

            encoder_types = {experiment["model"]["encoder_type"] for experiment in experiments}
            self.assertTrue(
                {"tiny", "eegnet", "multiscale_tcn", "convtransformer_base", "convtransformer_strong", "subject_adaptive"} <= encoder_types
            )
            self.assertTrue(plan_path.exists())

            config_path = out_dir / "S5.yaml"
            self.assertTrue(config_path.exists())
            config = load_config(config_path)
            self.assertEqual(config["experiment_id"], "S5")
            self.assertEqual(config["model"]["encoder_type"], "convtransformer_base")
            self.assertTrue(config["loss"]["use_supcon"])
            self.assertTrue(config["loss"]["use_aug_consistency"])
            self.assertEqual(config["output"]["dir"], "outputs/day4_search/S5")

    def test_launcher_uses_generated_candidate_order_not_filename_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "configs"
            generate_experiment_configs(out_dir=out_dir, plan_path=root / "plan.md")

            ordered = [path.stem for path in ordered_config_paths(out_dir)]

            self.assertEqual(ordered[:5], ["T0", "T1", "T2", "T3", "T4"])
            self.assertLess(ordered.index("S1"), ordered.index("G1"))

    def test_gpu_policy_matches_no_idle_alignment_strategy(self) -> None:
        self.assertEqual(allowed_concurrency(memory_used_gb=5.0, gpu_util=30, max_concurrent=4), 4)
        self.assertEqual(allowed_concurrency(memory_used_gb=5.0, gpu_util=10, max_concurrent=8), 8)
        self.assertEqual(allowed_concurrency(memory_used_gb=15.0, gpu_util=50, max_concurrent=4), 4)
        self.assertEqual(allowed_concurrency(memory_used_gb=15.0, gpu_util=50, max_concurrent=8), 6)
        self.assertEqual(allowed_concurrency(memory_used_gb=25.0, gpu_util=80, max_concurrent=4), 1)
        self.assertEqual(allowed_concurrency(memory_used_gb=2.0, gpu_util=10, max_concurrent=1), 1)

    def test_process_counter_ignores_train_align_worker_children(self) -> None:
        rows = [
            (100, 1, "bash scripts/run_day4_alignment_bcd.sh"),
            (101, 100, "python -m src.train.train_align --config c.yaml"),
            (102, 101, "python -m src.train.train_align --config c.yaml"),
            (103, 101, "python -m src.train.train_align --config c.yaml"),
            (201, 1, "python -m src.train.train_align --config other.yaml"),
        ]

        self.assertEqual(_count_top_level_alignment_processes(rows), 2)

    def test_launcher_accounts_for_external_alignment_jobs(self) -> None:
        self.assertEqual(available_internal_launch_slots(allowed_total=3, internal_running=1, external_running=1), 1)
        self.assertEqual(available_internal_launch_slots(allowed_total=2, internal_running=1, external_running=2), 0)

    def test_initial_board_rows_record_queued_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "configs"
            out_root = root / "outputs"
            generate_experiment_configs(out_dir=config_dir, plan_path=root / "plan.md")

            rows = build_initial_board_rows(ordered_config_paths(config_dir), out_root)

            self.assertEqual(len(rows), 15)
            self.assertEqual(rows["T0"].status, "queued")
            self.assertEqual(rows["G2"].status, "queued")

    def test_experiment_board_columns_match_goal_contract(self) -> None:
        self.assertEqual(
            EXPERIMENT_BOARD_COLUMNS,
            [
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

    def test_param_count_initializes_lazy_eegnet_parameters(self) -> None:
        config = {
            "data": {"train_manifest": "data/thought2text/train.jsonl"},
            "model": {
                "encoder_type": "eegnet",
                "eeg_channels": 64,
                "eeg_time_steps": 250,
                "eeg_embed_dim": 512,
                "clip_embed_dim": 512,
                "hidden_dim": 128,
                "transformer_layers": 2,
                "dropout": 0.2,
            },
            "loss": {"use_cls": False},
        }

        self.assertGreater(estimate_param_count(config), 0)

    def test_param_count_supports_subject_adaptive_encoder(self) -> None:
        config = {
            "data": {"train_manifest": "data/thought2text/train.jsonl"},
            "model": {
                "encoder_type": "subject_adaptive",
                "eeg_channels": 64,
                "eeg_time_steps": 250,
                "eeg_embed_dim": 512,
                "clip_embed_dim": 512,
                "hidden_dim": 128,
                "transformer_layers": 2,
                "dropout": 0.2,
                "num_subjects": 8,
            },
            "loss": {"use_cls": False},
        }

        self.assertGreater(estimate_param_count(config), 0)

    def test_day4_best_multiseed_cli_runs_without_pythonpath(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/run_day4_best_multiseed.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
