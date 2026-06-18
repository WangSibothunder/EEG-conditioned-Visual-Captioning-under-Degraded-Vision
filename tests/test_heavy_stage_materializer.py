from __future__ import annotations

import json
import sys
import unittest
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_metrics(path: Path, *, r5: float, class_acc: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {
            "unique_image": {
                "r@1": 0.1,
                "r@5": r5,
                "r@10": 0.6,
                "class_acc": class_acc,
                "mean_rank": 12.0,
            },
            "trial": {"r@5": 0.02},
        },
        "random": {"r@5": 0.015},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_semantic_gap_metrics(path: Path) -> None:
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
                "real_minus_shuffled": "0.1",
                "real_minus_random": "0.2",
                "real_top5_minus_vision": "-0.05",
            }
        )
        writer.writerow(
            {
                "corruption": "blur",
                "real_beats_controls": "false",
                "real_beats_vision": "false",
                "real_minus_vision": "-0.2",
                "real_minus_shuffled": "-0.01",
                "real_minus_random": "0.03",
                "real_top5_minus_vision": "-0.08",
            }
        )


class HeavyStageMaterializerTests(unittest.TestCase):
    def test_collect_alignment_rows_maps_configs_and_best_checkpoint(self) -> None:
        from scripts.materialize_heavy_stage_reports import collect_alignment_rows, select_best_row

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_a = root / "outputs" / "run_a"
            out_b = root / "outputs" / "run_b"
            _write_metrics(out_a / "alignment_metrics.json", r5=0.2, class_acc=0.4)
            _write_metrics(out_b / "alignment_metrics.json", r5=0.35, class_acc=0.3)
            (out_a / "checkpoints").mkdir()
            (out_b / "checkpoints").mkdir()
            (out_b / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            (cfg_dir / "a.yaml").write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "run_a",
                        "model": {"encoder_type": "tiny"},
                        "loss": {"loss_combo": "L1"},
                        "output": {"dir": str(out_a)},
                    }
                ),
                encoding="utf-8",
            )
            (cfg_dir / "b.yaml").write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "run_b",
                        "model": {"encoder_type": "raw_spectrogram_fusion"},
                        "loss": {"loss_combo": "L1+L2+L4"},
                        "output": {"dir": str(out_b)},
                    }
                ),
                encoding="utf-8",
            )

            rows = collect_alignment_rows(root / "outputs", [cfg_dir])
            best = select_best_row(rows)

            self.assertEqual(len(rows), 2)
            self.assertEqual(best["experiment_id"], "run_b")
            self.assertEqual(best["encoder_type"], "raw_spectrogram_fusion")
            self.assertEqual(best["loss_combo"], "L1+L2+L4")
            self.assertEqual(best["checkpoint"], out_b / "checkpoints" / "best.pt")

    def test_collect_alignment_rows_excludes_smoke_and_debug_outputs(self) -> None:
        from scripts.materialize_heavy_stage_reports import collect_alignment_rows, select_best_row

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            smoke_out = root / "outputs" / "cache_loader_align_smoke"
            full_out = root / "outputs" / "full_run"
            _write_metrics(smoke_out / "alignment_metrics.json", r5=0.95, class_acc=0.1)
            _write_metrics(full_out / "alignment_metrics.json", r5=0.3, class_acc=0.4)
            (full_out / "checkpoints").mkdir(parents=True)
            (full_out / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            (cfg_dir / "full.yaml").write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "full_run",
                        "model": {"encoder_type": "convtransformer_base"},
                        "loss": {"loss_combo": "L1"},
                        "output": {"dir": str(full_out)},
                    }
                ),
                encoding="utf-8",
            )

            rows = collect_alignment_rows(root / "outputs", [cfg_dir])
            best = select_best_row(rows)

            self.assertEqual([row["experiment_id"] for row in rows], ["full_run"])
            self.assertEqual(best["experiment_id"], "full_run")

    def test_collect_alignment_rows_uses_train_align_defaults_for_legacy_configs(self) -> None:
        from scripts.materialize_heavy_stage_reports import collect_alignment_rows

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "outputs" / "legacy_day4"
            _write_metrics(out / "alignment_metrics.json", r5=0.33, class_acc=0.29)
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            (cfg_dir / "legacy.yaml").write_text(
                yaml.safe_dump(
                    {
                        "model": {"hidden_dim": 128},
                        "loss": {
                            "use_infonce": True,
                            "use_multi_positive_infonce": True,
                            "use_class_ce": True,
                            "use_similarity_distillation": True,
                            "use_aug_consistency": True,
                            "use_prototype_alignment": True,
                        },
                        "output": {"dir": str(out)},
                    }
                ),
                encoding="utf-8",
            )

            rows = collect_alignment_rows(root / "outputs", [cfg_dir])

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["encoder_type"], "tiny")
            self.assertEqual(rows[0]["loss_combo"], "L1+L2+L4+L5+L6")

    def test_collect_alignment_rows_keeps_named_heavy_architecture_notes_and_skips_root_metrics(self) -> None:
        from scripts.materialize_heavy_stage_reports import collect_alignment_rows

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            root_metric = outputs / "alignment_metrics.json"
            a1 = outputs / "architectures" / "A1_dualbranch_eegconformer_full"
            a2 = outputs / "architectures" / "A2_temporal_spectral_spatial_full"
            a3 = outputs / "architectures" / "A3_subject_adaptive_graph_full"
            _write_metrics(root_metric, r5=0.99, class_acc=0.99)
            _write_metrics(a1 / "alignment_metrics.json", r5=0.24, class_acc=0.25)
            _write_metrics(a2 / "alignment_metrics.json", r5=0.29, class_acc=0.30)
            _write_metrics(a3 / "alignment_metrics.json", r5=0.03, class_acc=0.04)
            cfg_dir = root / "configs" / "heavy_architectures"
            cfg_dir.mkdir(parents=True)
            configs = [
                (
                    "A1.yaml",
                    "A1_dualbranch_eegconformer_full",
                    "dualbranch_eegconformer",
                    a1,
                    "Unit forward smoke passed; this is not a sample-limited smoke.",
                ),
                (
                    "A2.yaml",
                    "A2_temporal_spectral_spatial_full",
                    "temporal_spectral_spatial_transformer",
                    a2,
                    "Heavy-stage A2 full run. Unit forward smoke passed; this is not smoke.",
                ),
                (
                    "A3.yaml",
                    "A3_subject_adaptive_graph_full",
                    "subject_adaptive_graph",
                    a3,
                    "Full A3 run after smoke validation.",
                ),
            ]
            for filename, experiment_id, encoder_type, out_dir, notes in configs:
                (cfg_dir / filename).write_text(
                    yaml.safe_dump(
                        {
                            "experiment_id": experiment_id,
                            "model": {"encoder_type": encoder_type},
                            "loss": {"loss_combo": "L1+L2+L4"},
                            "output": {"dir": str(out_dir)},
                            "notes": notes,
                        }
                    ),
                    encoding="utf-8",
                )

            rows = collect_alignment_rows(outputs, [cfg_dir])

            by_id = {row["experiment_id"]: row for row in rows}
            self.assertNotIn("outputs", by_id)
            self.assertEqual(set(by_id), {item[1] for item in configs})
            self.assertEqual(by_id["A1_dualbranch_eegconformer_full"]["param_count"], "~10M-30M target")
            self.assertEqual(by_id["A2_temporal_spectral_spatial_full"]["param_count"], "temporal+spectral+spatial transformer")
            self.assertEqual(by_id["A3_subject_adaptive_graph_full"]["param_count"], "subject-adaptive graph encoder")

    def test_architecture_report_always_lists_named_heavy_architecture_runs(self) -> None:
        from scripts.materialize_heavy_stage_reports import collect_alignment_rows, write_reports

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            strong = outputs / "many_better_runs" / "best"
            _write_metrics(strong / "alignment_metrics.json", r5=0.8, class_acc=0.8)
            a3 = outputs / "architectures" / "A3_subject_adaptive_graph_full"
            _write_metrics(a3 / "alignment_metrics.json", r5=0.03, class_acc=0.04)
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            (cfg_dir / "strong.yaml").write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "strong_run",
                        "model": {"encoder_type": "raw_spectrogram_fusion"},
                        "loss": {"loss_combo": "L1"},
                        "output": {"dir": str(strong)},
                    }
                ),
                encoding="utf-8",
            )
            (cfg_dir / "a3.yaml").write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "A3_subject_adaptive_graph_full",
                        "model": {"encoder_type": "subject_adaptive_graph"},
                        "loss": {"loss_combo": "L1+L2+L4+L7"},
                        "output": {"dir": str(a3)},
                        "notes": "Full A3 run after smoke validation.",
                    }
                ),
                encoding="utf-8",
            )
            rows = collect_alignment_rows(outputs, [cfg_dir])
            write_reports(rows, outputs)

            report = (outputs / "architectures" / "ARCHITECTURE_SEARCH_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("Named Heavy Architectures", report)
            self.assertIn("A3_subject_adaptive_graph_full", report)
            self.assertIn("subject_adaptive_graph", report)

    def test_write_reports_creates_required_heavy_stage_artifacts(self) -> None:
        from scripts.materialize_heavy_stage_reports import collect_alignment_rows, write_reports

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "outputs" / "transfer" / "masked_pretrain_t2t_align"
            _write_metrics(out / "alignment_metrics.json", r5=0.25, class_acc=0.2)
            (out / "checkpoints").mkdir(parents=True)
            (out / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            (cfg_dir / "transfer.yaml").write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "masked_pretrain_t2t_align",
                        "model": {"encoder_type": "masked_pretrained"},
                        "loss": {"loss_combo": "L1+L2+L4+L5"},
                        "output": {"dir": str(out)},
                    }
                ),
                encoding="utf-8",
            )

            rows = collect_alignment_rows(root / "outputs", [cfg_dir])
            write_reports(rows, root / "outputs")

            self.assertTrue((root / "outputs" / "architectures" / "ARCHITECTURE_SEARCH_REPORT.md").exists())
            self.assertTrue((root / "outputs" / "architectures" / "checkpoints" / "best_encoder.pt").exists())
            self.assertTrue((root / "outputs" / "transfer" / "TRANSFER_TO_THOUGHT2TEXT_REPORT.md").exists())
            self.assertTrue((root / "outputs" / "transfer" / "best_transfer_encoder.pt").exists())
            self.assertTrue((root / "outputs" / "heavy_stage" / "BASELINES.md").exists())
            self.assertTrue((root / "outputs" / "heavy_stage" / "MASTER_REPORT.md").exists())
            report = (root / "outputs" / "transfer" / "TRANSFER_TO_THOUGHT2TEXT_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("masked_pretrain_t2t_align", report)
            master = (root / "outputs" / "heavy_stage" / "MASTER_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("Did larger datasets become loader-ready?", master)
            self.assertIn("Current status", master)
            baselines = (root / "outputs" / "heavy_stage" / "BASELINES.md").read_text(encoding="utf-8")
            self.assertIn("Current highest indexed EEG->image R@5", baselines)
            self.assertIn("0.2500", baselines)

    def test_master_report_names_current_pretraining_state_without_stale_eeg_imagenet_running_claim(self) -> None:
        from scripts.materialize_heavy_stage_reports import collect_alignment_rows, write_reports

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "outputs" / "transfer" / "masked_pretrain_t2t_align"
            _write_metrics(out / "alignment_metrics.json", r5=0.25, class_acc=0.2)
            (out / "checkpoints").mkdir(parents=True)
            (out / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            (cfg_dir / "transfer.yaml").write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "masked_pretrain_t2t_align",
                        "model": {"encoder_type": "masked_pretrained"},
                        "loss": {"loss_combo": "L1+L2+L4+L5"},
                        "output": {"dir": str(out)},
                    }
                ),
                encoding="utf-8",
            )

            rows = collect_alignment_rows(root / "outputs", [cfg_dir])
            write_reports(rows, root / "outputs")

            master = (root / "outputs" / "heavy_stage" / "MASTER_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("THINGS-EEG2 raw-window masked EEG pretraining", master)
            self.assertIn("Thought2Text transfer final metrics are not available yet", master)
            self.assertIn("EEG-ImageNet masked pretraining is complete", master)
            self.assertIn("EEG-ImageNet paired image+EEG exact-linked training is not complete in this report", master)
            self.assertIn("ImageNet CLS-LOC", master)
            self.assertNotIn("active high-utilization EEG-ImageNet pretraining", master)
            self.assertNotIn("Finish EEG-ImageNet masked pretraining", master)

    def test_master_report_does_not_call_completed_things_transfer_running(self) -> None:
        from scripts.materialize_heavy_stage_reports import collect_alignment_rows, write_reports

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "outputs" / "transfer" / "things_raw_pretrain_t2t_align"
            _write_metrics(out / "alignment_metrics.json", r5=0.1712, class_acc=0.1442)
            (out / "checkpoints").mkdir(parents=True)
            (out / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            (cfg_dir / "things_transfer.yaml").write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "things_raw_pretrain_t2t_align",
                        "model": {"encoder_type": "masked_pretrained"},
                        "loss": {"loss_combo": "L1+L2+L4+L5"},
                        "output": {"dir": str(out)},
                    }
                ),
                encoding="utf-8",
            )

            rows = collect_alignment_rows(root / "outputs", [cfg_dir])
            write_reports(rows, root / "outputs")

            master = (root / "outputs" / "heavy_stage" / "MASTER_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("THINGS raw-window transfer completed", master)
            self.assertNotIn("TRANSFER_THINGS_RAW_PRETRAIN_TO_THOUGHT2TEXT` has launched and is still running", master)
            self.assertNotIn("final `alignment_metrics.json` is not available yet", master)

    def test_master_report_names_strong_degradation_context_when_available(self) -> None:
        from scripts.materialize_heavy_stage_reports import collect_alignment_rows, write_reports

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "outputs" / "transfer" / "eeg_imagenet_pretrain_t2t_align"
            _write_metrics(out / "alignment_metrics.json", r5=0.018, class_acc=0.018)
            (out / "checkpoints").mkdir(parents=True)
            (out / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
            strong_gap = root / "outputs" / "final_semantic" / "strong_degradation_eval" / "SEMANTIC_GAP_METRICS.csv"
            strong_gap.parent.mkdir(parents=True, exist_ok=True)
            with strong_gap.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["corruption", "real_beats_controls", "real_beats_vision"],
                )
                writer.writeheader()
                writer.writerow({"corruption": "lowres16", "real_beats_controls": "true", "real_beats_vision": "true"})
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            (cfg_dir / "transfer.yaml").write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "eeg_imagenet_pretrain_t2t_align",
                        "model": {"encoder_type": "masked_pretrained"},
                        "loss": {"loss_combo": "L1+L2+L4+L5"},
                        "output": {"dir": str(out)},
                    }
                ),
                encoding="utf-8",
            )

            rows = collect_alignment_rows(root / "outputs", [cfg_dir])
            write_reports(rows, root / "outputs")

            master = (root / "outputs" / "heavy_stage" / "MASTER_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("strong degradation semantic evaluation completed", master)
            self.assertNotIn("finish the active THINGS raw-window transfer", master)

    def test_semantic_status_prefers_strong_degradation_when_present(self) -> None:
        from scripts.materialize_heavy_stage_reports import _semantic_control_status

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            transfer_gap = root / "final_semantic" / "eeg_imagenet_transfer_eval" / "SEMANTIC_GAP_METRICS.csv"
            _write_semantic_gap_metrics(transfer_gap)
            strong_gap = root / "final_semantic" / "strong_degradation_eval" / "SEMANTIC_GAP_METRICS.csv"
            strong_gap.parent.mkdir(parents=True, exist_ok=True)
            with strong_gap.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["corruption", "real_beats_controls", "real_beats_vision"],
                )
                writer.writeheader()
                for corruption, vision in [("lowres16", "true"), ("mixed", "true"), ("strong_blur", "false")]:
                    writer.writerow(
                        {
                            "corruption": corruption,
                            "real_beats_controls": "true",
                            "real_beats_vision": vision,
                        }
                    )

            status = _semantic_control_status(root)

            self.assertEqual(status["source"], strong_gap)
            self.assertEqual(status["control_wins"], 3)
            self.assertEqual(status["vision_wins"], 2)
            self.assertIn("strong degradation", status["control_answer"])

    def test_semantic_status_prefers_a2_classifier_eval_over_legacy_strong_degradation(self) -> None:
        from scripts.materialize_heavy_stage_reports import _semantic_control_status

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            legacy = root / "final_semantic" / "strong_degradation_eval" / "SEMANTIC_GAP_METRICS.csv"
            _write_semantic_gap_metrics(legacy)
            a2 = (
                root
                / "final_semantic"
                / "semantic_fusion_A2_temporal_spectral_spatial_full_strong_eval"
                / "SEMANTIC_GAP_METRICS.csv"
            )
            a2.parent.mkdir(parents=True, exist_ok=True)
            with a2.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["corruption", "real_beats_controls", "real_beats_vision"],
                )
                writer.writeheader()
                for corruption in ["clean", "lowres16", "mixed"]:
                    writer.writerow(
                        {
                            "corruption": corruption,
                            "real_beats_controls": "true",
                            "real_beats_vision": "true",
                        }
                    )

            status = _semantic_control_status(root)

            self.assertEqual(status["source"], a2)
            self.assertEqual(status["control_wins"], 3)
            self.assertEqual(status["vision_wins"], 3)
            self.assertIn("A2 semantic fusion", status["context"])

    def test_semantic_status_uses_a2_multiseed_summary_when_available(self) -> None:
        from scripts.materialize_heavy_stage_reports import _semantic_control_status

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            a2 = (
                root
                / "final_semantic"
                / "semantic_fusion_A2_temporal_spectral_spatial_full_strong_eval"
                / "SEMANTIC_GAP_METRICS.csv"
            )
            a2.parent.mkdir(parents=True, exist_ok=True)
            with a2.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["corruption", "real_beats_controls", "real_beats_vision"],
                )
                writer.writeheader()
                writer.writerow({"corruption": "clean", "real_beats_controls": "true", "real_beats_vision": "true"})
            multiseed = root / "final_semantic" / "A2_SEMANTIC_FUSION_MULTISEED_SUMMARY.csv"
            multiseed.parent.mkdir(parents=True, exist_ok=True)
            with multiseed.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["seed", "corruption", "real_beats_controls", "real_beats_vision"],
                )
                writer.writeheader()
                for seed in ["seed42", "seed123", "seed2025"]:
                    for corruption in ["clean", "mixed"]:
                        writer.writerow(
                            {
                                "seed": seed,
                                "corruption": corruption,
                                "real_beats_controls": "True",
                                "real_beats_vision": "True",
                            }
                        )

            status = _semantic_control_status(root)

            self.assertEqual(status["source"], multiseed)
            self.assertEqual(status["total"], 6)
            self.assertEqual(status["control_wins"], 6)
            self.assertEqual(status["vision_wins"], 6)
            self.assertIn("multi-seed", status["context"])
            self.assertIn("real EEG beats vision-only in `6/6`", status["vision_answer"])
            self.assertNotIn("Limited:", status["vision_answer"])

    def test_reports_state_completed_eeg_imagenet_transfer_as_negative_when_metrics_exist(self) -> None:
        from scripts.materialize_heavy_stage_reports import collect_alignment_rows, write_reports

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            transfer = root / "outputs" / "transfer" / "eeg_imagenet_pretrain_t2t_align"
            _write_metrics(transfer / "alignment_metrics.json", r5=0.018, class_acc=0.018)
            (transfer / "checkpoints").mkdir(parents=True)
            (transfer / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            (cfg_dir / "eeg_imagenet_transfer.yaml").write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "eeg_imagenet_pretrain_t2t_align",
                        "model": {
                            "encoder_type": "masked_pretrained",
                            "pretrained_eeg_checkpoint": "outputs/pretrain/masked_eeg_eeg_imagenet_dualbranch_heavy/checkpoints/best_masked_eeg.pt",
                        },
                        "loss": {"loss_combo": "L1+L2+L4+L5"},
                        "output": {"dir": str(transfer)},
                    }
                ),
                encoding="utf-8",
            )

            rows = collect_alignment_rows(root / "outputs", [cfg_dir])
            write_reports(rows, root / "outputs")

            transfer_report = (root / "outputs" / "transfer" / "TRANSFER_TO_THOUGHT2TEXT_REPORT.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("EEG-ImageNet transfer metrics are present", transfer_report)
            self.assertIn("did not improve over historical Day4 R@5", transfer_report)
            master = (root / "outputs" / "heavy_stage" / "MASTER_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("EEG-ImageNet transfer recovery completed but did not improve", master)
            self.assertNotIn("downstream transfer recovery is still running", master)

    def test_master_report_does_not_treat_completed_exact_eeg_imagenet_runs_as_active_next_work(self) -> None:
        from scripts.materialize_heavy_stage_reports import collect_alignment_rows, write_reports

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            transfer = root / "outputs" / "transfer" / "eeg_imagenet_pretrain_t2t_align"
            paired = root / "outputs" / "transfer" / "eeg_imagenet_paired_alignment_cached"
            _write_metrics(transfer / "alignment_metrics.json", r5=0.018, class_acc=0.018)
            _write_metrics(paired / "alignment_metrics.json", r5=0.002, class_acc=0.0156)
            for out in [transfer, paired]:
                (out / "checkpoints").mkdir(parents=True)
                (out / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
            exact_tri = root / "outputs" / "trimodal" / "eeg_imagenet_exact_full"
            exact_tri.mkdir(parents=True)
            (exact_tri / "TRIMODAL_FULL_REPORT.md").write_text(
                "EEG->image R@5 `0.0020`; EEG->text R@5 `0.0020`; class_acc `0.0169`.",
                encoding="utf-8",
            )
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            (cfg_dir / "transfer.yaml").write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "eeg_imagenet_pretrain_t2t_align",
                        "model": {"encoder_type": "masked_pretrained"},
                        "loss": {"loss_combo": "L1+L2+L4+L5"},
                        "output": {"dir": str(transfer)},
                    }
                ),
                encoding="utf-8",
            )
            (cfg_dir / "paired.yaml").write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "eeg_imagenet_paired_alignment_cached",
                        "model": {"encoder_type": "masked_pretrained"},
                        "loss": {"loss_combo": "L1+L2+L4+L5"},
                        "output": {"dir": str(paired)},
                    }
                ),
                encoding="utf-8",
            )

            rows = collect_alignment_rows(root / "outputs", [cfg_dir])
            write_reports(rows, root / "outputs")

            master = (root / "outputs" / "heavy_stage" / "MASTER_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("exact-linked EEG-ImageNet paired alignment and tri-modal results as negative controls", master)
            self.assertNotIn("finish the active exact-linked EEG-ImageNet paired alignment", master)
            self.assertNotIn("finish the active large-dataset tri-modal run", master)
            self.assertNotIn("active alignment", master)

    def test_master_report_distinguishes_completed_a4_exact_from_running_a2_exact_followup(self) -> None:
        from scripts.materialize_heavy_stage_reports import collect_alignment_rows, write_reports

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            a4 = root / "outputs" / "transfer" / "eeg_imagenet_exact_a4_scratch_full"
            _write_metrics(a4 / "alignment_metrics.json", r5=0.0036, class_acc=0.0215)
            (a4 / "checkpoints").mkdir(parents=True)
            (a4 / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
            paired = root / "outputs" / "transfer" / "eeg_imagenet_paired_alignment_cached"
            m0_transfer = root / "outputs" / "transfer" / "things_m0_convtransformer_pretrain_t2t_align"
            m1_transfer = root / "outputs" / "transfer" / "things_m1_dualbranch_pretrain_t2t_align"
            m2_transfer = root / "outputs" / "transfer" / "things_m2_tsst_pretrain_t2t_align"
            for out, r5, acc in [
                (paired, 0.002, 0.0156),
                (m0_transfer, 0.1742, 0.1335),
                (m1_transfer, 0.1201, 0.0905),
                (m2_transfer, 0.1742, 0.1400),
            ]:
                _write_metrics(out / "alignment_metrics.json", r5=r5, class_acc=acc)
                (out / "checkpoints").mkdir(parents=True)
                (out / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
            exact_tri = root / "outputs" / "trimodal" / "eeg_imagenet_exact_full"
            exact_tri.mkdir(parents=True)
            (exact_tri / "TRIMODAL_FULL_REPORT.md").write_text("completed negative exact tri-modal", encoding="utf-8")
            a2 = root / "outputs" / "trimodal" / "eeg_imagenet_exact_a2_scratch_full"
            a2.mkdir(parents=True)
            (a2 / "train_log.jsonl").write_text('{"epoch": 24, "total": 13.8}\n', encoding="utf-8")
            siglip_cache = root / "data" / "thought2text" / "cache"
            siglip_cache.mkdir(parents=True)
            (siglip_cache / "siglip_val.npy").write_bytes(b"cache")

            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            for filename, experiment_id, out_dir, encoder_type in [
                ("paired.yaml", "eeg_imagenet_paired_alignment_cached", paired, "masked_pretrained"),
                ("m0.yaml", "things_m0_convtransformer_pretrain_t2t_align", m0_transfer, "convtransformer_base"),
                ("m1.yaml", "things_m1_dualbranch_pretrain_t2t_align", m1_transfer, "dualbranch_eegconformer"),
                ("m2.yaml", "things_m2_tsst_pretrain_t2t_align", m2_transfer, "temporal_spectral_spatial_transformer"),
                ("a4_exact.yaml", "eeg_imagenet_exact_a4_scratch_full", a4, "raw_spectrogram_fusion"),
            ]:
                (cfg_dir / filename).write_text(
                    yaml.safe_dump(
                        {
                            "experiment_id": experiment_id,
                            "model": {"encoder_type": encoder_type},
                            "loss": {"loss_combo": "L1+L2+L3+L4+L5"},
                            "output": {"dir": str(out_dir)},
                        }
                    ),
                    encoding="utf-8",
                )

            rows = collect_alignment_rows(root / "outputs", [cfg_dir])
            write_reports(rows, root / "outputs")

            master = (root / "outputs" / "heavy_stage" / "MASTER_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("EEG-ImageNet exact A4 scratch alignment completed as a negative result", master)
            self.assertIn("exact A2 scratch tri-modal remains running", master)
            self.assertIn("let exact A2 scratch tri-modal finish", master)
            self.assertIn("SigLIP prototype/calibration cache exists", master)
            self.assertNotIn("then run SigLIP calibration", master)
            self.assertNotIn("continue queued EEG-ImageNet exact-linked A4/A2 follow-ups", master)
            self.assertNotIn("queued EEG-ImageNet exact-linked A4/A2 follow-ups and SigLIP calibration are the next useful jobs", master)
            self.assertNotIn("continue queued exact-linked EEG-ImageNet follow-ups", master)

    def test_master_report_marks_stage_complete_when_exact_a2_scratch_outputs_exist(self) -> None:
        from scripts.materialize_heavy_stage_reports import collect_alignment_rows, write_reports

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            paired = root / "outputs" / "transfer" / "eeg_imagenet_paired_alignment_cached"
            a4 = root / "outputs" / "transfer" / "eeg_imagenet_exact_a4_scratch_full"
            _write_metrics(paired / "alignment_metrics.json", r5=0.002, class_acc=0.0156)
            _write_metrics(a4 / "alignment_metrics.json", r5=0.0036, class_acc=0.0215)
            for out in [paired, a4]:
                (out / "checkpoints").mkdir(parents=True)
                (out / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
            exact_tri = root / "outputs" / "trimodal" / "eeg_imagenet_exact_full"
            exact_tri.mkdir(parents=True)
            (exact_tri / "TRIMODAL_FULL_REPORT.md").write_text("completed negative exact tri-modal", encoding="utf-8")
            a2 = root / "outputs" / "trimodal" / "eeg_imagenet_exact_a2_scratch_full"
            a2.mkdir(parents=True)
            (a2 / "TRIMODAL_FULL_REPORT.md").write_text("# Tri-Modal Alignment Status\n", encoding="utf-8")
            (a2 / "trimodal_metrics.json").write_text(
                json.dumps(
                    {
                        "image": {"r@1": 0.01, "r@5": 0.04, "r@10": 0.08, "class_acc": 0.22},
                        "text": {"r@1": 0.008, "r@5": 0.041, "r@10": 0.081, "class_acc": 0.22},
                    }
                ),
                encoding="utf-8",
            )
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            for filename, experiment_id, out_dir, encoder_type in [
                ("paired.yaml", "eeg_imagenet_paired_alignment_cached", paired, "masked_pretrained"),
                ("a4_exact.yaml", "eeg_imagenet_exact_a4_scratch_full", a4, "raw_spectrogram_fusion"),
            ]:
                (cfg_dir / filename).write_text(
                    yaml.safe_dump(
                        {
                            "experiment_id": experiment_id,
                            "model": {"encoder_type": encoder_type},
                            "loss": {"loss_combo": "L1+L2+L4"},
                            "output": {"dir": str(out_dir)},
                        }
                    ),
                    encoding="utf-8",
                )

            rows = collect_alignment_rows(root / "outputs", [cfg_dir])
            write_reports(rows, root / "outputs")

            master = (root / "outputs" / "heavy_stage" / "MASTER_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("Current status: `complete_with_mixed_results`", master)
            self.assertIn("EEG-ImageNet exact A2 scratch tri-modal completed", master)
            self.assertIn("not fully supported", master)
            self.assertNotIn("Current status: `in progress`", master)
            self.assertNotIn("Do not claim the final heavy-stage scientific target yet", master)

    def test_master_report_names_m2_only_after_m0_transfer_is_complete(self) -> None:
        from scripts.materialize_heavy_stage_reports import collect_alignment_rows, write_reports

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            eeg_transfer = root / "outputs" / "transfer" / "eeg_imagenet_pretrain_t2t_align"
            paired = root / "outputs" / "transfer" / "eeg_imagenet_paired_alignment_cached"
            m0_transfer = root / "outputs" / "transfer" / "things_m0_convtransformer_pretrain_t2t_align"
            _write_metrics(eeg_transfer / "alignment_metrics.json", r5=0.018, class_acc=0.018)
            _write_metrics(paired / "alignment_metrics.json", r5=0.002, class_acc=0.0156)
            _write_metrics(m0_transfer / "alignment_metrics.json", r5=0.1742, class_acc=0.1335)
            for out in [eeg_transfer, paired, m0_transfer]:
                (out / "checkpoints").mkdir(parents=True)
                (out / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
            exact_tri = root / "outputs" / "trimodal" / "eeg_imagenet_exact_full"
            exact_tri.mkdir(parents=True)
            (exact_tri / "TRIMODAL_FULL_REPORT.md").write_text("completed negative exact tri-modal", encoding="utf-8")

            m2 = root / "outputs" / "pretrain" / "masked_eeg_things_eeg2_m2_temporal_spectral_spatial"
            m2.mkdir(parents=True)
            (m2 / "train.stdout.log").write_text(
                '{"epoch": 26, "train_loss": 1.0, "val_loss": 2.0, "gpu_mem_peak_mb": 21062.8}\n',
                encoding="utf-8",
            )
            (root / "outputs" / "transfer" / "things_m2_tsst_pretrain_t2t_align").mkdir(parents=True)

            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            for filename, experiment_id, out_dir in [
                ("eeg_transfer.yaml", "eeg_imagenet_pretrain_t2t_align", eeg_transfer),
                ("paired.yaml", "eeg_imagenet_paired_alignment_cached", paired),
                ("m0.yaml", "things_m0_convtransformer_pretrain_t2t_align", m0_transfer),
            ]:
                (cfg_dir / filename).write_text(
                    yaml.safe_dump(
                        {
                            "experiment_id": experiment_id,
                            "model": {"encoder_type": "masked_pretrained"},
                            "loss": {"loss_combo": "L1+L2+L4+L5"},
                            "output": {"dir": str(out_dir)},
                        }
                    ),
                    encoding="utf-8",
                )

            rows = collect_alignment_rows(root / "outputs", [cfg_dir])
            write_reports(rows, root / "outputs")

            master = (root / "outputs" / "heavy_stage" / "MASTER_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("continue the M2 THINGS pretraining-to-transfer sequence", master)
            self.assertNotIn("active THINGS M0/M2 masked EEG pretraining", master)
            self.assertNotIn("continue the M0/M2 THINGS pretraining-to-transfer sequence", master)

    def test_master_report_names_active_m1_after_m0_m2_transfers_complete(self) -> None:
        from scripts.materialize_heavy_stage_reports import collect_alignment_rows, write_reports

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            eeg_transfer = root / "outputs" / "transfer" / "eeg_imagenet_pretrain_t2t_align"
            paired = root / "outputs" / "transfer" / "eeg_imagenet_paired_alignment_cached"
            m0_transfer = root / "outputs" / "transfer" / "things_m0_convtransformer_pretrain_t2t_align"
            m2_transfer = root / "outputs" / "transfer" / "things_m2_tsst_pretrain_t2t_align"
            for out, r5, acc in [
                (eeg_transfer, 0.018, 0.018),
                (paired, 0.002, 0.0156),
                (m0_transfer, 0.1742, 0.1335),
                (m2_transfer, 0.1742, 0.1400),
            ]:
                _write_metrics(out / "alignment_metrics.json", r5=r5, class_acc=acc)
                (out / "checkpoints").mkdir(parents=True)
                (out / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
            exact_tri = root / "outputs" / "trimodal" / "eeg_imagenet_exact_full"
            exact_tri.mkdir(parents=True)
            (exact_tri / "TRIMODAL_FULL_REPORT.md").write_text("completed negative exact tri-modal", encoding="utf-8")
            m1 = root / "outputs" / "pretrain" / "masked_eeg_things_eeg2_m1_dualbranch_eegconformer"
            m1.mkdir(parents=True)
            (m1 / "train.log").write_text('{"epoch": 12, "step": 2220}\n', encoding="utf-8")

            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            for filename, experiment_id, out_dir in [
                ("eeg_transfer.yaml", "eeg_imagenet_pretrain_t2t_align", eeg_transfer),
                ("paired.yaml", "eeg_imagenet_paired_alignment_cached", paired),
                ("m0.yaml", "things_m0_convtransformer_pretrain_t2t_align", m0_transfer),
                ("m2.yaml", "things_m2_tsst_pretrain_t2t_align", m2_transfer),
            ]:
                (cfg_dir / filename).write_text(
                    yaml.safe_dump(
                        {
                            "experiment_id": experiment_id,
                            "model": {"encoder_type": "masked_pretrained"},
                            "loss": {"loss_combo": "L1+L2+L4+L5"},
                            "output": {"dir": str(out_dir)},
                        }
                    ),
                    encoding="utf-8",
                )

            rows = collect_alignment_rows(root / "outputs", [cfg_dir])
            write_reports(rows, root / "outputs")

            master = (root / "outputs" / "heavy_stage" / "MASTER_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("active M1 THINGS masked EEG pretraining", master)
            self.assertIn("M1 Thought2Text transfer watcher is waiting for the M1 report/checkpoint", master)
            self.assertNotIn("launch M1 Thought2Text transfer after the M1 report/checkpoint", master)
            self.assertNotIn("queued M1/A4/H1/SigLIP", master)

    def test_master_report_states_completed_transfer_semantic_eval_as_limited_when_gap_metrics_exist(self) -> None:
        from scripts.materialize_heavy_stage_reports import collect_alignment_rows, write_reports

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            transfer = root / "outputs" / "transfer" / "eeg_imagenet_pretrain_t2t_align"
            _write_metrics(transfer / "alignment_metrics.json", r5=0.018, class_acc=0.018)
            (transfer / "checkpoints").mkdir(parents=True)
            (transfer / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
            _write_semantic_gap_metrics(
                root
                / "outputs"
                / "final_semantic"
                / "eeg_imagenet_transfer_eval"
                / "SEMANTIC_GAP_METRICS.csv"
            )
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            (cfg_dir / "eeg_imagenet_transfer.yaml").write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "eeg_imagenet_pretrain_t2t_align",
                        "model": {
                            "encoder_type": "masked_pretrained",
                            "pretrained_eeg_checkpoint": "outputs/pretrain/masked_eeg_eeg_imagenet_dualbranch_heavy/checkpoints/best_masked_eeg.pt",
                        },
                        "loss": {"loss_combo": "L1+L2+L4+L5"},
                        "output": {"dir": str(transfer)},
                    }
                ),
                encoding="utf-8",
            )

            rows = collect_alignment_rows(root / "outputs", [cfg_dir])
            write_reports(rows, root / "outputs")

            master = (root / "outputs" / "heavy_stage" / "MASTER_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("EEG-ImageNet transfer semantic evaluation completed", master)
            self.assertIn("1/2", master)
            self.assertIn("limited", master.lower())
            self.assertNotIn("still needs re-run after EEG-ImageNet transfer", master)
            self.assertNotIn("running or pending final `FULL_METRICS.csv`", master)


if __name__ == "__main__":
    unittest.main()
