from __future__ import annotations

from pathlib import Path
import unittest

import yaml


class HeavyStageSchedulerLoopTests(unittest.TestCase):
    def test_loop_script_runs_scheduler_with_idle_launch_and_refreshes_reports(self) -> None:
        script = Path("scripts/run_heavy_stage_scheduler_loop.sh")
        self.assertTrue(script.exists(), "scheduler loop script is missing")
        text = script.read_text(encoding="utf-8")

        self.assertIn("python scripts/gpu_monitor.py", text)
        self.assertIn("python scripts/heavy_stage_scheduler.py --launch-when-idle", text)
        self.assertIn("python scripts/materialize_heavy_stage_reports.py --outputs-root outputs", text)
        self.assertIn("python scripts/update_large_data_progress_report.py", text)
        self.assertIn('sleep "${HEAVY_STAGE_SCHEDULER_POLL_SECONDS:-300}"', text)
        self.assertIn("outputs/heavy_stage/heavy_stage_scheduler_loop.log", text)
        self.assertLess(text.index("python scripts/gpu_monitor.py"), text.index("python scripts/heavy_stage_scheduler.py --launch-when-idle"))

    def test_queue_has_scientific_idle_fallback_jobs(self) -> None:
        queue = yaml.safe_load(Path("configs/heavy_stage_queue.yaml").read_text(encoding="utf-8"))
        jobs = {item["id"]: item for item in queue["jobs"]}

        self.assertIn(jobs["FALLBACK_CLIPL_DEGRADED_TEST_CACHE"]["status"], {"queued", "running", "completed", "failed"})
        self.assertIn("openai/clip-vit-large-patch14", jobs["FALLBACK_CLIPL_DEGRADED_TEST_CACHE"]["command"])
        self.assertIn("--require_real_model", jobs["FALLBACK_CLIPL_DEGRADED_TEST_CACHE"]["command"])
        self.assertIn("degraded_test_clipL", jobs["FALLBACK_CLIPL_DEGRADED_TEST_CACHE"]["expected_output"])

        self.assertIn(jobs["FALLBACK_STRONG_DEGRADATION_SEMANTIC_EVAL"]["status"], {"queued", "running", "completed", "failed"})
        strong_command = jobs["FALLBACK_STRONG_DEGRADATION_SEMANTIC_EVAL"]["command"]
        for corruption in ["strong_blur", "strong_noise", "occlusion50", "lowres16", "mixed"]:
            self.assertIn(corruption, strong_command)
        self.assertIn("--require_real_model", strong_command)
        self.assertIn("--day5_csv outputs/final_semantic/strong_degradation_eval/FULL_METRICS.csv", strong_command)

        self.assertIn(jobs["FALLBACK_EEG_IMAGENET_TRANSFER_SEMANTIC_FUSION"]["status"], {"queued", "running", "completed", "failed"})
        self.assertIn("alignment_metrics.json", jobs["FALLBACK_EEG_IMAGENET_TRANSFER_SEMANTIC_FUSION"]["command"])
        self.assertIn("eeg_imagenet_transfer_fusion_classifier", jobs["FALLBACK_EEG_IMAGENET_TRANSFER_SEMANTIC_FUSION"]["expected_output"])

        self.assertEqual(jobs["EEG_IMAGENET_PAIRED_ALIGNMENT_AFTER_IMAGE_LINK"]["status"], "skipped")
        self.assertIn("Superseded", jobs["EEG_IMAGENET_PAIRED_ALIGNMENT_AFTER_IMAGE_LINK"]["notes"])
        self.assertEqual(jobs["EEG_IMAGENET_PAIRED_ALIGNMENT_CACHED_RECOVERY"]["status"], "completed")
        self.assertIn(
            "outputs/transfer/eeg_imagenet_paired_alignment_cached/alignment_metrics.json",
            jobs["EEG_IMAGENET_PAIRED_ALIGNMENT_CACHED_RECOVERY"]["expected_output"],
        )
        paired_command = jobs["EEG_IMAGENET_PAIRED_ALIGNMENT_AFTER_IMAGE_LINK"]["command"]
        self.assertNotIn("while [ ! -f", paired_command)
        self.assertIn("data/EEG-ImageNet/train_image_exact.jsonl", paired_command)
        self.assertNotIn("eeg_pretrain_train.npy", paired_command)
        self.assertIn("scripts/precompute_vision.py", paired_command)
        self.assertIn("configs/transfer/eeg_imagenet_paired_alignment.yaml", paired_command)
        self.assertIn("outputs/transfer/eeg_imagenet_paired_alignment/alignment_metrics.json", jobs["EEG_IMAGENET_PAIRED_ALIGNMENT_AFTER_IMAGE_LINK"]["expected_output"])

    def test_queue_has_remaining_heavy_stage_followups(self) -> None:
        queue = yaml.safe_load(Path("configs/heavy_stage_queue.yaml").read_text(encoding="utf-8"))
        jobs = {item["id"]: item for item in queue["jobs"]}
        runnable_statuses = {"queued", "waiting", "running", "completed"}

        self.assertIn("MASKED_EEG_PRETRAIN_THINGS_M1_DUALBRANCH_EEGCONFORMER", jobs)
        self.assertIn(jobs["MASKED_EEG_PRETRAIN_THINGS_M1_DUALBRANCH_EEGCONFORMER"]["status"], runnable_statuses)
        m1_command = jobs["MASKED_EEG_PRETRAIN_THINGS_M1_DUALBRANCH_EEGCONFORMER"]["command"]
        self.assertIn("scripts/launch_after_artifacts.py", m1_command)
        self.assertIn("things_m0_convtransformer_pretrain_t2t_align/alignment_metrics.json", m1_command)
        self.assertIn("things_m2_tsst_pretrain_t2t_align/alignment_metrics.json", m1_command)
        self.assertIn("masked_eeg_things_eeg2_m1_dualbranch_eegconformer.yaml", m1_command)

        self.assertIn("TRANSFER_THINGS_M1_DUALBRANCH_TO_THOUGHT2TEXT", jobs)
        self.assertIn(jobs["TRANSFER_THINGS_M1_DUALBRANCH_TO_THOUGHT2TEXT"]["status"], runnable_statuses)
        m1_transfer_command = jobs["TRANSFER_THINGS_M1_DUALBRANCH_TO_THOUGHT2TEXT"]["command"]
        self.assertIn("scripts/launch_transfer_after_pretrain_artifact.py", m1_transfer_command)
        self.assertIn("best_masked_eeg.pt", m1_transfer_command)
        self.assertIn("things_m1_dualbranch_pretrain_t2t_align.yaml", m1_transfer_command)

        self.assertIn("ARCH_A4_RAW_SPECTROGRAM_LATE_FUSION_FULL", jobs)
        self.assertIn(jobs["ARCH_A4_RAW_SPECTROGRAM_LATE_FUSION_FULL"]["status"], runnable_statuses)
        self.assertIn("A4_raw_spectrogram_late_fusion_full.yaml", jobs["ARCH_A4_RAW_SPECTROGRAM_LATE_FUSION_FULL"]["command"])

        self.assertIn("H1_P2_HARD_NEGATIVE_ALIGNMENT", jobs)
        self.assertIn(jobs["H1_P2_HARD_NEGATIVE_ALIGNMENT"]["status"], runnable_statuses)
        self.assertIn("H1_P2_hard_negative.yaml", jobs["H1_P2_HARD_NEGATIVE_ALIGNMENT"]["command"])

        self.assertIn("EEG_IMAGENET_EXACT_A4_SCRATCH_FULL", jobs)
        self.assertIn(jobs["EEG_IMAGENET_EXACT_A4_SCRATCH_FULL"]["status"], runnable_statuses)
        self.assertIn("eeg_imagenet_exact_a4_scratch_full.yaml", jobs["EEG_IMAGENET_EXACT_A4_SCRATCH_FULL"]["command"])

        self.assertIn("TRIMODAL_EEG_IMAGENET_EXACT_A2_SCRATCH_FULL", jobs)
        self.assertIn(jobs["TRIMODAL_EEG_IMAGENET_EXACT_A2_SCRATCH_FULL"]["status"], runnable_statuses)
        self.assertIn("eeg_imagenet_exact_a2_scratch_full.yaml", jobs["TRIMODAL_EEG_IMAGENET_EXACT_A2_SCRATCH_FULL"]["command"])

        self.assertIn("CLIP_ADAPTER_SIGLIP_PROTOTYPE_CALIBRATION", jobs)
        self.assertIn(jobs["CLIP_ADAPTER_SIGLIP_PROTOTYPE_CALIBRATION"]["status"], runnable_statuses)
        self.assertIn("siglip", jobs["CLIP_ADAPTER_SIGLIP_PROTOTYPE_CALIBRATION"]["command"].lower())


if __name__ == "__main__":
    unittest.main()
