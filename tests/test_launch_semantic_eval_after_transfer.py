from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class LaunchSemanticEvalAfterTransferTests(unittest.TestCase):
    def test_build_semantic_eval_command_uses_full_modes_and_transfer_checkpoint(self) -> None:
        from scripts.launch_semantic_eval_after_transfer import build_semantic_eval_command

        command = build_semantic_eval_command(
            checkpoint=Path("outputs/transfer/eeg_imagenet_pretrain_t2t_align/checkpoints/best.pt"),
            output_dir=Path("outputs/final_semantic/eeg_imagenet_transfer_eval"),
        )

        joined = " ".join(command)
        self.assertIn("--eeg_checkpoint outputs/transfer/eeg_imagenet_pretrain_t2t_align/checkpoints/best.pt", joined)
        self.assertIn("--output_dir outputs/final_semantic/eeg_imagenet_transfer_eval", joined)
        self.assertIn("--corruptions clean blur occlusion noise lowres", joined)
        self.assertIn("--modes vision_only real_eeg shuffled_eeg random_eeg eeg_only", joined)
        self.assertIn(
            "python scripts/materialize_final_semantic_report.py --source-dir outputs/final_semantic/eeg_imagenet_transfer_eval --robustness-dir outputs/robustness/eeg_imagenet_transfer_eval --out-dir outputs/final_semantic --primary-summary outputs/final_semantic/A2_SEMANTIC_FUSION_MULTISEED_SUMMARY.md",
            joined,
        )
        self.assertNotIn("--max_samples", joined)

    def test_wait_for_checkpoint_returns_false_when_missing(self) -> None:
        from scripts.launch_semantic_eval_after_transfer import wait_for_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "missing.pt"

            self.assertFalse(wait_for_checkpoint(checkpoint, poll_seconds=0, max_wait_seconds=0))

    def test_default_ready_marker_waits_for_transfer_metrics_not_checkpoint_only(self) -> None:
        import scripts.launch_semantic_eval_after_transfer as launcher

        self.assertEqual(
            launcher.DEFAULT_READY_MARKER,
            Path("outputs/transfer/eeg_imagenet_pretrain_t2t_align/alignment_metrics.json"),
        )


if __name__ == "__main__":
    unittest.main()
