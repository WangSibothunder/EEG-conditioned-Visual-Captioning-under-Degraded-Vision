from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class LaunchAfterManifestArtifactTests(unittest.TestCase):
    def test_manifests_ready_requires_all_paths(self) -> None:
        from scripts.launch_after_manifest_artifact import manifests_ready

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train = root / "train.jsonl"
            val = root / "val.jsonl"

            self.assertFalse(manifests_ready([train, val]))
            train.write_text("{}\n", encoding="utf-8")
            self.assertFalse(manifests_ready([train, val]))
            val.write_text("{}\n", encoding="utf-8")
            self.assertTrue(manifests_ready([train, val]))

    def test_link_reports_ready_requires_fully_image_linked_status(self) -> None:
        from scripts.launch_after_manifest_artifact import link_reports_ready

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train = root / "train_report.md"
            val = root / "val_report.md"

            self.assertFalse(link_reports_ready([train, val]))

            train.write_text("- Loader-ready status: `fully image-linked`\n", encoding="utf-8")
            val.write_text("- Loader-ready status: `not fully image-linked`\n", encoding="utf-8")
            self.assertFalse(link_reports_ready([train, val]))

            val.write_text("- Loader-ready status: `fully image-linked`\n", encoding="utf-8")
            self.assertTrue(link_reports_ready([train, val]))

    def test_link_reports_ready_accepts_exact_linked_subset_status(self) -> None:
        from scripts.launch_after_manifest_artifact import link_reports_ready

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train = root / "train_report.md"
            val = root / "val_report.md"
            for path in (train, val):
                path.write_text(
                    "\n".join(
                        [
                            "- Loader-ready status: `not fully image-linked`",
                            "- Exact-linked subset status: `ready for paired training`",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )

            self.assertTrue(link_reports_ready([train, val]))

    def test_build_queue_ready_command_sets_job_queued(self) -> None:
        from scripts.launch_after_manifest_artifact import build_queue_ready_command

        command = build_queue_ready_command(
            queue=Path("configs/heavy_stage_queue.yaml"),
            job_id="EEG_IMAGENET_PAIRED_ALIGNMENT_AFTER_IMAGE_LINK",
        )

        self.assertEqual(command[0], "python")
        self.assertEqual(command[1], "scripts/update_heavy_stage_queue_status.py")
        self.assertIn("--status", command)
        self.assertIn("queued", command)


if __name__ == "__main__":
    unittest.main()
