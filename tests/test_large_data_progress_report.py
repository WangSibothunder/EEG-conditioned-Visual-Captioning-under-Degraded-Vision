from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class LargeDataProgressReportTests(unittest.TestCase):
    def test_extraction_progress_uses_live_files_before_marker_exists(self) -> None:
        from scripts.update_large_data_progress_report import extraction_progress

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = root / "ILSVRC" / "Data" / "CLS-LOC" / "val" / "ILSVRC2012_val_00000001.JPEG"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"12345")

            progress = extraction_progress(root, {"extract_marker": None})

            self.assertEqual(progress["bytes"], 5)
            self.assertEqual(progress["files"], 1)
            self.assertEqual(progress["source"], "live")

    def test_extraction_progress_prefers_complete_marker(self) -> None:
        from scripts.update_large_data_progress_report import extraction_progress

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = root / "partial.JPEG"
            image.write_bytes(b"12345")

            progress = extraction_progress(
                root,
                {"extract_marker": {"status": "complete", "uncompressed_bytes": 99, "files": 7}},
            )

            self.assertEqual(progress["bytes"], 99)
            self.assertEqual(progress["files"], 7)
            self.assertEqual(progress["source"], "marker")

    def test_eeg_imagenet_pairing_status_reports_exact_subset_ready(self) -> None:
        from scripts.update_large_data_progress_report import eeg_imagenet_pairing_status

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_dir = root / "data" / "EEG-ImageNet"
            report_dir = root / "outputs" / "datasets"
            manifest_dir.mkdir(parents=True)
            report_dir.mkdir(parents=True)
            rows = {"train": 3, "val": 2, "test": 1}
            for split, count in rows.items():
                (manifest_dir / f"{split}_image_exact.jsonl").write_text("{}\n" * count, encoding="utf-8")
                (report_dir / f"EEG_IMAGENET_IMAGE_LINK_{split.upper()}_REPORT.md").write_text(
                    "\n".join(
                        [
                            "- Loader-ready status: `not fully image-linked`",
                            f"- Exact-linked filtered rows: `{count}`",
                            "- Exact-linked subset status: `ready for paired training`",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )

            status = eeg_imagenet_pairing_status(root)

            self.assertEqual(status["status"], "exact-linked subset ready")
            self.assertEqual(status["rows"], {"train": 3, "val": 2, "test": 1})
            self.assertTrue(status["ready"])

    def test_zip_status_mentions_cleaned_archive_after_complete_extraction(self) -> None:
        from scripts.update_large_data_progress_report import zip_status_text

        text = zip_status_text(
            zip_size=0,
            expected_zip_bytes=155 * 1024**3,
            central_directory_ok=False,
            extraction_status={"complete": True, "extract_marker": {"zip_size": 166_496_672_546}},
        )

        self.assertIn("archive cleaned after extraction", text)
        self.assertIn("original zip", text)
        self.assertIn("GiB", text)
        self.assertNotIn("0.0% of ~155GiB", text)
        self.assertNotIn("zip valid=False", text)


if __name__ == "__main__":
    unittest.main()
