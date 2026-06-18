from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class EEGImageNetImageMappingTests(unittest.TestCase):
    def test_rewrite_manifest_marks_rows_ready_when_imagenet_files_exist(self) -> None:
        from scripts.link_eeg_imagenet_images import rewrite_manifest_image_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "input.jsonl"
            image_root = root / "imagenet"
            output = root / "mapped.jsonl"
            report = root / "report.md"

            rows = [
                {
                    "image_id": "n02510455_4381",
                    "image_path": "images/n02510455/n02510455_4381.JPEG",
                    "eeg_path": "eeg/eeg_imagenet_000000.npy",
                    "caption": "a photo of a giant panda",
                    "label": 0,
                    "subject_id": "S00",
                    "split": "train",
                },
                {
                    "image_id": "n03452741_1234",
                    "image_path": "images/n03452741/n03452741_1234.JPEG",
                    "eeg_path": "eeg/eeg_imagenet_000001.npy",
                    "caption": "a photo of a grand piano",
                    "label": 1,
                    "subject_id": "S01",
                    "split": "val",
                },
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            (image_root / "train" / "n02510455").mkdir(parents=True)
            (image_root / "train" / "n03452741").mkdir(parents=True)
            (image_root / "train" / "n02510455" / "n02510455_4381.JPEG").write_bytes(b"fake")
            (image_root / "train" / "n03452741" / "n03452741_1234.JPEG").write_bytes(b"fake")

            stats = rewrite_manifest_image_paths(
                manifest,
                image_root=image_root,
                out=output,
                report=report,
                relative_to=root,
            )

            mapped_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(stats["matched_rows"], 2)
            self.assertEqual(stats["missing_rows"], 0)
            self.assertEqual(mapped_rows[0]["image_path"], "imagenet/train/n02510455/n02510455_4381.JPEG")
            self.assertEqual(mapped_rows[1]["image_path"], "imagenet/train/n03452741/n03452741_1234.JPEG")
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("Loader-ready status: `fully image-linked`", report_text)

    def test_rewrite_manifest_reports_missing_images_without_dropping_rows(self) -> None:
        from scripts.link_eeg_imagenet_images import rewrite_manifest_image_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "input.jsonl"
            image_root = root / "imagenet"
            output = root / "mapped.jsonl"
            report = root / "report.md"

            row = {
                "image_id": "n02510455_4381",
                "image_path": "images/n02510455/n02510455_4381.JPEG",
                "eeg_path": "eeg/eeg_imagenet_000000.npy",
                "caption": "a photo of a giant panda",
                "label": 0,
                "subject_id": "S00",
                "split": "train",
            }
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            image_root.mkdir()

            stats = rewrite_manifest_image_paths(
                manifest,
                image_root=image_root,
                out=output,
                report=report,
                relative_to=root,
            )

            mapped_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(stats["matched_rows"], 0)
            self.assertEqual(stats["missing_rows"], 1)
            self.assertEqual(mapped_rows[0]["image_path"], row["image_path"])
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("Loader-ready status: `not fully image-linked`", report_text)
            self.assertIn("n02510455_4381", report_text)

    def test_rewrite_manifest_writes_exact_linked_filtered_subset(self) -> None:
        from scripts.link_eeg_imagenet_images import rewrite_manifest_image_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "input.jsonl"
            image_root = root / "imagenet"
            output = root / "mapped.jsonl"
            filtered = root / "mapped_exact.jsonl"
            report = root / "report.md"

            rows = [
                {
                    "image_id": "n02510455_4381",
                    "image_path": "images/n02510455/n02510455_4381.JPEG",
                    "eeg_path": "eeg/eeg_imagenet_000000.npy",
                    "caption": "a photo of a giant panda",
                    "label": 0,
                    "subject_id": "S00",
                    "split": "train",
                },
                {
                    "image_id": "n02510455_missing",
                    "image_path": "images/n02510455/n02510455_missing.JPEG",
                    "eeg_path": "eeg/eeg_imagenet_000001.npy",
                    "caption": "a photo of a giant panda",
                    "label": 0,
                    "subject_id": "S01",
                    "split": "train",
                },
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            image_dir = image_root / "train" / "n02510455"
            image_dir.mkdir(parents=True)
            (image_dir / "n02510455_4381.JPEG").write_bytes(b"fake")

            stats = rewrite_manifest_image_paths(
                manifest,
                image_root=image_root,
                out=output,
                report=report,
                relative_to=root,
                filtered_out=filtered,
            )

            all_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            filtered_rows = [json.loads(line) for line in filtered.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(all_rows), 2)
            self.assertEqual(len(filtered_rows), 1)
            self.assertEqual(filtered_rows[0]["image_id"], "n02510455_4381")
            self.assertEqual(stats["filtered_rows"], 1)
            self.assertEqual(stats["missing_unique_images"], 1)
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("Exact-linked filtered manifest", report_text)
            self.assertIn("Missing unique image IDs: `1`", report_text)

    def test_rewrite_manifest_finds_kaggle_cls_loc_train_layout(self) -> None:
        from scripts.link_eeg_imagenet_images import rewrite_manifest_image_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "input.jsonl"
            image_root = root / "kaggle_cls_loc" / "extracted"
            output = root / "mapped.jsonl"
            report = root / "report.md"

            row = {
                "image_id": "n02510455_4381",
                "image_path": "images/n02510455/n02510455_4381.JPEG",
                "eeg_path": "eeg/eeg_imagenet_000000.npy",
                "caption": "a photo of a giant panda",
                "label": 0,
                "subject_id": "S00",
                "split": "train",
            }
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            kaggle_train_dir = image_root / "ILSVRC" / "Data" / "CLS-LOC" / "train" / "n02510455"
            kaggle_train_dir.mkdir(parents=True)
            (kaggle_train_dir / "n02510455_4381.JPEG").write_bytes(b"fake")

            stats = rewrite_manifest_image_paths(
                manifest,
                image_root=image_root,
                out=output,
                report=report,
                relative_to=root,
            )

            mapped_row = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertEqual(stats["matched_rows"], 1)
            self.assertEqual(mapped_row["image_path"], "kaggle_cls_loc/extracted/ILSVRC/Data/CLS-LOC/train/n02510455/n02510455_4381.JPEG")


if __name__ == "__main__":
    unittest.main()
