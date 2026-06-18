from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch


class EEGImageNetManifestTests(unittest.TestCase):
    def test_build_manifest_writes_sample_rows_eeg_arrays_and_not_ready_report(self) -> None:
        from scripts.build_eeg_imagenet_manifest import build_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "EEG-ImageNet_1.pth"
            out_dir = root / "converted"
            torch.save(
                {
                    "dataset": [
                        {
                            "eeg_data": torch.ones((62, 501), dtype=torch.float32),
                            "subject": 3,
                            "label": "n02510455",
                            "image": "n02510455_4381.JPEG",
                            "granularity": "coarse",
                        },
                        {
                            "eeg_data": torch.zeros((62, 501), dtype=torch.float32),
                            "subject": 4,
                            "label": "n03452741",
                            "image": "n03452741_1234.JPEG",
                            "granularity": "coarse",
                        },
                    ],
                    "labels": ["n02510455", "n03452741"],
                    "images": ["n02510455_4381.JPEG", "n03452741_1234.JPEG"],
                },
                source,
            )

            stats = build_manifest([source], out_dir=out_dir, max_samples=2)

            manifest = out_dir / "small_manifest.jsonl"
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(stats["sample_count"], 2)
            self.assertEqual(rows[0]["image_id"], "n02510455_4381")
            self.assertEqual(rows[0]["image_path"], "images/n02510455/n02510455_4381.JPEG")
            self.assertEqual(rows[0]["eeg_path"], "eeg/eeg_imagenet_000000.npy")
            self.assertEqual(rows[0]["caption"], "a photo of a giant panda")
            self.assertEqual(rows[0]["label"], 0)
            self.assertEqual(rows[0]["subject_id"], "S03")
            self.assertEqual(rows[0]["split"], "train")
            self.assertTrue((out_dir / rows[0]["eeg_path"]).exists())

            report_text = (out_dir / "EEG_IMAGENET_READY_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("Loader-ready status: `not fully loader-ready`", report_text)
            self.assertIn("Image path status: `logical paths only", report_text)
            self.assertIn("EEG shapes", report_text)


if __name__ == "__main__":
    unittest.main()
