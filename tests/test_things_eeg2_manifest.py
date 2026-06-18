import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_things_eeg2_manifest import build


class ThingsEEG2ManifestReportTest(unittest.TestCase):
    def test_partial_raw_dataset_reports_not_loader_ready_without_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "THINGS-EEG2"
            image_dir = root / "image_set" / "training_images" / "00001_aardvark"
            eeg_dir = root / "raw-eeg" / "sub-01" / "ses-01"
            image_dir.mkdir(parents=True)
            eeg_dir.mkdir(parents=True)
            (image_dir / "aardvark_01b.jpg").write_bytes(b"not-a-real-image")

            metadata = {
                "train_img_files": ["aardvark_01b.jpg"],
                "train_img_concepts": ["00001_aardvark"],
                "train_img_concepts_THINGS": ["00001_aardvark"],
                "test_img_files": [],
                "test_img_concepts": [],
                "test_img_concepts_THINGS": [],
            }
            np.save(root / "image_metadata.npy", metadata)
            np.save(
                eeg_dir / "raw_eeg_training.npy",
                {
                    "raw_eeg_data": np.zeros((64, 1000), dtype=np.float32),
                    "sfreq": 1000,
                    "ch_names": ["Cz"],
                    "ch_types": ["eeg"],
                    "highpass": 0.0,
                    "lowpass": 100,
                },
            )

            manifest = root / "small_manifest.jsonl"
            status = Path(tmp) / "THINGS_EEG2_READY_REPORT.md"
            smoke = Path(tmp) / "things_smoke.md"

            build(root, manifest, status, smoke)

            self.assertTrue(status.exists())
            text = status.read_text(encoding="utf-8")
            self.assertIn("Loader-ready: `False`", text)
            self.assertIn("Trial/image alignment ready: `False`", text)
            self.assertIn("raw continuous EEG", text)
            self.assertFalse(manifest.exists())
            smoke_text = smoke.read_text(encoding="utf-8")
            self.assertIn("Dataset usable for current image+EEG manifest: `False`", smoke_text)

    def test_project_manifest_pass_through_is_loader_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "THINGS-EEG2"
            root.mkdir()
            source = root / "project_ready.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "image_id": "img001",
                        "image_path": "images/img001.jpg",
                        "eeg_path": "eeg/img001.npy",
                        "caption": "a photo of an object",
                        "label": 0,
                        "subject_id": "sub-01",
                        "split": "train",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            status = Path(tmp) / "status.md"
            smoke = Path(tmp) / "smoke.md"
            build(root, root / "small_manifest.jsonl", status, smoke)

            text = status.read_text(encoding="utf-8")
            self.assertIn("Loader-ready: `True`", text)
            self.assertIn("Project-ready manifest detected", text)


if __name__ == "__main__":
    unittest.main()
