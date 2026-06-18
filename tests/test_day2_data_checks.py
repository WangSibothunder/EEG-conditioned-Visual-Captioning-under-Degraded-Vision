from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data.check_split_leakage import check_split_leakage, write_split_leakage_report
from src.data.clip_cache import precompute_clip_cache
from src.data.collate import caption_collate
from src.data.dataset import EEGVisionCaptionDataset


class Day2DataChecksTests(unittest.TestCase):
    def test_split_leakage_checker_counts_image_overlaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for split, image_ids in {
                "train": ["a", "b"],
                "val": ["b", "c"],
                "test": ["d", "a"],
            }.items():
                rows = [
                    {
                        "image_id": image_id,
                        "image_path": f"images/{image_id}.jpg",
                        "eeg_path": f"eeg/{image_id}.npy",
                        "caption": "a photo",
                        "label": 1,
                        "subject_id": "S01",
                        "split": split,
                    }
                    for image_id in image_ids
                ]
                (root / f"{split}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n",
                    encoding="utf-8",
                )
            stats = check_split_leakage(root / "train.jsonl", root / "val.jsonl", root / "test.jsonl")

        self.assertTrue(stats["has_leakage"])
        self.assertEqual(stats["overlaps"]["train_val"], 1)
        self.assertEqual(stats["overlaps"]["train_test"], 1)
        self.assertEqual(stats["overlaps"]["val_test"], 0)

    def test_split_leakage_report_writes_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for split, image_id in {"train": "a", "val": "b", "test": "c"}.items():
                (root / f"{split}.jsonl").write_text(
                    json.dumps({"image_id": image_id, "image_path": "", "caption": "", "label": 0}) + "\n",
                    encoding="utf-8",
                )
            out = root / "report.md"
            stats = check_split_leakage(root / "train.jsonl", root / "val.jsonl", root / "test.jsonl")
            write_split_leakage_report(out, stats)

            text = out.read_text(encoding="utf-8")
            self.assertIn("Leakage exists: `False`", text)
            self.assertIn("train unique images", text)

    def test_dataset_and_collate_preserve_subject_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            (root / "eeg").mkdir()
            from PIL import Image

            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(root / "images" / "a.jpg")
            np.save(root / "eeg" / "a.npy", np.zeros((64, 250), dtype=np.float32))
            manifest = root / "train.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "image_id": "a",
                        "image_path": "images/a.jpg",
                        "eeg_path": "eeg/a.npy",
                        "caption": "a red square",
                        "label": 3,
                        "subject_id": "S01",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            dataset = EEGVisionCaptionDataset(manifest)
            item = dataset[0]
            batch = caption_collate([item])

        self.assertEqual(item["subject_id"], "S01")
        self.assertEqual(batch["subject_id"], ["S01"])
        self.assertEqual(tuple(batch["image"].shape), (1, 3, 224, 224))
        self.assertEqual(tuple(batch["eeg"].shape), (1, 64, 250))

    def test_precompute_clip_cache_skips_existing_cache_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "train.jsonl"
            manifest.write_text(
                json.dumps({"image_id": "a", "image_path": "missing.jpg", "caption": "caption", "label": 0}) + "\n",
                encoding="utf-8",
            )
            cache = root / "clip.npy"
            np.save(cache, np.zeros((1, 512), dtype=np.float16))
            index = root / "clip_index.json"
            index.write_text(json.dumps([{"image_id": "a", "eeg_index": 3}]), encoding="utf-8")

            stats = precompute_clip_cache(manifest, cache, index, overwrite=False)

        self.assertTrue(stats["skipped_existing"])
        self.assertEqual(stats["embedding_shape"], [1, 512])
        self.assertEqual(stats["dtype"], "float16")
        self.assertEqual(stats["unique_images"], 1)


if __name__ == "__main__":
    unittest.main()
