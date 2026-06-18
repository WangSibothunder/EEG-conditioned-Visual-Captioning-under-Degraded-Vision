import unittest

import numpy as np
import torch

from src.train.train_align import AlignmentDataset, alignment_collate
from src.train.train_trimodal_align import _build_text_lookup, _lookup_text_index
from src.train.train_trimodal_align import TriModalAlignmentDataset, trimodal_collate


class _BaseDataset:
    rows = [
        {
            "image_id": "img0",
            "caption": "a photo of class",
            "label": 1,
            "subject_id": "S0",
            "split": "train",
            "caption_source": "human_class",
        }
    ]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        raise AssertionError("cache-backed dataset should not load base image/eeg item")


class TriModalTextLookupTests(unittest.TestCase):
    def test_lookup_uses_row_caption_source_for_wnid_fallback(self) -> None:
        text_index = [
            {
                "split": "train",
                "caption_source": "wnid_fallback",
                "image_id": "n01456756_123",
                "eeg_index": None,
            }
        ]
        lookup = _build_text_lookup(text_index, text_source="human")

        row = {
            "split": "train",
            "caption_source": "wnid_fallback",
            "image_id": "n01456756_123",
        }

        self.assertEqual(_lookup_text_index(lookup, row), 0)

    def test_alignment_dataset_uses_cache_without_loading_base_item(self) -> None:
        dataset = AlignmentDataset(
            _BaseDataset(),  # type: ignore[arg-type]
            torch.zeros((1, 512), dtype=torch.float32),
            np.zeros((1, 62, 501), dtype=np.float32),
        )

        item = dataset[0]

        self.assertEqual(item["image_id"], "img0")
        self.assertEqual(tuple(item["eeg"].shape), (62, 501))
        self.assertEqual(tuple(item["clip_emb"].shape), (512,))

    def test_alignment_collate_accepts_cache_only_items_without_image_tensor(self) -> None:
        batch = [
            {
                "eeg": torch.zeros((62, 501), dtype=torch.float32),
                "caption": "a photo of class",
                "image_id": "img0",
                "label": 1,
                "subject_id": "S0",
                "clip_emb": torch.ones((512,), dtype=torch.float32),
            }
        ]

        out = alignment_collate(batch)

        self.assertNotIn("image", out)
        self.assertEqual(tuple(out["eeg"].shape), (1, 62, 501))
        self.assertEqual(tuple(out["clip_emb"].shape), (1, 512))
        self.assertEqual(out["image_id"], ["img0"])

    def test_trimodal_dataset_uses_cache_without_loading_base_item(self) -> None:
        text_index = [
            {
                "split": "train",
                "caption_source": "human_class",
                "image_id": "img0",
                "eeg_index": None,
            }
        ]
        dataset = TriModalAlignmentDataset(
            _BaseDataset(),  # type: ignore[arg-type]
            torch.zeros((1, 512), dtype=torch.float32),
            torch.ones((1, 512), dtype=torch.float32),
            text_index,
            text_source="human",
            eeg_cache=np.zeros((1, 62, 501), dtype=np.float32),
        )

        item = dataset[0]

        self.assertEqual(item["image_id"], "img0")
        self.assertEqual(tuple(item["eeg"].shape), (62, 501))
        self.assertEqual(tuple(item["image_emb"].shape), (512,))
        self.assertEqual(tuple(item["text_emb"].shape), (512,))

    def test_trimodal_collate_accepts_cache_only_items_without_image_tensor(self) -> None:
        batch = [
            {
                "eeg": torch.zeros((62, 501), dtype=torch.float32),
                "caption": "a photo of class",
                "image_id": "img0",
                "label": 1,
                "subject_id": "S0",
                "image_emb": torch.ones((512,), dtype=torch.float32),
                "text_emb": torch.full((512,), 2.0, dtype=torch.float32),
                "text_cache_index": 3,
            }
        ]

        out = trimodal_collate(batch)

        self.assertNotIn("image", out)
        self.assertEqual(tuple(out["eeg"].shape), (1, 62, 501))
        self.assertEqual(tuple(out["image_emb"].shape), (1, 512))
        self.assertEqual(tuple(out["text_emb"].shape), (1, 512))
        self.assertEqual(out["text_cache_index"].tolist(), [3])


if __name__ == "__main__":
    unittest.main()
