from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from scripts import precompute_degraded_vision


class _FakeVariableDimEncoder(nn.Module):
    def __init__(self, output_dim: int, *, using_fallback: bool = False) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.using_fallback = using_fallback

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = torch.arange(1, self.output_dim + 1, dtype=torch.float32, device=images.device)
        return values.unsqueeze(0).repeat(images.shape[0], 1)


class PrecomputeDegradedVisionTests(unittest.TestCase):
    def test_degraded_cache_preserves_encoder_native_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            (root / "eeg").mkdir()
            rows = []
            for idx in range(2):
                image_id = f"img{idx}"
                Image.new("RGB", (8, 8), color=(idx * 50, 0, 128)).save(root / "images" / f"{image_id}.jpg")
                np.save(root / "eeg" / f"{image_id}.npy", np.zeros((64, 250), dtype=np.float32))
                rows.append(
                    {
                        "image_id": image_id,
                        "image_path": f"images/{image_id}.jpg",
                        "eeg_path": f"eeg/{image_id}.npy",
                        "caption": "a test image",
                        "label": idx,
                    }
                )
            manifest = root / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            stats = precompute_degraded_vision.precompute_degraded_clip_caches(
                manifest=manifest,
                out_dir=root / "cache",
                corruptions=["clean"],
                batch_size=2,
                image_size=8,
                device="cpu",
                overwrite=True,
                encoder_factory=lambda **_: _FakeVariableDimEncoder(768),
            )

            arr = np.load(root / "cache" / "clip_test_clean.npy")

        self.assertEqual(stats[0]["embedding_shape"], [2, 768])
        self.assertEqual(tuple(arr.shape), (2, 768))

    def test_require_real_model_rejects_fallback_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            (root / "eeg").mkdir()
            Image.new("RGB", (8, 8), color=(0, 0, 128)).save(root / "images" / "img0.jpg")
            np.save(root / "eeg" / "img0.npy", np.zeros((64, 250), dtype=np.float32))
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "image_id": "img0",
                        "image_path": "images/img0.jpg",
                        "eeg_path": "eeg/img0.npy",
                        "caption": "a test image",
                        "label": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "real vision model"):
                precompute_degraded_vision.precompute_degraded_clip_caches(
                    manifest=manifest,
                    out_dir=root / "cache",
                    corruptions=["clean"],
                    image_size=8,
                    device="cpu",
                    require_real_model=True,
                    encoder_factory=lambda **_: _FakeVariableDimEncoder(512, using_fallback=True),
                )

    def test_degraded_cache_passes_num_workers_to_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            (root / "eeg").mkdir()
            Image.new("RGB", (8, 8), color=(0, 0, 128)).save(root / "images" / "img0.jpg")
            np.save(root / "eeg" / "img0.npy", np.zeros((64, 250), dtype=np.float32))
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "image_id": "img0",
                        "image_path": "images/img0.jpg",
                        "eeg_path": "eeg/img0.npy",
                        "caption": "a test image",
                        "label": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            seen: dict[str, int] = {}
            original_loader = precompute_degraded_vision.DataLoader

            def recording_loader(*args, **kwargs):
                seen["num_workers"] = kwargs.get("num_workers")
                seen["prefetch_factor"] = kwargs.get("prefetch_factor", 0)
                return original_loader(*args, **kwargs)

            with unittest.mock.patch.object(precompute_degraded_vision, "DataLoader", recording_loader):
                precompute_degraded_vision.precompute_degraded_clip_caches(
                    manifest=manifest,
                    out_dir=root / "cache",
                    corruptions=["clean"],
                    image_size=8,
                    device="cpu",
                    overwrite=True,
                    num_workers=2,
                    prefetch_factor=3,
                    encoder_factory=lambda **_: _FakeVariableDimEncoder(8),
                )

        self.assertEqual(seen["num_workers"], 2)
        self.assertEqual(seen["prefetch_factor"], 3)


if __name__ == "__main__":
    unittest.main()
