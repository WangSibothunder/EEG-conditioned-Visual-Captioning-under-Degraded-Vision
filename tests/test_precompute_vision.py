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

from scripts import precompute_vision


class _FakeVariableDimEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.using_fallback = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        values = torch.arange(1, self.output_dim + 1, dtype=torch.float32, device=images.device)
        return values.unsqueeze(0).repeat(batch, 1)


class _FakeFallbackEncoder(_FakeVariableDimEncoder):
    def __init__(self, output_dim: int) -> None:
        super().__init__(output_dim)
        self.using_fallback = True


class _FakeSiglipOutput:
    def __init__(self, pooler_output: torch.Tensor) -> None:
        self.pooler_output = pooler_output


class _FakeSiglipVisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 768})()

    def forward(self, pixel_values: torch.Tensor) -> _FakeSiglipOutput:
        batch = pixel_values.shape[0]
        return _FakeSiglipOutput(torch.ones(batch, 768, device=pixel_values.device))


class PrecomputeVisionTests(unittest.TestCase):
    def test_native_encoder_loads_siglip_with_siglip_specific_loader(self) -> None:
        calls: list[str] = []

        def fake_loader(model_name: str):
            calls.append(model_name)
            return _FakeSiglipVisionModel()

        encoder = precompute_vision.NativeCLIPVisionEncoder(
            "google/siglip-base-patch16-224",
            vision_model_loader=fake_loader,
        )
        output = encoder(torch.zeros(2, 3, 224, 224))

        self.assertFalse(encoder.using_fallback)
        self.assertEqual(calls, ["google/siglip-base-patch16-224"])
        self.assertEqual(tuple(output.shape), (2, 768))

    def test_precompute_cache_preserves_encoder_output_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            (root / "eeg").mkdir()
            rows = []
            for idx in range(2):
                image_id = f"img{idx}"
                Image.new("RGB", (8, 8), color=(idx * 40, 0, 255)).save(root / "images" / f"{image_id}.jpg")
                np.save(root / "eeg" / f"{image_id}.npy", np.zeros((64, 250), dtype=np.float32))
                rows.append(
                    {
                        "image_id": image_id,
                        "image_path": f"images/{image_id}.jpg",
                        "eeg_path": f"eeg/{image_id}.npy",
                        "caption": "a small test image",
                        "label": idx,
                    }
                )
            manifest = root / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            cache = root / "clip.npy"
            index = root / "clip_index.json"

            stats = precompute_vision.precompute_vision_cache(
                manifest,
                cache,
                index,
                batch_size=2,
                image_size=8,
                eeg_shape=(64, 250),
                model_name="fake-variable-dim",
                device="cpu",
                overwrite=True,
                encoder_factory=lambda **_: _FakeVariableDimEncoder(output_dim=768),
            )

            embeddings = np.load(cache)

        self.assertEqual(stats["embedding_shape"], [2, 768])
        self.assertEqual(tuple(embeddings.shape), (2, 768))
        self.assertEqual(embeddings.dtype, np.float16)
        self.assertEqual(stats["dtype"], "float16")

    def test_require_real_model_rejects_fallback_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            (root / "eeg").mkdir()
            Image.new("RGB", (8, 8), color=(0, 0, 255)).save(root / "images" / "img0.jpg")
            np.save(root / "eeg" / "img0.npy", np.zeros((64, 250), dtype=np.float32))
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "image_id": "img0",
                        "image_path": "images/img0.jpg",
                        "eeg_path": "eeg/img0.npy",
                        "caption": "a small test image",
                        "label": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "real vision model"):
                precompute_vision.precompute_vision_cache(
                    manifest,
                    root / "clip.npy",
                    root / "clip_index.json",
                    image_size=8,
                    device="cpu",
                    overwrite=True,
                    require_real_model=True,
                    encoder_factory=lambda **_: _FakeFallbackEncoder(output_dim=512),
                )

    def test_precompute_cache_passes_num_workers_to_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            (root / "eeg").mkdir()
            Image.new("RGB", (8, 8), color=(0, 0, 255)).save(root / "images" / "img0.jpg")
            np.save(root / "eeg" / "img0.npy", np.zeros((64, 250), dtype=np.float32))
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "image_id": "img0",
                        "image_path": "images/img0.jpg",
                        "eeg_path": "eeg/img0.npy",
                        "caption": "a small test image",
                        "label": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            seen: dict[str, int] = {}
            original_loader = precompute_vision.DataLoader

            def recording_loader(*args, **kwargs):
                seen["num_workers"] = kwargs.get("num_workers")
                seen["prefetch_factor"] = kwargs.get("prefetch_factor", 0)
                return original_loader(*args, **kwargs)

            with unittest.mock.patch.object(precompute_vision, "DataLoader", recording_loader):
                precompute_vision.precompute_vision_cache(
                    manifest,
                    root / "clip.npy",
                    root / "clip_index.json",
                    batch_size=1,
                    image_size=8,
                    device="cpu",
                    overwrite=True,
                    num_workers=2,
                    prefetch_factor=3,
                    encoder_factory=lambda **_: _FakeVariableDimEncoder(output_dim=8),
                )

        self.assertEqual(seen["num_workers"], 2)
        self.assertEqual(seen["prefetch_factor"], 3)


if __name__ == "__main__":
    unittest.main()
