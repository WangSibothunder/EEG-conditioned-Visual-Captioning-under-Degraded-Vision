from __future__ import annotations

import unittest

import torch

from src.data.corruptions import apply_corruption


class CorruptionTests(unittest.TestCase):
    def test_heavy_stage_corruptions_keep_shape_and_range(self) -> None:
        torch.manual_seed(7)
        images = torch.rand(2, 3, 96, 96)

        for name in [
            "strong_blur",
            "strong_noise",
            "occlusion50",
            "lowres16",
            "mixed",
        ]:
            out = apply_corruption(images, name)
            self.assertEqual(tuple(out.shape), tuple(images.shape), name)
            self.assertGreaterEqual(float(out.min()), 0.0, name)
            self.assertLessEqual(float(out.max()), 1.0, name)

    def test_strong_corruptions_are_more_aggressive_than_default_variants(self) -> None:
        torch.manual_seed(11)
        images = torch.rand(2, 3, 96, 96)

        mild_noise_delta = (apply_corruption(images, "noise") - images).abs().mean()
        strong_noise_delta = (apply_corruption(images, "strong_noise") - images).abs().mean()
        self.assertGreater(float(strong_noise_delta), float(mild_noise_delta))

        mild_occlusion_zero = (apply_corruption(images, "occlusion") == 0.0).float().mean()
        strong_occlusion_zero = (apply_corruption(images, "occlusion50") == 0.0).float().mean()
        self.assertGreater(float(strong_occlusion_zero), float(mild_occlusion_zero))

        lowres = apply_corruption(images, "lowres")
        lowres16 = apply_corruption(images, "lowres16")
        self.assertGreater(float((lowres16 - images).abs().mean()), float((lowres - images).abs().mean()))


if __name__ == "__main__":
    unittest.main()
