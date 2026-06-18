from __future__ import annotations

import unittest

import torch
from torch import nn

from src.models.vision_encoder import FrozenCLIPVisionEncoder


class _FakeCLIPOutput:
    def __init__(self, image_embeds: torch.Tensor) -> None:
        self.image_embeds = image_embeds


class _RecordingCLIPEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_pixel_values: torch.Tensor | None = None

    def forward(self, *, pixel_values: torch.Tensor) -> _FakeCLIPOutput:
        self.last_pixel_values = pixel_values.detach().cpu()
        batch = pixel_values.shape[0]
        return _FakeCLIPOutput(torch.zeros(batch, 512, device=pixel_values.device))


class _TestableCLIPVisionEncoder(FrozenCLIPVisionEncoder):
    def _load_clip_or_fallback(self) -> tuple[nn.Module, nn.Module]:
        self.using_fallback = False
        self.recording_encoder = _RecordingCLIPEncoder()
        return self.recording_encoder, nn.Identity()


class VisionEncoderTests(unittest.TestCase):
    def test_real_clip_path_normalizes_raw_zero_one_pixels(self) -> None:
        encoder = _TestableCLIPVisionEncoder()
        images = torch.full((1, 3, 2, 2), 0.5)

        _ = encoder(images)

        received = encoder.recording_encoder.last_pixel_values
        self.assertIsNotNone(received)
        expected = torch.tensor(
            [
                (0.5 - 0.48145466) / 0.26862954,
                (0.5 - 0.4578275) / 0.26130258,
                (0.5 - 0.40821073) / 0.27577711,
            ]
        ).view(1, 3, 1, 1)
        torch.testing.assert_close(received, expected.expand_as(received), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
