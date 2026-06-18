from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.models.eeg_encoder import build_eeg_encoder
from src.models.masked_eeg_autoencoder import MaskedEEGAutoencoder
from src.models.alignment_model import EEGCLIPAlignmentModel
from src.train.train_align import load_pretrained_eeg_encoder


class MaskedPretrainTransferTests(unittest.TestCase):
    def test_masked_pretrained_encoder_returns_requested_embedding_dim(self) -> None:
        model = build_eeg_encoder(
            "masked_pretrained",
            channels=4,
            timesteps=16,
            output_dim=12,
            hidden_dim=32,
            transformer_layers=1,
            dropout=0.0,
        )

        out = model(torch.randn(2, 4, 16))

        self.assertEqual(tuple(out.shape), (2, 12))
        self.assertTrue(torch.isfinite(out).all())

    def test_masked_autoencoder_temporal_spectral_spatial_variant_has_spectral_branch(self) -> None:
        model = MaskedEEGAutoencoder(
            channels=4,
            timesteps=32,
            hidden_dim=32,
            layers=1,
            heads=4,
            ffn_dim=128,
            dropout=0.0,
            spatial_layers=1,
            variant="temporal_spectral_spatial",
        )

        out = model(torch.randn(2, 4, 32))

        self.assertEqual(tuple(out.shape), (2, 4, 32))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(any("spectral" in name for name, _ in model.named_parameters()))

    def test_load_pretrained_eeg_encoder_loads_masked_autoencoder_checkpoint(self) -> None:
        source = MaskedEEGAutoencoder(channels=4, timesteps=16, hidden_dim=32, layers=1, heads=4, ffn_dim=128, dropout=0.0)
        target = EEGCLIPAlignmentModel(
            eeg_channels=4,
            eeg_timesteps=16,
            eeg_dim=12,
            clip_dim=12,
            hidden_dim=32,
            transformer_layers=1,
            dropout=0.0,
            encoder_type="masked_pretrained",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "masked.pt"
            torch.save({"model": source.state_dict()}, checkpoint)

            report = load_pretrained_eeg_encoder(
                target,
                {
                    "pretrained_eeg_checkpoint": str(checkpoint),
                    "pretrained_key": "model",
                    "pretrained_strict": True,
                },
            )

        self.assertEqual(report["loaded"], True)
        for name, tensor in source.state_dict().items():
            loaded = target.eeg_encoder.autoencoder.state_dict()[name]
            self.assertTrue(torch.equal(tensor, loaded), msg=name)

    def test_load_pretrained_eeg_encoder_skips_mismatched_shapes_when_allowed(self) -> None:
        source = MaskedEEGAutoencoder(channels=5, timesteps=20, hidden_dim=32, layers=1, heads=4, ffn_dim=128, dropout=0.0)
        target = EEGCLIPAlignmentModel(
            eeg_channels=4,
            eeg_timesteps=16,
            eeg_dim=12,
            clip_dim=12,
            hidden_dim=32,
            transformer_layers=1,
            dropout=0.0,
            encoder_type="masked_pretrained",
        )
        before = {name: tensor.clone() for name, tensor in target.eeg_encoder.autoencoder.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "masked_cross_shape.pt"
            torch.save({"model": source.state_dict()}, checkpoint)

            report = load_pretrained_eeg_encoder(
                target,
                {
                    "pretrained_eeg_checkpoint": str(checkpoint),
                    "pretrained_key": "model",
                    "pretrained_strict": False,
                    "pretrained_allow_shape_mismatch": True,
                },
            )

        self.assertEqual(report["loaded"], True)
        self.assertGreater(report["loaded_key_count"], 0)
        self.assertGreater(report["skipped_shape_mismatch_count"], 0)
        self.assertIn("temporal_stem.0.weight", report["skipped_shape_mismatch"])
        for name, tensor in target.eeg_encoder.autoencoder.state_dict().items():
            if name in report["loaded_keys"]:
                self.assertTrue(torch.equal(tensor, source.state_dict()[name]), msg=name)
            elif name in report["skipped_shape_mismatch"]:
                self.assertTrue(torch.equal(tensor, before[name]), msg=name)


if __name__ == "__main__":
    unittest.main()
