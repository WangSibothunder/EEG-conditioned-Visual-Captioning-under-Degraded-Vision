from __future__ import annotations

import unittest

import torch

from src.models.eeg_encoder import build_eeg_encoder


class EEGEncoderVariantTests(unittest.TestCase):
    def test_encoder_variants_return_512d_embeddings(self) -> None:
        eeg = torch.randn(2, 64, 250)
        for encoder_type in ["tiny", "eegnet", "multiscale_tcn", "convtransformer_base", "convtransformer_strong"]:
            with self.subTest(encoder_type=encoder_type):
                model = build_eeg_encoder(
                    encoder_type=encoder_type,
                    channels=64,
                    timesteps=250,
                    output_dim=512,
                )
                out = model(eeg)
                self.assertEqual(tuple(out.shape), (2, 512))
                self.assertTrue(torch.isfinite(out).all())

    def test_encoder_variants_have_different_parameter_counts(self) -> None:
        tiny = build_eeg_encoder("tiny", channels=64, timesteps=250, output_dim=512)
        base = build_eeg_encoder("convtransformer_base", channels=64, timesteps=250, output_dim=512)

        tiny_params = sum(parameter.numel() for parameter in tiny.parameters())
        base_params = sum(parameter.numel() for parameter in base.parameters())

        self.assertGreater(base_params, tiny_params)

    def test_subject_adaptive_encoder_uses_subject_ids(self) -> None:
        eeg = torch.randn(2, 64, 250)
        model = build_eeg_encoder("subject_adaptive", channels=64, timesteps=250, output_dim=512, num_subjects=8)

        out_a = model(eeg, subject_ids=["1", "1"])
        out_b = model(eeg, subject_ids=["2", "2"])

        self.assertEqual(tuple(out_a.shape), (2, 512))
        self.assertFalse(torch.allclose(out_a, out_b))

    def test_heavy_goal_named_architectures_forward(self) -> None:
        eeg = torch.randn(2, 64, 250)
        variants = [
            "dualbranch_eegconformer",
            "temporal_spectral_spatial",
            "subject_adaptive_graph",
        ]

        for encoder_type in variants:
            with self.subTest(encoder_type=encoder_type):
                model = build_eeg_encoder(
                    encoder_type=encoder_type,
                    channels=64,
                    timesteps=250,
                    output_dim=512,
                    hidden_dim=128,
                    transformer_layers=2,
                    dropout=0.1,
                    num_subjects=8,
                )
                if encoder_type == "subject_adaptive_graph":
                    out = model(eeg, subject_ids=["S01", "S02"])
                else:
                    out = model(eeg)
                self.assertEqual(tuple(out.shape), (2, 512))
                self.assertTrue(torch.isfinite(out).all())

    def test_heavy_goal_named_architectures_are_not_tiny(self) -> None:
        tiny = build_eeg_encoder("tiny", channels=64, timesteps=250, output_dim=512)
        tiny_params = sum(parameter.numel() for parameter in tiny.parameters())

        for encoder_type in ["dualbranch_eegconformer", "temporal_spectral_spatial", "subject_adaptive_graph"]:
            with self.subTest(encoder_type=encoder_type):
                model = build_eeg_encoder(
                    encoder_type=encoder_type,
                    channels=64,
                    timesteps=250,
                    output_dim=512,
                    hidden_dim=128,
                    num_subjects=8,
                )
                params = sum(parameter.numel() for parameter in model.parameters())
                self.assertGreater(params, tiny_params)


if __name__ == "__main__":
    unittest.main()
