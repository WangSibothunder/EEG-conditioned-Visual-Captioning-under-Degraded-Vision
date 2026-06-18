from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from scripts.run_evlm_improve import (
    DEFAULT_CORRUPTIONS,
    EVLMEnhancer,
    REQUIRED_VARIANTS,
    RESIDUAL_VARIANTS,
    PROTO_VARIANTS,
    semantic_gap_rows,
    selection_row,
)


class EVLMImproveTests(unittest.TestCase):
    def test_all_required_variants_forward_to_class_logits(self) -> None:
        torch.manual_seed(3)
        prototypes = F.normalize(torch.randn(5, 16), dim=-1)
        image = F.normalize(torch.randn(7, 16), dim=-1)
        eeg = F.normalize(torch.randn(7, 16), dim=-1)

        for variant in REQUIRED_VARIANTS:
            with self.subTest(variant=variant):
                model = EVLMEnhancer(variant=variant, embed_dim=16, num_classes=5, hidden_dim=32, tau_cls=0.07)
                logits, aux = model(image, eeg, prototypes)

                self.assertEqual(tuple(logits.shape), (7, 5))
                self.assertTrue(torch.isfinite(logits).all())
                if variant in RESIDUAL_VARIANTS:
                    self.assertIn("alpha", aux)
                    self.assertIn("delta_norm", aux)
                if variant in PROTO_VARIANTS:
                    self.assertIn("gamma", aux)

    def test_semantic_gap_rows_matches_final_metric_schema(self) -> None:
        metrics = []
        for corruption in DEFAULT_CORRUPTIONS:
            metrics.extend(
                [
                    {"corruption": corruption, "mode": "vision_only", "accuracy": 0.2, "top5_accuracy": 0.5, "caption_class_hit": 0.2},
                    {"corruption": corruption, "mode": "real_eeg", "accuracy": 0.6, "top5_accuracy": 0.8, "caption_class_hit": 0.6},
                    {"corruption": corruption, "mode": "shuffled_eeg", "accuracy": 0.3, "top5_accuracy": 0.6, "caption_class_hit": 0.3},
                    {"corruption": corruption, "mode": "random_eeg", "accuracy": 0.1, "top5_accuracy": 0.4, "caption_class_hit": 0.1},
                ]
            )

        rows = semantic_gap_rows(metrics, model_name="A2_proto_bias", seed=42, eval_dir="outputs/x")  # type: ignore[arg-type]

        self.assertEqual(len(rows), len(DEFAULT_CORRUPTIONS))
        self.assertEqual(rows[0]["model"], "A2_proto_bias")
        self.assertAlmostEqual(rows[0]["real_minus_vision"], 0.4)
        self.assertTrue(rows[0]["real_beats_vision"])
        self.assertTrue(rows[0]["real_beats_controls"])

    def test_selection_row_scores_strong_degradation_only(self) -> None:
        rows = [
            {
                "model": "x",
                "seed": 42,
                "corruption": "clean",
                "real_top1": 0.99,
                "real_minus_vision": 0.01,
                "real_minus_shuffled": 0.1,
                "real_minus_random": 0.1,
                "real_beats_vision": True,
                "real_beats_controls": True,
            },
            {
                "model": "x",
                "seed": 42,
                "corruption": "mixed",
                "real_top1": 0.4,
                "real_minus_vision": 0.2,
                "real_minus_shuffled": 0.3,
                "real_minus_random": 0.4,
                "real_beats_vision": True,
                "real_beats_controls": True,
            },
        ]

        scored = selection_row("x", rows)

        self.assertAlmostEqual(scored["strong_real_top1_mean"], 0.4)
        self.assertAlmostEqual(scored["real_minus_vision_mean"], 0.2)


if __name__ == "__main__":
    unittest.main()
