from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.run_token_generative_evlm import (
    TrainablePrefixCaptionGenerator,
    TokenFusionModel,
    TokenGenConfig,
    WordCaptionTokenizer,
    valid_caption,
    write_qualitative_examples,
)


class TokenGenerativeEVLMTests(unittest.TestCase):
    def test_token_fusion_accepts_visual_patch_tokens(self) -> None:
        torch.manual_seed(7)
        model = TokenFusionModel(
            "VTF3_confidence_beta_margin_M4",
            embed_dim=512,
            hidden_dim=64,
            tau_cls=0.07,
        )
        visual_tokens = F.normalize(torch.randn(3, 50, 512), dim=-1)
        eeg = F.normalize(torch.randn(3, 512), dim=-1)
        prototypes = F.normalize(torch.randn(6, 512), dim=-1)

        logits, enhanced_img, aux = model(visual_tokens, eeg, prototypes)

        self.assertEqual(tuple(logits.shape), (3, 6))
        self.assertEqual(tuple(enhanced_img.shape), (3, 512))
        self.assertEqual(tuple(aux["enhanced_tokens"].shape), (3, 50, 512))
        self.assertEqual(tuple(aux["eeg_tokens"].shape), (3, 4, 512))
        self.assertTrue(torch.isfinite(logits).all())

    def test_qualitative_examples_include_required_free_form_examples(self) -> None:
        records = []
        for idx in range(35):
            records.append(
                {
                    "image_id": f"img{idx:03d}",
                    "true_class": "grand piano",
                    "corruption": "mixed",
                    "mode": "real_eeg",
                    "generated_caption": "a grand piano in a small room",
                    "valid": True,
                    "class_hit": 1.0,
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "examples.md"
            write_qualitative_examples(records, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("## Best examples for course report", text)
        self.assertGreaterEqual(text.count("img"), 30)
        ok, reason = valid_caption("a grand piano in a small room")
        self.assertTrue(ok, reason)
        weak, weak_reason = valid_caption("a on a")
        self.assertFalse(weak, weak_reason)

    def test_config_defaults_point_to_required_output_root(self) -> None:
        self.assertEqual(TokenGenConfig().output_dir, "outputs/token_generative_evlm")

    def test_trainable_caption_decoder_generates_text(self) -> None:
        captions = [
            "a man sitting at a piano",
            "a red train on the tracks",
            "a camera on a table",
        ]
        tokenizer = WordCaptionTokenizer.from_captions(captions, max_vocab_size=64)
        model = TrainablePrefixCaptionGenerator(
            cond_dim=8,
            tokenizer=tokenizer,
            hidden_dim=16,
            embed_dim=12,
            max_text_length=12,
        )
        cond = torch.randn(3, 8)
        loss = model(cond, captions)
        self.assertTrue(torch.isfinite(loss).all())
        generated = model.generate(cond[:2], max_new_tokens=6)
        self.assertEqual(len(generated), 2)
        self.assertTrue(all(isinstance(item, str) for item in generated))


if __name__ == "__main__":
    unittest.main()
