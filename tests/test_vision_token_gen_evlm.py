from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.run_token_generative_evlm import TokenFusionModel, WordCaptionTokenizer
from scripts.run_vision_token_gen_evlm import (
    EVG1TokenPrefixCaptionGenerator,
    EVG2QFormerCaptionGenerator,
    VisionTokenBatch,
    write_enhanced_token_source_report,
)


class VisionTokenGenerativeEVLMTests(unittest.TestCase):
    def _batch(self) -> VisionTokenBatch:
        torch.manual_seed(11)
        return VisionTokenBatch(
            visual_tokens=F.normalize(torch.randn(4, 50, 512), dim=-1),
            eeg_tokens=F.normalize(torch.randn(4, 4, 512), dim=-1),
            topk_prototypes=F.normalize(torch.randn(4, 5, 512), dim=-1),
            confidence=torch.rand(4, 1),
            corruption_ids=torch.tensor([0, 1, 2, 3]),
        )

    def test_evg1_generator_consumes_enhanced_visual_tokens(self) -> None:
        captions = ["a dog running on grass", "a red bus on a road", "a piano in a room", "a bird on a branch"]
        tokenizer = WordCaptionTokenizer.from_captions(captions, max_vocab_size=64)
        model = EVG1TokenPrefixCaptionGenerator(
            tokenizer=tokenizer,
            token_dim=512,
            hidden_dim=32,
            embed_dim=16,
            max_text_length=12,
            num_corruptions=6,
        )

        batch = self._batch()
        loss = model(batch, captions)
        generated = model.generate(batch, max_new_tokens=6)

        self.assertTrue(torch.isfinite(loss).all())
        self.assertEqual(model.last_seen_visual_tokens_shape, (4, 50, 512))
        self.assertEqual(len(generated), 4)
        self.assertTrue(all(isinstance(item, str) for item in generated))

    def test_evg2_qformer_bridge_preserves_token_path(self) -> None:
        captions = ["a dog running on grass", "a red bus on a road", "a piano in a room", "a bird on a branch"]
        tokenizer = WordCaptionTokenizer.from_captions(captions, max_vocab_size=64)
        model = EVG2QFormerCaptionGenerator(
            tokenizer=tokenizer,
            token_dim=512,
            hidden_dim=32,
            embed_dim=16,
            max_text_length=12,
            num_corruptions=6,
            num_queries=4,
        )

        batch = self._batch()
        loss = model(batch, captions)
        _ = model.generate(batch, max_new_tokens=6)

        self.assertTrue(torch.isfinite(loss).all())
        self.assertEqual(model.last_seen_visual_tokens_shape, (4, 50, 512))
        self.assertEqual(model.last_bridge_tokens_shape, (4, 4, 512))

    def test_vtf_aux_enhanced_tokens_can_feed_generator(self) -> None:
        torch.manual_seed(13)
        vtf = TokenFusionModel("VTF3_confidence_beta_margin_M4", embed_dim=512, hidden_dim=64)
        visual_tokens = F.normalize(torch.randn(2, 50, 512), dim=-1)
        eeg = F.normalize(torch.randn(2, 512), dim=-1)
        prototypes = F.normalize(torch.randn(6, 512), dim=-1)
        logits, _emb, aux = vtf(visual_tokens, eeg, prototypes)
        topk = prototypes[logits.topk(k=5, dim=-1).indices]

        tokenizer = WordCaptionTokenizer.from_captions(["a dog running on grass", "a piano in a room"], max_vocab_size=64)
        gen = EVG1TokenPrefixCaptionGenerator(tokenizer=tokenizer, hidden_dim=32, embed_dim=16, max_text_length=12)
        batch = VisionTokenBatch(
            visual_tokens=aux["enhanced_tokens"].detach(),
            eeg_tokens=aux["eeg_tokens"].detach(),
            topk_prototypes=topk.detach(),
            confidence=aux["vision_confidence"].detach(),
            corruption_ids=torch.zeros(2, dtype=torch.long),
        )
        loss = gen(batch, ["a dog running on grass", "a piano in a room"])

        self.assertTrue(torch.isfinite(loss).all())
        self.assertEqual(gen.last_seen_visual_tokens_shape, (2, 50, 512))

    def test_enhanced_token_source_report_contains_required_audit_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ENHANCED_TOKEN_SOURCE.md"
            write_enhanced_token_source_report(
                path,
                vtf_checkpoint=Path("outputs/token_generative_evlm/token_fusion/checkpoints/VTF3_confidence_beta_margin_M4_seed42_best.pt"),
                token_shape=(1997, 50, 512),
                eeg_token_shape=(1997, 4, 512),
                modes=["vision_only", "real_eeg", "shuffled_eeg", "random_eeg", "eeg_only"],
                frozen_modules=["CLIP", "A2 EEG encoder", "VTF3 token fusion"],
            )
            text = path.read_text(encoding="utf-8")

        self.assertIn("VTF3_confidence_beta_margin_M4", text)
        self.assertIn("[1997, 50, 512]", text)
        self.assertIn("real_eeg", text)
        self.assertIn("frozen", text.lower())


if __name__ == "__main__":
    unittest.main()
