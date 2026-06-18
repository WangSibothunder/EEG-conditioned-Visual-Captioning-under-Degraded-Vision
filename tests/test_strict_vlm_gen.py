from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from scripts.run_strict_vlm_gen import (
    StrictRouteConfig,
    StrictTokenBatch,
    build_semantic_prompt,
    route1_full_variant_configs,
    route1_variant_configs,
    route3_variant_configs,
    route4_variant_configs,
    route5_variant_configs,
)


class StrictVLMGenTests(unittest.TestCase):
    def test_semantic_prompt_contains_topk_and_corruption(self) -> None:
        prompt = build_semantic_prompt(
            ["grand piano", "reflex camera", "missile", "banana", "German shepherd"],
            "strong_noise",
        )

        self.assertIn("Write one short natural image caption.", prompt)
        self.assertIn("Candidate visual concepts:", prompt)
        self.assertIn("grand piano", prompt)
        self.assertIn("German shepherd", prompt)
        self.assertIn("Visual condition:", prompt)
        self.assertIn("strong_noise", prompt)
        self.assertTrue(prompt.strip().endswith("Caption:"))

    def test_route1_required_variants_are_configured(self) -> None:
        variants = route1_variant_configs()
        by_name = {variant.name: variant for variant in variants}

        self.assertIn("qwen_prefix_only", by_name)
        self.assertIn("qwen_semantic_prompt_only", by_name)
        self.assertIn("qwen_prefix_semantic_prompt", by_name)
        self.assertIn("qwen_prefix_semantic_prompt_lora_r8", by_name)
        self.assertFalse(by_name["qwen_prefix_only"].use_semantic_prompt)
        self.assertTrue(by_name["qwen_semantic_prompt_only"].use_semantic_prompt)
        self.assertFalse(by_name["qwen_semantic_prompt_only"].use_prefix)
        self.assertTrue(by_name["qwen_prefix_semantic_prompt_lora_r8"].use_lora)
        self.assertEqual(by_name["qwen_prefix_semantic_prompt_lora_r8"].lora_r, 8)

    def test_strict_token_batch_keeps_visual_token_sequence(self) -> None:
        batch = StrictTokenBatch(
            visual_tokens=F.normalize(torch.randn(2, 50, 512), dim=-1),
            eeg_tokens=F.normalize(torch.randn(2, 4, 512), dim=-1),
            topk_prototypes=F.normalize(torch.randn(2, 5, 512), dim=-1),
            confidence=torch.rand(2, 1),
            corruption_ids=torch.tensor([0, 1]),
        )
        self.assertEqual(tuple(batch.visual_tokens.shape), (2, 50, 512))

    def test_config_rejects_route_without_prefix_or_semantic_prompt(self) -> None:
        with self.assertRaises(ValueError):
            StrictRouteConfig(name="bad", use_prefix=False, use_semantic_prompt=False)

    def test_expanded_route_configs_cover_missing_routes(self) -> None:
        route1 = {variant.name for variant in route1_full_variant_configs()}
        self.assertIn("EVG1B_full_lora_r8", route1)
        self.assertIn("EVG1C_full_lora_r16", route1)

        route3 = {variant.name for variant in route3_variant_configs()}
        self.assertIn("LLaVAStyle_projector_frozenLLM", route3)
        self.assertIn("LLaVAStyle_projector_lora", route3)
        self.assertIn("LLaVAStyle_projector_topk_prompt_lora", route3)

        route4 = {variant.name for variant in route4_variant_configs()}
        self.assertIn("BLIP2_actual_OPT_prefix_adapter", route4)

        route5 = {variant.name for variant in route5_variant_configs()}
        self.assertIn("Qwen2VL_2B_prefix_semantic_adapter", route5)


if __name__ == "__main__":
    unittest.main()
