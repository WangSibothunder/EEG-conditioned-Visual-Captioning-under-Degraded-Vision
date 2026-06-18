from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import src.eval.retrieval as retrieval_mod
from src.eval.retrieval import random_retrieval_metrics, retrieval_metric_bundle, retrieval_metrics
from src.train.train_align import class_accuracy_from_logits


class RetrievalMetricTests(unittest.TestCase):
    def test_retrieval_can_count_duplicate_image_ids_as_positives(self) -> None:
        query = torch.tensor([[0.0, 1.0]])
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        strict = retrieval_metrics(query, target, ks=(1,))
        duplicate_aware = retrieval_metrics(
            query,
            target,
            ks=(1,),
            query_ids=["same-image"],
            target_ids=["same-image", "same-image"],
        )

        self.assertEqual(strict["r@1"], 0.0)
        self.assertEqual(duplicate_aware["r@1"], 1.0)

    def test_random_retrieval_uses_duplicate_positive_set_analytically(self) -> None:
        metrics = random_retrieval_metrics(
            4,
            ks=(1,),
            query_ids=["a", "b", "c", "d"],
            target_ids=["a", "a", "b", "c"],
        )

        self.assertAlmostEqual(metrics["r@1"], 0.25)

    def test_random_retrieval_caches_duplicate_positive_median(self) -> None:
        with patch.object(retrieval_mod, "_random_best_rank_median", return_value=3.0) as median:
            metrics = random_retrieval_metrics(
                5,
                ks=(1,),
                query_ids=["same"] * 5,
                target_ids=["same"] * 5,
            )

        self.assertEqual(metrics["median_rank"], 3.0)
        self.assertEqual(median.call_count, 1)

    def test_random_best_rank_median_matches_bruteforce_for_small_cases(self) -> None:
        def brute(target_count: int, positives: int) -> float:
            for rank in range(1, target_count + 1):
                if retrieval_mod._random_best_rank_recall(target_count, positives, rank) >= 0.5:
                    return float(rank)
            return float(target_count)

        for target_count in range(2, 16):
            for positives in range(1, target_count + 1):
                self.assertEqual(
                    retrieval_mod._random_best_rank_median(target_count, positives),
                    brute(target_count, positives),
                )

    def test_metric_bundle_reports_trial_and_unique_image_levels(self) -> None:
        query = torch.nn.functional.normalize(torch.eye(4), dim=-1)
        target = query.clone()
        image_ids = ["img_a", "img_a", "img_b", "img_c"]

        bundle = retrieval_metric_bundle(query, target, image_ids=image_ids)

        self.assertIn("trial", bundle)
        self.assertIn("unique_image", bundle)
        self.assertIn("random_trial", bundle)
        self.assertIn("random_unique_image", bundle)
        self.assertGreaterEqual(bundle["unique_image"]["r@5"], bundle["unique_image"]["r@1"])

    def test_class_accuracy_from_logits(self) -> None:
        logits = torch.tensor([[0.0, 2.0], [3.0, 1.0], [0.5, 0.7]])
        labels = torch.tensor([1, 0, 0])

        self.assertAlmostEqual(class_accuracy_from_logits(logits, labels), 2 / 3)


if __name__ == "__main__":
    unittest.main()
