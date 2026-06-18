from __future__ import annotations

import unittest

import torch

from src.train.train_align import compute_alignment_loss


class _DummyAlignmentModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.classifier = torch.nn.Linear(4, 2)

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = eeg.mean(dim=-1)
        pred = torch.nn.functional.normalize(self.proj(pooled), dim=-1)
        return pred, self.classifier(pred)


class TrainAlignLossHookTests(unittest.TestCase):
    def test_compute_alignment_loss_includes_supcon_and_same_image_subject_terms(self) -> None:
        model = _DummyAlignmentModel()
        batch = {
            "eeg": torch.randn(4, 4, 6),
            "clip_emb": torch.nn.functional.normalize(torch.randn(4, 4), dim=-1),
            "label": torch.tensor([0, 0, 1, 1]),
            "image_id": ["same", "same", "other_a", "other_b"],
        }
        loss_cfg = {
            "use_infonce": False,
            "use_mse": False,
            "use_cls": False,
            "use_similarity_distill": False,
            "use_prototype_alignment": False,
            "use_aug_consistency": False,
            "use_supcon": True,
            "use_same_image_subject": True,
            "lambda_supcon": 0.2,
            "lambda_same_image_subject": 0.2,
        }

        total, parts = compute_alignment_loss(model, batch, torch.device("cpu"), loss_cfg, set())

        self.assertGreater(float(total.detach()), 0.0)
        self.assertIn("supcon", parts)
        self.assertIn("same_image_subject", parts)

    def test_compute_alignment_loss_includes_hard_negative_term(self) -> None:
        model = _DummyAlignmentModel()
        batch = {
            "eeg": torch.randn(4, 4, 6),
            "clip_emb": torch.nn.functional.normalize(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.95, 0.05, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.95, 0.05, 0.0],
                    ]
                ),
                dim=-1,
            ),
            "label": torch.tensor([0, 1, 2, 3]),
            "image_id": ["a", "b", "c", "d"],
        }
        loss_cfg = {
            "use_infonce": False,
            "use_mse": False,
            "use_cls": False,
            "use_similarity_distill": False,
            "use_prototype_alignment": False,
            "use_aug_consistency": False,
            "use_same_image_subject": False,
            "use_hard_negative": True,
            "lambda_hard_negative": 0.1,
        }

        total, parts = compute_alignment_loss(model, batch, torch.device("cpu"), loss_cfg, set())

        self.assertGreaterEqual(float(total.detach()), 0.0)
        self.assertIn("hard_negative", parts)


if __name__ == "__main__":
    unittest.main()
