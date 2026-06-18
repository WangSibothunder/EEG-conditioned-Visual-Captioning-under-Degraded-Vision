from __future__ import annotations

import unittest

import torch

from src.losses.contrastive import (
    multi_positive_info_nce,
    prototype_alignment_loss,
    same_image_subject_consistency_loss,
    supervised_contrastive_loss,
    symmetric_info_nce,
)
from src.losses.eeg_aug import augment_eeg
from src.losses.similarity import similarity_distillation_loss


class AlignmentLossTests(unittest.TestCase):
    def test_symmetric_info_nce_prefers_matching_pairs(self) -> None:
        emb = torch.nn.functional.normalize(torch.eye(4), dim=-1)
        matched = symmetric_info_nce(emb, emb, temperature=0.07)
        mismatched = symmetric_info_nce(emb, emb.flip(0), temperature=0.07)

        self.assertLess(float(matched), float(mismatched))

    def test_multi_positive_info_nce_does_not_penalize_duplicate_image_ids(self) -> None:
        eeg = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        image = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        strict = symmetric_info_nce(eeg, image, temperature=0.07)
        multi = multi_positive_info_nce(eeg, image, ["same", "same", "other"], temperature=0.07)

        self.assertLess(float(multi), float(strict))

    def test_multi_positive_info_nce_uses_same_label_weak_positives(self) -> None:
        eeg = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        image = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        strict = multi_positive_info_nce(eeg, image, ["a", "b", "c"], temperature=0.07)
        weak = multi_positive_info_nce(
            eeg,
            image,
            ["a", "b", "c"],
            labels=torch.tensor([0, 0, 1]),
            temperature=0.07,
        )

        self.assertLess(float(weak), float(strict))

    def test_prototype_alignment_is_low_when_matching_class_prototypes(self) -> None:
        image = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
            dim=-1,
        )
        eeg = image.clone()
        labels = torch.tensor([0, 0, 1, 1])

        loss = prototype_alignment_loss(eeg, image, labels)

        self.assertLess(float(loss), 1e-5)

    def test_supervised_contrastive_prefers_same_class_neighbors(self) -> None:
        emb = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
            dim=-1,
        )
        labels = torch.tensor([0, 0, 1, 1])
        loss = supervised_contrastive_loss(emb, labels, temperature=0.1)

        self.assertLess(float(loss), 0.5)

    def test_same_image_subject_consistency_uses_repeated_image_ids(self) -> None:
        emb = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]), dim=-1)
        loss = same_image_subject_consistency_loss(emb, ["same", "same", "other"])

        self.assertLess(float(loss), 0.1)

    def test_same_image_subject_consistency_can_require_different_subjects(self) -> None:
        emb = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]), dim=-1)
        same_subject = same_image_subject_consistency_loss(emb[:2], ["same", "same"], subject_ids=["S1", "S1"])
        cross_subject = same_image_subject_consistency_loss(emb, ["same", "same", "other"], subject_ids=["S1", "S2", "S1"])

        self.assertEqual(float(same_subject), 0.0)
        self.assertGreater(float(cross_subject), 0.5)

    def test_similarity_distillation_is_small_for_identical_embeddings(self) -> None:
        emb = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        loss = similarity_distillation_loss(emb, emb, tau=0.1)

        self.assertLess(float(loss), 1e-5)

    def test_eeg_augmentation_preserves_shape(self) -> None:
        eeg = torch.randn(3, 64, 250)
        aug = augment_eeg(eeg)

        self.assertEqual(tuple(aug.shape), tuple(eeg.shape))
        self.assertTrue(torch.isfinite(aug).all())


if __name__ == "__main__":
    unittest.main()
