from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.eval.semantic_fusion_classifier_eval import (
    compose_classifier_inputs,
    grouped_image_summary,
    label_mismatched_indices,
    maybe_write_embedded_eeg_encoder_checkpoint,
    summarize_records,
    write_gap_metrics,
    write_metrics,
)
from src.train import train_semantic_fusion
from src.train.train_semantic_fusion import parse_args


class SemanticFusionClassifierEvalTests(unittest.TestCase):
    def test_semantic_fusion_train_cli_accepts_seed(self) -> None:
        args = parse_args(["--seed", "123", "--epochs", "1"])

        self.assertEqual(args.seed, 123)
        self.assertEqual(args.epochs, 1)

    def test_semantic_fusion_train_cli_accepts_reliability_gate(self) -> None:
        args = parse_args(["--fusion_type", "gated", "--epochs", "1"])

        self.assertEqual(args.fusion_type, "gated")

    def test_semantic_fusion_train_cli_accepts_encoder_train_mode(self) -> None:
        args = parse_args(["--encoder_train_mode", "unfreeze_last2", "--encoder_lr", "1e-5"])

        self.assertEqual(args.encoder_train_mode, "unfreeze_last2")
        self.assertEqual(args.encoder_lr, 1e-5)

    def test_configure_eeg_encoder_train_mode_unfreezes_only_tail(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Linear(2, 2),
            torch.nn.Linear(2, 2),
        )

        trainable = train_semantic_fusion.configure_eeg_encoder_train_mode(model, "unfreeze_last2")

        self.assertTrue(trainable)
        params = list(model.parameters())
        self.assertFalse(params[0].requires_grad)
        self.assertFalse(params[1].requires_grad)
        self.assertTrue(params[-1].requires_grad)
        self.assertTrue(params[-2].requires_grad)

    def test_configure_eeg_encoder_train_mode_uses_alignment_encoder_tail_not_projector(self) -> None:
        class Wrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.eeg_encoder = torch.nn.Sequential(
                    torch.nn.Linear(2, 2),
                    torch.nn.Linear(2, 2),
                    torch.nn.Linear(2, 2),
                )
                self.projector = torch.nn.Linear(2, 2)
                self.classifier = torch.nn.Linear(2, 2)

        model = Wrapper()

        trainable = train_semantic_fusion.configure_eeg_encoder_train_mode(model, "unfreeze_last2")

        self.assertTrue(trainable)
        self.assertFalse(any(parameter.requires_grad for parameter in model.projector.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.classifier.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.eeg_encoder[0].parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.eeg_encoder[1].parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.eeg_encoder[2].parameters()))

    def test_eval_can_materialize_embedded_eeg_encoder_checkpoint(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            classifier_checkpoint = out / "semantic_fusion_classifier.pt"
            torch.save(
                {
                    "model": {},
                    "label_values": torch.tensor([0, 1]),
                    "uses_eeg": True,
                    "args": {"eeg_checkpoint": "source.pt"},
                    "eeg_encoder_model": {"weight": torch.ones(1)},
                    "eeg_encoder_source_checkpoint": "source.pt",
                },
                classifier_checkpoint,
            )

            embedded = maybe_write_embedded_eeg_encoder_checkpoint(classifier_checkpoint, torch.load(classifier_checkpoint, weights_only=False))

            self.assertIsNotNone(embedded)
            self.assertTrue(embedded.exists())
            payload = torch.load(embedded, weights_only=False)
            self.assertIn("model", payload)
            self.assertEqual(payload["config"]["model"], {})

    def test_reliability_gated_classifier_outputs_gate_values(self) -> None:
        model = train_semantic_fusion.ReliabilityGatedSemanticFusionClassifier(
            image_dim=4,
            hidden_dim=8,
            num_classes=3,
        )
        image = torch.randn(2, 4)
        eeg = torch.randn(2, 4)

        logits = model(image, eeg)
        gate = model.gate_values(image, eeg)

        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(gate.shape), (2, 1))
        self.assertTrue(torch.all(gate >= 0.0))
        self.assertTrue(torch.all(gate <= 1.0))

    def test_compose_inputs_uses_zero_eeg_for_vision_only_and_zero_image_for_eeg_only(self) -> None:
        image = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        eeg = torch.tensor([[1.0, 0.0], [0.0, 5.0]])

        vision_image, vision_eeg = compose_classifier_inputs(image, eeg, mode="vision_only", uses_eeg=True)
        real_image, real_eeg = compose_classifier_inputs(image, eeg, mode="real_eeg", uses_eeg=True)
        global_image, global_eeg = compose_classifier_inputs(image, eeg, mode="global_shuffled_eeg", uses_eeg=True)
        eeg_only_image, eeg_only_eeg = compose_classifier_inputs(image, eeg, mode="eeg_only", uses_eeg=True)

        self.assertTrue(torch.allclose(vision_image.norm(dim=-1), torch.ones(2)))
        self.assertTrue(torch.allclose(vision_eeg, torch.zeros_like(vision_eeg)))
        self.assertTrue(torch.allclose(real_image.norm(dim=-1), torch.ones(2)))
        self.assertTrue(torch.allclose(real_eeg.norm(dim=-1), torch.ones(2)))
        self.assertTrue(torch.allclose(global_image.norm(dim=-1), torch.ones(2)))
        self.assertTrue(torch.allclose(global_eeg.norm(dim=-1), torch.ones(2)))
        self.assertTrue(torch.allclose(eeg_only_image, torch.zeros_like(eeg_only_image)))
        self.assertTrue(torch.allclose(eeg_only_eeg.norm(dim=-1), torch.ones(2)))

    def test_summarize_records_reports_top1_top5_and_invalid_rate(self) -> None:
        records = [
            {"class_correct": 1.0, "top5_correct": 1.0, "prediction": "a photo of a piano"},
            {"class_correct": 0.0, "top5_correct": 1.0, "prediction": ""},
        ]

        summary = summarize_records(records, corruption="mixed", mode="real_eeg", file_name="mixed_real_eeg.jsonl")

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["top5_accuracy"], 1.0)
        self.assertEqual(summary["invalid_caption_rate"], 0.5)

    def test_summarize_records_reports_gate_mean_when_available(self) -> None:
        records = [
            {"class_correct": 1.0, "top5_correct": 1.0, "prediction": "a photo of a piano", "gate_mean": 0.2},
            {"class_correct": 0.0, "top5_correct": 1.0, "prediction": "a photo of a car", "gate_mean": 0.4},
        ]

        summary = summarize_records(records, corruption="mixed", mode="real_eeg", file_name="mixed_real_eeg.jsonl")

        self.assertAlmostEqual(summary["gate_mean"], 0.3, places=6)

    def test_label_mismatched_indices_never_select_same_label_when_possible(self) -> None:
        rows = [
            {"image_id": "img0", "label": 0},
            {"image_id": "img0", "label": 0},
            {"image_id": "img1", "label": 1},
            {"image_id": "img1", "label": 1},
            {"image_id": "img2", "label": 2},
        ]

        indices = label_mismatched_indices(rows)

        self.assertEqual(len(indices), len(rows))
        for row_index, selected_index in enumerate(indices):
            self.assertNotEqual(rows[row_index]["label"], rows[selected_index]["label"])

    def test_grouped_image_summary_counts_unique_images_with_majority_vote(self) -> None:
        records = [
            {"image_id": "img0", "label": 0, "pred_label": 0, "top5_labels": [0, 1], "prediction": "a photo of piano", "class_correct": 1.0, "top5_correct": 1.0},
            {"image_id": "img0", "label": 0, "pred_label": 1, "top5_labels": [1, 0], "prediction": "a photo of car", "class_correct": 0.0, "top5_correct": 1.0},
            {"image_id": "img0", "label": 0, "pred_label": 0, "top5_labels": [0, 2], "prediction": "a photo of piano", "class_correct": 1.0, "top5_correct": 1.0},
            {"image_id": "img1", "label": 1, "pred_label": 2, "top5_labels": [2, 3], "prediction": "a photo of chair", "class_correct": 0.0, "top5_correct": 0.0},
            {"image_id": "img1", "label": 1, "pred_label": 1, "top5_labels": [1, 2], "prediction": "a photo of car", "class_correct": 1.0, "top5_correct": 1.0},
            {"image_id": "img1", "label": 1, "pred_label": 1, "top5_labels": [1, 3], "prediction": "a photo of car", "class_correct": 1.0, "top5_correct": 1.0},
        ]

        summary = grouped_image_summary(records, corruption="mixed", mode="real_eeg", file_name="mixed_real_eeg.jsonl")

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["trial_count"], 6)
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["top5_accuracy"], 1.0)

    def test_write_metrics_accepts_late_fields_from_image_level_rows(self) -> None:
        rows = [
            {
                "file": "mixed_real_eeg.jsonl",
                "corruption": "mixed",
                "mode": "real_eeg",
                "count": 6,
                "accuracy": 0.5,
            },
            {
                "file": "mixed_real_eeg.jsonl",
                "corruption": "mixed",
                "mode": "real_eeg_image_level",
                "count": 2,
                "trial_count": 6,
                "accuracy": 1.0,
            },
        ]

        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_metrics(rows, out / "metrics.csv", out / "metrics.md")

            text = (out / "metrics.csv").read_text(encoding="utf-8")
            self.assertIn("trial_count", text.splitlines()[0])
            self.assertIn("real_eeg_image_level", text)

    def test_write_gap_metrics_writes_image_level_control_report(self) -> None:
        metrics = [
            {"corruption": "mixed", "mode": "vision_only", "accuracy": 0.10},
            {"corruption": "mixed", "mode": "real_eeg", "accuracy": 0.40},
            {"corruption": "mixed", "mode": "shuffled_eeg", "accuracy": 0.12},
            {"corruption": "mixed", "mode": "global_shuffled_eeg", "accuracy": 0.08},
            {"corruption": "mixed", "mode": "random_eeg", "accuracy": 0.09},
            {"corruption": "mixed", "mode": "vision_only_image_level", "accuracy": 0.15},
            {"corruption": "mixed", "mode": "real_eeg_image_level", "accuracy": 0.55},
            {"corruption": "mixed", "mode": "shuffled_eeg_image_level", "accuracy": 0.20},
            {"corruption": "mixed", "mode": "global_shuffled_eeg_image_level", "accuracy": 0.18},
            {"corruption": "mixed", "mode": "random_eeg_image_level", "accuracy": 0.16},
        ]

        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_gap_metrics(metrics, out)

            text = (out / "SEMANTIC_GAP_METRICS_IMAGE_LEVEL.csv").read_text(encoding="utf-8")
            self.assertIn("real_minus_global_shuffled", text)
            self.assertIn("0.370", text)
            self.assertIn("True", text)


if __name__ == "__main__":
    unittest.main()
