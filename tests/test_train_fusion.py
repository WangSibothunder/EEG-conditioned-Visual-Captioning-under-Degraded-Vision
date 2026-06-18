from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import torch

from src.models.eeg_encoder import EEGEncoder
from src.train.train_fusion import (
    apply_cli_overrides,
    caption_checkpoint_state,
    evaluate_caption_loss,
    load_alignment_eeg_encoder,
    resolve_output_dir,
)


class TrainFusionTests(unittest.TestCase):
    def test_resolve_output_dir_can_use_cli_path_exactly(self) -> None:
        self.assertEqual(resolve_output_dir({"output_dir": "outputs/debug"}), Path("outputs/debug/fusion"))
        self.assertEqual(
            resolve_output_dir({"output_dir": "outputs/overnight/fusion_qwen15", "exact_output_dir": True}),
            Path("outputs/overnight/fusion_qwen15"),
        )

    def test_load_alignment_eeg_encoder_uses_eeg_encoder_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = EEGEncoder()
            source = EEGEncoder()
            for parameter in source.parameters():
                torch.nn.init.constant_(parameter, 0.123)
            checkpoint = Path(tmpdir) / "align.pt"
            torch.save({"eeg_encoder": source.state_dict()}, checkpoint)

            load_alignment_eeg_encoder(encoder, checkpoint, torch.device("cpu"))

        for parameter in encoder.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 0.123))

    def test_cli_llm_override_disables_tiny_debug_model(self) -> None:
        cfg = {
            "data": {},
            "model": {"use_tiny_debug_model": True},
            "train": {},
            "output_dir": "outputs/debug",
        }
        args = SimpleNamespace(
            train_manifest=None,
            val_manifest=None,
            root=None,
            clip_train_cache=None,
            clip_val_cache=None,
            eeg_ckpt=None,
            llm="Qwen/Qwen2.5-1.5B-Instruct",
            freeze_llm="true",
            freeze_eeg_encoder="true",
            epochs=None,
            batch_size=None,
            grad_accum_steps=None,
            max_steps=None,
            max_val_batches=None,
            train_mode=None,
            bf16=None,
            output_dir="outputs/overnight/fusion_qwen15",
        )

        apply_cli_overrides(cfg, args)

        self.assertEqual(cfg["model"]["llm_model"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertFalse(cfg["model"]["use_tiny_debug_model"])
        self.assertTrue(cfg["model"]["require_real_lm"])
        self.assertTrue(cfg["train"]["freeze_eeg_encoder"])
        self.assertEqual(cfg["train"]["max_steps"], 0)
        self.assertTrue(cfg["exact_output_dir"])

    def test_cli_train_mode_override_sets_training_mode(self) -> None:
        cfg = {
            "data": {},
            "model": {},
            "train": {},
            "output_dir": "outputs/debug",
        }
        args = SimpleNamespace(
            train_manifest=None,
            val_manifest=None,
            root=None,
            clip_train_cache=None,
            clip_val_cache=None,
            eeg_ckpt=None,
            llm=None,
            freeze_llm=None,
            freeze_eeg_encoder=None,
            epochs=None,
            batch_size=None,
            grad_accum_steps=None,
            max_steps=None,
            max_val_batches=None,
            train_mode="vision_only",
            bf16=None,
            output_dir=None,
        )

        apply_cli_overrides(cfg, args)

        self.assertEqual(cfg["train"]["mode"], "vision_only")

    def test_evaluate_caption_loss_respects_vision_only_mode(self) -> None:
        class RaisingEegEncoder(torch.nn.Module):
            def forward(self, eeg: torch.Tensor) -> torch.Tensor:
                raise AssertionError("vision_only evaluation should not encode EEG")

        class FusionProbe(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.used_image_only = False

            def forward(self, image_emb: torch.Tensor, eeg_emb: torch.Tensor | None = None) -> torch.Tensor:
                if eeg_emb is not None:
                    raise AssertionError("vision_only evaluation should not fuse EEG")
                return image_emb

            def image_only(self, image_emb: torch.Tensor) -> torch.Tensor:
                self.used_image_only = True
                return image_emb

        class CaptionProbe(torch.nn.Module):
            def forward(self, fused_emb: torch.Tensor, captions: list[str]) -> torch.Tensor:
                return fused_emb.sum() * 0 + torch.tensor(1.0, device=fused_emb.device)

        fusion = FusionProbe()
        batch = {
            "eeg": torch.zeros(2, 64, 250),
            "clip_emb": torch.ones(2, 512),
            "caption": ["a photo of a canoe", "a photo of a guitar"],
        }

        loss = evaluate_caption_loss(
            vision=torch.nn.Identity(),
            eeg_encoder=RaisingEegEncoder(),
            fusion=fusion,
            caption_model=CaptionProbe(),
            loader=[batch],
            device=torch.device("cpu"),
            amp_dtype=torch.float32,
            use_amp=False,
            freeze_eeg_encoder=True,
            max_batches=0,
            mode="vision_only",
        )

        self.assertEqual(loss, 1.0)
        self.assertTrue(fusion.used_image_only)

    def test_caption_checkpoint_state_omits_frozen_lm_weights(self) -> None:
        class CaptionLike(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lm = torch.nn.Linear(2, 2)
                self.prompt_projector = torch.nn.Linear(2, 2)
                for parameter in self.lm.parameters():
                    parameter.requires_grad_(False)

        state = caption_checkpoint_state(CaptionLike())  # type: ignore[arg-type]

        self.assertTrue(state)
        self.assertTrue(all(key.startswith("prompt_projector.") for key in state))
        self.assertFalse(any(key.startswith("lm.") for key in state))


if __name__ == "__main__":
    unittest.main()
