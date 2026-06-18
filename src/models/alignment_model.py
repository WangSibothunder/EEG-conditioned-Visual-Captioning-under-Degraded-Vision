from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.losses.contrastive import symmetric_info_nce
from src.models.eeg_encoder import build_eeg_encoder


class EEGCLIPAlignmentModel(nn.Module):
    def __init__(
        self,
        eeg_channels: int = 64,
        eeg_timesteps: int = 250,
        eeg_dim: int = 512,
        clip_dim: int = 512,
        hidden_dim: int = 128,
        transformer_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int | None = None,
        encoder_type: str = "tiny",
    ) -> None:
        super().__init__()
        self.eeg_encoder = build_eeg_encoder(
            encoder_type,
            channels=eeg_channels,
            timesteps=eeg_timesteps,
            output_dim=eeg_dim,
            hidden_dim=hidden_dim,
            transformer_layers=transformer_layers,
            dropout=dropout,
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(eeg_dim),
            nn.Linear(eeg_dim, clip_dim),
        )
        self.classifier = nn.Linear(clip_dim, num_classes) if num_classes else None

    def forward(self, eeg: torch.Tensor, subject_ids: list[str] | torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        try:
            eeg_emb = self.eeg_encoder(eeg, subject_ids=subject_ids)
        except TypeError:
            eeg_emb = self.eeg_encoder(eeg)
        clip_pred = F.normalize(self.projector(eeg_emb), dim=-1)
        logits = self.classifier(clip_pred) if self.classifier is not None else None
        return clip_pred, logits


def info_nce_loss(eeg_emb: torch.Tensor, image_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    return symmetric_info_nce(eeg_emb, image_emb, temperature=temperature)
