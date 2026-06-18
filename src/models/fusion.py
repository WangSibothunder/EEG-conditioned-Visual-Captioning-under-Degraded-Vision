from __future__ import annotations

import torch
from torch import nn


class GatedFusion(nn.Module):
    """Inject EEG information as a gated residual delta on image embeddings."""

    def __init__(self, dim: int = 512, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim
        hidden = hidden_dim or dim * 2
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )
        self.eeg_delta = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, image_emb: torch.Tensor, eeg_emb: torch.Tensor | None = None) -> torch.Tensor:
        if image_emb.ndim != 2 or image_emb.shape[-1] != self.dim:
            raise ValueError(f"image_emb must have shape [B, {self.dim}], got {tuple(image_emb.shape)}")
        if eeg_emb is None:
            return image_emb
        if eeg_emb.ndim != 2 or eeg_emb.shape != image_emb.shape:
            raise ValueError(
                f"eeg_emb must have shape {tuple(image_emb.shape)}, got {tuple(eeg_emb.shape)}"
            )

        gate = self.gate(torch.cat([image_emb, eeg_emb], dim=-1))
        eeg_delta = self.eeg_delta(eeg_emb)
        return image_emb + gate * eeg_delta

    def gate_values(self, image_emb: torch.Tensor, eeg_emb: torch.Tensor) -> torch.Tensor:
        if image_emb.ndim != 2 or image_emb.shape[-1] != self.dim:
            raise ValueError(f"image_emb must have shape [B, {self.dim}], got {tuple(image_emb.shape)}")
        if eeg_emb.ndim != 2 or eeg_emb.shape != image_emb.shape:
            raise ValueError(
                f"eeg_emb must have shape {tuple(image_emb.shape)}, got {tuple(eeg_emb.shape)}"
            )
        return self.gate(torch.cat([image_emb, eeg_emb], dim=-1))

    def image_only(self, image_emb: torch.Tensor) -> torch.Tensor:
        return self.forward(image_emb, None)
