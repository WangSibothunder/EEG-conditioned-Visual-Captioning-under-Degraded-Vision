from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MaskedEEGAutoencoder(nn.Module):
    """Heavy masked EEG reconstruction model for [B, C, T] EEG tensors."""

    def __init__(
        self,
        channels: int = 64,
        timesteps: int = 250,
        hidden_dim: int = 512,
        layers: int = 8,
        heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.15,
        spatial_layers: int = 2,
        variant: str = "dualbranch",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.timesteps = timesteps
        self.hidden_dim = hidden_dim
        self.variant = variant

        self.temporal_stem = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=8, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        token_count = (timesteps + 1) // 2
        self.pos = nn.Parameter(torch.zeros(1, token_count, hidden_dim))

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=layers)

        self.channel_embed = nn.Linear(timesteps, hidden_dim)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=max(ffn_dim // 2, hidden_dim * 2),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=spatial_layers)
        self.spatial_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.use_spectral_branch = variant.lower() in {
            "temporal_spectral_spatial",
            "m2_temporal_spectral_spatial",
            "tsst",
        }
        if self.use_spectral_branch:
            self.spectral_branch = nn.Sequential(
                nn.Linear((timesteps // 2) + 1, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.spectral_gate = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
        else:
            self.spectral_branch = None
            self.spectral_gate = None

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=5, padding=2, groups=8, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, channels, kernel_size=1),
        )

    def _encode_tokens(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3:
            raise ValueError(f"eeg must have shape [B, C, T], got {tuple(eeg.shape)}")
        if eeg.shape[1] != self.channels or eeg.shape[2] != self.timesteps:
            raise ValueError(f"expected EEG shape [B, {self.channels}, {self.timesteps}], got {tuple(eeg.shape)}")

        temporal = self.temporal_stem(eeg).transpose(1, 2)
        temporal = temporal + self.pos[:, : temporal.shape[1], :]

        spatial = self.channel_embed(eeg)
        spatial = self.spatial_encoder(spatial).mean(dim=1)
        temporal = temporal * (1.0 + self.spatial_gate(spatial).unsqueeze(1))
        if self.spectral_branch is not None and self.spectral_gate is not None:
            spectrum = torch.fft.rfft(eeg.float(), dim=-1).abs().log1p().mean(dim=1)
            spectral = self.spectral_branch(spectrum)
            gate = self.spectral_gate(torch.cat([spatial, spectral], dim=-1)).unsqueeze(1)
            temporal = temporal * (1.0 + gate) + spectral.unsqueeze(1)

        return self.temporal_encoder(temporal)

    def encode(self, eeg: torch.Tensor) -> torch.Tensor:
        return self._encode_tokens(eeg).mean(dim=1)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        encoded = self._encode_tokens(eeg).transpose(1, 2)
        recon = self.decoder(encoded)
        if recon.shape[-1] != self.timesteps:
            recon = F.interpolate(recon, size=self.timesteps, mode="linear", align_corners=False)
        return recon


def masked_eeg_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    lambda_spectral: float = 0.2,
    lambda_smoothness: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    diff = (reconstruction - target).float()
    if mask.any():
        masked_mse = diff[mask].pow(2).mean()
    else:
        masked_mse = diff.pow(2).mean()

    recon_spec = torch.fft.rfft(reconstruction.float(), dim=-1).abs()
    target_spec = torch.fft.rfft(target.float(), dim=-1).abs()
    spectral = F.l1_loss(torch.log1p(recon_spec), torch.log1p(target_spec))

    recon_delta = reconstruction[:, :, 1:] - reconstruction[:, :, :-1]
    target_delta = target[:, :, 1:] - target[:, :, :-1]
    smoothness = F.mse_loss(recon_delta.float(), target_delta.float())

    total = masked_mse + lambda_spectral * spectral + lambda_smoothness * smoothness
    parts = {
        "masked_mse": float(masked_mse.detach().cpu()),
        "spectral": float(spectral.detach().cpu()),
        "smoothness": float(smoothness.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, parts


def make_time_channel_mask(
    eeg: torch.Tensor,
    *,
    mask_ratio_time: float = 0.35,
    mask_ratio_channel: float = 0.15,
    span: int = 12,
) -> torch.Tensor:
    batch, channels, timesteps = eeg.shape
    device = eeg.device
    mask = torch.zeros((batch, channels, timesteps), dtype=torch.bool, device=device)

    n_time = max(1, int(round(timesteps * mask_ratio_time)))
    n_spans = max(1, n_time // max(1, span))
    for b_idx in range(batch):
        starts = torch.randint(0, max(1, timesteps - span + 1), (n_spans,), device=device)
        for start in starts.tolist():
            mask[b_idx, :, start : min(timesteps, start + span)] = True

    channel_mask = torch.rand((batch, channels), device=device) < mask_ratio_channel
    mask |= channel_mask.unsqueeze(-1)
    return mask


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
