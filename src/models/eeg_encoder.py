from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.models.masked_eeg_autoencoder import MaskedEEGAutoencoder


class EEGEncoder(nn.Module):
    """Lightweight EEG encoder for tensors shaped [B, channels, timesteps]."""

    def __init__(
        self,
        channels: int = 64,
        timesteps: int = 250,
        output_dim: int = 512,
        conv_dim: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.timesteps = timesteps
        self.output_dim = output_dim

        self.conv = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, conv_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(conv_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_dim,
            nhead=transformer_heads,
            dim_feedforward=conv_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.mlp = nn.Sequential(
            nn.LayerNorm(conv_dim),
            nn.Linear(conv_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3:
            raise ValueError(f"eeg must have shape [B, C, T], got {tuple(eeg.shape)}")
        if eeg.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} EEG channels, got {eeg.shape[1]}")
        if eeg.shape[2] != self.timesteps:
            raise ValueError(f"expected {self.timesteps} EEG timesteps, got {eeg.shape[2]}")

        x = self.conv(eeg)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.mlp(x)


class AttentionPool1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class EEGNetEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        timesteps: int = 250,
        output_dim: int = 512,
        temporal_filters: int = 32,
        depth_multiplier: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.timesteps = timesteps
        spatial_filters = temporal_filters * depth_multiplier
        self.net = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, 31), padding=(0, 15), bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                kernel_size=(channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(spatial_filters, spatial_filters, kernel_size=(1, 15), padding=(0, 7), groups=spatial_filters, bias=False),
            nn.Conv2d(spatial_filters, spatial_filters * 2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(spatial_filters * 2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        _validate_eeg(eeg, self.channels, self.timesteps)
        return self.projector(self.net(eeg.unsqueeze(1)))


class ResidualTCNBlock(nn.Module):
    def __init__(self, dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * 3
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=7, padding=padding, dilation=dilation),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class MultiScaleTemporalConvEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        timesteps: int = 250,
        output_dim: int = 512,
        branch_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.timesteps = timesteps
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels, branch_dim, kernel_size=kernel, padding=kernel // 2),
                    nn.BatchNorm1d(branch_dim),
                    nn.GELU(),
                )
                for kernel in [3, 7, 15, 31]
            ]
        )
        self.mixer = nn.Sequential(
            nn.Conv1d(branch_dim * 4, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.tcn = nn.Sequential(*(ResidualTCNBlock(hidden_dim, dilation, dropout) for dilation in [1, 2, 4, 8]))
        self.pool = AttentionPool1d(hidden_dim)
        self.projector = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim), nn.GELU(), nn.LayerNorm(output_dim))

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        _validate_eeg(eeg, self.channels, self.timesteps)
        x = torch.cat([branch(eeg) for branch in self.branches], dim=1)
        x = self.tcn(self.mixer(x)).transpose(1, 2)
        return self.projector(self.pool(x))


class ConvTransformerEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        timesteps: int = 250,
        output_dim: int = 512,
        hidden_dim: int = 256,
        layers: int = 4,
        heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.2,
        multiscale: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.timesteps = timesteps
        if multiscale:
            branch_dim = hidden_dim // 4
            self.stem = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(channels, branch_dim, kernel_size=kernel, stride=2, padding=kernel // 2),
                        nn.BatchNorm1d(branch_dim),
                        nn.GELU(),
                    )
                    for kernel in [3, 7, 15, 31]
                ]
            )
            stem_dim = branch_dim * 4
        else:
            self.stem = nn.Sequential(
                nn.Conv1d(channels, hidden_dim, kernel_size=9, stride=2, padding=4),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            )
            stem_dim = hidden_dim
        self.input_proj = nn.Linear(stem_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, max(1, timesteps), hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.pool = AttentionPool1d(hidden_dim)
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        _validate_eeg(eeg, self.channels, self.timesteps)
        if isinstance(self.stem, nn.ModuleList):
            x = torch.cat([branch(eeg) for branch in self.stem], dim=1)
        else:
            x = self.stem(eeg)
        x = self.input_proj(x.transpose(1, 2))
        x = x + self.pos[:, : x.shape[1], :]
        x = self.transformer(x)
        return self.projector(self.pool(x))


class SubjectAdaptiveEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        timesteps: int = 250,
        output_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        num_subjects: int = 16,
    ) -> None:
        super().__init__()
        self.base = ConvTransformerEncoder(
            channels=channels,
            timesteps=timesteps,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            layers=4,
            heads=8,
            ffn_dim=1024,
            dropout=max(dropout, 0.2),
        )
        self.num_subjects = max(1, int(num_subjects))
        self.subject_adapter = nn.Embedding(self.num_subjects, output_dim * 2)
        nn.init.zeros_(self.subject_adapter.weight)
        with torch.no_grad():
            self.subject_adapter.weight[:, :output_dim].fill_(1.0)

    def forward(self, eeg: torch.Tensor, subject_ids: list[str] | torch.Tensor | None = None) -> torch.Tensor:
        features = self.base(eeg)
        if subject_ids is None:
            return features
        indices = self._subject_indices(subject_ids, features.device)
        gamma_beta = self.subject_adapter(indices)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma * features + beta

    def _subject_indices(self, subject_ids: list[str] | torch.Tensor, device: torch.device) -> torch.Tensor:
        if isinstance(subject_ids, torch.Tensor):
            values = subject_ids.to(device=device, dtype=torch.long)
        else:
            parsed: list[int] = []
            for subject in subject_ids:
                text = str(subject)
                digits = "".join(ch for ch in text if ch.isdigit())
                parsed.append(int(digits) if digits else 0)
            values = torch.tensor(parsed, dtype=torch.long, device=device)
        return values.clamp_min(0).remainder(self.num_subjects)


class SpectrogramCNNEncoder(nn.Module):
    """Time-frequency EEG branch using a differentiable STFT front-end."""

    def __init__(
        self,
        channels: int = 64,
        timesteps: int = 250,
        output_dim: int = 512,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        n_fft: int = 64,
        hop_length: int = 16,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.timesteps = timesteps
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        _validate_eeg(eeg, self.channels, self.timesteps)
        batch, channels, timesteps = eeg.shape
        # STFT is more stable in fp32; downstream CNN still benefits from autocast.
        flat = eeg.float().reshape(batch * channels, timesteps)
        spec = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(device=eeg.device, dtype=torch.float32),
            center=True,
            return_complex=True,
        )
        spec = torch.log1p(spec.abs()).reshape(batch, channels, spec.shape[-2], spec.shape[-1])
        mean = spec.mean(dim=(2, 3), keepdim=True)
        std = spec.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        spec = (spec - mean) / std
        return self.projector(self.cnn(spec))


class RawSpectrogramFusionEncoder(nn.Module):
    """Late-fuse raw temporal and spectrogram EEG features."""

    def __init__(
        self,
        channels: int = 64,
        timesteps: int = 250,
        output_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.raw = ConvTransformerEncoder(
            channels=channels,
            timesteps=timesteps,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            layers=4,
            heads=8,
            ffn_dim=1024,
            dropout=max(dropout, 0.2),
        )
        self.spectrogram = SpectrogramCNNEncoder(
            channels=channels,
            timesteps=timesteps,
            output_dim=output_dim,
            hidden_dim=max(64, hidden_dim // 2),
            dropout=max(dropout, 0.2),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(output_dim * 2),
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        raw = self.raw(eeg)
        spec = self.spectrogram(eeg)
        return self.fusion(torch.cat([raw, spec], dim=-1))


class DualBranchEEGConformer(nn.Module):
    """Dual temporal/spatial EEG encoder used for the heavy-stage architecture sweep."""

    def __init__(
        self,
        channels: int = 64,
        timesteps: int = 250,
        output_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        num_subjects: int = 16,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.timesteps = timesteps
        self.temporal = MultiScaleTemporalConvEncoder(
            channels=channels,
            timesteps=timesteps,
            output_dim=output_dim,
            branch_dim=max(32, hidden_dim // 4),
            hidden_dim=hidden_dim,
            dropout=max(dropout, 0.2),
        )
        self.spatial_proj = nn.Linear(timesteps, hidden_dim)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=max(dropout, 0.2),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=2)
        self.channel_attention = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.subject_adapter = nn.Embedding(max(1, int(num_subjects)), output_dim * 2)
        nn.init.zeros_(self.subject_adapter.weight)
        with torch.no_grad():
            self.subject_adapter.weight[:, :output_dim].fill_(1.0)
        self.fusion = nn.Sequential(
            nn.LayerNorm(output_dim + hidden_dim),
            nn.Linear(output_dim + hidden_dim, output_dim),
            nn.GELU(),
            nn.Dropout(max(dropout, 0.2)),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, eeg: torch.Tensor, subject_ids: list[str] | torch.Tensor | None = None) -> torch.Tensor:
        _validate_eeg(eeg, self.channels, self.timesteps)
        temporal = self.temporal(eeg)
        spatial_tokens = self.spatial_proj(eeg)
        spatial_tokens = self.spatial_transformer(spatial_tokens)
        weights = torch.softmax(self.channel_attention(spatial_tokens).squeeze(-1), dim=-1)
        spatial = torch.sum(spatial_tokens * weights.unsqueeze(-1), dim=1)
        fused = self.fusion(torch.cat([temporal, spatial], dim=-1))
        if subject_ids is None:
            return fused
        indices = _subject_indices(subject_ids, fused.device, self.subject_adapter.num_embeddings)
        gamma, beta = self.subject_adapter(indices).chunk(2, dim=-1)
        return gamma * fused + beta


class TemporalSpectralSpatialTransformer(nn.Module):
    """Temporal, spectral, and channel-spatial branches with late transformer fusion."""

    def __init__(
        self,
        channels: int = 64,
        timesteps: int = 250,
        output_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.timesteps = timesteps
        self.temporal = ConvTransformerEncoder(
            channels=channels,
            timesteps=timesteps,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            layers=4,
            heads=8,
            ffn_dim=hidden_dim * 4,
            dropout=max(dropout, 0.2),
            multiscale=True,
        )
        self.spectral = SpectrogramCNNEncoder(
            channels=channels,
            timesteps=timesteps,
            output_dim=output_dim,
            hidden_dim=max(64, hidden_dim // 2),
            dropout=max(dropout, 0.2),
        )
        self.spatial = nn.Sequential(
            nn.Linear(timesteps, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.spatial_pool = AttentionPool1d(hidden_dim)
        self.spatial_out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim), nn.GELU(), nn.LayerNorm(output_dim))
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=8,
            dim_feedforward=output_dim * 2,
            dropout=max(dropout, 0.2),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(fusion_layer, num_layers=2)
        self.branch_type = nn.Parameter(torch.zeros(1, 3, output_dim))
        self.pool = AttentionPool1d(output_dim)
        self.projector = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(max(dropout, 0.2)),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        _validate_eeg(eeg, self.channels, self.timesteps)
        temporal = self.temporal(eeg)
        spectral = self.spectral(eeg)
        spatial = self.spatial_out(self.spatial_pool(self.spatial(eeg)))
        tokens = torch.stack([temporal, spectral, spatial], dim=1) + self.branch_type
        tokens = self.fusion_transformer(tokens)
        return self.projector(self.pool(tokens))


class SubjectAdaptiveGraphEncoder(nn.Module):
    """Subject-adaptive channel graph encoder for cross-subject EEG alignment."""

    def __init__(
        self,
        channels: int = 64,
        timesteps: int = 250,
        output_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        num_subjects: int = 16,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.timesteps = timesteps
        self.channel_proj = nn.Linear(timesteps, hidden_dim)
        self.adjacency_logits = nn.Parameter(torch.zeros(channels, channels))
        self.graph_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(max(dropout, 0.2)),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=max(dropout, 0.2),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.graph_transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.pool = AttentionPool1d(hidden_dim)
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Dropout(max(dropout, 0.2)),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.subject_adapter = nn.Embedding(max(1, int(num_subjects)), output_dim * 2)
        nn.init.zeros_(self.subject_adapter.weight)
        with torch.no_grad():
            self.subject_adapter.weight[:, :output_dim].fill_(1.0)

    def forward(self, eeg: torch.Tensor, subject_ids: list[str] | torch.Tensor | None = None) -> torch.Tensor:
        _validate_eeg(eeg, self.channels, self.timesteps)
        nodes = self.channel_proj(eeg)
        adjacency = torch.softmax(self.adjacency_logits, dim=-1)
        mixed = torch.einsum("cd,bdh->bch", adjacency, nodes)
        nodes = self.graph_transformer(nodes + self.graph_mlp(mixed))
        features = self.projector(self.pool(nodes))
        if subject_ids is None:
            return features
        indices = _subject_indices(subject_ids, features.device, self.subject_adapter.num_embeddings)
        gamma, beta = self.subject_adapter(indices).chunk(2, dim=-1)
        return gamma * features + beta


class MaskedPretrainedEEGEncoder(nn.Module):
    """Encoder wrapper for MaskedEEGAutoencoder checkpoints."""

    def __init__(
        self,
        channels: int = 64,
        timesteps: int = 250,
        output_dim: int = 512,
        hidden_dim: int = 512,
        layers: int = 8,
        heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.15,
        spatial_layers: int = 2,
        variant: str = "dualbranch",
    ) -> None:
        super().__init__()
        self.autoencoder = MaskedEEGAutoencoder(
            channels=channels,
            timesteps=timesteps,
            hidden_dim=hidden_dim,
            layers=layers,
            heads=heads,
            ffn_dim=int(ffn_dim or hidden_dim * 4),
            dropout=dropout,
            spatial_layers=spatial_layers,
            variant=variant,
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.projector(self.autoencoder.encode(eeg))


def _validate_eeg(eeg: torch.Tensor, channels: int, timesteps: int) -> None:
    if eeg.ndim != 3:
        raise ValueError(f"eeg must have shape [B, C, T], got {tuple(eeg.shape)}")
    if eeg.shape[1] != channels:
        raise ValueError(f"expected {channels} EEG channels, got {eeg.shape[1]}")
    if eeg.shape[2] != timesteps:
        raise ValueError(f"expected {timesteps} EEG timesteps, got {eeg.shape[2]}")


def _subject_indices(subject_ids: list[str] | torch.Tensor, device: torch.device, num_subjects: int) -> torch.Tensor:
    if isinstance(subject_ids, torch.Tensor):
        values = subject_ids.to(device=device, dtype=torch.long)
    else:
        parsed: list[int] = []
        for subject in subject_ids:
            text = str(subject)
            digits = "".join(ch for ch in text if ch.isdigit())
            parsed.append(int(digits) if digits else 0)
        values = torch.tensor(parsed, dtype=torch.long, device=device)
    return values.clamp_min(0).remainder(max(1, int(num_subjects)))


def build_eeg_encoder(
    encoder_type: str = "tiny",
    *,
    channels: int = 64,
    timesteps: int = 250,
    output_dim: int = 512,
    hidden_dim: int = 128,
    transformer_layers: int = 2,
    dropout: float = 0.1,
    num_subjects: int = 16,
) -> nn.Module:
    key = encoder_type.lower()
    if key in {"tiny", "e0", "baseline"}:
        return EEGEncoder(
            channels=channels,
            timesteps=timesteps,
            output_dim=output_dim,
            conv_dim=hidden_dim,
            transformer_layers=transformer_layers,
            dropout=dropout,
        )
    if key in {"eegnet", "e1"}:
        return EEGNetEncoder(channels=channels, timesteps=timesteps, output_dim=output_dim, dropout=max(dropout, 0.2))
    if key in {"multiscale_tcn", "e2"}:
        return MultiScaleTemporalConvEncoder(channels=channels, timesteps=timesteps, output_dim=output_dim, hidden_dim=256, dropout=max(dropout, 0.2))
    if key in {"convtransformer_base", "e3", "base"}:
        return ConvTransformerEncoder(
            channels=channels,
            timesteps=timesteps,
            output_dim=output_dim,
            hidden_dim=256,
            layers=4,
            heads=8,
            ffn_dim=1024,
            dropout=max(dropout, 0.2),
        )
    if key in {"convtransformer_strong", "e4", "strong"}:
        return ConvTransformerEncoder(
            channels=channels,
            timesteps=timesteps,
            output_dim=output_dim,
            hidden_dim=384,
            layers=6,
            heads=8,
            ffn_dim=1536,
            dropout=max(dropout, 0.3),
            multiscale=True,
        )
    if key in {"subject_adaptive", "e5"}:
        return SubjectAdaptiveEncoder(
            channels=channels,
            timesteps=timesteps,
            output_dim=output_dim,
            hidden_dim=256,
            dropout=max(dropout, 0.2),
            num_subjects=num_subjects,
        )
    if key in {"spectrogram_cnn", "spectrogram", "e7"}:
        return SpectrogramCNNEncoder(
            channels=channels,
            timesteps=timesteps,
            output_dim=output_dim,
            hidden_dim=max(64, hidden_dim),
            dropout=max(dropout, 0.2),
        )
    if key in {"raw_spectrogram_fusion", "e3_e7", "p2"}:
        return RawSpectrogramFusionEncoder(
            channels=channels,
            timesteps=timesteps,
            output_dim=output_dim,
            hidden_dim=256,
            dropout=max(dropout, 0.2),
        )
    if key in {"dualbranch_eegconformer", "dual_branch_eegconformer", "a1"}:
        return DualBranchEEGConformer(
            channels=channels,
            timesteps=timesteps,
            output_dim=output_dim,
            hidden_dim=max(256, hidden_dim),
            dropout=max(dropout, 0.2),
            num_subjects=num_subjects,
        )
    if key in {"temporal_spectral_spatial", "temporal_spectral_spatial_transformer", "tsst", "a2"}:
        return TemporalSpectralSpatialTransformer(
            channels=channels,
            timesteps=timesteps,
            output_dim=output_dim,
            hidden_dim=max(256, hidden_dim),
            dropout=max(dropout, 0.2),
        )
    if key in {"subject_adaptive_graph", "subject_graph", "a3"}:
        return SubjectAdaptiveGraphEncoder(
            channels=channels,
            timesteps=timesteps,
            output_dim=output_dim,
            hidden_dim=max(256, hidden_dim),
            dropout=max(dropout, 0.2),
            num_subjects=num_subjects,
        )
    if key in {"masked_pretrained", "masked_autoencoder_encoder", "masked_eeg", "masked_pretrained_tsst", "masked_tsst", "masked_m2"}:
        return MaskedPretrainedEEGEncoder(
            channels=channels,
            timesteps=timesteps,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            layers=transformer_layers,
            dropout=dropout,
            variant="temporal_spectral_spatial" if key in {"masked_pretrained_tsst", "masked_tsst", "masked_m2"} else "dualbranch",
        )
    raise ValueError(f"Unknown EEG encoder type: {encoder_type}")
