from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class _TinyVisionEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class FrozenCLIPVisionEncoder(nn.Module):
    """Frozen CLIP vision wrapper with an offline-friendly tiny fallback."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        model_name: str | None = None,
        output_dim: int | None = None,
        use_tiny_debug_model: bool | None = None,
    ) -> None:
        super().__init__()
        cfg = dict(config or {})
        self.model_name = model_name or str(cfg.get("vision_model", "openai/clip-vit-base-patch32"))
        self.output_dim = int(output_dim or cfg.get("image_dim", 512))
        self.using_fallback = True
        self.encoder: nn.Module
        self.proj: nn.Module
        self.register_buffer(
            "_clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False,
        )

        use_tiny = bool(cfg.get("use_tiny_debug_model", False))
        if use_tiny_debug_model is not None:
            use_tiny = use_tiny_debug_model

        if use_tiny:
            self.encoder = _TinyVisionEncoder(self.output_dim)
            self.proj = nn.Identity()
        else:
            self.encoder, self.proj = self._load_clip_or_fallback()

        self.freeze()

    def _load_clip_or_fallback(self) -> tuple[nn.Module, nn.Module]:
        try:
            from transformers import CLIPVisionModelWithProjection

            encoder = CLIPVisionModelWithProjection.from_pretrained(self.model_name)
            hidden_dim = int(encoder.config.projection_dim)
            proj: nn.Module = nn.Identity()
            if hidden_dim != self.output_dim:
                proj = nn.Linear(hidden_dim, self.output_dim, bias=False)
            self.using_fallback = False
            return encoder, proj
        except Exception:
            self.using_fallback = True
            return _TinyVisionEncoder(self.output_dim), nn.Identity()

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"images must have shape [B, 3, H, W], got {tuple(images.shape)}")

        if self.using_fallback:
            return self.encoder(images)

        pixel_values = self._normalize_clip_inputs(images)
        output = self.encoder(pixel_values=pixel_values)
        if getattr(output, "image_embeds", None) is not None:
            image_emb = output.image_embeds
        elif getattr(output, "pooler_output", None) is not None:
            image_emb = output.pooler_output
        else:
            image_emb = output.last_hidden_state[:, 0]
        return self.proj(image_emb)

    def _normalize_clip_inputs(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float()
        return (images - self._clip_mean.to(images.device, images.dtype)) / self._clip_std.to(
            images.device, images.dtype
        )
