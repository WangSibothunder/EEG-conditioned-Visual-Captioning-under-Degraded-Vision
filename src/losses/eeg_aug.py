from __future__ import annotations

import torch


def augment_eeg(
    eeg: torch.Tensor,
    *,
    noise_std: float = 0.01,
    channel_dropout: float = 0.05,
    max_time_shift: int = 8,
    scale_jitter: float = 0.05,
) -> torch.Tensor:
    if eeg.ndim != 3:
        raise ValueError(f"eeg must have shape [B, C, T], got {tuple(eeg.shape)}")

    out = eeg.float()
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    if scale_jitter > 0:
        scale = 1.0 + (torch.rand(out.shape[0], 1, 1, device=out.device) * 2.0 - 1.0) * scale_jitter
        out = out * scale
    if channel_dropout > 0:
        keep = torch.rand(out.shape[0], out.shape[1], 1, device=out.device) >= channel_dropout
        out = out * keep.to(out.dtype)
    if max_time_shift > 0:
        shifts = torch.randint(-max_time_shift, max_time_shift + 1, (out.shape[0],), device=out.device)
        out = torch.stack([torch.roll(sample, int(shift.item()), dims=-1) for sample, shift in zip(out, shifts)], dim=0)
    return out
