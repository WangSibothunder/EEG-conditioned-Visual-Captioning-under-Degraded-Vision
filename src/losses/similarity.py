from __future__ import annotations

import torch
import torch.nn.functional as F


def similarity_distillation_loss(eeg_emb: torch.Tensor, image_emb: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    eeg_emb = F.normalize(eeg_emb.float(), dim=-1)
    image_emb = F.normalize(image_emb.float(), dim=-1)
    image_sim = image_emb @ image_emb.T
    eeg_sim = eeg_emb @ eeg_emb.T
    target = F.softmax(image_sim / tau, dim=-1)
    pred = F.log_softmax(eeg_sim / tau, dim=-1)
    return F.kl_div(pred, target, reduction="batchmean")
