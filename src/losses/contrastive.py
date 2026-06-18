from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_info_nce(eeg_emb: torch.Tensor, image_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    eeg_emb = F.normalize(eeg_emb.float(), dim=-1)
    image_emb = F.normalize(image_emb.float(), dim=-1)
    logits = eeg_emb @ image_emb.T / temperature
    labels = torch.arange(eeg_emb.shape[0], device=eeg_emb.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def multi_positive_info_nce(
    eeg_emb: torch.Tensor,
    image_emb: torch.Tensor,
    image_ids: list[str],
    labels: torch.Tensor | None = None,
    temperature: float = 0.07,
    label_positive_weight: float = 0.25,
) -> torch.Tensor:
    if eeg_emb.shape[0] != image_emb.shape[0]:
        raise ValueError(f"batch mismatch: {eeg_emb.shape[0]} vs {image_emb.shape[0]}")
    if len(image_ids) != eeg_emb.shape[0]:
        raise ValueError(f"image_ids length {len(image_ids)} does not match batch size {eeg_emb.shape[0]}")

    eeg_emb = F.normalize(eeg_emb.float(), dim=-1)
    image_emb = F.normalize(image_emb.float(), dim=-1)
    logits = eeg_emb @ image_emb.T / temperature
    positive_weights = torch.tensor(
        [[left == right for right in image_ids] for left in image_ids],
        dtype=torch.float32,
        device=logits.device,
    )
    if labels is not None and label_positive_weight > 0:
        labels = labels.to(logits.device)
        label_mask = labels[:, None] == labels[None, :]
        positive_weights = torch.maximum(
            positive_weights,
            label_mask.float() * float(label_positive_weight),
        )
    if not (positive_weights > 0).any(dim=1).all():
        raise ValueError("each row must contain at least one positive pair")

    def _direction_loss(scores: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        target = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        log_probs = F.log_softmax(scores, dim=-1)
        return -(target * log_probs).sum(dim=-1).mean()

    return 0.5 * (_direction_loss(logits, positive_weights) + _direction_loss(logits.T, positive_weights.T))


def prototype_alignment_loss(eeg_emb: torch.Tensor, image_emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if eeg_emb.shape[0] != image_emb.shape[0] or eeg_emb.shape[0] != labels.shape[0]:
        raise ValueError("eeg_emb, image_emb, and labels must share batch size")
    eeg_emb = F.normalize(eeg_emb.float(), dim=-1)
    image_emb = F.normalize(image_emb.float(), dim=-1)
    labels = labels.to(eeg_emb.device)
    prototypes = torch.zeros_like(image_emb)
    for label in torch.unique(labels):
        mask = labels == label
        proto = F.normalize(image_emb[mask].mean(dim=0, keepdim=True), dim=-1)
        prototypes[mask] = proto
    return 1.0 - F.cosine_similarity(eeg_emb, prototypes, dim=-1).mean()


def supervised_contrastive_loss(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    emb = F.normalize(emb.float(), dim=-1)
    labels = labels.to(emb.device)
    logits = emb @ emb.T / temperature
    eye = torch.eye(emb.shape[0], dtype=torch.bool, device=emb.device)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    if not positive.any():
        return torch.zeros((), device=emb.device, dtype=emb.dtype)
    logits = logits.masked_fill(eye, float("-inf"))
    log_probs = F.log_softmax(logits, dim=-1)
    positive_counts = positive.sum(dim=-1).clamp_min(1)
    per_row = -(log_probs.masked_fill(~positive, 0.0).sum(dim=-1) / positive_counts)
    return per_row[positive.any(dim=-1)].mean()


def same_image_subject_consistency_loss(
    eeg_emb: torch.Tensor,
    image_ids: list[str],
    subject_ids: list[str] | None = None,
) -> torch.Tensor:
    if len(image_ids) != eeg_emb.shape[0]:
        raise ValueError("image_ids length must match batch size")
    if subject_ids is not None and len(subject_ids) != eeg_emb.shape[0]:
        raise ValueError("subject_ids length must match batch size")
    eeg_emb = F.normalize(eeg_emb.float(), dim=-1)
    losses: list[torch.Tensor] = []
    for image_id in sorted(set(image_ids)):
        indices = [idx for idx, current in enumerate(image_ids) if current == image_id]
        if len(indices) < 2:
            continue
        group = eeg_emb[torch.tensor(indices, device=eeg_emb.device)]
        sim = group @ group.T
        mask = ~torch.eye(group.shape[0], dtype=torch.bool, device=eeg_emb.device)
        if subject_ids is not None:
            subjects = [str(subject_ids[idx]) for idx in indices]
            subject_mask = torch.tensor(
                [[left != right for right in subjects] for left in subjects],
                dtype=torch.bool,
                device=eeg_emb.device,
            )
            mask = mask & subject_mask
        if mask.any():
            losses.append(1.0 - sim[mask].mean())
    if not losses:
        return torch.zeros((), device=eeg_emb.device, dtype=eeg_emb.dtype)
    return torch.stack(losses).mean()
