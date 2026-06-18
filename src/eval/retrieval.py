from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.clip_cache import load_cache
from src.data.collate import caption_collate
from src.data.dataset import EEGVisionCaptionDataset
from src.models.alignment_model import EEGCLIPAlignmentModel
from src.utils.checkpoint import load_checkpoint


def retrieval_metrics(
    query_emb: torch.Tensor,
    target_emb: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
    *,
    query_ids: list[str] | None = None,
    target_ids: list[str] | None = None,
) -> dict[str, Any]:
    query_emb = torch.nn.functional.normalize(query_emb.float(), dim=-1)
    target_emb = torch.nn.functional.normalize(target_emb.float(), dim=-1)
    sim = query_emb @ target_emb.T
    order = torch.argsort(sim, dim=1, descending=True)
    if query_ids is None and target_ids is None:
        gold = torch.arange(query_emb.shape[0], device=query_emb.device)
        ranks = (order == gold[:, None]).nonzero()[:, 1] + 1
    else:
        if query_ids is None or target_ids is None:
            raise ValueError("query_ids and target_ids must be provided together")
        if len(query_ids) != query_emb.shape[0]:
            raise ValueError(f"query_ids length {len(query_ids)} does not match query count {query_emb.shape[0]}")
        if len(target_ids) != target_emb.shape[0]:
            raise ValueError(f"target_ids length {len(target_ids)} does not match target count {target_emb.shape[0]}")
        id_to_code = {image_id: idx for idx, image_id in enumerate(sorted(set(query_ids) | set(target_ids)))}
        query_codes = torch.tensor([id_to_code[image_id] for image_id in query_ids], dtype=torch.long)
        target_codes = torch.tensor([id_to_code[image_id] for image_id in target_ids], dtype=torch.long)
        ordered_codes = target_codes[order.cpu()]
        positive = ordered_codes.eq(query_codes[:, None])
        first_positive = positive.float().argmax(dim=1).long() + 1
        ranks = torch.where(
            positive.any(dim=1),
            first_positive,
            torch.full_like(first_positive, target_emb.shape[0] + 1),
        )
    metrics: dict[str, Any] = {
        "mean_rank": float(ranks.float().mean().item()),
        "median_rank": float(ranks.float().median().item()),
    }
    for k in ks:
        metrics[f"r@{k}"] = float((ranks <= k).float().mean().item())
    return metrics


def _unique_embeddings(emb: torch.Tensor, image_ids: list[str]) -> tuple[torch.Tensor, list[str]]:
    groups: dict[str, list[int]] = {}
    for idx, image_id in enumerate(image_ids):
        groups.setdefault(str(image_id), []).append(idx)
    ids = sorted(groups)
    chunks = [emb[groups[image_id]].mean(dim=0) for image_id in ids]
    return torch.stack(chunks, dim=0), ids


def retrieval_metric_bundle(query_emb: torch.Tensor, target_emb: torch.Tensor, image_ids: list[str]) -> dict[str, Any]:
    trial = retrieval_metrics(query_emb, target_emb, query_ids=image_ids, target_ids=image_ids)
    random_trial = random_retrieval_metrics(query_emb.shape[0], query_ids=image_ids, target_ids=image_ids)
    unique_query, unique_ids = _unique_embeddings(query_emb, image_ids)
    unique_target, target_unique_ids = _unique_embeddings(target_emb, image_ids)
    unique = retrieval_metrics(unique_query, unique_target, query_ids=unique_ids, target_ids=target_unique_ids)
    random_unique = random_retrieval_metrics(unique_query.shape[0], query_ids=unique_ids, target_ids=target_unique_ids)
    bundle: dict[str, Any] = {
        "trial": trial,
        "unique_image": unique,
        "random_trial": random_trial,
        "random_unique_image": random_unique,
    }
    bundle.update(unique)
    return bundle


def random_retrieval_metrics(
    n: int,
    ks: tuple[int, ...] = (1, 5, 10),
    trials: int = 20,
    *,
    query_ids: list[str] | None = None,
    target_ids: list[str] | None = None,
) -> dict[str, Any]:
    if query_ids is not None or target_ids is not None:
        if query_ids is None or target_ids is None:
            raise ValueError("query_ids and target_ids must be provided together")
        if len(query_ids) != n:
            raise ValueError(f"query_ids length {len(query_ids)} does not match n={n}")
        target_count = len(target_ids)
        positive_counts = [sum(1 for target_id in target_ids if target_id == query_id) for query_id in query_ids]
        metrics: dict[str, Any] = {}
        unique_positive_counts = sorted(set(positive_counts))
        recall_cache = {
            (positives, k): _random_best_rank_recall(target_count, positives, k)
            for positives in unique_positive_counts
            for k in ks
        }
        median_cache = {
            positives: _random_best_rank_median(target_count, positives)
            for positives in unique_positive_counts
        }
        for k in ks:
            metrics[f"r@{k}"] = float(np.mean([recall_cache[(positives, k)] for positives in positive_counts]))
        metrics["mean_rank"] = float(
            np.mean(
                [
                    (target_count + 1) / (positives + 1) if positives > 0 else target_count + 1
                    for positives in positive_counts
                ]
            )
        )
        metrics["median_rank"] = float(np.mean([median_cache[positives] for positives in positive_counts]))
        return metrics

    values: dict[str, list[float]] = {f"r@{k}": [] for k in ks}
    values["mean_rank"] = []
    values["median_rank"] = []
    for _ in range(trials):
        query = torch.randn(n, 512)
        target = torch.randn(n, 512)
        metrics = retrieval_metrics(query, target, ks=ks, query_ids=query_ids, target_ids=target_ids)
        for key, value in metrics.items():
            values[key].append(float(value))
    return {key: float(np.mean(vals)) for key, vals in values.items()}


def _random_best_rank_recall(target_count: int, positives: int, k: int) -> float:
    if positives <= 0:
        return 0.0
    k = min(k, target_count)
    if k <= 0:
        return 0.0
    if positives > target_count - k:
        return 1.0
    no_positive = 1.0
    for offset in range(k):
        no_positive *= (target_count - positives - offset) / (target_count - offset)
    return 1.0 - no_positive


def _random_best_rank_median(target_count: int, positives: int) -> float:
    if positives <= 0:
        return float(target_count + 1)
    low = 1
    high = target_count
    answer = target_count
    while low <= high:
        mid = (low + high) // 2
        if _random_best_rank_recall(target_count, positives, mid) >= 0.5:
            answer = mid
            high = mid - 1
        else:
            low = mid + 1
    return float(answer)


def save_retrieval_report(path: str | Path, metrics: dict[str, Any], random_metrics: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Alignment Report",
        "",
        "## Retrieval Metrics",
        "",
        f"- model R@1: `{metrics.get('r@1', 0.0):.4f}`",
        f"- random R@1: `{random_metrics.get('r@1', 0.0):.4f}`",
        f"- model R@5: `{metrics.get('r@5', 0.0):.4f}`",
        f"- random R@5: `{random_metrics.get('r@5', 0.0):.4f}`",
        f"- model R@10: `{metrics.get('r@10', 0.0):.4f}`",
        f"- mean rank: `{metrics.get('mean_rank', 0.0):.2f}`",
        f"- median rank: `{metrics.get('median_rank', 0.0):.2f}`",
        "",
        "Do not overclaim. These are preliminary EEG-to-CLIP alignment metrics.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_metrics_json(path: str | Path, metrics: dict[str, Any], random_metrics: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"model": metrics, "random": random_metrics}, handle, indent=2, sort_keys=True)


def class_accuracy_from_logits(logits: torch.Tensor | None, labels: torch.Tensor) -> float:
    if logits is None or logits.numel() == 0 or labels.numel() == 0:
        return 0.0
    preds = logits.detach().argmax(dim=-1).to(labels.device)
    return float((preds == labels).float().mean().item())


def _num_classes_from_manifest(path: str | Path) -> int | None:
    labels: list[int] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("label") is not None:
                    labels.append(int(row["label"]))
    return max(labels) + 1 if labels else None


class _RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: EEGVisionCaptionDataset, clip_embeddings: torch.Tensor) -> None:
        if len(dataset) != clip_embeddings.shape[0]:
            raise ValueError(f"Dataset/cache length mismatch: {len(dataset)} vs {clip_embeddings.shape[0]}")
        self.dataset = dataset
        self.clip_embeddings = clip_embeddings.float()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        item["clip_emb"] = self.clip_embeddings[index]
        return item


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = caption_collate(batch)
    out["clip_emb"] = torch.stack([item["clip_emb"] for item in batch], dim=0).float()
    return out


def _infer_index_path(cache_path: str | Path) -> Path:
    path = Path(cache_path)
    name = path.name.replace("clip_", "clip_index_").replace(".npy", ".json")
    return path.with_name(name)


def evaluate_checkpoint(
    *,
    manifest: str | Path,
    clip_cache: str | Path,
    eeg_ckpt: str | Path,
    out: str | Path,
    index_path: str | Path | None = None,
    batch_size: int = 128,
    device: str = "auto",
) -> dict[str, Any]:
    torch_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    checkpoint = load_checkpoint(eeg_ckpt, map_location=torch_device)
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    eeg_channels = int(model_cfg.get("eeg_channels", 64))
    eeg_timesteps = int(model_cfg.get("eeg_time_steps", 250))
    model = EEGCLIPAlignmentModel(
        eeg_channels=eeg_channels,
        eeg_timesteps=eeg_timesteps,
        eeg_dim=int(model_cfg.get("eeg_embed_dim", 512)),
        clip_dim=int(model_cfg.get("clip_embed_dim", 512)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        transformer_layers=int(model_cfg.get("transformer_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_classes=_num_classes_from_manifest(manifest),
        encoder_type=str(model_cfg.get("encoder_type", "tiny")),
    ).to(torch_device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=False)
    model.eval()

    index = Path(index_path) if index_path else _infer_index_path(clip_cache)
    clip_embeddings, _ = load_cache(clip_cache, index)
    dataset = EEGVisionCaptionDataset(
        manifest,
        eeg_shape=(eeg_channels, eeg_timesteps),
        allow_missing_images=True,
    )
    loader = DataLoader(
        _RetrievalDataset(dataset, clip_embeddings),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    pred_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    class_acc_values: list[float] = []
    image_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(torch_device)
            pred, logits = model(batch["eeg"].to(torch_device), subject_ids=batch.get("subject_id"))
            pred_chunks.append(pred.cpu())
            target_chunks.append(batch["clip_emb"].cpu())
            if logits is not None:
                class_acc_values.append(class_accuracy_from_logits(logits, labels))
            image_ids.extend(str(image_id) for image_id in batch["image_id"])
    pred_all = torch.cat(pred_chunks, dim=0)
    target_all = torch.cat(target_chunks, dim=0)
    metrics = retrieval_metric_bundle(pred_all, target_all, image_ids=image_ids)
    if class_acc_values:
        metrics["class_acc"] = float(np.mean(class_acc_values))
        metrics.setdefault("unique_image", {})["class_acc"] = metrics["class_acc"]
    random_metrics = metrics.get("random_unique_image", random_retrieval_metrics(pred_all.shape[0], query_ids=image_ids, target_ids=image_ids))
    save_metrics_json(out, metrics, random_metrics)
    out_path = Path(out)
    save_retrieval_report(out_path.with_name("retrieval_report.md"), metrics, random_metrics)
    if out_path.name == "alignment_metrics.json":
        save_retrieval_report(out_path.with_name("alignment_report.md"), metrics, random_metrics)
    return {"model": metrics, "random": random_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EEG-to-image retrieval from an alignment checkpoint.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clip_cache", required=True)
    parser.add_argument("--clip_index", default=None)
    parser.add_argument("--eeg_ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    metrics = evaluate_checkpoint(
        manifest=args.manifest,
        clip_cache=args.clip_cache,
        index_path=args.clip_index,
        eeg_ckpt=args.eeg_ckpt,
        out=args.out,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
