from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.clip_cache import load_cache
from src.data.collate import caption_collate
from src.data.dataset import EEGVisionCaptionDataset
from src.eval.retrieval import class_accuracy_from_logits, random_retrieval_metrics, retrieval_metric_bundle
from src.models.eeg_encoder import build_eeg_encoder
from src.utils.config import load_config
from src.utils.seed import seed_everything


class CachedClipAdapterDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: EEGVisionCaptionDataset,
        clip_embeddings: torch.Tensor,
        eeg_cache: np.ndarray | None = None,
    ) -> None:
        if len(dataset) != int(clip_embeddings.shape[0]):
            raise ValueError(f"Dataset/cache length mismatch: {len(dataset)} vs {clip_embeddings.shape[0]}")
        if eeg_cache is not None and len(dataset) != int(eeg_cache.shape[0]):
            raise ValueError(f"Dataset/EEG cache length mismatch: {len(dataset)} vs {eeg_cache.shape[0]}")
        self.dataset = dataset
        self.clip_embeddings = clip_embeddings.float()
        self.eeg_cache = eeg_cache

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        if self.eeg_cache is not None:
            # 使用预处理 EEG mmap，避免每个 worker 反复 torch.load 3GB Thought2Text block。
            item["eeg"] = torch.from_numpy(np.asarray(self.eeg_cache[index], dtype=np.float32).copy())
        item["clip_emb"] = self.clip_embeddings[index]
        return item


def clip_adapter_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = caption_collate(batch)
    out["clip_emb"] = torch.stack([item["clip_emb"] for item in batch], dim=0).float()
    return out


class NeuralAwareClipAdapter(nn.Module):
    """Small EEG-conditioned adapter calibrated against cached CLIP class prototypes."""

    def __init__(
        self,
        *,
        eeg_channels: int = 64,
        eeg_timesteps: int = 250,
        eeg_dim: int = 512,
        clip_dim: int = 512,
        hidden_dim: int = 128,
        adapter_hidden_dim: int = 1024,
        transformer_layers: int = 2,
        dropout: float = 0.1,
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
        self.eeg_to_clip = nn.Sequential(nn.LayerNorm(eeg_dim), nn.Linear(eeg_dim, clip_dim))
        self.adapter = nn.Sequential(
            nn.LayerNorm(clip_dim * 2),
            nn.Linear(clip_dim * 2, adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden_dim, clip_dim),
        )
        self.gate = nn.Sequential(nn.LayerNorm(clip_dim * 2), nn.Linear(clip_dim * 2, clip_dim), nn.Sigmoid())

    def forward(self, eeg: torch.Tensor, clip_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        if eeg.ndim != 3:
            raise ValueError(f"eeg must have shape [B, C, T], got {tuple(eeg.shape)}")
        if clip_emb.ndim != 2:
            raise ValueError(f"clip_emb must have shape [B, D], got {tuple(clip_emb.shape)}")
        eeg_clip = F.normalize(self.eeg_to_clip(self.eeg_encoder(eeg)), dim=-1)
        base_clip = F.normalize(clip_emb.float(), dim=-1)
        combined = torch.cat([base_clip, eeg_clip], dim=-1)
        delta = self.adapter(combined)
        gate = self.gate(combined)
        adapted = F.normalize(base_clip + gate * delta, dim=-1)
        return {"eeg_clip": eeg_clip, "adapted_clip": adapted, "gate_mean": gate.mean()}


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _num_classes(path: str | Path) -> int:
    labels = [int(row["label"]) for row in _read_rows(path) if row.get("label") is not None]
    if not labels:
        raise ValueError(f"No labels found in manifest: {path}")
    return max(labels) + 1


def _subset(dataset: torch.utils.data.Dataset, max_samples: int) -> torch.utils.data.Dataset:
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def _build_loader(
    *,
    manifest: str | Path,
    cache: str | Path,
    index: str | Path,
    eeg_shape: tuple[int, int],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    max_samples: int,
    eeg_cache: str | Path | None = None,
) -> DataLoader:
    clip_embeddings, _ = load_cache(cache, index)
    base = EEGVisionCaptionDataset(manifest, eeg_shape=eeg_shape, allow_missing_images=True)
    eeg_array = np.load(eeg_cache, mmap_mode="r") if eeg_cache else None
    dataset = _subset(CachedClipAdapterDataset(base, clip_embeddings, eeg_array), max_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=clip_adapter_collate,
        pin_memory=torch.cuda.is_available(),
    )


def _class_prototypes(embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    prototypes = torch.zeros((num_classes, embeddings.shape[-1]), dtype=torch.float32)
    counts = torch.zeros((num_classes,), dtype=torch.float32)
    for cls in range(num_classes):
        mask = labels == cls
        if mask.any():
            prototypes[cls] = embeddings[mask].mean(dim=0)
            counts[cls] = float(mask.sum())
    missing = counts == 0
    if missing.any():
        fallback = embeddings.mean(dim=0)
        prototypes[missing] = fallback
    return F.normalize(prototypes, dim=-1)


def _collect_labels(loader: DataLoader) -> torch.Tensor:
    labels: list[torch.Tensor] = []
    for batch in loader:
        labels.append(batch["label"].long())
    return torch.cat(labels, dim=0)


def _collect_clip_embeddings(loader: DataLoader) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for batch in loader:
        chunks.append(batch["clip_emb"].float())
    return torch.cat(chunks, dim=0)


def _adapter_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    adapted = outputs["adapted_clip"]
    eeg_clip = outputs["eeg_clip"]
    target = F.normalize(target, dim=-1)
    temperature = float(loss_cfg.get("temperature", 0.07))
    logits = adapted @ prototypes.to(adapted.device).T / temperature
    eeg_logits = eeg_clip @ prototypes.to(eeg_clip.device).T / temperature

    mse = F.mse_loss(adapted, target)
    eeg_mse = F.mse_loss(eeg_clip, target)
    cosine = 1.0 - F.cosine_similarity(adapted, target, dim=-1).mean()
    cls = F.cross_entropy(logits, labels)
    eeg_cls = F.cross_entropy(eeg_logits, labels)
    total = (
        float(loss_cfg.get("lambda_mse", 0.5)) * mse
        + float(loss_cfg.get("lambda_eeg_mse", 0.5)) * eeg_mse
        + float(loss_cfg.get("lambda_cosine", 0.5)) * cosine
        + float(loss_cfg.get("lambda_cls", 1.0)) * cls
        + float(loss_cfg.get("lambda_eeg_cls", 0.3)) * eeg_cls
    )
    parts = {
        "loss": float(total.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "eeg_mse": float(eeg_mse.detach().cpu()),
        "cosine": float(cosine.detach().cpu()),
        "cls": float(cls.detach().cpu()),
        "eeg_cls": float(eeg_cls.detach().cpu()),
        "gate_mean": float(outputs["gate_mean"].detach().cpu()),
    }
    return total, parts, logits


def evaluate(
    model: NeuralAwareClipAdapter,
    loader: DataLoader,
    device: torch.device,
    prototypes: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    adapted_chunks: list[torch.Tensor] = []
    eeg_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    image_ids: list[str] = []
    loss_values: list[float] = []
    class_acc_values: list[float] = []
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device, non_blocking=True)
            target = batch["clip_emb"].to(device, non_blocking=True)
            labels = batch["label"].long().to(device, non_blocking=True)
            outputs = model(eeg, target)
            loss, _, logits = _adapter_loss(outputs, target, labels, prototypes, loss_cfg)
            loss_values.append(float(loss.detach().cpu()))
            class_acc_values.append(class_accuracy_from_logits(logits, labels))
            adapted_chunks.append(outputs["adapted_clip"].cpu())
            eeg_chunks.append(outputs["eeg_clip"].cpu())
            target_chunks.append(F.normalize(target.float(), dim=-1).cpu())
            image_ids.extend(str(image_id) for image_id in batch["image_id"])

    adapted = torch.cat(adapted_chunks, dim=0)
    eeg_clip = torch.cat(eeg_chunks, dim=0)
    target = torch.cat(target_chunks, dim=0)
    return {
        "loss": float(np.mean(loss_values)) if loss_values else 0.0,
        "class_acc": float(np.mean(class_acc_values)) if class_acc_values else 0.0,
        "retrieval": retrieval_metric_bundle(adapted, target, image_ids=image_ids),
        "eeg_retrieval": retrieval_metric_bundle(eeg_clip, target, image_ids=image_ids),
        "random": random_retrieval_metrics(adapted.shape[0], query_ids=image_ids, target_ids=image_ids),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_config_copy(config: dict[str, Any], out_dir: Path, source_config: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "config.yaml"
    if source_config and Path(source_config).exists():
        shutil.copyfile(source_config, target)
        return
    try:
        import yaml

        target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    except ModuleNotFoundError:
        target.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(out_dir: Path, metrics: dict[str, Any], config: dict[str, Any]) -> None:
    val = metrics["val"]
    retrieval = val["retrieval"]["unique_image"]
    eeg_retrieval = val["eeg_retrieval"]["unique_image"]
    lines = [
        "# CLIP Adapter Report",
        "",
        "Boundary: this is a cached-embedding Neural-Aware CLIP Adapter baseline. It uses Thought2Text manifests, EEG tensors, and precomputed CLIP embeddings; it does not load raw images or the CLIP model.",
        "",
        "## Validation Metrics",
        "",
        f"- adapted unique-image R@1: `{retrieval.get('r@1', 0.0):.4f}`",
        f"- adapted unique-image R@5: `{retrieval.get('r@5', 0.0):.4f}`",
        f"- EEG-only unique-image R@1: `{eeg_retrieval.get('r@1', 0.0):.4f}`",
        f"- EEG-only unique-image R@5: `{eeg_retrieval.get('r@5', 0.0):.4f}`",
        f"- class accuracy: `{val.get('class_acc', 0.0):.4f}`",
        f"- validation loss: `{val.get('loss', 0.0):.4f}`",
        "",
        "## Inputs",
        "",
        f"- train manifest: `{config['data']['train_manifest']}`",
        f"- val manifest: `{config['data']['val_manifest']}`",
        f"- train cache: `{config['data']['clip_train_cache']}`",
        f"- val cache: `{config['data']['clip_val_cache']}`",
        "",
        "Do not claim an EEG benefit from this job alone. Use it as a calibration/retrieval diagnostic for downstream degraded-vision fusion controls.",
    ]
    (out_dir / "CLIP_ADAPTER_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_files(config: dict[str, Any]) -> None:
    data_cfg = config["data"]
    for key in ("train_manifest", "val_manifest", "clip_train_cache", "clip_val_cache", "clip_index_train", "clip_index_val"):
        path = Path(data_cfg[key])
        if not path.exists():
            raise FileNotFoundError(f"Missing required {key}: {path}")


def train(config: dict[str, Any], *, source_config: str | None = None) -> dict[str, Any]:
    seed_everything(int(config.get("seed", 42)))
    _validate_files(config)
    data_cfg = config["data"]
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    loss_cfg = config.get("loss", {})
    out_dir = Path(config.get("output", {}).get("dir", "outputs/clip_adapter/run"))
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _write_config_copy(config, out_dir, source_config=source_config)

    device = _resolve_device(str(config.get("device", "auto")))
    eeg_shape = (
        int(model_cfg.get("eeg_channels", model_cfg.get("channels", 64))),
        int(model_cfg.get("eeg_time_steps", model_cfg.get("timesteps", 250))),
    )
    batch_size = int(train_cfg.get("batch_size", 128))
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = _build_loader(
        manifest=data_cfg["train_manifest"],
        cache=data_cfg["clip_train_cache"],
        index=data_cfg["clip_index_train"],
        eeg_shape=eeg_shape,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        max_samples=int(train_cfg.get("max_train_samples", 0)),
        eeg_cache=data_cfg.get("eeg_train_cache"),
    )
    val_loader = _build_loader(
        manifest=data_cfg["val_manifest"],
        cache=data_cfg["clip_val_cache"],
        index=data_cfg["clip_index_val"],
        eeg_shape=eeg_shape,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        max_samples=int(train_cfg.get("max_val_samples", 0)),
        eeg_cache=data_cfg.get("eeg_val_cache"),
    )
    num_classes = _num_classes(data_cfg["train_manifest"])
    clip_dim = int(model_cfg.get("clip_dim", model_cfg.get("clip_embed_dim", _collect_clip_embeddings(train_loader).shape[-1])))
    train_labels = _collect_labels(train_loader)
    train_clip = _collect_clip_embeddings(train_loader)
    prototypes = _class_prototypes(F.normalize(train_clip, dim=-1), train_labels, num_classes).to(device)

    model = NeuralAwareClipAdapter(
        eeg_channels=eeg_shape[0],
        eeg_timesteps=eeg_shape[1],
        eeg_dim=int(model_cfg.get("eeg_dim", model_cfg.get("eeg_embed_dim", 512))),
        clip_dim=clip_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        adapter_hidden_dim=int(model_cfg.get("adapter_hidden_dim", 1024)),
        transformer_layers=int(model_cfg.get("transformer_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        encoder_type=str(model_cfg.get("encoder_type", "tiny")),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    use_amp = bool(train_cfg.get("bf16", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    epochs = int(train_cfg.get("epochs", 20))
    log_every = int(train_cfg.get("log_every", 20))
    best_r5 = -1.0
    history: list[dict[str, Any]] = []
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            eeg = batch["eeg"].to(device, non_blocking=True)
            target = batch["clip_emb"].to(device, non_blocking=True)
            labels = batch["label"].long().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(eeg, target)
                loss, parts, _ = _adapter_loss(outputs, target, labels, prototypes, loss_cfg)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite CLIP adapter loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            optimizer.step()
            global_step += 1
            epoch_losses.append(float(loss.detach().cpu()))
            if global_step % max(log_every, 1) == 0:
                with (out_dir / "train.log").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": global_step, "epoch": epoch + 1, **parts}, sort_keys=True) + "\n")

        val_metrics = evaluate(model, val_loader, device, prototypes, loss_cfg)
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "val": val_metrics,
        }
        history.append(record)
        r5 = float(val_metrics["retrieval"]["unique_image"].get("r@5", 0.0))
        checkpoint = {
            "model": model.state_dict(),
            "prototypes": prototypes.detach().cpu(),
            "epoch": epoch,
            "metrics": val_metrics,
            "config": config,
        }
        torch.save(checkpoint, ckpt_dir / "last.pt")
        if r5 >= best_r5:
            best_r5 = r5
            torch.save(checkpoint, ckpt_dir / "best.pt")
        print(f"epoch={epoch + 1} train_loss={record['train_loss']:.4f} val_r@5={r5:.4f} class_acc={val_metrics['class_acc']:.4f}")

    final_metrics = {"train": {"last_loss": history[-1]["train_loss"] if history else 0.0}, "val": history[-1]["val"] if history else {}}
    _write_json(out_dir / "history.json", history)
    _write_json(out_dir / "metrics.json", final_metrics)
    _write_report(out_dir, final_metrics, config)
    return final_metrics


def _default_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed": args.seed,
        "device": args.device,
        "data": {
            "train_manifest": "data/thought2text/train.jsonl",
            "val_manifest": "data/thought2text/val.jsonl",
            "clip_train_cache": "data/thought2text/cache/clip_train.npy",
            "clip_val_cache": "data/thought2text/cache/clip_val.npy",
            "clip_index_train": "data/thought2text/cache/clip_index_train.json",
            "clip_index_val": "data/thought2text/cache/clip_index_val.json",
            "eeg_train_cache": "data/thought2text/cache/eeg_pretrain_train.npy",
            "eeg_val_cache": "data/thought2text/cache/eeg_pretrain_val.npy",
        },
        "model": {
            "eeg_channels": 64,
            "eeg_time_steps": 250,
            "eeg_embed_dim": 512,
            "clip_dim": 512,
            "hidden_dim": 128,
            "adapter_hidden_dim": 1024,
            "transformer_layers": 2,
            "dropout": 0.1,
            "encoder_type": "tiny",
        },
        "loss": {
            "temperature": 0.07,
            "lambda_mse": 0.5,
            "lambda_eeg_mse": 0.5,
            "lambda_cosine": 0.5,
            "lambda_cls": 1.0,
            "lambda_eeg_cls": 0.3,
        },
        "train": {
            "epochs": args.epochs or 20,
            "batch_size": args.batch_size or 128,
            "num_workers": args.num_workers if args.num_workers is not None else 4,
            "lr": args.lr,
            "weight_decay": 0.01,
            "bf16": True,
            "log_every": 20,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
        },
        "output": {"dir": args.output_dir},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cached-embedding Neural-Aware CLIP Adapter.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_dir", "--out_dir", dest="output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        source_config = args.config
    else:
        config = _default_config_from_args(args)
        source_config = None
    if args.output_dir is not None:
        config.setdefault("output", {})["dir"] = args.output_dir
    if args.device != "auto":
        config["device"] = args.device
    if args.seed != 42:
        config["seed"] = args.seed
    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("train", {})["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config.setdefault("train", {})["num_workers"] = args.num_workers
    if args.lr != 1e-4:
        config.setdefault("train", {})["lr"] = args.lr
    if args.max_train_samples:
        config.setdefault("train", {})["max_train_samples"] = args.max_train_samples
    if args.max_val_samples:
        config.setdefault("train", {})["max_val_samples"] = args.max_val_samples
    train(config, source_config=source_config)


if __name__ == "__main__":
    main()
