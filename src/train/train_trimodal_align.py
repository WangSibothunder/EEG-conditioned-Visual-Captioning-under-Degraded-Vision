from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.clip_cache import load_cache
from src.data.dataset import EEGVisionCaptionDataset
from src.eval.retrieval import class_accuracy_from_logits, random_retrieval_metrics, retrieval_metric_bundle, save_metrics_json
from src.losses.contrastive import multi_positive_info_nce, prototype_alignment_loss, symmetric_info_nce
from src.models.alignment_model import EEGCLIPAlignmentModel
from src.train.train_align import load_pretrained_eeg_encoder
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config
from src.utils.seed import seed_everything


class TriModalAlignmentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: EEGVisionCaptionDataset,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_index: list[dict[str, Any]],
        *,
        text_source: str = "human_class",
        eeg_cache: np.ndarray | None = None,
    ) -> None:
        if len(dataset) != image_embeddings.shape[0]:
            raise ValueError(f"Dataset/image cache length mismatch: {len(dataset)} vs {image_embeddings.shape[0]}")
        if eeg_cache is not None and len(dataset) != int(eeg_cache.shape[0]):
            raise ValueError(f"Dataset/EEG cache length mismatch: {len(dataset)} vs {eeg_cache.shape[0]}")
        self.dataset = dataset
        self.image_embeddings = image_embeddings.float()
        self.text_embeddings = text_embeddings.float()
        self.text_lookup = _build_text_lookup(text_index, text_source=text_source)
        self.eeg_cache = eeg_cache

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_row = self.dataset.rows[index]
        text_idx = _lookup_text_index(self.text_lookup, source_row)
        if self.eeg_cache is not None:
            item = {
                "eeg": torch.from_numpy(np.asarray(self.eeg_cache[index], dtype=np.float32).copy()),
                "caption": str(source_row["caption"]),
                "image_id": str(source_row["image_id"]),
                "label": int(source_row["label"]),
                "subject_id": source_row.get("subject_id"),
            }
        else:
            item = self.dataset[index]
        item["image_emb"] = self.image_embeddings[index]
        item["text_emb"] = self.text_embeddings[text_idx]
        item["text_cache_index"] = text_idx
        return item


def _source_aliases(source: str) -> set[str]:
    aliases = {source}
    if source == "human":
        aliases.update({"human_class", "wnid_fallback"})
    if source == "human_class":
        aliases.update({"human", "wnid_fallback"})
    return aliases


def _build_text_lookup(text_index: list[dict[str, Any]], *, text_source: str) -> dict[tuple[str, str, str, int | None], int]:
    aliases = _source_aliases(text_source)
    lookup: dict[tuple[str, str, str, int | None], int] = {}
    for idx, row in enumerate(text_index):
        source = str(row.get("caption_source", ""))
        if source not in aliases:
            continue
        split = str(row.get("split", ""))
        image_id = str(row.get("image_id", ""))
        eeg_index = row.get("eeg_index")
        eeg_key = int(eeg_index) if eeg_index is not None else None
        lookup.setdefault((split, source, image_id, eeg_key), idx)
        lookup.setdefault((split, source, image_id, None), idx)
        for alias in aliases:
            lookup.setdefault((split, alias, image_id, eeg_key), idx)
            lookup.setdefault((split, alias, image_id, None), idx)
    if not lookup:
        raise ValueError(f"No text cache rows matched text_source={text_source!r}")
    return lookup


def _lookup_text_index(lookup: dict[tuple[str, str, str, int | None], int], row: dict[str, Any]) -> int:
    split = str(row.get("split", ""))
    image_id = str(row.get("image_id", ""))
    eeg_index = int(row["eeg_index"]) if row.get("eeg_index") is not None else None
    row_source = str(row.get("caption_source", ""))
    sources = [source for source in (row_source, "human_class", "human", "wnid_fallback", "blip") if source]
    for source in dict.fromkeys(sources):
        for key in ((split, source, image_id, eeg_index), (split, source, image_id, None)):
            if key in lookup:
                return lookup[key]
    raise KeyError(f"No text embedding for split={split} image_id={image_id} eeg_index={eeg_index}")


def trimodal_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "eeg": torch.stack([item["eeg"] for item in batch], dim=0).float(),
        "caption": [str(item["caption"]) for item in batch],
        "image_id": [str(item["image_id"]) for item in batch],
        "label": torch.tensor([int(item["label"]) for item in batch], dtype=torch.long),
        "subject_id": [item.get("subject_id") for item in batch],
    }
    if "image" in batch[0]:
        out["image"] = torch.stack([item["image"] for item in batch], dim=0).float()
    out["image_emb"] = torch.stack([item["image_emb"] for item in batch], dim=0).float()
    out["text_emb"] = torch.stack([item["text_emb"] for item in batch], dim=0).float()
    out["text_cache_index"] = torch.tensor([int(item["text_cache_index"]) for item in batch], dtype=torch.long)
    return out


def _subset(dataset: torch.utils.data.Dataset, max_samples: int) -> torch.utils.data.Dataset:
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _num_classes(manifest: str | Path) -> int | None:
    labels = [int(row["label"]) for row in _read_rows(manifest) if row.get("label") is not None]
    return max(labels) + 1 if labels else None


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_text_cache(cache: str | Path, index: str | Path) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    return load_cache(cache, index)


def _build_loader(
    *,
    manifest: str | Path,
    image_cache: str | Path,
    image_index: str | Path,
    text_cache: str | Path,
    text_index: str | Path,
    text_source: str,
    batch_size: int,
    max_samples: int,
    eeg_shape: tuple[int, int],
    shuffle: bool,
    num_workers: int,
    eeg_cache: str | Path | None = None,
) -> DataLoader:
    image_embeddings, _ = load_cache(image_cache, image_index)
    text_embeddings, text_rows = _load_text_cache(text_cache, text_index)
    dataset = EEGVisionCaptionDataset(manifest, eeg_shape=eeg_shape, allow_missing_images=True)
    eeg_array = np.load(eeg_cache, mmap_mode="r") if eeg_cache else None
    wrapped = TriModalAlignmentDataset(
        dataset,
        image_embeddings,
        text_embeddings,
        text_rows,
        text_source=text_source,
        eeg_cache=eeg_array,
    )
    return DataLoader(
        _subset(wrapped, max_samples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=trimodal_collate,
        pin_memory=torch.cuda.is_available(),
    )


def _forward(model: EEGCLIPAlignmentModel, eeg: torch.Tensor, subject_ids: list[str] | torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    try:
        return model(eeg, subject_ids=subject_ids)
    except TypeError:
        return model(eeg)


def compute_trimodal_loss(
    model: EEGCLIPAlignmentModel,
    batch: dict[str, Any],
    device: torch.device,
    *,
    temperature: float,
    lambda_image: float,
    lambda_text: float,
    lambda_cls: float,
    lambda_proto: float,
    use_multi_positive: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    eeg = batch["eeg"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    image_target = F.normalize(batch["image_emb"].to(device, non_blocking=True), dim=-1)
    text_target = F.normalize(batch["text_emb"].to(device, non_blocking=True), dim=-1)
    pred, logits = _forward(model, eeg, batch.get("subject_id"))

    contrast_fn = multi_positive_info_nce if use_multi_positive else symmetric_info_nce
    if use_multi_positive:
        image_loss = contrast_fn(pred, image_target, [str(image_id) for image_id in batch["image_id"]], labels=labels, temperature=temperature)
        text_loss = contrast_fn(pred, text_target, [str(image_id) for image_id in batch["image_id"]], labels=labels, temperature=temperature)
    else:
        image_loss = contrast_fn(pred, image_target, temperature=temperature)
        text_loss = contrast_fn(pred, text_target, temperature=temperature)

    total = lambda_image * image_loss + lambda_text * text_loss
    parts = {
        "image_contrast": float(image_loss.detach().cpu()),
        "text_contrast": float(text_loss.detach().cpu()),
    }
    if logits is not None and lambda_cls > 0:
        cls = F.cross_entropy(logits, labels)
        total = total + lambda_cls * cls
        parts["cls"] = float(cls.detach().cpu())
    if lambda_proto > 0:
        proto = 0.5 * (
            prototype_alignment_loss(pred, image_target, labels)
            + prototype_alignment_loss(pred, text_target, labels)
        )
        total = total + lambda_proto * proto
        parts["prototype"] = float(proto.detach().cpu())
    parts["total"] = float(total.detach().cpu())
    return total, parts


def evaluate(model: EEGCLIPAlignmentModel, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    pred_chunks: list[torch.Tensor] = []
    image_chunks: list[torch.Tensor] = []
    text_chunks: list[torch.Tensor] = []
    image_ids: list[str] = []
    class_acc: list[float] = []
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device, non_blocking=True)
            pred, logits = _forward(model, eeg, batch.get("subject_id"))
            pred_chunks.append(pred.cpu())
            image_chunks.append(F.normalize(batch["image_emb"], dim=-1).cpu())
            text_chunks.append(F.normalize(batch["text_emb"], dim=-1).cpu())
            image_ids.extend(str(image_id) for image_id in batch["image_id"])
            if logits is not None:
                class_acc.append(class_accuracy_from_logits(logits, batch["label"].to(device)))
    pred_all = torch.cat(pred_chunks, dim=0)
    image_all = torch.cat(image_chunks, dim=0)
    text_all = torch.cat(text_chunks, dim=0)
    image_metrics = retrieval_metric_bundle(pred_all, image_all, image_ids)
    text_metrics = retrieval_metric_bundle(pred_all, text_all, image_ids)
    if class_acc:
        image_metrics["class_acc"] = float(np.mean(class_acc))
        text_metrics["class_acc"] = float(np.mean(class_acc))
    return {
        "image": image_metrics,
        "text": text_metrics,
        "random_image": image_metrics.get("random_unique_image", random_retrieval_metrics(pred_all.shape[0], query_ids=image_ids, target_ids=image_ids)),
        "random_text": text_metrics.get("random_unique_image", random_retrieval_metrics(pred_all.shape[0], query_ids=image_ids, target_ids=image_ids)),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    for path in [
        args.train_manifest,
        args.val_manifest,
        args.clip_train_cache,
        args.clip_val_cache,
        args.clip_index_train,
        args.clip_index_val,
        args.text_cache,
        args.text_index,
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Required tri-modal input missing: {path}")

    device = _resolve_device(args.device)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    eeg_shape = (args.eeg_channels, args.eeg_timesteps)
    train_loader = _build_loader(
        manifest=args.train_manifest,
        image_cache=args.clip_train_cache,
        image_index=args.clip_index_train,
        text_cache=args.text_cache,
        text_index=args.text_index,
        text_source=args.text_source,
        batch_size=args.batch_size,
        max_samples=args.max_train_samples,
        eeg_shape=eeg_shape,
        shuffle=True,
        num_workers=args.num_workers,
        eeg_cache=getattr(args, "eeg_train_cache", ""),
    )
    val_loader = _build_loader(
        manifest=args.val_manifest,
        image_cache=args.clip_val_cache,
        image_index=args.clip_index_val,
        text_cache=args.text_cache,
        text_index=args.text_index,
        text_source=args.text_source,
        batch_size=args.batch_size,
        max_samples=args.max_val_samples,
        eeg_shape=eeg_shape,
        shuffle=False,
        num_workers=args.num_workers,
        eeg_cache=getattr(args, "eeg_val_cache", ""),
    )

    model = EEGCLIPAlignmentModel(
        eeg_channels=args.eeg_channels,
        eeg_timesteps=args.eeg_timesteps,
        eeg_dim=args.eeg_embed_dim,
        clip_dim=args.clip_embed_dim,
        hidden_dim=args.hidden_dim,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
        num_classes=_num_classes(args.train_manifest),
        encoder_type=args.encoder_type,
    ).to(device)
    pretrained_report = load_pretrained_eeg_encoder(model, vars(args))
    if pretrained_report.get("loaded"):
        print(json.dumps({"pretrained_eeg": pretrained_report}, sort_keys=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    global_step = 0
    start_epoch = 0
    if getattr(args, "resume_checkpoint", ""):
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        previous_epoch = int(checkpoint.get("epoch", -1))
        start_epoch = previous_epoch + 1
        metrics = checkpoint.get("metrics", {})
        if isinstance(metrics, dict):
            image_metrics = metrics.get("image", {}) if isinstance(metrics.get("image"), dict) else {}
            text_metrics = metrics.get("text", {}) if isinstance(metrics.get("text"), dict) else {}
            best_score = float(image_metrics.get("r@5", 0.0)) + float(text_metrics.get("r@5", 0.0))
            best_epoch = previous_epoch + 1
        print(
            json.dumps(
                {
                    "resume": {
                        "checkpoint": str(args.resume_checkpoint),
                        "start_epoch": start_epoch + 1,
                        "previous_epoch": previous_epoch + 1,
                        "best_score_from_checkpoint": best_score,
                    }
                },
                sort_keys=True,
            )
        )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(train_loader, start=1):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss, parts = compute_trimodal_loss(
                    model,
                    batch,
                    device,
                    temperature=args.temperature,
                    lambda_image=args.lambda_image,
                    lambda_text=args.lambda_text,
                    lambda_cls=args.lambda_cls,
                    lambda_proto=args.lambda_proto,
                    use_multi_positive=args.use_multi_positive,
                )
                scaled = loss / args.grad_accum_steps
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite tri-modal alignment loss")
            scaled.backward()
            if micro_step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                epoch_losses.append(float(loss.detach().cpu()))
                if global_step % max(args.log_every, 1) == 0:
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"step": global_step, "epoch": epoch + 1, **parts}) + "\n")

        metrics = evaluate(model, val_loader, device)
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        score = float(metrics["image"].get("r@5", 0.0)) + float(metrics["text"].get("r@5", 0.0))
        record = {"epoch": epoch + 1, "loss": mean_loss, "metrics": metrics}
        history.append(record)
        checkpoint = {
            "model": model.state_dict(),
            "eeg_encoder": model.eeg_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
        }
        save_checkpoint(ckpt_dir / "last.pt", checkpoint)
        if score >= best_score:
            best_score = score
            best_epoch = epoch + 1
            stale_epochs = 0
            save_checkpoint(ckpt_dir / "best.pt", checkpoint)
        else:
            stale_epochs += 1
        print(
            f"epoch={epoch + 1} loss={mean_loss:.4f} "
            f"image_r5={metrics['image'].get('r@5', 0.0):.4f} text_r5={metrics['text'].get('r@5', 0.0):.4f}"
        )
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"early_stop epoch={epoch + 1} best_epoch={best_epoch} best_score={best_score:.4f}")
            break

    final_metrics = evaluate(model, val_loader, device)
    with (out_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, sort_keys=True)
    with (out_dir / "trimodal_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2, sort_keys=True)
    save_metrics_json(out_dir / "image_alignment_metrics.json", final_metrics["image"], final_metrics["random_image"])
    save_metrics_json(out_dir / "text_alignment_metrics.json", final_metrics["text"], final_metrics["random_text"])
    _write_status(out_dir / "TRIMODAL_STATUS.md", args, final_metrics, pretrained_report, best_epoch, best_score)
    _write_status(out_dir / "TRIMODAL_FULL_REPORT.md", args, final_metrics, pretrained_report, best_epoch, best_score)
    if out_dir.parent.name == "trimodal":
        _write_status(out_dir.parent / "TRIMODAL_FULL_REPORT.md", args, final_metrics, pretrained_report, best_epoch, best_score)
    return final_metrics


def _write_status(
    path: Path,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    pretrained_report: dict[str, Any],
    best_epoch: int,
    best_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tri-Modal Alignment Status",
        "",
        "## What Ran",
        "",
        f"- Train manifest: `{args.train_manifest}`",
        f"- Val manifest: `{args.val_manifest}`",
        f"- Text cache: `{args.text_cache}`",
        f"- Text index: `{args.text_index}`",
        f"- Text source: `{args.text_source}`",
        f"- Encoder: `{args.encoder_type}`",
        f"- Pretrained EEG checkpoint: `{args.pretrained_eeg_checkpoint or 'none'}`",
        f"- Pretrained load target: `{pretrained_report.get('target', 'none')}`",
        f"- Epochs: `{args.epochs}`",
        f"- Best epoch: `{best_epoch}`",
        f"- Best image+text R@5 score: `{best_score:.4f}`",
        f"- Max train samples: `{args.max_train_samples}`",
        f"- Max val samples: `{args.max_val_samples}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Gradient accumulation: `{args.grad_accum_steps}`",
        f"- Effective batch size: `{args.batch_size * args.grad_accum_steps}`",
        "",
        "## Validation Metrics",
        "",
        f"- EEG->image R@5: `{metrics['image'].get('r@5', 0.0):.4f}`",
        f"- EEG->text R@5: `{metrics['text'].get('r@5', 0.0):.4f}`",
        f"- Class accuracy: `{metrics['image'].get('class_acc', 0.0):.4f}`",
        "",
        "## Recipe Gap Appendix",
        "",
        "- This is tri-modal EEG-image-text contrastive training, not free-form Qwen caption training.",
        "- Promote this checkpoint only if validation retrieval/class metrics beat the historical alignment baseline.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _config_defaults(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    config = load_config(config_path)
    defaults: dict[str, Any] = {}
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    train_cfg = config.get("train", {})
    output_cfg = config.get("output", {})

    mapping = {
        "train_manifest": data_cfg.get("train_manifest"),
        "val_manifest": data_cfg.get("val_manifest"),
        "clip_train_cache": data_cfg.get("clip_train_cache"),
        "clip_val_cache": data_cfg.get("clip_val_cache"),
        "clip_index_train": data_cfg.get("clip_index_train"),
        "clip_index_val": data_cfg.get("clip_index_val"),
        "text_cache": data_cfg.get("text_cache"),
        "text_index": data_cfg.get("text_index"),
        "text_source": data_cfg.get("text_source"),
        "eeg_train_cache": data_cfg.get("eeg_train_cache"),
        "eeg_val_cache": data_cfg.get("eeg_val_cache"),
        "encoder_type": model_cfg.get("encoder_type"),
        "eeg_channels": model_cfg.get("eeg_channels"),
        "eeg_timesteps": model_cfg.get("eeg_time_steps", model_cfg.get("eeg_timesteps")),
        "eeg_embed_dim": model_cfg.get("eeg_embed_dim"),
        "clip_embed_dim": model_cfg.get("clip_embed_dim"),
        "hidden_dim": model_cfg.get("hidden_dim"),
        "transformer_layers": model_cfg.get("transformer_layers"),
        "dropout": model_cfg.get("dropout"),
        "pretrained_eeg_checkpoint": model_cfg.get("pretrained_eeg_checkpoint"),
        "pretrained_key": model_cfg.get("pretrained_key"),
        "pretrained_strict": model_cfg.get("pretrained_strict"),
        "temperature": loss_cfg.get("temperature"),
        "lambda_image": loss_cfg.get("lambda_image"),
        "lambda_text": loss_cfg.get("lambda_text"),
        "lambda_cls": loss_cfg.get("lambda_cls", loss_cfg.get("lambda_class_ce")),
        "lambda_proto": loss_cfg.get("lambda_proto"),
        "use_multi_positive": loss_cfg.get("use_multi_positive", loss_cfg.get("use_multi_positive_infonce")),
        "seed": config.get("seed"),
        "device": config.get("device"),
        "batch_size": train_cfg.get("batch_size"),
        "grad_accum_steps": train_cfg.get("grad_accum_steps"),
        "epochs": train_cfg.get("epochs"),
        "max_train_samples": train_cfg.get("max_train_samples"),
        "max_val_samples": train_cfg.get("max_val_samples"),
        "num_workers": train_cfg.get("num_workers"),
        "lr": train_cfg.get("lr"),
        "weight_decay": train_cfg.get("weight_decay"),
        "bf16": train_cfg.get("bf16"),
        "log_every": train_cfg.get("log_every"),
        "patience": train_cfg.get("patience"),
        "resume_checkpoint": train_cfg.get("resume_checkpoint"),
        "out_dir": output_cfg.get("dir"),
    }
    for key, value in mapping.items():
        if value is not None:
            defaults[key] = value
    return defaults


def parse_args() -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", default=None)
    known, _ = base.parse_known_args()
    defaults = _config_defaults(known.config)

    parser = argparse.ArgumentParser(
        description="Full tri-modal EEG-image-text contrastive alignment.",
        parents=[base],
    )
    parser.add_argument("--train_manifest", default=defaults.get("train_manifest", "data/thought2text/train.jsonl"))
    parser.add_argument("--val_manifest", default=defaults.get("val_manifest", "data/thought2text/val.jsonl"))
    parser.add_argument("--clip_train_cache", default=defaults.get("clip_train_cache", "data/thought2text/cache/clip_train.npy"))
    parser.add_argument("--clip_val_cache", default=defaults.get("clip_val_cache", "data/thought2text/cache/clip_val.npy"))
    parser.add_argument("--clip_index_train", default=defaults.get("clip_index_train", "data/thought2text/cache/clip_index_train.json"))
    parser.add_argument("--clip_index_val", default=defaults.get("clip_index_val", "data/thought2text/cache/clip_index_val.json"))
    parser.add_argument("--text_cache", default=defaults.get("text_cache", "data/thought2text/cache/text_embeddings.npy"))
    parser.add_argument("--text_index", default=defaults.get("text_index", "data/thought2text/cache/text_index.json"))
    parser.add_argument("--text_source", default=defaults.get("text_source", "human_class"))
    parser.add_argument("--eeg_train_cache", default=defaults.get("eeg_train_cache", ""))
    parser.add_argument("--eeg_val_cache", default=defaults.get("eeg_val_cache", ""))
    parser.add_argument("--out_dir", default=defaults.get("out_dir", "outputs/trimodal/full"))
    parser.add_argument("--device", default=defaults.get("device", "auto"))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 256))
    parser.add_argument("--grad_accum_steps", type=int, default=defaults.get("grad_accum_steps", 1))
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 100))
    parser.add_argument("--max_train_samples", type=int, default=defaults.get("max_train_samples", 0))
    parser.add_argument("--max_val_samples", type=int, default=defaults.get("max_val_samples", 0))
    parser.add_argument("--num_workers", type=int, default=defaults.get("num_workers", 8))
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 1e-4))
    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 0.05))
    parser.add_argument("--bf16", action="store_true", default=defaults.get("bf16", False))
    parser.add_argument("--encoder_type", default=defaults.get("encoder_type", "masked_pretrained"))
    parser.add_argument("--pretrained_eeg_checkpoint", default=defaults.get("pretrained_eeg_checkpoint", ""))
    parser.add_argument("--pretrained_key", default=defaults.get("pretrained_key", "eeg_encoder"))
    parser.add_argument("--pretrained_strict", action="store_true", default=defaults.get("pretrained_strict", False))
    parser.add_argument("--eeg_channels", type=int, default=defaults.get("eeg_channels", 64))
    parser.add_argument("--eeg_timesteps", type=int, default=defaults.get("eeg_timesteps", 250))
    parser.add_argument("--eeg_embed_dim", type=int, default=defaults.get("eeg_embed_dim", 512))
    parser.add_argument("--clip_embed_dim", type=int, default=defaults.get("clip_embed_dim", 512))
    parser.add_argument("--hidden_dim", type=int, default=defaults.get("hidden_dim", 512))
    parser.add_argument("--transformer_layers", type=int, default=defaults.get("transformer_layers", 8))
    parser.add_argument("--dropout", type=float, default=defaults.get("dropout", 0.15))
    parser.add_argument("--temperature", type=float, default=defaults.get("temperature", 0.07))
    parser.add_argument("--lambda_image", type=float, default=defaults.get("lambda_image", 1.0))
    parser.add_argument("--lambda_text", type=float, default=defaults.get("lambda_text", 1.0))
    parser.add_argument("--lambda_cls", type=float, default=defaults.get("lambda_cls", 0.3))
    parser.add_argument("--lambda_proto", type=float, default=defaults.get("lambda_proto", 0.2))
    parser.add_argument("--use_multi_positive", action="store_true", default=defaults.get("use_multi_positive", False))
    parser.add_argument("--log_every", type=int, default=defaults.get("log_every", 10))
    parser.add_argument("--patience", type=int, default=defaults.get("patience", 15))
    parser.add_argument("--resume_checkpoint", default=defaults.get("resume_checkpoint", ""))
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
