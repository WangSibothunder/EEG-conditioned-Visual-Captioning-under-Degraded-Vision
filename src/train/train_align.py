from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.clip_cache import load_cache
from src.data.dataset import EEGVisionCaptionDataset
from src.eval.retrieval import (
    class_accuracy_from_logits,
    random_retrieval_metrics,
    retrieval_metric_bundle,
    retrieval_metrics,
    save_metrics_json,
    save_retrieval_report,
)
from src.losses.contrastive import (
    multi_positive_info_nce,
    prototype_alignment_loss,
    same_image_subject_consistency_loss,
    supervised_contrastive_loss,
    symmetric_info_nce,
)
from src.losses.eeg_aug import augment_eeg
from src.losses.similarity import similarity_distillation_loss
from src.models.alignment_model import EEGCLIPAlignmentModel
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config
from src.utils.seed import seed_everything


class AlignmentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: EEGVisionCaptionDataset,
        clip_embeddings: torch.Tensor,
        eeg_cache: np.ndarray | None = None,
    ) -> None:
        if len(dataset) != clip_embeddings.shape[0]:
            raise ValueError(f"Dataset/cache length mismatch: {len(dataset)} vs {clip_embeddings.shape[0]}")
        if eeg_cache is not None and len(dataset) != int(eeg_cache.shape[0]):
            raise ValueError(f"Dataset/EEG cache length mismatch: {len(dataset)} vs {eeg_cache.shape[0]}")
        self.dataset = dataset
        self.clip_embeddings = clip_embeddings.float()
        self.eeg_cache = eeg_cache

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.eeg_cache is not None:
            row = self.dataset.rows[index]
            item = {
                "eeg": torch.from_numpy(np.asarray(self.eeg_cache[index], dtype=np.float32).copy()),
                "caption": str(row["caption"]),
                "image_id": str(row["image_id"]),
                "label": int(row["label"]),
                "subject_id": row.get("subject_id"),
            }
        else:
            item = self.dataset[index]
        item["clip_emb"] = self.clip_embeddings[index]
        return item


def alignment_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "eeg": torch.stack([item["eeg"] for item in batch], dim=0).float(),
        "caption": [str(item["caption"]) for item in batch],
        "image_id": [str(item["image_id"]) for item in batch],
        "label": torch.tensor([int(item["label"]) for item in batch], dtype=torch.long),
        "subject_id": [item.get("subject_id") for item in batch],
    }
    if "image" in batch[0]:
        out["image"] = torch.stack([item["image"] for item in batch], dim=0).float()
    out["clip_emb"] = torch.stack([item["clip_emb"] for item in batch], dim=0).float()
    return out


def subset_dataset(dataset: torch.utils.data.Dataset, max_samples: int | None) -> torch.utils.data.Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _num_classes(manifest: str | Path) -> int | None:
    rows = _read_rows(manifest)
    labels = [int(row["label"]) for row in rows if row.get("label") is not None]
    return max(labels) + 1 if labels else None


def _build_loader(
    *,
    manifest: str | Path,
    cache: str | Path,
    index: str | Path,
    batch_size: int,
    max_samples: int,
    eeg_shape: tuple[int, int],
    shuffle: bool,
    num_workers: int,
    eeg_cache: str | Path | None = None,
) -> DataLoader:
    clip_embeddings, _ = load_cache(cache, index)
    base = EEGVisionCaptionDataset(manifest, eeg_shape=eeg_shape, allow_missing_images=True)
    eeg_array = np.load(eeg_cache, mmap_mode="r") if eeg_cache else None
    dataset = subset_dataset(AlignmentDataset(base, clip_embeddings, eeg_array), max_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=alignment_collate,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(model: EEGCLIPAlignmentModel, loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    pred_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    class_acc_values: list[float] = []
    image_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device, non_blocking=True)
            target = F.normalize(batch["clip_emb"].to(device, non_blocking=True), dim=-1)
            labels = batch["label"].to(device, non_blocking=True)
            pred, logits = _model_forward(model, eeg, subject_ids=batch.get("subject_id"))
            pred_chunks.append(pred.cpu())
            target_chunks.append(target.cpu())
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
    return metrics, random_metrics


def _loss_with_guard(name: str, enabled: bool, fn, disabled: set[str]) -> torch.Tensor | None:
    if not enabled or name in disabled:
        return None
    try:
        loss = fn()
    except Exception:
        disabled.add(name)
        return None
    if not torch.isfinite(loss):
        disabled.add(name)
        return None
    return loss


def _model_forward(
    model: EEGCLIPAlignmentModel,
    eeg: torch.Tensor,
    subject_ids: list[str] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    try:
        return model(eeg, subject_ids=subject_ids)
    except TypeError:
        return model(eeg)


def _hard_negative_contrast_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    image_ids: list[str],
    *,
    tau: float = 0.07,
    top_k: int = 1,
    margin: float = 0.0,
) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    batch_size = pred.shape[0]
    if batch_size < 2:
        return pred.new_zeros(())

    image_similarity = target @ target.T
    eeg_similarity = pred @ pred.T
    label_mismatch = labels[:, None] != labels[None, :]
    image_mismatch = torch.tensor(
        [[image_ids[row] != image_ids[col] for col in range(batch_size)] for row in range(batch_size)],
        device=pred.device,
        dtype=torch.bool,
    )
    candidate_mask = label_mismatch & image_mismatch
    candidate_mask.fill_diagonal_(False)
    if not candidate_mask.any():
        return pred.new_zeros(())

    masked_image_similarity = image_similarity.masked_fill(~candidate_mask, -torch.inf)
    selected = masked_image_similarity.topk(k=max(1, min(int(top_k), batch_size - 1)), dim=-1).indices
    selected_mask = torch.zeros_like(candidate_mask)
    selected_mask.scatter_(1, selected, True)
    selected_mask &= candidate_mask
    if not selected_mask.any():
        return pred.new_zeros(())

    hard_scores = eeg_similarity[selected_mask] / max(float(tau), 1e-6)
    if margin > 0:
        hard_scores = hard_scores - float(margin)
    return F.softplus(hard_scores).mean()


def compute_alignment_loss(
    model: EEGCLIPAlignmentModel,
    batch: dict[str, Any],
    device: torch.device,
    loss_cfg: dict[str, Any],
    disabled_terms: set[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    eeg = batch["eeg"].to(device, non_blocking=True)
    target = F.normalize(batch["clip_emb"].to(device, non_blocking=True), dim=-1)
    labels = batch["label"].to(device, non_blocking=True)
    pred, logits = _model_forward(model, eeg, subject_ids=batch.get("subject_id"))

    total = torch.zeros((), device=device)
    parts: dict[str, float] = {}

    infonce = _loss_with_guard(
        "infonce",
        bool(loss_cfg.get("use_infonce", True)),
        lambda: symmetric_info_nce(pred, target, temperature=float(loss_cfg.get("temperature", 0.07))),
        disabled_terms,
    )
    if infonce is not None:
        total = total + float(loss_cfg.get("lambda_infonce", 1.0)) * infonce
        parts["infonce"] = float(infonce.detach().cpu())

    multi_positive = _loss_with_guard(
        "multi_positive",
        bool(loss_cfg.get("use_multi_positive_infonce", False)),
        lambda: multi_positive_info_nce(
            pred,
            target,
            [str(image_id) for image_id in batch["image_id"]],
            labels=labels,
            temperature=float(loss_cfg.get("temperature", 0.07)),
        )
        if batch.get("image_id") is not None
        else symmetric_info_nce(pred, target, temperature=float(loss_cfg.get("temperature", 0.07))),
        disabled_terms,
    )
    if multi_positive is not None:
        total = total + float(loss_cfg.get("lambda_multi_positive", loss_cfg.get("lambda_infonce", 1.0))) * multi_positive
        parts["multi_positive"] = float(multi_positive.detach().cpu())

    cosine = _loss_with_guard(
        "cosine",
        bool(loss_cfg.get("use_cosine", False)),
        lambda: 1.0 - F.cosine_similarity(pred, target, dim=-1).mean(),
        disabled_terms,
    )
    if cosine is not None:
        total = total + float(loss_cfg.get("lambda_cosine", 0.5)) * cosine
        parts["cosine"] = float(cosine.detach().cpu())

    mse = _loss_with_guard(
        "mse",
        bool(loss_cfg.get("use_mse", True)),
        lambda: F.mse_loss(pred, target),
        disabled_terms,
    )
    if mse is not None:
        total = total + float(loss_cfg.get("lambda_mse", 0.5)) * mse
        parts["mse"] = float(mse.detach().cpu())

    cls = _loss_with_guard(
        "cls",
        bool(loss_cfg.get("use_cls", loss_cfg.get("use_class_ce", True))) and logits is not None,
        lambda: F.cross_entropy(logits, labels),
        disabled_terms,
    )
    if cls is not None:
        total = total + float(loss_cfg.get("lambda_cls", loss_cfg.get("lambda_class_ce", 0.2))) * cls
        parts["cls"] = float(cls.detach().cpu())

    supcon = _loss_with_guard(
        "supcon",
        bool(loss_cfg.get("use_supcon", loss_cfg.get("use_supervised_contrastive", False))),
        lambda: supervised_contrastive_loss(pred, labels, temperature=float(loss_cfg.get("temperature", 0.07))),
        disabled_terms,
    )
    if supcon is not None:
        total = total + float(loss_cfg.get("lambda_supcon", 0.2)) * supcon
        parts["supcon"] = float(supcon.detach().cpu())

    sim = _loss_with_guard(
        "similarity",
        bool(loss_cfg.get("use_similarity_distill", loss_cfg.get("use_similarity_distillation", True))),
        lambda: similarity_distillation_loss(pred, target, tau=float(loss_cfg.get("tau_sim", 0.1))),
        disabled_terms,
    )
    if sim is not None:
        total = total + float(loss_cfg.get("lambda_sim", loss_cfg.get("lambda_similarity", 0.2))) * sim
        parts["similarity"] = float(sim.detach().cpu())

    proto = _loss_with_guard(
        "prototype",
        bool(loss_cfg.get("use_prototype_alignment", False)),
        lambda: prototype_alignment_loss(pred, target, labels),
        disabled_terms,
    )
    if proto is not None:
        total = total + float(loss_cfg.get("lambda_proto", 0.2)) * proto
        parts["prototype"] = float(proto.detach().cpu())

    hard_negative = _loss_with_guard(
        "hard_negative",
        bool(loss_cfg.get("use_hard_negative", False)),
        lambda: _hard_negative_contrast_loss(
            pred,
            target,
            labels,
            [str(image_id) for image_id in batch.get("image_id", [])],
            tau=float(loss_cfg.get("temperature", 0.07)),
            top_k=int(loss_cfg.get("hard_negative_top_k", 1)),
            margin=float(loss_cfg.get("hard_negative_margin", 0.0)),
        ),
        disabled_terms,
    )
    if hard_negative is not None:
        total = total + float(loss_cfg.get("lambda_hard_negative", 0.05)) * hard_negative
        parts["hard_negative"] = float(hard_negative.detach().cpu())

    same_image_subject = _loss_with_guard(
        "same_image_subject",
        bool(loss_cfg.get("use_same_image_subject", loss_cfg.get("use_same_image_subject_consistency", False))),
        lambda: same_image_subject_consistency_loss(
            pred,
            [str(image_id) for image_id in batch["image_id"]],
            subject_ids=[str(subject_id) for subject_id in batch.get("subject_id", [])] if batch.get("subject_id") is not None else None,
        ),
        disabled_terms,
    )
    if same_image_subject is not None:
        total = total + float(loss_cfg.get("lambda_same_image_subject", 0.2)) * same_image_subject
        parts["same_image_subject"] = float(same_image_subject.detach().cpu())

    aug = _loss_with_guard(
        "augmentation",
        bool(loss_cfg.get("use_aug_consistency", True)),
        lambda: 1.0
        - F.cosine_similarity(
            _model_forward(model, augment_eeg(eeg), subject_ids=batch.get("subject_id"))[0],
            _model_forward(model, augment_eeg(eeg), subject_ids=batch.get("subject_id"))[0],
            dim=-1,
        ).mean(),
        disabled_terms,
    )
    if aug is not None:
        total = total + float(loss_cfg.get("lambda_aug", 0.1)) * aug
        parts["augmentation"] = float(aug.detach().cpu())

    parts["total"] = float(total.detach().cpu())
    return total, parts


def _default_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed": args.seed,
        "data": {
            "train_manifest": args.train_manifest,
            "val_manifest": args.val_manifest,
            "clip_train_cache": args.train_cache,
            "clip_val_cache": args.val_cache,
            "clip_index_train": args.train_index,
            "clip_index_val": args.val_index,
        },
        "model": {
            "eeg_embed_dim": 512,
            "clip_embed_dim": 512,
            "eeg_channels": args.eeg_channels,
            "eeg_time_steps": args.eeg_timesteps,
            "hidden_dim": 128,
            "transformer_layers": 2,
            "dropout": 0.1,
        },
        "loss": {
            "use_infonce": True,
            "use_mse": True,
            "use_cls": True,
            "use_similarity_distill": False,
            "use_aug_consistency": False,
            "temperature": 0.07,
            "lambda_mse": 0.5,
            "lambda_cls": 0.2,
        },
        "train": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": 0.01,
            "bf16": True,
            "num_workers": 0,
            "log_every": 20,
            "patience": 0,
        },
        "output": {"dir": args.out_dir},
    }


def _config_value(config: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    value = config.get(section, {}).get(key, default)
    return default if value is None else value


def load_pretrained_eeg_encoder(model: EEGCLIPAlignmentModel, model_cfg: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = model_cfg.get("pretrained_eeg_checkpoint")
    if not checkpoint_path:
        return {"loaded": False, "reason": "no pretrained_eeg_checkpoint"}

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"pretrained_eeg_checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu")
    key = str(model_cfg.get("pretrained_key", "eeg_encoder"))
    if isinstance(payload, dict) and key in payload:
        state_dict = payload[key]
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"pretrained checkpoint `{path}` key `{key}` is not a state_dict")

    candidates: list[tuple[str, torch.nn.Module]] = [("eeg_encoder", model.eeg_encoder)]
    autoencoder = getattr(model.eeg_encoder, "autoencoder", None)
    if isinstance(autoencoder, torch.nn.Module):
        candidates.append(("eeg_encoder.autoencoder", autoencoder))

    def match_count(module: torch.nn.Module) -> int:
        module_keys = set(module.state_dict().keys())
        return sum(1 for state_key in state_dict if state_key in module_keys)

    target_name, target_module = max(candidates, key=lambda item: match_count(item[1]))
    if match_count(target_module) == 0:
        raise RuntimeError(f"No matching EEG encoder keys found in pretrained checkpoint: {path}")

    strict = bool(model_cfg.get("pretrained_strict", False))
    allow_shape_mismatch = bool(model_cfg.get("pretrained_allow_shape_mismatch", False))
    loaded_keys: list[str] = []
    skipped_shape_mismatch: list[str] = []
    unexpected_keys: list[str] = []

    if allow_shape_mismatch:
        target_state = target_module.state_dict()
        filtered_state: dict[str, torch.Tensor] = {}
        for state_key, tensor in state_dict.items():
            if state_key not in target_state:
                unexpected_keys.append(state_key)
                continue
            if tuple(target_state[state_key].shape) != tuple(tensor.shape):
                skipped_shape_mismatch.append(state_key)
                continue
            filtered_state[state_key] = tensor
            loaded_keys.append(state_key)
        if not loaded_keys:
            raise RuntimeError(f"No shape-compatible EEG encoder keys found in pretrained checkpoint: {path}")
        incompatible = target_module.load_state_dict(filtered_state, strict=False)
    else:
        incompatible = target_module.load_state_dict(state_dict, strict=strict)
        loaded_keys = [key for key in state_dict.keys() if key in target_module.state_dict()]

    return {
        "loaded": True,
        "checkpoint": str(path),
        "key": key,
        "target": target_name,
        "strict": strict if not allow_shape_mismatch else False,
        "allow_shape_mismatch": allow_shape_mismatch,
        "loaded_key_count": len(loaded_keys),
        "loaded_keys": loaded_keys,
        "skipped_shape_mismatch_count": len(skipped_shape_mismatch),
        "skipped_shape_mismatch": skipped_shape_mismatch,
        "missing_keys": list(getattr(incompatible, "missing_keys", [])),
        "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])) + unexpected_keys,
    }


def _validate_required_files(paths: list[str | Path]) -> None:
    for path in paths:
        if not Path(path).exists():
            print(f"ERROR: required file missing: {path}")
            if "cache" in str(path):
                print(
                    "Run `python scripts/precompute_vision.py --manifest data/thought2text/train.jsonl "
                    "--image_root data/thought2text --out data/thought2text/cache/clip_train.npy "
                    "--index_out data/thought2text/cache/clip_index_train.json` first."
                )
                print("If this reports missing images, place the ImageNet files under `data/thought2text/images/`.")
            raise SystemExit(2)


def train(config: dict[str, Any], *, max_train_samples: int = 0, max_val_samples: int = 0, epochs_override: int | None = None, output_dir: str | None = None) -> None:
    seed_everything(int(config.get("seed", 42)))
    torch.backends.cudnn.benchmark = True

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    loss_cfg = config["loss"]
    out_dir = Path(output_dir or _config_value(config, "output", "dir", "outputs/align"))
    ckpt_dir = out_dir / "checkpoints"
    log_path = out_dir / "train_log.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_index = data_cfg.get("clip_index_train", "data/thought2text/cache/clip_index_train.json")
    val_index = data_cfg.get("clip_index_val", "data/thought2text/cache/clip_index_val.json")
    _validate_required_files(
        [
            data_cfg["train_manifest"],
            data_cfg["val_manifest"],
            data_cfg["clip_train_cache"],
            data_cfg["clip_val_cache"],
            train_index,
            val_index,
        ]
    )

    device = _resolve_device(str(config.get("device", "auto")))
    eeg_shape = (int(model_cfg.get("eeg_channels", 64)), int(model_cfg.get("eeg_time_steps", 250)))
    batch_size = int(train_cfg.get("batch_size", 128))
    train_loader = _build_loader(
        manifest=data_cfg["train_manifest"],
        cache=data_cfg["clip_train_cache"],
        index=train_index,
        batch_size=batch_size,
        max_samples=max_train_samples,
        eeg_shape=eeg_shape,
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        eeg_cache=data_cfg.get("eeg_train_cache"),
    )
    val_loader = _build_loader(
        manifest=data_cfg["val_manifest"],
        cache=data_cfg["clip_val_cache"],
        index=val_index,
        batch_size=batch_size,
        max_samples=max_val_samples,
        eeg_shape=eeg_shape,
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        eeg_cache=data_cfg.get("eeg_val_cache"),
    )

    num_classes = _num_classes(data_cfg["train_manifest"]) if bool(loss_cfg.get("use_cls", True)) else None
    model = EEGCLIPAlignmentModel(
        eeg_channels=eeg_shape[0],
        eeg_timesteps=eeg_shape[1],
        eeg_dim=int(model_cfg.get("eeg_embed_dim", 512)),
        clip_dim=int(model_cfg.get("clip_embed_dim", 512)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        transformer_layers=int(model_cfg.get("transformer_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_classes=num_classes,
        encoder_type=str(model_cfg.get("encoder_type", "tiny")),
    ).to(device)
    pretrained_report = load_pretrained_eeg_encoder(model, model_cfg)
    if pretrained_report["loaded"]:
        print(json.dumps({"pretrained_eeg": pretrained_report}, sort_keys=True))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    use_amp = bool(train_cfg.get("bf16", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    epochs = int(epochs_override if epochs_override is not None else train_cfg.get("epochs", 80))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    patience = int(train_cfg.get("patience", 0))
    log_every = int(train_cfg.get("log_every", 20))

    best_r5 = -1.0
    best_epoch = -1
    disabled_terms: set[str] = set()
    global_step = 0
    history: list[dict[str, Any]] = []
    start_epoch = 0

    resume_path = train_cfg.get("resume_checkpoint")
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        previous_epoch = int(checkpoint.get("epoch", -1))
        start_epoch = previous_epoch + 1
        metrics = checkpoint.get("metrics", {})
        if isinstance(metrics, dict):
            best_r5 = float(metrics.get("r@5", best_r5))
            best_epoch = previous_epoch
        disabled_terms = set(checkpoint.get("disabled_terms", []))
        print(
            json.dumps(
                {
                    "resume": {
                        "checkpoint": str(resume_path),
                        "start_epoch": start_epoch + 1,
                        "previous_epoch": previous_epoch + 1,
                        "best_r5_from_checkpoint": best_r5,
                    }
                },
                sort_keys=True,
            )
        )

    for epoch in range(start_epoch, epochs):
        model.train()
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(train_loader, start=1):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss, parts = compute_alignment_loss(model, batch, device, loss_cfg, disabled_terms)
                scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()
            if micro_step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                losses.append(float(loss.detach().cpu()))
                if global_step % max(log_every, 1) == 0:
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"step": global_step, "epoch": epoch + 1, **parts}) + "\n")

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite alignment loss after guarded term handling")

        metrics, random_metrics = evaluate(model, val_loader, device)
        mean_loss = float(np.mean(losses)) if losses else 0.0
        epoch_record = {
            "epoch": epoch + 1,
            "loss": mean_loss,
            "metrics": metrics,
            "random": random_metrics,
            "disabled_terms": sorted(disabled_terms),
        }
        history.append(epoch_record)
        print(f"epoch={epoch + 1} loss={mean_loss:.4f} r@1={metrics['r@1']:.4f} r@5={metrics['r@5']:.4f}")
        checkpoint = {
            "model": model.state_dict(),
            "eeg_encoder": model.eeg_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "random_metrics": random_metrics,
            "config": config,
            "disabled_terms": sorted(disabled_terms),
        }
        save_checkpoint(ckpt_dir / "last.pt", checkpoint)
        if metrics["r@5"] >= best_r5:
            best_r5 = metrics["r@5"]
            best_epoch = epoch
            save_checkpoint(ckpt_dir / "best.pt", checkpoint)

        if patience > 0 and epoch - best_epoch >= patience:
            break

    metrics, random_metrics = evaluate(model, val_loader, device)
    save_metrics_json(out_dir / "retrieval_metrics.json", metrics, random_metrics)
    save_retrieval_report(out_dir / "retrieval_report.md", metrics, random_metrics)
    save_metrics_json(out_dir / "alignment_metrics.json", metrics, random_metrics)
    save_retrieval_report(out_dir / "alignment_report.md", metrics, random_metrics)
    save_metrics_json("outputs/alignment_metrics.json", metrics, random_metrics)
    save_retrieval_report("outputs/alignment_report.md", metrics, random_metrics)
    with (out_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EEG -> CLIP alignment.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--train_manifest", default="data/thought2text/train.jsonl")
    parser.add_argument("--val_manifest", default="data/thought2text/val.jsonl")
    parser.add_argument("--train_cache", default="data/thought2text/cache/clip_train.npy")
    parser.add_argument("--val_cache", default="data/thought2text/cache/clip_val.npy")
    parser.add_argument("--train_index", default="data/thought2text/cache/clip_index_train.json")
    parser.add_argument("--val_index", default="data/thought2text/cache/clip_index_val.json")
    parser.add_argument("--out_dir", "--output_dir", dest="out_dir", default="outputs/align")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--eeg_channels", type=int, default=64)
    parser.add_argument("--eeg_timesteps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else _default_config_from_args(args)
    if args.config and args.out_dir != "outputs/align":
        config.setdefault("output", {})["dir"] = args.out_dir
    train(
        config,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        epochs_override=args.epochs,
        output_dir=args.out_dir if args.config and args.out_dir != "outputs/align" else None,
    )


if __name__ == "__main__":
    main()
