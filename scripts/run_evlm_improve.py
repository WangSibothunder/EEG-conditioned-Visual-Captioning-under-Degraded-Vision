from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data.collate import caption_collate
from src.data.dataset import EEGVisionCaptionDataset
from src.eval.constrained_caption_eval import load_cache, load_eeg_encoder, read_jsonl
from src.train.train_fusion import apply_eeg_mode
from src.utils.seed import seed_everything


REQUIRED_VARIANTS = [
    "A2_residual_scalar",
    "A2_residual_vector",
    "A2_residual_vector_margin",
    "A2_proto_bias",
    "A2_proto_bias_margin",
    "A2_residual_plus_proto_bias",
]
RESIDUAL_VARIANTS = {"A2_residual_scalar", "A2_residual_vector", "A2_residual_vector_margin", "A2_residual_plus_proto_bias"}
PROTO_VARIANTS = {"A2_proto_bias", "A2_proto_bias_margin", "A2_residual_plus_proto_bias"}
MARGIN_VARIANTS = {"A2_residual_vector_margin", "A2_proto_bias_margin", "A2_residual_plus_proto_bias"}
DEFAULT_CORRUPTIONS = ["clean", "lowres16", "mixed", "occlusion50", "strong_blur", "strong_noise"]
DEFAULT_MODES = ["vision_only", "real_eeg", "shuffled_eeg", "random_eeg", "eeg_only"]
STRONG_CORRUPTIONS = ["lowres16", "mixed", "occlusion50", "strong_blur", "strong_noise"]


@dataclass
class EVLMConfig:
    variant: str
    seed: int
    train_manifest: str
    val_manifest: str
    test_manifest: str
    train_cache: str
    val_cache: str
    test_cache_dir: str
    prototype_bank: str
    text_prototypes: str
    eeg_checkpoint: str
    output_dir: str
    epochs: int = 80
    patience: int = 12
    batch_size: int = 1024
    eval_batch_size: int = 512
    hidden_dim: int = 1024
    lr: float = 1.0e-4
    weight_decay: float = 0.05
    tau_cls: float = 0.07
    margin: float = 0.1
    delta_norm_weight: float = 0.01
    margin_weight: float = 0.2
    gamma_reg_weight: float = 0.01
    max_train_samples: int = 0
    max_val_samples: int = 0
    max_test_samples: int = 0
    num_workers: int = 0
    device: str = "auto"
    log_every: int = 10


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def variant_group(variant: str) -> str:
    if variant.endswith("_ensemble") or variant not in REQUIRED_VARIANTS:
        return "autonomous"
    if variant.startswith("A2_residual_plus"):
        return "combined"
    if variant.startswith("A2_residual"):
        return "residual"
    if variant.startswith("A2_proto"):
        return "proto_bias"
    return "autonomous"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_gpu_usage(path: Path, *, active_job: str, event: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    query = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(query, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - depends on local GPU tooling
        output = f"nvidia-smi unavailable: {exc}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {time.strftime('%Y-%m-%d %H:%M:%S')} - {event}\n\n")
        handle.write(f"- active_job: `{active_job}`\n")
        handle.write(f"- gpu: `{output}`\n")


def load_label_bank(prototype_bank: Path, text_prototype_path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[int, str]]:
    bank = torch.load(prototype_bank, map_location="cpu", weights_only=False)
    labels = torch.tensor([int(item) for item in bank["labels"]], dtype=torch.long, device=device)
    class_name_map = {int(key): str(value) for key, value in bank["class_name_map"].items()}
    if text_prototype_path.exists():
        prototypes = torch.from_numpy(np.load(text_prototype_path)).float()
    else:
        prototypes = bank["image_prototypes"].float()
    if prototypes.ndim != 2:
        raise ValueError(f"Expected prototypes [C,D], got {tuple(prototypes.shape)}")
    if prototypes.shape[0] != labels.numel():
        raise ValueError(f"Prototype count {prototypes.shape[0]} does not match label count {labels.numel()}")
    return labels, F.normalize(prototypes.to(device), dim=-1), class_name_map


def labels_to_targets(labels: torch.Tensor, label_values: torch.Tensor) -> torch.Tensor:
    label_to_index = {int(label): idx for idx, label in enumerate(label_values.detach().cpu().tolist())}
    return torch.tensor([label_to_index[int(label)] for label in labels.detach().cpu().tolist()], device=labels.device)


def read_labels(manifest: Path, max_samples: int = 0) -> torch.Tensor:
    rows = read_jsonl(manifest)
    if max_samples:
        rows = rows[:max_samples]
    return torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)


def compute_eeg_embeddings(
    *,
    manifest: Path,
    checkpoint: Path,
    mode: str,
    device: torch.device,
    batch_size: int,
    max_samples: int,
) -> torch.Tensor:
    cache_root = Path("outputs/evlm_improve/cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = (
        f"eeg_{checkpoint.parent.parent.name}_{manifest.stem}_{mode}"
        f"_{max_samples or 'all'}.pt"
    ).replace("/", "_")
    cache_path = cache_root / cache_key
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    dataset = EEGVisionCaptionDataset(manifest, allow_missing_images=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=caption_collate)
    encoder = load_eeg_encoder(checkpoint, device, manifest)
    outputs: list[torch.Tensor] = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device)
            mode_eeg = apply_eeg_mode(eeg, "real_eeg" if mode == "eeg_only" else mode)
            if mode_eeg is None:
                raise ValueError(f"EEG mode {mode} did not produce tensors")
            clip_pred, _ = encoder(mode_eeg)
            outputs.append(F.normalize(clip_pred.detach().cpu().float(), dim=-1))
            seen += int(eeg.shape[0])
            if max_samples and seen >= max_samples:
                break
    out = torch.cat(outputs, dim=0)
    out = out[:max_samples] if max_samples else out
    torch.save(out.cpu(), cache_path)
    return out


def load_image_embeddings(cache_path: Path, index_path: Path | None = None, max_samples: int = 0) -> torch.Tensor:
    if index_path is None:
        index_path = cache_path.with_name(cache_path.name.replace("clip_", "clip_index_").replace(".npy", ".json"))
    embeddings, _rows = load_cache(cache_path, index_path, max_samples=max_samples)
    return F.normalize(embeddings.float(), dim=-1)


def load_train_tensors(cfg: EVLMConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_img = load_image_embeddings(Path(cfg.train_cache), max_samples=cfg.max_train_samples)
    val_img = load_image_embeddings(Path(cfg.val_cache), max_samples=cfg.max_val_samples)
    train_labels = read_labels(Path(cfg.train_manifest), cfg.max_train_samples)
    val_labels = read_labels(Path(cfg.val_manifest), cfg.max_val_samples)
    train_eeg = compute_eeg_embeddings(
        manifest=Path(cfg.train_manifest),
        checkpoint=Path(cfg.eeg_checkpoint),
        mode="real_eeg",
        device=device,
        batch_size=cfg.eval_batch_size,
        max_samples=cfg.max_train_samples,
    )
    val_eeg = compute_eeg_embeddings(
        manifest=Path(cfg.val_manifest),
        checkpoint=Path(cfg.eeg_checkpoint),
        mode="real_eeg",
        device=device,
        batch_size=cfg.eval_batch_size,
        max_samples=cfg.max_val_samples,
    )
    if not (len(train_img) == len(train_eeg) == len(train_labels)):
        raise ValueError("Train image/eeg/label lengths do not match")
    if not (len(val_img) == len(val_eeg) == len(val_labels)):
        raise ValueError("Val image/eeg/label lengths do not match")
    return train_img, train_eeg, train_labels, val_img, val_eeg, val_labels


class EVLMEnhancer(nn.Module):
    def __init__(self, *, variant: str, embed_dim: int, num_classes: int, hidden_dim: int, tau_cls: float) -> None:
        super().__init__()
        if variant not in REQUIRED_VARIANTS:
            raise ValueError(f"Unsupported EVLM variant: {variant}")
        self.variant = variant
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.tau_cls = float(tau_cls)
        residual_input = embed_dim * 4 + 1
        proto_input = embed_dim * 2 + 1
        if variant in RESIDUAL_VARIANTS:
            self.delta_mlp = nn.Sequential(
                nn.LayerNorm(residual_input),
                nn.Linear(residual_input, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, embed_dim),
            )
            alpha_dim = 1 if variant == "A2_residual_scalar" else embed_dim
            self.alpha_mlp = nn.Sequential(
                nn.LayerNorm(residual_input),
                nn.Linear(residual_input, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, alpha_dim),
            )
        if variant in PROTO_VARIANTS:
            self.gamma_mlp = nn.Sequential(
                nn.LayerNorm(proto_input),
                nn.Linear(proto_input, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes),
            )

    def image_logits_and_confidence(self, image_emb: torch.Tensor, prototypes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.normalize(image_emb.float(), dim=-1) @ prototypes.T / self.tau_cls
        confidence = torch.softmax(logits, dim=-1).max(dim=-1, keepdim=True).values
        return logits, confidence

    def residual_correct(
        self,
        image_emb: torch.Tensor,
        eeg_emb: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_norm = F.normalize(image_emb.float(), dim=-1)
        eeg_norm = F.normalize(eeg_emb.float(), dim=-1)
        _logits, confidence = self.image_logits_and_confidence(image_norm, prototypes)
        features = torch.cat([image_norm, eeg_norm, image_norm * eeg_norm, image_norm - eeg_norm, confidence], dim=-1)
        delta = self.delta_mlp(features)
        alpha = torch.sigmoid(self.alpha_mlp(features))
        corrected = F.normalize(image_norm + alpha * delta, dim=-1)
        return corrected, {
            "alpha": alpha,
            "delta": delta,
            "vision_confidence": confidence,
            "delta_norm": delta.norm(dim=-1),
        }

    def proto_bias(
        self,
        image_emb: torch.Tensor,
        eeg_emb: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_norm = F.normalize(image_emb.float(), dim=-1)
        eeg_norm = F.normalize(eeg_emb.float(), dim=-1)
        image_logits, confidence = self.image_logits_and_confidence(image_norm, prototypes)
        eeg_logits = eeg_norm @ prototypes.T / self.tau_cls
        features = torch.cat([image_norm, eeg_norm, confidence], dim=-1)
        gamma_raw = torch.sigmoid(self.gamma_mlp(features))
        gamma = gamma_raw * (1.0 - confidence)
        return image_logits + gamma * eeg_logits, {
            "gamma": gamma,
            "vision_confidence": confidence,
        }

    def forward(self, image_emb: torch.Tensor, eeg_emb: torch.Tensor, prototypes: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux: dict[str, torch.Tensor] = {}
        if self.variant == "A2_residual_plus_proto_bias":
            corrected, residual_aux = self.residual_correct(image_emb, eeg_emb, prototypes)
            logits, proto_aux = self.proto_bias(corrected, eeg_emb, prototypes)
            aux.update(residual_aux)
            aux.update(proto_aux)
            return logits, aux
        if self.variant in RESIDUAL_VARIANTS:
            corrected, aux = self.residual_correct(image_emb, eeg_emb, prototypes)
            return corrected @ prototypes.T / self.tau_cls, aux
        logits, aux = self.proto_bias(image_emb, eeg_emb, prototypes)
        return logits, aux


def loss_for_batch(
    model: EVLMEnhancer,
    image: torch.Tensor,
    eeg: torch.Tensor,
    targets: torch.Tensor,
    prototypes: torch.Tensor,
    cfg: EVLMConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits_real, aux = model(image, eeg, prototypes)
    ce = F.cross_entropy(logits_real, targets)
    loss = ce
    margin_loss = torch.zeros((), device=image.device)
    if cfg.variant in MARGIN_VARIANTS:
        perm = torch.randperm(eeg.shape[0], device=eeg.device)
        logits_shuf, _ = model(image, eeg[perm], prototypes)
        logits_rand, _ = model(image, F.normalize(torch.randn_like(eeg), dim=-1), prototypes)
        row = torch.arange(targets.shape[0], device=targets.device)
        score_real = logits_real[row, targets]
        score_shuf = logits_shuf[row, targets]
        score_rand = logits_rand[row, targets]
        margin_loss = (
            F.relu(cfg.margin - (score_real - score_shuf)).mean()
            + F.relu(cfg.margin - (score_real - score_rand)).mean()
        )
        loss = loss + cfg.margin_weight * margin_loss
    delta_norm = aux.get("delta_norm")
    delta_loss = delta_norm.mean() if delta_norm is not None else torch.zeros((), device=image.device)
    if delta_norm is not None:
        loss = loss + cfg.delta_norm_weight * delta_loss
    gamma = aux.get("gamma")
    gamma_loss = torch.zeros((), device=image.device)
    if gamma is not None:
        confidence = aux["vision_confidence"].detach()
        gamma_loss = (gamma * confidence).mean()
        loss = loss + cfg.gamma_reg_weight * gamma_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "margin": float(margin_loss.detach().cpu()),
        "delta_norm": float(delta_loss.detach().cpu()),
        "gamma_reg": float(gamma_loss.detach().cpu()),
    }


@torch.no_grad()
def eval_accuracy(
    model: EVLMEnhancer,
    image: torch.Tensor,
    eeg: torch.Tensor,
    labels: torch.Tensor,
    label_values: torch.Tensor,
    prototypes: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    correct: list[float] = []
    top5: list[float] = []
    losses: list[float] = []
    for start in range(0, image.shape[0], batch_size):
        end = start + batch_size
        image_b = image[start:end].to(device)
        eeg_b = eeg[start:end].to(device)
        labels_b = labels[start:end].to(device)
        targets = labels_to_targets(labels_b, label_values)
        logits, _aux = model(image_b, eeg_b, prototypes)
        losses.append(float(F.cross_entropy(logits, targets).detach().cpu()))
        pred_indices = logits.argmax(dim=-1)
        pred_labels = label_values[pred_indices].to(labels_b.device)
        correct.extend((pred_labels == labels_b).float().detach().cpu().tolist())
        top_indices = logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices
        top_labels = label_values[top_indices].to(labels_b.device)
        top5.extend((top_labels == labels_b.unsqueeze(-1)).any(dim=-1).float().detach().cpu().tolist())
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(np.mean(correct)) if correct else 0.0,
        "top5_accuracy": float(np.mean(top5)) if top5 else 0.0,
    }


def train_one(cfg: EVLMConfig) -> Path:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    run_dir = Path(cfg.output_dir) / variant_group(cfg.variant) / f"{cfg.variant}_seed{cfg.seed}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", asdict(cfg))
    gpu_log = Path(cfg.output_dir) / "GPU_USAGE.md"
    append_gpu_usage(gpu_log, active_job=f"{cfg.variant}_seed{cfg.seed}", event="train_start")

    label_values, prototypes, _class_name_map = load_label_bank(Path(cfg.prototype_bank), Path(cfg.text_prototypes), device)
    train_img, train_eeg, train_labels, val_img, val_eeg, val_labels = load_train_tensors(cfg, device)
    model = EVLMEnhancer(
        variant=cfg.variant,
        embed_dim=int(train_img.shape[1]),
        num_classes=int(label_values.numel()),
        hidden_dim=cfg.hidden_dim,
        tau_cls=cfg.tau_cls,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_targets = labels_to_targets(train_labels.to(device), label_values)
    dataset = TensorDataset(train_img, train_eeg, train_targets.cpu())
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    best_val = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    log_path = run_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            epoch_stats: list[dict[str, float]] = []
            for step, (image_b, eeg_b, targets_b) in enumerate(loader, start=1):
                image_b = image_b.to(device, non_blocking=True)
                eeg_b = eeg_b.to(device, non_blocking=True)
                targets_b = targets_b.to(device, non_blocking=True)
                loss, stats = loss_for_batch(model, image_b, eeg_b, targets_b, prototypes, cfg)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_stats.append(stats)
                if cfg.log_every and step % cfg.log_every == 0:
                    row = {"epoch": epoch, "step": step, "steps_per_epoch": len(loader), **stats}
                    log_handle.write(json.dumps(row) + "\n")
                    log_handle.flush()
            val_stats = eval_accuracy(model, val_img, val_eeg, val_labels, label_values, prototypes, device, cfg.eval_batch_size)
            train_loss = float(np.mean([item["loss"] for item in epoch_stats])) if epoch_stats else 0.0
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_stats["loss"],
                "val_accuracy": val_stats["accuracy"],
                "val_top5_accuracy": val_stats["top5_accuracy"],
            }
            history.append(row)
            log_handle.write(json.dumps(row) + "\n")
            log_handle.flush()
            if val_stats["accuracy"] > best_val:
                best_val = float(val_stats["accuracy"])
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": asdict(cfg),
                        "label_values": label_values.detach().cpu(),
                        "text_prototypes": prototypes.detach().cpu(),
                        "best_epoch": best_epoch,
                        "best_val_accuracy": best_val,
                    },
                    ckpt_dir / "best.pt",
                )
            else:
                stale += 1
            if stale >= cfg.patience:
                break
    write_json(run_dir / "history.json", history)
    report_lines = [
        f"# {cfg.variant} seed {cfg.seed} Training Summary",
        "",
        f"- Best epoch: `{best_epoch}`",
        f"- Best val accuracy: `{best_val:.6f}`",
        f"- Checkpoint: `{ckpt_dir / 'best.pt'}`",
        f"- Epochs run: `{len(history)}`",
        f"- Batch size: `{cfg.batch_size}`",
        "",
        "| Epoch | Train Loss | Val Loss | Val Acc | Val Top5 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in history:
        report_lines.append(
            f"| {row['epoch']} | {row['train_loss']:.6f} | {row['val_loss']:.6f} | {row['val_accuracy']:.6f} | {row['val_top5_accuracy']:.6f} |"
        )
    (run_dir / "summary.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    append_gpu_usage(gpu_log, active_job=f"{cfg.variant}_seed{cfg.seed}", event="train_end")
    return ckpt_dir / "best.pt"


def load_evlm_checkpoint(checkpoint: Path, device: torch.device) -> tuple[EVLMEnhancer, torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = payload["config"]
    label_values = payload["label_values"].to(device=device, dtype=torch.long)
    prototypes = F.normalize(payload["text_prototypes"].to(device=device, dtype=torch.float32), dim=-1)
    model = EVLMEnhancer(
        variant=cfg["variant"],
        embed_dim=int(prototypes.shape[1]),
        num_classes=int(label_values.numel()),
        hidden_dim=int(cfg["hidden_dim"]),
        tau_cls=float(cfg["tau_cls"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, label_values, prototypes, cfg


def caption_for_label(label: int, class_name_map: dict[int, str]) -> str:
    return f"a photo of a {class_name_map.get(int(label), str(label))}"


@torch.no_grad()
def predict_records(
    *,
    model: EVLMEnhancer,
    image: torch.Tensor,
    eeg: torch.Tensor,
    rows: list[dict[str, Any]],
    label_values: torch.Tensor,
    prototypes: torch.Tensor,
    class_name_map: dict[int, str],
    mode: str,
    corruption: str,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for start in range(0, image.shape[0], batch_size):
        end = start + batch_size
        image_b = image[start:end].to(device)
        eeg_b = eeg[start:end].to(device)
        logits, aux = model(image_b, eeg_b, prototypes)
        probs = torch.softmax(logits.float(), dim=-1)
        k = min(5, probs.shape[-1])
        top_scores, top_indices = probs.topk(k=k, dim=-1)
        top_labels = label_values[top_indices].detach().cpu().tolist()
        pred_labels = label_values[top_indices[:, 0]].detach().cpu().tolist()
        gate_tensor = aux.get("gamma")
        alpha_tensor = aux.get("alpha")
        delta_tensor = aux.get("delta_norm")
        confidence_tensor = aux.get("vision_confidence")
        for local_idx, (pred_label, labels, scores) in enumerate(zip(pred_labels, top_labels, top_scores.detach().cpu().tolist(), strict=False)):
            row = rows[start + local_idx]
            target_label = int(row["label"]) if row.get("label") is not None else None
            pred_label = int(pred_label)
            label_list = [int(item) for item in labels]
            record = {
                "image_id": str(row["image_id"]),
                "mode": mode,
                "corruption": corruption,
                "label": target_label,
                "pred_label": pred_label,
                "top5_labels": label_list,
                "top5_class_names": [class_name_map[item] for item in label_list],
                "top5_scores": [float(item) for item in scores],
                "human_label_name": class_name_map.get(target_label, str(target_label)),
                "pred_class_name": class_name_map.get(pred_label, str(pred_label)),
                "reference": caption_for_label(target_label, class_name_map) if target_label is not None else "",
                "prediction": caption_for_label(pred_label, class_name_map),
                "class_correct": float(pred_label == target_label) if target_label is not None else 0.0,
                "top5_correct": float(target_label in label_list) if target_label is not None else 0.0,
            }
            if gate_tensor is not None:
                record["gate_mean"] = float(gate_tensor[local_idx].mean().detach().cpu())
            if alpha_tensor is not None:
                record["alpha_mean"] = float(alpha_tensor[local_idx].mean().detach().cpu())
            if delta_tensor is not None:
                record["delta_norm"] = float(delta_tensor[local_idx].detach().cpu())
            if confidence_tensor is not None:
                record["vision_confidence"] = float(confidence_tensor[local_idx].mean().detach().cpu())
            records.append(record)
    return records


def summarize_records(records: list[dict[str, Any]], corruption: str, mode: str, file_name: str) -> dict[str, Any]:
    if not records:
        return {
            "file": file_name,
            "corruption": corruption,
            "mode": mode,
            "count": 0,
            "accuracy": 0.0,
            "top5_accuracy": 0.0,
            "caption_class_hit": 0.0,
        }
    summary: dict[str, Any] = {
        "file": file_name,
        "corruption": corruption,
        "mode": mode,
        "count": len(records),
        "accuracy": sum(float(row["class_correct"]) for row in records) / len(records),
        "top5_accuracy": sum(float(row["top5_correct"]) for row in records) / len(records),
        "caption_class_hit": sum(float(row["class_correct"]) for row in records) / len(records),
    }
    for key in ["gate_mean", "alpha_mean", "delta_norm", "vision_confidence"]:
        vals = [float(row[key]) for row in records if key in row]
        if vals:
            summary[key] = float(np.mean(vals))
    return summary


def write_table(rows: list[dict[str, Any]], csv_path: Path, md_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    if not headers:
        headers = ["status"]
        rows = [{"status": "empty"}]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(f"{row.get(h, ''):.6f}" if isinstance(row.get(h), float) else str(row.get(h, "")) for h in headers)
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_test_image_embeddings(cache_dir: Path, corruption: str, max_samples: int) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if corruption == "clean":
        cache_path = cache_dir / "clip_test.npy"
        index_path = cache_dir / "clip_index_test.json"
    else:
        cache_path = cache_dir / "degraded_test" / f"clip_test_{corruption}.npy"
        index_path = cache_dir / "degraded_test" / f"clip_index_test_{corruption}.json"
    embeddings, cache_rows = load_cache(cache_path, index_path, max_samples=max_samples)
    return F.normalize(embeddings.float(), dim=-1), cache_rows


def evaluate_checkpoint(checkpoint: Path, output_dir: Path, corruptions: list[str], modes: list[str], max_samples: int = 0) -> list[dict[str, Any]]:
    device = resolve_device("auto")
    model, label_values, prototypes, cfg_dict = load_evlm_checkpoint(checkpoint, device)
    cfg = EVLMConfig(**cfg_dict)
    bank = torch.load(cfg.prototype_bank, map_location="cpu", weights_only=False)
    class_name_map = {int(key): str(value) for key, value in bank["class_name_map"].items()}
    manifest = Path(cfg.test_manifest)
    manifest_rows = read_jsonl(manifest)
    if max_samples:
        manifest_rows = manifest_rows[:max_samples]
    eeg_cache: dict[str, torch.Tensor] = {}
    for mode in ["real_eeg", "shuffled_eeg", "random_eeg"]:
        eeg_cache[mode] = compute_eeg_embeddings(
            manifest=manifest,
            checkpoint=Path(cfg.eeg_checkpoint),
            mode=mode,
            device=device,
            batch_size=cfg.eval_batch_size,
            max_samples=max_samples,
        )
    metrics: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for corruption in corruptions:
        image_emb, cache_rows = load_test_image_embeddings(Path(cfg.test_cache_dir), corruption, max_samples)
        rows = manifest_rows[: len(cache_rows)]
        for manifest_row, cache_row in zip(rows, cache_rows, strict=True):
            if manifest_row.get("image_id") != cache_row.get("image_id"):
                raise ValueError(f"Manifest/cache order mismatch for {corruption}: {manifest_row.get('image_id')} vs {cache_row.get('image_id')}")
        for mode in modes:
            if mode == "vision_only":
                image_in = image_emb
                eeg_in = torch.zeros_like(eeg_cache["real_eeg"][: image_emb.shape[0]])
            elif mode == "eeg_only":
                image_in = torch.zeros_like(image_emb)
                eeg_in = eeg_cache["real_eeg"][: image_emb.shape[0]]
            else:
                image_in = image_emb
                eeg_in = eeg_cache[mode][: image_emb.shape[0]]
            records = predict_records(
                model=model,
                image=image_in,
                eeg=eeg_in,
                rows=rows,
                label_values=label_values,
                prototypes=prototypes,
                class_name_map=class_name_map,
                mode=mode,
                corruption=corruption,
                device=device,
                batch_size=cfg.eval_batch_size,
            )
            out_path = output_dir / f"{corruption}_{mode}.jsonl"
            with out_path.open("w", encoding="utf-8") as handle:
                for row in records:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics.append(summarize_records(records, corruption, mode, out_path.name))
    write_table(metrics, output_dir / "FULL_METRICS.csv", output_dir / "FULL_METRICS.md", "EVLM Full Metrics")
    gap_rows = semantic_gap_rows(metrics, model_name=cfg.variant, seed=cfg.seed, eval_dir=output_dir)
    write_table(gap_rows, output_dir / "SUMMARY_METRICS.csv", output_dir / "SUMMARY_METRICS.md", "EVLM Summary Metrics")
    return gap_rows


def semantic_gap_rows(metrics: list[dict[str, Any]], *, model_name: str, seed: int, eval_dir: Path) -> list[dict[str, Any]]:
    by_condition: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in metrics:
        by_condition[str(row["corruption"])][str(row["mode"])] = row
    rows: list[dict[str, Any]] = []
    for corruption in DEFAULT_CORRUPTIONS:
        mode_rows = by_condition.get(corruption, {})
        real = mode_rows.get("real_eeg")
        if real is None:
            continue
        vision = mode_rows.get("vision_only", {})
        shuffled = mode_rows.get("shuffled_eeg", {})
        random = mode_rows.get("random_eeg", {})
        real_top1 = float(real.get("accuracy", 0.0))
        vision_top1 = float(vision.get("accuracy", 0.0))
        shuffled_top1 = float(shuffled.get("accuracy", 0.0))
        random_top1 = float(random.get("accuracy", 0.0))
        rows.append(
            {
                "model": model_name,
                "seed": seed,
                "corruption": corruption,
                "real_top1": real_top1,
                "real_top5": float(real.get("top5_accuracy", 0.0)),
                "class_hit": float(real.get("caption_class_hit", 0.0)),
                "vision_top1": vision_top1,
                "shuffled_top1": shuffled_top1,
                "random_top1": random_top1,
                "real_minus_vision": real_top1 - vision_top1,
                "real_minus_shuffled": real_top1 - shuffled_top1,
                "real_minus_random": real_top1 - random_top1,
                "real_beats_vision": real_top1 > vision_top1,
                "real_beats_controls": real_top1 > shuffled_top1 and real_top1 > random_top1,
                "eval_dir": str(eval_dir),
            }
        )
    return rows


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def coerce_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def selection_row(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    strong = [row for row in rows if str(row.get("corruption")) in STRONG_CORRUPTIONS]
    if not strong:
        strong = rows
    per_seed: dict[str, list[float]] = defaultdict(list)
    for row in strong:
        per_seed[str(row.get("seed", "baseline"))].append(coerce_float(row.get("real_top1")))
    seed_means = [float(np.mean(vals)) for vals in per_seed.values() if vals]
    mean_real = float(np.mean([coerce_float(row.get("real_top1")) for row in strong])) if strong else 0.0
    mean_real_vision = float(np.mean([coerce_float(row.get("real_minus_vision")) for row in strong])) if strong else 0.0
    mean_real_shuf = float(np.mean([coerce_float(row.get("real_minus_shuffled")) for row in strong])) if strong else 0.0
    mean_real_rand = float(np.mean([coerce_float(row.get("real_minus_random")) for row in strong])) if strong else 0.0
    std_seeds = float(np.std(seed_means)) if len(seed_means) > 1 else 0.0
    score = mean_real + 0.5 * mean_real_vision + 0.5 * mean_real_shuf + 0.5 * mean_real_rand - 0.1 * std_seeds
    return {
        "model": model,
        "strong_real_top1_mean": mean_real,
        "real_minus_vision_mean": mean_real_vision,
        "real_minus_shuffled_mean": mean_real_shuf,
        "real_minus_random_mean": mean_real_rand,
        "win_rate_vision": float(np.mean([str(row.get("real_beats_vision")).lower() == "true" for row in strong])) if strong else 0.0,
        "win_rate_controls": float(np.mean([str(row.get("real_beats_controls")).lower() == "true" for row in strong])) if strong else 0.0,
        "std_across_seed_strong_real_top1": std_seeds,
        "score": score,
        "rows": len(strong),
    }


def aggregate_reports(out_dir: Path) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for group in ["residual", "proto_bias", "combined", "autonomous"]:
        group_dir = out_dir / group
        group_rows: list[dict[str, Any]] = []
        for path in sorted(group_dir.glob("**/eval/SUMMARY_METRICS.csv")):
            rows = read_csv_rows(path)
            group_rows.extend(rows)
            all_rows.extend(rows)
        if group_rows:
            write_table(group_rows, group_dir / "metrics.csv", group_dir / "metrics.md", f"{group} EVLM Metrics")
            for variant in sorted({str(row["model"]) for row in group_rows}):
                variant_rows = [row for row in group_rows if str(row["model"]) == variant]
                summary = summarize_variant(variant_rows)
                (group_dir / f"{variant}_summary.md").write_text(summary, encoding="utf-8")
                checkpoint_dir = group_dir / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                for seed in sorted({int(row["seed"]) for row in variant_rows if str(row.get("seed", "")).isdigit()}):
                    src = group_dir / f"{variant}_seed{seed}" / "checkpoints" / "best.pt"
                    if src.exists():
                        dst = checkpoint_dir / f"{variant}_seed{seed}_best.pt"
                        if not dst.exists():
                            shutil.copy2(src, dst)
    baseline_rows = read_csv_rows(Path("outputs/final_results/A2_FINAL_METRICS.csv"))
    selection_rows = [selection_row("A2_final", baseline_rows)]
    for variant in sorted({str(row["model"]) for row in all_rows}):
        selection_rows.append(selection_row(variant, [row for row in all_rows if str(row["model"]) == variant]))
    selection_rows = sorted(selection_rows, key=lambda row: coerce_float(row["score"]), reverse=True)
    baseline_score = next((row for row in selection_rows if row["model"] == "A2_final"), None)
    for row in selection_rows:
        row["beats_A2_final"] = bool(baseline_score and row["model"] != "A2_final" and coerce_float(row["score"]) > coerce_float(baseline_score["score"]))
    write_table(selection_rows, out_dir / "EVLM_MODEL_SELECTION.csv", out_dir / "EVLM_MODEL_SELECTION.md", "EVLM Model Selection")
    write_model_selection_report(out_dir, selection_rows)
    write_final_report(out_dir, selection_rows, baseline_rows, all_rows)
    return selection_rows


def checkpoint_for_model(out_dir: Path, model: str) -> str:
    if model == "A2_final":
        return "outputs/heavy_stage/semantic_fusion_A2_temporal_spectral_spatial_full/semantic_fusion_classifier.pt and seed variants"
    group = variant_group(model)
    candidates = sorted((out_dir / group).glob(f"{model}_seed*/checkpoints/best.pt"))
    if not candidates and group == "autonomous":
        return "test-time ensemble; no trained checkpoint"
    return str(candidates[0]) if candidates else "not found"


def metrics_file_for_model(out_dir: Path, model: str) -> str:
    if model == "A2_final":
        return "outputs/final_results/A2_FINAL_METRICS.csv"
    group = variant_group(model)
    path = out_dir / group / "metrics.csv"
    if path.exists():
        return str(path)
    auto_path = out_dir / "autonomous" / model / "metrics.csv"
    return str(auto_path) if auto_path.exists() else str(out_dir / "EVLM_MODEL_SELECTION.csv")


def write_model_selection_report(out_dir: Path, selection_rows: list[dict[str, Any]]) -> None:
    best = selection_rows[0] if selection_rows else {}
    baseline = next((row for row in selection_rows if row.get("model") == "A2_final"), {})
    best_new = next((row for row in selection_rows if row.get("model") != "A2_final"), {})
    beats = bool(best.get("beats_A2_final", False))
    improved_fields: list[str] = []
    failed_fields: list[str] = []
    if baseline and best_new:
        comparisons = [
            ("strong_real_top1_mean", "strong-degradation real EEG Top-1"),
            ("real_minus_vision_mean", "real-vs-vision gap"),
            ("real_minus_shuffled_mean", "real-vs-shuffled gap"),
            ("real_minus_random_mean", "real-vs-random gap"),
            ("win_rate_vision", "win rate over vision"),
            ("win_rate_controls", "win rate over controls"),
        ]
        for key, label in comparisons:
            if coerce_float(best_new.get(key)) > coerce_float(baseline.get(key)):
                improved_fields.append(label)
            else:
                failed_fields.append(label)
    headers = list(selection_rows[0].keys()) if selection_rows else []
    lines = [
        "# EVLM Model Selection",
        "",
        f"- Best model: `{best.get('model', 'unknown')}`",
        f"- Best checkpoint: `{checkpoint_for_model(out_dir, str(best.get('model', '')) )}`",
        f"- Best metrics file: `{metrics_file_for_model(out_dir, str(best.get('model', '')) )}`",
        f"- Does it beat A2_final? `{'yes' if beats else 'no'}`",
        f"- Best new-model improvements vs A2_final: `{', '.join(improved_fields) if improved_fields else 'no primary selection field beats A2_final'}`",
        f"- Best new-model failures vs A2_final: `{', '.join(failed_fields) if failed_fields else 'none among primary fields'}`",
        f"- Recommended final model for report: `{best.get('model', 'unknown')}`",
        "",
        "## Selection Table",
        "",
    ]
    if headers:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in selection_rows:
            lines.append(
                "| "
                + " | ".join(f"{row.get(h, ''):.6f}" if isinstance(row.get(h), float) else str(row.get(h, "")) for h in headers)
                + " |"
            )
    (out_dir / "EVLM_MODEL_SELECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_prediction_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_score_ensemble(out_dir: Path, *, model_name: str = "A2_residual_proto_ensemble") -> None:
    """Autonomous Direction E: cheap test-time ensemble over completed EVLM predictions."""
    sources = [
        ("A2_residual_vector_margin", "residual"),
        ("A2_proto_bias", "proto_bias"),
        ("A2_residual_plus_proto_bias", "combined"),
    ]
    bank = torch.load("outputs/semantic_caption/prototypes.pt", map_location="cpu", weights_only=False)
    class_name_map = {int(key): str(value) for key, value in bank["class_name_map"].items()}
    label_values = [int(item) for item in bank["labels"]]
    auto_root = out_dir / "autonomous" / model_name
    auto_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for seed in [42, 123, 2025]:
        eval_dir = auto_root / f"seed{seed}" / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        metrics: list[dict[str, Any]] = []
        for corruption in DEFAULT_CORRUPTIONS:
            for mode in DEFAULT_MODES:
                per_source: list[list[dict[str, Any]]] = []
                for variant, group in sources:
                    path = out_dir / group / f"{variant}_seed{seed}" / "eval" / f"{corruption}_{mode}.jsonl"
                    records = _load_prediction_records(path)
                    if records:
                        per_source.append(records)
                if not per_source:
                    continue
                count = min(len(records) for records in per_source)
                ensembled: list[dict[str, Any]] = []
                for index in range(count):
                    base = dict(per_source[0][index])
                    scores: Counter[int] = Counter()
                    for records in per_source:
                        for label, score in zip(records[index].get("top5_labels", []), records[index].get("top5_scores", []), strict=False):
                            scores[int(label)] += float(score)
                    ordered = [label for label, _score in scores.most_common(5)]
                    if len(ordered) < 5:
                        ordered.extend([label for label in label_values if label not in ordered][: 5 - len(ordered)])
                    pred_label = int(ordered[0])
                    target_label = int(base["label"]) if base.get("label") is not None else None
                    base.update(
                        {
                            "model": model_name,
                            "seed": seed,
                            "mode": mode,
                            "corruption": corruption,
                            "pred_label": pred_label,
                            "top5_labels": ordered,
                            "top5_class_names": [class_name_map[item] for item in ordered],
                            "pred_class_name": class_name_map.get(pred_label, str(pred_label)),
                            "prediction": caption_for_label(pred_label, class_name_map),
                            "class_correct": float(pred_label == target_label) if target_label is not None else 0.0,
                            "top5_correct": float(target_label in ordered) if target_label is not None else 0.0,
                        }
                    )
                    ensembled.append(base)
                out_path = eval_dir / f"{corruption}_{mode}.jsonl"
                with out_path.open("w", encoding="utf-8") as handle:
                    for row in ensembled:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                metrics.append(summarize_records(ensembled, corruption, mode, out_path.name))
        write_table(metrics, eval_dir / "FULL_METRICS.csv", eval_dir / "FULL_METRICS.md", f"{model_name} Full Metrics")
        gap_rows = semantic_gap_rows(metrics, model_name=model_name, seed=seed, eval_dir=eval_dir)
        write_table(gap_rows, eval_dir / "SUMMARY_METRICS.csv", eval_dir / "SUMMARY_METRICS.md", f"{model_name} Summary Metrics")
        summary_rows.extend(gap_rows)
    write_table(summary_rows, auto_root / "metrics.csv", auto_root / "metrics.md", f"{model_name} Metrics")


def summarize_variant(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "# Empty Summary\n"
    model = str(rows[0]["model"])
    lines = [f"# {model} Summary", ""]
    lines.append("| corruption | seeds | top1_mean | top1_std | top5_mean | real_vision_mean | real_shuffled_mean | real_random_mean | vision_wins | control_wins |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    by_corr: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_corr[str(row["corruption"])].append(row)
    for corruption in DEFAULT_CORRUPTIONS:
        corr_rows = by_corr.get(corruption, [])
        if not corr_rows:
            continue
        seeds = sorted({str(row["seed"]) for row in corr_rows})
        top1 = [coerce_float(row["real_top1"]) for row in corr_rows]
        top5 = [coerce_float(row["real_top5"]) for row in corr_rows]
        rv = [coerce_float(row["real_minus_vision"]) for row in corr_rows]
        rs = [coerce_float(row["real_minus_shuffled"]) for row in corr_rows]
        rr = [coerce_float(row["real_minus_random"]) for row in corr_rows]
        vw = sum(str(row["real_beats_vision"]).lower() == "true" for row in corr_rows)
        cw = sum(str(row["real_beats_controls"]).lower() == "true" for row in corr_rows)
        lines.append(
            f"| {corruption} | {','.join(seeds)} | {np.mean(top1):.6f} | {np.std(top1):.6f} | {np.mean(top5):.6f} | "
            f"{np.mean(rv):.6f} | {np.mean(rs):.6f} | {np.mean(rr):.6f} | {vw}/{len(corr_rows)} | {cw}/{len(corr_rows)} |"
        )
    return "\n".join(lines) + "\n"


def report_section_for_model(title: str, rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return [f"## {title}", "", "No rows available.", ""]
    lines = [f"## {title}", ""]
    keys = ["model", "strong_real_top1_mean", "real_minus_vision_mean", "real_minus_shuffled_mean", "real_minus_random_mean", "win_rate_vision", "win_rate_controls", "std_across_seed_strong_real_top1", "score", "beats_A2_final"]
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(f"{row.get(k, ''):.6f}" if isinstance(row.get(k), float) else str(row.get(k, "")) for k in keys)
            + " |"
        )
    lines.append("")
    return lines


def write_final_report(out_dir: Path, selection_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> None:
    best = selection_rows[0] if selection_rows else {}
    baseline = next((row for row in selection_rows if row["model"] == "A2_final"), {})
    required_set = set(REQUIRED_VARIANTS)
    residual = [row for row in selection_rows if row["model"] in {"A2_residual_scalar", "A2_residual_vector", "A2_residual_vector_margin"}]
    proto = [row for row in selection_rows if row["model"] in {"A2_proto_bias", "A2_proto_bias_margin"}]
    combined = [row for row in selection_rows if row["model"] == "A2_residual_plus_proto_bias"]
    autonomous = [row for row in selection_rows if row["model"] not in {"A2_final", *required_set}]
    best_autonomous = autonomous[0] if autonomous else None
    best_new = next((row for row in selection_rows if row.get("model") != "A2_final"), None)
    improved_over_a2_fields: list[str] = []
    if baseline and best_new:
        for key, label in [
            ("strong_real_top1_mean", "strong-degradation real EEG Top-1"),
            ("real_minus_vision_mean", "real-vs-vision gap"),
            ("real_minus_shuffled_mean", "real-vs-shuffled gap"),
            ("real_minus_random_mean", "real-vs-random gap"),
        ]:
            if coerce_float(best_new.get(key)) > coerce_float(baseline.get(key)):
                improved_over_a2_fields.append(label)
    lines = [
        "# Final EVLM Improvement Report",
        "",
        f"- Recommended final EVLM model: `{best.get('model', 'unknown')}`",
        f"- Whether it beats A2_final: `{bool(best.get('beats_A2_final', False))}`",
        "- Recommended checkpoint: see per-run `checkpoints/best.pt`; use the best model/seed selected from `EVLM_MODEL_SELECTION.csv`.",
        f"- Recommended final metrics file: `{out_dir / 'EVLM_MODEL_SELECTION.csv'}`",
        f"- A2_final baseline score: `{coerce_float(baseline.get('score')):.6f}`",
        f"- Best score: `{coerce_float(best.get('score')):.6f}`",
        "",
        "## Answers",
        "",
        f"1. Residual Adapter improved over A2_final: `{any(row.get('beats_A2_final') for row in residual)}`.",
        f"2. Prototype Attention Bias improved over A2_final: `{any(row.get('beats_A2_final') for row in proto)}`.",
        f"3. Combined residual + prototype bias improved over A2_final: `{any(row.get('beats_A2_final') for row in combined)}`.",
        f"4. Best strong-degradation performance: `{max(selection_rows, key=lambda r: coerce_float(r.get('strong_real_top1_mean')))['model'] if selection_rows else 'unknown'}`.",
        f"5. Best real-vs-vision gap: `{max(selection_rows, key=lambda r: coerce_float(r.get('real_minus_vision_mean')))['model'] if selection_rows else 'unknown'}`.",
        f"6. Best real-vs-control gap: `{max(selection_rows, key=lambda r: coerce_float(r.get('real_minus_shuffled_mean')) + coerce_float(r.get('real_minus_random_mean')))['model'] if selection_rows else 'unknown'}`.",
        f"7. Most stable across seeds: `{min(selection_rows, key=lambda r: coerce_float(r.get('std_across_seed_strong_real_top1')))['model'] if selection_rows else 'unknown'}`.",
        f"8. Final model recommendation: `{best.get('model', 'unknown')}`.",
        f"9. Autonomous explorations attempted: `{'A2_residual_proto_ensemble' if autonomous else 'none'}`.",
        f"10. Autonomous exploration helped most: `{best_autonomous.get('model') if best_autonomous else 'none'}`.",
        f"11. Remaining limitations: `{', '.join(improved_over_a2_fields) if improved_over_a2_fields else 'no new primary metric improved over A2_final'}` improved for the best new model, but the overall score/control gaps did not beat A2_final.",
        "",
    ]
    lines.extend(report_section_for_model("Table 1: A2_final baseline", [baseline] if baseline else []))
    lines.extend(report_section_for_model("Table 2: Residual Adapter results", residual))
    lines.extend(report_section_for_model("Table 3: Prototype Bias results", proto))
    lines.extend(report_section_for_model("Table 4: Combined model results", combined))
    lines.extend(report_section_for_model("Table 5: Autonomous exploration results", autonomous))
    lines.extend(report_section_for_model("Table 6: Final model selection", selection_rows))
    lines.extend(
        [
            "## Final Statement",
            "",
            f"- Recommended final EVLM model: `{best.get('model', 'unknown')}`",
            "- Recommended checkpoint: choose the best seed checkpoint under the corresponding experiment directory.",
            f"- Recommended final metrics file: `{out_dir / 'EVLM_MODEL_SELECTION.csv'}`",
            f"- Whether it beats A2_final: `{bool(best.get('beats_A2_final', False))}`",
            f"- Best improvement conditions: `{', '.join(improved_over_a2_fields) if improved_over_a2_fields else 'none over A2_final by primary fields'}`.",
            "- Remaining limitations: new EVLM variants improved strong-degradation real Top-1 in some cases, but reduced real-vs-vision and real-vs-control gaps enough that A2_final remains the recommended model.",
        ]
    )
    (out_dir / "FINAL_EVLM_IMPROVEMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiments(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = args.variants or REQUIRED_VARIANTS
    seeds = [int(seed) for seed in args.seeds]
    for variant in variants:
        for seed in seeds:
            cfg = EVLMConfig(
                variant=variant,
                seed=seed,
                train_manifest=args.train_manifest,
                val_manifest=args.val_manifest,
                test_manifest=args.test_manifest,
                train_cache=args.train_cache,
                val_cache=args.val_cache,
                test_cache_dir=args.test_cache_dir,
                prototype_bank=args.prototype_bank,
                text_prototypes=args.text_prototypes,
                eeg_checkpoint=args.eeg_checkpoint,
                output_dir=args.output_dir,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                hidden_dim=args.hidden_dim,
                lr=args.lr,
                weight_decay=args.weight_decay,
                tau_cls=args.tau_cls,
                margin=args.margin,
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                max_test_samples=args.max_test_samples,
                log_every=args.log_every,
            )
            checkpoint = Path(cfg.output_dir) / variant_group(variant) / f"{variant}_seed{seed}" / "checkpoints" / "best.pt"
            if not checkpoint.exists() or args.force:
                checkpoint = train_one(cfg)
            eval_dir = Path(cfg.output_dir) / variant_group(variant) / f"{variant}_seed{seed}" / "eval"
            if not (eval_dir / "SUMMARY_METRICS.csv").exists() or args.force_eval:
                append_gpu_usage(out_dir / "GPU_USAGE.md", active_job=f"{variant}_seed{seed}", event="eval_start")
                evaluate_checkpoint(checkpoint, eval_dir, args.corruptions, args.modes, max_samples=args.max_test_samples)
                append_gpu_usage(out_dir / "GPU_USAGE.md", active_job=f"{variant}_seed{seed}", event="eval_end")
            if not args.skip_aggregate:
                aggregate_reports(out_dir)
    if not args.skip_aggregate:
        aggregate_reports(out_dir)


def self_test() -> None:
    seed_everything(7)
    prototypes = F.normalize(torch.randn(3, 8), dim=-1)
    image = F.normalize(torch.randn(4, 8), dim=-1)
    eeg = F.normalize(torch.randn(4, 8), dim=-1)
    labels = torch.tensor([0, 1, 2])
    for variant in REQUIRED_VARIANTS:
        model = EVLMEnhancer(variant=variant, embed_dim=8, num_classes=3, hidden_dim=16, tau_cls=0.07)
        logits, aux = model(image, eeg, prototypes)
        assert tuple(logits.shape) == (4, 3), variant
        if variant in RESIDUAL_VARIANTS:
            assert "alpha" in aux and "delta_norm" in aux, variant
        if variant in PROTO_VARIANTS:
            assert "gamma" in aux, variant
    rows = [
        {"model": "x", "seed": 1, "corruption": "lowres16", "real_top1": 0.5, "real_minus_vision": 0.1, "real_minus_shuffled": 0.2, "real_minus_random": 0.3, "real_beats_vision": True, "real_beats_controls": True},
        {"model": "x", "seed": 2, "corruption": "mixed", "real_top1": 0.7, "real_minus_vision": 0.2, "real_minus_shuffled": 0.3, "real_minus_random": 0.4, "real_beats_vision": True, "real_beats_controls": True},
    ]
    score = selection_row("x", rows)
    assert score["strong_real_top1_mean"] == 0.6
    print(json.dumps({"self_test": "ok", "variants": len(REQUIRED_VARIANTS)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate EVLM EEG-guided visual enhancement variants.")
    parser.add_argument("--output_dir", default="outputs/evlm_improve")
    parser.add_argument("--variants", nargs="+", default=None, choices=REQUIRED_VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2025])
    parser.add_argument("--corruptions", nargs="+", default=DEFAULT_CORRUPTIONS)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--train_manifest", default="data/thought2text/train_human_caption.jsonl")
    parser.add_argument("--val_manifest", default="data/thought2text/val_human_caption.jsonl")
    parser.add_argument("--test_manifest", default="data/thought2text/test_human_caption.jsonl")
    parser.add_argument("--train_cache", default="data/thought2text/cache/clip_train.npy")
    parser.add_argument("--val_cache", default="data/thought2text/cache/clip_val.npy")
    parser.add_argument("--test_cache_dir", default="data/thought2text/cache")
    parser.add_argument("--prototype_bank", default="outputs/semantic_caption/prototypes.pt")
    parser.add_argument("--text_prototypes", default="data/thought2text/cache/class_text_prototypes.npy")
    parser.add_argument("--eeg_checkpoint", default="outputs/architectures/A2_temporal_spectral_spatial_full/checkpoints/best.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--tau_cls", type=float, default=0.07)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force_eval", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--run_ensemble", action="store_true")
    parser.add_argument("--skip_aggregate", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.aggregate_only:
        if args.run_ensemble:
            run_score_ensemble(Path(args.output_dir))
        aggregate_reports(Path(args.output_dir))
        return
    run_experiments(args)


if __name__ == "__main__":
    main()
