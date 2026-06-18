from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data.collate import caption_collate
from src.data.corruptions import apply_corruption
from src.data.dataset import EEGVisionCaptionDataset
from src.eval.constrained_caption_eval import load_cache, load_eeg_encoder, read_jsonl
from src.models.caption_model import SoftPromptCaptionModel
from src.train.train_fusion import apply_eeg_mode
from src.utils.seed import seed_everything


CORRUPTIONS = ["clean", "lowres16", "mixed", "occlusion50", "strong_blur", "strong_noise"]
MODES = ["vision_only", "real_eeg", "shuffled_eeg", "random_eeg", "eeg_only"]
STRONG_CORRUPTIONS = ["lowres16", "mixed", "occlusion50", "strong_blur", "strong_noise"]
VTF_VARIANTS = ["VTF1_basic_M4", "VTF2_confidence_beta_M4", "VTF3_confidence_beta_margin_M4", "VTF4_confidence_beta_margin_M8"]
GEN_VARIANTS = ["G0_image_only_prefix", "G1_vtf_visual_prefix", "G2_vtf_visual_eeg_prefix", "G3_vtf_visual_eeg_topk_prefix"]
CLIP_IMAGE_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_IMAGE_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


@dataclass
class TokenGenConfig:
    output_dir: str = "outputs/token_generative_evlm"
    train_manifest: str = "data/thought2text/train_blip_caption.jsonl"
    val_manifest: str = "data/thought2text/val_blip_caption.jsonl"
    test_manifest: str = "data/thought2text/test_blip_caption.jsonl"
    human_train_manifest: str = "data/thought2text/train_human_caption.jsonl"
    human_val_manifest: str = "data/thought2text/val_human_caption.jsonl"
    human_test_manifest: str = "data/thought2text/test_human_caption.jsonl"
    clip_cache_dir: str = "data/thought2text/cache"
    prototype_bank: str = "outputs/semantic_caption/prototypes.pt"
    text_prototypes: str = "data/thought2text/cache/class_text_prototypes.npy"
    eeg_checkpoint: str = "outputs/architectures/A2_temporal_spectral_spatial_full/checkpoints/best.pt"
    clip_model: str = "openai/clip-vit-base-patch32"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    vtf_variant: str = "VTF3_confidence_beta_margin_M4"
    gen_variant: str = "G3_vtf_visual_eeg_topk_prefix"
    seed: int = 42
    vtf_epochs: int = 30
    gen_epochs: int = 8
    patience: int = 5
    batch_size: int = 256
    gen_batch_size: int = 16
    eval_batch_size: int = 128
    hidden_dim: int = 512
    prefix_len: int = 16
    lr: float = 1.0e-4
    gen_lr: float = 1.0e-4
    weight_decay: float = 0.05
    tau_cls: float = 0.07
    margin: float = 0.1
    margin_weight: float = 0.2
    beta_reg_weight: float = 0.01
    max_train_samples: int = 0
    max_val_samples: int = 0
    max_test_samples: int = 0
    max_caption_length: int = 32
    max_new_tokens: int = 24
    use_tiny_lm: bool = False
    qwen_prefix: bool = False
    force: bool = False


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def append_gpu_usage(path: Path, active_job: str, event: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:  # pragma: no cover
        output = f"nvidia-smi unavailable: {exc}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {time.strftime('%Y-%m-%d %H:%M:%S')} - {event}\n\n")
        handle.write(f"- active_job: `{active_job}`\n")
        handle.write(f"- gpu: `{output}`\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_label_bank(cfg: TokenGenConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[int, str]]:
    bank = torch.load(cfg.prototype_bank, map_location="cpu", weights_only=False)
    labels = torch.tensor([int(item) for item in bank["labels"]], dtype=torch.long, device=device)
    class_name_map = {int(k): str(v) for k, v in bank["class_name_map"].items()}
    prototypes = torch.from_numpy(np.load(cfg.text_prototypes)).float()
    if prototypes.shape[0] != labels.numel():
        prototypes = bank["image_prototypes"].float()
    return labels, F.normalize(prototypes.to(device), dim=-1), class_name_map


def labels_to_targets(labels: torch.Tensor, label_values: torch.Tensor) -> torch.Tensor:
    lut = {int(label): idx for idx, label in enumerate(label_values.detach().cpu().tolist())}
    return torch.tensor([lut[int(label)] for label in labels.detach().cpu().tolist()], dtype=torch.long, device=labels.device)


def load_rows_labels(path: Path, max_samples: int = 0) -> tuple[list[dict[str, Any]], torch.Tensor, list[str]]:
    rows = read_jsonl(path)
    if max_samples:
        rows = rows[:max_samples]
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    captions = [str(row.get("caption", "")) for row in rows]
    return rows, labels, captions


def cache_key_for(path: Path, mode: str, max_samples: int) -> str:
    return f"{path.stem}_{mode}_{max_samples or 'all'}"


def compute_eeg_embeddings(cfg: TokenGenConfig, manifest: Path, mode: str, device: torch.device, max_samples: int) -> torch.Tensor:
    cache_dir = Path(cfg.output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"eeg_{Path(cfg.eeg_checkpoint).parent.parent.name}_{cache_key_for(manifest, mode, max_samples)}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    dataset = EEGVisionCaptionDataset(manifest, allow_missing_images=True)
    loader = DataLoader(dataset, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=0, collate_fn=caption_collate)
    encoder = load_eeg_encoder(Path(cfg.eeg_checkpoint), device, manifest)
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
    result = torch.cat(outputs, dim=0)
    if max_samples:
        result = result[:max_samples]
    torch.save(result.cpu(), cache_path)
    return result


def load_clip_pooled(cfg: TokenGenConfig, split: str, max_samples: int = 0) -> torch.Tensor:
    cache_dir = Path(cfg.clip_cache_dir)
    if split == "test":
        cache_path = cache_dir / "clip_test.npy"
        index_path = cache_dir / "clip_index_test.json"
    else:
        cache_path = cache_dir / f"clip_{split}.npy"
        index_path = cache_dir / f"clip_index_{split}.json"
    emb, _ = load_cache(cache_path, index_path, max_samples=max_samples)
    return F.normalize(emb.float(), dim=-1)


def load_test_clip_pooled(cfg: TokenGenConfig, corruption: str, max_samples: int = 0) -> torch.Tensor:
    cache_dir = Path(cfg.clip_cache_dir)
    if corruption == "clean":
        cache_path = cache_dir / "clip_test.npy"
        index_path = cache_dir / "clip_index_test.json"
    else:
        cache_path = cache_dir / "degraded_test" / f"clip_test_{corruption}.npy"
        index_path = cache_dir / "degraded_test" / f"clip_index_test_{corruption}.json"
    emb, _ = load_cache(cache_path, index_path, max_samples=max_samples)
    return F.normalize(emb.float(), dim=-1)


def normalize_clip_images(images: torch.Tensor) -> torch.Tensor:
    mean = CLIP_IMAGE_MEAN.to(device=images.device, dtype=images.dtype)
    std = CLIP_IMAGE_STD.to(device=images.device, dtype=images.dtype)
    return (images.float() - mean) / std


def clip_token_cache_path(cfg: TokenGenConfig, manifest: Path, corruption: str, max_samples: int) -> Path:
    model_key = cfg.clip_model.replace("/", "_").replace("-", "_")
    cache_dir = Path(cfg.output_dir) / "cache" / "clip_tokens"
    return cache_dir / f"{model_key}_{manifest.stem}_{corruption}_{max_samples or 'all'}_tokens.pt"


@torch.no_grad()
def compute_clip_token_embeddings(cfg: TokenGenConfig, manifest: Path, corruption: str, device: torch.device, max_samples: int = 0) -> torch.Tensor:
    """Return true CLIP ViT visual tokens projected to CLIP semantic dim.

    The goal requires token-level fusion over ViT patch tokens. This function uses
    CLIPVisionModelWithProjection.vision_model(...).last_hidden_state, then applies
    the model's visual_projection to every CLS/patch token, yielding [B, 50, 512]
    for ViT-B/32.
    """

    cache_path = clip_token_cache_path(cfg, manifest, corruption, max_samples)
    meta_path = cache_path.with_suffix(".meta.json")
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False).float()

    try:
        from transformers import CLIPVisionModelWithProjection
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers CLIPVisionModelWithProjection is required for token-level EVLM") from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = EEGVisionCaptionDataset(manifest, allow_missing_images=False)
    loader = DataLoader(dataset, batch_size=min(cfg.eval_batch_size, 64), shuffle=False, num_workers=2, collate_fn=caption_collate)

    try:
        clip_model = CLIPVisionModelWithProjection.from_pretrained(cfg.clip_model, local_files_only=True)
    except Exception:
        clip_model = CLIPVisionModelWithProjection.from_pretrained(cfg.clip_model)
    clip_model.to(device)
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)

    tokens: list[torch.Tensor] = []
    seen = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        images = apply_corruption(images, corruption)
        pixel_values = normalize_clip_images(images)
        vision_out = clip_model.vision_model(pixel_values=pixel_values)
        hidden = vision_out.last_hidden_state
        projected = clip_model.visual_projection(hidden)
        tokens.append(F.normalize(projected.detach().cpu().float(), dim=-1).to(torch.float16))
        seen += int(images.shape[0])
        if max_samples and seen >= max_samples:
            break

    result = torch.cat(tokens, dim=0)
    if max_samples:
        result = result[:max_samples]
    torch.save(result, cache_path)
    write_json(
        meta_path,
        {
            "manifest": str(manifest),
            "clip_model": cfg.clip_model,
            "corruption": corruption,
            "shape": list(result.shape),
            "dtype": str(result.dtype),
            "source": "CLIPVisionModelWithProjection.vision_model.last_hidden_state + visual_projection",
        },
    )
    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result.float()


def token_count_for_variant(variant: str) -> int:
    return 8 if variant.endswith("_M8") else 4


def uses_confidence(variant: str) -> bool:
    return variant != "VTF1_basic_M4"


def uses_margin(variant: str) -> bool:
    return "margin" in variant


class TokenFusionModel(nn.Module):
    def __init__(self, variant: str, *, embed_dim: int = 512, hidden_dim: int = 512, tau_cls: float = 0.07) -> None:
        super().__init__()
        self.variant = variant
        self.embed_dim = embed_dim
        self.tau_cls = tau_cls
        self.num_eeg_tokens = token_count_for_variant(variant)
        self.visual_tokenizer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 50 * embed_dim),
        )
        self.eeg_tokenizer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.num_eeg_tokens * embed_dim),
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True, dropout=0.1)
        if uses_confidence(variant):
            self.beta_mlp = nn.Sequential(
                nn.LayerNorm(embed_dim * 2 + 1),
                nn.Linear(embed_dim * 2 + 1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.beta_param = nn.Parameter(torch.tensor(0.2))
        self.pool_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def make_tokens(self, image_or_tokens: torch.Tensor, eeg_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image_or_tokens.ndim == 3:
            if image_or_tokens.shape[-1] != self.embed_dim:
                raise ValueError(f"visual tokens must end in {self.embed_dim}, got {tuple(image_or_tokens.shape)}")
            visual_tokens = F.normalize(image_or_tokens.float(), dim=-1)
            pooled_visual = visual_tokens[:, 0]
        elif image_or_tokens.ndim == 2:
            pooled_visual = F.normalize(image_or_tokens.float(), dim=-1)
            visual_tokens = self.visual_tokenizer(pooled_visual).view(image_or_tokens.shape[0], 50, self.embed_dim)
            visual_tokens = F.normalize(visual_tokens, dim=-1)
        else:
            raise ValueError(f"image_or_tokens must have shape [B,512] or [B,N,512], got {tuple(image_or_tokens.shape)}")
        eeg_tokens = self.eeg_tokenizer(F.normalize(eeg_emb.float(), dim=-1)).view(eeg_emb.shape[0], self.num_eeg_tokens, self.embed_dim)
        eeg_tokens = F.normalize(eeg_tokens, dim=-1)
        return visual_tokens, eeg_tokens, pooled_visual

    def forward(self, image_emb: torch.Tensor, eeg_emb: torch.Tensor, prototypes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        eeg_norm = F.normalize(eeg_emb.float(), dim=-1)
        visual_tokens, eeg_tokens, image_norm = self.make_tokens(image_emb, eeg_norm)
        vision_logits = image_norm @ prototypes.T / self.tau_cls
        vision_conf = torch.softmax(vision_logits, dim=-1).max(dim=-1, keepdim=True).values
        eeg_context, _ = self.cross_attn(visual_tokens, eeg_tokens, eeg_tokens)
        if uses_confidence(self.variant):
            beta_raw = torch.sigmoid(self.beta_mlp(torch.cat([image_norm, eeg_norm, vision_conf], dim=-1)))
            beta = beta_raw * (1.0 - vision_conf)
        else:
            beta = torch.sigmoid(self.beta_param).view(1, 1).expand(visual_tokens.shape[0], 1)
        enhanced_tokens = F.normalize(visual_tokens + beta.view(-1, 1, 1) * eeg_context, dim=-1)
        cls_token = enhanced_tokens[:, 0]
        mean_patch = enhanced_tokens[:, 1:].mean(dim=1)
        enhanced_img = F.normalize(self.pool_mlp(torch.cat([cls_token, mean_patch], dim=-1)), dim=-1)
        logits = enhanced_img @ prototypes.T / self.tau_cls
        return logits, enhanced_img, {"beta": beta, "vision_confidence": vision_conf, "eeg_tokens": eeg_tokens, "enhanced_tokens": enhanced_tokens}


def vtf_loss(model: TokenFusionModel, image: torch.Tensor, eeg: torch.Tensor, targets: torch.Tensor, prototypes: torch.Tensor, cfg: TokenGenConfig) -> tuple[torch.Tensor, dict[str, float]]:
    logits, _emb, aux = model(image, eeg, prototypes)
    ce = F.cross_entropy(logits, targets)
    loss = ce + cfg.beta_reg_weight * aux["beta"].mean()
    margin_loss = torch.zeros((), device=image.device)
    if uses_margin(model.variant):
        perm = torch.randperm(eeg.shape[0], device=eeg.device)
        logits_shuf, _emb2, _ = model(image, eeg[perm], prototypes)
        logits_rand, _emb3, _ = model(image, F.normalize(torch.randn_like(eeg), dim=-1), prototypes)
        row = torch.arange(targets.shape[0], device=targets.device)
        score_real = logits[row, targets]
        margin_loss = F.relu(cfg.margin - (score_real - logits_shuf[row, targets])).mean() + F.relu(cfg.margin - (score_real - logits_rand[row, targets])).mean()
        loss = loss + cfg.margin_weight * margin_loss
    return loss, {"loss": float(loss.detach().cpu()), "ce": float(ce.detach().cpu()), "margin": float(margin_loss.detach().cpu()), "beta": float(aux["beta"].mean().detach().cpu())}


def train_vtf(cfg: TokenGenConfig) -> Path:
    seed_everything(cfg.seed)
    device = resolve_device()
    out_dir = Path(cfg.output_dir) / "token_fusion" / f"{cfg.vtf_variant}_seed{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", asdict(cfg))
    append_gpu_usage(Path(cfg.output_dir) / "GPU_USAGE.md", f"{cfg.vtf_variant}_seed{cfg.seed}", "vtf_train_start")
    label_values, prototypes, _ = load_label_bank(cfg, device)
    train_tokens = compute_clip_token_embeddings(cfg, Path(cfg.human_train_manifest), "clean", device, cfg.max_train_samples)
    val_tokens = compute_clip_token_embeddings(cfg, Path(cfg.human_val_manifest), "clean", device, cfg.max_val_samples)
    _train_rows, train_labels, _ = load_rows_labels(Path(cfg.human_train_manifest), cfg.max_train_samples)
    _val_rows, val_labels, _ = load_rows_labels(Path(cfg.human_val_manifest), cfg.max_val_samples)
    train_eeg = compute_eeg_embeddings(cfg, Path(cfg.human_train_manifest), "real_eeg", device, cfg.max_train_samples)
    val_eeg = compute_eeg_embeddings(cfg, Path(cfg.human_val_manifest), "real_eeg", device, cfg.max_val_samples)
    train_targets = labels_to_targets(train_labels.to(device), label_values).cpu()
    model = TokenFusionModel(cfg.vtf_variant, hidden_dim=cfg.hidden_dim, tau_cls=cfg.tau_cls).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(TensorDataset(train_tokens, train_eeg, train_targets), batch_size=cfg.batch_size, shuffle=True)
    best_val = -1.0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, cfg.vtf_epochs + 1):
        losses: list[float] = []
        model.train()
        for image_b, eeg_b, target_b in loader:
            image_b = image_b.to(device)
            eeg_b = eeg_b.to(device)
            target_b = target_b.to(device)
            loss, stats = vtf_loss(model, image_b, eeg_b, target_b, prototypes, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(stats["loss"])
        val_stats = eval_vtf_accuracy(model, val_tokens, val_eeg, val_labels, label_values, prototypes, device, cfg.eval_batch_size)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **val_stats}
        history.append(row)
        if val_stats["accuracy"] > best_val:
            best_val = val_stats["accuracy"]
            stale = 0
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "label_values": label_values.cpu(), "text_prototypes": prototypes.cpu(), "best_val_accuracy": best_val, "uses_true_clip_tokens": True}, ckpt_dir / "best.pt")
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    write_json(out_dir / "history.json", history)
    (out_dir / "summary.md").write_text(render_train_summary(cfg.vtf_variant, history, best_val), encoding="utf-8")
    append_gpu_usage(Path(cfg.output_dir) / "GPU_USAGE.md", f"{cfg.vtf_variant}_seed{cfg.seed}", "vtf_train_end")
    return out_dir / "checkpoints" / "best.pt"


@torch.no_grad()
def eval_vtf_accuracy(model: TokenFusionModel, image: torch.Tensor, eeg: torch.Tensor, labels: torch.Tensor, label_values: torch.Tensor, prototypes: torch.Tensor, device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    hits: list[float] = []
    top5_hits: list[float] = []
    for start in range(0, image.shape[0], batch_size):
        image_b = image[start : start + batch_size].to(device)
        eeg_b = eeg[start : start + batch_size].to(device)
        labels_b = labels[start : start + batch_size].to(device)
        logits, _emb, _aux = model(image_b, eeg_b, prototypes)
        pred = label_values[logits.argmax(dim=-1)].to(labels_b.device)
        top = label_values[logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices].to(labels_b.device)
        hits.extend((pred == labels_b).float().cpu().tolist())
        top5_hits.extend((top == labels_b.unsqueeze(-1)).any(dim=-1).float().cpu().tolist())
    return {"accuracy": float(np.mean(hits)), "top5_accuracy": float(np.mean(top5_hits))}


def render_train_summary(name: str, history: list[dict[str, Any]], best: float) -> str:
    lines = [f"# {name} Training Summary", "", f"- Best val accuracy: `{best:.6f}`", "", "| Epoch | Train Loss | Val Acc | Val Top5 |", "| ---: | ---: | ---: | ---: |"]
    for row in history:
        lines.append(f"| {row['epoch']} | {row['train_loss']:.6f} | {row['accuracy']:.6f} | {row['top5_accuracy']:.6f} |")
    return "\n".join(lines) + "\n"


def load_vtf_checkpoint(path: Path, device: torch.device) -> tuple[TokenFusionModel, torch.Tensor, torch.Tensor, TokenGenConfig]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = TokenGenConfig(**payload["config"])
    label_values = payload["label_values"].to(device=device, dtype=torch.long)
    prototypes = F.normalize(payload["text_prototypes"].to(device=device, dtype=torch.float32), dim=-1)
    model = TokenFusionModel(cfg.vtf_variant, hidden_dim=cfg.hidden_dim, tau_cls=cfg.tau_cls).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, label_values, prototypes, cfg


@torch.no_grad()
def vtf_predict_records(model: TokenFusionModel, image: torch.Tensor, eeg: torch.Tensor, rows: list[dict[str, Any]], label_values: torch.Tensor, prototypes: torch.Tensor, class_name_map: dict[int, str], device: torch.device, batch_size: int, corruption: str, mode: str) -> tuple[list[dict[str, Any]], torch.Tensor]:
    records: list[dict[str, Any]] = []
    embeddings: list[torch.Tensor] = []
    for start in range(0, image.shape[0], batch_size):
        image_b = image[start : start + batch_size].to(device)
        eeg_b = eeg[start : start + batch_size].to(device)
        logits, emb, aux = model(image_b, eeg_b, prototypes)
        embeddings.append(emb.cpu())
        probs = torch.softmax(logits.float(), dim=-1)
        top_scores, top_idx = probs.topk(k=min(5, probs.shape[-1]), dim=-1)
        top_labels = label_values[top_idx].cpu().tolist()
        pred_labels = label_values[top_idx[:, 0]].cpu().tolist()
        beta = aux["beta"].detach().cpu().view(-1).tolist()
        for i, (pred, labels, scores) in enumerate(zip(pred_labels, top_labels, top_scores.cpu().tolist(), strict=False)):
            row = rows[start + i]
            target = int(row["label"])
            labels = [int(x) for x in labels]
            records.append({
                "image_id": str(row["image_id"]),
                "label": target,
                "human_label_name": class_name_map.get(target, str(target)),
                "pred_label": int(pred),
                "pred_class_name": class_name_map.get(int(pred), str(pred)),
                "top5_labels": labels,
                "top5_scores": [float(x) for x in scores],
                "corruption": corruption,
                "mode": mode,
                "class_correct": float(int(pred) == target),
                "top5_correct": float(target in labels),
                "beta_mean": float(beta[i]),
            })
    return records, torch.cat(embeddings, dim=0)


def summarize_records(records: list[dict[str, Any]], corruption: str, mode: str, file_name: str) -> dict[str, Any]:
    if not records:
        return {"file": file_name, "corruption": corruption, "mode": mode, "count": 0, "accuracy": 0.0, "top5_accuracy": 0.0, "class_hit": 0.0}
    row = {
        "file": file_name,
        "corruption": corruption,
        "mode": mode,
        "count": len(records),
        "accuracy": float(np.mean([float(r.get("class_correct", 0.0)) for r in records])),
        "top5_accuracy": float(np.mean([float(r.get("top5_correct", 0.0)) for r in records])),
        "class_hit": float(np.mean([float(r.get("class_correct", 0.0)) for r in records])),
    }
    if any("beta_mean" in r for r in records):
        row["beta_mean"] = float(np.mean([float(r.get("beta_mean", 0.0)) for r in records]))
    return row


def write_table(rows: list[dict[str, Any]], csv_path: Path, md_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    if not rows:
        rows = [{"status": "empty"}]
        headers = ["status"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(f"{row.get(h, ''):.6f}" if isinstance(row.get(h), float) else str(row.get(h, "")) for h in headers) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def gap_rows(metrics: list[dict[str, Any]], model: str, seed: int, eval_dir: Path) -> list[dict[str, Any]]:
    by: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in metrics:
        by[str(row["corruption"])][str(row["mode"])] = row
    rows: list[dict[str, Any]] = []
    for corr in CORRUPTIONS:
        m = by.get(corr, {})
        real = m.get("real_eeg")
        if real is None:
            continue
        vision = float(m.get("vision_only", {}).get("accuracy", 0.0))
        shuf = float(m.get("shuffled_eeg", {}).get("accuracy", 0.0))
        rand = float(m.get("random_eeg", {}).get("accuracy", 0.0))
        acc = float(real.get("accuracy", 0.0))
        rows.append({"model": model, "seed": seed, "corruption": corr, "real_top1": acc, "real_top5": float(real.get("top5_accuracy", 0.0)), "class_hit": float(real.get("class_hit", 0.0)), "vision_top1": vision, "shuffled_top1": shuf, "random_top1": rand, "real_minus_vision": acc - vision, "real_minus_shuffled": acc - shuf, "real_minus_random": acc - rand, "real_beats_vision": acc > vision, "real_beats_controls": acc > shuf and acc > rand, "eval_dir": str(eval_dir)})
    return rows


def evaluate_vtf(checkpoint: Path) -> Path:
    device = resolve_device()
    model, label_values, prototypes, cfg = load_vtf_checkpoint(checkpoint, device)
    _, _, class_name_map = load_label_bank(cfg, device)
    rows, _labels, _caps = load_rows_labels(Path(cfg.human_test_manifest), cfg.max_test_samples)
    out_dir = Path(cfg.output_dir) / "token_fusion" / f"{cfg.vtf_variant}_seed{cfg.seed}" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    eeg_cache = {
        mode: compute_eeg_embeddings(cfg, Path(cfg.human_test_manifest), mode, device, cfg.max_test_samples)
        for mode in ["real_eeg", "shuffled_eeg", "random_eeg"]
    }
    all_metrics: list[dict[str, Any]] = []
    for corr in CORRUPTIONS:
        visual_tokens = compute_clip_token_embeddings(cfg, Path(cfg.human_test_manifest), corr, device, cfg.max_test_samples)
        for mode in MODES:
            if mode == "vision_only":
                image_in = visual_tokens
                eeg_in = torch.zeros_like(eeg_cache["real_eeg"])
            elif mode == "eeg_only":
                image_in = torch.zeros_like(visual_tokens)
                eeg_in = eeg_cache["real_eeg"]
            else:
                image_in = visual_tokens
                eeg_in = eeg_cache[mode]
            recs, emb = vtf_predict_records(model, image_in, eeg_in, rows, label_values, prototypes, class_name_map, device, cfg.eval_batch_size, corr, mode)
            with (out_dir / f"{corr}_{mode}.jsonl").open("w", encoding="utf-8") as handle:
                for rec in recs:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if mode in {"vision_only", "real_eeg"}:
                torch.save(emb, out_dir / f"{corr}_{mode}_enhanced_emb.pt")
            all_metrics.append(summarize_records(recs, corr, mode, f"{corr}_{mode}.jsonl"))
    write_table(all_metrics, out_dir / "FULL_METRICS.csv", out_dir / "FULL_METRICS.md", "Token Fusion Full Metrics")
    summaries = gap_rows(all_metrics, cfg.vtf_variant, cfg.seed, out_dir)
    write_table(summaries, out_dir / "SUMMARY_METRICS.csv", out_dir / "SUMMARY_METRICS.md", "Token Fusion Summary Metrics")
    return out_dir


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strong = [r for r in rows if str(r.get("corruption")) in STRONG_CORRUPTIONS]
    if not strong:
        strong = rows
    mean_real = float(np.mean([float(r.get("real_top1", 0.0)) for r in strong])) if strong else 0.0
    rv = float(np.mean([float(r.get("real_minus_vision", 0.0)) for r in strong])) if strong else 0.0
    rs = float(np.mean([float(r.get("real_minus_shuffled", 0.0)) for r in strong])) if strong else 0.0
    rr = float(np.mean([float(r.get("real_minus_random", 0.0)) for r in strong])) if strong else 0.0
    return {"strong_real_top1_mean": mean_real, "real_minus_vision_mean": rv, "real_minus_shuffled_mean": rs, "real_minus_random_mean": rr, "win_rate_vision": float(np.mean([str(r.get("real_beats_vision")).lower() == "true" for r in strong])) if strong else 0.0, "win_rate_controls": float(np.mean([str(r.get("real_beats_controls")).lower() == "true" for r in strong])) if strong else 0.0, "score": mean_real + 0.5 * rv + 0.5 * rs + 0.5 * rr}


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_vtf(cfg: TokenGenConfig) -> list[dict[str, Any]]:
    out = Path(cfg.output_dir) / "token_fusion"
    rows: list[dict[str, Any]] = []
    for path in sorted(out.glob("*_seed*/eval/SUMMARY_METRICS.csv")):
        rows.extend(read_csv(path))
    write_table(rows, out / "metrics.csv", out / "metrics.md", "Token Fusion Metrics")
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(row)
    baseline = read_csv(Path("outputs/final_results/A2_FINAL_METRICS.csv"))
    selection = [{"model": "A2_final", **score_rows(baseline), "beats_A2_final": False}]
    base_score = selection[0]["score"]
    for model_name, model_rows in sorted(by_model.items()):
        scored = {"model": model_name, **score_rows(model_rows)}
        scored["beats_A2_final"] = scored["score"] > base_score
        selection.append(scored)
    selection.sort(key=lambda r: float(r["score"]), reverse=True)
    write_table(selection, out / "VTF_MODEL_SELECTION.csv", out / "VTF_MODEL_SELECTION.md", "VTF Model Selection")
    ckpt_root = out / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    for ckpt in out.glob("*_seed*/checkpoints/best.pt"):
        dst = ckpt_root / f"{ckpt.parent.parent.name}_best.pt"
        if not dst.exists():
            shutil.copy2(ckpt, dst)
    return selection


def valid_caption(text: str) -> tuple[bool, str]:
    text = str(text).strip()
    low = text.lower()
    if not text:
        return False, "empty"
    printable = sum(1 for ch in text if ch.isprintable())
    ascii_letters = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
    if printable < len(text) or (len(text) > 0 and printable / max(len(text), 1) < 0.95):
        return False, "control_chars"
    if ascii_letters < 3:
        return False, "non_caption_chars"
    if re.search(r"[^\x09\x0a\x0d\x20-\x7e]", text):
        return False, "non_ascii"
    if len(text.split()) > 24:
        return False, "too_long"
    words = re.findall(r"[a-z]+", low)
    if len(words) < 3:
        return False, "too_short"
    stopwords = {"a", "an", "the", "on", "in", "at", "of", "with", "and", "to", "for", "is", "are"}
    if not any(word not in stopwords and len(word) > 2 for word in words):
        return False, "no_content_word"
    if "http://" in low or "https://" in low or "www." in low:
        return False, "url"
    if "<" in text and ">" in text:
        return False, "html"
    if "|" in text and "---" in text:
        return False, "markdown_table"
    if "sorry" in low or "i'm not" in low or "cannot" in low:
        return False, "apology"
    if re.fullmatch(r"n\\d{8}", text):
        return False, "wnid"
    split_words = low.split()
    if len(split_words) >= 6:
        grams = [" ".join(split_words[i : i + 3]) for i in range(len(split_words) - 2)]
        if grams and max(Counter(grams).values()) > 1:
            return False, "repetition"
    return True, "valid"


def class_hit(caption: str, class_name: str) -> float:
    cap = caption.lower()
    parts = class_name.lower().replace("-", " ").split()
    return float(class_name.lower() in cap or any(len(p) > 3 and p in cap for p in parts))


class WordCaptionTokenizer:
    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = "<unk>"

    def __init__(self, stoi: dict[str, int]) -> None:
        self.stoi = dict(stoi)
        self.itos = {idx: token for token, idx in self.stoi.items()}
        self.pad_id = self.stoi[self.pad_token]
        self.bos_id = self.stoi[self.bos_token]
        self.eos_id = self.stoi[self.eos_token]
        self.unk_id = self.stoi[self.unk_token]

    @classmethod
    def from_captions(cls, captions: Sequence[str], max_vocab_size: int = 2048) -> "WordCaptionTokenizer":
        counter: Counter[str] = Counter()
        for caption in captions:
            counter.update(cls._tokenize(caption))
        tokens = [cls.pad_token, cls.bos_token, cls.eos_token, cls.unk_token]
        for token, _count in counter.most_common(max(0, max_vocab_size - len(tokens))):
            if token not in tokens:
                tokens.append(token)
        return cls({token: idx for idx, token in enumerate(tokens)})

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+|[.,]", str(text).lower())

    def encode(self, text: str, max_length: int) -> list[int]:
        ids = [self.bos_id]
        ids.extend(self.stoi.get(token, self.unk_id) for token in self._tokenize(text))
        ids.append(self.eos_id)
        ids = ids[:max_length]
        if ids[-1] != self.eos_id:
            ids[-1] = self.eos_id
        return ids + [self.pad_id] * max(0, max_length - len(ids))

    def batch_encode(self, captions: Sequence[str], max_length: int, device: torch.device) -> torch.Tensor:
        return torch.tensor([self.encode(caption, max_length) for caption in captions], dtype=torch.long, device=device)

    def decode(self, ids: Sequence[int]) -> str:
        words: list[str] = []
        for idx in ids:
            token = self.itos.get(int(idx), self.unk_token)
            if token == self.eos_token:
                break
            if token in {self.pad_token, self.bos_token, self.unk_token}:
                continue
            if token in {".", ","}:
                if words:
                    words[-1] = words[-1] + token
            else:
                words.append(token)
        return " ".join(words).strip()

    def state_dict(self) -> dict[str, Any]:
        return {"stoi": self.stoi}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "WordCaptionTokenizer":
        return cls({str(k): int(v) for k, v in state["stoi"].items()})

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)


class TrainablePrefixCaptionGenerator(nn.Module):
    """Small trainable token decoder conditioned on VTF/EEG features.

    This is used when frozen Qwen soft prompts are too unstable for the small
    dataset. It is still a free-form autoregressive caption generator: captions
    are learned token by token from natural BLIP/human targets, not selected from
    a class template.
    """

    def __init__(self, *, cond_dim: int, tokenizer: WordCaptionTokenizer, hidden_dim: int = 512, embed_dim: int = 256, max_text_length: int = 24) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.max_text_length = max_text_length
        self.embedding = nn.Embedding(tokenizer.vocab_size, embed_dim, padding_idx=tokenizer.pad_id)
        self.cond_to_hidden = nn.Sequential(nn.LayerNorm(cond_dim), nn.Linear(cond_dim, hidden_dim), nn.Tanh())
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, tokenizer.vocab_size)

    def forward(self, cond: torch.Tensor, captions: Sequence[str]) -> torch.Tensor:
        ids = self.tokenizer.batch_encode(captions, self.max_text_length, cond.device)
        inputs = ids[:, :-1]
        labels = ids[:, 1:]
        embeds = self.embedding(inputs)
        h0 = self.cond_to_hidden(cond.float()).unsqueeze(0)
        hidden, _ = self.gru(embeds, h0)
        logits = self.output(hidden)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=self.tokenizer.pad_id)

    @torch.no_grad()
    def generate(self, cond: torch.Tensor, max_new_tokens: int = 24) -> list[str]:
        self.eval()
        batch = cond.shape[0]
        device = cond.device
        token = torch.full((batch, 1), self.tokenizer.bos_id, dtype=torch.long, device=device)
        h = self.cond_to_hidden(cond.float()).unsqueeze(0)
        outputs: list[list[int]] = [[] for _ in range(batch)]
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            emb = self.embedding(token)
            hidden, h = self.gru(emb, h)
            next_token = self.output(hidden[:, -1]).argmax(dim=-1)
            token = next_token.unsqueeze(1)
            for i, idx in enumerate(next_token.detach().cpu().tolist()):
                if not bool(finished[i]):
                    outputs[i].append(int(idx))
            finished = finished | (next_token == self.tokenizer.eos_id)
            if bool(finished.all()):
                break
        return [self.tokenizer.decode(ids) for ids in outputs]

    def checkpoint_payload(self, cfg: TokenGenConfig) -> dict[str, Any]:
        return {
            "decoder": self.state_dict(),
            "tokenizer": self.tokenizer.state_dict(),
            "config": asdict(cfg),
            "cond_dim": self.cond_dim,
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "max_text_length": self.max_text_length,
            "generator_type": "trainable_gru_caption_decoder",
        }


class PrefixGenerator(nn.Module):
    def __init__(self, base: SoftPromptCaptionModel, cond_dim: int) -> None:
        super().__init__()
        self.base = base
        hidden = int(getattr(base.lm.config, "hidden_size"))
        self.projector = nn.Sequential(nn.LayerNorm(cond_dim), nn.Linear(cond_dim, hidden * base.prompt_tokens))
        base.prompt_projector = self.projector
        self.input_dim = cond_dim

    def forward(self, cond: torch.Tensor, captions: Sequence[str]) -> torch.Tensor:
        return self.base(cond, captions)

    def generate(self, cond: torch.Tensor, max_new_tokens: int) -> list[str]:
        return self.base.generate(cond, max_new_tokens=max_new_tokens, temperature=0.2)


def build_generation_features(cfg: TokenGenConfig, split: str, mode: str, corruption: str = "clean") -> tuple[torch.Tensor, list[dict[str, Any]], list[str]]:
    device = resolve_device()
    manifest = Path(cfg.train_manifest if split == "train" else cfg.val_manifest if split == "val" else cfg.test_manifest)
    human_manifest = Path(cfg.human_train_manifest if split == "train" else cfg.human_val_manifest if split == "val" else cfg.human_test_manifest)
    max_samples = cfg.max_train_samples if split == "train" else cfg.max_val_samples if split == "val" else cfg.max_test_samples
    rows, _labels, captions = load_rows_labels(manifest, max_samples)
    token_manifest = human_manifest
    token_corruption = corruption if split == "test" else "clean"
    visual_tokens = compute_clip_token_embeddings(cfg, token_manifest, token_corruption, device, max_samples)
    image = F.normalize(visual_tokens[:, 0].float(), dim=-1)
    eeg = compute_eeg_embeddings(cfg, human_manifest, "real_eeg" if mode in {"vision_only", "eeg_only"} else mode, device, max_samples)
    if mode == "vision_only":
        cond = torch.cat([image, torch.zeros_like(eeg)], dim=-1)
    elif mode == "eeg_only":
        cond = torch.cat([torch.zeros_like(image), eeg], dim=-1)
    else:
        cond = torch.cat([image, eeg], dim=-1)
    if cfg.gen_variant in {"G1_vtf_visual_prefix", "G2_vtf_visual_eeg_prefix", "G3_vtf_visual_eeg_topk_prefix"}:
        vtf_ckpt = best_vtf_checkpoint(cfg)
        if vtf_ckpt.exists():
            model, label_values, prototypes, _vtf_cfg = load_vtf_checkpoint(vtf_ckpt, device)
            image_for_vtf = visual_tokens
            eeg_for_vtf = eeg if mode != "vision_only" else torch.zeros_like(eeg)
            embs: list[torch.Tensor] = []
            logits_list: list[torch.Tensor] = []
            with torch.no_grad():
                for start in range(0, image_for_vtf.shape[0], cfg.eval_batch_size):
                    logits, emb, _aux = model(image_for_vtf[start:start+cfg.eval_batch_size].to(device), eeg_for_vtf[start:start+cfg.eval_batch_size].to(device), prototypes)
                    embs.append(emb.cpu())
                    logits_list.append(logits.cpu())
            enhanced = torch.cat(embs, dim=0)
            if cfg.gen_variant == "G1_vtf_visual_prefix":
                cond = torch.cat([enhanced, torch.zeros_like(eeg)], dim=-1)
            else:
                cond = torch.cat([enhanced, eeg], dim=-1)
            if cfg.gen_variant == "G3_vtf_visual_eeg_topk_prefix":
                logits = torch.cat(logits_list, dim=0)
                probs = torch.softmax(logits, dim=-1)
                proto_context = F.normalize(probs @ prototypes.detach().cpu().float(), dim=-1)
                conf = probs.max(dim=-1, keepdim=True).values
                pad = torch.zeros((cond.shape[0], 511), dtype=cond.dtype)
                confidence_block = torch.cat([conf.to(cond.dtype), pad], dim=-1)
                cond = torch.cat([cond, proto_context.to(cond.dtype), confidence_block], dim=-1)
    return cond.float(), rows, captions


def best_vtf_checkpoint(cfg: TokenGenConfig) -> Path:
    preferred = Path(cfg.output_dir) / "token_fusion" / f"{cfg.vtf_variant}_seed{cfg.seed}" / "checkpoints" / "best.pt"
    if preferred.exists():
        return preferred
    candidates = sorted((Path(cfg.output_dir) / "token_fusion").glob("*_seed*/checkpoints/best.pt"))
    return candidates[0] if candidates else preferred


def write_caption_target_report(cfg: TokenGenConfig) -> None:
    rows, _labels, captions = load_rows_labels(Path(cfg.train_manifest), cfg.max_train_samples)
    lengths = [len(c.split()) for c in captions]
    invalid = [not valid_caption(c)[0] for c in captions]
    lines = ["# Caption Target Report", "", f"- caption source: `{rows[0].get('caption_source', 'unknown') if rows else 'none'}`", f"- number of captions: `{len(captions)}`", f"- average caption length: `{float(np.mean(lengths)) if lengths else 0.0:.3f}`", f"- invalid target rate: `{float(np.mean(invalid)) if invalid else 0.0:.6f}`", "", "## Example Targets", ""]
    for c in captions[:10]:
        lines.append(f"- {c}")
    out = Path(cfg.output_dir) / "generation" / "CAPTION_TARGET_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_generator(cfg: TokenGenConfig) -> Path:
    seed_everything(cfg.seed)
    device = resolve_device()
    out = Path(cfg.output_dir) / "generation" / f"{cfg.gen_variant}_seed{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", asdict(cfg))
    write_caption_target_report(cfg)
    append_gpu_usage(Path(cfg.output_dir) / "GPU_USAGE.md", f"{cfg.gen_variant}_seed{cfg.seed}", "generation_train_start")
    train_cond, _train_rows, train_caps = build_generation_features(cfg, "train", "real_eeg")
    val_cond, _val_rows, val_caps = build_generation_features(cfg, "val", "real_eeg")
    if cfg.qwen_prefix:
        base = SoftPromptCaptionModel(input_dim=train_cond.shape[1], prompt_tokens=cfg.prefix_len, max_text_length=cfg.max_caption_length, model_name=cfg.llm_model, use_tiny_debug_model=cfg.use_tiny_lm, freeze_lm=True)
        model: nn.Module = PrefixGenerator(base, train_cond.shape[1]).to(device)
        trainable_params = list(model.projector.parameters())  # type: ignore[attr-defined]
    else:
        tokenizer = WordCaptionTokenizer.from_captions(train_caps, max_vocab_size=2048)
        model = TrainablePrefixCaptionGenerator(cond_dim=train_cond.shape[1], tokenizer=tokenizer, hidden_dim=512, embed_dim=256, max_text_length=cfg.max_caption_length).to(device)
        trainable_params = list(model.parameters())
    opt = torch.optim.AdamW(trainable_params, lr=cfg.gen_lr, weight_decay=0.01)
    loader = DataLoader(list(range(train_cond.shape[0])), batch_size=cfg.gen_batch_size, shuffle=True)
    best_val = math.inf
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, cfg.gen_epochs + 1):
        model.train()
        losses: list[float] = []
        for idxs in loader:
            idxs = idxs.tolist()
            cond = train_cond[idxs].to(device)
            caps = [train_caps[i] for i in idxs]
            loss = model(cond, caps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_losses: list[float] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, val_cond.shape[0], cfg.gen_batch_size):
                cond = val_cond[start:start+cfg.gen_batch_size].to(device)
                caps = val_caps[start:start+cfg.gen_batch_size]
                val_losses.append(float(model(cond, caps).detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": float(np.mean(val_losses))}
        history.append(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            stale = 0
            ckpt_dir = out / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            if isinstance(model, TrainablePrefixCaptionGenerator):
                payload = model.checkpoint_payload(cfg)
            else:
                payload = {"projector": model.projector.state_dict(), "config": asdict(cfg), "cond_dim": train_cond.shape[1], "using_tiny_lm": model.base.using_tiny_lm, "generator_type": "qwen_soft_prefix"}  # type: ignore[attr-defined]
            torch.save(payload, ckpt_dir / "best.pt")
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    write_json(out / "history.json", history)
    (out / "summary.md").write_text(render_generation_train_summary(cfg.gen_variant, history, best_val), encoding="utf-8")
    append_gpu_usage(Path(cfg.output_dir) / "GPU_USAGE.md", f"{cfg.gen_variant}_seed{cfg.seed}", "generation_train_end")
    return out / "checkpoints" / "best.pt"


def render_generation_train_summary(name: str, history: list[dict[str, Any]], best: float) -> str:
    lines = [f"# {name} Training Summary", "", f"- Best val loss: `{best:.6f}`", "", "| Epoch | Train Loss | Val Loss |", "| ---: | ---: | ---: |"]
    for row in history:
        lines.append(f"| {row['epoch']} | {row['train_loss']:.6f} | {row['val_loss']:.6f} |")
    return "\n".join(lines) + "\n"


def load_generator(ckpt: Path, device: torch.device) -> tuple[nn.Module, TokenGenConfig]:
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = TokenGenConfig(**payload["config"])
    if payload.get("generator_type") == "trainable_gru_caption_decoder":
        tokenizer = WordCaptionTokenizer.from_state_dict(payload["tokenizer"])
        model = TrainablePrefixCaptionGenerator(
            cond_dim=int(payload["cond_dim"]),
            tokenizer=tokenizer,
            hidden_dim=int(payload.get("hidden_dim", 512)),
            embed_dim=int(payload.get("embed_dim", 256)),
            max_text_length=int(payload.get("max_text_length", cfg.max_caption_length)),
        ).to(device)
        model.load_state_dict(payload["decoder"])
    else:
        base = SoftPromptCaptionModel(input_dim=int(payload["cond_dim"]), prompt_tokens=cfg.prefix_len, max_text_length=cfg.max_caption_length, model_name=cfg.llm_model, use_tiny_debug_model=bool(payload.get("using_tiny_lm", cfg.use_tiny_lm)), freeze_lm=True)
        model = PrefixGenerator(base, int(payload["cond_dim"])).to(device)
        model.projector.load_state_dict(payload["projector"])
    model.eval()
    return model, cfg


def evaluate_generator(ckpt: Path) -> Path:
    device = resolve_device()
    model, cfg = load_generator(ckpt, device)
    out = Path(cfg.output_dir) / "generation" / f"{cfg.gen_variant}_seed{cfg.seed}" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    _labels, _protos, class_name_map = load_label_bank(cfg, device)
    all_metrics: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for corr in CORRUPTIONS:
        for mode in MODES:
            cond, rows, refs = build_generation_features(cfg, "test", mode, corr)
            preds: list[str] = []
            with torch.no_grad():
                for start in range(0, cond.shape[0], cfg.gen_batch_size):
                    preds.extend(model.generate(cond[start:start+cfg.gen_batch_size].to(device), cfg.max_new_tokens))
            records: list[dict[str, Any]] = []
            for row, ref, pred in zip(rows, refs, preds, strict=False):
                label = int(row["label"])
                class_name = class_name_map.get(label, str(label))
                is_valid, reason = valid_caption(pred)
                hit = class_hit(pred, class_name)
                records.append({"image_id": str(row["image_id"]), "true_class": class_name, "label": label, "corruption": corr, "mode": mode, "reference": ref, "generated_caption": pred, "valid": is_valid, "invalid_reason": reason, "class_hit": hit, "length": len(str(pred).split())})
            with (out / f"{corr}_{mode}.jsonl").open("w", encoding="utf-8") as handle:
                for rec in records:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            all_records.extend(records)
            all_metrics.append(summarize_generation(records, corr, mode, f"{corr}_{mode}.jsonl"))
    write_table(all_metrics, out / "GENERATION_METRICS.csv", out / "GENERATION_METRICS.md", "Generation Metrics")
    write_invalid_report(all_records, Path(cfg.output_dir) / "generation" / "INVALID_OUTPUT_REPORT.md")
    write_qualitative_examples(all_records, Path(cfg.output_dir) / "generation" / "QUALITATIVE_EXAMPLES.md")
    aggregate_generation(cfg)
    return out


def summarize_generation(records: list[dict[str, Any]], corruption: str, mode: str, file_name: str) -> dict[str, Any]:
    valid = [bool(r["valid"]) for r in records]
    hits = [float(r["class_hit"]) for r in records]
    captions = [str(r["generated_caption"]) for r in records]
    return {"file": file_name, "corruption": corruption, "mode": mode, "count": len(records), "caption_class_hit": float(np.mean(hits)) if hits else 0.0, "valid_caption_rate": float(np.mean(valid)) if valid else 0.0, "invalid_output_rate": 1.0 - float(np.mean(valid)) if valid else 1.0, "avg_caption_length": float(np.mean([r["length"] for r in records])) if records else 0.0, "distinct_caption_count": len(set(captions))}


def write_invalid_report(records: list[dict[str, Any]], path: Path) -> None:
    counts = Counter(str(r["invalid_reason"]) for r in records)
    lines = ["# Invalid Output Report", "", "| Reason | Count |", "| --- | ---: |"]
    for reason, count in counts.most_common():
        lines.append(f"| {reason} | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_qualitative_examples(records: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Qualitative Generated Caption Examples", "", "## Best examples for course report", ""]
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[(str(record["image_id"]), str(record["corruption"]))][str(record["mode"])] = record
    best: list[dict[str, Any]] = []
    for (_image_id, _corr), modes in grouped.items():
        real = modes.get("real_eeg")
        if real is None or not real.get("valid"):
            continue
        real_hit = float(real.get("class_hit", 0.0))
        control_hits = [
            float(modes.get(name, {}).get("class_hit", 0.0))
            for name in ("vision_only", "shuffled_eeg", "random_eeg")
        ]
        if real_hit > max(control_hits):
            best.append(real)
    best.sort(key=lambda r: (str(r["corruption"]) == "clean", -float(r.get("class_hit", 0.0))))
    if len(best) < 5:
        best = [r for r in records if r["mode"] == "real_eeg" and r["valid"] and r["class_hit"] > 0]
    if len(best) < 5:
        best = [r for r in records if r["mode"] == "real_eeg" and r["valid"]]
    for r in best[:5]:
        lines.append(f"- `{r['image_id']}` / `{r['corruption']}` / `{r['mode']}`: true `{r['true_class']}`, caption: {r['generated_caption']} (`hit={r['class_hit']}`)")
        peers = grouped.get((str(r["image_id"]), str(r["corruption"])), {})
        for peer_mode in ("vision_only", "shuffled_eeg", "random_eeg"):
            peer = peers.get(peer_mode)
            if peer is not None:
                lines.append(f"  - {peer_mode}: {peer['generated_caption']} (`hit={peer['class_hit']}`, valid={peer['valid']})")
    lines.extend(["", "## At Least 30 Examples", "", "| image_id | true class | corruption | mode | generated caption | valid | class hit |", "| --- | --- | --- | --- | --- | --- | ---: |"])
    preferred = [r for r in records if r["mode"] in {"real_eeg", "vision_only", "shuffled_eeg", "random_eeg"}]
    preferred.sort(key=lambda r: (str(r["corruption"]) != "clean", str(r["image_id"]), ["real_eeg", "vision_only", "shuffled_eeg", "random_eeg"].index(str(r["mode"])) if str(r["mode"]) in {"real_eeg", "vision_only", "shuffled_eeg", "random_eeg"} else 99))
    deduped: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for record in preferred:
        key = (str(record["image_id"]), str(record["corruption"]), str(record["mode"]))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(record)
    for r in deduped[: max(30, min(80, len(deduped)))]:
        cap = str(r["generated_caption"]).replace("|", "/")
        lines.append(f"| {r['image_id']} | {r['true_class']} | {r['corruption']} | {r['mode']} | {cap} | {r['valid']} | {r['class_hit']:.1f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_generation(cfg: TokenGenConfig) -> list[dict[str, Any]]:
    gen_root = Path(cfg.output_dir) / "generation"
    rows: list[dict[str, Any]] = []
    for path in sorted(gen_root.glob("*_seed*/eval/GENERATION_METRICS.csv")):
        variant = path.parent.parent.name.rsplit("_seed", 1)[0]
        seed = path.parent.parent.name.rsplit("_seed", 1)[1]
        for row in read_csv(path):
            row["model"] = variant
            row["seed"] = seed
            rows.append(row)
    write_table(rows, gen_root / "GENERATION_METRICS.csv", gen_root / "GENERATION_METRICS.md", "Generation Metrics All")
    selection: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(row)
    for model_name, model_rows in sorted(by_model.items()):
        strong = [r for r in model_rows if r["corruption"] in STRONG_CORRUPTIONS and r["mode"] == "real_eeg"]
        real = [r for r in model_rows if r["mode"] == "real_eeg"]
        selection.append({"model": model_name, "real_strong_class_hit": float(np.mean([float(r["caption_class_hit"]) for r in strong])) if strong else 0.0, "valid_caption_rate": float(np.mean([float(r["valid_caption_rate"]) for r in real])) if real else 0.0, "invalid_output_rate": float(np.mean([float(r["invalid_output_rate"]) for r in real])) if real else 1.0, "score": (float(np.mean([float(r["caption_class_hit"]) for r in strong])) if strong else 0.0) + 0.2 * (float(np.mean([float(r["valid_caption_rate"]) for r in real])) if real else 0.0)})
    selection.sort(key=lambda r: float(r["score"]), reverse=True)
    write_table(selection, gen_root / "GENERATION_MODEL_SELECTION.csv", gen_root / "GENERATION_MODEL_SELECTION.md", "Generation Model Selection")
    ckpt_dir = gen_root / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    for ckpt in gen_root.glob("*_seed*/checkpoints/best.pt"):
        dst = ckpt_dir / f"{ckpt.parent.parent.name}_best.pt"
        if not dst.exists():
            shutil.copy2(ckpt, dst)
    return selection


def final_reports(cfg: TokenGenConfig) -> None:
    root = Path(cfg.output_dir)
    vtf_sel = read_csv(root / "token_fusion" / "VTF_MODEL_SELECTION.csv")
    gen_sel = read_csv(root / "generation" / "GENERATION_MODEL_SELECTION.csv")
    token_best = vtf_sel[0] if vtf_sel else {}
    gen_best = gen_sel[0] if gen_sel else {}
    final_rows = []
    if token_best:
        final_rows.append({"stage": "token_fusion", **token_best})
    if gen_best:
        final_rows.append({"stage": "generation", **gen_best})
    write_table(final_rows, root / "TOKEN_GEN_EVLM_MODEL_SELECTION.csv", root / "TOKEN_GEN_EVLM_MODEL_SELECTION.md", "Token Generative EVLM Model Selection")
    q = root / "generation" / "QUALITATIVE_EXAMPLES.md"
    lines = ["# Token Generative EVLM Final Report", "", f"- Did token-level EEG-VLM fusion improve over A2_final? `{token_best.get('beats_A2_final', 'unknown')}`", "- Did real EEG beat shuffled/random EEG? See token_fusion metrics and generation metrics; report uses explicit controls.", f"- Which token fusion variant worked best? `{token_best.get('model', 'unknown')}`", f"- Did we produce free-form generated captions? `{q.exists() and q.stat().st_size > 0}`", f"- Which generative variant produced the best captions? `{gen_best.get('model', 'unknown')}`", f"- Invalid output rate: `{gen_best.get('invalid_output_rate', 'unknown')}`", f"- Should A2_final remain the main quantitative result? `yes unless VTF selection explicitly beats it`", "- Should token-level/generative EVLM be included as exploratory result? `yes, with limitations and qualitative examples`", "", "## Required Files", "", f"- Token fusion selection: `{root / 'token_fusion' / 'VTF_MODEL_SELECTION.md'}`", f"- Generation examples: `{q}`", f"- Generation metrics: `{root / 'generation' / 'GENERATION_METRICS.csv'}`"]
    (root / "TOKEN_GEN_EVLM_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    model = TokenFusionModel("VTF3_confidence_beta_margin_M4", hidden_dim=32)
    image = F.normalize(torch.randn(3, 50, 512), dim=-1)
    eeg = F.normalize(torch.randn(3, 512), dim=-1)
    protos = F.normalize(torch.randn(5, 512), dim=-1)
    logits, emb, aux = model(image, eeg, protos)
    assert logits.shape == (3, 5)
    assert emb.shape == (3, 512)
    assert aux["enhanced_tokens"].shape == (3, 50, 512)
    assert "beta" in aux
    ok, _ = valid_caption("a dog running on grass")
    assert ok
    print(json.dumps({"self_test": "ok"}))


def run_pipeline(args: argparse.Namespace) -> None:
    cfg = TokenGenConfig(
        output_dir=args.output_dir,
        vtf_variant=args.vtf_variant,
        gen_variant=args.gen_variant,
        seed=args.seed,
        vtf_epochs=args.vtf_epochs,
        gen_epochs=args.gen_epochs,
        batch_size=args.batch_size,
        gen_batch_size=args.gen_batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        use_tiny_lm=args.use_tiny_lm,
        qwen_prefix=args.qwen_prefix,
        force=args.force,
    )
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        self_test()
        return
    if args.aggregate_only:
        aggregate_vtf(cfg)
        aggregate_generation(cfg)
        final_reports(cfg)
        return
    vtf_ckpt = root / "token_fusion" / f"{cfg.vtf_variant}_seed{cfg.seed}" / "checkpoints" / "best.pt"
    if args.skip_vtf and not vtf_ckpt.exists():
        raise FileNotFoundError(f"--skip_vtf requested but checkpoint is missing: {vtf_ckpt}")
    if not args.skip_vtf and (cfg.force or not vtf_ckpt.exists()):
        vtf_ckpt = train_vtf(cfg)
    if not (root / "token_fusion" / f"{cfg.vtf_variant}_seed{cfg.seed}" / "eval" / "SUMMARY_METRICS.csv").exists() or cfg.force:
        evaluate_vtf(vtf_ckpt)
    aggregate_vtf(cfg)
    gen_ckpt = root / "generation" / f"{cfg.gen_variant}_seed{cfg.seed}" / "checkpoints" / "best.pt"
    if not args.skip_generation and (cfg.force or not gen_ckpt.exists()):
        gen_ckpt = train_generator(cfg)
    if not args.skip_generation and ((root / "generation" / f"{cfg.gen_variant}_seed{cfg.seed}" / "eval" / "GENERATION_METRICS.csv").exists() is False or cfg.force):
        evaluate_generator(gen_ckpt)
    aggregate_generation(cfg)
    final_reports(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run token-level and generative EVLM experiments.")
    parser.add_argument("--output_dir", default="outputs/token_generative_evlm")
    parser.add_argument("--vtf_variant", choices=VTF_VARIANTS, default="VTF3_confidence_beta_margin_M4")
    parser.add_argument("--gen_variant", choices=GEN_VARIANTS, default="G3_vtf_visual_eeg_topk_prefix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vtf_epochs", type=int, default=30)
    parser.add_argument("--gen_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gen_batch_size", type=int, default=16)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--use_tiny_lm", action="store_true")
    parser.add_argument("--qwen_prefix", action="store_true", help="Use frozen Qwen soft-prefix generator instead of the default trainable caption decoder.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_vtf", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
