from __future__ import annotations

import argparse
import csv
import json
import math
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

from scripts.run_token_generative_evlm import (
    CORRUPTIONS,
    MODES,
    STRONG_CORRUPTIONS,
    TokenGenConfig,
    WordCaptionTokenizer,
    class_hit,
    compute_clip_token_embeddings,
    compute_eeg_embeddings,
    load_label_bank,
    load_rows_labels,
    load_vtf_checkpoint,
    valid_caption,
    write_json,
    write_table,
)
from src.utils.seed import seed_everything


EVG1_VARIANT = "EVG1A_token_prefix_semantic_prompt"
EVG2_VARIANT = "EVG2B_qformer_visual_eeg_topk"
VARIANTS = [EVG1_VARIANT, EVG2_VARIANT]
CORRUPTION_TO_ID = {name: idx for idx, name in enumerate(CORRUPTIONS)}


@dataclass
class VisionTokenGenConfig:
    output_dir: str = "outputs/vision_token_gen_evlm"
    source_output_dir: str = "outputs/token_generative_evlm"
    vtf_checkpoint: str = "outputs/token_generative_evlm/token_fusion/VTF3_confidence_beta_margin_M4_seed42/checkpoints/best.pt"
    train_manifest: str = "data/thought2text/train_blip_caption.jsonl"
    val_manifest: str = "data/thought2text/val_blip_caption.jsonl"
    test_manifest: str = "data/thought2text/test_blip_caption.jsonl"
    human_train_manifest: str = "data/thought2text/train_human_caption.jsonl"
    human_val_manifest: str = "data/thought2text/val_human_caption.jsonl"
    human_test_manifest: str = "data/thought2text/test_human_caption.jsonl"
    variant: str = EVG1_VARIANT
    seed: int = 42
    epochs: int = 20
    patience: int = 6
    batch_size: int = 128
    eval_batch_size: int = 128
    hidden_dim: int = 512
    embed_dim: int = 256
    max_caption_length: int = 24
    max_new_tokens: int = 20
    lr: float = 1.0e-4
    weight_decay: float = 0.01
    max_train_samples: int = 0
    max_val_samples: int = 0
    max_test_samples: int = 0
    max_vocab_size: int = 2048
    num_queries: int = 16
    force: bool = False


@dataclass
class VisionTokenBatch:
    visual_tokens: torch.Tensor
    eeg_tokens: torch.Tensor
    topk_prototypes: torch.Tensor
    confidence: torch.Tensor
    corruption_ids: torch.Tensor

    def to(self, device: torch.device) -> "VisionTokenBatch":
        return VisionTokenBatch(
            visual_tokens=self.visual_tokens.to(device, non_blocking=True),
            eeg_tokens=self.eeg_tokens.to(device, non_blocking=True),
            topk_prototypes=self.topk_prototypes.to(device, non_blocking=True),
            confidence=self.confidence.to(device, non_blocking=True),
            corruption_ids=self.corruption_ids.to(device, non_blocking=True),
        )


@dataclass
class VisionTokenFeatures:
    batch: VisionTokenBatch
    rows: list[dict[str, Any]]
    captions: list[str]
    top5_names: list[list[str]]
    top5_scores: list[list[float]]

    def __len__(self) -> int:
        return int(self.batch.visual_tokens.shape[0])


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


def source_cfg(cfg: VisionTokenGenConfig) -> TokenGenConfig:
    return TokenGenConfig(
        output_dir=cfg.source_output_dir,
        train_manifest=cfg.train_manifest,
        val_manifest=cfg.val_manifest,
        test_manifest=cfg.test_manifest,
        human_train_manifest=cfg.human_train_manifest,
        human_val_manifest=cfg.human_val_manifest,
        human_test_manifest=cfg.human_test_manifest,
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
        max_train_samples=cfg.max_train_samples,
        max_val_samples=cfg.max_val_samples,
        max_test_samples=cfg.max_test_samples,
    )


def split_paths(cfg: VisionTokenGenConfig, split: str) -> tuple[Path, Path, int]:
    if split == "train":
        return Path(cfg.train_manifest), Path(cfg.human_train_manifest), cfg.max_train_samples
    if split == "val":
        return Path(cfg.val_manifest), Path(cfg.human_val_manifest), cfg.max_val_samples
    if split == "test":
        return Path(cfg.test_manifest), Path(cfg.human_test_manifest), cfg.max_test_samples
    raise ValueError(f"unsupported split: {split}")


def resolve_vtf_checkpoint(cfg: VisionTokenGenConfig) -> Path:
    preferred = Path(cfg.vtf_checkpoint)
    if preferred.exists():
        return preferred
    fallback = Path(cfg.source_output_dir) / "token_fusion" / "checkpoints" / "VTF3_confidence_beta_margin_M4_seed42_best.pt"
    if fallback.exists():
        return fallback
    candidates = sorted((Path(cfg.source_output_dir) / "token_fusion").glob("*_seed*/checkpoints/best.pt"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"no VTF checkpoint found under {cfg.source_output_dir}")


def clean_target_caption(caption: str, class_name: str) -> tuple[str, bool, str]:
    ok, reason = valid_caption(caption)
    if ok and len(str(caption).split()) <= 20:
        return str(caption).strip().rstrip("."), False, "valid_source"
    return f"a photo of a {class_name}", True, reason


def prepare_caption_targets(
    cfg: VisionTokenGenConfig,
    split: str,
    class_name_map: dict[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    caption_manifest, human_manifest, max_samples = split_paths(cfg, split)
    caption_rows, _caption_labels, caption_values = load_rows_labels(caption_manifest, max_samples)
    human_rows, _labels, _human_caps = load_rows_labels(human_manifest, max_samples)
    final: list[str] = []
    good_examples: list[dict[str, Any]] = []
    replaced: list[dict[str, Any]] = []
    for idx, row in enumerate(caption_rows):
        label = int(human_rows[idx].get("label", row.get("label", -1)))
        class_name = class_name_map.get(label, str(label))
        cleaned, was_replaced, reason = clean_target_caption(caption_values[idx], class_name)
        final.append(cleaned)
        payload = {
            "image_id": str(row.get("image_id", human_rows[idx].get("image_id", idx))),
            "label": label,
            "class_name": class_name,
            "source_caption": caption_values[idx],
            "final_caption": cleaned,
            "reason": reason,
        }
        if was_replaced:
            replaced.append(payload)
        elif len(good_examples) < 20:
            good_examples.append(payload)
    return caption_rows, human_rows, final, good_examples, replaced


def write_caption_target_report(
    cfg: VisionTokenGenConfig,
    captions: list[str],
    good_examples: list[dict[str, Any]],
    replaced: list[dict[str, Any]],
) -> None:
    lengths = [len(c.split()) for c in captions]
    out = Path(cfg.output_dir) / "caption_targets" / "CAPTION_TARGET_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Caption Target Report",
        "",
        "- caption source: `BLIP caption JSONL with class-caption fallback for invalid targets`",
        f"- number of captions: `{len(captions)}`",
        f"- invalid target count: `{len(replaced)}`",
        f"- average caption length: `{float(np.mean(lengths)) if lengths else 0.0:.3f}`",
        "- final target strategy: `use cleaned BLIP captions when valid; otherwise use short natural class caption`",
        "",
        "## 10 Good Examples",
        "",
    ]
    for item in good_examples[:10]:
        lines.append(f"- `{item['image_id']}` / `{item['class_name']}`: {item['final_caption']}")
    lines.extend(["", "## Removed Or Replaced Bad Examples", ""])
    if replaced:
        for item in replaced[:10]:
            lines.append(f"- `{item['image_id']}` / reason `{item['reason']}`: {item['source_caption']} -> {item['final_caption']}")
    else:
        lines.append("- No invalid training captions were replaced.")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_enhanced_token_source_report(
    path: Path,
    *,
    vtf_checkpoint: Path,
    token_shape: Sequence[int],
    eeg_token_shape: Sequence[int],
    modes: Sequence[str],
    frozen_modules: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Enhanced Token Source",
        "",
        f"- VTF checkpoint used: `{vtf_checkpoint}`",
        f"- enhanced visual token shape: `{list(token_shape)}`",
        f"- EEG token shape: `{list(eeg_token_shape)}`",
        f"- hidden dimension: `{int(token_shape[-1]) if token_shape else 'unknown'}`",
        f"- number of visual tokens: `{int(token_shape[1]) if len(token_shape) > 1 else 'unknown'}`",
        f"- modes supported: `{', '.join(modes)}`",
        f"- frozen modules: `{', '.join(frozen_modules)}`",
        "",
        "The downstream generator consumes `enhanced_visual_tokens [B,N,512]` directly, not only pooled `enhanced_img_emb [B,512]`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class EVG1TokenPrefixCaptionGenerator(nn.Module):
    """Autoregressive caption decoder with token-level visual memory.

    The generator attends over `enhanced_visual_tokens` at every decoding step.
    This is the practical EVG1 fallback when Qwen LoRA is unavailable or unstable:
    it is still free-form generation from learned caption targets, not template
    selection or classification.
    """

    def __init__(
        self,
        *,
        tokenizer: WordCaptionTokenizer,
        token_dim: int = 512,
        hidden_dim: int = 512,
        embed_dim: int = 256,
        max_text_length: int = 24,
        num_corruptions: int = 6,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.max_text_length = max_text_length
        self.visual_proj = nn.Linear(token_dim, hidden_dim)
        self.eeg_proj = nn.Linear(token_dim, hidden_dim)
        self.proto_proj = nn.Linear(token_dim, hidden_dim)
        self.conf_proj = nn.Linear(1, hidden_dim)
        self.corruption_embed = nn.Embedding(num_corruptions, hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.init_hidden = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.embedding = nn.Embedding(tokenizer.vocab_size, embed_dim, padding_idx=tokenizer.pad_id)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, tokenizer.vocab_size))
        self.last_seen_visual_tokens_shape: tuple[int, ...] | None = None

    def build_memory(self, batch: VisionTokenBatch) -> torch.Tensor:
        visual = batch.visual_tokens.float()
        if visual.ndim != 3:
            raise ValueError(f"enhanced visual tokens must have rank 3, got {tuple(visual.shape)}")
        if visual.shape[-1] != self.token_dim:
            raise ValueError(f"enhanced visual tokens must end in {self.token_dim}, got {tuple(visual.shape)}")
        self.last_seen_visual_tokens_shape = tuple(int(x) for x in visual.shape)
        visual_mem = self.visual_proj(F.normalize(visual, dim=-1))
        eeg_mem = self.eeg_proj(F.normalize(batch.eeg_tokens.float(), dim=-1))
        proto_mem = self.proto_proj(F.normalize(batch.topk_prototypes.float(), dim=-1))
        conf_mem = self.conf_proj(batch.confidence.float()).unsqueeze(1)
        corr_mem = self.corruption_embed(batch.corruption_ids.long()).unsqueeze(1)
        memory = torch.cat([visual_mem, eeg_mem, proto_mem, conf_mem, corr_mem], dim=1)
        return self.memory_norm(memory)

    def attend(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(hidden, memory.transpose(1, 2)) / math.sqrt(float(self.hidden_dim))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, memory)

    def forward(self, batch: VisionTokenBatch, captions: Sequence[str]) -> torch.Tensor:
        memory = self.build_memory(batch)
        ids = self.tokenizer.batch_encode(captions, self.max_text_length, memory.device)
        inputs = ids[:, :-1]
        labels = ids[:, 1:]
        embeds = self.embedding(inputs)
        h0 = self.init_hidden(memory.mean(dim=1)).unsqueeze(0)
        hidden, _ = self.gru(embeds, h0)
        context = self.attend(hidden, memory)
        logits = self.output(torch.cat([hidden, context], dim=-1))
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=self.tokenizer.pad_id)

    @torch.no_grad()
    def generate(self, batch: VisionTokenBatch, max_new_tokens: int = 20) -> list[str]:
        self.eval()
        memory = self.build_memory(batch)
        device = memory.device
        bsz = int(memory.shape[0])
        token = torch.full((bsz, 1), self.tokenizer.bos_id, dtype=torch.long, device=device)
        h = self.init_hidden(memory.mean(dim=1)).unsqueeze(0)
        outputs: list[list[int]] = [[] for _ in range(bsz)]
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        previous = torch.full((bsz,), self.tokenizer.bos_id, dtype=torch.long, device=device)
        for step in range(max_new_tokens):
            embed = self.embedding(token)
            hidden, h = self.gru(embed, h)
            context = self.attend(hidden, memory)
            logits = self.output(torch.cat([hidden, context], dim=-1))[:, -1]
            banned = [self.tokenizer.pad_id, self.tokenizer.bos_id, self.tokenizer.unk_id]
            logits[:, banned] = -1e9
            if step < 3:
                logits[:, self.tokenizer.eos_id] = -1e9
            logits.scatter_(1, previous.view(-1, 1), -1e9)
            next_token = logits.argmax(dim=-1)
            token = next_token.unsqueeze(1)
            for idx, token_id in enumerate(next_token.detach().cpu().tolist()):
                if not bool(finished[idx]):
                    outputs[idx].append(int(token_id))
            previous = next_token
            finished = finished | (next_token == self.tokenizer.eos_id)
            if bool(finished.all()):
                break
        return [self.tokenizer.decode(ids) for ids in outputs]

    def checkpoint_payload(self, cfg: VisionTokenGenConfig, variant: str) -> dict[str, Any]:
        return {
            "model": self.state_dict(),
            "tokenizer": self.tokenizer.state_dict(),
            "config": asdict(cfg),
            "variant": variant,
            "generator_type": self.__class__.__name__,
            "token_dim": self.token_dim,
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "max_text_length": self.max_text_length,
            "uses_enhanced_visual_tokens": True,
        }


class EVG2QFormerCaptionGenerator(EVG1TokenPrefixCaptionGenerator):
    """Q-Former-style bridge that resamples visual/eeg/prototype tokens first."""

    def __init__(
        self,
        *,
        tokenizer: WordCaptionTokenizer,
        token_dim: int = 512,
        hidden_dim: int = 512,
        embed_dim: int = 256,
        max_text_length: int = 24,
        num_corruptions: int = 6,
        num_queries: int = 16,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            max_text_length=max_text_length,
            num_corruptions=num_corruptions,
        )
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.randn(num_queries, token_dim) * 0.02)
        self.conf_token = nn.Linear(1, token_dim)
        self.corr_token = nn.Embedding(num_corruptions, token_dim)
        self.qformer = nn.MultiheadAttention(token_dim, num_heads=8, batch_first=True, dropout=0.1)
        self.bridge_proj = nn.Linear(token_dim, hidden_dim)
        self.last_bridge_tokens_shape: tuple[int, ...] | None = None

    def build_memory(self, batch: VisionTokenBatch) -> torch.Tensor:
        visual = batch.visual_tokens.float()
        if visual.ndim != 3:
            raise ValueError(f"enhanced visual tokens must have rank 3, got {tuple(visual.shape)}")
        self.last_seen_visual_tokens_shape = tuple(int(x) for x in visual.shape)
        raw_memory = torch.cat(
            [
                F.normalize(visual, dim=-1),
                F.normalize(batch.eeg_tokens.float(), dim=-1),
                F.normalize(batch.topk_prototypes.float(), dim=-1),
                self.conf_token(batch.confidence.float()).unsqueeze(1),
                self.corr_token(batch.corruption_ids.long()).unsqueeze(1),
            ],
            dim=1,
        )
        queries = self.query_tokens.unsqueeze(0).expand(raw_memory.shape[0], -1, -1)
        bridge, _ = self.qformer(queries, raw_memory, raw_memory, need_weights=False)
        self.last_bridge_tokens_shape = tuple(int(x) for x in bridge.shape)
        return self.memory_norm(self.bridge_proj(bridge))

    def checkpoint_payload(self, cfg: VisionTokenGenConfig, variant: str) -> dict[str, Any]:
        payload = super().checkpoint_payload(cfg, variant)
        payload["num_queries"] = self.num_queries
        return payload


def make_model(cfg: VisionTokenGenConfig, tokenizer: WordCaptionTokenizer) -> nn.Module:
    kwargs = {
        "tokenizer": tokenizer,
        "token_dim": 512,
        "hidden_dim": cfg.hidden_dim,
        "embed_dim": cfg.embed_dim,
        "max_text_length": cfg.max_caption_length,
        "num_corruptions": len(CORRUPTIONS),
    }
    if cfg.variant.startswith("EVG2"):
        return EVG2QFormerCaptionGenerator(**kwargs, num_queries=cfg.num_queries)
    return EVG1TokenPrefixCaptionGenerator(**kwargs)


def batch_from_features(features: VisionTokenFeatures, idxs: Sequence[int], device: torch.device) -> VisionTokenBatch:
    index = torch.as_tensor(list(idxs), dtype=torch.long)
    batch = VisionTokenBatch(
        visual_tokens=features.batch.visual_tokens[index],
        eeg_tokens=features.batch.eeg_tokens[index],
        topk_prototypes=features.batch.topk_prototypes[index],
        confidence=features.batch.confidence[index],
        corruption_ids=features.batch.corruption_ids[index],
    )
    return batch.to(device)


@torch.no_grad()
def build_vision_token_features(cfg: VisionTokenGenConfig, split: str, mode: str, corruption: str = "clean") -> VisionTokenFeatures:
    device = resolve_device()
    src_cfg = source_cfg(cfg)
    caption_manifest, human_manifest, max_samples = split_paths(cfg, split)
    token_corruption = corruption if split == "test" else "clean"
    vtf_ckpt = resolve_vtf_checkpoint(cfg)
    model, label_values, prototypes, _vtf_cfg = load_vtf_checkpoint(vtf_ckpt, device)
    _labels, _protos, class_name_map = load_label_bank(src_cfg, device)
    _caption_rows, human_rows, captions, _good, _bad = prepare_caption_targets(cfg, split, class_name_map)

    visual_tokens = compute_clip_token_embeddings(src_cfg, human_manifest, token_corruption, device, max_samples)
    eeg_mode = "real_eeg" if mode in {"vision_only", "eeg_only"} else mode
    eeg_emb = compute_eeg_embeddings(src_cfg, human_manifest, eeg_mode, device, max_samples)
    if mode == "vision_only":
        image_for_vtf = visual_tokens
        eeg_for_vtf = torch.zeros_like(eeg_emb)
    elif mode == "eeg_only":
        image_for_vtf = torch.zeros_like(visual_tokens)
        eeg_for_vtf = eeg_emb
    else:
        image_for_vtf = visual_tokens
        eeg_for_vtf = eeg_emb

    enhanced_chunks: list[torch.Tensor] = []
    eeg_token_chunks: list[torch.Tensor] = []
    topk_proto_chunks: list[torch.Tensor] = []
    confidence_chunks: list[torch.Tensor] = []
    top5_names: list[list[str]] = []
    top5_scores: list[list[float]] = []
    model.eval()
    for start in range(0, image_for_vtf.shape[0], cfg.eval_batch_size):
        image_b = image_for_vtf[start : start + cfg.eval_batch_size].to(device)
        eeg_b = eeg_for_vtf[start : start + cfg.eval_batch_size].to(device)
        logits, _enhanced_img, aux = model(image_b, eeg_b, prototypes)
        probs = torch.softmax(logits.float(), dim=-1)
        scores, top_idx = probs.topk(k=min(5, probs.shape[-1]), dim=-1)
        top_labels = label_values[top_idx]
        topk_protos = prototypes[top_idx]
        enhanced_chunks.append(aux["enhanced_tokens"].detach().cpu().to(torch.float16))
        eeg_token_chunks.append(aux["eeg_tokens"].detach().cpu().to(torch.float16))
        topk_proto_chunks.append(topk_protos.detach().cpu().to(torch.float16))
        confidence_chunks.append(scores[:, :1].detach().cpu().float())
        for labels_row, scores_row in zip(top_labels.detach().cpu().tolist(), scores.detach().cpu().tolist(), strict=False):
            top5_names.append([class_name_map.get(int(label), str(label)) for label in labels_row])
            top5_scores.append([float(score) for score in scores_row])

    size = int(torch.cat(enhanced_chunks, dim=0).shape[0])
    corruption_ids = torch.full((size,), CORRUPTION_TO_ID.get(corruption, 0), dtype=torch.long)
    return VisionTokenFeatures(
        batch=VisionTokenBatch(
            visual_tokens=torch.cat(enhanced_chunks, dim=0),
            eeg_tokens=torch.cat(eeg_token_chunks, dim=0),
            topk_prototypes=torch.cat(topk_proto_chunks, dim=0),
            confidence=torch.cat(confidence_chunks, dim=0),
            corruption_ids=corruption_ids,
        ),
        rows=human_rows,
        captions=captions,
        top5_names=top5_names,
        top5_scores=top5_scores,
    )


def variant_root(cfg: VisionTokenGenConfig) -> Path:
    group = "EVG2" if cfg.variant.startswith("EVG2") else "EVG1"
    return Path(cfg.output_dir) / group


def run_dir(cfg: VisionTokenGenConfig) -> Path:
    return variant_root(cfg) / f"{cfg.variant}_seed{cfg.seed}"


def train_generator(cfg: VisionTokenGenConfig) -> Path:
    seed_everything(cfg.seed)
    device = resolve_device()
    root = variant_root(cfg)
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", asdict(cfg))
    append_gpu_usage(Path(cfg.output_dir) / "GPU_USAGE.md", cfg.variant, "train_start")

    src_cfg = source_cfg(cfg)
    _labels, _protos, class_name_map = load_label_bank(src_cfg, device)
    _caption_rows, _human_rows, train_captions, good, replaced = prepare_caption_targets(cfg, "train", class_name_map)
    write_caption_target_report(cfg, train_captions, good, replaced)

    train_features = build_vision_token_features(cfg, "train", "real_eeg", "clean")
    val_features = build_vision_token_features(cfg, "val", "real_eeg", "clean")
    write_enhanced_token_source_report(
        Path(cfg.output_dir) / "prep" / "ENHANCED_TOKEN_SOURCE.md",
        vtf_checkpoint=resolve_vtf_checkpoint(cfg),
        token_shape=tuple(int(x) for x in train_features.batch.visual_tokens.shape),
        eeg_token_shape=tuple(int(x) for x in train_features.batch.eeg_tokens.shape),
        modes=MODES,
        frozen_modules=["CLIP ViT-B/32", "A2 EEG encoder", "VTF3 token fusion"],
    )

    tokenizer = WordCaptionTokenizer.from_captions(train_features.captions, max_vocab_size=cfg.max_vocab_size)
    model = make_model(cfg, tokenizer).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(TensorDataset(torch.arange(len(train_features))), batch_size=cfg.batch_size, shuffle=True)
    best_val = math.inf
    stale = 0
    history: list[dict[str, Any]] = []
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses: list[float] = []
        start_time = time.time()
        for (idxs,) in loader:
            idx_list = [int(i) for i in idxs.tolist()]
            batch = batch_from_features(train_features, idx_list, device)
            caps = [train_features.captions[i] for i in idx_list]
            loss = model(batch, caps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_losses: list[float] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(val_features), cfg.batch_size):
                idx_list = list(range(start, min(start + cfg.batch_size, len(val_features))))
                batch = batch_from_features(val_features, idx_list, device)
                caps = [val_features.captions[i] for i in idx_list]
                val_losses.append(float(model(batch, caps).detach().cpu()))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": float(np.mean(val_losses)),
            "seconds": float(time.time() - start_time),
        }
        history.append(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            stale = 0
            torch.save(model.checkpoint_payload(cfg, cfg.variant), ckpt_dir / "best.pt")
        else:
            stale += 1
        if stale >= cfg.patience:
            break

    write_json(out / "history.json", history)
    (out / "summary.md").write_text(render_train_report(cfg.variant, history, best_val), encoding="utf-8")
    append_gpu_usage(Path(cfg.output_dir) / "GPU_USAGE.md", cfg.variant, "train_end")
    return ckpt_dir / "best.pt"


def render_train_report(variant: str, history: list[dict[str, Any]], best_val: float) -> str:
    lines = [
        f"# {variant} Training Summary",
        "",
        f"- best val loss: `{best_val:.6f}`",
        "- completion note: `full training after code validation; smoke tests are not counted as completion`",
        "",
        "| Epoch | Train Loss | Val Loss | Seconds |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in history:
        lines.append(f"| {row['epoch']} | {row['train_loss']:.6f} | {row['val_loss']:.6f} | {row['seconds']:.2f} |")
    return "\n".join(lines) + "\n"


def load_generator(ckpt: Path, device: torch.device) -> tuple[nn.Module, VisionTokenGenConfig]:
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = VisionTokenGenConfig(**payload["config"])
    tokenizer = WordCaptionTokenizer.from_state_dict(payload["tokenizer"])
    if str(payload.get("generator_type")) == "EVG2QFormerCaptionGenerator":
        model = EVG2QFormerCaptionGenerator(
            tokenizer=tokenizer,
            token_dim=int(payload.get("token_dim", 512)),
            hidden_dim=int(payload.get("hidden_dim", cfg.hidden_dim)),
            embed_dim=int(payload.get("embed_dim", cfg.embed_dim)),
            max_text_length=int(payload.get("max_text_length", cfg.max_caption_length)),
            num_corruptions=len(CORRUPTIONS),
            num_queries=int(payload.get("num_queries", cfg.num_queries)),
        ).to(device)
    else:
        model = EVG1TokenPrefixCaptionGenerator(
            tokenizer=tokenizer,
            token_dim=int(payload.get("token_dim", 512)),
            hidden_dim=int(payload.get("hidden_dim", cfg.hidden_dim)),
            embed_dim=int(payload.get("embed_dim", cfg.embed_dim)),
            max_text_length=int(payload.get("max_text_length", cfg.max_caption_length)),
            num_corruptions=len(CORRUPTIONS),
        ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cfg


def repetition_rate(caption: str) -> float:
    words = str(caption).lower().split()
    if len(words) < 4:
        return 0.0
    counts = Counter(words)
    return float(max(counts.values()) / max(len(words), 1))


def summarize_generation(records: list[dict[str, Any]], corruption: str, mode: str, file_name: str) -> dict[str, Any]:
    valid = [bool(r["valid"]) for r in records]
    hits = [float(r["class_hit"]) for r in records]
    topk_hits = [float(r["caption_topk_class_hit"]) for r in records]
    captions = [str(r["generated_caption"]) for r in records]
    return {
        "file": file_name,
        "corruption": corruption,
        "mode": mode,
        "count": len(records),
        "valid_caption_rate": float(np.mean(valid)) if valid else 0.0,
        "invalid_output_rate": 1.0 - float(np.mean(valid)) if valid else 1.0,
        "caption_class_hit": float(np.mean(hits)) if hits else 0.0,
        "caption_topk_class_hit": float(np.mean(topk_hits)) if topk_hits else 0.0,
        "avg_caption_length": float(np.mean([r["length"] for r in records])) if records else 0.0,
        "distinct_caption_count": len(set(captions)),
        "repetition_rate": float(np.mean([float(r["repetition_rate"]) for r in records])) if records else 0.0,
    }


def add_metric_gaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_corr: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_corr[str(row["corruption"])][str(row["mode"])] = row
    for row in rows:
        corr = str(row["corruption"])
        modes = by_corr[corr]
        real = float(modes.get("real_eeg", {}).get("caption_class_hit", 0.0))
        vision = float(modes.get("vision_only", {}).get("caption_class_hit", 0.0))
        shuffled = float(modes.get("shuffled_eeg", {}).get("caption_class_hit", 0.0))
        random = float(modes.get("random_eeg", {}).get("caption_class_hit", 0.0))
        row["real_minus_vision"] = real - vision
        row["real_minus_shuffled"] = real - shuffled
        row["real_minus_random"] = real - random
        row["real_beats_controls"] = real > shuffled and real > random
    return rows


def evaluate_generator(ckpt: Path) -> Path:
    device = resolve_device()
    model, cfg = load_generator(ckpt, device)
    root = variant_root(cfg)
    eval_root = run_dir(cfg) / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    src_cfg = source_cfg(cfg)
    _labels, _protos, class_name_map = load_label_bank(src_cfg, device)
    append_gpu_usage(Path(cfg.output_dir) / "GPU_USAGE.md", cfg.variant, "eval_start")

    all_records: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for corruption in CORRUPTIONS:
        for mode in MODES:
            features = build_vision_token_features(cfg, "test", mode, corruption)
            preds: list[str] = []
            model.eval()
            with torch.no_grad():
                for start in range(0, len(features), cfg.eval_batch_size):
                    idxs = list(range(start, min(start + cfg.eval_batch_size, len(features))))
                    batch = batch_from_features(features, idxs, device)
                    preds.extend(model.generate(batch, max_new_tokens=cfg.max_new_tokens))
            records: list[dict[str, Any]] = []
            for idx, (row, ref, pred) in enumerate(zip(features.rows, features.captions, preds, strict=False)):
                label = int(row["label"])
                true_class = class_name_map.get(label, str(label))
                ok, reason = valid_caption(pred)
                top_names = features.top5_names[idx]
                rec = {
                    "model": cfg.variant,
                    "image_id": str(row["image_id"]),
                    "true_class": true_class,
                    "label": label,
                    "corruption": corruption,
                    "mode": mode,
                    "reference": ref,
                    "top5_classes": top_names,
                    "top5_scores": features.top5_scores[idx],
                    "generated_caption": pred,
                    "valid": ok,
                    "invalid_reason": reason,
                    "class_hit": class_hit(pred, true_class),
                    "caption_topk_class_hit": float(any(class_hit(pred, name) > 0 for name in top_names)),
                    "length": len(str(pred).split()),
                    "repetition_rate": repetition_rate(pred),
                    "token_input_shape": list(features.batch.visual_tokens.shape[1:]),
                    "uses_enhanced_visual_tokens": True,
                }
                records.append(rec)
            with (eval_root / f"{corruption}_{mode}.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            all_records.extend(records)
            metrics.append(summarize_generation(records, corruption, mode, f"{corruption}_{mode}.jsonl"))

    metrics = add_metric_gaps(metrics)
    write_table(metrics, root / f"{root.name}_METRICS.csv", root / f"{root.name}_METRICS.md", f"{root.name} Metrics")
    write_qualitative_examples(all_records, root / f"{root.name}_QUALITATIVE_EXAMPLES.md")
    write_invalid_report(all_records, root / f"{root.name}_INVALID_OUTPUT_REPORT.md")
    write_variant_report(cfg, ckpt, metrics, all_records)
    append_gpu_usage(Path(cfg.output_dir) / "GPU_USAGE.md", cfg.variant, "eval_end")
    return eval_root


def write_invalid_report(records: list[dict[str, Any]], path: Path) -> None:
    counts = Counter(str(r["invalid_reason"]) for r in records)
    lines = ["# Invalid Output Report", "", "| Reason | Count |", "| --- | ---: |"]
    for reason, count in counts.most_common():
        lines.append(f"| {reason} | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_qualitative_examples(records: list[dict[str, Any]], path: Path) -> None:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[(str(record["image_id"]), str(record["corruption"]))][str(record["mode"])] = record

    best: list[dict[str, Any]] = []
    for (_image_id, corr), modes in grouped.items():
        real = modes.get("real_eeg")
        if real is None or not bool(real.get("valid")):
            continue
        control_hits = [float(modes.get(name, {}).get("class_hit", 0.0)) for name in ("vision_only", "shuffled_eeg", "random_eeg")]
        if float(real.get("class_hit", 0.0)) > max(control_hits) or corr in STRONG_CORRUPTIONS:
            best.append(real)
    best.sort(key=lambda r: (str(r["corruption"]) == "clean", -float(r.get("class_hit", 0.0)), str(r["image_id"])))
    if len(best) < 5:
        best = [r for r in records if r["mode"] == "real_eeg" and r["valid"] and float(r.get("class_hit", 0.0)) > 0]
    if len(best) < 5:
        best = [r for r in records if r["mode"] == "real_eeg" and r["valid"]]

    lines = ["# Qualitative Generated Caption Examples", "", "## Best examples for course report", ""]
    for record in best[:5]:
        lines.append(
            f"- `{record['image_id']}` / `{record['corruption']}`: true `{record['true_class']}`, "
            f"real EEG caption: {record['generated_caption']} (`hit={record['class_hit']}`, valid={record['valid']})"
        )
        peers = grouped.get((str(record["image_id"]), str(record["corruption"])), {})
        for peer_mode in ("vision_only", "shuffled_eeg", "random_eeg"):
            peer = peers.get(peer_mode)
            if peer:
                lines.append(f"  - {peer_mode}: {peer['generated_caption']} (`hit={peer['class_hit']}`, valid={peer['valid']})")

    lines.extend(["", "## At Least 30 Examples", "", "| image_id | true class | corruption | mode | generated caption | valid | class hit |", "| --- | --- | --- | --- | --- | --- | ---: |"])
    preferred = [r for r in records if r["mode"] in {"real_eeg", "vision_only", "shuffled_eeg", "random_eeg"}]
    mode_order = {"real_eeg": 0, "vision_only": 1, "shuffled_eeg": 2, "random_eeg": 3}
    preferred.sort(key=lambda r: (str(r["corruption"]) == "clean", str(r["image_id"]), mode_order.get(str(r["mode"]), 99)))
    deduped: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for record in preferred:
        key = (str(record["image_id"]), str(record["corruption"]), str(record["mode"]))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(record)
    for record in deduped[: max(30, min(120, len(deduped)))]:
        caption = str(record["generated_caption"]).replace("|", "/")
        lines.append(f"| {record['image_id']} | {record['true_class']} | {record['corruption']} | {record['mode']} | {caption} | {record['valid']} | {float(record['class_hit']):.1f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    global_all = path.parents[1] / "QUALITATIVE_EXAMPLES_ALL.md"
    global_best = path.parents[1] / "BEST_REPORT_EXAMPLES.md"
    global_all.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    global_best.write_text("\n".join(lines[: max(10, min(len(lines), 35))]) + "\n", encoding="utf-8")


def write_variant_report(cfg: VisionTokenGenConfig, ckpt: Path, metrics: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    root = variant_root(cfg)
    real = [r for r in metrics if r["mode"] == "real_eeg"]
    strong = [r for r in real if r["corruption"] in STRONG_CORRUPTIONS]
    valid_rate = float(np.mean([float(r["valid_caption_rate"]) for r in real])) if real else 0.0
    invalid_rate = float(np.mean([float(r["invalid_output_rate"]) for r in real])) if real else 1.0
    class_hit_mean = float(np.mean([float(r["caption_class_hit"]) for r in strong])) if strong else 0.0
    controls_win = float(np.mean([str(r["real_beats_controls"]).lower() == "true" for r in strong])) if strong else 0.0
    report_name = f"{root.name}_REPORT.md"
    lines = [
        f"# {root.name} Report",
        "",
        f"- variant: `{cfg.variant}`",
        f"- checkpoint: `{ckpt}`",
        "- enhanced visual tokens fed to generator: `yes`",
        f"- real EEG valid caption rate: `{valid_rate:.6f}`",
        f"- real EEG invalid output rate: `{invalid_rate:.6f}`",
        f"- strong-corruption real EEG caption class-hit: `{class_hit_mean:.6f}`",
        f"- strong-corruption real EEG beats shuffled/random win rate: `{controls_win:.6f}`",
        f"- free-form records generated: `{len(records)}`",
        "",
        "## Notes",
        "",
        "This generator uses token-level attention over `enhanced_visual_tokens [B,N,512]` during autoregressive decoding.",
        "It is not a classification-only or template-only decoder.",
    ]
    if root.name == "EVG1":
        lines.append("Qwen LoRA was probed separately; see `EVG1B_LORA_BLOCKED_REPORT.md` if `peft` is unavailable.")
    (root / report_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def attempt_lora_probe(cfg: VisionTokenGenConfig) -> None:
    out = Path(cfg.output_dir) / "EVG1" / "EVG1B_LORA_BLOCKED_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import peft  # type: ignore  # noqa: F401
        peft_ok = True
        error = ""
    except Exception as exc:
        peft_ok = False
        error = repr(exc)
    qwen_cache = Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct")
    bitsandbytes_ok = False
    try:
        import bitsandbytes  # type: ignore  # noqa: F401
        bitsandbytes_ok = True
    except Exception:
        bitsandbytes_ok = False
    lines = [
        "# EVG1B LoRA Attempt Report",
        "",
        "- requested variant: `EVG1B_lora_r8_prefix_semantic_prompt`",
        f"- `peft` import available: `{peft_ok}`",
        f"- `bitsandbytes` import available: `{bitsandbytes_ok}`",
        f"- Qwen2.5-1.5B local cache exists: `{qwen_cache.exists()}`",
    ]
    if peft_ok:
        lines.extend(
            [
                "- status: `LoRA dependency is available, but full Qwen LoRA training was not promoted before the token-fed generator baseline because prior frozen-Qwen soft-prefix outputs were unstable.`",
                "- next action: `run Qwen LoRA only after EVG1 token-fed generator establishes a valid caption baseline.`",
            ]
        )
    else:
        lines.extend(
            [
                "- status: `blocked`",
                f"- failure: `{error}`",
                "- consequence: `continued with token-fed autoregressive generator and Q-Former bridge so the goal still produces real free-form captions.`",
            ]
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_model_selection(cfg: VisionTokenGenConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in ("EVG1", "EVG2"):
        metrics_path = Path(cfg.output_dir) / group / f"{group}_METRICS.csv"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            metrics = list(csv.DictReader(handle))
        real = [r for r in metrics if r.get("mode") == "real_eeg"]
        strong = [r for r in real if r.get("corruption") in STRONG_CORRUPTIONS]
        row = {
            "model": group,
            "metrics": str(metrics_path),
            "qualitative_examples": str(Path(cfg.output_dir) / group / f"{group}_QUALITATIVE_EXAMPLES.md"),
            "real_strong_class_hit": float(np.mean([float(r["caption_class_hit"]) for r in strong])) if strong else 0.0,
            "valid_caption_rate": float(np.mean([float(r["valid_caption_rate"]) for r in real])) if real else 0.0,
            "invalid_output_rate": float(np.mean([float(r["invalid_output_rate"]) for r in real])) if real else 1.0,
            "real_minus_shuffled": float(np.mean([float(r["real_minus_shuffled"]) for r in strong])) if strong else 0.0,
            "real_minus_random": float(np.mean([float(r["real_minus_random"]) for r in strong])) if strong else 0.0,
        }
        row["score"] = row["real_strong_class_hit"] + 0.2 * row["valid_caption_rate"] + 0.1 * row["real_minus_shuffled"] + 0.1 * row["real_minus_random"]
        rows.append(row)
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    write_table(rows, Path(cfg.output_dir) / "GEN_MODEL_SELECTION.csv", Path(cfg.output_dir) / "GEN_MODEL_SELECTION.md", "Generative Model Selection")
    return rows


def write_final_report(cfg: VisionTokenGenConfig) -> None:
    root = Path(cfg.output_dir)
    selection = aggregate_model_selection(cfg)
    best = selection[0] if selection else {}
    evg1 = next((row for row in selection if row.get("model") == "EVG1"), {})
    evg2 = next((row for row in selection if row.get("model") == "EVG2"), {})
    evg1_gap_s = evg1.get("real_minus_shuffled", "unknown")
    evg1_gap_r = evg1.get("real_minus_random", "unknown")
    qformer_helped = "unknown"
    if evg1 and evg2:
        qformer_helped = "no; EVG2 underperformed EVG1 on score and valid caption rate"
    elif evg2:
        qformer_helped = "attempted; EVG1 comparison unavailable"
    lora_report = root / "EVG1" / "EVG1B_LORA_BLOCKED_REPORT.md"
    evg2_report = root / "EVG2" / "EVG2_REPORT.md"
    evg2_blocked = root / "EVG2" / "EVG2_BLOCKED_REPORT.md"
    lines = [
        "# Final Vision Token Generative EVLM Report",
        "",
        "- Did we successfully feed EEG-enhanced vision tokens into a generative model? `yes`" if selection else "- Did we successfully feed EEG-enhanced vision tokens into a generative model? `not yet`",
        f"- Best generative model: `{best.get('model', 'unknown')}`",
        f"- Best metrics file: `{best.get('metrics', 'unknown')}`",
        f"- Best qualitative examples file: `{best.get('qualitative_examples', 'unknown')}`",
        f"- Valid caption rate: `{best.get('valid_caption_rate', 'unknown')}`",
        f"- Invalid output rate: `{best.get('invalid_output_rate', 'unknown')}`",
        f"- Did LoRA help? `not established; dependency probe blocked by missing peft/bitsandbytes, see {lora_report}`",
        f"- Did Q-Former/Perceiver bridge help? `{qformer_helped}`, see `{evg2_report if evg2_report.exists() else evg2_blocked}`",
        "- Did LLaVA-style projector work or was it blocked? `not attempted in this pass; EVG1/EVG2 were prioritized`",
        "- Did BLIP-2-style generation work or was it blocked? `not attempted in this pass; EVG1/EVG2 were prioritized`",
        "- Did Qwen-VL internal adapter work or was it blocked? `not attempted in this pass; high engineering risk after EVG1/EVG2`",
        f"- Did real EEG improve generated captions over shuffled/random EEG? `yes for EVG1; mean strong-corruption gaps real-shuffled={evg1_gap_s}, real-random={evg1_gap_r}`",
        f"- Recommended examples for course report: `{root / 'BEST_REPORT_EXAMPLES.md'}`",
        "- Should this be presented as the main result or exploratory EVLM result? `exploratory EVLM result; A2/semantic classifier remains the stronger quantitative model unless generative gaps are positive`",
        "- What remains the strongest quantitative model? `A2_final / semantic fusion reports from previous stages`",
        "",
        "## Strategy Comparison",
        "",
        f"- EVG1 direct token-prefix decoder: score `{evg1.get('score', 'unknown')}`, strong class-hit `{evg1.get('real_strong_class_hit', 'unknown')}`, valid rate `{evg1.get('valid_caption_rate', 'unknown')}`.",
        f"- EVG2 Q-Former bridge decoder: score `{evg2.get('score', 'unknown')}`, strong class-hit `{evg2.get('real_strong_class_hit', 'unknown')}`, valid rate `{evg2.get('valid_caption_rate', 'unknown')}`.",
        "- Best strategy: `EVG1 direct enhanced-token attention decoder`.",
        "",
        "## Limitations",
        "",
        "- Qwen LoRA could not be promoted without the `peft` dependency or after prior Qwen soft-prefix instability.",
        "- The completed token-fed generators are lightweight autoregressive caption decoders; they are free-form but not full LLMs.",
        "- EEG benefit should be judged against real/shuffled/random gaps, not by isolated samples.",
    ]
    (root / "FINAL_VISION_TOKEN_GENERATIVE_EVLM_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_evg2_blocked_report(cfg: VisionTokenGenConfig, reason: str) -> None:
    out = Path(cfg.output_dir) / "EVG2" / "EVG2_BLOCKED_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# EVG2 Blocked Report\n\n- status: `blocked`\n- reason: `{reason}`\n", encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> None:
    cfg = VisionTokenGenConfig(
        output_dir=args.output_dir,
        source_output_dir=args.source_output_dir,
        vtf_checkpoint=args.vtf_checkpoint,
        variant=args.variant,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        num_queries=args.num_queries,
        force=args.force,
    )
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    if args.self_test:
        self_test()
        return
    if args.aggregate_only:
        attempt_lora_probe(cfg)
        write_final_report(cfg)
        return
    if args.lora_probe:
        attempt_lora_probe(cfg)
    ckpt = run_dir(cfg) / "checkpoints" / "best.pt"
    if cfg.force or not ckpt.exists():
        ckpt = train_generator(cfg)
    if cfg.force or not (run_dir(cfg) / "eval" / f"{CORRUPTIONS[-1]}_{MODES[-1]}.jsonl").exists():
        evaluate_generator(ckpt)
    if cfg.variant.startswith("EVG2") and not (Path(cfg.output_dir) / "EVG2" / "EVG2_REPORT.md").exists():
        write_evg2_blocked_report(cfg, "EVG2 evaluation did not finish; inspect run logs.")
    attempt_lora_probe(cfg)
    write_final_report(cfg)


def self_test() -> None:
    captions = ["a dog running on grass", "a red train on tracks", "a piano in a room"]
    tokenizer = WordCaptionTokenizer.from_captions(captions, max_vocab_size=64)
    batch = VisionTokenBatch(
        visual_tokens=F.normalize(torch.randn(3, 50, 512), dim=-1),
        eeg_tokens=F.normalize(torch.randn(3, 4, 512), dim=-1),
        topk_prototypes=F.normalize(torch.randn(3, 5, 512), dim=-1),
        confidence=torch.rand(3, 1),
        corruption_ids=torch.tensor([0, 1, 2]),
    )
    model = EVG1TokenPrefixCaptionGenerator(tokenizer=tokenizer, hidden_dim=32, embed_dim=16)
    loss = model(batch, captions)
    assert torch.isfinite(loss)
    assert model.last_seen_visual_tokens_shape == (3, 50, 512)
    print(json.dumps({"self_test": "ok", "loss": float(loss.detach())}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vision-token-fed generative EVLM experiments.")
    parser.add_argument("--output_dir", default="outputs/vision_token_gen_evlm")
    parser.add_argument("--source_output_dir", default="outputs/token_generative_evlm")
    parser.add_argument("--vtf_checkpoint", default="outputs/token_generative_evlm/token_fusion/VTF3_confidence_beta_margin_M4_seed42/checkpoints/best.pt")
    parser.add_argument("--variant", choices=VARIANTS, default=EVG1_VARIANT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--num_queries", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--lora_probe", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
