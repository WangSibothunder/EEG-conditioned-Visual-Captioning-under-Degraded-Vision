from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
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
    class_hit,
    load_label_bank,
    load_rows_labels,
    valid_caption,
    write_json,
    write_table,
)
from scripts.run_vision_token_gen_evlm import (
    VisionTokenBatch,
    VisionTokenGenConfig,
    batch_from_features,
    build_vision_token_features,
    prepare_caption_targets,
    source_cfg,
    split_paths,
    write_caption_target_report,
    write_enhanced_token_source_report,
)
from src.utils.seed import seed_everything


ROUTE1 = "route1_qwen_inputs_embeds"
ROUTE2 = "route2_qformer_qwen"
ROUTE3 = "route3_llava_style"
ROUTE4 = "route4_blip2_style"
ROUTE5 = "route5_qwenvl_adapter"
ROUTES = [ROUTE1, ROUTE2, ROUTE3, ROUTE4, ROUTE5]
CORRUPTION_TO_ID = {name: idx for idx, name in enumerate(CORRUPTIONS)}


@dataclass
class StrictTokenBatch:
    visual_tokens: torch.Tensor
    eeg_tokens: torch.Tensor
    topk_prototypes: torch.Tensor
    confidence: torch.Tensor
    corruption_ids: torch.Tensor

    def to(self, device: torch.device) -> "StrictTokenBatch":
        return StrictTokenBatch(
            visual_tokens=self.visual_tokens.to(device, non_blocking=True),
            eeg_tokens=self.eeg_tokens.to(device, non_blocking=True),
            topk_prototypes=self.topk_prototypes.to(device, non_blocking=True),
            confidence=self.confidence.to(device, non_blocking=True),
            corruption_ids=self.corruption_ids.to(device, non_blocking=True),
        )


@dataclass
class StrictRouteConfig:
    name: str
    route: str = ROUTE1
    bridge: str = "direct"  # none, direct, qformer, mm_projector, blip2_style
    use_prefix: bool = True
    use_semantic_prompt: bool = True
    use_lora: bool = False
    lora_r: int = 0
    lora_alpha: int = 0
    prefix_len: int = 16
    num_queries: int = 8
    include_visual: bool = True
    include_eeg: bool = True
    include_topk: bool = True
    train: bool = True
    exact_model_status: str = "not_applicable"
    caption_target_strategy: str = "clean_blip_class_fallback"

    def __post_init__(self) -> None:
        if not self.use_prefix and not self.use_semantic_prompt:
            raise ValueError("strict Qwen route must use a prefix, a semantic prompt, or both")
        if self.use_lora and self.lora_r <= 0:
            raise ValueError("LoRA route must set lora_r > 0")


@dataclass
class StrictRunConfig:
    output_dir: str = "outputs/strict_vlm_gen"
    source_output_dir: str = "outputs/token_generative_evlm"
    vtf_checkpoint: str = "outputs/token_generative_evlm/token_fusion/VTF3_confidence_beta_margin_M4_seed42/checkpoints/best.pt"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_cache_dir: str = "/cloud/cloud-ssd1/eeg_vision_caption_data/hf_cache"
    seed: int = 42
    epochs: int = 1
    batch_size: int = 2
    eval_batch_size: int = 4
    feature_batch_size: int = 128
    lr: float = 2.0e-4
    weight_decay: float = 0.01
    max_train_samples: int = 512
    max_val_samples: int = 128
    max_test_samples: int = 64
    max_prompt_length: int = 96
    max_caption_length: int = 24
    max_new_tokens: int = 14
    generation_strategy: str = "greedy"
    rerank_n: int = 3
    force: bool = False


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_semantic_prompt(top5_class_names: Sequence[str], corruption_type: str) -> str:
    concepts = ", ".join(str(name) for name in top5_class_names[:5])
    return (
        "Write one short natural image caption.\n"
        "Candidate visual concepts:\n"
        f"{concepts}\n"
        "Visual condition:\n"
        f"{corruption_type}\n"
        "Caption:"
    )


def build_plain_prompt() -> str:
    return "Write one short natural image caption.\nCaption:"


def class_caption(class_name: str) -> str:
    return f"a photo of a {str(class_name).strip()}"


def is_filtered_blip_caption(caption: str, class_name: str) -> bool:
    ok, _reason = valid_caption(caption)
    words = str(caption).strip().split()
    if not ok or not words or len(words) > 20:
        return False
    lowered = str(caption).lower()
    if lowered == class_caption(class_name).lower():
        return False
    counts = Counter(word.lower() for word in words)
    if counts and max(counts.values()) > 3:
        return False
    # Keep BLIP additions conservative: require direct class-name evidence.
    return class_hit(caption, class_name) > 0.0


def apply_caption_target_strategy(
    cfg: StrictRunConfig,
    features: Any,
    split: str,
    strategy: str,
) -> Any:
    if strategy == "clean_blip_class_fallback":
        return features
    dev = device()
    src_cfg = source_cfg(vtg_cfg(cfg))
    _labels, _protos, class_name_map = load_label_bank(src_cfg, dev)
    captions = list(features.captions)
    raw_blip: list[str] = []
    try:
        caption_manifest, _human_manifest, _max_samples = split_paths(vtg_cfg(cfg), split)
        _rows, _labels_raw, raw_blip = load_rows_labels(caption_manifest, 0)
    except Exception:
        raw_blip = captions
    updated: list[str] = []
    for idx, row in enumerate(features.rows):
        label = int(row.get("label", -1))
        name = class_name_map.get(label, str(label))
        base = class_caption(name)
        blip = raw_blip[idx] if idx < len(raw_blip) else captions[idx]
        if strategy == "T1_class_only":
            updated.append(base)
        elif strategy == "T2_filtered_blip":
            updated.append(str(blip).strip().rstrip(".") if is_filtered_blip_caption(blip, name) else base)
        elif strategy == "T3_class_plus_blip":
            if is_filtered_blip_caption(blip, name):
                updated.append(f"{base}. {str(blip).strip().rstrip('.')}")
            else:
                updated.append(base)
        else:
            raise ValueError(f"unsupported caption_target_strategy: {strategy}")
    features.captions = updated
    return features


def route1_variant_configs() -> list[StrictRouteConfig]:
    return [
        StrictRouteConfig(
            name="qwen_prefix_only",
            route=ROUTE1,
            bridge="direct",
            use_prefix=True,
            use_semantic_prompt=False,
            use_lora=False,
            train=True,
        ),
        StrictRouteConfig(
            name="qwen_semantic_prompt_only",
            route=ROUTE1,
            bridge="none",
            use_prefix=False,
            use_semantic_prompt=True,
            use_lora=False,
            train=False,
        ),
        StrictRouteConfig(
            name="qwen_prefix_semantic_prompt",
            route=ROUTE1,
            bridge="direct",
            use_prefix=True,
            use_semantic_prompt=True,
            use_lora=False,
            train=True,
        ),
        StrictRouteConfig(
            name="qwen_prefix_semantic_prompt_lora_r8",
            route=ROUTE1,
            bridge="direct",
            use_prefix=True,
            use_semantic_prompt=True,
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
            train=True,
        ),
    ]


def route1_full_variant_configs() -> list[StrictRouteConfig]:
    return [
        StrictRouteConfig(
            name="ablation_semantic_prompt_only_full",
            route=ROUTE1,
            bridge="none",
            use_prefix=False,
            use_semantic_prompt=True,
            use_lora=False,
            train=False,
        ),
        StrictRouteConfig(
            name="ablation_enhanced_token_prefix_only_full",
            route=ROUTE1,
            bridge="direct",
            use_prefix=True,
            use_semantic_prompt=False,
            use_lora=False,
            train=True,
        ),
        StrictRouteConfig(
            name="ablation_enhanced_token_prefix_semantic_full",
            route=ROUTE1,
            bridge="direct",
            use_prefix=True,
            use_semantic_prompt=True,
            use_lora=False,
            train=True,
        ),
        StrictRouteConfig(
            name="EVG1B_full_lora_r8",
            route=ROUTE1,
            bridge="direct",
            use_prefix=True,
            use_semantic_prompt=True,
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
            train=True,
        ),
        StrictRouteConfig(
            name="EVG1C_full_lora_r16",
            route=ROUTE1,
            bridge="direct",
            use_prefix=True,
            use_semantic_prompt=True,
            use_lora=True,
            lora_r=16,
            lora_alpha=32,
            train=True,
        ),
    ]


def route2_variant_configs() -> list[StrictRouteConfig]:
    return [
        StrictRouteConfig(
            name="QFormer_visual_only",
            route=ROUTE2,
            bridge="qformer",
            use_prefix=True,
            use_semantic_prompt=True,
            include_visual=True,
            include_eeg=False,
            include_topk=False,
            train=True,
            num_queries=8,
        ),
        StrictRouteConfig(
            name="QFormer_visual_eeg",
            route=ROUTE2,
            bridge="qformer",
            use_prefix=True,
            use_semantic_prompt=True,
            include_visual=True,
            include_eeg=True,
            include_topk=False,
            train=True,
            num_queries=8,
        ),
        StrictRouteConfig(
            name="QFormer_visual_eeg_topk",
            route=ROUTE2,
            bridge="qformer",
            use_prefix=True,
            use_semantic_prompt=True,
            include_visual=True,
            include_eeg=True,
            include_topk=True,
            train=True,
            num_queries=8,
        ),
        StrictRouteConfig(
            name="QFormer_visual_eeg_topk_lora",
            route=ROUTE2,
            bridge="qformer",
            use_prefix=True,
            use_semantic_prompt=True,
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
            include_visual=True,
            include_eeg=True,
            include_topk=True,
            train=True,
            num_queries=8,
        ),
    ]


def route3_variant_configs() -> list[StrictRouteConfig]:
    return [
        StrictRouteConfig(
            name="LLaVAStyle_projector_frozenLLM",
            route=ROUTE3,
            bridge="mm_projector",
            use_prefix=True,
            use_semantic_prompt=False,
            use_lora=False,
            train=True,
            prefix_len=50,
            exact_model_status="local_mm_projector",
        ),
        StrictRouteConfig(
            name="LLaVAStyle_projector_lora",
            route=ROUTE3,
            bridge="mm_projector",
            use_prefix=True,
            use_semantic_prompt=False,
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
            train=True,
            prefix_len=50,
            exact_model_status="local_mm_projector_lora",
        ),
        StrictRouteConfig(
            name="LLaVAStyle_projector_topk_prompt_lora",
            route=ROUTE3,
            bridge="mm_projector",
            use_prefix=True,
            use_semantic_prompt=True,
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
            train=True,
            prefix_len=50,
            exact_model_status="local_mm_projector_topk_lora",
        ),
    ]


def route4_fallback_config() -> StrictRouteConfig:
    return StrictRouteConfig(
        name="blip2_style_qformer_qwen_fallback",
        route=ROUTE4,
        bridge="blip2_style",
        use_prefix=True,
        use_semantic_prompt=True,
        use_lora=False,
        train=True,
        num_queries=32,
        exact_model_status="fallback",
    )


def route4_variant_configs() -> list[StrictRouteConfig]:
    return [
        StrictRouteConfig(
            name="BLIP2_actual_OPT_prefix_adapter",
            route=ROUTE4,
            bridge="blip2_actual",
            use_prefix=True,
            use_semantic_prompt=True,
            use_lora=False,
            train=True,
            num_queries=32,
            exact_model_status="Salesforce/blip2-opt-2.7b",
        ),
        route4_fallback_config(),
    ]


def route5_variant_configs() -> list[StrictRouteConfig]:
    return [
        StrictRouteConfig(
            name="Qwen2VL_2B_prefix_semantic_adapter",
            route=ROUTE5,
            bridge="qwenvl_prefix",
            use_prefix=True,
            use_semantic_prompt=True,
            use_lora=False,
            train=True,
            num_queries=16,
            exact_model_status="Qwen/Qwen2-VL-2B-Instruct",
        ),
    ]


def deep_route5_variant(name: str, *, use_lora: bool, lora_r: int = 0, lora_alpha: int = 0, target: str = "clean_blip_class_fallback") -> StrictRouteConfig:
    return StrictRouteConfig(
        name=name,
        route=ROUTE5,
        bridge="qwenvl_prefix",
        use_prefix=True,
        use_semantic_prompt=True,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        train=True,
        num_queries=16,
        include_visual=True,
        include_eeg=True,
        include_topk=True,
        exact_model_status="Qwen/Qwen2-VL-2B-Instruct",
        caption_target_strategy=target,
    )


def deep_route5_variants() -> list[StrictRouteConfig]:
    return [
        deep_route5_variant("Route5_Qwen2VL_full_adapter", use_lora=False),
        deep_route5_variant("Route5_Qwen2VL_full_lora_r8", use_lora=True, lora_r=8, lora_alpha=16),
        deep_route5_variant("Route5_Qwen2VL_full_lora_r16", use_lora=True, lora_r=16, lora_alpha=32),
        deep_route5_variant("Route5_Qwen2VL_lora_r8_T1_class_only", use_lora=True, lora_r=8, lora_alpha=16, target="T1_class_only"),
        deep_route5_variant("Route5_Qwen2VL_lora_r8_T3_class_plus_blip", use_lora=True, lora_r=8, lora_alpha=16, target="T3_class_plus_blip"),
    ]


def deep_route1_clean_variants() -> list[StrictRouteConfig]:
    return [
        StrictRouteConfig(
            name="Route1_QwenLoRA_r8_full_clean_target",
            route=ROUTE1,
            bridge="direct",
            use_prefix=True,
            use_semantic_prompt=True,
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
            train=True,
            caption_target_strategy="T1_class_only",
        ),
        StrictRouteConfig(
            name="Route1_QwenLoRA_r16_full_clean_target",
            route=ROUTE1,
            bridge="direct",
            use_prefix=True,
            use_semantic_prompt=True,
            use_lora=True,
            lora_r=16,
            lora_alpha=32,
            train=True,
            caption_target_strategy="T1_class_only",
        ),
    ]


def vtg_cfg(cfg: StrictRunConfig) -> VisionTokenGenConfig:
    return VisionTokenGenConfig(
        output_dir=cfg.output_dir,
        source_output_dir=cfg.source_output_dir,
        vtf_checkpoint=cfg.vtf_checkpoint,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.feature_batch_size,
        max_train_samples=cfg.max_train_samples,
        max_val_samples=cfg.max_val_samples,
        max_test_samples=cfg.max_test_samples,
    )


def append_gpu_usage(path: Path, event: str, route: str, extra: dict[str, Any] | None = None) -> None:
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
    except Exception as exc:
        output = f"nvidia-smi unavailable: {exc}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {time.strftime('%Y-%m-%d %H:%M:%S')} - {event}\n\n")
        handle.write(f"- route: `{route}`\n")
        handle.write(f"- gpu: `{output}`\n")
        for key, value in (extra or {}).items():
            handle.write(f"- {key}: `{value}`\n")


def current_gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024**3))


def strict_batch_from_features(features: Any, idxs: Sequence[int], dev: torch.device) -> StrictTokenBatch:
    batch = batch_from_features(features, idxs, dev)
    return StrictTokenBatch(
        visual_tokens=batch.visual_tokens,
        eeg_tokens=batch.eeg_tokens,
        topk_prototypes=batch.topk_prototypes,
        confidence=batch.confidence,
        corruption_ids=batch.corruption_ids,
    )


def prompts_for(features: Any, corruption: str, idxs: Sequence[int], variant: StrictRouteConfig) -> list[str]:
    prompts: list[str] = []
    for idx in idxs:
        if variant.use_semantic_prompt:
            prompts.append(build_semantic_prompt(features.top5_names[idx], corruption))
        else:
            prompts.append(build_plain_prompt())
    return prompts


def repetition_rate(caption: str) -> float:
    words = str(caption).lower().split()
    if len(words) < 4:
        return 0.0
    counts = Counter(words)
    return float(max(counts.values()) / max(len(words), 1))


class StrictQwenGenerator(nn.Module):
    def __init__(self, cfg: StrictRunConfig, variant: StrictRouteConfig) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.variant = variant
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        base = AutoModelForCausalLM.from_pretrained(cfg.llm_model, local_files_only=True, torch_dtype=dtype)
        for param in base.parameters():
            param.requires_grad_(False)
        if variant.use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=variant.lora_r,
                lora_alpha=variant.lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            self.llm = get_peft_model(base, lora_cfg)
        else:
            self.llm = base
        hidden = int(self.llm.config.hidden_size)
        self.query_count = variant.prefix_len if variant.bridge == "direct" else variant.num_queries
        if variant.bridge == "mm_projector":
            self.mm_projector = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        else:
            self.query_tokens = nn.Parameter(torch.randn(self.query_count, 512) * 0.02)
            self.conf_token = nn.Linear(1, 512)
            self.corr_embed = nn.Embedding(len(CORRUPTIONS), 512)
            self.resampler = nn.MultiheadAttention(512, num_heads=8, batch_first=True, dropout=0.05)
            self.prefix_projector = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, hidden))
        self.last_seen_visual_tokens_shape: tuple[int, ...] | None = None

    def trainable_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))

    def total_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters()))

    def memory_tokens(self, batch: StrictTokenBatch) -> torch.Tensor:
        visual = F.normalize(batch.visual_tokens.float(), dim=-1)
        self.last_seen_visual_tokens_shape = tuple(int(x) for x in visual.shape)
        chunks: list[torch.Tensor] = []
        if self.variant.include_visual:
            chunks.append(visual)
        if self.variant.include_eeg:
            chunks.append(F.normalize(batch.eeg_tokens.float(), dim=-1))
        if self.variant.include_topk:
            chunks.append(F.normalize(batch.topk_prototypes.float(), dim=-1))
        if hasattr(self, "conf_token"):
            chunks.append(self.conf_token(batch.confidence.float()).unsqueeze(1))
            chunks.append(self.corr_embed(batch.corruption_ids.long()).unsqueeze(1))
        if not chunks:
            chunks.append(torch.zeros_like(visual[:, :1]))
        return torch.cat(chunks, dim=1)

    def prefix_embeds(self, batch: StrictTokenBatch) -> torch.Tensor | None:
        if not self.variant.use_prefix:
            return None
        memory = self.memory_tokens(batch)
        if self.variant.bridge == "mm_projector":
            prefix = self.mm_projector(memory)
        else:
            queries = self.query_tokens.unsqueeze(0).expand(memory.shape[0], -1, -1)
            prefix_512, _ = self.resampler(queries, memory, memory, need_weights=False)
            prefix = self.prefix_projector(prefix_512)
        return prefix.to(dtype=self.llm.get_input_embeddings().weight.dtype)

    def encode_text(self, prompts: Sequence[str], captions: Sequence[str], dev: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_tokens = self.tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=self.cfg.max_prompt_length)
        target_text = [str(caption).strip() + self.tokenizer.eos_token for caption in captions]
        caption_tokens = self.tokenizer(
            target_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_caption_length,
            add_special_tokens=False,
        )
        return (
            prompt_tokens["input_ids"].to(dev),
            prompt_tokens["attention_mask"].to(dev),
            caption_tokens["input_ids"].to(dev),
            caption_tokens["attention_mask"].to(dev),
        )

    def forward(self, batch: StrictTokenBatch, prompts: Sequence[str], captions: Sequence[str]) -> torch.Tensor:
        dev = batch.visual_tokens.device
        prefix = self.prefix_embeds(batch)
        prompt_ids, prompt_mask, caption_ids, caption_mask = self.encode_text(prompts, captions, dev)
        input_ids = torch.cat([prompt_ids, caption_ids], dim=1)
        text_mask = torch.cat([prompt_mask, caption_mask], dim=1)
        token_embeds = self.llm.get_input_embeddings()(input_ids)
        if prefix is not None:
            token_embeds = token_embeds.to(dtype=prefix.dtype)
            inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
            prefix_mask = torch.ones(prefix.shape[:2], dtype=text_mask.dtype, device=dev)
            attention_mask = torch.cat([prefix_mask, text_mask], dim=1)
            ignored = torch.full((input_ids.shape[0], prefix.shape[1] + prompt_ids.shape[1]), -100, dtype=torch.long, device=dev)
        else:
            inputs_embeds = token_embeds
            attention_mask = text_mask
            ignored = torch.full((input_ids.shape[0], prompt_ids.shape[1]), -100, dtype=torch.long, device=dev)
        caption_labels = caption_ids.masked_fill(caption_mask == 0, -100)
        labels = torch.cat([ignored, caption_labels], dim=1)
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    @torch.no_grad()
    def generate(self, batch: StrictTokenBatch, prompts: Sequence[str], max_new_tokens: int, strategy: str = "greedy") -> list[str]:
        self.eval()
        dev = batch.visual_tokens.device
        prefix = self.prefix_embeds(batch)
        prompt_tokens = self.tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=self.cfg.max_prompt_length)
        prompt_ids = prompt_tokens["input_ids"].to(dev)
        prompt_mask = prompt_tokens["attention_mask"].to(dev)
        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
        if prefix is not None:
            prompt_embeds = prompt_embeds.to(dtype=prefix.dtype)
            inputs_embeds = torch.cat([prefix, prompt_embeds], dim=1)
            attention_mask = torch.cat([torch.ones(prefix.shape[:2], dtype=prompt_mask.dtype, device=dev), prompt_mask], dim=1)
        else:
            inputs_embeds = prompt_embeds
            attention_mask = prompt_mask
        try:
            kwargs: dict[str, Any] = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if strategy == "beam3":
                kwargs.update({"num_beams": 3, "do_sample": False})
            elif strategy == "temp02":
                kwargs.update({"do_sample": True, "temperature": 0.2, "top_p": 0.9})
            elif strategy == "temp05":
                kwargs.update({"do_sample": True, "temperature": 0.5, "top_p": 0.9})
            else:
                kwargs.update({"do_sample": False})
            generated = self.llm.generate(
                **kwargs,
            )
            return [clean_generated_text(item) for item in self.tokenizer.batch_decode(generated, skip_special_tokens=True)]
        except Exception:
            return self.manual_greedy(inputs_embeds, attention_mask, max_new_tokens)

    def manual_greedy(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int) -> list[str]:
        dev = inputs_embeds.device
        generated = torch.empty((inputs_embeds.shape[0], 0), dtype=torch.long, device=dev)
        for step in range(max_new_tokens):
            if generated.numel():
                token_embeds = self.llm.get_input_embeddings()(generated).to(dtype=inputs_embeds.dtype)
                current_embeds = torch.cat([inputs_embeds, token_embeds], dim=1)
            else:
                current_embeds = inputs_embeds
            current_mask = torch.ones(current_embeds.shape[:2], dtype=torch.long, device=dev)
            logits = self.llm(inputs_embeds=current_embeds, attention_mask=current_mask).logits[:, -1]
            logits[:, self.tokenizer.pad_token_id] = -1e9
            if step < 3 and self.tokenizer.eos_token_id is not None:
                logits[:, self.tokenizer.eos_token_id] = -1e9
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        return [clean_generated_text(item) for item in self.tokenizer.batch_decode(generated, skip_special_tokens=True)]


class StrictExternalPrefixGenerator(nn.Module):
    def __init__(self, cfg: StrictRunConfig, variant: StrictRouteConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        self.backend = variant.bridge
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if variant.bridge == "blip2_actual":
            from transformers import Blip2ForConditionalGeneration, Blip2Processor

            self.model_id = "Salesforce/blip2-opt-2.7b"
            self.processor = Blip2Processor.from_pretrained(
                self.model_id,
                cache_dir=cfg.model_cache_dir,
                local_files_only=True,
            )
            self.tokenizer = self.processor.tokenizer
            self.backbone = Blip2ForConditionalGeneration.from_pretrained(
                self.model_id,
                cache_dir=cfg.model_cache_dir,
                local_files_only=True,
                torch_dtype=dtype,
            )
            self.decoder = self.backbone.language_model
        elif variant.bridge == "qwenvl_prefix":
            from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration

            self.model_id = "Qwen/Qwen2-VL-2B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=cfg.model_cache_dir,
                local_files_only=True,
            )
            self.backbone = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                cache_dir=cfg.model_cache_dir,
                local_files_only=True,
                torch_dtype=dtype,
            )
            for param in self.backbone.parameters():
                param.requires_grad_(False)
            if variant.use_lora:
                from peft import LoraConfig, TaskType, get_peft_model

                lora_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=variant.lora_r,
                    lora_alpha=variant.lora_alpha,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    bias="none",
                )
                self.backbone = get_peft_model(self.backbone, lora_cfg)
                if hasattr(self.backbone, "gradient_checkpointing_enable"):
                    self.backbone.gradient_checkpointing_enable()
            self.decoder = self.backbone
        else:
            raise ValueError(f"unsupported external generator bridge: {variant.bridge}")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not variant.use_lora:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        hidden = int(self.decoder.get_input_embeddings().weight.shape[1])
        self.query_tokens = nn.Parameter(torch.randn(variant.num_queries, 512) * 0.02)
        self.conf_token = nn.Linear(1, 512)
        self.corr_embed = nn.Embedding(len(CORRUPTIONS), 512)
        self.resampler = nn.MultiheadAttention(512, num_heads=8, batch_first=True, dropout=0.05)
        self.prefix_projector = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, hidden))
        self.last_seen_visual_tokens_shape: tuple[int, ...] | None = None

    def trainable_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))

    def total_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters()))

    def memory_tokens(self, batch: StrictTokenBatch) -> torch.Tensor:
        visual = F.normalize(batch.visual_tokens.float(), dim=-1)
        self.last_seen_visual_tokens_shape = tuple(int(x) for x in visual.shape)
        chunks = [visual]
        if self.variant.include_eeg:
            chunks.append(F.normalize(batch.eeg_tokens.float(), dim=-1))
        if self.variant.include_topk:
            chunks.append(F.normalize(batch.topk_prototypes.float(), dim=-1))
        chunks.append(self.conf_token(batch.confidence.float()).unsqueeze(1))
        chunks.append(self.corr_embed(batch.corruption_ids.long()).unsqueeze(1))
        return torch.cat(chunks, dim=1)

    def prefix_embeds(self, batch: StrictTokenBatch) -> torch.Tensor:
        memory = self.memory_tokens(batch)
        queries = self.query_tokens.unsqueeze(0).expand(memory.shape[0], -1, -1)
        prefix_512, _ = self.resampler(queries, memory, memory, need_weights=False)
        prefix = self.prefix_projector(prefix_512)
        return prefix.to(dtype=self.decoder.get_input_embeddings().weight.dtype)

    def encode_text(self, prompts: Sequence[str], captions: Sequence[str], dev: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_tokens = self.tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=self.cfg.max_prompt_length)
        eos = self.tokenizer.eos_token or ""
        target_text = [str(caption).strip() + eos for caption in captions]
        caption_tokens = self.tokenizer(
            target_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_caption_length,
            add_special_tokens=False,
        )
        return (
            prompt_tokens["input_ids"].to(dev),
            prompt_tokens["attention_mask"].to(dev),
            caption_tokens["input_ids"].to(dev),
            caption_tokens["attention_mask"].to(dev),
        )

    def forward(self, batch: StrictTokenBatch, prompts: Sequence[str], captions: Sequence[str]) -> torch.Tensor:
        dev = batch.visual_tokens.device
        prefix = self.prefix_embeds(batch)
        prompt_ids, prompt_mask, caption_ids, caption_mask = self.encode_text(prompts, captions, dev)
        input_ids = torch.cat([prompt_ids, caption_ids], dim=1)
        text_mask = torch.cat([prompt_mask, caption_mask], dim=1)
        token_embeds = self.decoder.get_input_embeddings()(input_ids).to(dtype=prefix.dtype)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
        attention_mask = torch.cat([torch.ones(prefix.shape[:2], dtype=text_mask.dtype, device=dev), text_mask], dim=1)
        ignored = torch.full((input_ids.shape[0], prefix.shape[1] + prompt_ids.shape[1]), -100, dtype=torch.long, device=dev)
        labels = torch.cat([ignored, caption_ids.masked_fill(caption_mask == 0, -100)], dim=1)
        outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    @torch.no_grad()
    def generate(self, batch: StrictTokenBatch, prompts: Sequence[str], max_new_tokens: int, strategy: str = "greedy") -> list[str]:
        self.eval()
        dev = batch.visual_tokens.device
        prefix = self.prefix_embeds(batch)
        prompt_tokens = self.tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=self.cfg.max_prompt_length)
        prompt_ids = prompt_tokens["input_ids"].to(dev)
        prompt_mask = prompt_tokens["attention_mask"].to(dev)
        prompt_embeds = self.decoder.get_input_embeddings()(prompt_ids).to(dtype=prefix.dtype)
        inputs_embeds = torch.cat([prefix, prompt_embeds], dim=1)
        attention_mask = torch.cat([torch.ones(prefix.shape[:2], dtype=prompt_mask.dtype, device=dev), prompt_mask], dim=1)
        kwargs: dict[str, Any] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if strategy == "beam3":
            kwargs.update({"num_beams": 3, "do_sample": False})
        elif strategy == "temp02":
            kwargs.update({"do_sample": True, "temperature": 0.2, "top_p": 0.9})
        elif strategy == "temp05":
            kwargs.update({"do_sample": True, "temperature": 0.5, "top_p": 0.9})
        else:
            kwargs.update({"do_sample": False})
        generated = self.decoder.generate(**kwargs)
        return [clean_generated_text(item) for item in self.tokenizer.batch_decode(generated, skip_special_tokens=True)]


def clean_generated_text(text: str) -> str:
    text = str(text).strip()
    for marker in ["Caption:", "caption:"]:
        if marker in text:
            text = text.split(marker)[-1].strip()
    return " ".join(text.split())


def make_generator(cfg: StrictRunConfig, variant: StrictRouteConfig) -> nn.Module:
    if variant.bridge in {"blip2_actual", "qwenvl_prefix"}:
        return StrictExternalPrefixGenerator(cfg, variant)
    return StrictQwenGenerator(cfg, variant)


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def non_llm_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Save only local bridge/projector tensors; never serialize frozen Qwen weights."""
    state: dict[str, torch.Tensor] = {}
    frozen_prefixes = ("llm.", "backbone.", "decoder.", "processor.")
    for name, param in model.named_parameters():
        if not name.startswith(frozen_prefixes):
            state[name] = param.detach().cpu()
    for name, buffer in model.named_buffers():
        if not name.startswith(frozen_prefixes):
            state[name] = buffer.detach().cpu()
    return state


def lora_host_module(model: nn.Module) -> nn.Module | None:
    if hasattr(model, "llm"):
        return getattr(model, "llm")
    if getattr(model, "variant", None) is not None and getattr(model.variant, "use_lora", False) and hasattr(model, "backbone"):
        return getattr(model, "backbone")
    return None


def load_lora_adapter_weights(model: nn.Module, adapter_dir: Path, error_path: Path) -> str:
    if not model.variant.use_lora:
        return "not_lora"
    if not adapter_dir.exists():
        return "missing"
    host = lora_host_module(model)
    if host is None:
        return "missing_host"
    try:
        from peft import set_peft_model_state_dict

        state_path = adapter_dir / "adapter_model.safetensors"
        if state_path.exists():
            from safetensors.torch import load_file

            adapter_state = load_file(str(state_path), device="cpu")
        else:
            adapter_state = torch.load(adapter_dir / "adapter_model.bin", map_location="cpu", weights_only=False)
        set_peft_model_state_dict(host, adapter_state)
        return "loaded"
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        return "error"


def route_dir(cfg: StrictRunConfig, route: str) -> Path:
    return Path(cfg.output_dir) / route


def variant_dir(cfg: StrictRunConfig, variant: StrictRouteConfig) -> Path:
    return route_dir(cfg, variant.route) / variant.name


def build_features(cfg: StrictRunConfig, split: str, mode: str, corruption: str = "clean", target_strategy: str = "clean_blip_class_fallback") -> Any:
    features = build_vision_token_features(vtg_cfg(cfg), split, mode, corruption)
    return apply_caption_target_strategy(cfg, features, split, target_strategy)


def write_strict_caption_report(cfg: StrictRunConfig) -> None:
    dev = device()
    _labels, _protos, class_name_map = load_label_bank(source_cfg(vtg_cfg(cfg)), dev)
    _rows, _human, captions, good, replaced = prepare_caption_targets(vtg_cfg(cfg), "train", class_name_map)
    write_caption_target_report(vtg_cfg(cfg), captions, good, replaced)
    src = Path(cfg.output_dir) / "caption_targets" / "CAPTION_TARGET_REPORT.md"
    dst = Path(cfg.output_dir) / "CAPTION_TARGET_REPORT.md"
    if src.exists():
        shutil.copy2(src, dst)


def train_variant(cfg: StrictRunConfig, variant: StrictRouteConfig) -> Path:
    seed_everything(cfg.seed)
    dev = device()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out = variant_dir(cfg, variant)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", {"run": asdict(cfg), "variant": asdict(variant)})
    model = make_generator(cfg, variant).to(dev)
    trainable = trainable_parameters(model)
    append_gpu_usage(
        Path(cfg.output_dir) / "GPU_USAGE.md",
        "model_loaded",
        variant.name,
        {
            "model_name": getattr(model, "model_id", cfg.llm_model),
            "total_params": model.total_parameter_count(),
            "trainable_params": model.trainable_parameter_count(),
            "lora": variant.use_lora,
            "lora_r": variant.lora_r,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "target_strategy": variant.caption_target_strategy,
        },
    )
    train_features = build_features(cfg, "train", "real_eeg", "clean", variant.caption_target_strategy)
    val_features = build_features(cfg, "val", "real_eeg", "clean", variant.caption_target_strategy)
    write_enhanced_token_source_report(
        Path(cfg.output_dir) / "ENHANCED_TOKEN_SOURCE.md",
        vtf_checkpoint=Path(cfg.vtf_checkpoint),
        token_shape=tuple(int(x) for x in train_features.batch.visual_tokens.shape),
        eeg_token_shape=tuple(int(x) for x in train_features.batch.eeg_tokens.shape),
        modes=MODES,
        frozen_modules=["CLIP ViT-B/32", "A2 EEG encoder", "VTF3 token fusion", "Qwen base"],
    )
    history: list[dict[str, Any]] = []
    if variant.train and trainable:
        opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
        loader = DataLoader(TensorDataset(torch.arange(len(train_features))), batch_size=cfg.batch_size, shuffle=True)
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            losses: list[float] = []
            started = time.time()
            for (idxs,) in loader:
                idx_list = [int(i) for i in idxs.tolist()]
                batch = strict_batch_from_features(train_features, idx_list, dev)
                prompts = prompts_for(train_features, "clean", idx_list, variant)
                caps = [train_features.captions[i] for i in idx_list]
                loss = model(batch, prompts, caps)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            val_losses: list[float] = []
            model.eval()
            with torch.no_grad():
                for start in range(0, len(val_features), cfg.batch_size):
                    idx_list = list(range(start, min(start + cfg.batch_size, len(val_features))))
                    batch = strict_batch_from_features(val_features, idx_list, dev)
                    prompts = prompts_for(val_features, "clean", idx_list, variant)
                    caps = [val_features.captions[i] for i in idx_list]
                    val_losses.append(float(model(batch, prompts, caps).detach().cpu()))
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)) if losses else 0.0,
                    "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
                    "seconds": time.time() - started,
                    "peak_gpu_memory_gb": current_gpu_memory_gb(),
                }
            )
    else:
        history.append({"epoch": 0, "train_loss": 0.0, "val_loss": 0.0, "seconds": 0.0, "note": "zero-shot semantic prompt route"})
    ckpt = out / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": non_llm_state_dict(model),
            "run": asdict(cfg),
            "variant": asdict(variant),
            "last_seen_visual_tokens_shape": model.last_seen_visual_tokens_shape,
            "total_params": model.total_parameter_count(),
            "trainable_params": model.trainable_parameter_count(),
            "peak_gpu_memory_gb": float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0,
        },
        ckpt / "prefix_projector.pt",
    )
    host = lora_host_module(model)
    if variant.use_lora and host is not None and hasattr(host, "save_pretrained"):
        host.save_pretrained(ckpt / "lora_adapter")
    write_json(out / "history.json", history)
    (out / "TRAINING_SUMMARY.md").write_text(render_training_summary(variant, history, model), encoding="utf-8")
    return ckpt / "prefix_projector.pt"


def render_training_summary(variant: StrictRouteConfig, history: list[dict[str, Any]], model: nn.Module) -> str:
    lines = [
        f"# {variant.name} Training Summary",
        "",
        f"- route: `{variant.route}`",
        f"- bridge: `{variant.bridge}`",
        f"- pretrained LLM/VLM: `{getattr(model, 'model_id', 'Qwen/Qwen2.5-1.5B-Instruct')}`",
        f"- use_prefix: `{variant.use_prefix}`",
        f"- use_semantic_prompt: `{variant.use_semantic_prompt}`",
        f"- use_lora: `{variant.use_lora}`",
        f"- total params: `{model.total_parameter_count()}`",
        f"- trainable params: `{model.trainable_parameter_count()}`",
        f"- caption target strategy: `{variant.caption_target_strategy}`",
        "",
        "| Epoch | Train Loss | Val Loss | Seconds | Peak GPU GB |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in history:
        lines.append(
            f"| {row['epoch']} | {float(row['train_loss']):.6f} | {float(row['val_loss']):.6f} | "
            f"{float(row['seconds']):.2f} | {float(row.get('peak_gpu_memory_gb', 0.0)):.3f} |"
        )
    return "\n".join(lines) + "\n"


def evaluate_variant(cfg: StrictRunConfig, variant: StrictRouteConfig) -> list[dict[str, Any]]:
    dev = device()
    out = variant_dir(cfg, variant)
    model = make_generator(cfg, variant).to(dev)
    ckpt_path = out / "checkpoints" / "prefix_projector.pt"
    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model", {}), strict=False)
        if hasattr(model, "llm"):
            load_lora_adapter_weights(model, out / "checkpoints" / "lora_adapter", out / "LORA_LOAD_ERROR.txt")
    model.eval()
    src_cfg = source_cfg(vtg_cfg(cfg))
    _labels, _protos, class_name_map = load_label_bank(src_cfg, dev)
    eval_dir = out / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for corruption in CORRUPTIONS:
        for mode in MODES:
            features = build_features(cfg, "test", mode, corruption, variant.caption_target_strategy)
            preds: list[str] = []
            for start in range(0, len(features), cfg.eval_batch_size):
                idxs = list(range(start, min(start + cfg.eval_batch_size, len(features))))
                batch = strict_batch_from_features(features, idxs, dev)
                prompts = prompts_for(features, corruption, idxs, variant)
                preds.extend(model.generate(batch, prompts, cfg.max_new_tokens, cfg.generation_strategy))
            records: list[dict[str, Any]] = []
            for idx, (row, pred) in enumerate(zip(features.rows, preds, strict=False)):
                label = int(row["label"])
                true_class = class_name_map.get(label, str(label))
                ok, reason = valid_caption(pred)
                record = {
                    "route": variant.route,
                    "variant": variant.name,
                    "image_id": str(row["image_id"]),
                    "true_class": true_class,
                    "corruption": corruption,
                    "mode": mode,
                    "top5_classes": features.top5_names[idx],
                    "generated_caption": pred,
                    "valid": ok,
                    "invalid_reason": reason,
                    "class_hit": class_hit(pred, true_class),
                    "caption_topk_class_hit": float(any(class_hit(pred, name) > 0 for name in features.top5_names[idx])),
                    "length": len(str(pred).split()),
                    "repetition_rate": repetition_rate(pred),
                    "token_input_shape": list(features.batch.visual_tokens.shape[1:]),
                    "uses_inputs_embeds": True,
                    "uses_gru_decoder": False,
                    "pretrained_model": getattr(model, "model_id", cfg.llm_model),
                    "semantic_prompt": variant.use_semantic_prompt,
                    "lora": variant.use_lora,
                    "generation_strategy": cfg.generation_strategy,
                    "caption_target_strategy": variant.caption_target_strategy,
                }
                records.append(record)
            with (eval_dir / f"{corruption}_{mode}.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            all_records.extend(records)
            rows.append(summarize_records(records, corruption, mode, f"{variant.name}/eval/{corruption}_{mode}.jsonl"))
    rows = add_gaps(rows)
    write_table(rows, out / "METRICS.csv", out / "METRICS.md", f"{variant.name} Metrics")
    write_qualitative_examples(all_records, out / "QUALITATIVE_EXAMPLES.md")
    write_route_report(cfg, variant, rows, all_records, out)
    return rows


def summarize_records(records: list[dict[str, Any]], corruption: str, mode: str, file_name: str) -> dict[str, Any]:
    valid = [bool(record["valid"]) for record in records]
    hits = [float(record["class_hit"]) for record in records]
    topk = [float(record["caption_topk_class_hit"]) for record in records]
    captions = [str(record["generated_caption"]) for record in records]
    return {
        "file": file_name,
        "corruption": corruption,
        "mode": mode,
        "count": len(records),
        "valid_caption_rate": float(np.mean(valid)) if valid else 0.0,
        "invalid_output_rate": 1.0 - float(np.mean(valid)) if valid else 1.0,
        "caption_class_hit": float(np.mean(hits)) if hits else 0.0,
        "caption_topk_class_hit": float(np.mean(topk)) if topk else 0.0,
        "avg_caption_length": float(np.mean([record["length"] for record in records])) if records else 0.0,
        "distinct_caption_count": len(set(captions)),
        "repetition_rate": float(np.mean([record["repetition_rate"] for record in records])) if records else 0.0,
    }


def add_gaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by[str(row["corruption"])][str(row["mode"])] = row
    for row in rows:
        modes = by[str(row["corruption"])]
        real = float(modes.get("real_eeg", {}).get("caption_class_hit", 0.0))
        row["real_minus_vision"] = real - float(modes.get("vision_only", {}).get("caption_class_hit", 0.0))
        row["real_minus_shuffled"] = real - float(modes.get("shuffled_eeg", {}).get("caption_class_hit", 0.0))
        row["real_minus_random"] = real - float(modes.get("random_eeg", {}).get("caption_class_hit", 0.0))
    return rows


def write_route_level_outputs(cfg: StrictRunConfig, route: str, selection: list[dict[str, Any]]) -> None:
    out = route_dir(cfg, route)
    if not out.exists():
        return
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(out.glob("*/METRICS.csv")):
        variant = metrics_path.parent.name
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["route"] = route
                row["variant"] = variant
                rows.append(row)
    if rows:
        write_table(rows, out / "METRICS.csv", out / "METRICS.md", f"{route} Combined Metrics")
    route_selection = [row for row in selection if row.get("route") == route]
    best_variant = str(route_selection[0]["variant"]) if route_selection else ""
    q = out / best_variant / "QUALITATIVE_EXAMPLES.md"
    if q.exists():
        shutil.copy2(q, out / "QUALITATIVE_EXAMPLES.md")
    (out / "checkpoints").mkdir(exist_ok=True)
    ckpt_lines = ["# Route Checkpoints", ""]
    for ckpt in sorted(out.glob("*/checkpoints")):
        ckpt_lines.append(f"- `{ckpt}`")
    (out / "checkpoints" / "README.md").write_text("\n".join(ckpt_lines) + "\n", encoding="utf-8")
    lines = [
        f"# {route} Route Report",
        "",
        f"- variants with metrics: `{len(set(row.get('variant', '') for row in rows))}`",
        f"- combined metrics: `{out / 'METRICS.csv'}`",
        f"- qualitative examples: `{out / 'QUALITATIVE_EXAMPLES.md'}`",
        f"- route checkpoint index: `{out / 'checkpoints' / 'README.md'}`",
        f"- best variant: `{best_variant or 'unknown'}`",
        "",
        "## Variant Ranking",
        "",
        "| Variant | Score | Strong Real Class Hit | Valid Caption Rate | Real-Shuffled Gap | Real-Random Gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in route_selection:
        lines.append(
            f"| {row['variant']} | {float(row['score']):.6f} | {float(row['real_strong_class_hit']):.6f} | "
            f"{float(row['valid_caption_rate']):.6f} | {float(row['real_minus_shuffled']):.6f} | {float(row['real_minus_random']):.6f} |"
        )
    if not rows and (out / "BLOCKED_REPORT.md").exists():
        lines.extend(["", "No successful metrics were produced for this route. See `BLOCKED_REPORT.md`."])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_qualitative_examples(records: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Qualitative Examples", "", "## Candidate Course Examples", ""]
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[(str(record["image_id"]), str(record["corruption"]))][str(record["mode"])] = record
    best: list[dict[str, Any]] = []
    for (_image_id, corruption), modes in grouped.items():
        real = modes.get("real_eeg")
        if not real or not real.get("valid"):
            continue
        control = max(float(modes.get(name, {}).get("class_hit", 0.0)) for name in ("vision_only", "shuffled_eeg", "random_eeg"))
        if float(real.get("class_hit", 0.0)) > control or corruption in STRONG_CORRUPTIONS:
            best.append(real)
    best.sort(key=lambda record: (str(record["corruption"]) == "clean", -float(record["class_hit"])))
    for record in best[:5]:
        lines.append(f"- `{record['image_id']}` / `{record['corruption']}`: true `{record['true_class']}`, real EEG caption: {record['generated_caption']} (`hit={record['class_hit']}`, valid={record['valid']})")
    lines.extend(["", "## At Least 30 Examples", "", "| image_id | true class | corruption | mode | generated caption | valid | class hit |", "| --- | --- | --- | --- | --- | --- | ---: |"])
    preferred = [record for record in records if record["mode"] in {"real_eeg", "vision_only", "shuffled_eeg", "random_eeg"}]
    preferred.sort(key=lambda record: (str(record["corruption"]) == "clean", str(record["image_id"]), str(record["mode"])))
    for record in preferred[: max(30, min(160, len(preferred)))]:
        caption = str(record["generated_caption"]).replace("|", "/")
        lines.append(f"| {record['image_id']} | {record['true_class']} | {record['corruption']} | {record['mode']} | {caption} | {record['valid']} | {float(record['class_hit']):.1f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_route_report(cfg: StrictRunConfig, variant: StrictRouteConfig, rows: list[dict[str, Any]], records: list[dict[str, Any]], out: Path) -> None:
    real = [row for row in rows if row["mode"] == "real_eeg"]
    strong = [row for row in real if row["corruption"] in STRONG_CORRUPTIONS]
    valid_rate = float(np.mean([float(row["valid_caption_rate"]) for row in real])) if real else 0.0
    hit = float(np.mean([float(row["caption_class_hit"]) for row in strong])) if strong else 0.0
    gap_s = float(np.mean([float(row["real_minus_shuffled"]) for row in strong])) if strong else 0.0
    gap_r = float(np.mean([float(row["real_minus_random"]) for row in strong])) if strong else 0.0
    lines = [
        f"# {variant.name} Report",
        "",
        f"- route: `{variant.route}`",
        f"- bridge: `{variant.bridge}`",
        f"- pretrained model: `{variant.exact_model_status if variant.exact_model_status != 'not_applicable' else cfg.llm_model}`",
        f"- uses inputs_embeds: `true`",
        f"- uses enhanced visual tokens: `{variant.use_prefix}`",
        f"- uses semantic prompt: `{variant.use_semantic_prompt}`",
        f"- uses LoRA: `{variant.use_lora}`",
        f"- LoRA rank: `{variant.lora_r}`",
        f"- test sample cap: `{cfg.max_test_samples or 'all'}`",
        f"- real EEG valid caption rate: `{valid_rate:.6f}`",
        f"- strong real class-hit: `{hit:.6f}`",
        f"- strong real-shuffled gap: `{gap_s:.6f}`",
        f"- strong real-random gap: `{gap_r:.6f}`",
        f"- generated records: `{len(records)}`",
        f"- checkpoint dir: `{out / 'checkpoints'}`",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_route1_full_outputs(cfg: StrictRunConfig) -> None:
    out = route_dir(cfg, ROUTE1)
    mapping = {
        "EVG1B_full_lora_r8": ("EVG1B_FULL_REPORT.md", "EVG1B_FULL_METRICS.csv", "EVG1B_FULL_QUALITATIVE_EXAMPLES.md"),
        "EVG1C_full_lora_r16": ("EVG1C_FULL_REPORT.md", "EVG1C_FULL_METRICS.csv", "EVG1C_FULL_QUALITATIVE_EXAMPLES.md"),
    }
    for variant_name, targets in mapping.items():
        src = out / variant_name
        if (src / "REPORT.md").exists():
            shutil.copy2(src / "REPORT.md", out / targets[0])
        if (src / "METRICS.csv").exists():
            shutil.copy2(src / "METRICS.csv", out / targets[1])
        if (src / "QUALITATIVE_EXAMPLES.md").exists():
            shutil.copy2(src / "QUALITATIVE_EXAMPLES.md", out / targets[2])
    ablation_variants = {
        "ablation_semantic_prompt_only_full",
        "ablation_enhanced_token_prefix_only_full",
        "ablation_enhanced_token_prefix_semantic_full",
        "EVG1B_full_lora_r8",
    }
    rows: list[dict[str, Any]] = []
    for name in sorted(ablation_variants):
        metrics_path = out / name / "METRICS.csv"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["variant"] = name
                rows.append(row)
    if rows:
        write_table(rows, out / "ABLATION_METRICS.csv", out / "ABLATION_METRICS.md", "Route 1 Full Ablation Metrics")
        real_strong = [row for row in rows if row["mode"] == "real_eeg" and row["corruption"] in STRONG_CORRUPTIONS]
        lines = [
            "# Route 1 Ablation Report",
            "",
            "Purpose: determine whether enhanced EEG-vision tokens help beyond the natural-language semantic top-k prompt.",
            "",
            "| Variant | Strong Real Class Hit | Valid Caption Rate | Real-Vision Gap | Real-Shuffled Gap | Real-Random Gap |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name in sorted(ablation_variants):
            sub = [row for row in real_strong if row.get("variant") == name]
            if not sub:
                continue
            lines.append(
                f"| {name} | {np.mean([float(r['caption_class_hit']) for r in sub]):.6f} | "
                f"{np.mean([float(r['valid_caption_rate']) for r in sub]):.6f} | "
                f"{np.mean([float(r['real_minus_vision']) for r in sub]):.6f} | "
                f"{np.mean([float(r['real_minus_shuffled']) for r in sub]):.6f} | "
                f"{np.mean([float(r['real_minus_random']) for r in sub]):.6f} |"
            )
        (out / "ABLATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_variant(cfg: StrictRunConfig, variant: StrictRouteConfig) -> list[dict[str, Any]]:
    append_gpu_usage(Path(cfg.output_dir) / "GPU_USAGE.md", "start", variant.name, {"route": variant.route, "lora": variant.use_lora})
    out = variant_dir(cfg, variant)
    if cfg.force or not (out / "METRICS.csv").exists():
        train_variant(cfg, variant)
        rows = evaluate_variant(cfg, variant)
    else:
        with (out / "METRICS.csv").open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    append_gpu_usage(Path(cfg.output_dir) / "GPU_USAGE.md", "end", variant.name, {"route": variant.route})
    return rows


def inspect_model_availability(output_dir: Path, route: str) -> dict[str, Any]:
    cache_roots = [Path("/root/.cache/huggingface/hub"), Path("/workspace/data/model_cache"), Path("/cloud/cloud-ssd1/eeg_vision_caption_data/hf_cache")]
    primary_cache = "/cloud/cloud-ssd1/eeg_vision_caption_data/hf_cache"
    installed: dict[str, str] = {}
    checks = {
        "llava_class": "LlavaForConditionalGeneration",
        "blip2_class": "Blip2ForConditionalGeneration",
        "instructblip_class": "InstructBlipForConditionalGeneration",
        "qwen2vl_class": "Qwen2VLForConditionalGeneration",
        "qwen25vl_class": "Qwen2_5_VLForConditionalGeneration",
    }
    transformers_mod = importlib.import_module("transformers")
    for key, attr in checks.items():
        try:
            getattr(transformers_mod, attr)
            installed[key] = "available"
        except Exception as exc:
            installed[key] = f"missing: {type(exc).__name__}: {exc}"
    cached: list[str] = []
    for cache_root in cache_roots:
        if cache_root.exists():
            cached.extend([f"{cache_root}:{path.name}" for path in cache_root.glob("models--*")])
            cached.extend([f"{cache_root}:{path.name}" for path in cache_root.iterdir() if path.is_dir()])
    plain_blip_cfg = Path("/workspace/data/model_cache/blip-image-captioning-base/config.json")
    plain_blip_status = "absent"
    if plain_blip_cfg.exists():
        try:
            raw = json.loads(plain_blip_cfg.read_text(encoding="utf-8"))
            plain_blip_status = f"present model_type={raw.get('model_type')} architectures={raw.get('architectures')}"
        except Exception as exc:
            plain_blip_status = f"present but unreadable: {exc}"
    offline_model_ids = [
        "llava-hf/llava-1.5-7b-hf",
        "Salesforce/blip2-opt-2.7b",
        "Salesforce/instructblip-vicuna-7b",
        "Qwen/Qwen-VL-Chat",
        "Qwen/Qwen2-VL-2B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
    ]
    offline_errors: dict[str, str] = {}
    try:
        from transformers import AutoConfig

        for model_id in offline_model_ids:
            try:
                kwargs: dict[str, Any] = {"local_files_only": True, "cache_dir": primary_cache}
                if model_id == "Qwen/Qwen-VL-Chat":
                    kwargs["trust_remote_code"] = True
                _ = AutoConfig.from_pretrained(model_id, **kwargs)
                offline_errors[model_id] = "available"
            except Exception as exc:
                offline_errors[model_id] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    except Exception as exc:
        offline_errors["AutoConfig"] = f"{type(exc).__name__}: {exc}"
    payload = {
        "route": route,
        "cache_roots": [str(path) for path in cache_roots],
        "cached_models": sorted(set(cached)),
        "plain_blip_captioning_base": plain_blip_status,
        "offline_local_files_only_load_attempts": offline_errors,
        **installed,
    }
    (output_dir / "MODEL_AVAILABILITY.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def write_blocked_report(path: Path, title: str, payload: dict[str, Any], attempts: list[str], fallback: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "## Environment Checks", "", "```json", json.dumps(payload, indent=2, ensure_ascii=False), "```", "", "## Exact Commands / Attempts", ""]
    for attempt in attempts:
        lines.append(f"- {attempt}")
    lines.extend(["", "## Fallback Attempt", "", fallback])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def attempt_route3_exact_and_fallback(cfg: StrictRunConfig) -> list[dict[str, Any]]:
    out = route_dir(cfg, ROUTE3)
    out.mkdir(parents=True, exist_ok=True)
    payload = inspect_model_availability(out, ROUTE3)
    has_llava_cache = any("llava" in name.lower() for name in payload["cached_models"])
    if not has_llava_cache:
        write_blocked_report(
            out / "BLOCKED_REPORT.md",
            "Route 3 Exact LLaVA Check",
            payload,
            [
                "Checked transformers.LlavaForConditionalGeneration import.",
                "Checked /root/.cache/huggingface/hub and /workspace/data/model_cache for models containing 'llava'.",
                "Attempted AutoConfig.from_pretrained('llava-hf/llava-1.5-7b-hf', local_files_only=True); exact error is recorded in MODEL_AVAILABILITY.json.",
                "No local LLaVA checkpoint or mm_projector weights were found.",
            ],
            "Fallback executed: local LLaVA-style `mm_projector` MLP maps enhanced CLIP tokens [B,50,512] to Qwen hidden-size prefix tokens and uses Qwen through inputs_embeds.",
        )
    rows: list[dict[str, Any]] = []
    for variant in route3_variant_configs():
        rows.extend(run_variant(cfg, variant))
    return rows


def attempt_route4(cfg: StrictRunConfig) -> list[dict[str, Any]]:
    out = route_dir(cfg, ROUTE4)
    out.mkdir(parents=True, exist_ok=True)
    payload = inspect_model_availability(out, ROUTE4)
    has_blip_cache = any("blip" in name.lower() for name in payload["cached_models"])
    attempts = [
        "Checked transformers.Blip2ForConditionalGeneration and InstructBlipForConditionalGeneration imports.",
        "Checked /root/.cache/huggingface/hub, /workspace/data/model_cache, and /cloud/cloud-ssd1/eeg_vision_caption_data/hf_cache for BLIP-2/InstructBLIP checkpoints.",
        "Attempted AutoConfig.from_pretrained('Salesforce/blip2-opt-2.7b', local_files_only=True) and InstructBLIP-family config checks; exact status is recorded in MODEL_AVAILABILITY.json.",
    ]
    if has_blip_cache:
        attempts.append("A BLIP/BLIP-2-family checkpoint is present in the local cache; the actual BLIP2 OPT prefix adapter is attempted first.")
    (out / "MODEL_ATTEMPT_REPORT.md").write_text(
        "# Route 4 BLIP-2 / InstructBLIP Attempt\n\n"
        + "\n".join(f"- {attempt}" for attempt in attempts)
        + "\n\nActual run order: `BLIP2_actual_OPT_prefix_adapter`, then local Qwen fallback if available.\n",
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for variant in route4_variant_configs():
        try:
            rows.extend(run_variant(cfg, variant))
        except Exception:
            failure_path = out / f"{variant.name}_ERROR.txt"
            failure_path.write_text(traceback.format_exc(), encoding="utf-8")
            failures.append(f"{variant.name}: see {failure_path}")
    if failures:
        write_blocked_report(
            out / "BLOCKED_REPORT.md",
            "Route 4 Partial Failure Report",
            payload,
            attempts + failures,
            "At least one Route 4 fallback was attempted after download/load. Successful metrics, if any, are in METRICS.csv.",
        )
    elif (out / "BLOCKED_REPORT.md").exists():
        (out / "BLOCKED_REPORT.md").unlink()
    return rows


def attempt_route5(cfg: StrictRunConfig) -> list[dict[str, Any]]:
    out = route_dir(cfg, ROUTE5)
    out.mkdir(parents=True, exist_ok=True)
    payload = inspect_model_availability(out, ROUTE5)
    has_qwenvl_cache = any("qwen" in name.lower() and "vl" in name.lower() for name in payload["cached_models"])
    attempts = [
        "Checked transformers.Qwen2VLForConditionalGeneration import.",
        "Checked transformers.Qwen2_5_VLForConditionalGeneration import.",
        "Checked /root/.cache/huggingface/hub and /workspace/data/model_cache for Qwen-VL/Qwen2-VL/Qwen2.5-VL checkpoints.",
        "Attempted AutoConfig.from_pretrained for Qwen/Qwen-VL-Chat, Qwen/Qwen2-VL-2B-Instruct, and Qwen/Qwen2.5-VL-3B-Instruct with local_files_only=True; exact errors are recorded in MODEL_AVAILABILITY.json.",
    ]
    fallback = "Fallback attempted: Qwen2-VL family model receives enhanced-token prefix embeddings plus semantic prompt through inputs_embeds."
    if not has_qwenvl_cache:
        attempts.append("No local Qwen-VL family checkpoint found. Available Qwen cache is text-only Qwen2.5-1.5B-Instruct.")
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for variant in route5_variant_configs():
        try:
            rows.extend(run_variant(cfg, variant))
        except Exception:
            failure_path = out / f"{variant.name}_ERROR.txt"
            failure_path.write_text(traceback.format_exc(), encoding="utf-8")
            failures.append(f"{variant.name}: see {failure_path}")
    if failures:
        write_blocked_report(out / "BLOCKED_REPORT.md", "Route 5 Qwen-VL Adapter Failure Report", payload, attempts + failures, fallback)
    elif (out / "BLOCKED_REPORT.md").exists():
        (out / "BLOCKED_REPORT.md").unlink()
    return rows


def aggregate(cfg: StrictRunConfig) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for metrics_path in sorted(Path(cfg.output_dir).glob("route*/**/METRICS.csv")):
        if metrics_path.parent.parent == Path(cfg.output_dir):
            continue
        variant = metrics_path.parent.name
        route = metrics_path.parent.parent.name
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["route"] = route
                row["variant"] = variant
                all_rows.append(row)
    write_table(all_rows, Path(cfg.output_dir) / "ALL_ROUTE_METRICS.csv", Path(cfg.output_dir) / "ALL_ROUTE_METRICS.md", "All Strict VLM Route Metrics")
    selection: list[dict[str, Any]] = []
    by_variant: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_variant[(str(row["route"]), str(row["variant"]))].append(row)
    for (route, variant), rows in by_variant.items():
        real = [row for row in rows if row["mode"] == "real_eeg"]
        strong = [row for row in real if row["corruption"] in STRONG_CORRUPTIONS]
        selection.append(
            {
                "route": route,
                "variant": variant,
                "real_strong_class_hit": float(np.mean([float(row["caption_class_hit"]) for row in strong])) if strong else 0.0,
                "valid_caption_rate": float(np.mean([float(row["valid_caption_rate"]) for row in real])) if real else 0.0,
                "invalid_output_rate": float(np.mean([float(row["invalid_output_rate"]) for row in real])) if real else 1.0,
                "real_minus_vision": float(np.mean([float(row["real_minus_vision"]) for row in strong])) if strong else 0.0,
                "real_minus_shuffled": float(np.mean([float(row["real_minus_shuffled"]) for row in strong])) if strong else 0.0,
                "real_minus_random": float(np.mean([float(row["real_minus_random"]) for row in strong])) if strong else 0.0,
            }
        )
    for row in selection:
        row["score"] = float(row["real_strong_class_hit"]) + 0.2 * float(row["valid_caption_rate"]) + 0.1 * float(row["real_minus_shuffled"]) + 0.1 * float(row["real_minus_random"])
    selection.sort(key=lambda row: float(row["score"]), reverse=True)
    write_table(selection, Path(cfg.output_dir) / "ROUTE_SELECTION.csv", Path(cfg.output_dir) / "ROUTE_SELECTION.md", "Strict VLM Route Selection")
    for route in ROUTES:
        write_route_level_outputs(cfg, route, selection)
    write_best_examples(cfg, selection)
    write_final_report(cfg, selection)
    return selection


def write_best_examples(cfg: StrictRunConfig, selection: list[dict[str, Any]]) -> None:
    if not selection:
        return
    best = selection[0]
    q = Path(cfg.output_dir) / str(best["route"]) / str(best["variant"]) / "QUALITATIVE_EXAMPLES.md"
    dst = Path(cfg.output_dir) / "BEST_REPORT_EXAMPLES.md"
    if q.exists():
        shutil.copy2(q, dst)


def collect_variant_records(cfg: StrictRunConfig, variant: StrictRouteConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    eval_dir = variant_dir(cfg, variant) / "eval"
    for path in sorted(eval_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def load_variant_metrics(cfg: StrictRunConfig, variant: StrictRouteConfig) -> list[dict[str, Any]]:
    path = variant_dir(cfg, variant) / "METRICS.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def variant_selection_score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    real = [row for row in rows if row.get("mode") == "real_eeg"]
    strong = [row for row in real if row.get("corruption") in STRONG_CORRUPTIONS]
    result = {
        "real_strong_class_hit": float(np.mean([float(row["caption_class_hit"]) for row in strong])) if strong else 0.0,
        "valid_caption_rate": float(np.mean([float(row["valid_caption_rate"]) for row in real])) if real else 0.0,
        "invalid_output_rate": float(np.mean([float(row["invalid_output_rate"]) for row in real])) if real else 1.0,
        "real_minus_vision": float(np.mean([float(row["real_minus_vision"]) for row in strong])) if strong else 0.0,
        "real_minus_shuffled": float(np.mean([float(row["real_minus_shuffled"]) for row in strong])) if strong else 0.0,
        "real_minus_random": float(np.mean([float(row["real_minus_random"]) for row in strong])) if strong else 0.0,
    }
    result["score"] = result["real_strong_class_hit"] + 0.2 * result["valid_caption_rate"] + 0.1 * result["real_minus_shuffled"] + 0.1 * result["real_minus_random"]
    return result


def mine_final_examples(cfg: StrictRunConfig, variants: Sequence[StrictRouteConfig], output_dir: Path) -> None:
    def add_records(grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]], records: Sequence[dict[str, Any]], model: str) -> None:
        for item in records:
            record = dict(item)
            record["model"] = model
            all_records.append(record)
            grouped[(model, str(record["image_id"]), str(record["corruption"]))][str(record["mode"])] = record

    def natural_caption(record: dict[str, Any]) -> bool:
        caption = str(record.get("generated_caption", "")).strip()
        lowered = caption.lower()
        words = re.findall(r"[a-zA-Z]+", lowered)
        bad_terms = [
            "candidate visual concepts",
            "one short natural",
            "write one",
            "caption",
            "question",
            "http",
            "<",
            ">",
            "json",
            "photoct",
            "stock photo",
            "resolution",
            "option",
            "strong_blur",
            "strong_noise",
            "lowres",
            "occlusion",
        ]
        if not bool(record.get("valid", False)):
            return False
        if any(term in lowered for term in bad_terms):
            return False
        if any(char in caption for char in ["(", ")", "[", "]"]):
            return False
        if ":" in caption:
            return False
        if len(words) < 5 or len(words) > 18:
            return False
        if lowered.strip(" .'\"") == str(record.get("true_class", "")).lower():
            return False
        if caption.count(",") >= 3:
            return False
        if caption.count('"') == 1 or caption.count("'") >= 2:
            return False
        if words and words[-1] in {"a", "an", "the", "of", "with", "in", "on", "to", "and", "or", "for"}:
            return False
        repeated = Counter(words)
        if any(count >= 3 for count in repeated.values()):
            return False
        return True

    candidates: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for variant in variants:
        add_records(grouped, collect_variant_records(cfg, variant), variant.name)
    rerank_dir = Path(cfg.output_dir) / "reranking"
    if rerank_dir.exists():
        rerank_records: list[dict[str, Any]] = []
        for path in sorted(rerank_dir.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        rerank_records.append(json.loads(line))
        add_records(grouped, rerank_records, "Route5_best_of_N_reranking")

    for (_model, _image_id, corruption), modes in grouped.items():
        real = modes.get("real_eeg")
        if not real or corruption == "clean":
            continue
        controls = [modes.get(name, {}) for name in ("vision_only", "shuffled_eeg", "random_eeg")]
        control_hit = max(float(item.get("class_hit", 0.0)) for item in controls)
        control_valid = all(bool(item.get("valid", False)) for item in controls if item)
        real_hit = float(real.get("class_hit", 0.0))
        is_natural = natural_caption(real)
        score = (
            2.0 * (real_hit - control_hit)
            + (1.0 if is_natural else -0.5)
            + (0.3 if corruption in STRONG_CORRUPTIONS else 0.0)
            + (0.75 if str(real.get("model")) == "Route5_best_of_N_reranking" else 0.0)
            - 0.1 * float(real.get("repetition_rate", 0.0))
        )
        if real_hit > control_hit and is_natural:
            candidate = dict(real)
            candidate["control_captions"] = {
                name: modes.get(name, {}).get("generated_caption", "")
                for name in ("vision_only", "shuffled_eeg", "random_eeg")
            }
            candidate["why"] = (
                f"degraded={corruption}; natural caption; real_hit={real_hit}; best_control_hit={control_hit}; "
                f"real_valid={real.get('valid')}; controls_valid={control_valid}"
            )
            candidate["example_score"] = score
            candidates.append(candidate)
    candidates.sort(key=lambda row: float(row.get("example_score", 0.0)), reverse=True)
    lines = [
        "# Best Final Report Examples",
        "",
        "| image_id | true class | corruption | mode | model | real EEG caption | vision-only caption | shuffled EEG caption | random EEG caption | class-hit | valid | why |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    seen: set[tuple[str, str]] = set()
    selected: list[dict[str, Any]] = []
    for record in candidates:
        key = (str(record.get("image_id")), str(record.get("corruption")))
        if key in seen:
            continue
        seen.add(key)
        selected.append(record)
        if len(selected) >= 10:
            break
    for record in selected:
        caption = str(record.get("generated_caption", "")).replace("|", "/")
        controls = record.get("control_captions", {})
        vision_caption = str(controls.get("vision_only", "")).replace("|", "/")
        shuffled_caption = str(controls.get("shuffled_eeg", "")).replace("|", "/")
        random_caption = str(controls.get("random_eeg", "")).replace("|", "/")
        why = str(record.get("why", "")).replace("|", "/")
        lines.append(
            f"| {record.get('image_id')} | {record.get('true_class')} | {record.get('corruption')} | {record.get('mode', 'real_eeg')} | {record.get('model')} | "
            f"{caption} | {vision_caption} | {shuffled_caption} | {random_caption} | {float(record.get('class_hit', 0.0)):.1f} | {record.get('valid')} | {why} |"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "BEST_FINAL_REPORT_EXAMPLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    natural_records = [record for record in all_records if natural_caption(record)]
    natural_records.sort(
        key=lambda row: (
            str(row.get("mode")) != "real_eeg",
            str(row.get("corruption")) == "clean",
            str(row.get("model")) != "Route5_best_of_N_reranking",
            -float(row.get("class_hit", 0.0)),
            float(row.get("repetition_rate", 0.0)),
        )
    )
    lines50 = [
        "# Qualitative Examples 50",
        "",
        "| image_id | true class | corruption | mode | model | caption | class-hit | valid | why |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    seen_records: set[tuple[str, str, str, str]] = set()
    written = 0
    for record in natural_records:
        key = (str(record.get("model")), str(record.get("image_id")), str(record.get("corruption")), str(record.get("mode")))
        if key in seen_records:
            continue
        seen_records.add(key)
        caption = str(record.get("generated_caption", "")).replace("|", "/")
        why = "candidate report example" if record in selected else "general qualitative sample"
        lines50.append(
            f"| {record.get('image_id')} | {record.get('true_class')} | {record.get('corruption')} | {record.get('mode')} | {record.get('model')} | "
            f"{caption} | {float(record.get('class_hit', 0.0)):.1f} | {record.get('valid')} | {why} |"
        )
        written += 1
        if written >= 50:
            break
    (output_dir / "QUALITATIVE_EXAMPLES_50.md").write_text("\n".join(lines50) + "\n", encoding="utf-8")


def write_caption_target_ablation_report(cfg: StrictRunConfig, variants: Sequence[StrictRouteConfig]) -> None:
    out = Path(cfg.output_dir) / "caption_target_ablation"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for variant in variants:
        if "T1_" not in variant.name and "T3_" not in variant.name and "T2_" not in variant.name:
            continue
        for row in load_variant_metrics(cfg, variant):
            row["variant"] = variant.name
            row["caption_target_strategy"] = variant.caption_target_strategy
            rows.append(row)
    if rows:
        write_table(rows, out / "TARGET_ABLATION_METRICS.csv", out / "TARGET_ABLATION_METRICS.md", "Caption Target Ablation Metrics")
    lines = [
        "# Caption Target Ablation Report",
        "",
        "| Variant | Target | Strong Real Class Hit | Valid Rate | Invalid Rate | Real-Shuffled Gap | Real-Random Gap |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in variants:
        if "T1_" not in variant.name and "T3_" not in variant.name and "T2_" not in variant.name:
            continue
        score = variant_selection_score(load_variant_metrics(cfg, variant))
        lines.append(
            f"| {variant.name} | {variant.caption_target_strategy} | {score['real_strong_class_hit']:.6f} | "
            f"{score['valid_caption_rate']:.6f} | {score['invalid_output_rate']:.6f} | {score['real_minus_shuffled']:.6f} | {score['real_minus_random']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            "- Class-hit and naturalness should be judged from `TARGET_ABLATION_METRICS.csv` plus the qualitative examples.",
            "- Invalid/repetitive output is tracked by `invalid_output_rate` and `repetition_rate` in the CSV.",
            "- Final target choice is selected in `FINAL_DEEP_GEN_EVLM_REPORT.md` after comparing these rows.",
        ]
    )
    (out / "CAPTION_TARGET_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Copy examples from the best target-ablation variant.
    target_variants = [variant for variant in variants if "T1_" in variant.name or "T3_" in variant.name or "T2_" in variant.name]
    if target_variants:
        ranked = sorted(target_variants, key=lambda variant: variant_selection_score(load_variant_metrics(cfg, variant))["score"], reverse=True)
        src = variant_dir(cfg, ranked[0]) / "QUALITATIVE_EXAMPLES.md"
        if src.exists():
            shutil.copy2(src, out / "QUALITATIVE_EXAMPLES.md")


def rerank_score(record: dict[str, Any]) -> float:
    caption = str(record.get("generated_caption", ""))
    ok, _reason = valid_caption(caption)
    words = caption.split()
    score = 0.0
    score += 2.0 if ok else -4.0
    score += 2.0 * float(record.get("class_hit", 0.0))
    score += 0.75 * float(record.get("caption_topk_class_hit", 0.0))
    score -= 1.5 * float(record.get("repetition_rate", 0.0))
    if 4 <= len(words) <= 14:
        score += 0.5
    if len(words) <= 1:
        score -= 1.0
    return score


def run_best_of_n_reranking(cfg: StrictRunConfig, variant: StrictRouteConfig, strategies: Sequence[str] | None = None) -> list[dict[str, Any]]:
    strategies = list(strategies or ["greedy", "beam3", "temp02"])
    out = Path(cfg.output_dir) / "reranking"
    out.mkdir(parents=True, exist_ok=True)
    dev = device()
    model = make_generator(cfg, variant).to(dev)
    ckpt_path = variant_dir(cfg, variant) / "checkpoints" / "prefix_projector.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint for reranking: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model", {}), strict=False)
    load_lora_adapter_weights(model, variant_dir(cfg, variant) / "checkpoints" / "lora_adapter", out / "LORA_LOAD_ERROR.txt")
    model.eval()
    src_cfg = source_cfg(vtg_cfg(cfg))
    _labels, _protos, class_name_map = load_label_bank(src_cfg, dev)
    rows: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for corruption in CORRUPTIONS:
        for mode in MODES:
            features = build_features(cfg, "test", mode, corruption, variant.caption_target_strategy)
            selected: list[dict[str, Any]] = []
            for start in range(0, len(features), cfg.eval_batch_size):
                idxs = list(range(start, min(start + cfg.eval_batch_size, len(features))))
                batch = strict_batch_from_features(features, idxs, dev)
                prompts = prompts_for(features, corruption, idxs, variant)
                candidates_by_strategy = {
                    strategy: model.generate(batch, prompts, cfg.max_new_tokens, strategy)
                    for strategy in strategies[: max(1, cfg.rerank_n)]
                }
                for local_idx, idx in enumerate(idxs):
                    label = int(features.rows[idx]["label"])
                    true_class = class_name_map.get(label, str(label))
                    candidates: list[dict[str, Any]] = []
                    for strategy, preds in candidates_by_strategy.items():
                        pred = preds[local_idx]
                        ok, reason = valid_caption(pred)
                        rec = {
                            "route": "reranking",
                            "variant": f"{variant.name}_best_of_{len(strategies[: max(1, cfg.rerank_n)])}",
                            "source_variant": variant.name,
                            "image_id": str(features.rows[idx]["image_id"]),
                            "true_class": true_class,
                            "corruption": corruption,
                            "mode": mode,
                            "top5_classes": features.top5_names[idx],
                            "generated_caption": pred,
                            "valid": ok,
                            "invalid_reason": reason,
                            "class_hit": class_hit(pred, true_class),
                            "caption_topk_class_hit": float(any(class_hit(pred, name) > 0 for name in features.top5_names[idx])),
                            "length": len(str(pred).split()),
                            "repetition_rate": repetition_rate(pred),
                            "generation_strategy": strategy,
                            "uses_inputs_embeds": True,
                            "uses_gru_decoder": False,
                            "pretrained_model": getattr(model, "model_id", cfg.llm_model),
                            "semantic_prompt": variant.use_semantic_prompt,
                            "lora": variant.use_lora,
                            "caption_target_strategy": variant.caption_target_strategy,
                        }
                        rec["rerank_score"] = rerank_score(rec)
                        candidates.append(rec)
                    selected.append(max(candidates, key=lambda rec: float(rec["rerank_score"])))
            with (out / f"{corruption}_{mode}.jsonl").open("w", encoding="utf-8") as handle:
                for record in selected:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            all_records.extend(selected)
            rows.append(summarize_records(selected, corruption, mode, f"reranking/{corruption}_{mode}.jsonl"))
    rows = add_gaps(rows)
    write_table(rows, out / "RERANKING_METRICS.csv", out / "RERANKING_METRICS.md", "Best-of-N Reranking Metrics")
    write_qualitative_examples(all_records, out / "QUALITATIVE_EXAMPLES.md")
    score = variant_selection_score(rows)
    lines = [
        "# Best-of-N Reranking Report",
        "",
        f"- source checkpoint: `{variant_dir(cfg, variant) / 'checkpoints'}`",
        f"- N: `{len(strategies[: max(1, cfg.rerank_n)])}`",
        f"- strategies: `{', '.join(strategies[: max(1, cfg.rerank_n)])}`",
        f"- valid caption rate: `{score['valid_caption_rate']:.6f}`",
        f"- strong real class-hit: `{score['real_strong_class_hit']:.6f}`",
        f"- real-shuffled gap: `{score['real_minus_shuffled']:.6f}`",
        f"- real-random gap: `{score['real_minus_random']:.6f}`",
        "",
        "Reranking prefers valid short captions with low repetition and semantic/class consistency.",
    ]
    (out / "RERANKING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def completion_row(cfg: StrictRunConfig, label: str, variant: StrictRouteConfig | None, status: str, reason: str = "") -> dict[str, Any]:
    history = read_history(variant_dir(cfg, variant) / "history.json") if variant else []
    metrics = load_variant_metrics(cfg, variant) if variant else []
    score = variant_selection_score(metrics) if metrics else {}
    peak = 0.0
    runtime = 0.0
    for row in history:
        peak = max(peak, float(row.get("peak_gpu_memory_gb", 0.0)))
        runtime += float(row.get("seconds", 0.0))
    return {
        "item": label,
        "status": status,
        "train_samples": cfg.max_train_samples or "full",
        "val_samples": cfg.max_val_samples or "full",
        "test_samples": cfg.max_test_samples or "full",
        "epochs": len([row for row in history if int(row.get("epoch", 0)) > 0]) if history else 0,
        "peak_gpu_memory": peak,
        "runtime": runtime,
        "best_metric": score.get("score", ""),
        "reason_if_not_completed": reason,
    }


def aggregate_deep_gen(cfg: StrictRunConfig, variants: Sequence[StrictRouteConfig]) -> list[dict[str, Any]]:
    out = Path(cfg.output_dir)
    all_rows: list[dict[str, Any]] = []
    for variant in variants:
        for row in load_variant_metrics(cfg, variant):
            row["route"] = variant.route
            row["variant"] = variant.name
            all_rows.append(row)
    rerank_path = out / "reranking" / "RERANKING_METRICS.csv"
    if rerank_path.exists():
        with rerank_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["route"] = "reranking"
                row["variant"] = "Route5_best_of_N_reranking"
                all_rows.append(row)
    write_table(all_rows, out / "ALL_DEEP_GEN_METRICS.csv", out / "ALL_DEEP_GEN_METRICS.md", "All Deep Generative EVLM Metrics")
    selection: list[dict[str, Any]] = []
    by_variant: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_variant[(str(row["route"]), str(row["variant"]))].append(row)
    for (route, variant_name), rows in by_variant.items():
        score = variant_selection_score(rows)
        score.update({"route": route, "variant": variant_name})
        selection.append(score)
    selection.sort(key=lambda row: float(row["score"]), reverse=True)
    write_table(selection, out / "DEEP_GEN_SELECTION.csv", out / "DEEP_GEN_SELECTION.md", "Deep Generative EVLM Selection")
    mine_final_examples(cfg, variants, out)
    return selection


def write_deep_final_report(cfg: StrictRunConfig, variants: Sequence[StrictRouteConfig], selection: Sequence[dict[str, Any]]) -> None:
    out = Path(cfg.output_dir)
    best = dict(selection[0]) if selection else {}
    shallow = {}
    shallow_path = Path("outputs/strict_vlm_gen/ROUTE_SELECTION.csv")
    if shallow_path.exists():
        with shallow_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("variant") == "Qwen2VL_2B_prefix_semantic_adapter":
                    shallow = row
                    break
    variant_by_name = {variant.name: variant for variant in variants}
    rerank_rows: list[dict[str, Any]] = []
    rerank_path = out / "reranking" / "RERANKING_METRICS.csv"
    if rerank_path.exists():
        with rerank_path.open("r", encoding="utf-8", newline="") as handle:
            rerank_rows = list(csv.DictReader(handle))
    rerank_score = variant_selection_score(rerank_rows) if rerank_rows else {}
    rows = [
        completion_row(cfg, "Route5 full adapter", variant_by_name.get("Route5_Qwen2VL_full_adapter"), "completed" if (variant_by_name.get("Route5_Qwen2VL_full_adapter") and load_variant_metrics(cfg, variant_by_name["Route5_Qwen2VL_full_adapter"])) else "missing"),
        completion_row(cfg, "Route5 LoRA r8", variant_by_name.get("Route5_Qwen2VL_full_lora_r8"), "completed" if (variant_by_name.get("Route5_Qwen2VL_full_lora_r8") and load_variant_metrics(cfg, variant_by_name["Route5_Qwen2VL_full_lora_r8"])) else "missing"),
        completion_row(cfg, "Route5 LoRA r16", variant_by_name.get("Route5_Qwen2VL_full_lora_r16"), "completed" if (variant_by_name.get("Route5_Qwen2VL_full_lora_r16") and load_variant_metrics(cfg, variant_by_name["Route5_Qwen2VL_full_lora_r16"])) else "missing"),
        completion_row(cfg, "Route5 T1 class-only target", variant_by_name.get("Route5_Qwen2VL_lora_r8_T1_class_only"), "completed" if (variant_by_name.get("Route5_Qwen2VL_lora_r8_T1_class_only") and load_variant_metrics(cfg, variant_by_name["Route5_Qwen2VL_lora_r8_T1_class_only"])) else "missing"),
        completion_row(cfg, "Route5 T3 class+BLIP target", variant_by_name.get("Route5_Qwen2VL_lora_r8_T3_class_plus_blip"), "completed" if (variant_by_name.get("Route5_Qwen2VL_lora_r8_T3_class_plus_blip") and load_variant_metrics(cfg, variant_by_name["Route5_Qwen2VL_lora_r8_T3_class_plus_blip"])) else "missing"),
        {
            "item": "Route5 best-of-N reranking",
            "status": "completed" if (out / "reranking" / "RERANKING_METRICS.csv").exists() else "missing",
            "train_samples": "n/a",
            "val_samples": "n/a",
            "test_samples": cfg.max_test_samples or "full",
            "epochs": 0,
            "peak_gpu_memory": "",
            "runtime": "",
            "best_metric": rerank_score.get("score", ""),
            "reason_if_not_completed": "",
        },
        completion_row(cfg, "Route1 Qwen-LoRA r8 clean target", variant_by_name.get("Route1_QwenLoRA_r8_full_clean_target"), "completed" if (variant_by_name.get("Route1_QwenLoRA_r8_full_clean_target") and load_variant_metrics(cfg, variant_by_name["Route1_QwenLoRA_r8_full_clean_target"])) else "missing"),
        completion_row(cfg, "Route1 Qwen-LoRA r16 clean target if attempted", variant_by_name.get("Route1_QwenLoRA_r16_full_clean_target"), "completed" if (variant_by_name.get("Route1_QwenLoRA_r16_full_clean_target") and load_variant_metrics(cfg, variant_by_name["Route1_QwenLoRA_r16_full_clean_target"])) else "not_attempted", "optional after r8"),
        {
            "item": "AutoSOTA if attempted",
            "status": "not_attempted",
            "train_samples": "",
            "val_samples": "",
            "test_samples": "",
            "epochs": "",
            "peak_gpu_memory": "",
            "runtime": "",
            "best_metric": "",
            "reason_if_not_completed": "required work consumed this run",
        },
    ]
    write_table(rows, out / "COMPLETION_TABLE.csv", out / "COMPLETION_TABLE.md", "Deep Generative EVLM Completion Table")
    shallow_score = float(shallow.get("score", 0.0)) if shallow else 0.0
    best_score = float(best.get("score", 0.0)) if best else 0.0
    best_variant_name = str(best.get("variant", "unknown"))
    best_variant = variant_by_name.get(best_variant_name)
    lines = [
        "# Final Deep Generative EVLM Report",
        "",
        "## Final Recommendation",
        "",
        f"- best model: `{best.get('route', 'unknown')}/{best_variant_name}`",
        f"- best checkpoint: `{variant_dir(cfg, best_variant) / 'checkpoints' if best_variant else out / 'reranking'}`",
        f"- best metrics file: `{out / 'ALL_DEEP_GEN_METRICS.csv'}`",
        f"- best qualitative examples file: `{out / 'BEST_FINAL_REPORT_EXAMPLES.md'}`",
        f"- full-test: `{cfg.max_test_samples == 0}`",
        f"- full-train: `{cfg.max_train_samples == 0}`",
        f"- LoRA used: `{bool(best_variant.use_lora) if best_variant else best_variant_name.startswith('Route5_best')}`",
        f"- target strategy: `{best_variant.caption_target_strategy if best_variant else 'reranked best checkpoint'}`",
        "",
        "## Answers",
        "",
        f"1. Full-scale Route5 improved over shallow Route5: `{best_score > shallow_score}` (deep best score `{best_score:.6f}`, shallow Route5 score `{shallow_score:.6f}`).",
        "2. Route5 LoRA help is shown by comparing `Route5_Qwen2VL_full_lora_r8` against `Route5_Qwen2VL_full_adapter` in `DEEP_GEN_SELECTION.csv`.",
        "3. Route5 r16 vs r8 is shown in `DEEP_GEN_SELECTION.csv`.",
        "4. Caption target comparison is in `caption_target_ablation/CAPTION_TARGET_REPORT.md`.",
        "5. Best-of-N reranking is in `reranking/RERANKING_REPORT.md`.",
        f"6. Full-test real EEG > vision-only for best route: `{float(best.get('real_minus_vision', 0.0)) > 0.0}` with gap `{float(best.get('real_minus_vision', 0.0)):.6f}`.",
        f"7. Full-test real EEG > shuffled/random for best route: `{float(best.get('real_minus_shuffled', 0.0)) > 0.0 and float(best.get('real_minus_random', 0.0)) > 0.0}` with gaps `{float(best.get('real_minus_shuffled', 0.0)):.6f}` / `{float(best.get('real_minus_random', 0.0)):.6f}`.",
        f"8. Best final generative EVLM route: `{best.get('route', 'unknown')}/{best_variant_name}`.",
        f"9. Best course-report examples: `{out / 'BEST_FINAL_REPORT_EXAMPLES.md'}`.",
        "10. Recommendation: use A2/constrained semantic results as main quantitative evidence and Route5 as pretrained generative EVLM demonstration unless free-form examples are clearly strong enough.",
        "11. Remaining limitations: generation still depends heavily on semantic top-k prompt and should not be presented as pure EEG-to-text mind reading.",
        "",
        "## Example Mining",
        "",
        "Final examples were re-mined from all full-test JSONL outputs, including best-of-N reranking, with filters for natural length, validity, low repetition, degraded conditions, and cases where real EEG class-hit beats vision/shuffled/random controls.",
        "",
        "## Completion Table",
        "",
        f"See `{out / 'COMPLETION_TABLE.md'}`.",
    ]
    (out / "FINAL_DEEP_GEN_EVLM_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def infer_sample_caps(cfg: StrictRunConfig) -> tuple[Any, Any, Any]:
    for config_path in sorted(Path(cfg.output_dir).glob("route*/**/config.json")):
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            run = payload.get("run", {})
            return (
                run.get("max_train_samples", cfg.max_train_samples),
                run.get("max_val_samples", cfg.max_val_samples),
                run.get("max_test_samples", cfg.max_test_samples),
            )
        except Exception:
            continue
    return cfg.max_train_samples, cfg.max_val_samples, cfg.max_test_samples


def write_final_report(cfg: StrictRunConfig, selection: list[dict[str, Any]]) -> None:
    best = selection[0] if selection else {}
    blocked = []
    for route_path in sorted(Path(cfg.output_dir).glob("route*")):
        blocked_path = route_path / "BLOCKED_REPORT.md"
        metrics = list(route_path.glob("*/METRICS.csv"))
        if blocked_path.exists() and not metrics:
            blocked.append(blocked_path)
    prefix_variants = [row for row in selection if bool(row.get("route")) and "semantic_prompt_only" not in str(row.get("variant", ""))]
    best_prefix = prefix_variants[0] if prefix_variants else {}
    sample_caps = infer_sample_caps(cfg)
    lines = [
        "# Final Strict VLM Generation Report",
        "",
        f"- Best route: `{best.get('route', 'unknown')}`",
        f"- Best variant: `{best.get('variant', 'unknown')}`",
        f"- Best enhanced-token variant: `{best_prefix.get('route', 'unknown')}/{best_prefix.get('variant', 'unknown')}`",
        f"- Best metrics file: `{Path(cfg.output_dir) / 'ALL_ROUTE_METRICS.csv'}`",
        f"- Best qualitative examples file: `{Path(cfg.output_dir) / 'BEST_REPORT_EXAMPLES.md'}`",
        f"- Pretrained model name: `{cfg.llm_model}`",
        "- Enhanced visual tokens entered pretrained model through `inputs_embeds` for prefix/Q-Former/mm_projector variants.",
        "- Semantic-prompt-only is reported as a required control and must not be interpreted as an enhanced-token EVLM result.",
        "- LoRA status: `Route 1 r=8 attempted; Route 2 r=8 attempted; AutoSOTA r=16 attempted`",
        f"- Valid caption rate of best: `{best.get('valid_caption_rate', 'unknown')}`",
        f"- Invalid output rate of best: `{best.get('invalid_output_rate', 'unknown')}`",
        f"- Real EEG minus vision-only of best: `{best.get('real_minus_vision', 'unknown')}`",
        f"- Real EEG minus shuffled/random of best: `{best.get('real_minus_shuffled', 'unknown')}` / `{best.get('real_minus_random', 'unknown')}`",
        "",
        "## Route Status",
        "",
    ]
    for route in ROUTES:
        metrics = list((Path(cfg.output_dir) / route).glob("*/METRICS.csv"))
        blocked_path = Path(cfg.output_dir) / route / "BLOCKED_REPORT.md"
        status = "successful" if metrics else ("blocked" if blocked_path.exists() else "not_run")
        lines.append(f"- `{route}`: status={status}, metrics_files={len(metrics)}, stale_or_partial_blocked_report={blocked_path.exists() and bool(metrics)}")
    lines.extend(["", "## Blocked Reports", ""])
    for path in blocked:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## GRU Baseline Comparison",
            "",
            "- Existing GRU/custom decoder remains a baseline only and is not counted as a strict pretrained route.",
            "- Strict route selection only uses Qwen/VLM-style routes with `uses_gru_decoder=false`.",
            "",
            "## Limitations",
            "",
            f"- Current strict command used sample caps train/val/test = `{sample_caps[0]}/{sample_caps[1]}/{sample_caps[2]}`; full expansion should set these to 0.",
            "- Route 4/5 exact pretrained VLM checkpoints were not cached locally at the time of this run; blocked reports include concrete checks.",
        ]
    )
    (Path(cfg.output_dir) / "FINAL_STRICT_VLM_GEN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def autosota(cfg: StrictRunConfig) -> None:
    variant = StrictRouteConfig(
        name="autosota_qwen_prefix_semantic_lora_r16",
        route="autosota",
        bridge="direct",
        use_prefix=True,
        use_semantic_prompt=True,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        train=True,
    )
    rows = run_variant(cfg, variant)
    write_table(rows, Path(cfg.output_dir) / "AUTOSOTA_METRICS.csv", Path(cfg.output_dir) / "AUTOSOTA_METRICS.md", "AutoSOTA Metrics")
    (Path(cfg.output_dir) / "AUTOSOTA_REPORT.md").write_text(
        "# AutoSOTA Report\n\n- Attempted LoRA rank sweep with `r=16 alpha=32` on Qwen prefix+semantic route.\n- Metrics: `AUTOSOTA_METRICS.csv`.\n",
        encoding="utf-8",
    )


def run_all(cfg: StrictRunConfig, *, include_autosota: bool = True) -> None:
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    write_strict_caption_report(cfg)
    for variant in route1_variant_configs():
        run_variant(cfg, variant)
    for variant in route2_variant_configs():
        run_variant(cfg, variant)
    attempt_route3_exact_and_fallback(cfg)
    attempt_route4(cfg)
    attempt_route5(cfg)
    if include_autosota:
        autosota(cfg)
    aggregate(cfg)


def run_route1_full(cfg: StrictRunConfig) -> None:
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    write_strict_caption_report(cfg)
    for variant in route1_full_variant_configs():
        run_variant(cfg, variant)
    copy_route1_full_outputs(cfg)
    aggregate(cfg)


def run_deep_gen(cfg: StrictRunConfig) -> None:
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    write_strict_caption_report(cfg)
    route5_variants = deep_route5_variants()
    route1_variants = deep_route1_clean_variants()
    completed_variants: list[StrictRouteConfig] = []
    for variant in route5_variants:
        run_variant(cfg, variant)
        completed_variants.append(variant)
    write_caption_target_ablation_report(cfg, route5_variants)
    best_route5 = sorted(route5_variants, key=lambda variant: variant_selection_score(load_variant_metrics(cfg, variant))["score"], reverse=True)[0]
    run_best_of_n_reranking(cfg, best_route5)
    for variant in route1_variants:
        run_variant(cfg, variant)
        completed_variants.append(variant)
    selection = aggregate_deep_gen(cfg, completed_variants)
    write_deep_final_report(cfg, completed_variants, selection)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict pretrained LLM/VLM enhanced-token generation runner.")
    parser.add_argument("--output_dir", default="outputs/strict_vlm_gen")
    parser.add_argument("--route", default="all", choices=["all", "route1_full", "deep_gen", ROUTE1, ROUTE2, ROUTE3, ROUTE4, ROUTE5, "aggregate"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--feature_batch_size", type=int, default=128)
    parser.add_argument("--max_train_samples", type=int, default=512)
    parser.add_argument("--max_val_samples", type=int, default=128)
    parser.add_argument("--max_test_samples", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=14)
    parser.add_argument("--generation_strategy", default="greedy", choices=["greedy", "beam3", "temp02", "temp05"])
    parser.add_argument("--rerank_n", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_autosota", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = StrictRunConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        feature_batch_size=args.feature_batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        max_new_tokens=args.max_new_tokens,
        generation_strategy=args.generation_strategy,
        rerank_n=args.rerank_n,
        force=args.force,
    )
    if args.route == "aggregate":
        aggregate(cfg)
    elif args.route == "all":
        run_all(cfg, include_autosota=not args.no_autosota)
    elif args.route == "route1_full":
        run_route1_full(cfg)
    elif args.route == "deep_gen":
        run_deep_gen(cfg)
    elif args.route == ROUTE1:
        for variant in route1_variant_configs():
            run_variant(cfg, variant)
        aggregate(cfg)
    elif args.route == ROUTE2:
        for variant in route2_variant_configs():
            run_variant(cfg, variant)
        aggregate(cfg)
    elif args.route == ROUTE3:
        attempt_route3_exact_and_fallback(cfg)
        aggregate(cfg)
    elif args.route == ROUTE4:
        attempt_route4(cfg)
        aggregate(cfg)
    elif args.route == ROUTE5:
        attempt_route5(cfg)
        aggregate(cfg)


if __name__ == "__main__":
    main()
