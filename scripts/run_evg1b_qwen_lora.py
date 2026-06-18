from __future__ import annotations

import argparse
import csv
import json
import math
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
    class_hit,
    load_label_bank,
    valid_caption,
    write_json,
    write_table,
)
from scripts.run_vision_token_gen_evlm import (
    VisionTokenBatch,
    VisionTokenGenConfig,
    batch_from_features,
    build_vision_token_features,
    resolve_vtf_checkpoint,
    source_cfg,
    write_enhanced_token_source_report,
)
from src.utils.seed import seed_everything


@dataclass
class EVG1BLoRAConfig:
    output_dir: str = "outputs/vision_token_gen_evlm"
    source_output_dir: str = "outputs/token_generative_evlm"
    vtf_checkpoint: str = "outputs/token_generative_evlm/token_fusion/VTF3_confidence_beta_margin_M4_seed42/checkpoints/best.pt"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    seed: int = 42
    epochs: int = 2
    batch_size: int = 2
    eval_batch_size: int = 4
    lr: float = 2.0e-4
    weight_decay: float = 0.01
    max_train_samples: int = 512
    max_val_samples: int = 128
    max_test_samples: int = 32
    prefix_len: int = 16
    max_prompt_length: int = 96
    max_caption_length: int = 24
    max_new_tokens: int = 18
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_vtg_cfg(cfg: EVG1BLoRAConfig) -> VisionTokenGenConfig:
    return VisionTokenGenConfig(
        output_dir=cfg.output_dir,
        source_output_dir=cfg.source_output_dir,
        vtf_checkpoint=cfg.vtf_checkpoint,
        variant="EVG1B_lora_r8_prefix_semantic_prompt",
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        eval_batch_size=max(16, cfg.eval_batch_size),
        max_train_samples=cfg.max_train_samples,
        max_val_samples=cfg.max_val_samples,
        max_test_samples=cfg.max_test_samples,
    )


def format_prompt(top5_names: Sequence[str], corruption: str) -> str:
    concepts = ", ".join(str(name) for name in top5_names[:5])
    return (
        "Write one short natural image caption. Do not mention EEG. "
        "Do not output JSON, code, markdown, URLs, or explanations.\n"
        f"Candidate visual concepts: {concepts}\n"
        f"Visual condition: {corruption}\n"
        "Caption:"
    )


class QwenVisualTokenPrefixLoRA(nn.Module):
    def __init__(self, cfg: EVG1BLoRAConfig) -> None:
        super().__init__()
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        base = AutoModelForCausalLM.from_pretrained(cfg.llm_model, local_files_only=True, torch_dtype=dtype)
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.target_modules),
            bias="none",
        )
        self.llm = get_peft_model(base, lora_cfg)
        hidden = int(self.llm.config.hidden_size)
        self.query_tokens = nn.Parameter(torch.randn(cfg.prefix_len, 512) * 0.02)
        self.conf_token = nn.Linear(1, 512)
        self.corr_embed = nn.Embedding(len(CORRUPTIONS), 512)
        self.resampler = nn.MultiheadAttention(512, num_heads=8, batch_first=True, dropout=0.05)
        self.prefix_projector = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, hidden))
        self.last_seen_visual_tokens_shape: tuple[int, ...] | None = None

    def prefix_embeds(self, batch: VisionTokenBatch) -> torch.Tensor:
        visual = F.normalize(batch.visual_tokens.float(), dim=-1)
        self.last_seen_visual_tokens_shape = tuple(int(x) for x in visual.shape)
        memory = torch.cat(
            [
                visual,
                F.normalize(batch.eeg_tokens.float(), dim=-1),
                F.normalize(batch.topk_prototypes.float(), dim=-1),
                self.conf_token(batch.confidence.float()).unsqueeze(1),
                self.corr_embed(batch.corruption_ids.long()).unsqueeze(1),
            ],
            dim=1,
        )
        queries = self.query_tokens.unsqueeze(0).expand(memory.shape[0], -1, -1)
        prefix_512, _ = self.resampler(queries, memory, memory, need_weights=False)
        prefix = self.prefix_projector(prefix_512)
        embed_dtype = self.llm.get_input_embeddings().weight.dtype
        return prefix.to(dtype=embed_dtype)

    def encode_text(self, prompts: Sequence[str], captions: Sequence[str], dev: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_tokens = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_prompt_length,
        )
        target_text = [str(caption).strip() + self.tokenizer.eos_token for caption in captions]
        caption_tokens = self.tokenizer(
            target_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_caption_length,
            add_special_tokens=False,
        )
        prompt_ids = prompt_tokens["input_ids"].to(dev)
        prompt_mask = prompt_tokens["attention_mask"].to(dev)
        caption_ids = caption_tokens["input_ids"].to(dev)
        caption_mask = caption_tokens["attention_mask"].to(dev)
        return prompt_ids, prompt_mask, caption_ids, caption_mask

    def forward(self, batch: VisionTokenBatch, prompts: Sequence[str], captions: Sequence[str]) -> torch.Tensor:
        dev = batch.visual_tokens.device
        prefix = self.prefix_embeds(batch)
        prompt_ids, prompt_mask, caption_ids, caption_mask = self.encode_text(prompts, captions, dev)
        input_ids = torch.cat([prompt_ids, caption_ids], dim=1)
        text_mask = torch.cat([prompt_mask, caption_mask], dim=1)
        token_embeds = self.llm.get_input_embeddings()(input_ids).to(dtype=prefix.dtype)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
        prefix_mask = torch.ones(prefix.shape[:2], dtype=text_mask.dtype, device=dev)
        attention_mask = torch.cat([prefix_mask, text_mask], dim=1)
        ignore_prefix = torch.full((input_ids.shape[0], prefix.shape[1] + prompt_ids.shape[1]), -100, dtype=torch.long, device=dev)
        caption_labels = caption_ids.masked_fill(caption_mask == 0, -100)
        labels = torch.cat([ignore_prefix, caption_labels], dim=1)
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    @torch.no_grad()
    def generate(self, batch: VisionTokenBatch, prompts: Sequence[str], max_new_tokens: int) -> list[str]:
        self.eval()
        dev = batch.visual_tokens.device
        prefix = self.prefix_embeds(batch)
        prompt_tokens = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_prompt_length,
        )
        prompt_ids = prompt_tokens["input_ids"].to(dev)
        prompt_mask = prompt_tokens["attention_mask"].to(dev)
        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids).to(dtype=prefix.dtype)
        inputs_embeds = torch.cat([prefix, prompt_embeds], dim=1)
        attention_mask = torch.cat(
            [torch.ones(prefix.shape[:2], dtype=prompt_mask.dtype, device=dev), prompt_mask],
            dim=1,
        )
        try:
            generated = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        except Exception:
            return self._manual_greedy(prefix, prompt_ids, prompt_mask, max_new_tokens)

    def _manual_greedy(self, prefix: torch.Tensor, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor, max_new_tokens: int) -> list[str]:
        dev = prefix.device
        generated = torch.empty((prefix.shape[0], 0), dtype=torch.long, device=dev)
        finished = torch.zeros(prefix.shape[0], dtype=torch.bool, device=dev)
        for step in range(max_new_tokens):
            if generated.numel():
                ids = torch.cat([prompt_ids, generated], dim=1)
            else:
                ids = prompt_ids
            token_embeds = self.llm.get_input_embeddings()(ids).to(dtype=prefix.dtype)
            inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=dev)
            logits = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits[:, -1]
            logits[:, [self.tokenizer.pad_token_id]] = -1e9
            if step < 3 and self.tokenizer.eos_token_id is not None:
                logits[:, self.tokenizer.eos_token_id] = -1e9
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            if self.tokenizer.eos_token_id is not None:
                finished = finished | (next_token == self.tokenizer.eos_token_id)
                if bool(finished.all()):
                    break
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


def optimizer_params(model: QwenVisualTokenPrefixLoRA) -> list[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def build_prompts(features: Any, corruption: str, idxs: Sequence[int]) -> list[str]:
    return [format_prompt(features.top5_names[i], corruption) for i in idxs]


def train(cfg: EVG1BLoRAConfig) -> Path:
    seed_everything(cfg.seed)
    dev = device()
    out = Path(cfg.output_dir) / "EVG1B" / "EVG1B_lora_r8_prefix_semantic_prompt_seed42"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", asdict(cfg))

    vtg = to_vtg_cfg(cfg)
    train_features = build_vision_token_features(vtg, "train", "real_eeg", "clean")
    val_features = build_vision_token_features(vtg, "val", "real_eeg", "clean")
    write_enhanced_token_source_report(
        Path(cfg.output_dir) / "EVG1B" / "ENHANCED_TOKEN_SOURCE_FOR_LORA.md",
        vtf_checkpoint=resolve_vtf_checkpoint(vtg),
        token_shape=tuple(int(x) for x in train_features.batch.visual_tokens.shape),
        eeg_token_shape=tuple(int(x) for x in train_features.batch.eeg_tokens.shape),
        modes=MODES,
        frozen_modules=["CLIP ViT-B/32", "A2 EEG encoder", "VTF3 token fusion", "Qwen base weights"],
    )

    model = QwenVisualTokenPrefixLoRA(cfg).to(dev)
    params = optimizer_params(model)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(TensorDataset(torch.arange(len(train_features))), batch_size=cfg.batch_size, shuffle=True)
    best_val = math.inf
    history: list[dict[str, Any]] = []
    ckpt_dir = out / "checkpoints"
    adapter_dir = ckpt_dir / "lora_adapter"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses: list[float] = []
        started = time.time()
        for (idxs,) in loader:
            idx_list = [int(i) for i in idxs.tolist()]
            batch = batch_from_features(train_features, idx_list, dev)
            prompts = build_prompts(train_features, "clean", idx_list)
            caps = [train_features.captions[i] for i in idx_list]
            loss = model(batch, prompts, caps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for start in range(0, len(val_features), max(1, cfg.batch_size)):
                idx_list = list(range(start, min(start + max(1, cfg.batch_size), len(val_features))))
                batch = batch_from_features(val_features, idx_list, dev)
                prompts = build_prompts(val_features, "clean", idx_list)
                caps = [val_features.captions[i] for i in idx_list]
                val_losses.append(float(model(batch, prompts, caps).detach().cpu()))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
            "seconds": float(time.time() - started),
            "trainable_params": int(sum(p.numel() for p in params)),
        }
        history.append(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            model.llm.save_pretrained(adapter_dir)
            torch.save(
                {
                    "prefix": {
                        "query_tokens": model.query_tokens.detach().cpu(),
                        "conf_token": model.conf_token.state_dict(),
                        "corr_embed": model.corr_embed.state_dict(),
                        "resampler": model.resampler.state_dict(),
                        "prefix_projector": model.prefix_projector.state_dict(),
                    },
                    "config": asdict(cfg),
                    "best_val_loss": best_val,
                    "uses_enhanced_visual_tokens": True,
                    "generator_type": "QwenVisualTokenPrefixLoRA",
                },
                ckpt_dir / "best_prefix.pt",
            )
    write_json(out / "history.json", history)
    lines = [
        "# EVG1B Qwen LoRA Training Summary",
        "",
        f"- best val loss: `{best_val:.6f}`",
        f"- adapter dir: `{adapter_dir}`",
        "",
        "| Epoch | Train Loss | Val Loss | Seconds | Trainable Params |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in history:
        lines.append(f"| {row['epoch']} | {row['train_loss']:.6f} | {row['val_loss']:.6f} | {row['seconds']:.2f} | {row['trainable_params']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ckpt_dir / "best_prefix.pt"


def load_trained(cfg: EVG1BLoRAConfig, ckpt: Path, dev: torch.device) -> QwenVisualTokenPrefixLoRA:
    model = QwenVisualTokenPrefixLoRA(cfg).to(dev)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    prefix = payload["prefix"]
    model.query_tokens.data.copy_(prefix["query_tokens"].to(model.query_tokens.device))
    model.conf_token.load_state_dict(prefix["conf_token"])
    model.corr_embed.load_state_dict(prefix["corr_embed"])
    model.resampler.load_state_dict(prefix["resampler"])
    model.prefix_projector.load_state_dict(prefix["prefix_projector"])
    from peft import PeftModel

    adapter_dir = ckpt.parent / "lora_adapter"
    model.llm = PeftModel.from_pretrained(model.llm.base_model.model, adapter_dir).to(dev)
    model.eval()
    return model


def summarize(records: list[dict[str, Any]], corruption: str, mode: str, file_name: str) -> dict[str, Any]:
    valid = [bool(r["valid"]) for r in records]
    hits = [float(r["class_hit"]) for r in records]
    return {
        "file": file_name,
        "corruption": corruption,
        "mode": mode,
        "count": len(records),
        "valid_caption_rate": float(np.mean(valid)) if valid else 0.0,
        "invalid_output_rate": 1.0 - float(np.mean(valid)) if valid else 1.0,
        "caption_class_hit": float(np.mean(hits)) if hits else 0.0,
        "avg_caption_length": float(np.mean([r["length"] for r in records])) if records else 0.0,
        "distinct_caption_count": len({str(r["generated_caption"]) for r in records}),
    }


def add_gaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_corr: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_corr[str(row["corruption"])][str(row["mode"])] = row
    for row in rows:
        modes = by_corr[str(row["corruption"])]
        real = float(modes.get("real_eeg", {}).get("caption_class_hit", 0.0))
        row["real_minus_shuffled"] = real - float(modes.get("shuffled_eeg", {}).get("caption_class_hit", 0.0))
        row["real_minus_random"] = real - float(modes.get("random_eeg", {}).get("caption_class_hit", 0.0))
        row["real_minus_vision"] = real - float(modes.get("vision_only", {}).get("caption_class_hit", 0.0))
    return rows


def evaluate(cfg: EVG1BLoRAConfig, ckpt: Path) -> None:
    dev = device()
    model = load_trained(cfg, ckpt, dev)
    vtg = to_vtg_cfg(cfg)
    src_cfg = source_cfg(vtg)
    _labels, _protos, class_name_map = load_label_bank(src_cfg, dev)
    out_root = Path(cfg.output_dir) / "EVG1B"
    eval_dir = out_root / "EVG1B_lora_r8_prefix_semantic_prompt_seed42" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for corruption in CORRUPTIONS:
        for mode in MODES:
            features = build_vision_token_features(vtg, "test", mode, corruption)
            preds: list[str] = []
            with torch.no_grad():
                for start in range(0, len(features), cfg.eval_batch_size):
                    idxs = list(range(start, min(start + cfg.eval_batch_size, len(features))))
                    batch = batch_from_features(features, idxs, dev)
                    prompts = build_prompts(features, corruption, idxs)
                    preds.extend(model.generate(batch, prompts, cfg.max_new_tokens))
            records: list[dict[str, Any]] = []
            for idx, (row, pred) in enumerate(zip(features.rows, preds, strict=False)):
                label = int(row["label"])
                true_class = class_name_map.get(label, str(label))
                ok, reason = valid_caption(pred)
                rec = {
                    "model": "EVG1B_lora_r8_prefix_semantic_prompt",
                    "image_id": str(row["image_id"]),
                    "true_class": true_class,
                    "corruption": corruption,
                    "mode": mode,
                    "top5_classes": features.top5_names[idx],
                    "generated_caption": pred,
                    "valid": ok,
                    "invalid_reason": reason,
                    "class_hit": class_hit(pred, true_class),
                    "length": len(str(pred).split()),
                    "token_input_shape": list(features.batch.visual_tokens.shape[1:]),
                    "uses_enhanced_visual_tokens": True,
                    "qwen_lora": True,
                }
                records.append(rec)
            with (eval_dir / f"{corruption}_{mode}.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            all_records.extend(records)
            metrics.append(summarize(records, corruption, mode, f"{corruption}_{mode}.jsonl"))
    metrics = add_gaps(metrics)
    write_table(metrics, out_root / "EVG1B_METRICS.csv", out_root / "EVG1B_METRICS.md", "EVG1B Qwen LoRA Metrics")
    write_examples(all_records, out_root / "EVG1B_QUALITATIVE_EXAMPLES.md")
    write_report(cfg, metrics, all_records, ckpt)


def write_examples(records: list[dict[str, Any]], path: Path) -> None:
    lines = ["# EVG1B Qwen LoRA Qualitative Examples", "", "| image_id | true class | corruption | mode | generated caption | valid | class hit |", "| --- | --- | --- | --- | --- | --- | ---: |"]
    preferred = [r for r in records if r["mode"] == "real_eeg" and r["corruption"] in STRONG_CORRUPTIONS]
    if len(preferred) < 30:
        preferred = records
    for record in preferred[:120]:
        cap = str(record["generated_caption"]).replace("|", "/")
        lines.append(f"| {record['image_id']} | {record['true_class']} | {record['corruption']} | {record['mode']} | {cap} | {record['valid']} | {float(record['class_hit']):.1f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(cfg: EVG1BLoRAConfig, metrics: list[dict[str, Any]], records: list[dict[str, Any]], ckpt: Path) -> None:
    real = [r for r in metrics if r["mode"] == "real_eeg"]
    strong = [r for r in real if r["corruption"] in STRONG_CORRUPTIONS]
    valid_rate = float(np.mean([float(r["valid_caption_rate"]) for r in real])) if real else 0.0
    invalid_rate = float(np.mean([float(r["invalid_output_rate"]) for r in real])) if real else 1.0
    hit = float(np.mean([float(r["caption_class_hit"]) for r in strong])) if strong else 0.0
    gap_s = float(np.mean([float(r["real_minus_shuffled"]) for r in strong])) if strong else 0.0
    gap_r = float(np.mean([float(r["real_minus_random"]) for r in strong])) if strong else 0.0
    out = Path(cfg.output_dir) / "EVG1B" / "EVG1B_QWEN_LORA_REPORT.md"
    lines = [
        "# EVG1B Qwen LoRA Report",
        "",
        "- status: `completed subset LoRA run`",
        f"- checkpoint prefix: `{ckpt}`",
        f"- Qwen model: `{cfg.llm_model}`",
        f"- LoRA r/alpha/dropout: `{cfg.lora_r}/{cfg.lora_alpha}/{cfg.lora_dropout}`",
        f"- target modules: `{', '.join(cfg.target_modules)}`",
        f"- train/val/test sample caps: `{cfg.max_train_samples}/{cfg.max_val_samples}/{cfg.max_test_samples}`",
        "- enhanced visual tokens fed into Qwen: `yes, via prefix inputs_embeds projected from [B,50,512] tokens`",
        f"- real EEG valid caption rate: `{valid_rate:.6f}`",
        f"- real EEG invalid output rate: `{invalid_rate:.6f}`",
        f"- strong-corruption real EEG class-hit: `{hit:.6f}`",
        f"- strong-corruption real-shuffled gap: `{gap_s:.6f}`",
        f"- strong-corruption real-random gap: `{gap_r:.6f}`",
        f"- generated records: `{len(records)}`",
        "",
        "This run resolves the previous dependency blocker for the LoRA path. It is a subset run and should not replace the full EVG1 direct-token decoder unless expanded to all test samples.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_final_report(cfg: EVG1BLoRAConfig) -> None:
    final = Path(cfg.output_dir) / "FINAL_VISION_TOKEN_GENERATIVE_EVLM_REPORT.md"
    text = final.read_text(encoding="utf-8") if final.exists() else "# Final Vision Token Generative EVLM Report\n"
    text = text.replace(
        "- Did LoRA help? `not established; dependency probe blocked by missing peft/bitsandbytes, see outputs/vision_token_gen_evlm/EVG1/EVG1B_LORA_BLOCKED_REPORT.md`",
        "- Did LoRA help? `attempted after installing peft/bitsandbytes; subset Qwen-LoRA run completed, but not promoted over full EVG1 because it is subset-sized`",
    )
    text = text.replace(
        "- Qwen LoRA could not be promoted without the `peft` dependency or after prior Qwen soft-prefix instability.",
        "- Qwen LoRA dependency blocker was resolved and a subset run completed; it remains exploratory until expanded to full test coverage.",
    )
    marker = "\n## EVG1B Qwen LoRA Follow-up\n"
    addition = (
        marker +
        "\n"
        "- EVG1B LoRA dependency blocker was revisited after installing `peft`, `accelerate`, and `bitsandbytes`.\n"
        "- A Qwen2.5-1.5B LoRA subset run now exists under `outputs/vision_token_gen_evlm/EVG1B/`.\n"
        "- Enhanced visual tokens are projected into Qwen prefix `inputs_embeds`; this is an actual LLM+LoRA attempt, but subset-sized and not promoted over EVG1.\n"
        "- See `outputs/vision_token_gen_evlm/EVG1B/EVG1B_QWEN_LORA_REPORT.md`.\n"
    )
    if marker in text:
        text = text.split(marker)[0].rstrip() + addition
    else:
        text = text.rstrip() + "\n" + addition
    final.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EVG1B Qwen LoRA over enhanced vision tokens.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=512)
    parser.add_argument("--max_val_samples", type=int, default=128)
    parser.add_argument("--max_test_samples", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EVG1BLoRAConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        max_new_tokens=args.max_new_tokens,
    )
    ckpt = train(cfg)
    evaluate(cfg, ckpt)
    refresh_final_report(cfg)


if __name__ == "__main__":
    main()
