from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class _ByteTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 259

    def __call__(self, texts: Sequence[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        return_tensors = kwargs.get("return_tensors", "pt")
        padding = bool(kwargs.get("padding", True))
        truncation = bool(kwargs.get("truncation", True))
        max_length = kwargs.get("max_length")
        if return_tensors != "pt":
            raise ValueError("tiny tokenizer only supports return_tensors='pt'")
        encoded = [self._encode(text, max_length=max_length, truncation=truncation) for text in texts]
        width = max(len(ids) for ids in encoded) if padding else None
        if max_length is not None and padding:
            width = max_length
        rows = []
        masks = []
        for ids in encoded:
            row = ids if width is None else ids[:width] + [self.pad_token_id] * max(0, width - len(ids))
            rows.append(row)
            masks.append([0 if token == self.pad_token_id else 1 for token in row])
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def batch_decode(self, token_ids: torch.Tensor | list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        if isinstance(token_ids, torch.Tensor):
            rows = token_ids.detach().cpu().tolist()
        else:
            rows = token_ids
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in rows]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        values: list[int] = []
        for token_id in token_ids:
            token = int(token_id)
            if token == self.eos_token_id:
                break
            if skip_special_tokens and token in {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            }:
                continue
            if token >= 3:
                values.append((token - 3) % 256)
        return bytes(values).decode("utf-8", errors="ignore").strip()

    def _encode(self, text: str, *, max_length: int | None, truncation: bool) -> list[int]:
        tokens = [self.bos_token_id] + [byte + 3 for byte in text.encode("utf-8")] + [self.eos_token_id]
        if max_length is not None and truncation:
            tokens = tokens[:max_length]
            tokens[-1] = self.eos_token_id
        return tokens


class _TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 259, hidden_size: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=layers, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, *, input_ids: torch.Tensor | None = None, inputs_embeds: torch.Tensor | None = None, **_: Any) -> Any:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = self.embed_tokens(input_ids)
        hidden, _ = self.rnn(inputs_embeds)
        return SimpleNamespace(logits=self.lm_head(hidden))


class SoftPromptCaptionModel(nn.Module):
    """Map a 512-d image/fused embedding to soft prompts for a frozen causal LM."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        input_dim: int | None = None,
        prompt_tokens: int | None = None,
        max_text_length: int | None = None,
        model_name: str | None = None,
        use_tiny_debug_model: bool | None = None,
        freeze_lm: bool = True,
    ) -> None:
        super().__init__()
        cfg = dict(config or {})
        self.input_dim = int(input_dim or cfg.get("image_dim", 512))
        self.prompt_tokens = int(prompt_tokens or cfg.get("prompt_tokens", 8))
        self.max_text_length = int(max_text_length or cfg.get("max_text_length", 64))
        self.model_name = model_name or str(cfg.get("llm_model", "Qwen/Qwen2.5-1.5B-Instruct"))
        self.using_tiny_lm = bool(cfg.get("use_tiny_debug_model", False))
        if use_tiny_debug_model is not None:
            self.using_tiny_lm = use_tiny_debug_model

        self.tokenizer, self.lm = self._load_lm()
        hidden_size = int(getattr(self.lm.config, "hidden_size"))
        self.prompt_projector = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_size * self.prompt_tokens),
        )
        self._init_projector(hidden_size)

        if freeze_lm:
            self.freeze_lm()

    def _load_lm(self) -> tuple[Any, nn.Module]:
        if self.using_tiny_lm:
            tokenizer = _ByteTokenizer()
            return tokenizer, _TinyCausalLM(tokenizer.vocab_size)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer, AutoModelForCausalLM.from_pretrained(self.model_name)
        except Exception:
            self.using_tiny_lm = True
            tokenizer = _ByteTokenizer()
            return tokenizer, _TinyCausalLM(tokenizer.vocab_size)

    def _init_projector(self, hidden_size: int) -> None:
        linear = self.prompt_projector[-1]
        if isinstance(linear, nn.Linear):
            nn.init.normal_(linear.weight, mean=0.0, std=hidden_size**-0.5)
            nn.init.zeros_(linear.bias)

    def freeze_lm(self) -> None:
        for parameter in self.lm.parameters():
            parameter.requires_grad_(False)

    def unfreeze_lm(self) -> None:
        for parameter in self.lm.parameters():
            parameter.requires_grad_(True)

    def encode_captions(self, captions: Sequence[str], device: torch.device) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(list(captions), return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_length)
        return {key: value.to(device) for key, value in tokens.items()}

    def soft_prompt(self, conditioning_emb: torch.Tensor) -> torch.Tensor:
        if conditioning_emb.ndim != 2 or conditioning_emb.shape[-1] != self.input_dim:
            raise ValueError(
                f"conditioning_emb must have shape [B, {self.input_dim}], got {tuple(conditioning_emb.shape)}"
            )
        hidden_size = int(getattr(self.lm.config, "hidden_size"))
        prompt = self.prompt_projector(conditioning_emb)
        return prompt.view(conditioning_emb.shape[0], self.prompt_tokens, hidden_size)

    def forward(self, conditioning_emb: torch.Tensor, captions: Sequence[str]) -> torch.Tensor:
        tokens = self.encode_captions(captions, conditioning_emb.device)
        input_ids = tokens["input_ids"]
        attention_mask = tokens.get("attention_mask", torch.ones_like(input_ids))
        token_embeds = self.lm.get_input_embeddings()(input_ids)
        prompt_embeds = self.soft_prompt(conditioning_emb).to(dtype=token_embeds.dtype)
        inputs_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)
        prompt_mask = torch.ones(input_ids.shape[0], self.prompt_tokens, dtype=attention_mask.dtype, device=input_ids.device)
        full_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        outputs = self.lm(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask)
        ignore = torch.full((input_ids.shape[0], self.prompt_tokens), -100, dtype=torch.long, device=input_ids.device)
        token_labels = input_ids.masked_fill(attention_mask == 0, -100)
        return self._causal_loss(outputs.logits, torch.cat([ignore, token_labels], dim=1))

    def _causal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), ignore_index=-100)

    def _start_token_id(self) -> int:
        return int(
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.eos_token_id
        )

    @torch.no_grad()
    def generate(
        self,
        conditioning_emb: torch.Tensor,
        *,
        max_new_tokens: int = 24,
        temperature: float = 0.0,
    ) -> list[str]:
        self.eval()
        batch_size = conditioning_emb.shape[0]
        device = conditioning_emb.device
        generated = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        embed_dtype = self.lm.get_input_embeddings().weight.dtype

        for _ in range(max_new_tokens):
            prompt_embeds = self.soft_prompt(conditioning_emb).to(dtype=embed_dtype)
            if generated.shape[1] > 0:
                token_embeds = self.lm.get_input_embeddings()(generated)
                inputs_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)
            else:
                inputs_embeds = prompt_embeds
            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.long,
                device=device,
            )
            logits = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
            next_logits = logits[:, -1, :]
            if temperature and temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if self.tokenizer.eos_token_id is not None and torch.all(
                next_token.squeeze(1) == int(self.tokenizer.eos_token_id)
            ):
                break

        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
