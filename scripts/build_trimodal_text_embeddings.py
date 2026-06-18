from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


class CaptionDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("image_id", "caption"):
                if key not in row:
                    raise ValueError(f"{path}:{line_number} missing required key: {key}")
            rows.append(row)
    return rows


def _caption_files(data_dir: Path, splits: list[str], sources: list[str]) -> list[Path]:
    files: list[Path] = []
    for split in splits:
        for source in sources:
            path = data_dir / f"{split}_{source}_caption.jsonl"
            if path.exists():
                files.append(path)
    if not files:
        raise FileNotFoundError(f"No caption JSONL files found under {data_dir} for splits={splits} sources={sources}")
    return files


def collect_caption_rows(data_dir: str | Path, splits: list[str], sources: list[str]) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for path in _caption_files(data_dir, splits, sources):
        stem = path.name.removesuffix("_caption.jsonl")
        split, source = stem.split("_", 1)
        for row_number, row in enumerate(_read_jsonl(path)):
            entry = dict(row)
            entry["split"] = str(entry.get("split") or split)
            entry["caption_source"] = str(entry.get("caption_source") or source)
            entry["caption_file"] = str(path)
            entry["caption_row"] = row_number
            key = (str(entry["split"]), str(entry["caption_source"]), str(entry["image_id"]), int(row_number))
            if key not in seen:
                seen.add(key)
                rows.append(entry)
    return rows


def _collate_captions(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "caption": [str(item["caption"]) for item in batch],
        "rows": batch,
    }


class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        *,
        use_tiny_debug_model: bool = False,
        fallback_dim: int = 512,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.using_fallback = True
        self.tokenizer: Any | None = None
        self.text_model: nn.Module | None = None
        self.fallback_dim = fallback_dim

        if not use_tiny_debug_model:
            try:
                from transformers import CLIPTextModelWithProjection, CLIPTokenizer

                self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
                self.text_model = CLIPTextModelWithProjection.from_pretrained(model_name)
                self.using_fallback = False
            except Exception:
                self.tokenizer = None
                self.text_model = None

        if self.text_model is not None:
            for parameter in self.text_model.parameters():
                parameter.requires_grad_(False)

    def forward(self, captions: list[str], device: torch.device) -> torch.Tensor:
        if self.using_fallback or self.text_model is None or self.tokenizer is None:
            return self._fallback_encode(captions, device)
        tokens = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        output = self.text_model(**tokens)
        if getattr(output, "text_embeds", None) is not None:
            return output.text_embeds
        if getattr(output, "pooler_output", None) is not None:
            return output.pooler_output
        return output.last_hidden_state[:, 0]

    def _fallback_encode(self, captions: list[str], device: torch.device) -> torch.Tensor:
        # 中文注释：离线 smoke 路径使用确定性哈希 bag-of-words，避免下载模型阻塞集成测试。
        vectors = torch.zeros((len(captions), self.fallback_dim), dtype=torch.float32, device=device)
        for row_idx, caption in enumerate(captions):
            for token in caption.lower().replace(",", " ").replace(".", " ").split():
                bucket = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16) % self.fallback_dim
                vectors[row_idx, bucket] += 1.0
        return vectors


def build_text_embedding_cache(
    *,
    data_dir: str | Path = "data/thought2text",
    out: str | Path = "data/thought2text/cache/text_embeddings.npy",
    index_out: str | Path = "data/thought2text/cache/text_index.json",
    report: str | Path = "outputs/trimodal/text_embedding_report.md",
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 64,
    device: str = "auto",
    splits: list[str] | None = None,
    sources: list[str] | None = None,
    overwrite: bool = False,
    use_tiny_debug_model: bool = False,
    require_real_model: bool = False,
) -> dict[str, Any]:
    splits = splits or ["train", "val", "test"]
    sources = sources or ["human", "blip"]
    out = Path(out)
    index_out = Path(index_out)
    report = Path(report)
    if out.exists() and index_out.exists() and not overwrite:
        array = np.load(out)
        with index_out.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        stats = _stats(index, array, model_name, out, index_out, skipped_existing=True, using_fallback=False)
        _write_report(report, stats)
        return stats

    rows = collect_caption_rows(data_dir, splits, sources)
    torch_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    encoder = CLIPTextEncoder(model_name, use_tiny_debug_model=use_tiny_debug_model).to(torch_device)
    encoder.eval()
    if require_real_model and encoder.using_fallback:
        raise RuntimeError(f"Requested real CLIP text model for {model_name}, but fallback encoder is active.")

    loader = DataLoader(CaptionDataset(rows), batch_size=batch_size, shuffle=False, collate_fn=_collate_captions)
    embeddings: list[torch.Tensor] = []
    index: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            emb = encoder(batch["caption"], torch_device)
            if emb.ndim != 2:
                raise ValueError(f"text encoder must return [B, D], got {tuple(emb.shape)}")
            emb = F.normalize(emb.float(), dim=-1)
            embeddings.append(emb.cpu())
            for row in batch["rows"]:
                index.append(
                    {
                        "image_id": str(row["image_id"]),
                        "caption": str(row["caption"]),
                        "caption_source": str(row.get("caption_source", "")),
                        "split": str(row.get("split", "")),
                        "label": int(row["label"]) if row.get("label") is not None else None,
                        "subject_id": row.get("subject_id"),
                        "eeg_index": int(row["eeg_index"]) if row.get("eeg_index") is not None else None,
                        "image_index": int(row["image_index"]) if row.get("image_index") is not None else None,
                        "caption_file": str(row.get("caption_file", "")),
                        "caption_row": int(row.get("caption_row", len(index))),
                    }
                )

    array = torch.cat(embeddings, dim=0).numpy().astype(np.float16)
    out.parent.mkdir(parents=True, exist_ok=True)
    index_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, array)
    with index_out.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)
    stats = _stats(index, array, model_name, out, index_out, skipped_existing=False, using_fallback=encoder.using_fallback)
    _write_report(report, stats)
    return stats


def _stats(
    index: list[dict[str, Any]],
    array: np.ndarray,
    model_name: str,
    out: Path,
    index_out: Path,
    *,
    skipped_existing: bool,
    using_fallback: bool,
) -> dict[str, Any]:
    by_split = Counter(str(item.get("split", "")) for item in index)
    by_source = Counter(str(item.get("caption_source", "")) for item in index)
    return {
        "model_name": model_name,
        "num_texts": int(array.shape[0]),
        "unique_images": len({str(item.get("image_id")) for item in index}),
        "embedding_shape": list(array.shape),
        "dtype": str(array.dtype),
        "by_split": dict(sorted(by_split.items())),
        "by_source": dict(sorted(by_source.items())),
        "out": str(out),
        "index_out": str(index_out),
        "cache_file_size": out.stat().st_size if out.exists() else 0,
        "using_tiny_debug_model": bool(using_fallback),
        "skipped_existing": bool(skipped_existing),
    }


def _write_report(path: Path, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tri-Modal Text Embedding Report",
        "",
        f"- CLIP text model: `{stats['model_name']}`",
        f"- Output: `{stats['out']}`",
        f"- Index: `{stats['index_out']}`",
        f"- Text rows: `{stats['num_texts']}`",
        f"- Unique images: `{stats['unique_images']}`",
        f"- Embedding shape: `{stats['embedding_shape']}`",
        f"- dtype: `{stats['dtype']}`",
        f"- Tiny fallback used: `{stats['using_tiny_debug_model']}`",
        f"- Skipped existing cache: `{stats['skipped_existing']}`",
        "",
        "## Rows By Split",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in stats["by_split"].items())
    lines.extend(["", "## Rows By Source", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in stats["by_source"].items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CLIP text embeddings for Thought2Text human/BLIP captions.")
    parser.add_argument("--data_dir", default="data/thought2text")
    parser.add_argument("--out", default="data/thought2text/cache/text_embeddings.npy")
    parser.add_argument("--index_out", default="data/thought2text/cache/text_index.json")
    parser.add_argument("--report", default="outputs/trimodal/text_embedding_report.md")
    parser.add_argument("--model_name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--sources", default="human,blip")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_tiny_debug_model", action="store_true")
    parser.add_argument("--require_real_model", action="store_true")
    args = parser.parse_args()

    stats = build_text_embedding_cache(
        data_dir=args.data_dir,
        out=args.out,
        index_out=args.index_out,
        report=args.report,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        splits=[part.strip() for part in args.splits.split(",") if part.strip()],
        sources=[part.strip() for part in args.sources.split(",") if part.strip()],
        overwrite=args.overwrite,
        use_tiny_debug_model=args.use_tiny_debug_model,
        require_real_model=args.require_real_model,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
