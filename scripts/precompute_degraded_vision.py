from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collate import caption_collate
from src.data.corruptions import apply_corruption
from src.data.dataset import EEGVisionCaptionDataset
from scripts.precompute_vision import NativeCLIPVisionEncoder


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _write_index(path: Path, dataset: EEGVisionCaptionDataset, batch: dict[str, Any], start: int) -> list[dict[str, Any]]:
    labels = batch.get("label")
    subject_ids = batch.get("subject_id", [None] * len(batch["image_id"]))
    entries: list[dict[str, Any]] = []
    for offset, (image_id, caption) in enumerate(zip(batch["image_id"], batch["caption"], strict=False)):
        source_row = dataset.rows[start + offset]
        entry: dict[str, Any] = {
            "image_id": image_id,
            "caption": caption,
        }
        if labels is not None:
            entry["label"] = int(labels[offset])
        if subject_ids:
            entry["subject_id"] = subject_ids[offset]
        if "eeg_index" in source_row:
            entry["eeg_index"] = int(source_row["eeg_index"])
        entries.append(entry)
    return entries


def _existing_stats(out: Path, index_out: Path, corruption: str, model_name: str) -> dict[str, Any]:
    array = np.load(out)
    with index_out.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    return {
        "corruption": corruption,
        "model_name": model_name,
        "num_images": int(array.shape[0]),
        "unique_images": len({str(item.get("image_id")) for item in index}),
        "embedding_shape": list(array.shape),
        "dtype": str(array.dtype),
        "cache_file_size": out.stat().st_size,
        "out": str(out),
        "index_out": str(index_out),
        "skipped_existing": True,
    }


def precompute_degraded_clip_caches(
    *,
    manifest: str | Path,
    out_dir: str | Path,
    corruptions: list[str],
    batch_size: int = 16,
    image_size: int = 224,
    eeg_shape: tuple[int, int] = (64, 250),
    model_name: str = "openai/clip-vit-base-patch32",
    use_tiny_debug_model: bool = False,
    device: str = "auto",
    overwrite: bool = False,
    require_real_model: bool = False,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    encoder_factory: Callable[..., nn.Module] | None = None,
) -> list[dict[str, Any]]:
    manifest = Path(manifest)
    out_dir = Path(out_dir)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    torch_device = _resolve_device(device)
    dataset = EEGVisionCaptionDataset(manifest, image_size=image_size, eeg_shape=eeg_shape)
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": max(0, int(num_workers)),
        "collate_fn": caption_collate,
        "pin_memory": torch.cuda.is_available(),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)
    if encoder_factory is None:
        model = NativeCLIPVisionEncoder(model_name, use_tiny_debug_model=use_tiny_debug_model)
    else:
        model = encoder_factory(model_name=model_name, use_tiny_debug_model=use_tiny_debug_model)
    model = model.to(torch_device)
    model.eval()
    if require_real_model and bool(getattr(model, "using_fallback", False)):
        raise RuntimeError(
            f"Requested real vision model for {model_name}, but the encoder is using a fallback model."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    stats: list[dict[str, Any]] = []
    for corruption in corruptions:
        out = out_dir / f"clip_test_{corruption}.npy"
        index_out = out_dir / f"clip_index_test_{corruption}.json"
        if out.exists() and index_out.exists() and not overwrite:
            stats.append(_existing_stats(out, index_out, corruption, model_name))
            continue

        all_embeddings: list[torch.Tensor] = []
        index: list[dict[str, Any]] = []
        seen = 0
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(torch_device)
                images = apply_corruption(images, corruption)
                emb = model(images)
                emb = F.normalize(emb.float(), dim=-1)
                all_embeddings.append(emb.cpu())
                index.extend(_write_index(index_out, dataset, batch, seen))
                seen += len(batch["image_id"])

        embeddings = torch.cat(all_embeddings, dim=0).numpy().astype(np.float16)
        np.save(out, embeddings)
        with index_out.open("w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2)
        stats.append(
            {
                "corruption": corruption,
                "model_name": model_name,
                "num_images": int(embeddings.shape[0]),
                "unique_images": len({str(item.get("image_id")) for item in index}),
                "embedding_shape": list(embeddings.shape),
                "dtype": str(embeddings.dtype),
                "cache_file_size": out.stat().st_size,
                "out": str(out),
                "index_out": str(index_out),
                "using_tiny_debug_model": bool(model.using_fallback),
                "num_workers": loader_kwargs["num_workers"],
                "prefetch_factor": loader_kwargs.get("prefetch_factor", 0),
                "skipped_existing": False,
            }
        )
    return stats


def write_report(path: str | Path, *, manifest: str | Path, stats: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Degraded CLIP Cache Report", "", f"- Manifest: `{manifest}`", ""]
    if stats:
        headers = [
            "corruption",
            "model_name",
            "num_images",
            "unique_images",
            "embedding_shape",
            "dtype",
            "cache_file_size",
            "out",
            "index_out",
            "skipped_existing",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in stats:
            lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    else:
        lines.append("No corruptions requested.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute CLIP caches for degraded test images.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image_root", default=None, help="Kept for command compatibility; manifest paths define roots.")
    parser.add_argument("--corruptions", nargs="+", default=["clean", "blur", "occlusion", "noise", "lowres"])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--eeg_channels", type=int, default=64)
    parser.add_argument("--eeg_timesteps", type=int, default=250)
    parser.add_argument("--model_name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--use_tiny_debug_model", action="store_true")
    parser.add_argument("--require_real_model", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", default="outputs/day3/degraded_clip_cache_report.md")
    args = parser.parse_args()

    stats = precompute_degraded_clip_caches(
        manifest=args.manifest,
        out_dir=args.out_dir,
        corruptions=args.corruptions,
        batch_size=args.batch_size,
        image_size=args.image_size,
        eeg_shape=(args.eeg_channels, args.eeg_timesteps),
        model_name=args.model_name,
        use_tiny_debug_model=args.use_tiny_debug_model,
        device=args.device,
        overwrite=args.overwrite,
        require_real_model=args.require_real_model,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    write_report(args.report, manifest=args.manifest, stats=stats)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
