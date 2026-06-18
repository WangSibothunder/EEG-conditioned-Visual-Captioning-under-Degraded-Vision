from __future__ import annotations

import argparse
from collections.abc import Callable
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
from torch.utils.data import DataLoader

from src.data.clip_cache import MissingImagesError, validate_manifest_images
from src.data.collate import caption_collate
from src.data.dataset import EEGVisionCaptionDataset
from src.models.vision_encoder import FrozenCLIPVisionEncoder


class NativeCLIPVisionEncoder(nn.Module):
    """Frozen vision encoder that keeps the model's native output dimension."""

    def __init__(
        self,
        model_name: str,
        *,
        use_tiny_debug_model: bool = False,
        fallback_dim: int = 512,
        vision_model_loader: Callable[[str], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.using_fallback = True
        self.encoder: nn.Module
        self.vision_family = "clip"
        self.register_buffer(
            "_clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False,
        )

        if use_tiny_debug_model:
            self.encoder = FrozenCLIPVisionEncoder(
                model_name=model_name,
                output_dim=fallback_dim,
                use_tiny_debug_model=True,
            )
        else:
            try:
                self.encoder = self._load_native_vision_model(model_name, vision_model_loader)
                self.using_fallback = False
            except Exception:
                self.encoder = FrozenCLIPVisionEncoder(
                    model_name=model_name,
                    output_dim=fallback_dim,
                    use_tiny_debug_model=True,
                )

        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _load_native_vision_model(
        self,
        model_name: str,
        vision_model_loader: Callable[[str], nn.Module] | None,
    ) -> nn.Module:
        if "siglip" in model_name.lower():
            self.vision_family = "siglip"
            if vision_model_loader is not None:
                return vision_model_loader(model_name)
            from transformers import SiglipVisionModel

            return SiglipVisionModel.from_pretrained(model_name)

        self.vision_family = "clip"
        if vision_model_loader is not None:
            return vision_model_loader(model_name)
        from transformers import CLIPVisionModelWithProjection

        return CLIPVisionModelWithProjection.from_pretrained(model_name)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"images must have shape [B, 3, H, W], got {tuple(images.shape)}")
        if self.using_fallback:
            return self.encoder(images)

        images = images.float()
        pixel_values = self._normalize_inputs(images)
        output = self.encoder(pixel_values=pixel_values)
        if getattr(output, "image_embeds", None) is not None:
            return output.image_embeds
        if getattr(output, "pooler_output", None) is not None:
            return output.pooler_output
        return output.last_hidden_state[:, 0]

    def _normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        if self.vision_family == "siglip":
            return (images - 0.5) / 0.5
        return (images - self._clip_mean.to(images.device, images.dtype)) / self._clip_std.to(
            images.device, images.dtype
        )


def precompute_vision_cache(
    manifest: str | Path,
    out: str | Path,
    index_out: str | Path,
    *,
    image_root: str | Path | None = None,
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
) -> dict[str, Any]:
    manifest = Path(manifest)
    out = Path(out)
    index_out = Path(index_out)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    if out.exists() and index_out.exists() and not overwrite:
        array = np.load(out)
        with index_out.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        return {
            "model_name": model_name,
            "num_images": int(array.shape[0]),
            "unique_images": len({str(item.get("image_id")) for item in index}),
            "embedding_shape": list(array.shape),
            "dtype": str(array.dtype),
            "missing_images": 0,
            "cache_file_size": out.stat().st_size,
            "out": str(out),
            "index_out": str(index_out),
            "using_tiny_debug_model": False,
            "skipped_existing": True,
        }

    image_stats = validate_manifest_images(
        manifest,
        image_root=image_root,
        report_path="outputs/missing_vision_images.md",
    )

    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

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
        model = NativeCLIPVisionEncoder(
            model_name,
            use_tiny_debug_model=use_tiny_debug_model,
        )
    else:
        model = encoder_factory(model_name=model_name, use_tiny_debug_model=use_tiny_debug_model)
    model = model.to(torch_device)
    model.eval()
    if require_real_model and bool(getattr(model, "using_fallback", False)):
        raise RuntimeError(
            f"Requested real vision model for {model_name}, but the encoder is using a fallback model."
        )

    all_embeddings: list[torch.Tensor] = []
    index: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(torch_device)
            emb = model(images)
            if emb.ndim != 2:
                raise ValueError(f"vision encoder must return [B, D], got {tuple(emb.shape)}")
            emb = F.normalize(emb.float(), dim=-1)
            all_embeddings.append(emb.cpu())
            labels = batch.get("label")
            subject_ids = batch.get("subject_id", [None] * len(batch["image_id"]))
            for row_idx, (image_id, caption) in enumerate(zip(batch["image_id"], batch["caption"], strict=False)):
                entry: dict[str, Any] = {"image_id": image_id, "caption": caption}
                if labels is not None:
                    entry["label"] = int(labels[row_idx])
                if subject_ids:
                    entry["subject_id"] = subject_ids[row_idx]
                source_row = dataset.rows[len(index)]
                if "eeg_index" in source_row:
                    entry["eeg_index"] = int(source_row["eeg_index"])
                index.append(entry)

    embeddings = torch.cat(all_embeddings, dim=0).numpy().astype(np.float16)
    out.parent.mkdir(parents=True, exist_ok=True)
    index_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, embeddings)
    with index_out.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)

    return {
        "model_name": model_name,
        "num_images": int(embeddings.shape[0]),
        "unique_images": len({str(item.get("image_id")) for item in index}),
        "embedding_shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "missing_images": 0,
        "validated_images": image_stats["num_images"],
        "cache_file_size": out.stat().st_size,
        "out": str(out),
        "index_out": str(index_out),
        "using_tiny_debug_model": bool(getattr(model, "using_fallback", False)),
        "num_workers": loader_kwargs["num_workers"],
        "prefetch_factor": loader_kwargs.get("prefetch_factor", 0),
        "skipped_existing": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute normalized CLIP image embeddings.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image_root", default=None, help="Kept for CLI compatibility; manifest paths define roots.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--index_out", required=True)
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
    parser.add_argument("--report", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        stats = precompute_vision_cache(
            args.manifest,
            args.out,
            args.index_out,
            image_root=args.image_root,
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
    except MissingImagesError as exc:
        print(f"ERROR: {exc}")
        print(
            json.dumps(
                {
                    "manifest": str(exc.manifest),
                    "missing_images": len(exc.missing),
                    "example_missing_images": [str(path) for path in exc.missing[:10]],
                    "report": str(exc.report_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(2) from None
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2) from None

    print(json.dumps(stats, indent=2, sort_keys=True))
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "\n".join(
                [
                    "# CLIP Cache Report",
                    "",
                    f"- Manifest: `{args.manifest}`",
                    f"- CLIP model name: `{stats['model_name']}`",
                    f"- Output: `{args.out}`",
                    f"- Index: `{args.index_out}`",
                    f"- Images processed: `{stats['num_images']}`",
                    f"- Unique images: `{stats['unique_images']}`",
                    f"- Embedding shape: `{stats['embedding_shape']}`",
                    f"- dtype: `{stats['dtype']}`",
                    f"- Missing images: `{stats['missing_images']}`",
                    f"- Cache file size: `{stats['cache_file_size']}`",
                    f"- Tiny fallback used: `{stats['using_tiny_debug_model']}`",
                    f"- Skipped existing cache: `{stats['skipped_existing']}`",
                ]
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
