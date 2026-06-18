from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    class _TqdmFallback:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs) -> None:
            return None

    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return _TqdmFallback(iterable, **kwargs)

from src.data.collate import caption_collate
from src.data.clip_cache import load_cache
from src.data.dataset import EEGVisionCaptionDataset
from src.models.caption_model import SoftPromptCaptionModel
from src.models.eeg_encoder import EEGEncoder
from src.models.fusion import GatedFusion
from src.models.vision_encoder import FrozenCLIPVisionEncoder
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config import load_config
from src.utils.logger import log_jsonl, print_rank0
from src.utils.seed import seed_everything


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_output_dir(cfg: dict[str, Any]) -> Path:
    output_dir = Path(cfg["output_dir"])
    return output_dir if bool(cfg.get("exact_output_dir", False)) else output_dir / "fusion"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.train_manifest is not None:
        cfg["data"]["train_manifest"] = args.train_manifest
    if args.val_manifest is not None:
        cfg["data"]["val_manifest"] = args.val_manifest
    if args.root is not None:
        cfg["data"]["root"] = args.root
    if args.clip_train_cache is not None:
        cfg["data"]["clip_train_cache"] = args.clip_train_cache
    if args.clip_val_cache is not None:
        cfg["data"]["clip_val_cache"] = args.clip_val_cache
    if args.eeg_ckpt is not None:
        cfg["eeg_ckpt"] = args.eeg_ckpt
    if args.llm is not None:
        cfg["model"]["llm_model"] = args.llm
        cfg["model"]["use_tiny_debug_model"] = False
        cfg["model"]["require_real_lm"] = True
        cfg["train"]["max_steps"] = 0
    if args.freeze_llm is not None:
        cfg["model"]["freeze_lm"] = _as_bool(args.freeze_llm)
    if args.freeze_eeg_encoder is not None:
        cfg["train"]["freeze_eeg_encoder"] = _as_bool(args.freeze_eeg_encoder)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.grad_accum_steps is not None:
        cfg["train"]["grad_accum_steps"] = args.grad_accum_steps
    if getattr(args, "max_steps", None) is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if getattr(args, "max_val_batches", None) is not None:
        cfg["train"]["max_val_batches"] = args.max_val_batches
    if getattr(args, "train_mode", None) is not None:
        cfg["train"]["mode"] = args.train_mode
    if args.bf16 is not None:
        cfg["train"]["amp"] = _as_bool(args.bf16)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
        cfg["exact_output_dir"] = True
    return cfg


def load_alignment_eeg_encoder(
    eeg_encoder: EEGEncoder,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    state = checkpoint.get("eeg_encoder", checkpoint.get("model", checkpoint))
    eeg_encoder.load_state_dict(state, strict=False)


class CachedFusionDataset(Dataset[dict[str, Any]]):
    def __init__(self, dataset: EEGVisionCaptionDataset, clip_embeddings: torch.Tensor) -> None:
        if len(dataset) != clip_embeddings.shape[0]:
            raise ValueError(f"Dataset/cache length mismatch: {len(dataset)} vs {clip_embeddings.shape[0]}")
        self.dataset = dataset
        self.clip_embeddings = clip_embeddings.float()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        item["clip_emb"] = self.clip_embeddings[index]
        return item


def fusion_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = caption_collate(batch)
    if "clip_emb" in batch[0]:
        out["clip_emb"] = torch.stack([item["clip_emb"] for item in batch], dim=0).float()
    return out


def build_loader(cfg: dict, split: str, shuffle: bool) -> DataLoader:
    data_cfg = cfg["data"]
    manifest = data_cfg["train_manifest"] if split == "train" else data_cfg["val_manifest"]
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute() and not manifest_path.exists():
        manifest_path = Path(data_cfg["root"]) / manifest_path
    dataset = EEGVisionCaptionDataset(
        manifest_path=manifest_path,
        image_size=data_cfg["image_size"],
        eeg_shape=(int(data_cfg["eeg_channels"]), int(data_cfg["eeg_timesteps"])),
    )
    cache_key = "clip_train_cache" if split == "train" else "clip_val_cache"
    index_key = "clip_index_train" if split == "train" else "clip_index_val"
    cache_path = data_cfg.get(cache_key)
    if cache_path:
        index_path = data_cfg.get(index_key)
        if index_path is None:
            path = Path(cache_path)
            index_path = path.with_name(path.name.replace("clip_", "clip_index_").replace(".npy", ".json"))
        clip_embeddings, _ = load_cache(cache_path, index_path)
        dataset = CachedFusionDataset(dataset, clip_embeddings)
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=0,
        collate_fn=fusion_collate,
    )


def normalize_eeg_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized == "vision_only":
        return "image_only"
    return normalized


def caption_checkpoint_state(caption_model: SoftPromptCaptionModel) -> dict[str, torch.Tensor]:
    """Save trainable caption adapter weights without duplicating frozen LLM weights."""
    if any(parameter.requires_grad for parameter in caption_model.lm.parameters()):
        return caption_model.state_dict()
    return {
        name: value.detach().cpu()
        for name, value in caption_model.state_dict().items()
        if name.startswith("prompt_projector.")
    }


def apply_eeg_mode(eeg: torch.Tensor, mode: str) -> torch.Tensor | None:
    mode = normalize_eeg_mode(mode)
    if mode == "image_only":
        return None
    if mode == "real_eeg":
        return eeg
    if mode == "shuffled_eeg":
        return eeg[torch.randperm(eeg.shape[0], device=eeg.device)]
    if mode == "random_eeg":
        return torch.randn_like(eeg)
    raise ValueError(f"Unsupported EEG mode: {mode}")


def generate_samples(
    vision: FrozenCLIPVisionEncoder,
    eeg_encoder: EEGEncoder,
    fusion: GatedFusion,
    caption_model: SoftPromptCaptionModel,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    mode: str,
    max_new_tokens: int,
    limit: int,
) -> None:
    vision.eval()
    eeg_encoder.eval()
    fusion.eval()
    caption_model.eval()
    seen = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            eeg = batch["eeg"].to(device)
            image_emb = vision(images)
            mode_eeg = apply_eeg_mode(eeg, mode)
            if mode_eeg is None:
                fused_emb = fusion.image_only(image_emb)
            else:
                eeg_emb = eeg_encoder(mode_eeg)
                fused_emb = fusion(image_emb, eeg_emb)
            predictions = caption_model.generate(fused_emb, max_new_tokens=max_new_tokens)
            for image_id, reference, prediction in zip(
                batch["image_id"], batch["caption"], predictions, strict=False
            ):
                log_jsonl(
                    output_path,
                    {
                        "image_id": image_id,
                        "mode": mode,
                        "reference": reference,
                        "prediction": prediction,
                    },
                )
                seen += 1
                if seen >= limit:
                    return


def train(config_path: str | None = None, cfg_override: dict | None = None) -> None:
    cfg = cfg_override if cfg_override is not None else load_config(config_path or "configs/debug.yaml")
    seed_everything(int(cfg["seed"]))
    torch.backends.cudnn.benchmark = True

    device = resolve_device(cfg.get("device", "auto"))
    output_dir = resolve_output_dir(cfg)
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_loader(cfg, "train", shuffle=True)
    val_loader = build_loader(cfg, "val", shuffle=False)

    model_cfg = cfg["model"]
    vision = FrozenCLIPVisionEncoder(model_cfg).to(device)
    eeg_encoder = EEGEncoder(
        channels=int(cfg["data"]["eeg_channels"]),
        timesteps=int(cfg["data"]["eeg_timesteps"]),
        output_dim=int(model_cfg["eeg_dim"]),
    ).to(device)
    fusion = GatedFusion(dim=int(model_cfg["image_dim"])).to(device)
    caption_model = SoftPromptCaptionModel(model_cfg, freeze_lm=bool(model_cfg.get("freeze_lm", True))).to(device)
    if bool(model_cfg.get("require_real_lm", False)) and caption_model.using_tiny_lm:
        raise RuntimeError(f"Failed to load required real LM: {model_cfg.get('llm_model')}")

    eeg_ckpt = cfg.get("eeg_ckpt") or cfg.get("data", {}).get("eeg_ckpt")
    if eeg_ckpt:
        load_alignment_eeg_encoder(eeg_encoder, eeg_ckpt, device)
        print_rank0(f"Loaded aligned EEG encoder from {eeg_ckpt}")

    vision.eval()
    if bool(model_cfg.get("freeze_lm", True)):
        caption_model.freeze_lm()
    if bool(cfg["train"].get("freeze_eeg_encoder", False)):
        eeg_encoder.eval()
        for parameter in eeg_encoder.parameters():
            parameter.requires_grad_(False)
    trainable = (
        [p for p in eeg_encoder.parameters() if p.requires_grad]
        + list(fusion.parameters())
        + [p for p in caption_model.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    max_steps = int(cfg["train"].get("max_steps", 0))
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    train_mode = str(cfg["train"].get("mode", "real_eeg"))
    global_step = 0
    best_val_loss = float("inf")

    eeg_encoder.train()
    fusion.train()
    caption_model.train()
    if bool(cfg["train"].get("freeze_eeg_encoder", False)):
        eeg_encoder.eval()
    for epoch in range(int(cfg["train"]["epochs"])):
        progress = tqdm(train_loader, desc=f"fusion epoch {epoch + 1}")
        optimizer.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(progress, start=1):
            eeg = batch["eeg"].to(device)
            captions = batch["caption"]

            with torch.no_grad():
                if "clip_emb" in batch:
                    image_emb = batch["clip_emb"].to(device)
                else:
                    image_emb = vision(batch["image"].to(device))

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                mode_eeg = apply_eeg_mode(eeg, train_mode)
                if mode_eeg is None:
                    fused_emb = fusion.image_only(image_emb)
                elif bool(cfg["train"].get("freeze_eeg_encoder", False)):
                    with torch.no_grad():
                        eeg_emb = eeg_encoder(mode_eeg)
                    fused_emb = fusion(image_emb, eeg_emb)
                else:
                    eeg_emb = eeg_encoder(mode_eeg)
                    fused_emb = fusion(image_emb, eeg_emb)
                loss = caption_model(fused_emb, captions)
                scaled_loss = loss / grad_accum_steps

            scaled_loss.backward()

            if micro_step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.set_postfix(loss=f"{loss.item():.4f}")
                log_jsonl(output_dir / "train_log.jsonl", {"step": global_step, "loss": loss.item()})

                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite fusion loss")
                if max_steps and global_step >= max_steps:
                    break

        checkpoint_path = output_dir / "checkpoint_last.pt"
        last_checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "checkpoint_format": "fusion_compact_caption_v1",
            "caption_model": caption_checkpoint_state(caption_model),
            "eeg_encoder": eeg_encoder.state_dict(),
            "fusion": fusion.state_dict(),
            "config": cfg,
        }
        save_checkpoint(
            checkpoint_path,
            last_checkpoint,
        )
        save_checkpoint(ckpt_dir / "last.pt", last_checkpoint)
        val_loss = evaluate_caption_loss(
            vision,
            eeg_encoder,
            fusion,
            caption_model,
            val_loader,
            device,
            amp_dtype,
            use_amp,
            bool(cfg["train"].get("freeze_eeg_encoder", False)),
            int(cfg["train"].get("max_val_batches", 0)),
            train_mode,
        )
        log_jsonl(output_dir / "val_log.jsonl", {"epoch": epoch + 1, "val_loss": val_loss})
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = dict(last_checkpoint)
            best_checkpoint["val_loss"] = val_loss
            save_checkpoint(ckpt_dir / "best.pt", best_checkpoint)
        generate_samples(
            vision,
            eeg_encoder,
            fusion,
            caption_model,
            val_loader,
            device,
            output_dir / "samples.jsonl",
            train_mode,
            int(cfg["generation"]["max_new_tokens"]),
            int(cfg["generation"]["num_samples"]),
        )
        if max_steps and global_step >= max_steps:
            break

    print_rank0(f"Saved fusion checkpoint to {output_dir / 'checkpoint_last.pt'}")


def evaluate_caption_loss(
    vision: FrozenCLIPVisionEncoder,
    eeg_encoder: EEGEncoder,
    fusion: GatedFusion,
    caption_model: SoftPromptCaptionModel,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    freeze_eeg_encoder: bool,
    max_batches: int = 0,
    mode: str = "real_eeg",
) -> float:
    vision.eval()
    eeg_encoder.eval()
    fusion.eval()
    caption_model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            eeg = batch["eeg"].to(device)
            if "clip_emb" in batch:
                image_emb = batch["clip_emb"].to(device)
            else:
                image_emb = vision(batch["image"].to(device))
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                mode_eeg = apply_eeg_mode(eeg, mode)
                if mode_eeg is None:
                    fused_emb = fusion.image_only(image_emb)
                else:
                    eeg_emb = eeg_encoder(mode_eeg)
                    fused_emb = fusion(image_emb, eeg_emb)
                loss = caption_model(fused_emb, batch["caption"])
            losses.append(float(loss.detach().cpu()))
            if max_batches and batch_index >= max_batches:
                break
    if not freeze_eeg_encoder:
        eeg_encoder.train()
    fusion.train()
    caption_model.train()
    return float(np.mean(losses)) if losses else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument("--train_manifest", default=None)
    parser.add_argument("--val_manifest", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--clip_train_cache", default=None)
    parser.add_argument("--clip_val_cache", default=None)
    parser.add_argument("--eeg_ckpt", default=None)
    parser.add_argument("--llm", default=None)
    parser.add_argument("--freeze_llm", default=None)
    parser.add_argument("--freeze_eeg_encoder", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument(
        "--train_mode",
        choices=["vision_only", "real_eeg", "shuffled_eeg", "random_eeg"],
        default=None,
    )
    parser.add_argument("--bf16", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    if any(
        value is not None
        for value in [
            args.train_manifest,
            args.val_manifest,
            args.root,
            args.llm,
            args.epochs,
            args.batch_size,
            args.output_dir,
        ]
    ):
        cfg = apply_cli_overrides(load_config(args.config), args)
        train(None, cfg_override=cfg)
    else:
        train(args.config)


if __name__ == "__main__":
    main()
