from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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
from src.data.dataset import EEGVisionCaptionDataset
from src.models.caption_model import SoftPromptCaptionModel
from src.models.vision_encoder import FrozenCLIPVisionEncoder
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config
from src.utils.logger import log_jsonl, print_rank0
from src.utils.seed import seed_everything


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_loader(cfg: dict, split: str, shuffle: bool) -> DataLoader:
    data_cfg = cfg["data"]
    manifest = data_cfg["train_manifest"] if split == "train" else data_cfg["val_manifest"]
    dataset = EEGVisionCaptionDataset(
        manifest_path=Path(data_cfg["root"]) / manifest,
        image_size=data_cfg["image_size"],
        eeg_shape=(int(data_cfg["eeg_channels"]), int(data_cfg["eeg_timesteps"])),
    )
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=0,
        collate_fn=caption_collate,
    )


def generate_samples(
    vision: FrozenCLIPVisionEncoder,
    caption_model: SoftPromptCaptionModel,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    max_new_tokens: int,
    limit: int,
) -> None:
    vision.eval()
    caption_model.eval()
    seen = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            image_emb = vision(images)
            predictions = caption_model.generate(image_emb, max_new_tokens=max_new_tokens)
            for image_id, reference, prediction in zip(
                batch["image_id"], batch["caption"], predictions, strict=False
            ):
                log_jsonl(
                    output_path,
                    {
                        "image_id": image_id,
                        "mode": "image_only",
                        "reference": reference,
                        "prediction": prediction,
                    },
                )
                seen += 1
                if seen >= limit:
                    return


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(int(cfg["seed"]))
    torch.backends.cudnn.benchmark = True

    device = resolve_device(cfg.get("device", "auto"))
    output_dir = Path(cfg["output_dir"]) / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_loader(cfg, "train", shuffle=True)
    val_loader = build_loader(cfg, "val", shuffle=False)

    model_cfg = cfg["model"]
    vision = FrozenCLIPVisionEncoder(model_cfg).to(device)
    caption_model = SoftPromptCaptionModel(model_cfg).to(device)

    vision.eval()
    caption_model.freeze_lm()
    trainable = [p for p in caption_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    max_steps = int(cfg["train"].get("max_steps", 0))
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    global_step = 0

    caption_model.train()
    for epoch in range(int(cfg["train"]["epochs"])):
        progress = tqdm(train_loader, desc=f"baseline epoch {epoch + 1}")
        optimizer.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(progress, start=1):
            images = batch["image"].to(device)
            captions = batch["caption"]

            with torch.no_grad():
                image_emb = vision(images)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = caption_model(image_emb, captions)
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
                    raise RuntimeError("Non-finite baseline loss")
                if max_steps and global_step >= max_steps:
                    break

        checkpoint_path = output_dir / "checkpoint_last.pt"
        save_checkpoint(
            checkpoint_path,
            {
                "epoch": epoch,
                "global_step": global_step,
                "caption_model": caption_model.state_dict(),
                "config": cfg,
            },
        )
        generate_samples(
            vision,
            caption_model,
            val_loader,
            device,
            output_dir / "samples.jsonl",
            int(cfg["generation"]["max_new_tokens"]),
            int(cfg["generation"]["num_samples"]),
        )
        if max_steps and global_step >= max_steps:
            break

    print_rank0(f"Saved baseline checkpoint to {output_dir / 'checkpoint_last.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
