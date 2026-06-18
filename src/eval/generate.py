from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.collate import caption_collate
from src.data.dataset import EEGVisionCaptionDataset
from src.models.caption_model import SoftPromptCaptionModel
from src.models.eeg_encoder import EEGEncoder
from src.models.fusion import GatedFusion
from src.models.vision_encoder import FrozenCLIPVisionEncoder
from src.train.train_fusion import apply_eeg_mode, generate_samples, resolve_device
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--mode",
        choices=["image_only", "real_eeg", "shuffled_eeg", "random_eeg"],
        default="image_only",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    device = resolve_device(cfg.get("device", "auto"))
    model_cfg = cfg["model"]

    dataset = EEGVisionCaptionDataset(
        manifest_path=Path(cfg["data"]["root"]) / cfg["data"]["val_manifest"],
        image_size=cfg["data"]["image_size"],
        eeg_shape=(int(cfg["data"]["eeg_channels"]), int(cfg["data"]["eeg_timesteps"])),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=caption_collate,
    )

    vision = FrozenCLIPVisionEncoder(model_cfg).to(device)
    eeg_encoder = EEGEncoder(
        channels=int(cfg["data"]["eeg_channels"]),
        timesteps=int(cfg["data"]["eeg_timesteps"]),
        output_dim=int(model_cfg["eeg_dim"]),
    ).to(device)
    fusion = GatedFusion(dim=int(model_cfg["image_dim"])).to(device)
    caption_model = SoftPromptCaptionModel(model_cfg).to(device)

    state = load_checkpoint(args.checkpoint, map_location=device)
    if "caption_model" in state:
        caption_model.load_state_dict(state["caption_model"], strict=False)
    if "eeg_encoder" in state:
        eeg_encoder.load_state_dict(state["eeg_encoder"], strict=False)
    if "fusion" in state:
        fusion.load_state_dict(state["fusion"], strict=False)

    output = Path(args.output) if args.output else Path(cfg["output_dir"]) / "generation" / f"{args.mode}.jsonl"
    if output.exists():
        output.unlink()
    generate_samples(
        vision,
        eeg_encoder,
        fusion,
        caption_model,
        loader,
        device,
        output,
        args.mode,
        int(cfg["generation"]["max_new_tokens"]),
        int(cfg["generation"]["num_samples"]),
    )
    print(f"Saved generations to {output}")


if __name__ == "__main__":
    main()
