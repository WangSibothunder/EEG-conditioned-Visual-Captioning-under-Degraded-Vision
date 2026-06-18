from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.collate import caption_collate
from src.data.corruptions import apply_corruption
from src.data.dataset import EEGVisionCaptionDataset
from src.models.caption_model import SoftPromptCaptionModel
from src.models.eeg_encoder import EEGEncoder
from src.models.fusion import GatedFusion
from src.models.vision_encoder import FrozenCLIPVisionEncoder
from src.train.train_fusion import apply_eeg_mode, resolve_device
from src.utils.checkpoint import load_checkpoint
from src.utils.config import deep_update, load_config


def load_sanity_config(config_path: str, checkpoint_path: str | None) -> dict:
    cfg = load_config(config_path)
    if checkpoint_path and Path(checkpoint_path).exists():
        state = load_checkpoint(checkpoint_path, map_location="cpu")
        checkpoint_cfg = state.get("config")
        if isinstance(checkpoint_cfg, dict):
            cfg = deep_update(cfg, checkpoint_cfg)
    return cfg


def build_prediction_record(
    *,
    image_id: str,
    corruption: str,
    mode: str,
    reference: str,
    prediction: str,
    label: int | None,
    gate_mean: float | None,
    human_label_name: str | None = None,
) -> dict:
    record = {
        "image_id": image_id,
        "corruption": corruption,
        "mode": mode,
        "reference": reference,
        "prediction": prediction,
    }
    if label is not None:
        record["label"] = int(label)
    if human_label_name:
        record["human_label_name"] = str(human_label_name)
    if gate_mean is not None:
        record["gate_mean"] = float(gate_mean)
    return record


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "y"}


def load_degraded_clip_cache(
    cache_dir: str | Path,
    corruption: str,
    *,
    expected_len: int,
    max_samples: int = 0,
) -> torch.Tensor:
    cache_path = Path(cache_dir) / f"clip_test_{corruption}.npy"
    if not cache_path.exists():
        raise FileNotFoundError(f"degraded CLIP cache not found: {cache_path}")
    embeddings = torch.from_numpy(np.load(cache_path)).float()
    if embeddings.ndim != 2:
        raise ValueError(f"degraded CLIP cache must have shape [N, D], got {tuple(embeddings.shape)} for {cache_path}")
    if embeddings.shape[0] != expected_len:
        raise ValueError(
            f"degraded CLIP cache length mismatch for {corruption}: "
            f"cache has {embeddings.shape[0]}, manifest has {expected_len}"
        )
    if max_samples and max_samples > 0:
        embeddings = embeddings[:max_samples]
    return embeddings


def run_mode(config: str, checkpoint: str, mode: str, output_dir: Path) -> None:
    import sys

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "generate",
            "--config",
            config,
            "--checkpoint",
            checkpoint,
            "--mode",
            mode,
            "--output",
            str(output_dir / f"{mode}.jsonl"),
        ]
        generate_main()
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--caption_ckpt", default=None)
    parser.add_argument("--eeg_ckpt", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--modes", nargs="+", default=None)
    parser.add_argument("--corruptions", nargs="+", default=None)
    parser.add_argument("--use_degraded_clip_cache", default=None)
    parser.add_argument("--degraded_cache_dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--output-dir", default="outputs/debug/sanity")
    args = parser.parse_args()

    if args.manifest or args.modes or args.corruptions or args.out:
        run_real_sanity(args)
        return

    checkpoint = args.checkpoint or args.caption_ckpt
    if checkpoint is None:
        raise SystemExit("ERROR: --checkpoint is required for debug sanity mode")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ["image_only", "real_eeg", "shuffled_eeg", "random_eeg"]:
        run_mode(args.config, checkpoint, mode, output_dir)
    print(f"Saved sanity outputs to {output_dir}")


def run_real_sanity(args: argparse.Namespace) -> None:
    checkpoint = args.caption_ckpt or args.checkpoint
    cfg = load_sanity_config(args.config, checkpoint)
    manifest = Path(args.manifest or cfg["data"]["root"]) 
    if args.manifest is None:
        manifest = Path(cfg["data"]["root"]) / cfg["data"]["val_manifest"]
    if not manifest.exists():
        print(f"ERROR: manifest not found: {manifest}")
        print("If using Thought2Text, run `bash scripts/inspect_thought2text.sh` and `bash scripts/build_thought2text_manifest.sh` first.")
        raise SystemExit(2)

    if checkpoint is None or not Path(checkpoint).exists():
        print(f"ERROR: caption checkpoint not found: {checkpoint}")
        print("Run fusion training first or pass --caption_ckpt / --checkpoint.")
        raise SystemExit(2)

    modes = args.modes or ["vision_only", "real_eeg", "shuffled_eeg", "random_eeg", "eeg_only"]
    corruptions = args.corruptions or ["clean"]
    output_dir = Path(args.out or args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_degraded_cache = _as_bool(args.use_degraded_clip_cache)

    device = resolve_device(cfg.get("device", "auto"))
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    dataset = EEGVisionCaptionDataset(
        manifest,
        image_size=int(data_cfg.get("image_size", 224)),
        eeg_shape=(int(data_cfg.get("eeg_channels", 64)), int(data_cfg.get("eeg_timesteps", 250))),
        allow_missing_images=use_degraded_cache,
    )
    if args.max_samples and args.max_samples > 0:
        dataset.rows = dataset.rows[: args.max_samples]
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"].get("batch_size", 2)),
        shuffle=False,
        num_workers=0,
        collate_fn=caption_collate,
    )

    degraded_caches: dict[str, torch.Tensor] = {}
    if use_degraded_cache:
        if not args.degraded_cache_dir:
            raise SystemExit("ERROR: --degraded_cache_dir is required when --use_degraded_clip_cache is true")
        original_len = len(dataset.rows) if not (args.max_samples and args.max_samples > 0) else int(args.max_samples)
        full_manifest_len = sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
        for corruption in corruptions:
            degraded_caches[corruption] = load_degraded_clip_cache(
                args.degraded_cache_dir,
                corruption,
                expected_len=full_manifest_len,
                max_samples=original_len,
            )

    vision = None if use_degraded_cache else FrozenCLIPVisionEncoder(model_cfg).to(device)
    eeg_encoder = EEGEncoder(
        channels=int(data_cfg.get("eeg_channels", 64)),
        timesteps=int(data_cfg.get("eeg_timesteps", 250)),
        output_dim=int(model_cfg.get("eeg_dim", 512)),
    ).to(device)
    fusion = GatedFusion(dim=int(model_cfg.get("image_dim", 512))).to(device)
    caption_model = SoftPromptCaptionModel(model_cfg).to(device)

    state = load_checkpoint(checkpoint, map_location=device)
    caption_model.load_state_dict(state.get("caption_model", state.get("model", {})), strict=False)
    if "eeg_encoder" in state:
        eeg_encoder.load_state_dict(state["eeg_encoder"], strict=False)
    if "fusion" in state:
        fusion.load_state_dict(state["fusion"], strict=False)
    if args.eeg_ckpt and Path(args.eeg_ckpt).exists():
        eeg_state = load_checkpoint(args.eeg_ckpt, map_location=device)
        eeg_encoder.load_state_dict(eeg_state.get("eeg_encoder", eeg_state.get("model", {})), strict=False)

    if vision is not None:
        vision.eval()
    eeg_encoder.eval()
    fusion.eval()
    caption_model.eval()

    with torch.no_grad():
        for corruption in corruptions:
            for requested_mode in modes:
                mode = "image_only" if requested_mode == "vision_only" else requested_mode
                path = output_dir / f"{corruption}_{requested_mode}.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    seen = 0
                    for batch in loader:
                        eeg = batch["eeg"].to(device)
                        if corruption in degraded_caches:
                            start = seen
                            stop = seen + len(batch["image_id"])
                            image_emb = degraded_caches[corruption][start:stop].to(device)
                        else:
                            images = apply_corruption(batch["image"].to(device), corruption)
                            if vision is None:
                                raise RuntimeError(f"No vision encoder available for uncached corruption: {corruption}")
                            image_emb = vision(images)
                        if requested_mode == "eeg_only":
                            image_emb = torch.zeros_like(image_emb)
                            mode_eeg = eeg
                        else:
                            mode_eeg = apply_eeg_mode(eeg, mode)
                        gate_mean_by_row: list[float | None]
                        if mode_eeg is None:
                            fused_emb = image_emb
                            gate_mean_by_row = [None] * image_emb.shape[0]
                        else:
                            eeg_emb = eeg_encoder(mode_eeg)
                            gate_values = fusion.gate_values(image_emb, eeg_emb)
                            gate_mean_by_row = [
                                float(value.detach().cpu())
                                for value in gate_values.float().mean(dim=1)
                            ]
                            fused_emb = fusion(image_emb, eeg_emb)
                        predictions = caption_model.generate(
                            fused_emb,
                            max_new_tokens=int(cfg["generation"].get("max_new_tokens", 16)),
                        )
                        labels = batch.get("label")
                        for row_index, (image_id, reference, prediction) in enumerate(
                            zip(batch["image_id"], batch["caption"], predictions, strict=False)
                        ):
                            label = int(labels[row_index]) if labels is not None else None
                            source_row = dataset.rows[seen + row_index]
                            human_label_name = source_row.get("human_label_name")
                            record = build_prediction_record(
                                image_id=image_id,
                                corruption=corruption,
                                mode=requested_mode,
                                reference=reference,
                                prediction=prediction,
                                label=label,
                                gate_mean=gate_mean_by_row[row_index],
                                human_label_name=human_label_name,
                            )
                            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                        seen += len(batch["image_id"])
                print(f"Saved {path}")


if __name__ == "__main__":
    main()
