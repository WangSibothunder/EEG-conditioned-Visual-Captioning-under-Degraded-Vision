from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import EEGVisionCaptionDataset
from src.models.masked_eeg_autoencoder import (
    MaskedEEGAutoencoder,
    count_parameters,
    make_time_channel_mask,
    masked_eeg_loss,
)
from src.utils.seed import seed_everything


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_eeg_cache(
    *,
    manifest: str | Path,
    out_path: str | Path,
    eeg_shape: tuple[int, int],
    force: bool = False,
) -> Path:
    manifest = Path(manifest)
    out_path = Path(out_path)
    rows = _read_jsonl(manifest)
    if out_path.exists() and not force:
        arr = np.load(out_path, mmap_mode="r")
        if arr.shape == (len(rows), eeg_shape[0], eeg_shape[1]):
            return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = EEGVisionCaptionDataset(manifest, eeg_shape=eeg_shape, allow_missing_images=True)
    cache = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), eeg_shape[0], eeg_shape[1]),
    )
    for idx, row in enumerate(dataset.rows):
        cache[idx] = dataset._load_row_eeg(row).numpy()
        if (idx + 1) % 1000 == 0:
            print(json.dumps({"cache": str(out_path), "rows": idx + 1, "total": len(dataset)}), flush=True)
    cache.flush()
    return out_path


def resolve_eeg_cache(
    *,
    split_name: str,
    data_cfg: dict[str, Any],
    cache_key: str,
    manifest_key: str,
    eeg_shape: tuple[int, int],
) -> Path:
    cache_path = Path(data_cfg[cache_key])
    manifest_value = data_cfg.get(manifest_key)
    if manifest_value:
        return build_eeg_cache(
            manifest=manifest_value,
            out_path=cache_path,
            eeg_shape=eeg_shape,
            force=bool(data_cfg.get("force_rebuild_cache", False)),
        )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{split_name} EEG cache does not exist and `{manifest_key}` was not provided: {cache_path}"
        )
    arr = np.load(cache_path, mmap_mode="r")
    if arr.ndim != 3 or tuple(arr.shape[1:]) != eeg_shape:
        raise ValueError(f"{split_name} EEG cache expected [N, {eeg_shape[0]}, {eeg_shape[1]}], got {arr.shape}")
    return cache_path


class CachedEEGDataset(Dataset[torch.Tensor]):
    def __init__(self, cache_path: str | Path, max_samples: int = 0) -> None:
        self.path = Path(cache_path)
        self.eeg = np.load(self.path, mmap_mode="r")
        self.length = int(self.eeg.shape[0] if max_samples <= 0 else min(max_samples, self.eeg.shape[0]))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.eeg[index], dtype=np.float32).copy())


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    import yaml

    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _candidate_batches(initial: int, target: int) -> list[int]:
    values = sorted({128, int(initial), 256, 512, int(target), 1024}, reverse=True)
    return [value for value in values if value > 0]


def _select_batch_size(model: torch.nn.Module, cfg: dict[str, Any], device: torch.device) -> tuple[int, int]:
    train_cfg = cfg["train"]
    model.train()
    candidates = _candidate_batches(int(train_cfg.get("initial_batch_size", 256)), int(train_cfg.get("target_effective_batch_size", 512)))
    channels = int(cfg["data"].get("eeg_channels", 64))
    timesteps = int(cfg["data"].get("eeg_timesteps", 250))
    use_amp = bool(train_cfg.get("bf16", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    for batch_size in candidates:
        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
            dummy = torch.randn(batch_size, channels, timesteps, device=device)
            mask = make_time_channel_mask(
                dummy,
                mask_ratio_time=float(train_cfg.get("mask_ratio_time", 0.35)),
                mask_ratio_channel=float(train_cfg.get("mask_ratio_channel", 0.15)),
            )
            masked = dummy.masked_fill(mask, 0.0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                recon = model(masked)
                loss, _ = masked_eeg_loss(
                    recon,
                    dummy,
                    mask,
                    lambda_spectral=float(train_cfg.get("lambda_spectral", 0.2)),
                    lambda_smoothness=float(train_cfg.get("lambda_smoothness", 0.1)),
                )
            loss.backward()
            optimizer.zero_grad(set_to_none=True)
            del dummy, mask, masked, recon, loss, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
            effective = max(batch_size, int(train_cfg.get("target_effective_batch_size", batch_size)))
            accum = max(1, int(np.ceil(effective / batch_size)))
            return batch_size, accum
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(json.dumps({"event": "oom_batch_search", "batch_size": batch_size, "error": str(exc)[:200]}), flush=True)
    return 64, max(1, int(np.ceil(int(train_cfg.get("target_effective_batch_size", 512)) / 64)))


def _make_loader(cache_path: Path, batch_size: int, num_workers: int, shuffle: bool, max_samples: int, device: torch.device) -> DataLoader:
    return DataLoader(
        CachedEEGDataset(cache_path, max_samples=max_samples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def train(config: dict[str, Any]) -> None:
    seed_everything(int(config.get("seed", 42)))
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    out_dir = Path(config.get("output", {}).get("dir", "outputs/pretrain/masked_eeg"))
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(out_dir / "config.yaml", config)

    eeg_shape = (int(data_cfg.get("eeg_channels", 64)), int(data_cfg.get("eeg_timesteps", 250)))
    data_cfg.setdefault("train_eeg_cache", "data/thought2text/cache/eeg_pretrain_train.npy")
    data_cfg.setdefault("val_eeg_cache", "data/thought2text/cache/eeg_pretrain_val.npy")
    train_cache = resolve_eeg_cache(
        split_name="train",
        data_cfg=data_cfg,
        cache_key="train_eeg_cache",
        manifest_key="train_manifest",
        eeg_shape=eeg_shape,
    )
    val_cache = resolve_eeg_cache(
        split_name="val",
        data_cfg=data_cfg,
        cache_key="val_eeg_cache",
        manifest_key="val_manifest",
        eeg_shape=eeg_shape,
    )

    device = _device(str(config.get("device", "auto")))
    model = MaskedEEGAutoencoder(
        channels=eeg_shape[0],
        timesteps=eeg_shape[1],
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        layers=int(model_cfg.get("layers", 8)),
        heads=int(model_cfg.get("heads", 8)),
        ffn_dim=int(model_cfg.get("ffn_dim", 2048)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        spatial_layers=int(model_cfg.get("spatial_layers", 2)),
        variant=str(model_cfg.get("variant", "dualbranch")),
    ).to(device)

    if train_cfg.get("batch_size", "auto") == "auto":
        batch_size, grad_accum_steps = _select_batch_size(model, config, device)
    else:
        batch_size = int(train_cfg["batch_size"])
        grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    config["train"]["resolved_batch_size"] = batch_size
    config["train"]["resolved_grad_accum_steps"] = grad_accum_steps
    config["train"]["effective_batch_size"] = batch_size * grad_accum_steps
    _write_yaml(out_dir / "config.yaml", config)

    num_workers = int(train_cfg.get("num_workers", 8))
    train_loader = _make_loader(train_cache, batch_size, num_workers, True, int(train_cfg.get("max_train_samples", 0)), device)
    val_loader = _make_loader(val_cache, batch_size, num_workers, False, int(train_cfg.get("max_val_samples", 0)), device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        betas=(0.9, 0.95),
        eps=1e-10,
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    epochs = int(train_cfg.get("epochs", 200))
    patience = int(train_cfg.get("patience", 25))
    log_every = int(train_cfg.get("log_every", 10))
    use_amp = bool(train_cfg.get("bf16", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    global_step = 0
    step_times: list[float] = []
    log_path = out_dir / "train.log"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for micro_step, eeg in enumerate(train_loader, start=1):
            step_start = time.time()
            eeg = eeg.to(device, non_blocking=True)
            mask = make_time_channel_mask(
                eeg,
                mask_ratio_time=float(train_cfg.get("mask_ratio_time", 0.35)),
                mask_ratio_channel=float(train_cfg.get("mask_ratio_channel", 0.15)),
                span=int(train_cfg.get("mask_span", 12)),
            )
            masked = eeg.masked_fill(mask, 0.0)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                recon = model(masked)
                loss, parts = masked_eeg_loss(
                    recon,
                    eeg,
                    mask,
                    lambda_spectral=float(train_cfg.get("lambda_spectral", 0.2)),
                    lambda_smoothness=float(train_cfg.get("lambda_smoothness", 0.1)),
                )
                scaled = loss / grad_accum_steps
            scaled.backward()
            if micro_step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                losses.append(float(loss.detach().cpu()))
                if device.type == "cuda":
                    torch.cuda.synchronize()
                step_time = time.time() - step_start
                step_times.append(step_time)
                if global_step % max(1, log_every) == 0:
                    payload = {
                        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "epoch": epoch,
                        "step": global_step,
                        "batch_size": batch_size,
                        "effective_batch_size": batch_size * grad_accum_steps,
                        "step_time": step_time,
                        **parts,
                    }
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(payload) + "\n")
                    print(json.dumps(payload), flush=True)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite masked EEG pretraining loss")

        val_loss = evaluate(model, val_loader, config, device, amp_dtype, use_amp)
        train_loss = float(np.mean(losses)) if losses else 0.0
        epoch_time = time.time() - epoch_start
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_seconds": epoch_time,
            "avg_step_time": float(np.mean(step_times[-max(1, len(train_loader)) :])) if step_times else 0.0,
            "gpu_mem_peak_mb": peak_mem,
        }
        history.append(row)
        (out_dir / "metrics.json").write_text(json.dumps({"history": history, "best_val_loss": best_val}, indent=2), encoding="utf-8")
        save_payload = {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_loss": val_loss,
            "param_count": count_parameters(model),
        }
        torch.save(save_payload, ckpt_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(save_payload, ckpt_dir / "best_masked_eeg.pt")
        print(json.dumps(row), flush=True)
        if patience > 0 and epoch - best_epoch >= patience:
            break

    _write_report(out_dir, config, history, count_parameters(model), best_val, best_epoch)


def evaluate(
    model: MaskedEEGAutoencoder,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> float:
    model.eval()
    train_cfg = config["train"]
    values: list[float] = []
    with torch.no_grad():
        for eeg in loader:
            eeg = eeg.to(device, non_blocking=True)
            mask = make_time_channel_mask(
                eeg,
                mask_ratio_time=float(train_cfg.get("mask_ratio_time", 0.35)),
                mask_ratio_channel=float(train_cfg.get("mask_ratio_channel", 0.15)),
                span=int(train_cfg.get("mask_span", 12)),
            )
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                recon = model(eeg.masked_fill(mask, 0.0))
                loss, _ = masked_eeg_loss(
                    recon,
                    eeg,
                    mask,
                    lambda_spectral=float(train_cfg.get("lambda_spectral", 0.2)),
                    lambda_smoothness=float(train_cfg.get("lambda_smoothness", 0.1)),
                )
            values.append(float(loss.detach().cpu()))
    model.train()
    return float(np.mean(values)) if values else 0.0


def _write_report(out_dir: Path, config: dict[str, Any], history: list[dict[str, Any]], param_count: int, best_val: float, best_epoch: int) -> None:
    train_cfg = config["train"]
    data_cfg = config["data"]
    last = history[-1] if history else {}
    lines = [
        "# Masked EEG Pretraining Report",
        "",
        f"- Dataset: `{data_cfg.get('name', 'unknown')}`",
        f"- Train manifest: `{data_cfg.get('train_manifest', 'cache-only')}`",
        f"- Val manifest: `{data_cfg.get('val_manifest', 'cache-only')}`",
        f"- Train samples: `{len(CachedEEGDataset(data_cfg.get('train_eeg_cache', 'data/thought2text/cache/eeg_pretrain_train.npy')) )}`",
        f"- Val samples: `{len(CachedEEGDataset(data_cfg.get('val_eeg_cache', 'data/thought2text/cache/eeg_pretrain_val.npy')) )}`",
        f"- Parameter count: `{param_count}`",
        f"- Mask ratio time/channel: `{train_cfg.get('mask_ratio_time')}` / `{train_cfg.get('mask_ratio_channel')}`",
        f"- Batch size: `{train_cfg.get('resolved_batch_size')}`",
        f"- Grad accumulation steps: `{train_cfg.get('resolved_grad_accum_steps')}`",
        f"- Effective batch size: `{train_cfg.get('effective_batch_size')}`",
        f"- Best val masked loss: `{best_val:.6f}` at epoch `{best_epoch}`",
        f"- Peak GPU memory MB: `{last.get('gpu_mem_peak_mb', 0):.1f}`",
        f"- Avg step time seconds: `{last.get('avg_step_time', 0):.4f}`",
        f"- Checkpoint: `{out_dir / 'checkpoints' / 'best_masked_eeg.pt'}`",
        "",
        "| Epoch | Train Loss | Val Loss | Epoch Seconds | GPU Mem Peak MB |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in history:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.6f} | {row['val_loss']:.6f} | "
            f"{row['epoch_seconds']:.2f} | {row['gpu_mem_peak_mb']:.1f} |"
        )
    report = "\n".join(lines) + "\n"
    (out_dir / "MASKED_EEG_PRETRAIN_REPORT.md").write_text(report, encoding="utf-8")
    mirror = Path("outputs/pretrain/MASKED_EEG_PRETRAIN_REPORT.md")
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(report, encoding="utf-8")
    target_ckpt = Path("outputs/pretrain/checkpoints/best_masked_eeg.pt")
    target_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if not target_ckpt.exists():
        target_ckpt.symlink_to((out_dir / "checkpoints" / "best_masked_eeg.pt").resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train masked EEG autoencoder on full EEG data.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(_load_yaml(args.config))


if __name__ == "__main__":
    main()
