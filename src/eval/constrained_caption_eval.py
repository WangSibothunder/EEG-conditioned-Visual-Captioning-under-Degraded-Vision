from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.collate import caption_collate
from src.data.dataset import EEGVisionCaptionDataset
from src.eval.metrics import bleu_n, rouge_l
from src.eval.prototype_captioner import PrototypeCaptioner
from src.models.alignment_model import EEGCLIPAlignmentModel
from src.train.train_fusion import apply_eeg_mode
from src.utils.checkpoint import load_checkpoint


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_cache(cache_path: Path, index_path: Path, max_samples: int = 0) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    embeddings = torch.from_numpy(np.load(cache_path)).float()
    with index_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if len(rows) != embeddings.shape[0]:
        raise ValueError(f"Cache/index length mismatch: {cache_path} vs {index_path}")
    if max_samples:
        embeddings = embeddings[:max_samples]
        rows = rows[:max_samples]
    return embeddings, rows


def _num_classes_from_manifest(path: Path) -> int | None:
    labels: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("label") is not None:
                    labels.append(int(row["label"]))
    return max(labels) + 1 if labels else None


def load_eeg_encoder(checkpoint_path: Path, device: torch.device, manifest: Path | None = None) -> EEGCLIPAlignmentModel:
    state = load_checkpoint(checkpoint_path, map_location=device)
    cfg = state.get("config", {})
    model_cfg = cfg.get("model", {})
    channels = int(model_cfg.get("eeg_channels", 64))
    timesteps = int(model_cfg.get("eeg_time_steps", model_cfg.get("eeg_timesteps", 250)))
    model = EEGCLIPAlignmentModel(
        eeg_channels=channels,
        eeg_timesteps=timesteps,
        eeg_dim=int(model_cfg.get("eeg_embed_dim", 512)),
        clip_dim=int(model_cfg.get("clip_embed_dim", 512)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        transformer_layers=int(model_cfg.get("transformer_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_classes=_num_classes_from_manifest(manifest) if manifest is not None and manifest.exists() else None,
        encoder_type=str(model_cfg.get("encoder_type", "tiny")),
    ).to(device)
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    if len(missing) > 8:
        raise RuntimeError(
            f"Could not load alignment model from {checkpoint_path}; many missing keys: {missing[:12]}"
        )
    if unexpected and len(unexpected) > 8:
        raise RuntimeError(
            f"Could not load alignment model from {checkpoint_path}; many unexpected keys: {unexpected[:12]}"
        )
    model.eval()
    return model


def eeg_embeddings(
    manifest: Path,
    mode: str,
    checkpoint_path: Path,
    *,
    max_samples: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    dataset = EEGVisionCaptionDataset(manifest, allow_missing_images=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=caption_collate)
    encoder = load_eeg_encoder(checkpoint_path, device, manifest)
    outputs: list[torch.Tensor] = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device)
            mode_eeg = apply_eeg_mode(eeg, mode)
            if mode_eeg is None:
                raise ValueError("eeg_embeddings requires an EEG mode")
            clip_pred, _ = encoder(mode_eeg)
            outputs.append(clip_pred.detach().cpu())
            seen += eeg.shape[0]
            if max_samples and seen >= max_samples:
                break
    out = torch.cat(outputs, dim=0)
    return out[:max_samples] if max_samples else out


def summarize(records: list[dict[str, Any]], corruption: str, mode: str, file_name: str) -> dict[str, Any]:
    if not records:
        return {
            "file": file_name,
            "corruption": corruption,
            "mode": mode,
            "count": 0,
            "accuracy": 0.0,
            "bleu_1": 0.0,
            "bleu_4": 0.0,
            "rouge_l": 0.0,
            "avg_prediction_length": 0.0,
            "distinct_prediction_ratio": 0.0,
        }
    return {
        "file": file_name,
        "corruption": corruption,
        "mode": mode,
        "count": len(records),
        "accuracy": sum(float(row["class_correct"]) for row in records if row["class_correct"] is not None)
        / max(1, sum(1 for row in records if row["class_correct"] is not None)),
        "top5_accuracy": sum(float(row.get("top5_correct", 0.0)) for row in records if row.get("top5_correct") is not None)
        / max(1, sum(1 for row in records if row.get("top5_correct") is not None)),
        "bleu_1": sum(bleu_n(row["reference"], row["prediction"], 1) for row in records) / len(records),
        "bleu_4": sum(bleu_n(row["reference"], row["prediction"], 4) for row in records) / len(records),
        "rouge_l": sum(rouge_l(row["reference"], row["prediction"]) for row in records) / len(records),
        "avg_prediction_length": sum(len(str(row["prediction"]).split()) for row in records) / len(records),
        "distinct_prediction_ratio": len({row["prediction"] for row in records}) / len(records),
    }


def write_metrics(rows: list[dict[str, Any]], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else ["file", "corruption", "mode", "count", "accuracy"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Controlled Semantic Caption Metrics", ""]
    if rows:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    f"{row[h]:.4f}" if isinstance(row[h], float) else str(row[h])
                    for h in headers
                )
                + " |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(
    *,
    prototype_bank: Path,
    manifest: Path,
    cache_dir: Path,
    output_dir: Path,
    corruptions: list[str],
    modes: list[str],
    eeg_checkpoint: Path | None,
    max_samples: int,
    batch_size: int,
    device_name: str,
) -> list[dict[str, Any]]:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    captioner = PrototypeCaptioner.from_file(prototype_bank, device="cpu")
    metrics: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    eeg_cache: dict[str, torch.Tensor] = {}
    manifest_rows = read_jsonl(manifest)
    if max_samples:
        manifest_rows = manifest_rows[:max_samples]

    for corruption in corruptions:
        if corruption == "clean":
            cache_path = cache_dir / "clip_test.npy"
            index_path = cache_dir / "clip_index_test.json"
        else:
            cache_path = cache_dir / "degraded_test" / f"clip_test_{corruption}.npy"
            index_path = cache_dir / "degraded_test" / f"clip_index_test_{corruption}.json"
        image_emb, cache_rows = load_cache(cache_path, index_path, max_samples=max_samples)
        rows = manifest_rows[: len(cache_rows)]
        for manifest_row, cache_row in zip(rows, cache_rows, strict=True):
            if manifest_row.get("image_id") != cache_row.get("image_id"):
                raise ValueError(
                    f"Manifest/cache order mismatch for {corruption}: {manifest_row.get('image_id')} vs {cache_row.get('image_id')}"
                )
        for mode in modes:
            if mode == "vision_only":
                emb = image_emb
            elif mode == "eeg_only":
                if eeg_checkpoint is None:
                    continue
                if mode not in eeg_cache:
                    eeg_cache[mode] = eeg_embeddings(manifest, "real_eeg", eeg_checkpoint, max_samples=max_samples, batch_size=batch_size, device=device)
                emb = eeg_cache[mode]
            elif mode in {"real_eeg", "shuffled_eeg", "random_eeg"}:
                if eeg_checkpoint is None:
                    continue
                if mode not in eeg_cache:
                    eeg_cache[mode] = eeg_embeddings(manifest, mode, eeg_checkpoint, max_samples=max_samples, batch_size=batch_size, device=device)
                emb = F.normalize(image_emb, dim=-1) + F.normalize(eeg_cache[mode], dim=-1)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            records = captioner.predict_records(emb, rows)
            for row in records:
                row["mode"] = mode
                row["corruption"] = corruption
            out_path = output_dir / f"{corruption}_{mode}.jsonl"
            with out_path.open("w", encoding="utf-8") as handle:
                for row in records:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics.append(summarize(records, corruption, mode, out_path.name))
    write_metrics(metrics, output_dir / "FULL_METRICS.csv", output_dir / "FULL_METRICS.md")
    return metrics


def self_test() -> None:
    from scripts.build_text_prototypes import build_prototypes

    out_dir = Path("outputs/semantic_caption")
    bank = out_dir / "self_test_eval_prototypes.pt"
    build_prototypes(Path("data/thought2text"), bank, splits=["train"], clip_prefix="clip", report_path=out_dir / "self_test_eval_prototypes.md")
    rows = evaluate(
        prototype_bank=bank,
        manifest=Path("data/thought2text/test_human_caption.jsonl"),
        cache_dir=Path("data/thought2text/cache"),
        output_dir=out_dir / "self_test_eval",
        corruptions=["clean"],
        modes=["vision_only"],
        eeg_checkpoint=None,
        max_samples=16,
        batch_size=8,
        device_name="cpu",
    )
    assert rows and rows[0]["count"] == 16
    print(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate controlled semantic captions from prototype predictions.")
    parser.add_argument("--prototype_bank", default="outputs/semantic_caption/prototypes.pt")
    parser.add_argument("--manifest", default="data/thought2text/test_human_caption.jsonl")
    parser.add_argument("--cache_dir", default="data/thought2text/cache")
    parser.add_argument("--output_dir", default="outputs/semantic_caption")
    parser.add_argument("--corruptions", nargs="+", default=["clean", "blur", "occlusion", "noise", "lowres"])
    parser.add_argument("--modes", nargs="+", default=["vision_only"])
    parser.add_argument("--eeg_checkpoint", default=None)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    metrics = evaluate(
        prototype_bank=Path(args.prototype_bank),
        manifest=Path(args.manifest),
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        corruptions=args.corruptions,
        modes=args.modes,
        eeg_checkpoint=Path(args.eeg_checkpoint) if args.eeg_checkpoint else None,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
