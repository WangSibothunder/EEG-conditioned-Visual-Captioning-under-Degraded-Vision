from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from src.eval.constrained_caption_eval import load_cache, load_eeg_encoder, read_jsonl
from src.eval.metrics import bleu_n, rouge_l
from src.eval.prototype_captioner import PrototypeCaptioner
from src.train.train_fusion import apply_eeg_mode
from src.train.train_semantic_fusion import (
    ReliabilityGatedSemanticFusionClassifier,
    SemanticFusionClassifier,
    build_semantic_classifier,
    load_class_prototypes,
    vision_confidence_from_prototypes,
)


def compose_classifier_inputs(
    image_emb: torch.Tensor,
    eeg_emb: torch.Tensor | None,
    *,
    mode: str,
    uses_eeg: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    image_norm = F.normalize(image_emb.float(), dim=-1)
    if not uses_eeg:
        if mode not in {"vision_only", "real_eeg"}:
            raise ValueError(f"Classifier checkpoint does not use EEG; unsupported mode: {mode}")
        return image_norm, None
    if eeg_emb is None:
        raise ValueError(f"Mode {mode} requires EEG embeddings for an EEG-enabled classifier")
    eeg_norm = F.normalize(eeg_emb.float(), dim=-1)
    if mode == "vision_only":
        return image_norm, torch.zeros_like(eeg_norm)
    if mode in {"real_eeg", "shuffled_eeg", "global_shuffled_eeg", "random_eeg"}:
        return image_norm, eeg_norm
    if mode == "eeg_only":
        return torch.zeros_like(image_norm), eeg_norm
    raise ValueError(f"Unsupported mode: {mode}")


def summarize_records(records: list[dict[str, Any]], corruption: str, mode: str, file_name: str) -> dict[str, Any]:
    if not records:
        summary = {
            "file": file_name,
            "corruption": corruption,
            "mode": mode,
            "count": 0,
            "accuracy": 0.0,
            "top5_accuracy": 0.0,
            "caption_class_hit": 0.0,
            "bleu_1": 0.0,
            "bleu_4": 0.0,
            "rouge_l": 0.0,
            "invalid_caption_rate": 0.0,
        }
        return summary
    valid_predictions = [bool(str(row.get("prediction", "")).strip()) for row in records]
    summary = {
        "file": file_name,
        "corruption": corruption,
        "mode": mode,
        "count": len(records),
        "accuracy": sum(float(row["class_correct"]) for row in records) / len(records),
        "top5_accuracy": sum(float(row["top5_correct"]) for row in records) / len(records),
        "caption_class_hit": sum(float(row["class_correct"]) for row in records) / len(records),
        "bleu_1": sum(bleu_n(str(row.get("reference", "")), str(row.get("prediction", "")), 1) for row in records) / len(records),
        "bleu_4": sum(bleu_n(str(row.get("reference", "")), str(row.get("prediction", "")), 4) for row in records) / len(records),
        "rouge_l": sum(rouge_l(str(row.get("reference", "")), str(row.get("prediction", ""))) for row in records) / len(records),
        "invalid_caption_rate": 1.0 - (sum(float(item) for item in valid_predictions) / len(valid_predictions)),
    }
    gate_values = [float(row["gate_mean"]) for row in records if row.get("gate_mean") is not None]
    if gate_values:
        summary["gate_mean"] = sum(gate_values) / len(gate_values)
    return summary


def label_mismatched_indices(rows: list[dict[str, Any]]) -> list[int]:
    """Deterministically map each row to an EEG row with a different class label."""
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_label[int(row["label"])].append(index)
    labels = sorted(by_label)
    if len(labels) < 2:
        raise ValueError("Global label-mismatched EEG control requires at least two labels")
    result: list[int] = []
    offsets = {label: 0 for label in labels}
    for row in rows:
        label = int(row["label"])
        candidate_labels = [item for item in labels if item != label]
        target_label = candidate_labels[offsets[label] % len(candidate_labels)]
        candidates = by_label[target_label]
        candidate_index = candidates[offsets[target_label] % len(candidates)]
        offsets[label] += 1
        offsets[target_label] += 1
        result.append(candidate_index)
    return result


def grouped_image_summary(records: list[dict[str, Any]], corruption: str, mode: str, file_name: str) -> dict[str, Any]:
    if not records:
        return {
            "file": file_name,
            "corruption": corruption,
            "mode": mode,
            "count": 0,
            "trial_count": 0,
            "accuracy": 0.0,
            "top5_accuracy": 0.0,
            "caption_class_hit": 0.0,
        }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row["image_id"])].append(row)
    image_hits: list[float] = []
    top5_hits: list[float] = []
    gate_means: list[float] = []
    for group in groups.values():
        target_label = int(group[0]["label"]) if group[0].get("label") is not None else None
        pred_counter = Counter(int(item["pred_label"]) for item in group)
        pred_label = pred_counter.most_common(1)[0][0]
        top5_counter: Counter[int] = Counter()
        for item in group:
            top5_counter.update(int(label) for label in item.get("top5_labels", []))
            if item.get("gate_mean") is not None:
                gate_means.append(float(item["gate_mean"]))
        top5_labels = [label for label, _count in top5_counter.most_common(5)]
        image_hits.append(float(pred_label == target_label) if target_label is not None else 0.0)
        top5_hits.append(float(target_label in top5_labels) if target_label is not None else 0.0)
    summary = {
        "file": file_name,
        "corruption": corruption,
        "mode": mode,
        "count": len(groups),
        "trial_count": len(records),
        "accuracy": sum(image_hits) / len(image_hits),
        "top5_accuracy": sum(top5_hits) / len(top5_hits),
        "caption_class_hit": sum(image_hits) / len(image_hits),
    }
    if gate_means:
        summary["gate_mean"] = sum(gate_means) / len(gate_means)
    return summary


def load_classifier(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, torch.Tensor, bool, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    label_values = payload["label_values"].to(device=device, dtype=torch.long)
    uses_eeg = bool(payload.get("uses_eeg", True))
    args = payload.get("args", {})
    hidden_dim = int(args.get("hidden_dim", 1024))
    fusion_type = str(payload.get("fusion_type") or args.get("fusion_type") or "concat")
    image_dim = 512
    state = payload["model"]
    if fusion_type == "gated":
        gate_weight = state.get("gate_net.1.weight")
        if isinstance(gate_weight, torch.Tensor):
            image_dim = (int(gate_weight.shape[1]) - 1) // 2
    else:
        first_weight = state.get("net.1.weight")
        if isinstance(first_weight, torch.Tensor):
            input_dim = int(first_weight.shape[1])
            image_dim = input_dim // 2 if uses_eeg else input_dim
    model = build_semantic_classifier(
        fusion_type=fusion_type,
        image_dim=image_dim,
        hidden_dim=hidden_dim,
        num_classes=len(label_values),
        uses_eeg=uses_eeg,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, label_values, uses_eeg, args


def maybe_write_embedded_eeg_encoder_checkpoint(classifier_checkpoint: Path, payload: dict[str, Any]) -> Path | None:
    eeg_state = payload.get("eeg_encoder_model")
    if eeg_state is None:
        return None
    args = payload.get("args", {})
    source_path = args.get("eeg_checkpoint") or payload.get("eeg_encoder_source_checkpoint")
    source_payload: dict[str, Any] = {}
    if source_path and Path(source_path).exists():
        source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
    materialized = classifier_checkpoint.with_name("finetuned_eeg_encoder.pt")
    torch.save(
        {
            "model": eeg_state,
            "config": source_payload.get("config", {"model": {}}),
            "source_checkpoint": str(source_path) if source_path else "",
            "finetuned_from_classifier": str(classifier_checkpoint),
        },
        materialized,
    )
    return materialized


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
                raise ValueError(f"EEG mode {mode} did not produce EEG tensors")
            clip_pred, _ = encoder(mode_eeg)
            outputs.append(clip_pred.detach().cpu())
            seen += eeg.shape[0]
            if max_samples and seen >= max_samples:
                break
    out = torch.cat(outputs, dim=0)
    return out[:max_samples] if max_samples else out


def _caption_records(
    *,
    logits: torch.Tensor,
    label_values: torch.Tensor,
    rows: list[dict[str, Any]],
    captioner: PrototypeCaptioner,
    mode: str,
    corruption: str,
) -> list[dict[str, Any]]:
    probs = torch.softmax(logits.float(), dim=-1)
    k = min(5, probs.shape[-1])
    top_scores, top_indices = probs.topk(k=k, dim=-1)
    top_labels = label_values.detach().cpu()[top_indices.detach().cpu()].tolist()
    pred_labels = label_values.detach().cpu()[top_indices[:, 0].detach().cpu()].tolist()
    records: list[dict[str, Any]] = []
    for row, pred_label, labels, scores in zip(rows, pred_labels, top_labels, top_scores.detach().cpu().tolist(), strict=False):
        target_label = int(row["label"]) if row.get("label") is not None else None
        pred_label = int(pred_label)
        top5_labels = [int(label) for label in labels]
        records.append(
            {
                "image_id": str(row["image_id"]),
                "mode": mode,
                "corruption": corruption,
                "label": target_label,
                "pred_label": pred_label,
                "top5_labels": top5_labels,
                "top5_class_names": [captioner.class_name_map[item] for item in top5_labels],
                "top5_scores": [float(item) for item in scores],
                "human_label_name": captioner.class_name_map.get(target_label, str(target_label)),
                "pred_class_name": captioner.class_name_map[pred_label],
                "reference": f"a photo of a {captioner.class_name_map.get(target_label, str(target_label))}",
                "prediction": captioner.caption_for_label(pred_label),
                "class_correct": float(pred_label == target_label) if target_label is not None else 0.0,
                "top5_correct": float(target_label in top5_labels) if target_label is not None else 0.0,
            }
        )
    return records


def write_metrics(rows: list[dict[str, Any]], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        preferred = [
            "file",
            "corruption",
            "mode",
            "count",
            "trial_count",
            "accuracy",
            "top5_accuracy",
            "caption_class_hit",
        ]
        all_keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in all_keys:
                    all_keys.append(key)
        headers = [key for key in preferred if key in all_keys] + [key for key in all_keys if key not in preferred]
    else:
        headers = ["file", "corruption", "mode", "count", "accuracy", "top5_accuracy"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Semantic Fusion Classifier Metrics", ""]
    if rows:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    f"{row.get(h, ''):.4f}" if isinstance(row.get(h, ""), float) else str(row.get(h, ""))
                    for h in headers
                )
                + " |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _semantic_gap_rows(metrics: list[dict[str, Any]], *, mode_suffix: str = "") -> list[dict[str, Any]]:
    by_condition: dict[str, dict[str, dict[str, Any]]] = {}
    for row in metrics:
        by_condition.setdefault(str(row["corruption"]), {})[str(row["mode"])] = row
    rows: list[dict[str, Any]] = []
    for corruption, mode_rows in sorted(by_condition.items()):
        real = mode_rows.get(f"real_eeg{mode_suffix}")
        vision = mode_rows.get(f"vision_only{mode_suffix}")
        shuffled = mode_rows.get(f"shuffled_eeg{mode_suffix}")
        global_shuffled = mode_rows.get(f"global_shuffled_eeg{mode_suffix}")
        random = mode_rows.get(f"random_eeg{mode_suffix}")
        if real is None:
            continue
        real_acc = float(real.get("accuracy", 0.0))
        vision_acc = float(vision.get("accuracy", 0.0)) if vision else 0.0
        shuffled_acc = float(shuffled.get("accuracy", 0.0)) if shuffled else 0.0
        global_shuffled_acc = float(global_shuffled.get("accuracy", 0.0)) if global_shuffled else 0.0
        random_acc = float(random.get("accuracy", 0.0)) if random else 0.0
        control_accs = [item for item in [shuffled_acc, global_shuffled_acc, random_acc] if item is not None]
        row = {
            "corruption": corruption,
            "vision_only_acc": vision_acc,
            "real_eeg_acc": real_acc,
            "shuffled_eeg_acc": shuffled_acc,
            "global_shuffled_eeg_acc": global_shuffled_acc,
            "random_eeg_acc": random_acc,
            "real_minus_vision": real_acc - vision_acc,
            "real_minus_shuffled": real_acc - shuffled_acc,
            "real_minus_global_shuffled": real_acc - global_shuffled_acc,
            "real_minus_random": real_acc - random_acc,
            "real_beats_controls": bool(all(real_acc > control for control in control_accs)),
            "real_beats_vision": bool(real_acc > vision_acc),
        }
        rows.append(row)
    return rows


def write_gap_metrics(metrics: list[dict[str, Any]], output_dir: Path) -> None:
    rows = _semantic_gap_rows(metrics)
    write_metrics(rows, output_dir / "SEMANTIC_GAP_METRICS.csv", output_dir / "SEMANTIC_GAP_METRICS.md")
    image_rows = _semantic_gap_rows(metrics, mode_suffix="_image_level")
    if image_rows:
        write_metrics(
            image_rows,
            output_dir / "SEMANTIC_GAP_METRICS_IMAGE_LEVEL.csv",
            output_dir / "SEMANTIC_GAP_METRICS_IMAGE_LEVEL.md",
        )


def write_examples(output_dir: Path, records: list[dict[str, Any]], limit: int = 40) -> None:
    lines = ["# Semantic Fusion Classifier Examples", ""]
    for row in records[:limit]:
        lines.append(
            f"- `{row['corruption']}` / `{row['mode']}` / `{row['image_id']}`: "
            f"true `{row['human_label_name']}`, pred `{row['pred_class_name']}`"
        )
    (output_dir / "qualitative_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(
    *,
    classifier_checkpoint: Path,
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
    classifier_payload = torch.load(classifier_checkpoint, map_location="cpu", weights_only=False)
    embedded_eeg_checkpoint = maybe_write_embedded_eeg_encoder_checkpoint(classifier_checkpoint, classifier_payload)
    if embedded_eeg_checkpoint is not None:
        eeg_checkpoint = embedded_eeg_checkpoint
    classifier, label_values, uses_eeg, checkpoint_args = load_classifier(classifier_checkpoint, device)
    image_dim = int(getattr(classifier, "image_dim", 512))
    class_prototypes = load_class_prototypes(
        str(checkpoint_args.get("class_prototypes") or (cache_dir / "class_image_prototypes.npy")),
        device,
        image_dim,
    )
    captioner = PrototypeCaptioner.from_file(prototype_bank, device="cpu")
    manifest_rows = read_jsonl(manifest)
    if max_samples:
        manifest_rows = manifest_rows[:max_samples]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    eeg_cache: dict[str, torch.Tensor] = {}

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
                raise ValueError(f"Manifest/cache order mismatch for {corruption}: {manifest_row.get('image_id')} vs {cache_row.get('image_id')}")
        for mode in modes:
            eeg_emb: torch.Tensor | None = None
            if mode != "vision_only" or uses_eeg:
                if mode in {"real_eeg", "shuffled_eeg", "global_shuffled_eeg", "random_eeg", "eeg_only", "vision_only"} and uses_eeg:
                    if eeg_checkpoint is None:
                        continue
                    cache_key = "real_eeg" if mode in {"vision_only", "eeg_only", "global_shuffled_eeg"} else mode
                    if cache_key not in eeg_cache:
                        eeg_cache[cache_key] = eeg_embeddings(manifest, cache_key, eeg_checkpoint, max_samples=max_samples, batch_size=batch_size, device=device)
                    eeg_emb = eeg_cache[cache_key]
                    if mode == "global_shuffled_eeg":
                        eeg_emb = eeg_emb[label_mismatched_indices(rows)]
            image_in, eeg_in = compose_classifier_inputs(image_emb.to(device), eeg_emb.to(device) if eeg_emb is not None else None, mode=mode, uses_eeg=uses_eeg)
            with torch.no_grad():
                vision_confidence = vision_confidence_from_prototypes(image_in, class_prototypes)
                if isinstance(classifier, ReliabilityGatedSemanticFusionClassifier):
                    logits = classifier(image_in, eeg_in, vision_confidence)
                    gate_mean_tensor = classifier.gate_values(image_in, eeg_in, vision_confidence).detach().cpu().view(-1)
                else:
                    logits = classifier(image_in, eeg_in)
                    gate_mean_tensor = None
            records = _caption_records(logits=logits, label_values=label_values, rows=rows, captioner=captioner, mode=mode, corruption=corruption)
            if gate_mean_tensor is not None:
                for row, gate_value in zip(records, gate_mean_tensor.tolist(), strict=False):
                    row["gate_mean"] = float(gate_value)
            out_path = output_dir / f"{corruption}_{mode}.jsonl"
            with out_path.open("w", encoding="utf-8") as handle:
                for row in records:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            all_records.extend(records)
            metrics.append(summarize_records(records, corruption, mode, out_path.name))
            metrics.append(grouped_image_summary(records, corruption, f"{mode}_image_level", out_path.name))

    write_metrics(metrics, output_dir / "FULL_METRICS.csv", output_dir / "FULL_METRICS.md")
    write_gap_metrics(metrics, output_dir)
    write_examples(output_dir, all_records)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained semantic fusion classifier under degraded vision.")
    parser.add_argument("--classifier_checkpoint", default="outputs/heavy_stage/semantic_fusion_best_encoder_full/semantic_fusion_classifier.pt")
    parser.add_argument("--prototype_bank", default="outputs/semantic_caption/prototypes.pt")
    parser.add_argument("--manifest", default="data/thought2text/test_human_caption.jsonl")
    parser.add_argument("--cache_dir", default="data/thought2text/cache")
    parser.add_argument("--output_dir", default="outputs/final_semantic/semantic_fusion_classifier_eval")
    parser.add_argument("--corruptions", nargs="+", default=["clean", "strong_blur", "strong_noise", "occlusion50", "lowres16", "mixed"])
    parser.add_argument("--modes", nargs="+", default=["vision_only", "real_eeg", "shuffled_eeg", "global_shuffled_eeg", "random_eeg", "eeg_only"])
    parser.add_argument("--eeg_checkpoint", default="outputs/architectures/checkpoints/best_encoder.pt")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    metrics = evaluate(
        classifier_checkpoint=Path(args.classifier_checkpoint),
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
