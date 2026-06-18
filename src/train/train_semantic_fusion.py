from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import EEGVisionCaptionDataset
from src.eval.constrained_caption_eval import load_eeg_encoder
from src.utils.seed import seed_everything


class SemanticFusionDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest: Path, cache_path: Path, index_path: Path, max_samples: int = 0) -> None:
        self.rows = self._read_jsonl(manifest)
        self.image_emb = torch.from_numpy(np.load(cache_path)).float()
        with index_path.open("r", encoding="utf-8") as handle:
            self.index_rows = json.load(handle)
        if len(self.rows) != self.image_emb.shape[0] or len(self.index_rows) != self.image_emb.shape[0]:
            raise ValueError("Manifest, cache, and cache index must have the same length")
        if max_samples:
            self.rows = self.rows[:max_samples]
            self.index_rows = self.index_rows[:max_samples]
            self.image_emb = self.image_emb[:max_samples]
        self.eeg_dataset = EEGVisionCaptionDataset(manifest, allow_missing_images=True)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Semantic fusion consumes cached CLIP features, so loading/resizing images here only starves the GPU.
        eeg = self.eeg_dataset._load_row_eeg(self.rows[index])
        return {
            "image_emb": self.image_emb[index],
            "eeg": eeg,
            "label": int(self.rows[index]["label"]),
            "image_id": str(self.rows[index]["image_id"]),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image_emb": torch.stack([item["image_emb"] for item in batch], dim=0).float(),
        "eeg": torch.stack([item["eeg"] for item in batch], dim=0).float(),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "image_id": [item["image_id"] for item in batch],
    }


class SemanticFusionClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image_emb: torch.Tensor, eeg_emb: torch.Tensor | None = None) -> torch.Tensor:
        image_emb = F.normalize(image_emb.float(), dim=-1)
        if eeg_emb is None:
            features = image_emb
        else:
            features = torch.cat([image_emb, F.normalize(eeg_emb.float(), dim=-1)], dim=-1)
        return self.net(features)


class ReliabilityGatedSemanticFusionClassifier(nn.Module):
    def __init__(self, image_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.image_dim = int(image_dim)
        self.eeg_delta = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, image_dim),
        )
        self.gate_net = nn.Sequential(
            nn.LayerNorm(image_dim * 2 + 1),
            nn.Linear(image_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def gate_values(
        self,
        image_emb: torch.Tensor,
        eeg_emb: torch.Tensor | None,
        vision_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image_norm = F.normalize(image_emb.float(), dim=-1)
        if eeg_emb is None:
            eeg_norm = torch.zeros_like(image_norm)
        else:
            eeg_norm = F.normalize(eeg_emb.float(), dim=-1)
        if vision_confidence is None:
            vision_confidence = image_norm.norm(dim=-1, keepdim=True)
        else:
            vision_confidence = vision_confidence.float().view(image_norm.shape[0], 1).to(image_norm.device)
        gate_input = torch.cat([image_norm, eeg_norm, vision_confidence], dim=-1)
        return torch.sigmoid(self.gate_net(gate_input))

    def forward(
        self,
        image_emb: torch.Tensor,
        eeg_emb: torch.Tensor | None = None,
        vision_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image_norm = F.normalize(image_emb.float(), dim=-1)
        if eeg_emb is None:
            eeg_norm = torch.zeros_like(image_norm)
        else:
            eeg_norm = F.normalize(eeg_emb.float(), dim=-1)
        gate = self.gate_values(image_norm, eeg_norm, vision_confidence)
        fused = image_norm + gate * self.eeg_delta(eeg_norm)
        fused = F.normalize(fused, dim=-1)
        return self.classifier(fused)


def load_class_prototypes(path: str | None, device: torch.device, expected_dim: int) -> torch.Tensor | None:
    if not path:
        return None
    proto_path = Path(path)
    if not proto_path.exists():
        raise FileNotFoundError(f"Prototype path not found: {proto_path}")
    prototypes = torch.from_numpy(np.load(proto_path)).float().to(device)
    if prototypes.ndim != 2 or prototypes.shape[1] != expected_dim:
        raise ValueError(f"Expected class prototypes with shape [C, {expected_dim}], got {tuple(prototypes.shape)}")
    return F.normalize(prototypes, dim=-1)


def vision_confidence_from_prototypes(image_emb: torch.Tensor, prototypes: torch.Tensor | None) -> torch.Tensor | None:
    if prototypes is None:
        return None
    image_norm = F.normalize(image_emb.float(), dim=-1)
    logits = image_norm @ prototypes.to(image_norm.device).T
    return torch.softmax(logits, dim=-1).max(dim=-1, keepdim=True).values


def build_semantic_classifier(
    *,
    fusion_type: str,
    image_dim: int,
    hidden_dim: int,
    num_classes: int,
    uses_eeg: bool,
) -> nn.Module:
    if fusion_type == "concat":
        input_dim = image_dim * 2 if uses_eeg else image_dim
        return SemanticFusionClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    if fusion_type == "gated":
        if not uses_eeg:
            raise ValueError("fusion_type='gated' requires an EEG checkpoint")
        return ReliabilityGatedSemanticFusionClassifier(image_dim=image_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    raise ValueError(f"Unsupported fusion_type: {fusion_type}")


def configure_eeg_encoder_train_mode(eeg_encoder: nn.Module, train_mode: str) -> bool:
    """Return True when the EEG encoder should receive gradients."""
    if train_mode == "frozen":
        for parameter in eeg_encoder.parameters():
            parameter.requires_grad_(False)
        return False
    if train_mode != "unfreeze_last2":
        raise ValueError(f"Unsupported encoder_train_mode: {train_mode}")
    for parameter in eeg_encoder.parameters():
        parameter.requires_grad_(False)
    tail_root = getattr(eeg_encoder, "eeg_encoder", eeg_encoder)
    child_modules = list(tail_root.children())
    trainable_modules = child_modules[-2:] if len(child_modules) >= 2 else child_modules
    if not trainable_modules:
        trainable_modules = [tail_root]
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return any(parameter.requires_grad for parameter in eeg_encoder.parameters())


def accuracy(logits: torch.Tensor, labels: torch.Tensor, label_values: torch.Tensor) -> float:
    pred_indices = logits.argmax(dim=-1)
    pred_labels = label_values.to(logits.device)[pred_indices]
    return float((pred_labels == labels).float().mean().detach().cpu())


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    eeg_encoder: nn.Module | None,
    label_values: torch.Tensor,
    device: torch.device,
    class_prototypes: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    if eeg_encoder is not None:
        eeg_encoder.eval()
    losses: list[float] = []
    accs: list[float] = []
    label_to_index = {int(label): idx for idx, label in enumerate(label_values.tolist())}
    with torch.no_grad():
        for batch in loader:
            image_emb = batch["image_emb"].to(device)
            labels = batch["label"].to(device)
            targets = torch.tensor([label_to_index[int(label)] for label in labels.tolist()], device=device)
            eeg_out = eeg_encoder(batch["eeg"].to(device)) if eeg_encoder is not None else None
            eeg_emb = eeg_out[0] if isinstance(eeg_out, tuple) else eeg_out
            vision_confidence = vision_confidence_from_prototypes(image_emb, class_prototypes)
            if isinstance(model, ReliabilityGatedSemanticFusionClassifier):
                logits = model(image_emb, eeg_emb, vision_confidence)
            else:
                logits = model(image_emb, eeg_emb)
            losses.append(float(F.cross_entropy(logits, targets).detach().cpu()))
            accs.append(accuracy(logits, labels, label_values))
    model.train()
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(np.mean(accs)) if accs else 0.0,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(int(getattr(args, "seed", 42)))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    train_data = SemanticFusionDataset(Path(args.train_manifest), Path(args.train_cache), Path(args.train_index), args.max_train_samples)
    val_data = SemanticFusionDataset(Path(args.val_manifest), Path(args.val_cache), Path(args.val_index), args.max_val_samples)
    label_values = torch.tensor(sorted({int(row["label"]) for row in train_data.rows + val_data.rows}), dtype=torch.long)
    label_to_index = {int(label): idx for idx, label in enumerate(label_values.tolist())}

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    eeg_encoder = load_eeg_encoder(Path(args.eeg_checkpoint), device) if args.eeg_checkpoint else None
    train_eeg_encoder = False
    if eeg_encoder is not None:
        train_eeg_encoder = configure_eeg_encoder_train_mode(eeg_encoder, args.encoder_train_mode)
    image_dim = int(train_data.image_emb.shape[1])
    class_prototypes = load_class_prototypes(args.class_prototypes, device, image_dim)
    model = build_semantic_classifier(
        fusion_type=args.fusion_type,
        image_dim=image_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(label_values),
        uses_eeg=eeg_encoder is not None,
    ).to(device)
    param_groups: list[dict[str, Any]] = [{"params": model.parameters(), "lr": args.lr}]
    if eeg_encoder is not None and train_eeg_encoder:
        encoder_params = [parameter for parameter in eeg_encoder.parameters() if parameter.requires_grad]
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": args.encoder_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        for step, batch in enumerate(train_loader, start=1):
            image_emb = batch["image_emb"].to(device)
            labels = batch["label"].to(device)
            targets = torch.tensor([label_to_index[int(label)] for label in labels.tolist()], device=device)
            if eeg_encoder is not None and train_eeg_encoder:
                eeg_out = eeg_encoder(batch["eeg"].to(device))
            else:
                with torch.no_grad():
                    eeg_out = eeg_encoder(batch["eeg"].to(device)) if eeg_encoder is not None else None
            eeg_emb = eeg_out[0] if isinstance(eeg_out, tuple) else eeg_out
            vision_confidence = vision_confidence_from_prototypes(image_emb, class_prototypes)
            if isinstance(model, ReliabilityGatedSemanticFusionClassifier):
                logits = model(image_emb, eeg_emb, vision_confidence)
            else:
                logits = model(image_emb, eeg_emb)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if args.log_every and step % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": step,
                            "steps_per_epoch": len(train_loader),
                            "train_loss_running": float(np.mean(losses)),
                        }
                    ),
                    flush=True,
                )
        val_stats = evaluate(model, val_loader, eeg_encoder, label_values, device, class_prototypes)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_loss": val_stats["loss"],
            "val_accuracy": val_stats["accuracy"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "label_values": label_values,
        "uses_eeg": eeg_encoder is not None,
        "fusion_type": args.fusion_type,
        "history": history,
        "args": vars(args),
    }
    if eeg_encoder is not None and train_eeg_encoder:
        checkpoint["eeg_encoder_model"] = eeg_encoder.state_dict()
        checkpoint["eeg_encoder_source_checkpoint"] = args.eeg_checkpoint
    torch.save(checkpoint, output_dir / "semantic_fusion_classifier.pt")
    with (output_dir / "semantic_fusion_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    report = [
        "# Semantic Fusion Classifier Report",
        "",
        f"- Train samples: `{len(train_data)}`",
        f"- Val samples: `{len(val_data)}`",
        f"- Classes: `{len(label_values)}`",
        f"- Uses EEG encoder: `{eeg_encoder is not None}`",
        f"- EEG encoder train mode: `{args.encoder_train_mode}`",
        f"- Trainable EEG encoder: `{train_eeg_encoder}`",
        f"- Fusion type: `{args.fusion_type}`",
        f"- Class prototypes: `{args.class_prototypes or 'none'}`",
        f"- Checkpoint: `{output_dir / 'semantic_fusion_classifier.pt'}`",
        "",
        "| Epoch | Train Loss | Val Loss | Val Accuracy |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in history:
        report.append(f"| {row['epoch']} | {row['train_loss']:.4f} | {row['val_loss']:.4f} | {row['val_accuracy']:.4f} |")
    (output_dir / "semantic_fusion_train_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return history[-1] if history else {}


def self_test() -> None:
    namespace = argparse.Namespace(
        train_manifest="data/thought2text/train_human_caption.jsonl",
        val_manifest="data/thought2text/val_human_caption.jsonl",
        train_cache="data/thought2text/cache/clip_train.npy",
        val_cache="data/thought2text/cache/clip_val.npy",
        train_index="data/thought2text/cache/clip_index_train.json",
        val_index="data/thought2text/cache/clip_index_val.json",
        eeg_checkpoint=None,
        output_dir="outputs/semantic_caption/self_test_train",
        max_train_samples=32,
        max_val_samples=16,
        batch_size=8,
        epochs=1,
        hidden_dim=64,
        lr=1e-3,
        weight_decay=0.01,
        num_workers=0,
        log_every=0,
        device="cpu",
        seed=42,
        fusion_type="concat",
        class_prototypes=None,
        encoder_train_mode="frozen",
        encoder_lr=1e-5,
    )
    stats = train(namespace)
    assert "val_accuracy" in stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight semantic class head for controlled captions.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_manifest", default="data/thought2text/train_human_caption.jsonl")
    parser.add_argument("--val_manifest", default="data/thought2text/val_human_caption.jsonl")
    parser.add_argument("--train_cache", default="data/thought2text/cache/clip_train.npy")
    parser.add_argument("--val_cache", default="data/thought2text/cache/clip_val.npy")
    parser.add_argument("--train_index", default="data/thought2text/cache/clip_index_train.json")
    parser.add_argument("--val_index", default="data/thought2text/cache/clip_index_val.json")
    parser.add_argument("--eeg_checkpoint", default=None)
    parser.add_argument("--output_dir", default="outputs/semantic_caption")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--fusion_type", choices=["concat", "gated"], default="concat")
    parser.add_argument("--class_prototypes", default="data/thought2text/cache/class_image_prototypes.npy")
    parser.add_argument("--encoder_train_mode", choices=["frozen", "unfreeze_last2"], default="frozen")
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    train(args)


if __name__ == "__main__":
    main()
