from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


class PrototypeCaptioner:
    """Nearest-prototype classifier that emits controlled class-template captions."""

    def __init__(
        self,
        labels: list[int],
        class_name_map: dict[int, str],
        image_prototypes: torch.Tensor,
    ) -> None:
        if image_prototypes.ndim != 2:
            raise ValueError(f"image_prototypes must be [num_classes, dim], got {tuple(image_prototypes.shape)}")
        if len(labels) != image_prototypes.shape[0]:
            raise ValueError("labels and image_prototypes have different class counts")
        self.labels = [int(label) for label in labels]
        self.class_name_map = {int(k): str(v) for k, v in class_name_map.items()}
        self.image_prototypes = F.normalize(image_prototypes.float(), dim=-1)

    @classmethod
    def from_file(cls, path: str | Path, device: str | torch.device = "cpu") -> "PrototypeCaptioner":
        bank = torch.load(path, map_location=device, weights_only=False)
        class_name_map = {int(k): str(v) for k, v in bank["class_name_map"].items()}
        return cls(
            labels=[int(label) for label in bank["labels"]],
            class_name_map=class_name_map,
            image_prototypes=bank["image_prototypes"].to(device),
        )

    def classify(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be [B, D], got {tuple(embeddings.shape)}")
        scores = F.normalize(embeddings.float(), dim=-1) @ self.image_prototypes.T
        prototype_indices = scores.argmax(dim=-1)
        labels = torch.tensor(self.labels, device=embeddings.device, dtype=torch.long)[prototype_indices]
        return labels, scores.max(dim=-1).values

    def topk(self, embeddings: torch.Tensor, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be [B, D], got {tuple(embeddings.shape)}")
        scores = F.normalize(embeddings.float(), dim=-1) @ self.image_prototypes.T
        k = min(k, scores.shape[-1])
        values, indices = scores.topk(k=k, dim=-1)
        label_tensor = torch.tensor(self.labels, device=embeddings.device, dtype=torch.long)
        return label_tensor[indices], values

    def caption_for_label(self, label: int) -> str:
        name = self.class_name_map[int(label)]
        return f"a photo of a {name}"

    def caption(self, embeddings: torch.Tensor) -> list[str]:
        labels, _ = self.classify(embeddings)
        return [self.caption_for_label(int(label)) for label in labels.detach().cpu().tolist()]

    def predict_records(self, embeddings: torch.Tensor, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        labels, scores = self.classify(embeddings)
        top5_labels, top5_scores = self.topk(embeddings, k=5)
        records: list[dict[str, Any]] = []
        for row, label, score, top_labels, top_scores in zip(
            rows,
            labels.detach().cpu().tolist(),
            scores.detach().cpu().tolist(),
            top5_labels.detach().cpu().tolist(),
            top5_scores.detach().cpu().tolist(),
            strict=False,
        ):
            target_label = int(row["label"]) if row.get("label") is not None else None
            pred_label = int(label)
            top_labels = [int(item) for item in top_labels]
            records.append(
                {
                    "image_id": str(row["image_id"]),
                    "label": target_label,
                    "pred_label": pred_label,
                    "top5_labels": top_labels,
                    "top5_class_names": [self.class_name_map[item] for item in top_labels],
                    "top5_scores": [float(item) for item in top_scores],
                    "human_label_name": self.class_name_map.get(target_label, str(target_label)),
                    "pred_class_name": self.class_name_map[pred_label],
                    "reference": f"a photo of a {self.class_name_map.get(target_label, str(target_label))}",
                    "prediction": self.caption_for_label(pred_label),
                    "prototype_score": float(score),
                    "class_correct": float(pred_label == target_label) if target_label is not None else None,
                    "top5_correct": float(target_label in top_labels) if target_label is not None else None,
                }
            )
        return records


def self_test() -> None:
    labels = [1, 2]
    names = {1: "alpha", 2: "beta"}
    prototypes = torch.eye(2)
    captioner = PrototypeCaptioner(labels, names, prototypes)
    records = captioner.predict_records(torch.tensor([[0.9, 0.1], [0.2, 0.8]]), [{"image_id": "a", "label": 1}, {"image_id": "b", "label": 2}])
    assert records[0]["prediction"] == "a photo of a alpha"
    assert records[1]["pred_label"] == 2
    print(json.dumps(records, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test prototype captioner.")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.error("Use --self_test, or import PrototypeCaptioner from evaluation code.")


if __name__ == "__main__":
    main()
