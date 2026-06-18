from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_cache_paths(data_root: Path, split: str, clip_prefix: str) -> tuple[Path, Path]:
    cache_dir = data_root / "cache"
    return cache_dir / f"{clip_prefix}_{split}.npy", cache_dir / f"clip_index_{split}.json"


def normalize_class_name(row: dict[str, Any]) -> str:
    if row.get("human_label_name"):
        return str(row["human_label_name"]).strip()
    caption = str(row.get("caption", "")).strip()
    prefix = "a photo of a "
    if caption.lower().startswith(prefix):
        return caption[len(prefix) :].strip()
    return caption.replace("a photo of ", "", 1).strip()


def build_class_maps(manifest_paths: list[Path]) -> tuple[dict[int, str], dict[int, str], Counter[int]]:
    names: dict[int, Counter[str]] = defaultdict(Counter)
    wnids: dict[int, Counter[str]] = defaultdict(Counter)
    counts: Counter[int] = Counter()
    for path in manifest_paths:
        for row in read_jsonl(path):
            label = int(row["label"])
            counts[label] += 1
            names[label][normalize_class_name(row)] += 1
            if row.get("label_name"):
                wnids[label][str(row["label_name"])] += 1

    class_name_map = {label: counter.most_common(1)[0][0] for label, counter in names.items()}
    class_wnid_map = {
        label: counter.most_common(1)[0][0]
        for label, counter in wnids.items()
        if counter
    }
    return class_name_map, class_wnid_map, counts


def build_image_prototypes(
    data_root: Path,
    manifest_paths: list[Path],
    splits: list[str],
    clip_prefix: str,
    class_name_map: dict[int, str],
) -> tuple[torch.Tensor, list[int], dict[int, int]]:
    sums: dict[int, torch.Tensor] = {}
    counts: Counter[int] = Counter()
    dim: int | None = None
    for split, manifest_path in zip(splits, manifest_paths, strict=True):
        cache_path, index_path = resolve_cache_paths(data_root, split, clip_prefix)
        if not cache_path.exists() or not index_path.exists():
            continue
        manifest_rows = read_jsonl(manifest_path)
        embeddings = torch.from_numpy(np.load(cache_path)).float()
        with index_path.open("r", encoding="utf-8") as handle:
            index_rows = json.load(handle)
        if len(index_rows) != embeddings.shape[0] or len(manifest_rows) != embeddings.shape[0]:
            raise ValueError(f"Cache/index length mismatch for {split}: {embeddings.shape[0]} vs {len(index_rows)}")
        dim = int(embeddings.shape[1])
        for manifest_row, index_row, emb in zip(manifest_rows, index_rows, embeddings, strict=True):
            if manifest_row.get("image_id") != index_row.get("image_id"):
                raise ValueError(
                    f"Image order mismatch in {split}: manifest {manifest_row.get('image_id')} vs cache {index_row.get('image_id')}"
                )
            label = int(manifest_row["label"])
            if label not in class_name_map:
                continue
            if label not in sums:
                sums[label] = torch.zeros_like(emb)
            sums[label] += emb
            counts[label] += 1

    if dim is None:
        raise FileNotFoundError(f"No CLIP cache found under {data_root / 'cache'} with prefix {clip_prefix}")

    labels = sorted(class_name_map)
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    prototypes = torch.zeros((len(labels), dim), dtype=torch.float32)
    for label in labels:
        if counts[label]:
            prototypes[label_to_index[label]] = sums[label] / counts[label]
    prototypes = F.normalize(prototypes, dim=-1)
    return prototypes, labels, dict(counts)


def write_report(
    path: Path,
    *,
    manifest_paths: list[Path],
    output_path: Path,
    labels: list[int],
    class_name_map: dict[int, str],
    manifest_counts: Counter[int],
    prototype_counts: dict[int, int],
) -> None:
    lines = [
        "# Semantic Caption Prototype Bank",
        "",
        "This artifact supports controlled captioning only: `a photo of a {class_name}`.",
        "",
        f"- Output bank: `{output_path}`",
        f"- Manifests: `{', '.join(str(path) for path in manifest_paths)}`",
        f"- Classes: `{len(labels)}`",
        f"- Manifest rows: `{sum(manifest_counts.values())}`",
        f"- Prototype rows from CLIP cache: `{sum(prototype_counts.values())}`",
        "",
        "## Class Map Preview",
        "",
        "| Label | Class Name | Manifest Rows | Prototype Rows |",
        "| ---: | --- | ---: | ---: |",
    ]
    for label in labels[:30]:
        lines.append(
            f"| {label} | {class_name_map[label]} | {manifest_counts[label]} | {prototype_counts.get(label, 0)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_prototypes(
    data_root: Path,
    output_path: Path,
    *,
    splits: list[str],
    clip_prefix: str,
    report_path: Path,
) -> dict[str, Any]:
    manifest_paths = [data_root / f"{split}_human_caption.jsonl" for split in splits]
    missing = [path for path in manifest_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing human-caption manifests: {missing}")

    class_name_map, class_wnid_map, manifest_counts = build_class_maps(manifest_paths)
    image_prototypes, labels, prototype_counts = build_image_prototypes(data_root, manifest_paths, splits, clip_prefix, class_name_map)
    captions = [f"a photo of a {class_name_map[label]}" for label in labels]
    bank = {
        "labels": labels,
        "class_name_map": {str(k): v for k, v in class_name_map.items()},
        "class_wnid_map": {str(k): v for k, v in class_wnid_map.items()},
        "caption_templates": {str(label): caption for label, caption in zip(labels, captions, strict=True)},
        "image_prototypes": image_prototypes,
        "prototype_counts": {str(k): int(v) for k, v in prototype_counts.items()},
        "source_splits": splits,
        "clip_prefix": clip_prefix,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, output_path)
    write_report(
        report_path,
        manifest_paths=manifest_paths,
        output_path=output_path,
        labels=labels,
        class_name_map=class_name_map,
        manifest_counts=manifest_counts,
        prototype_counts=prototype_counts,
    )
    return {"classes": len(labels), "rows": sum(manifest_counts.values()), "output": str(output_path)}


def self_test() -> None:
    out = Path("outputs/semantic_caption/self_test_prototypes.pt")
    report = Path("outputs/semantic_caption/self_test_prototypes.md")
    stats = build_prototypes(
        Path("data/thought2text"),
        out,
        splits=["train"],
        clip_prefix="clip",
        report_path=report,
    )
    bank = torch.load(out, map_location="cpu", weights_only=False)
    assert stats["classes"] > 0
    assert bank["image_prototypes"].ndim == 2
    assert len(bank["labels"]) == bank["image_prototypes"].shape[0]
    print(json.dumps(stats, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build class text and image prototype bank for controlled captioning.")
    parser.add_argument("--data_root", default="data/thought2text")
    parser.add_argument("--output", default="outputs/semantic_caption/prototypes.pt")
    parser.add_argument("--report", default="outputs/semantic_caption/prototype_bank.md")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--clip_prefix", default="clip")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    stats = build_prototypes(
        Path(args.data_root),
        Path(args.output),
        splits=args.splits,
        clip_prefix=args.clip_prefix,
        report_path=Path(args.report),
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
