from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        handle.flush()


def _load_caption_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id", ""))
            caption = str(row.get("caption", "")).strip()
            if image_id and caption:
                cache[image_id] = row
    return cache


def _resolve_image_path(manifest: Path, row: dict[str, Any]) -> Path:
    image_path = Path(str(row.get("image_path", "")))
    if image_path.is_absolute():
        return image_path
    return manifest.parent / image_path


def _unique_images(manifests: list[Path]) -> list[dict[str, Any]]:
    by_image: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        for row in _read_jsonl(manifest):
            image_id = str(row.get("image_id", ""))
            if not image_id or image_id in by_image:
                continue
            by_image[image_id] = {
                "image_id": image_id,
                "image_path": str(_resolve_image_path(manifest, row)),
                "reference": row.get("caption", ""),
                "human_label_name": row.get("human_label_name"),
            }
    return list(by_image.values())


def _load_blip(model_name_or_path: str, device: str):
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(model_name_or_path, local_files_only=Path(model_name_or_path).exists())
    model = BlipForConditionalGeneration.from_pretrained(
        model_name_or_path,
        local_files_only=Path(model_name_or_path).exists(),
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    )
    model.to(device)
    model.eval()
    return processor, model


@torch.inference_mode()
def _caption_batch(
    processor: Any,
    model: Any,
    batch: list[dict[str, Any]],
    *,
    device: str,
    max_new_tokens: int,
    num_beams: int,
) -> list[str]:
    images = [Image.open(row["image_path"]).convert("RGB") for row in batch]
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    if device.startswith("cuda"):
        inputs = {key: value.half() if torch.is_floating_point(value) else value for key, value in inputs.items()}
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
    )
    captions = processor.batch_decode(generated, skip_special_tokens=True)
    return [" ".join(caption.strip().split()) for caption in captions]


def generate_captions(
    manifests: list[str | Path],
    *,
    caption_cache_path: str | Path,
    model_name_or_path: str,
    batch_size: int,
    max_images: int | None,
    overwrite: bool,
    device: str,
    max_new_tokens: int,
    num_beams: int,
) -> dict[str, Any]:
    manifest_paths = [Path(path) for path in manifests]
    unique = _unique_images(manifest_paths)
    if max_images is not None and max_images >= 0:
        unique = unique[:max_images]

    caption_cache_path = Path(caption_cache_path)
    if overwrite and caption_cache_path.exists():
        caption_cache_path.unlink()
    cache = _load_caption_cache(caption_cache_path)
    pending = [row for row in unique if row["image_id"] not in cache]

    processor, model = _load_blip(model_name_or_path, device)
    generated = 0
    failed: list[dict[str, str]] = []
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        usable: list[dict[str, Any]] = []
        for row in batch:
            if Path(row["image_path"]).exists():
                usable.append(row)
            else:
                failed.append({"image_id": row["image_id"], "reason": f"missing image: {row['image_path']}"})
        if not usable:
            continue
        try:
            captions = _caption_batch(
                processor,
                model,
                usable,
                device=device,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        except Exception as exc:  # keep long overnight jobs moving
            for row in usable:
                failed.append({"image_id": row["image_id"], "reason": repr(exc)})
            continue
        for row, caption in zip(usable, captions, strict=False):
            payload = {
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "caption": caption,
                "caption_source": "blip",
                "human_label_name": row.get("human_label_name"),
            }
            cache[row["image_id"]] = payload
            _append_jsonl(caption_cache_path, payload)
            generated += 1

    return {
        "requested_unique_images": len(unique),
        "cached_before": len(unique) - len(pending),
        "generated": generated,
        "cached_after": len(cache),
        "failed": failed,
    }


def build_blip_manifest(source: str | Path, target: str | Path, caption_cache_path: str | Path) -> dict[str, int]:
    cache = _load_caption_cache(caption_cache_path)
    rows = _read_jsonl(source)
    converted: list[dict[str, Any]] = []
    used = 0
    missing = 0
    for row in rows:
        new_row = dict(row)
        image_id = str(row.get("image_id", ""))
        cached = cache.get(image_id)
        if cached and cached.get("caption"):
            new_row["caption"] = cached["caption"]
            new_row["caption_source"] = "blip"
            used += 1
        else:
            missing += 1
        converted.append(new_row)
    _write_jsonl(target, converted)
    return {"rows": len(rows), "blip_used": used, "missing": missing}


def write_report(path: str | Path, stats: dict[str, Any], split_stats: dict[str, dict[str, int]], caption_cache_path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    failed = stats.get("failed", [])
    lines = [
        "# BLIP Caption Report",
        "",
        f"- Caption cache: `{caption_cache_path}`",
        f"- Requested unique images: `{stats.get('requested_unique_images', 0)}`",
        f"- Cached before: `{stats.get('cached_before', 0)}`",
        f"- Generated this run: `{stats.get('generated', 0)}`",
        f"- Cached after: `{stats.get('cached_after', 0)}`",
        f"- Failed images: `{len(failed)}`",
        "",
        "| Split | Rows | BLIP captions used | Missing BLIP captions |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split, row in split_stats.items():
        lines.append(f"| {split} | {row['rows']} | {row['blip_used']} | {row['missing']} |")
    if failed:
        lines.extend(["", "## Failures", ""])
        for row in failed[:20]:
            lines.append(f"- `{row['image_id']}`: {row['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BLIP captions for Thought2Text unique images and build BLIP manifests.")
    parser.add_argument("--root", default="data/thought2text")
    parser.add_argument("--model", default="data/model_cache/blip-image-captioning-base")
    parser.add_argument("--caption_cache", default="data/thought2text/blip_captions.jsonl")
    parser.add_argument("--report", default="outputs/day4_caption_targets/blip_caption_report.md")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_images", type=int, default=-1, help="Limit unique images for smoke tests; -1 means all.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--num_beams", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.root)
    manifests = [root / f"{split}_human_caption.jsonl" for split in ["train", "val", "test"]]
    max_images = None if args.max_images is None or args.max_images < 0 else args.max_images
    stats = generate_captions(
        manifests,
        caption_cache_path=args.caption_cache,
        model_name_or_path=args.model,
        batch_size=args.batch_size,
        max_images=max_images,
        overwrite=args.overwrite,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    split_stats: dict[str, dict[str, int]] = {}
    for split in ["train", "val", "test"]:
        split_stats[split] = build_blip_manifest(
            root / f"{split}_human_caption.jsonl",
            root / f"{split}_blip_caption.jsonl",
            args.caption_cache,
        )
    write_report(args.report, stats, split_stats, args.caption_cache)
    print(args.report)


if __name__ == "__main__":
    main()
