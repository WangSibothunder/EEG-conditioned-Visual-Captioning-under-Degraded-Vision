from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def convert_split(raw_dir: Path, out_dir: Path, split: str, limit: int | None) -> int:
    rows: list[dict] = []
    image_dir = out_dir / "images"
    eeg_dir = out_dir / "eeg"
    image_dir.mkdir(parents=True, exist_ok=True)
    eeg_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted((raw_dir / "data").glob(f"{split}-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for split {split} under {raw_dir / 'data'}")

    written = 0
    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=64):
            data = batch.to_pydict()
            count = len(data["caption"])
            for idx in range(count):
                if limit is not None and written >= limit:
                    write_jsonl(out_dir / f"{split}.jsonl", rows)
                    return written

                image_id = f"{split}_{written:06d}"
                image_rel = Path("images") / f"{image_id}.jpg"
                eeg_rel = Path("eeg") / f"{image_id}.npy"

                image_record = data["image"][idx]
                image_bytes = image_record.get("bytes") if isinstance(image_record, dict) else None
                if not image_bytes:
                    raise ValueError(f"Missing image bytes in {parquet_path} row {idx}")
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image.save(out_dir / image_rel, format="JPEG", quality=95)

                eeg = np.asarray(data["conditioning_image"][idx], dtype=np.float32)
                if eeg.ndim != 2:
                    raise ValueError(f"Expected 2D EEG array, got shape {eeg.shape}")
                if eeg.shape[0] != 128 and eeg.shape[1] == 128:
                    eeg = eeg.T
                if eeg.shape[0] != 128:
                    raise ValueError(f"Expected EEG channel dimension 128, got shape {eeg.shape}")
                np.save(out_dir / eeg_rel, eeg)

                rows.append(
                    {
                        "image_id": image_id,
                        "image_path": str(image_rel),
                        "eeg_path": str(eeg_rel),
                        "caption": str(data["caption"][idx]),
                        "label": int(data["label"][idx]),
                        "label_folder": str(data["label_folder"][idx]),
                        "subject": int(data["subject"][idx]),
                    }
                )
                written += 1

    write_jsonl(out_dir / f"{split}.jsonl", rows)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert EEG_Image_CVPR_ALL_subj parquet shards.")
    parser.add_argument("--raw-dir", default="data/raw/eeg_image_cvpr_all_subj")
    parser.add_argument("--out-dir", default="data/real/eeg_image_cvpr_sample")
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--val-limit", type=int, default=64)
    parser.add_argument("--test-limit", type=int, default=64)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "train": convert_split(raw_dir, out_dir, "train", args.train_limit),
        "validation": convert_split(raw_dir, out_dir, "validation", args.val_limit),
        "test": convert_split(raw_dir, out_dir, "test", args.test_limit),
    }
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
