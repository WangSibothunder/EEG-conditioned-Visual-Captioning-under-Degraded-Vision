from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import EEGVisionCaptionDataset


def build_cache(
    *,
    manifest: str | Path,
    out: str | Path,
    eeg_shape: tuple[int, int],
    report: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest)
    out_path = Path(out)
    dataset = EEGVisionCaptionDataset(
        manifest_path,
        image_size=8,
        eeg_shape=eeg_shape,
        allow_missing_images=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), eeg_shape[0], eeg_shape[1]),
    )
    for idx, row in enumerate(dataset.rows):
        eeg_path = dataset._resolve_path(str(row["eeg_path"]))
        cache[idx] = dataset._load_eeg(eeg_path).numpy()
    cache.flush()

    stats: dict[str, Any] = {
        "manifest": str(manifest_path),
        "out": str(out_path),
        "rows": len(dataset),
        "shape": [len(dataset), eeg_shape[0], eeg_shape[1]],
        "dtype": "float32",
        "bytes": out_path.stat().st_size,
    }
    if report is not None:
        write_report(report, stats)
    return stats


def write_report(path: str | Path, stats: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manifest EEG Cache Report",
        "",
        f"- Manifest: `{stats['manifest']}`",
        f"- Output: `{stats['out']}`",
        f"- Rows: `{stats['rows']}`",
        f"- Shape: `{stats['shape']}`",
        f"- dtype: `{stats['dtype']}`",
        f"- File size: `{stats['bytes']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a row-aligned EEG .npy cache from a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eeg_channels", type=int, default=64)
    parser.add_argument("--eeg_timesteps", type=int, default=250)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    stats = build_cache(
        manifest=args.manifest,
        out=args.out,
        eeg_shape=(args.eeg_channels, args.eeg_timesteps),
        report=args.report,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
