from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class RawSession:
    path: Path
    split: str
    subject_id: str
    session_id: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _discover_sessions(root: Path) -> list[RawSession]:
    sessions: list[RawSession] = []
    for path in sorted((root / "raw-eeg").glob("sub-*/ses-*/raw_eeg_*.npy")):
        name = path.name.lower()
        if "training" in name:
            split = "train"
        elif "test" in name:
            split = "val"
        else:
            continue
        sessions.append(RawSession(path=path, split=split, subject_id=path.parents[1].name, session_id=path.parent.name))
    return sessions


def _load_raw_array(path: Path, channels: int | None = None) -> np.ndarray:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()
    if isinstance(payload, dict):
        if "raw_eeg_data" not in payload:
            raise KeyError(f"{path} has no raw_eeg_data key")
        arr = payload["raw_eeg_data"]
    else:
        arr = payload
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{path} raw EEG must be [channels, time], got {arr.shape}")
    if channels is not None and arr.shape[0] != channels:
        raise ValueError(f"{path} expected {channels} channels, got {arr.shape[0]}")
    return arr


def _iter_windows(
    sessions: Iterable[RawSession],
    *,
    window_size: int,
    stride: int,
    channels: int | None,
) -> Iterable[tuple[RawSession, int, np.ndarray]]:
    for session in sessions:
        arr = _load_raw_array(session.path, channels=channels)
        if arr.shape[1] < window_size:
            continue
        for start in range(0, arr.shape[1] - window_size + 1, stride):
            yield session, start, arr[:, start : start + window_size]


def _count_windows(sessions: list[RawSession], *, window_size: int, stride: int, channels: int | None, limit: int) -> int:
    if limit > 0:
        return limit
    count = 0
    for session in sessions:
        arr = _load_raw_array(session.path, channels=channels)
        count += max(0, (arr.shape[1] - window_size) // stride + 1)
        if limit > 0 and count >= limit:
            return limit
    return count


def _write_cache(
    sessions: list[RawSession],
    *,
    out_path: Path,
    index_path: Path,
    window_size: int,
    stride: int,
    channels: int,
    limit: int,
) -> int:
    count = _count_windows(sessions, window_size=window_size, stride=stride, channels=channels, limit=limit)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "cache_start", "cache": str(out_path), "sessions": len(sessions), "target_rows": count}), flush=True)
    cache = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(count, channels, window_size))
    index_rows: list[dict[str, Any]] = []
    row_idx = 0
    for session, start, window in _iter_windows(sessions, window_size=window_size, stride=stride, channels=channels):
        if row_idx == 0:
            print(json.dumps({"event": "first_window", "cache": str(out_path), "source": str(session.path), "start": start}), flush=True)
        if row_idx >= count:
            break
        cache[row_idx] = window
        index_rows.append(
            {
                "row": row_idx,
                "source": str(session.path),
                "subject_id": session.subject_id,
                "session_id": session.session_id,
                "split": session.split,
                "start": start,
                "stop": start + window_size,
            }
        )
        row_idx += 1
        if row_idx % 10000 == 0:
            print(json.dumps({"cache": str(out_path), "rows": row_idx, "target": count}), flush=True)
    cache.flush()
    index_path.write_text("\n".join(json.dumps(row) for row in index_rows) + ("\n" if index_rows else ""), encoding="utf-8")
    return row_idx


def build_raw_window_cache(
    *,
    root: str | Path,
    out_dir: str | Path,
    window_size: int = 250,
    stride: int = 125,
    max_train_windows: int = 0,
    max_val_windows: int = 0,
    channels: int = 64,
) -> dict[str, Any]:
    root = Path(root)
    out_dir = Path(out_dir)
    sessions = _discover_sessions(root)
    train_sessions = [session for session in sessions if session.split == "train"]
    val_sessions = [session for session in sessions if session.split == "val"]
    if not train_sessions or not val_sessions:
        raise FileNotFoundError(f"Expected raw_eeg_training.npy and raw_eeg_test.npy files under {root / 'raw-eeg'}")

    train_cache = out_dir / "things_eeg2_train_windows.npy"
    val_cache = out_dir / "things_eeg2_val_windows.npy"
    train_windows = _write_cache(
        train_sessions,
        out_path=train_cache,
        index_path=out_dir / "things_eeg2_train_windows.jsonl",
        window_size=window_size,
        stride=stride,
        channels=channels,
        limit=max_train_windows,
    )
    val_windows = _write_cache(
        val_sessions,
        out_path=val_cache,
        index_path=out_dir / "things_eeg2_val_windows.jsonl",
        window_size=window_size,
        stride=stride,
        channels=channels,
        limit=max_val_windows,
    )
    stats = {
        "created_at": _utc_now(),
        "root": str(root),
        "out_dir": str(out_dir),
        "window_size": window_size,
        "stride": stride,
        "channels": channels,
        "train_sessions": len(train_sessions),
        "val_sessions": len(val_sessions),
        "train_windows": train_windows,
        "val_windows": val_windows,
        "train_cache": str(train_cache),
        "val_cache": str(val_cache),
        "note": "Raw continuous EEG windows for masked EEG pretraining only; this is not trial/image alignment.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "things_eeg2_window_cache_manifest.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    report = [
        "# THINGS-EEG2 Raw Window Cache",
        "",
        f"- Created UTC: `{stats['created_at']}`",
        f"- Root: `{root}`",
        f"- Train sessions: `{len(train_sessions)}`",
        f"- Val/test sessions: `{len(val_sessions)}`",
        f"- Channels: `{channels}`",
        f"- Window size: `{window_size}`",
        f"- Stride: `{stride}`",
        f"- Train windows: `{train_windows}`",
        f"- Val windows: `{val_windows}`",
        f"- Train cache: `{train_cache}`",
        f"- Val cache: `{val_cache}`",
        "",
        "This cache is valid for masked EEG pretraining only. It does not provide image-level or trial-level alignment.",
    ]
    (out_dir / "THINGS_EEG2_RAW_WINDOW_CACHE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed-length THINGS-EEG2 raw EEG window caches for masked pretraining.")
    parser.add_argument("--root", default="/cloud/cloud-ssd1/eeg_vision_caption_data/THINGS-EEG2")
    parser.add_argument("--out_dir", default="data/THINGS-EEG2/cache")
    parser.add_argument("--window_size", type=int, default=250)
    parser.add_argument("--stride", type=int, default=125)
    parser.add_argument("--max_train_windows", type=int, default=0)
    parser.add_argument("--max_val_windows", type=int, default=0)
    parser.add_argument("--channels", type=int, default=64)
    args = parser.parse_args()
    stats = build_raw_window_cache(
        root=args.root,
        out_dir=args.out_dir,
        window_size=args.window_size,
        stride=args.stride,
        max_train_windows=args.max_train_windows,
        max_val_windows=args.max_val_windows,
        channels=args.channels,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
