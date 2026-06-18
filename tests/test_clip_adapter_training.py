from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_manifest(path: Path, split: str, count: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(count):
            row = {
                "image_id": f"{split}_{idx:03d}",
                "image_path": f"images/{split}_{idx:03d}.jpg",
                "eeg_path": f"eeg/{split}_{idx:03d}.npy",
                "caption": f"class {idx % 3}",
                "label": idx % 3,
                "subject_id": "S01",
                "split": split,
            }
            handle.write(json.dumps(row) + "\n")


def _write_cache(path: Path, index_path: Path, count: int, dim: int) -> None:
    rng = np.random.default_rng(123)
    emb = rng.normal(size=(count, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    np.save(path, emb.astype(np.float16))
    index = [{"image_id": f"{path.stem.split('_')[-1]}_{idx:03d}", "caption": f"class {idx % 3}"} for idx in range(count)]
    index_path.write_text(json.dumps(index), encoding="utf-8")


def _write_eeg(root: Path, split: str, count: int) -> None:
    eeg_dir = root / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(456)
    for idx in range(count):
        np.save(eeg_dir / f"{split}_{idx:03d}.npy", rng.normal(size=(64, 250)).astype(np.float32))


class ClipAdapterTrainingTests(unittest.TestCase):
    def test_clip_adapter_training_writes_required_artifacts(self) -> None:
        from src.train.train_clip_adapter import train

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_manifest = tmp_path / "train.jsonl"
            val_manifest = tmp_path / "val.jsonl"
            _write_manifest(train_manifest, "train", 12)
            _write_manifest(val_manifest, "val", 6)
            _write_eeg(tmp_path, "train", 12)
            _write_eeg(tmp_path, "val", 6)

            cache_dir = tmp_path / "cache"
            cache_dir.mkdir()
            train_cache = cache_dir / "clip_train.npy"
            val_cache = cache_dir / "clip_val.npy"
            train_index = cache_dir / "clip_index_train.json"
            val_index = cache_dir / "clip_index_val.json"
            _write_cache(train_cache, train_index, 12, 16)
            _write_cache(val_cache, val_index, 6, 16)

            out_dir = tmp_path / "outputs" / "adapter"
            config = {
                "seed": 7,
                "device": "cpu",
                "data": {
                    "train_manifest": str(train_manifest),
                    "val_manifest": str(val_manifest),
                    "clip_train_cache": str(train_cache),
                    "clip_val_cache": str(val_cache),
                    "clip_index_train": str(train_index),
                    "clip_index_val": str(val_index),
                },
                "model": {
                    "clip_dim": 16,
                    "adapter_hidden_dim": 32,
                    "dropout": 0.0,
                },
                "train": {
                    "epochs": 1,
                    "batch_size": 4,
                    "num_workers": 0,
                    "lr": 1.0e-3,
                    "bf16": False,
                },
                "output": {"dir": str(out_dir)},
            }

            metrics = train(config)

            self.assertTrue(torch.isfinite(torch.tensor(metrics["val"]["loss"])))
            self.assertTrue((out_dir / "config.yaml").exists())
            self.assertTrue((out_dir / "history.json").exists())
            self.assertTrue((out_dir / "metrics.json").exists())
            self.assertTrue((out_dir / "CLIP_ADAPTER_REPORT.md").exists())
            self.assertTrue((out_dir / "checkpoints" / "best.pt").exists())
            saved_metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("retrieval", saved_metrics["val"])
            self.assertIn("class_acc", saved_metrics["val"])


if __name__ == "__main__":
    unittest.main()
