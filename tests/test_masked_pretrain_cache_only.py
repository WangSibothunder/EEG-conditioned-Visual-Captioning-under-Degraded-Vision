from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.train.train_masked_eeg_pretrain import resolve_eeg_cache


class MaskedPretrainCacheOnlyTest(unittest.TestCase):
    def test_resolve_eeg_cache_accepts_existing_cache_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.npy"
            np.save(cache, np.zeros((3, 4, 16), dtype=np.float32))

            resolved = resolve_eeg_cache(
                split_name="train",
                data_cfg={"train_eeg_cache": str(cache)},
                cache_key="train_eeg_cache",
                manifest_key="train_manifest",
                eeg_shape=(4, 16),
            )

            self.assertEqual(resolved, cache)

    def test_resolve_eeg_cache_rejects_wrong_shape_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.npy"
            np.save(cache, np.zeros((3, 4, 15), dtype=np.float32))

            with self.assertRaises(ValueError):
                resolve_eeg_cache(
                    split_name="train",
                    data_cfg={"train_eeg_cache": str(cache)},
                    cache_key="train_eeg_cache",
                    manifest_key="train_manifest",
                    eeg_shape=(4, 16),
                )


if __name__ == "__main__":
    unittest.main()
