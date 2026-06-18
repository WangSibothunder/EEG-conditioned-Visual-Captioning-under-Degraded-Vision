from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_things_eeg2_raw_window_cache import build_raw_window_cache


class ThingsEEG2RawWindowCacheTest(unittest.TestCase):
    def test_builds_train_and_val_caches_from_raw_session_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "THINGS-EEG2"
            session = root / "raw-eeg" / "sub-01" / "ses-01"
            session.mkdir(parents=True)
            train = np.arange(4 * 1000, dtype=np.float32).reshape(4, 1000)
            test = np.ones((4, 500), dtype=np.float32)
            np.save(session / "raw_eeg_training.npy", {"raw_eeg_data": train, "sfreq": 1000})
            np.save(session / "raw_eeg_test.npy", {"raw_eeg_data": test, "sfreq": 1000})

            out = Path(tmp) / "cache"
            stats = build_raw_window_cache(
                root=root,
                out_dir=out,
                window_size=250,
                stride=250,
                max_train_windows=3,
                max_val_windows=2,
                channels=4,
            )

            train_cache = np.load(out / "things_eeg2_train_windows.npy", mmap_mode="r")
            val_cache = np.load(out / "things_eeg2_val_windows.npy", mmap_mode="r")
            self.assertEqual(tuple(train_cache.shape), (3, 4, 250))
            self.assertEqual(tuple(val_cache.shape), (2, 4, 250))
            np.testing.assert_array_equal(train_cache[0], train[:, :250])
            self.assertEqual(stats["train_windows"], 3)
            self.assertEqual(stats["val_windows"], 2)

            manifest = json.loads((out / "things_eeg2_window_cache_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["window_size"], 250)
            self.assertEqual(manifest["channels"], 4)


if __name__ == "__main__":
    unittest.main()
