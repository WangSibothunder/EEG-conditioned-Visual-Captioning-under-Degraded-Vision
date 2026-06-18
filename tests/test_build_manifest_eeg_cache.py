from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import build_manifest_eeg_cache


class BuildManifestEEGCacheTests(unittest.TestCase):
    def test_builds_cache_in_manifest_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "eeg").mkdir()
            rows = []
            for idx in [2, 0, 1]:
                arr = np.full((2, 3), idx, dtype=np.float32)
                np.save(root / "eeg" / f"sample_{idx}.npy", arr)
                rows.append(
                    {
                        "image_id": f"img{idx}",
                        "image_path": f"images/img{idx}.jpg",
                        "eeg_path": f"eeg/sample_{idx}.npy",
                        "caption": "x",
                        "label": idx,
                    }
                )
            manifest = root / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            out = root / "cache.npy"
            report = root / "report.md"

            stats = build_manifest_eeg_cache.build_cache(
                manifest=manifest,
                out=out,
                eeg_shape=(2, 3),
                report=report,
            )

            cached = np.load(out)
            report_text = report.read_text(encoding="utf-8")

        self.assertEqual(stats["rows"], 3)
        self.assertEqual(stats["shape"], [3, 2, 3])
        self.assertEqual(tuple(cached.shape), (3, 2, 3))
        self.assertTrue(np.all(cached[0] == 2))
        self.assertTrue(np.all(cached[1] == 0))
        self.assertTrue(np.all(cached[2] == 1))
        self.assertIn("Manifest EEG Cache Report", report_text)


if __name__ == "__main__":
    unittest.main()
