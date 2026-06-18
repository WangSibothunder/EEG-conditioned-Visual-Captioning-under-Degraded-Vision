from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.train.train_align import _build_loader


class OvernightPipelineCliTests(unittest.TestCase):
    def test_alignment_loader_does_not_require_image_files_when_clip_cache_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            eeg_path = root / "eeg.pth"
            torch.save({"eeg": torch.randn(2, 64, 250)}, eeg_path)
            manifest = root / "train.jsonl"
            rows = [
                {
                    "image_id": f"sample_{idx}",
                    "image_path": f"missing_{idx}.jpg",
                    "eeg_path": "eeg.pth",
                    "eeg_index": idx,
                    "caption": "a photo of an object",
                    "label": idx,
                }
                for idx in range(2)
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            cache = root / "clip.npy"
            np.save(cache, np.random.randn(2, 512).astype("float16"))
            index = root / "clip_index.json"
            index.write_text(json.dumps([{"image_id": row["image_id"]} for row in rows]), encoding="utf-8")

            loader = _build_loader(
                manifest=manifest,
                cache=cache,
                index=index,
                batch_size=2,
                max_samples=0,
                eeg_shape=(64, 250),
                shuffle=False,
                num_workers=0,
            )
            batch = next(iter(loader))

        self.assertEqual(tuple(batch["eeg"].shape), (2, 64, 250))
        self.assertEqual(tuple(batch["clip_emb"].shape), (2, 512))
        self.assertEqual(batch["image_id"], ["sample_0", "sample_1"])

    def test_metrics_cli_accepts_goal_pred_dir_and_out_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pred_dir = root / "preds"
            pred_dir.mkdir()
            (pred_dir / "clean_vision_only.jsonl").write_text(
                json.dumps(
                    {
                        "image_id": "a",
                        "corruption": "clean",
                        "mode": "vision_only",
                        "reference": "a red object",
                        "prediction": "a red object",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            out = root / "metrics.md"

            result = subprocess.run(
                [sys.executable, "-m", "src.eval.metrics", "--pred_dir", str(pred_dir), "--out", str(out)],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertTrue(out.exists())
            self.assertIn("# Sanity Metrics", out.read_text(encoding="utf-8"))

    def test_goal_cli_help_options_are_present(self) -> None:
        root = Path(__file__).resolve().parents[1]
        retrieval = subprocess.run(
            [sys.executable, "-m", "src.eval.retrieval", "--help"],
            cwd=root,
            check=False,
            text=True,
            capture_output=True,
        )
        sanity = subprocess.run(
            [sys.executable, "-m", "src.eval.sanity_check", "--help"],
            cwd=root,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(retrieval.returncode, 0, retrieval.stderr)
        self.assertIn("--manifest", retrieval.stdout)
        self.assertIn("--clip_cache", retrieval.stdout)
        self.assertIn("--eeg_ckpt", retrieval.stdout)
        self.assertEqual(sanity.returncode, 0, sanity.stderr)
        self.assertIn("--max_samples", sanity.stdout)


if __name__ == "__main__":
    unittest.main()
