from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class Day3RealEvalTests(unittest.TestCase):
    def test_precompute_degraded_vision_cli_writes_caches_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "images"
            eeg_dir = root / "eeg"
            image_dir.mkdir()
            eeg_dir.mkdir()
            for index, color in enumerate([(255, 0, 0), (0, 255, 0)]):
                Image.new("RGB", (24, 24), color=color).save(image_dir / f"{index}.jpg")
                np.save(eeg_dir / f"{index}.npy", np.zeros((64, 250), dtype=np.float32))
            manifest = root / "test.jsonl"
            rows = [
                {
                    "image_id": str(index),
                    "image_path": f"images/{index}.jpg",
                    "eeg_path": f"eeg/{index}.npy",
                    "caption": "a photo of an object",
                    "label": index,
                    "split": "test",
                }
                for index in range(2)
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            out_dir = root / "cache" / "degraded_test"
            report = root / "degraded_clip_cache_report.md"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/precompute_degraded_vision.py",
                    "--manifest",
                    str(manifest),
                    "--image_root",
                    str(root),
                    "--corruptions",
                    "clean",
                    "lowres",
                    "--out_dir",
                    str(out_dir),
                    "--report",
                    str(report),
                    "--use_tiny_debug_model",
                    "--batch_size",
                    "2",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual(np.load(out_dir / "clip_test_clean.npy").shape, (2, 512))
            self.assertEqual(np.load(out_dir / "clip_test_lowres.npy").shape, (2, 512))
            self.assertTrue((out_dir / "clip_index_test_clean.json").exists())
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("clean", report_text)
            self.assertIn("lowres", report_text)

    def test_sanity_helpers_prefer_checkpoint_config_and_include_label_gate(self) -> None:
        from src.eval.sanity_check import build_prediction_record, load_sanity_config

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "best.pt"
            torch.save(
                {
                    "config": {
                        "seed": 123,
                        "device": "cpu",
                        "data": {"root": str(root), "image_size": 224, "eeg_channels": 64, "eeg_timesteps": 250},
                        "model": {"use_tiny_debug_model": False, "llm_model": "checkpoint-llm", "image_dim": 512, "eeg_dim": 512},
                        "train": {"batch_size": 3},
                        "generation": {"max_new_tokens": 7},
                    }
                },
                checkpoint,
            )

            cfg = load_sanity_config("configs/debug.yaml", str(checkpoint))
            self.assertEqual(cfg["seed"], 123)
            self.assertEqual(cfg["model"]["llm_model"], "checkpoint-llm")
            self.assertFalse(cfg["model"]["use_tiny_debug_model"])
            record = build_prediction_record(
                image_id="img0",
                corruption="blur",
                mode="real_eeg",
                reference="a photo of an object",
                prediction="a photo of an object",
                label=4,
                gate_mean=0.25,
            )
            self.assertEqual(record["label"], 4)
            self.assertEqual(record["gate_mean"], 0.25)

    def test_sanity_degraded_clip_cache_loader_validates_and_slices(self) -> None:
        from src.eval.sanity_check import load_degraded_clip_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            np.save(cache_dir / "clip_test_clean.npy", np.zeros((3, 512), dtype=np.float16))

            cache = load_degraded_clip_cache(cache_dir, "clean", expected_len=3, max_samples=2)

        self.assertEqual(tuple(cache.shape), (2, 512))
        self.assertEqual(cache.dtype, torch.float32)

    def test_sanity_degraded_clip_cache_loader_rejects_length_mismatch(self) -> None:
        from src.eval.sanity_check import load_degraded_clip_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            np.save(cache_dir / "clip_test_clean.npy", np.zeros((2, 512), dtype=np.float16))

            with self.assertRaisesRegex(ValueError, "length mismatch"):
                load_degraded_clip_cache(cache_dir, "clean", expected_len=3, max_samples=0)

    def test_metrics_cli_includes_gate_mean_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pred_dir = root / "preds"
            pred_dir.mkdir()
            records = [
                {
                    "image_id": "a",
                    "corruption": "blur",
                    "mode": "real_eeg",
                    "reference": "a photo of an object",
                    "prediction": "a photo of an object",
                    "label": 1,
                    "gate_mean": 0.2,
                },
                {
                    "image_id": "b",
                    "corruption": "blur",
                    "mode": "real_eeg",
                    "reference": "a photo of an object",
                    "prediction": "object",
                    "label": 2,
                    "gate_mean": 0.4,
                },
            ]
            (pred_dir / "blur_real_eeg.jsonl").write_text(
                "\n".join(json.dumps(row) for row in records) + "\n",
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
            text = out.read_text(encoding="utf-8")
            self.assertIn("gate_mean", text)
            self.assertIn("0.3000", text)
            self.assertNotIn("sample_predictions.jsonl", text)

    def test_metrics_and_gate_ignore_sample_predictions_file(self) -> None:
        from src.eval.gate_analysis import load_prediction_records
        from src.eval.metrics import evaluate_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pred_dir = root / "preds"
            pred_dir.mkdir()
            valid_record = {
                "image_id": "a",
                "corruption": "clean",
                "mode": "real_eeg",
                "reference": "a photo of a canoe",
                "prediction": "a canoe",
                "gate_mean": 0.2,
            }
            sample_record = {
                "image_id": "sample",
                "corruption": "sample",
                "mode": "sample",
                "reference": "sample",
                "prediction": "sample",
                "gate_mean": 0.9,
            }
            (pred_dir / "clean_real_eeg.jsonl").write_text(json.dumps(valid_record) + "\n", encoding="utf-8")
            (pred_dir / "sample_predictions.jsonl").write_text(json.dumps(sample_record) + "\n", encoding="utf-8")

            evaluate_directory(pred_dir, root / "metrics.csv", root / "metrics.md")
            records = load_prediction_records(pred_dir)
            metrics_text = (root / "metrics.md").read_text(encoding="utf-8")

            self.assertIn("clean_real_eeg.jsonl", metrics_text)
            self.assertNotIn("sample_predictions.jsonl", metrics_text)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["image_id"], "a")

    def test_metrics_cli_includes_class_hit_when_human_label_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pred_dir = root / "preds"
            pred_dir.mkdir()
            records = [
                {
                    "image_id": "a",
                    "corruption": "clean",
                    "mode": "real_eeg",
                    "reference": "a photo of a grand piano",
                    "prediction": "a blurry grand piano",
                    "human_label_name": "grand piano",
                },
                {
                    "image_id": "b",
                    "corruption": "clean",
                    "mode": "real_eeg",
                    "reference": "a photo of a canoe",
                    "prediction": "a red car",
                    "human_label_name": "canoe",
                },
            ]
            (pred_dir / "clean_real_eeg.jsonl").write_text(
                "\n".join(json.dumps(row) for row in records) + "\n",
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
            text = out.read_text(encoding="utf-8")
            self.assertIn("class_hit", text)
            self.assertIn("0.5000", text)

    def test_gate_analysis_cli_writes_report_and_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pred_dir = root / "preds"
            pred_dir.mkdir()
            records = [
                {
                    "image_id": "a",
                    "corruption": "noise",
                    "mode": "real_eeg",
                    "reference": "a photo of an object",
                    "prediction": "a photo of an object",
                    "label": 1,
                    "gate_mean": 0.25,
                },
                {
                    "image_id": "b",
                    "corruption": "noise",
                    "mode": "shuffled_eeg",
                    "reference": "a photo of an object",
                    "prediction": "object",
                    "label": 2,
                    "gate_mean": 0.55,
                },
            ]
            (pred_dir / "noise_modes.jsonl").write_text(
                "\n".join(json.dumps(row) for row in records) + "\n",
                encoding="utf-8",
            )
            report = root / "gate_analysis.md"
            samples = root / "sample_predictions.jsonl"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.eval.gate_analysis",
                    "--pred_dir",
                    str(pred_dir),
                    "--out",
                    str(report),
                    "--sample_out",
                    str(samples),
                    "--sample_limit",
                    "1",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("noise", report.read_text(encoding="utf-8"))
            self.assertIn("0.2500", report.read_text(encoding="utf-8"))
            self.assertEqual(len(samples.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
