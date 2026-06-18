from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class Day4CaptionTargetTests(unittest.TestCase):
    def test_imagenet_lookup_returns_human_class_name_for_wnid(self) -> None:
        from src.data.imagenet_labels import caption_for_wnid, human_name_for_wnid

        self.assertEqual(human_name_for_wnid("n03452741"), "grand piano")
        self.assertEqual(caption_for_wnid("n03452741"), "a photo of a grand piano")
        self.assertEqual(caption_for_wnid("n02607072"), "a photo of an anemone fish")
        self.assertEqual(human_name_for_wnid("n99999999"), None)
        self.assertEqual(caption_for_wnid("n99999999"), "a photo of an object from class n99999999")

    def test_human_caption_manifest_rewrites_caption_and_reports_examples(self) -> None:
        from scripts.build_human_caption_manifest import convert_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "train.jsonl"
            target = root / "train_human_caption.jsonl"
            report = root / "report.md"
            rows = [
                {
                    "image_id": "n03452741_1",
                    "image_path": "images/n03452741_1.jpg",
                    "eeg_path": "block/eeg.pth",
                    "eeg_index": 0,
                    "caption": "a photo of an object from class n03452741",
                    "label": 1,
                    "label_name": "n03452741",
                    "split": "train",
                },
                {
                    "image_id": "n99999999_1",
                    "image_path": "images/n99999999_1.jpg",
                    "eeg_path": "block/eeg.pth",
                    "eeg_index": 1,
                    "caption": "a photo of an object from class n99999999",
                    "label": 2,
                    "label_name": "n99999999",
                    "split": "train",
                },
            ]
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            stats = convert_manifest(source, target, report_path=report)

            converted = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(converted[0]["caption"], "a photo of a grand piano")
            self.assertEqual(converted[0]["caption_source"], "human_class")
            self.assertEqual(converted[0]["human_label_name"], "grand piano")
            self.assertEqual(converted[1]["caption_source"], "wnid_fallback")
            self.assertEqual(stats["converted"], 1)
            self.assertEqual(stats["unknown"], 1)
            text = report.read_text(encoding="utf-8")
            self.assertIn("grand piano", text)
            self.assertIn("n99999999", text)

    def test_blip_manifest_uses_cached_caption_when_available(self) -> None:
        from scripts.generate_blip_captions import build_blip_manifest, write_report

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "test_human_caption.jsonl"
            target = root / "test_blip_caption.jsonl"
            cache = root / "blip_captions.jsonl"
            report = root / "report.md"
            source.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {
                            "image_id": "n03452741_1",
                            "image_path": "images/n03452741_1.jpg",
                            "caption": "a photo of a grand piano",
                            "caption_source": "human_class",
                        },
                        {
                            "image_id": "n99999999_1",
                            "image_path": "images/n99999999_1.jpg",
                            "caption": "a photo of an object from class n99999999",
                            "caption_source": "wnid_fallback",
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cache.write_text(
                json.dumps(
                    {
                        "image_id": "n03452741_1",
                        "caption": "a person playing a grand piano",
                        "caption_source": "blip",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            stats = build_blip_manifest(source, target, cache)
            converted = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(converted[0]["caption"], "a person playing a grand piano")
            self.assertEqual(converted[0]["caption_source"], "blip")
            self.assertEqual(converted[1]["caption_source"], "wnid_fallback")
            self.assertEqual(stats, {"rows": 2, "blip_used": 1, "missing": 1})

            write_report(report, {"requested_unique_images": 1, "cached_before": 0, "generated": 1, "cached_after": 1, "failed": []}, {"test": stats}, cache)
            self.assertIn("BLIP captions used", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
