from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.link_thought2text_images import link_thought2text_images


class LinkThought2TextImagesTests(unittest.TestCase):
    def test_links_images_from_flat_and_wnid_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "data" / "thought2text" / "train.jsonl"
            manifest.parent.mkdir(parents=True)
            rows = [
                {
                    "image_id": "n02951358_31190",
                    "image_path": "images/n02951358_31190.jpg",
                    "label_name": "n02951358",
                },
                {
                    "image_id": "n03272562_1177",
                    "image_path": "images/n03272562_1177.jpg",
                    "label_name": "n03272562",
                },
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            source = root / "imagenet"
            (source / "n02951358").mkdir(parents=True)
            (source / "n02951358" / "n02951358_31190.JPEG").write_bytes(b"image-a")
            source.mkdir(exist_ok=True)
            (source / "n03272562_1177.jpg").write_bytes(b"image-b")

            report = root / "outputs" / "link_report.md"
            stats = link_thought2text_images(
                manifests=[manifest],
                source_roots=[source],
                thought2text_root=manifest.parent,
                mode="copy",
                report_path=report,
            )

            self.assertEqual(stats["required"], 2)
            self.assertEqual(stats["linked"], 2)
            self.assertEqual(stats["missing"], 0)
            self.assertEqual((manifest.parent / "images" / "n02951358_31190.jpg").read_bytes(), b"image-a")
            self.assertEqual((manifest.parent / "images" / "n03272562_1177.jpg").read_bytes(), b"image-b")
            self.assertIn("Linked/copied images: `2`", report.read_text(encoding="utf-8"))

    def test_reports_missing_images_without_creating_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "data" / "thought2text" / "train.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "image_id": "n02951358_31190",
                        "image_path": "images/n02951358_31190.jpg",
                        "label_name": "n02951358",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = root / "outputs" / "link_report.md"
            stats = link_thought2text_images(
                manifests=[manifest],
                source_roots=[root / "missing_source"],
                thought2text_root=manifest.parent,
                mode="copy",
                report_path=report,
            )

            self.assertEqual(stats["required"], 1)
            self.assertEqual(stats["linked"], 0)
            self.assertEqual(stats["missing"], 1)
            self.assertFalse((manifest.parent / "images" / "n02951358_31190.jpg").exists())
            self.assertIn("n02951358_31190", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
