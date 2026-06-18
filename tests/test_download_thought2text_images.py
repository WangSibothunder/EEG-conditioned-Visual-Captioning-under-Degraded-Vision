from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.download_thought2text_images import (
    build_drive_image_index,
    plan_image_downloads,
)


class DownloadThought2TextImagesTests(unittest.TestCase):
    def test_plans_only_original_drive_images_for_manifest_rows(self) -> None:
        listing = [
            {
                "url": "https://drive.google.com/uc?id=original",
                "path": "images/n02951358/n02951358_31190.JPEG",
            },
            {
                "url": "https://drive.google.com/uc?id=sketch",
                "path": "images/n02951358/n02951358_31190_sketch.JPEG",
            },
            {
                "url": "https://drive.google.com/uc?id=spectro",
                "path": "images/n02951358/n02951358_31190_spectro_123.JPEG",
            },
            {
                "url": "https://drive.google.com/uc?id=caption",
                "path": "images/n02951358/n02951358_31190_caption.txt",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "data" / "thought2text"
            root.mkdir(parents=True)
            manifest = root / "train.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "image_id": "n02951358_31190",
                        "image_path": "images/n02951358_31190.jpg",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            index = build_drive_image_index(listing)
            planned, missing = plan_image_downloads(
                manifests=[manifest],
                drive_index=index,
                thought2text_root=root,
            )

        self.assertEqual(missing, [])
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0].url, "https://drive.google.com/uc?id=original")
        self.assertEqual(planned[0].target_path, root / "images" / "n02951358_31190.jpg")


if __name__ == "__main__":
    unittest.main()
