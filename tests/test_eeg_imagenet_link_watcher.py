from __future__ import annotations

import unittest
from pathlib import Path
import tempfile


class EEGImageNetLinkWatcherTests(unittest.TestCase):
    def test_build_link_commands_cover_train_val_test(self) -> None:
        from scripts.watch_and_link_eeg_imagenet_images import build_link_commands

        commands = build_link_commands(
            image_root=Path("/data/imagenet/extracted"),
            relative_to=Path("/workspace"),
            report_dir=Path("outputs/datasets"),
        )

        joined = "\n".join(" ".join(command) for command in commands)
        self.assertIn("--manifest data/EEG-ImageNet/train.jsonl", joined)
        self.assertIn("--manifest data/EEG-ImageNet/val.jsonl", joined)
        self.assertIn("--manifest data/EEG-ImageNet/test.jsonl", joined)
        self.assertIn("--out data/EEG-ImageNet/train_image_linked.jsonl", joined)
        self.assertIn("--out data/EEG-ImageNet/val_image_linked.jsonl", joined)
        self.assertIn("--out data/EEG-ImageNet/test_image_linked.jsonl", joined)
        self.assertIn("--filtered-out data/EEG-ImageNet/train_image_exact.jsonl", joined)
        self.assertIn("--filtered-out data/EEG-ImageNet/val_image_exact.jsonl", joined)
        self.assertIn("--filtered-out data/EEG-ImageNet/test_image_exact.jsonl", joined)
        self.assertIn("--image-root /data/imagenet/extracted", joined)
        self.assertIn("--relative-to /workspace", joined)

    def test_image_root_ready_requires_extracted_files(self) -> None:
        from scripts.watch_and_link_eeg_imagenet_images import image_root_ready

        self.assertFalse(image_root_ready(Path("/definitely/not/present")))

    def test_image_root_ready_rejects_partial_nested_extraction(self) -> None:
        from scripts.watch_and_link_eeg_imagenet_images import image_root_ready

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = root / "ILSVRC" / "Data" / "CLS-LOC" / "train" / "n00000001" / "n00000001_0001.JPEG"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"jpeg")

            self.assertFalse(image_root_ready(root))

    def test_image_root_ready_requires_complete_markers_and_images(self) -> None:
        from scripts.watch_and_link_eeg_imagenet_images import image_root_ready

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".extract_complete.json").write_text('{"status":"complete"}\n', encoding="utf-8")
            (root / ".nested_extract_complete.json").write_text('{"status":"complete"}\n', encoding="utf-8")

            self.assertFalse(image_root_ready(root))

            train_image = root / "ILSVRC" / "Data" / "CLS-LOC" / "train" / "n00000001" / "n00000001_0001.JPEG"
            val_image = root / "ILSVRC" / "Data" / "CLS-LOC" / "val" / "ILSVRC2012_val_00000001.JPEG"
            train_image.parent.mkdir(parents=True)
            val_image.parent.mkdir(parents=True)
            train_image.write_bytes(b"jpeg")
            val_image.write_bytes(b"jpeg")

            self.assertTrue(image_root_ready(root))

    def test_image_root_ready_rejects_complete_markers_without_kaggle_layout(self) -> None:
        from scripts.watch_and_link_eeg_imagenet_images import image_root_ready

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".extract_complete.json").write_text('{"status":"complete"}\n', encoding="utf-8")
            (root / ".nested_extract_complete.json").write_text('{"status":"complete"}\n', encoding="utf-8")
            random_image = root / "some_partial_dir" / "sample.JPEG"
            random_image.parent.mkdir(parents=True)
            random_image.write_bytes(b"jpeg")

            self.assertFalse(image_root_ready(root))


if __name__ == "__main__":
    unittest.main()
