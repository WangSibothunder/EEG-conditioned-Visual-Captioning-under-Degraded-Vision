from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class LaunchAfterArtifactsTests(unittest.TestCase):
    def test_all_artifacts_ready_requires_existing_nonempty_files(self) -> None:
        from scripts.launch_after_artifacts import all_artifacts_ready

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ready = root / "ready.txt"
            missing = root / "missing.txt"

            self.assertFalse(all_artifacts_ready([ready]))
            ready.write_text("", encoding="utf-8")
            self.assertFalse(all_artifacts_ready([ready]))
            ready.write_text("ok", encoding="utf-8")
            self.assertTrue(all_artifacts_ready([ready]))
            self.assertFalse(all_artifacts_ready([ready, missing]))

    def test_shell_command_uses_bash_lc(self) -> None:
        from scripts.launch_after_artifacts import build_shell_command

        command = build_shell_command("echo hello")

        self.assertEqual(command, ["bash", "-lc", "echo hello"])


if __name__ == "__main__":
    unittest.main()
