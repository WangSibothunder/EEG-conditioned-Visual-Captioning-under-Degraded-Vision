from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class HeavyStageQueueStatusTests(unittest.TestCase):
    def test_update_queue_status_rewrites_target_job_only(self) -> None:
        from scripts.update_heavy_stage_queue_status import update_queue_status

        with tempfile.TemporaryDirectory() as tmpdir:
            queue = Path(tmpdir) / "queue.yaml"
            queue.write_text(
                "\n".join(
                    [
                        "jobs:",
                        "  - id: JOB_A",
                        "    priority: 1",
                        "    status: waiting",
                        "    command: \"echo a\"",
                        "  - id: JOB_B",
                        "    priority: 2",
                        "    status: completed",
                        "    command: \"echo b\"",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            changed = update_queue_status(queue, "JOB_A", "running")

            self.assertTrue(changed)
            text = queue.read_text(encoding="utf-8")
            self.assertIn("  - id: JOB_A\n    priority: 1\n    status: running", text)
            self.assertIn("  - id: JOB_B\n    priority: 2\n    status: completed", text)

    def test_update_queue_status_returns_false_for_missing_job(self) -> None:
        from scripts.update_heavy_stage_queue_status import update_queue_status

        with tempfile.TemporaryDirectory() as tmpdir:
            queue = Path(tmpdir) / "queue.yaml"
            original = "jobs:\n  - id: JOB_A\n    status: waiting\n"
            queue.write_text(original, encoding="utf-8")

            changed = update_queue_status(queue, "MISSING", "running")

            self.assertFalse(changed)
            self.assertEqual(queue.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
