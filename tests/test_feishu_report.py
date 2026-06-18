from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.utils.feishu_report import (
    build_report_payload,
    detect_exceptions,
    read_goal,
    read_terminal_tail,
    summarize_training,
)


class FeishuReportTests(unittest.TestCase):
    def test_payload_is_chinese_feishu_card(self) -> None:
        payload = build_report_payload(
            goal="完成 Day1-Day2 MVP",
            progress="Done: dummy data works",
            training="fusion: step=2 loss=1.2345",
            exceptions="未发现异常关键词",
            terminal_tail=["line1", "line2"],
            generated_at="2026-06-15 00:00 UTC",
        )

        self.assertEqual(payload["msg_type"], "interactive")
        card_text = json.dumps(payload["card"], ensure_ascii=False)
        self.assertIn("项目进度汇报", card_text)
        self.assertIn("目标", card_text)
        self.assertIn("训练效果", card_text)
        self.assertIn("异常", card_text)
        self.assertIn("最近 5 行终端", card_text)

    def test_summarize_training_reads_latest_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log = root / "outputs" / "debug" / "fusion" / "train_log.jsonl"
            log.parent.mkdir(parents=True)
            log.write_text(
                "\n".join(
                    [
                        json.dumps({"step": 1, "loss": 2.0}),
                        json.dumps({"step": 2, "loss": 1.25}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = summarize_training(root)

        self.assertIn("fusion", summary)
        self.assertIn("step=2", summary)
        self.assertIn("loss=1.2500", summary)

    def test_detect_exceptions_scans_recent_terminal_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "terminal.log"
            log.write_text(
                "ok\nTraceback (most recent call last):\nRuntimeError: CUDA out of memory\n",
                encoding="utf-8",
            )

            summary = detect_exceptions(log)

        self.assertIn("CUDA out of memory", summary)
        self.assertIn("RuntimeError", summary)

    def test_terminal_tail_returns_last_five_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "terminal.log"
            log.write_text("\n".join(f"line {idx}" for idx in range(1, 8)), encoding="utf-8")

            tail = read_terminal_tail(log, lines=5)

        self.assertEqual(tail, ["line 3", "line 4", "line 5", "line 6", "line 7"])

    def test_read_goal_includes_multiline_current_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            goal_dir = Path(tmpdir) / "goal"
            goal_dir.mkdir()
            (goal_dir / "day1-2goal.md").write_text(
                "Current coding goal:\nFinish Phase 0, Phase 1, and minimal Phase 2 skeleton.\n",
                encoding="utf-8",
            )

            goal = read_goal(tmpdir)

        self.assertIn("Finish Phase 0", goal)


if __name__ == "__main__":
    unittest.main()
