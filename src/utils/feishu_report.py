from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request


DEFAULT_TERMINAL_LOG = Path("outputs/monitor/terminal.log")
DEFAULT_ENV_PATH = Path(".env.feishu")
EXCEPTION_KEYWORDS = (
    "Traceback",
    "RuntimeError",
    "Exception",
    "Error:",
    "CUDA out of memory",
    "OutOfMemoryError",
    "Killed",
)


def read_terminal_tail(log_path: str | Path, lines: int = 5) -> list[str]:
    path = Path(log_path)
    if not path.exists():
        return [f"终端日志不存在: {path}"]
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = content[-lines:]
    return tail if tail else ["终端日志为空"]


def summarize_training(project_root: str | Path = ".") -> str:
    root = Path(project_root)
    summaries: list[str] = []
    for name in ("baseline", "fusion"):
        log_path = root / "outputs" / "debug" / name / "train_log.jsonl"
        if not log_path.exists():
            summaries.append(f"{name}: 暂无训练日志")
            continue
        record = _latest_json_record(log_path)
        if record is None:
            summaries.append(f"{name}: 训练日志为空或无法解析")
            continue
        step = record.get("step", "未知")
        loss = _format_loss(record.get("loss"))
        summaries.append(f"{name}: step={step}, loss={loss}")

    sample_paths = [
        root / "outputs" / "debug" / "baseline" / "samples.jsonl",
        root / "outputs" / "debug" / "fusion" / "samples.jsonl",
        root / "outputs" / "debug" / "generation" / "real_eeg.jsonl",
    ]
    generated = [path for path in sample_paths if path.exists()]
    if generated:
        summaries.append("样例输出: " + ", ".join(str(path) for path in generated))
    return "\n".join(summaries)


def detect_exceptions(log_path: str | Path) -> str:
    path = Path(log_path)
    if not path.exists():
        return f"未发现异常关键词；终端日志不存在: {path}"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    matches = [
        line.strip()
        for line in lines[-200:]
        if any(keyword in line for keyword in EXCEPTION_KEYWORDS)
    ]
    if not matches:
        return "未发现异常关键词"
    return "\n".join(matches[-5:])


def read_goal(project_root: str | Path = ".") -> str:
    root = Path(project_root)
    goal_file = root / "goal" / "day1-2goal.md"
    if not goal_file.exists():
        return "当前目标: 完成 EEG + Vision -> Caption 的 Day1-Day2 MVP"
    text = goal_file.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Current coding goal:"):
            inline = stripped.removeprefix("Current coding goal:").strip()
            if inline:
                return f"当前目标: {inline}"
            for follow in lines[idx + 1 :]:
                follow = follow.strip().strip("`")
                if follow:
                    return f"当前目标: {follow}"
            return "当前目标: Finish Phase 0, Phase 1, and minimal Phase 2 skeleton"
    return "当前目标: Finish Phase 0, Phase 1, and minimal Phase 2 skeleton"


def read_progress(project_root: str | Path = ".") -> str:
    status_file = Path(project_root) / "docs" / "STATUS.md"
    if not status_file.exists():
        return "docs/STATUS.md 不存在，无法读取项目进度"
    text = status_file.read_text(encoding="utf-8", errors="replace")
    sections = _extract_status_sections(text, ["## Done", "## In progress", "## Blocked", "## Next action"])
    if not sections:
        return text.strip()[:1500] or "docs/STATUS.md 为空"
    return "\n\n".join(sections)


def build_report_payload(
    *,
    goal: str,
    progress: str,
    training: str,
    exceptions: str,
    terminal_tail: list[str],
    generated_at: str,
) -> dict[str, Any]:
    host = socket.gethostname()
    tail_text = "\n".join(terminal_tail)
    return {
        "msg_type": "interactive",
        "card": {
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {
                "template": "blue",
                "title": {"tag": "plain_text", "content": "EEG + Vision 项目进度汇报"},
            },
            "body": {
                "elements": [
                    _markdown(f"**时间:** {generated_at}\n**主机:** {host}"),
                    _hr(),
                    _section("目标", goal),
                    _section("目前项目进度", progress),
                    _section("训练效果", training),
                    _section("是否出现异常", exceptions),
                    _section("最近 5 行终端", f"```text\n{tail_text}\n```"),
                ]
            },
        },
    }


def send_feishu_payload(webhook: str, payload: dict[str, Any], timeout: int = 15) -> str:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        webhook,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
        if response.status >= 400:
            raise RuntimeError(f"Feishu webhook returned HTTP {response.status}: {body}")
        return body


def load_webhook(env_path: str | Path = DEFAULT_ENV_PATH) -> str | None:
    webhook = os.environ.get("FEISHU_WEBHOOK")
    if webhook:
        return webhook
    path = Path(env_path)
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        if key.strip() == "FEISHU_WEBHOOK":
            return value.strip().strip("\"'")
    return None


def build_report_from_workspace(
    *,
    project_root: str | Path = ".",
    terminal_log: str | Path = DEFAULT_TERMINAL_LOG,
    generated_at: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    timestamp = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return build_report_payload(
        goal=read_goal(root),
        progress=read_progress(root),
        training=summarize_training(root),
        exceptions=detect_exceptions(root / terminal_log),
        terminal_tail=read_terminal_tail(root / terminal_log, lines=5),
        generated_at=timestamp,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="发送 EEG + Vision 项目飞书进度卡片。")
    parser.add_argument("--project-root", default=".", help="项目根目录。")
    parser.add_argument("--terminal-log", default=str(DEFAULT_TERMINAL_LOG), help="终端日志路径。")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="本地 webhook 环境文件。")
    parser.add_argument("--dry-run", action="store_true", help="只打印 JSON，不发送。")
    args = parser.parse_args()

    payload = build_report_from_workspace(
        project_root=args.project_root,
        terminal_log=args.terminal_log,
    )
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    webhook = load_webhook(args.env_file)
    if not webhook:
        raise SystemExit("缺少 FEISHU_WEBHOOK。请设置环境变量或创建 .env.feishu。")
    response = send_feishu_payload(webhook, payload)
    print(response)


def _latest_json_record(path: Path) -> dict[str, Any] | None:
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            return record
    return None


def _format_loss(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "未知"


def _extract_status_sections(text: str, headers: list[str]) -> list[str]:
    lines = text.splitlines()
    sections: list[str] = []
    for header in headers:
        start = next((idx for idx, line in enumerate(lines) if line.strip().lower() == header.lower()), None)
        if start is None:
            continue
        end = len(lines)
        for idx in range(start + 1, len(lines)):
            if lines[idx].startswith("## "):
                end = idx
                break
        body = "\n".join(lines[start:end]).strip()
        if body:
            sections.append(body)
    return sections


def _markdown(content: str) -> dict[str, Any]:
    return {"tag": "markdown", "content": content}


def _section(title: str, content: str) -> dict[str, Any]:
    return _markdown(f"**{title}**\n{content}")


def _hr() -> dict[str, Any]:
    return {"tag": "hr"}


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
