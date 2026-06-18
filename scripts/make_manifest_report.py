from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Thought2Text dataset smoke test and save a manifest report.")
    parser.add_argument("--manifest", default="data/thought2text/train.jsonl")
    parser.add_argument("--root", default="data/thought2text")
    parser.add_argument("--out", default="outputs/overnight/manifest_report.md")
    parser.add_argument("--allow_missing_images", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        "-m",
        "src.data.dataset",
        "--manifest",
        args.manifest,
        "--root",
        args.root,
        "--smoke_test",
    ]
    if args.allow_missing_images:
        command.append("--allow_missing_images")
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    lines = [
        "# Thought2Text Manifest Report",
        "",
        "## Command",
        "",
        "```bash",
        " ".join(command),
        "```",
        "",
        f"- Exit code: `{result.returncode}`",
        "",
        "## stdout",
        "",
        "```text",
        result.stdout.strip(),
        "```",
        "",
        "## stderr",
        "",
        "```text",
        result.stderr.strip(),
        "```",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
