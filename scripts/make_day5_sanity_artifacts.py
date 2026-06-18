from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _by_image(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("image_id")): row for row in records}


def write_qualitative(root: str | Path, out: str | Path) -> None:
    root = Path(root)
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for path in sorted(root.glob("*.jsonl")):
        rows = _read_jsonl(path)
        if not rows:
            continue
        corruption = str(rows[0].get("corruption", path.stem.split("_")[0]))
        mode = str(rows[0].get("mode", "_".join(path.stem.split("_")[1:])))
        groups[corruption][mode] = rows

    lines = [
        "# Qualitative Examples",
        "",
        "| Corruption | Image ID | Label | Reference | Vision Only | Real EEG | Shuffled EEG | Random EEG | Gate Values |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for corruption in ["clean", "blur", "occlusion", "noise", "lowres"]:
        mode_maps = {mode: _by_image(rows) for mode, rows in groups.get(corruption, {}).items()}
        real_rows = groups.get(corruption, {}).get("real_eeg", [])[:6]
        for row in real_rows:
            image_id = str(row.get("image_id"))
            values = {mode: mode_maps.get(mode, {}).get(image_id, {}) for mode in ["vision_only", "real_eeg", "shuffled_eeg", "random_eeg"]}
            gates = ", ".join(
                f"{mode}={values[mode].get('gate_mean', 'NA')}"
                for mode in ["real_eeg", "shuffled_eeg", "random_eeg"]
            )
            lines.append(
                f"| {corruption} | {image_id} | {row.get('human_label_name', row.get('label', ''))} | "
                f"{row.get('reference', '')} | {values['vision_only'].get('prediction', '')} | "
                f"{values['real_eeg'].get('prediction', '')} | {values['shuffled_eeg'].get('prediction', '')} | "
                f"{values['random_eeg'].get('prediction', '')} | {gates} |"
            )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create qualitative examples from full sanity JSONL files.")
    parser.add_argument("--root", default="outputs/day5_sanity")
    parser.add_argument("--out", default="outputs/day5_sanity/qualitative_examples.md")
    args = parser.parse_args()
    write_qualitative(args.root, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
