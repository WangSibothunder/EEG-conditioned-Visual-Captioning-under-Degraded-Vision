from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _count(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _status(paths: list[Path]) -> str:
    if all(path.exists() for path in paths):
        return ", ".join(f"`{path}`" for path in paths)
    return "not generated yet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create caption target variant report.")
    parser.add_argument("--root", default="data/thought2text")
    parser.add_argument("--out", default="outputs/day4_caption_targets/caption_target_report.md")
    args = parser.parse_args()

    root = Path(args.root)
    splits = ["train", "val", "test"]
    wnid_paths = [root / f"{split}.jsonl" for split in splits]
    human_paths = [root / f"{split}_human_caption.jsonl" for split in splits]
    blip_paths = [root / f"{split}_blip_caption.jsonl" for split in splits]
    blip_available = all(path.exists() and _count(path) > 0 for path in blip_paths)

    lines = [
        "# Caption Target Report",
        "",
        "## Variants",
        "",
        "| Variant | Manifest status | Priority | Notes |",
        "| --- | --- | --- | --- |",
        f"| class_wnid_caption | {_status(wnid_paths)} | fallback | Original class-token target; caused code-like generations in Day3. |",
        f"| human_class_caption | {_status(human_paths)} | primary fallback | WNIDs mapped to human-readable ImageNet class names. |",
        f"| blip_caption | {_status(blip_paths)} | {'primary candidate' if blip_available else 'optional'} | Generated captions for unique images; inspect quality before retraining fusion. |",
        "",
        "## Counts",
        "",
        "| Split | WNID rows | Human-caption rows | BLIP-caption rows | BLIP captions used |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in splits:
        blip_rows = _read_jsonl(root / f"{split}_blip_caption.jsonl")
        blip_used = sum(1 for row in blip_rows if row.get("caption_source") == "blip")
        lines.append(
            f"| {split} | {_count(root / f'{split}.jsonl')} | {_count(root / f'{split}_human_caption.jsonl')} | "
            f"{len(blip_rows)} | {blip_used} |"
        )

    lines.extend(["", "## Examples", "", "| Image ID | WNID caption | Human class caption | BLIP caption | Human class |", "| --- | --- | --- | --- | --- |"])
    wnid_rows = _read_jsonl(root / "train.jsonl", limit=20)
    human_by_id = {str(row.get("image_id", "")): row for row in _read_jsonl(root / "train_human_caption.jsonl", limit=200)}
    blip_by_id = {str(row.get("image_id", "")): row for row in _read_jsonl(root / "train_blip_caption.jsonl", limit=200)}
    for row in wnid_rows[:20]:
        image_id = str(row.get("image_id", ""))
        human = human_by_id.get(image_id, {})
        blip = blip_by_id.get(image_id, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    image_id,
                    str(row.get("caption", "")),
                    str(human.get("caption", "")),
                    str(blip.get("caption", "")),
                    str(human.get("human_label_name", "")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Use `human_class_caption` for completed Day5 fusion runs because those runs already use that target.",
            "- Use `blip_caption` as the next fusion-training candidate now that BLIP captions are generated and complete.",
            "- Do not use the original WNID-only captions except as a fallback/control.",
        ]
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
