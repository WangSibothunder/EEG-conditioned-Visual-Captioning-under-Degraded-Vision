from __future__ import annotations

import argparse
from pathlib import Path


def inspect(root: str | Path, out: str | Path) -> None:
    root = Path(root)
    files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
    eeg_files = [path for path in files if any(token in path.name.lower() for token in ["eeg", "preprocessed"])]
    image_files = [
        path
        for path in files
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    subjects = sorted(
        {
            part
            for path in files
            for part in path.parts
            if part.lower().startswith(("sub-", "subject", "subj"))
        }
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# THINGS-EEG2 Status",
        "",
        f"- Data root: `{root}`",
        f"- Exists: `{root.exists()}`",
        f"- File count: `{len(files)}`",
        f"- Total file size bytes: `{sum(path.stat().st_size for path in files) if files else 0}`",
        f"- Available subjects: `{', '.join(subjects) if subjects else 'none detected'}`",
        f"- Preprocessed/EEG-like files: `{len(eeg_files)}`",
        f"- Image files: `{len(image_files)}`",
        "",
        "## Adaptability",
        "",
    ]
    if eeg_files and image_files:
        lines.append("THINGS-EEG2 appears partially present and may be adaptable to the current manifest schema after inspecting metadata.")
    elif root.exists():
        lines.append("Directory exists, but no usable EEG/image files were detected within the current tree scan. Treat as not ready for smoke alignment.")
    else:
        lines.append("Dataset root is missing. THINGS-EEG2 smoke cannot run yet.")
    lines.extend(["", "## Example Files", ""])
    for path in files[:30]:
        lines.append(f"- `{path}` ({path.stat().st_size} bytes)")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local THINGS-EEG2 availability.")
    parser.add_argument("--root", default="data/THINGS-EEG2")
    parser.add_argument("--out", default="outputs/day5_datasets/things_eeg2_status.md")
    args = parser.parse_args()
    inspect(args.root, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
