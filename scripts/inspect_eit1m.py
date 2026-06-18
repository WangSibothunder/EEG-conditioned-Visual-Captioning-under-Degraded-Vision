from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


def _zip_members(path: Path, limit: int = 200) -> tuple[list[str], int]:
    if not path.exists() or path.suffix.lower() != ".zip":
        return [], 0
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    return names[:limit], len(names)


def inspect(root: str | Path, out: str | Path) -> None:
    root = Path(root)
    files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
    zip_files = [path for path in files if path.suffix.lower() == ".zip"]
    member_names: list[str] = []
    total_members = 0
    for zip_path in zip_files[:2]:
        members, count = _zip_members(zip_path)
        total_members += count
        member_names.extend([f"{zip_path.name}:{name}" for name in members])
    lower_names = [name.lower() for name in member_names] + [str(path).lower() for path in files]
    has_eeg = any("eeg" in name or name.endswith((".mat", ".npy", ".pth")) for name in lower_names)
    has_image = any(name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")) for name in lower_names)
    has_text = any("text" in name or "caption" in name or name.endswith((".txt", ".json", ".jsonl", ".csv")) for name in lower_names)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# EIT-1M Status",
        "",
        f"- Data root: `{root}`",
        f"- Exists: `{root.exists()}`",
        f"- File count: `{len(files)}`",
        f"- Total file size bytes: `{sum(path.stat().st_size for path in files) if files else 0}`",
        f"- Zip files: `{len(zip_files)}`",
        f"- Zip member count inspected: `{total_members}`",
        f"- EEG exists: `{has_eeg}`",
        f"- Image exists: `{has_image}`",
        f"- Text/caption exists: `{has_text}`",
        f"- Sample count: `unknown without extracting/indexing zip members`",
        "",
        "## Adaptability",
        "",
    ]
    if has_eeg and has_image and has_text:
        lines.append("EIT-1M appears to contain EEG, image, and text-like assets. A small manifest conversion may be possible after controlled extraction.")
    elif root.exists():
        lines.append("EIT-1M is present but not yet converted. Avoid blocking Day4/Day5 core tasks on full extraction.")
    else:
        lines.append("Dataset root is missing.")
    lines.extend(["", "## Local Files", ""])
    for path in files[:30]:
        lines.append(f"- `{path}` ({path.stat().st_size} bytes)")
    if member_names:
        lines.extend(["", "## Zip Member Sample", ""])
        for name in member_names[:80]:
            lines.append(f"- `{name}`")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local EIT-1M availability without extracting large archives.")
    parser.add_argument("--root", default="data/EIT-1M")
    parser.add_argument("--out", default="outputs/day5_datasets/eit1m_status.md")
    args = parser.parse_args()
    inspect(args.root, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
