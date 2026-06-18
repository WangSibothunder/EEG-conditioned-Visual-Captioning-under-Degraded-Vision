from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> str:
    path = Path(path)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_json(path: str | Path) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _artifact_line(label: str, path: str | Path) -> str:
    path = Path(path)
    status = "available" if path.exists() else "pending"
    return f"- {label}: `{path}` ({status})"


def _pending_artifacts(artifacts: list[tuple[str, str | Path]]) -> list[str]:
    return [f"- {label}: `{Path(path)}`" for label, path in artifacts if not Path(path).exists()]


def _preferred_sanity_score(row: dict[str, str]) -> float:
    for key in ["class_hit", "rouge_l"]:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                continue
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the final Day5 research report.")
    parser.add_argument("--out", default="outputs/day5_final/NEXT_48H_RESEARCH_REPORT.md")
    args = parser.parse_args()

    paths = {
        "alignment": Path("outputs/day4_alignment/ALIGNMENT_ABLATION_REPORT.md"),
        "multiseed": Path("outputs/day4_alignment/multiseed_summary.md"),
        "caption_targets": Path("outputs/day4_caption_targets/caption_target_report.md"),
        "alignment_extensions": Path("outputs/day5_clipL/clipL_alignment_report.md"),
        "fusion": Path("outputs/day5_fusion/FUSION_COMPARISON_REPORT.md"),
        "sanity_csv": Path("outputs/day5_sanity/FULL_SANITY_METRICS.csv"),
        "sanity_md": Path("outputs/day5_sanity/FULL_SANITY_METRICS.md"),
        "day3": Path("outputs/day3/DAY3_REPORT.md"),
        "gate": Path("outputs/day5_sanity/gate_analysis.md"),
        "qualitative": Path("outputs/day5_sanity/qualitative_examples.md"),
        "best_ckpt": Path("outputs/day4_alignment/best_overall.pt"),
        "sanity_caption_ckpt": Path("outputs/day5_sanity/F1_real_eeg_epoch1_snapshot.pt"),
    }
    artifacts = [
        ("Alignment ablation report", paths["alignment"]),
        ("Alignment multi-seed summary", paths["multiseed"]),
        ("Alignment best checkpoint", paths["best_ckpt"]),
        ("Caption target report", paths["caption_targets"]),
        ("CLIP-L/Strong/Subject alignment comparison", paths["alignment_extensions"]),
        ("Fusion comparison report", paths["fusion"]),
        ("Full sanity metrics CSV", paths["sanity_csv"]),
        ("Full sanity metrics markdown", paths["sanity_md"]),
        ("Gate analysis", paths["gate"]),
        ("Qualitative examples", paths["qualitative"]),
        ("Day3 reference report", paths["day3"]),
        ("Full sanity caption checkpoint", paths["sanity_caption_ckpt"]),
    ]

    alignment = _read(paths["alignment"])
    multiseed = _read(paths["multiseed"])
    caption_targets = _read(paths["caption_targets"])
    alignment_extensions = _read(paths["alignment_extensions"])
    fusion = _read(paths["fusion"])
    sanity_rows = _read_csv(paths["sanity_csv"])
    day3 = _read(paths["day3"])
    gate = _read(paths["gate"])
    qualitative = _read(paths["qualitative"])
    pending = _pending_artifacts(artifacts)

    real_beats_controls = "not verified"
    eeg_claim_conclusion = "Full sanity metrics are still pending, so EEG benefit cannot be claimed yet."
    if sanity_rows:
        grouped: dict[str, dict[str, float]] = {}
        for row in sanity_rows:
            grouped.setdefault(row.get("corruption", ""), {})[row.get("mode", "")] = _preferred_sanity_score(row)
        wins = 0
        total = 0
        for modes in grouped.values():
            if all(mode in modes for mode in ["real_eeg", "shuffled_eeg", "random_eeg"]):
                total += 1
                if modes["real_eeg"] > modes["shuffled_eeg"] and modes["real_eeg"] > modes["random_eeg"]:
                    wins += 1
        real_beats_controls = f"{wins}/{total} corruptions" if total else "not available"
        if total and wins == total:
            eeg_claim_conclusion = "Paired EEG beats both shuffled and random controls across all evaluated corruptions by the selected metric; this supports a cautious EEG-benefit claim."
        elif total and wins > 0:
            eeg_claim_conclusion = "Paired EEG beats controls only on some corruptions; evidence is mixed and should be reported as preliminary."
        elif total:
            eeg_claim_conclusion = "Paired EEG does not beat both shuffled and random controls on the evaluated corruptions; do not claim an EEG benefit."
        else:
            eeg_claim_conclusion = "Sanity rows exist, but required real/shuffled/random control groups are incomplete; do not claim an EEG benefit."

    lines = [
        "# Next 48H Research Report",
        "",
        "## What Was Wrong With Day3",
        "",
        "- Caption generation was weak and code-like.",
        "- EEG alignment beat random but remained weak.",
        "- Real EEG did not consistently beat shuffled/random controls.",
        "- Gate values were small and close across modes.",
        "",
        "## What Was Fixed",
        "",
        "- Added human-readable ImageNet class caption targets.",
        "- Added class-hit sanity metric and preserved human class labels in sanity JSONL outputs.",
        "- Added Day4 alignment ablation/multiseed reporting infrastructure.",
        "- Added Day5 dataset status inspections for THINGS-EEG2 and EIT-1M.",
        "",
        "## Key Artifact Paths",
        "",
        *[_artifact_line(label, path) for label, path in artifacts],
        "",
        "## Pending Artifacts",
        "",
        *(pending or ["- None."]),
        "",
        "## Caption Target Improvement",
        "",
        caption_targets or "Caption target report missing.",
        "",
        "## Alignment Ablation Results",
        "",
        alignment or "Alignment ablation is still running or missing.",
        "",
        "## Multi-seed Results",
        "",
        multiseed or "Multi-seed summary is still pending.",
        "",
        f"- Best checkpoint path: `{paths['best_ckpt']}` exists={paths['best_ckpt'].exists()}",
        "",
        "## CLIP-L / Strong / Subject Alignment",
        "",
        alignment_extensions or "CLIP-L/Strong/Subject comparison is still pending.",
        "",
        "## Fusion Comparison",
        "",
        fusion or "Fusion comparison is still pending.",
        "",
        "## Full Degraded Sanity Results",
        "",
        f"- Caption checkpoint used for sanity generation: `{paths['sanity_caption_ckpt']}` exists={paths['sanity_caption_ckpt'].exists()}",
        f"- FULL_SANITY_METRICS.csv rows: `{len(sanity_rows)}`",
        f"- Full metrics table: `{paths['sanity_md']}` exists={paths['sanity_md'].exists()}",
        "",
        "## Gate Analysis",
        "",
        gate or "Gate analysis is still pending.",
        "",
        "## Qualitative Examples",
        "",
        qualitative or "Qualitative examples are still pending.",
        "",
        "## Real EEG vs Controls",
        "",
        f"- Real EEG beats shuffled and random: `{real_beats_controls}`",
        "- Do not claim EEG benefit unless this is consistent across controls and conditions.",
        "",
        "## Can We Claim EEG Benefit?",
        "",
        eeg_claim_conclusion,
        "",
        "## Recommended Next Step",
        "",
        "- Stronger data.",
        "- Better captions.",
        "- Larger CLIP.",
        "- THINGS-EEG2 pretraining.",
        "- EIT-1M tri-modal training.",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
