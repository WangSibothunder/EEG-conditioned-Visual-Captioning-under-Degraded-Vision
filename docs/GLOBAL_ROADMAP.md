# Global Roadmap

Last updated: 2026-06-15

Day1-Day2 are only the first part of the global plan. They create the runnable foundation for later real-data and robustness experiments.

## Phase 0: Project Skeleton And Context

Goal: make the project understandable and runnable.

- Create repo structure for configs, data, models, training, evaluation, scripts, outputs, and docs.
- Define module contracts before code grows.
- Add dummy data generation and debug config.
- Keep status, decisions, and commands current.

## Phase 1: Image-Only Caption Baseline

Goal: establish the baseline that every EEG result must beat or match.

Pipeline:

```text
image -> frozen CLIP -> trainable soft prompt projector -> frozen LLM -> caption
```

Expected outputs:

- Finite training loss.
- Saved baseline checkpoint.
- Sample validation captions.

## Phase 2: EEG Encoder And Gated Fusion

Goal: add the smallest useful EEG path.

Pipeline:

```text
image -> frozen CLIP -> image_emb
EEG -> lightweight EEG encoder -> eeg_emb
image_emb + eeg_emb -> gated fusion -> fused_emb
fused_emb -> soft prompt projector -> frozen LLM -> caption
```

Expected outputs:

- EEG encoder and gated fusion forward pass.
- Fusion training checkpoint.
- Generation with real, shuffled, random, and ignored EEG modes.

## Phase 3: Robustness Experiment

Goal: test whether EEG helps when vision is degraded.

Image conditions:

- Clean.
- Blur.
- Occlusion.
- Noise.
- Low resolution.

Model/input comparisons:

- Vision-only.
- EEG-only if implemented.
- Vision + real EEG.
- Vision + shuffled EEG.
- Vision + random EEG.

Do not claim EEG helps until shuffled and random EEG controls are implemented and reported.

## Phase 4: Real Dataset Migration

Goal: replace dummy data with real synchronized EEG/image/caption data.

Candidate sources:

- Thought2Text or CVPR2017 EEG-style data if available.
- THINGS-EEG2 if available.
- EEG-ImageNet if available.

Migration steps:

- Convert real samples into the manifest contract.
- Cache or precompute expensive image embeddings if needed.
- Train or pretrain EEG alignment to CLIP-like image embeddings.
- Re-run Phase 2 and Phase 3 checks.

## Current Two-Day Milestone

Day 1:

- Finish Phase 0.
- Build dummy data, dataset loader, and collate path.
- Train image-only baseline for a few debug steps.
- Generate sample captions.

Day 2:

- Add lightweight EEG encoder.
- Add gated fusion.
- Train fusion path for a few debug steps.
- Run generation and sanity-check modes.

The two-day milestone is complete only when commands in `docs/RUNBOOK.md` execute successfully or every failure is assigned to the owning agent with a concrete blocker.
