# Project Brief

Last updated: 2026-06-15

## Goal

Build a good-enough research prototype for EEG-assisted image captioning:

```text
image + optional EEG -> text caption
```

The research direction is EEG-assisted robust image captioning under visual degradation. The claim to test later is whether synchronized EEG can help caption degraded or ambiguous visual input compared with vision-only, shuffled EEG, and random EEG controls.

Day1-Day2 are only the first part of the global plan. They establish the repo context, dummy data, image-only baseline, minimal EEG fusion skeleton, and sanity-check entrypoints. They are not the full research result.

## Scope

This is not a pure EEG-to-text system. The visual stream remains the primary captioning signal, and EEG is an auxiliary perceptual signal used for fusion and ablation studies.

This is also not a large multimodal foundation model. The project should run on a single GPU, prioritize debuggability, and train only small modules first.

## Inputs and Outputs

Inputs:

- Image: RGB image resized to 224 x 224.
- EEG: optional time series, expected as 64 channels x 250 time steps for the debug pipeline.
- Caption target: text string for supervised training.

Output:

- Generated caption text.

## Main Technical Route

Phase 1 baseline:

```text
image -> frozen CLIP vision encoder -> trainable soft prompt projector -> frozen LLM -> caption
```

Phase 2 fusion:

```text
image -> frozen CLIP vision encoder -> image embedding
EEG -> lightweight EEG encoder -> EEG embedding
image embedding + EEG embedding -> gated fusion -> soft prompt projector -> frozen LLM -> caption
```

The first implementation should use frozen CLIP, a frozen small LLM, trainable projector/fusion/EEG modules, and dummy data before migrating to real EEG datasets.

## Success Criteria For The Current Slice

- Dummy image, EEG, and caption samples can be generated and loaded.
- Image-only baseline trains for a few debug steps and writes a checkpoint.
- EEG+vision fusion trains for a few debug steps and writes a checkpoint.
- Generation runs for image-only, real EEG, shuffled EEG, and random EEG modes.
- Documentation clearly records interfaces, decisions, status, and commands.
