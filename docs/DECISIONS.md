# Decisions

## 2026-06-15 — Feishu Hourly Progress Cards

Decision:
Use a local webhook configuration file or `FEISHU_WEBHOOK` environment variable, and send Chinese Feishu interactive cards from `src.utils.feishu_report`.

Reason:
The webhook must not be committed, and hourly project status should be generated from existing project artifacts: `docs/STATUS.md`, `goal/day1-2goal.md`, train JSONL logs, and `outputs/monitor/terminal.log`.

Alternatives considered:
Hard-code the webhook in scripts, or send plain text messages only.

Consequences:
The reporter is easy to run locally, avoids committed secrets, and keeps the "最近 5 行终端" field reliable when project commands append to `outputs/monitor/terminal.log`.

Last updated: 2026-06-15

Day1-Day2 are only the first part of the global plan. These decisions favor a runnable prototype over final accuracy.

## Current Decisions

1. Use an image-only caption baseline before any EEG claim.
   - Reason: EEG value is not interpretable without a stable vision-only comparison.

2. Use dummy data before real EEG datasets.
   - Reason: the first risk is system wiring, tensor shapes, and command reliability.

3. Use `openai/clip-vit-base-patch32` first, not CLIP ViT-L/14.
   - Reason: ViT-B/32 is faster and usually produces 512-dimensional embeddings suitable for debugging.

4. Freeze CLIP by default.
   - Reason: the debug milestone should train only small modules and avoid vision encoder instability.

5. Use `Qwen/Qwen2.5-1.5B-Instruct` as the first real LLM target, not a 7B model.
   - Reason: 1.5B is more practical for single-GPU iteration. The current debug config also allows a tiny debug model path.

6. Freeze the LLM by default.
   - Reason: trainable soft prompts and small adapters are enough to verify the pipeline.

7. Train the soft prompt projector first.
   - Reason: it is the smallest bridge from CLIP embeddings into the caption decoder.

8. Use gated fusion first, not cross-attention.
   - Reason: gated fusion is easy to inspect, shape-stable, and sufficient for sanity checks.

9. Include real, shuffled, random, and ignored EEG generation modes.
   - Reason: random and shuffled controls are mandatory before claiming EEG helps.

10. Keep Day1-Day2 scope to Phase 0, Phase 1, and a minimal Phase 2 skeleton.
    - Reason: robustness experiments and real dataset migration depend on a working dummy-data pipeline.

## Deferred Decisions

- Whether to add LoRA after the frozen-LLM baseline works.
- Whether to cache CLIP embeddings after the training loop is stable.
- Which real EEG dataset is easiest to convert into the manifest contract.
- Whether degraded-image experiments should be generated on the fly or precomputed.
- Whether an EEG-to-CLIP pretraining step is needed before fusion captioning.
