# EVLM Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate EEG-guided visual enhancement variants against A2_final under the EVLM goal.

**Architecture:** Add a standalone cached-embedding EVLM trainer/evaluator that reuses Thought2Text manifests, CLIP degraded caches, class prototypes, and the existing A2/P2 EEG encoder loader. It writes per-run metrics/checkpoints and aggregate model-selection reports under `outputs/evlm_improve/`.

**Tech Stack:** Python 3.10, PyTorch, NumPy, CSV/Markdown reports.

---

### Task 1: Core EVLM Script

**Files:**
- Create: `scripts/run_evlm_improve.py`
- Test: `tests/test_evlm_improve.py`

- [ ] Implement cached EEG/image dataset loaders using `data/thought2text/*_human_caption.jsonl`, `data/thought2text/cache/clip_*.npy`, and degraded test CLIP caches.
- [ ] Implement EVLM variants: `A2_residual_scalar`, `A2_residual_vector`, `A2_residual_vector_margin`, `A2_proto_bias`, `A2_proto_bias_margin`, `A2_residual_plus_proto_bias`.
- [ ] Implement training loss with CE, optional real-vs-shuffled/random margin, delta norm, and gamma regularization.
- [ ] Implement evaluation for clean, lowres16, mixed, occlusion50, strong_blur, strong_noise and modes vision_only, real_eeg, shuffled_eeg, random_eeg, eeg_only.
- [ ] Write per-run config, log, checkpoint, metrics CSV/MD, and summary MD.

### Task 2: Aggregation and Reports

**Files:**
- Modify: `scripts/run_evlm_improve.py`

- [ ] Aggregate all seeds into residual/proto_bias/combined metrics.
- [ ] Compare every model against `outputs/final_results/A2_FINAL_METRICS.csv`.
- [ ] Write `outputs/evlm_improve/EVLM_MODEL_SELECTION.csv` and `.md`.
- [ ] Write `outputs/evlm_improve/FINAL_EVLM_IMPROVEMENT_REPORT.md` with the six required tables.

### Task 3: Verification and Execution

**Files:**
- Test: `tests/test_evlm_improve.py`

- [ ] Run unit tests on model forward and metric aggregation.
- [ ] Run a tiny smoke only to validate code.
- [ ] Immediately launch full 80-epoch required variants across seeds after smoke.
- [ ] Maintain `outputs/evlm_improve/GPU_USAGE.md`.

