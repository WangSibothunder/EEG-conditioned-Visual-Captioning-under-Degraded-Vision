# Token Generative EVLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a token-level EEG-enhanced VLM prototype and a free-form generative EVLM that outputs actual caption examples.

**Architecture:** Reuse Thought2Text manifests, CLIP caches/images, A2 EEG encoder, text prototypes, and existing caption model primitives. Implement a standalone pipeline that trains VTF token-fusion semantic models, then trains/evaluates frozen-LM soft-prefix caption generators using enhanced VTF features and EEG controls.

**Tech Stack:** Python, PyTorch, HuggingFace Transformers when available, repo tiny LM fallback, CSV/Markdown/JSONL artifacts.

---

### Task 1: Token Fusion

**Files:**
- Create: `scripts/run_token_generative_evlm.py`
- Test: `tests/test_token_generative_evlm.py`

- [ ] Implement `ViTTokenFusionModel` with visual tokens, EEG tokenization, cross-attention, beta gating, pooling, and prototype logits.
- [ ] Support VTF variants `VTF1_basic_M4`, `VTF2_confidence_beta_M4`, `VTF3_confidence_beta_margin_M4`, `VTF4_confidence_beta_margin_M8`.
- [ ] Train at least one full VTF model and evaluate all six corruptions and five modes.
- [ ] Write `token_fusion/VTF_MODEL_SELECTION.md`, `token_fusion/VTF_MODEL_SELECTION.csv`, `token_fusion/metrics.csv`, and checkpoints.

### Task 2: Generative EVLM

**Files:**
- Create/modify: `scripts/run_token_generative_evlm.py`
- Test: `tests/test_token_generative_evlm.py`

- [ ] Implement free-form prefix generator variants `G0_image_only_prefix`, `G2_vtf_visual_eeg_prefix`, `G3_vtf_visual_eeg_topk_prefix`.
- [ ] Train at least one generator with teacher forcing and evaluate generated captions under all required corruptions/modes.
- [ ] Save actual captions in JSONL and `generation/QUALITATIVE_EXAMPLES.md` with at least 30 examples and 5 best examples if possible.
- [ ] Write invalid output report, metrics, model selection, and target report.

### Task 3: Final Reports and Verification

**Files:**
- Create/modify: `scripts/run_token_generative_evlm.py`

- [ ] Write `TOKEN_GEN_EVLM_FINAL_REPORT.md`, model-selection CSV/MD, and GPU usage log.
- [ ] Verify generated captions are non-template free-form text where possible and report honestly if quality is poor.
- [ ] Run unit tests and an artifact audit before completion.
