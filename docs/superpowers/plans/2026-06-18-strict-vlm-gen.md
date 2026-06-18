# Strict VLM Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete every feasible pretrained LLM/VLM route for feeding EEG-enhanced visual tokens into generative model token interfaces.

**Architecture:** Reuse the existing VTF enhanced-token feature builder, but write strict outputs under `outputs/strict_vlm_gen`. Route 1/2/3 successful attempts all use pretrained Qwen through `inputs_embeds`; Route 4/5 perform concrete model/API checks and either run if feasible or write detailed evidence-backed blocked reports.

**Tech Stack:** PyTorch, Transformers Qwen2.5-1.5B, PEFT LoRA, existing VTF enhanced tokens `[B,50,512]`, existing A2 EEG encoder.

---

### Task 1: Strict Runner And Reports

**Files:**
- Create: `scripts/run_strict_vlm_gen.py`
- Create: `tests/test_strict_vlm_gen.py`

- [ ] Add tests that assert Qwen prefix route consumes `[B,50,512]` tokens and writes strict reports.
- [ ] Implement shared dataclasses, prompt formatting, metrics, qualitative examples, GPU logging.
- [ ] Implement `CAPTION_TARGET_REPORT.md`, `ALL_ROUTE_METRICS.csv`, `BEST_REPORT_EXAMPLES.md`, and final report aggregation.

### Task 2: Route 1 Qwen Inputs Embeds + LoRA

**Files:**
- Modify: `scripts/run_strict_vlm_gen.py`

- [ ] Implement direct resampler over enhanced visual tokens + EEG tokens + top-k prototypes.
- [ ] Project to Qwen hidden size and concatenate with prompt embeddings through `inputs_embeds`.
- [ ] Train projector + LoRA r=8.
- [ ] If r=8 succeeds, attempt r=16 as a bounded AutoSOTA/rank sweep.
- [ ] Evaluate all required modes and corruptions.

### Task 3: Route 2 Q-Former / Perceiver + Qwen

**Files:**
- Modify: `scripts/run_strict_vlm_gen.py`

- [ ] Implement variants `QFormer_visual_only`, `QFormer_visual_eeg`, `QFormer_visual_eeg_topk`, `QFormer_visual_eeg_topk_lora`.
- [ ] Use query count 8 first; if time allows, query count 16 in AutoSOTA.
- [ ] Evaluate all required modes and corruptions for the most complete variant.

### Task 4: Route 3 LLaVA-Style Projector

**Files:**
- Modify: `scripts/run_strict_vlm_gen.py`

- [ ] Check local/cache availability for LLaVA classes and checkpoints.
- [ ] If no exact LLaVA projector exists, implement a local mm_projector MLP fallback from `[B,50,512]` to Qwen hidden-size visual prefix tokens.
- [ ] Train/evaluate fallback if exact route unavailable.
- [ ] Write blocked report only for exact LLaVA availability if fallback succeeds.

### Task 5: Route 4 BLIP-2 / InstructBLIP

**Files:**
- Modify: `scripts/run_strict_vlm_gen.py`

- [ ] Check installed classes and local cache for BLIP-2/InstructBLIP.
- [ ] Attempt minimal load or class/API inspection.
- [ ] Attempt external embedding/prefix fallback if direct Q-Former replacement is not feasible.
- [ ] Write detailed `BLOCKED_REPORT.md` with commands and errors if unavailable.

### Task 6: Route 5 Qwen-VL Adapter

**Files:**
- Modify: `scripts/run_strict_vlm_gen.py`

- [ ] Check installed classes and local cache for Qwen-VL/Qwen2-VL/Qwen2.5-VL.
- [ ] Inspect visual embedding API if class exists.
- [ ] Attempt prefix fallback through Qwen-VL-like tokenizer/prompt format if no internal adapter is exposed.
- [ ] Write detailed `BLOCKED_REPORT.md` with commands and errors if unavailable.

### Task 7: Verification And Package

**Files:**
- Create/update: `outputs/strict_vlm_gen/*`

- [ ] Run unit tests.
- [ ] Run strict route command.
- [ ] Verify required reports, metrics, examples, blocked reports, and final report.
- [ ] Create refreshed zip containing strict outputs and scripts, excluding large checkpoints.
