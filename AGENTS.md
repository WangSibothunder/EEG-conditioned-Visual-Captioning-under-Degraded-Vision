# AGENTS.md — EEG + Vision → Caption Project Agent Guide

This file is the **repository-level instruction document** for AI coding agents working on this project.

It is intentionally structured as an engineering specification, not as a paper draft. Follow it when planning, editing, testing, and reporting code changes.

---

## 0. One-Sentence Project Definition

Build a **single-GPU, good-enough, reproducible research prototype** for:

```text
Image + EEG → Text Caption
```

The target research framing is:

```text
EEG-assisted robust image captioning under degraded visual conditions
```

The core claim we want to test:

> When the visual input is degraded or ambiguous, synchronized EEG may provide auxiliary perceptual information that improves caption generation compared with vision-only, random-EEG, and shuffled-EEG baselines.

This is **not** a pure EEG-to-text mind-reading system.  
This is **not** a large-scale multimodal foundation model.  
This is a fast, controlled, course-project-friendly EEG + vision captioning prototype.

---

## 1. Agent Operating Principles

### 1.1 Primary rule

Before editing code, understand the current stage:

1. Are we still building the dummy-data MVP?
2. Are we migrating to real EEG data?
3. Are we training alignment?
4. Are we running degraded-vision experiments?
5. Are we preparing final reports?

Do not jump to a later stage until the earlier stage passes its acceptance checks.

### 1.2 Good agent behavior

A good agent should:

- Keep changes small and testable.
- Prefer simple, robust implementations over clever ones.
- Add shape assertions around multimodal tensors.
- Update documentation when interfaces change.
- Save runnable commands in `docs/RUNBOOK.md`.
- Save experiment status in `docs/STATUS.md`.
- Report what was tested and what was not tested.
- Avoid downloading huge datasets or models unless explicitly requested.
- Avoid changing public interfaces silently.
- Avoid “fixing” unrelated code.

### 1.3 Bad agent behavior

Do not:

- Full-finetune an LLM by default.
- Start with 7B/8B models before the 1.5B debug path works.
- Add distributed training before single-GPU training works.
- Replace the planned gated fusion with complex cross-attention in the first version.
- Spend time on real dataset preprocessing before the dummy pipeline runs end-to-end.
- Claim EEG helps unless random/shuffled EEG controls are implemented.
- Store multiple augmented copies of large image datasets.
- Put secrets, tokens, or private paths into committed files.

---

## 2. Global Roadmap

### Phase 0 — Repository Skeleton and Context Control

Goal:

- Create project structure.
- Create configs.
- Create dummy data.
- Create interface contracts.
- Create run scripts.
- Create minimal tests.

Acceptance:

- `bash scripts/run_smoke.sh` or equivalent runs without crashing.
- A dummy batch can be loaded.
- Docs exist and match code.

---

### Phase 1 — Image-Only Caption Baseline

Goal:

```text
Image → Frozen CLIP → Soft Prompt Projector → Frozen LLM → Caption
```

Acceptance:

- Image-only baseline trains on dummy data.
- Loss is finite.
- At least one checkpoint is saved.
- Sample captions are generated.
- No EEG is required for this mode.

---

### Phase 2 — EEG + Vision Fusion MVP

Goal:

```text
Image → Frozen CLIP → image_emb
EEG   → EEG Encoder → eeg_emb
image_emb + eeg_emb → Gated Fusion → Soft Prompt → Frozen LLM → Caption
```

Acceptance:

- Fusion model trains on dummy data.
- `image_only`, `real_eeg`, `shuffled_eeg`, and `random_eeg` generation modes all run.
- Output JSONL files are saved for all modes.

---

### Phase 3 — Real Dataset Migration

Goal:

Replace dummy data with real processed EEG/image/caption triplets.

Candidate datasets:

- Thought2Text / CVPR2017 EEG visual dataset.
- THINGS-EEG2.
- EEG-ImageNet.
- Other datasets only after the above path works.

Acceptance:

- Real manifest loads with the same dataset interface as dummy data.
- Shape normalization is documented.
- Missing files are reported clearly.

---

### Phase 4 — EEG-to-CLIP Alignment

Goal:

Pretrain EEG encoder:

```text
EEG → CLIP image embedding
```

Loss:

```text
L = InfoNCE + MSE + optional class CE
```

Acceptance:

- Retrieval metrics are reported:
  - EEG → image R@1
  - EEG → image R@5
  - optional class accuracy
- Alignment checkpoint can be loaded into fusion model.

---

### Phase 5 — Degraded Vision Experiment

Goal:

Test whether EEG helps more when vision is degraded.

Visual conditions:

- clean
- blur
- occlusion
- noise
- low-resolution / downsample-upsample

Baselines:

- vision-only
- EEG-only
- vision + real EEG
- vision + shuffled EEG
- vision + random EEG

Acceptance:

- Results are saved in tables.
- Qualitative examples are saved.
- Do not claim EEG helps without the controls.

---

## 3. Current Near-Term Mission

The immediate two-day mission is:

```text
Finish Phase 0 + Phase 1 + minimal Phase 2 skeleton.
```

This means:

1. Dummy data works.
2. Image-only baseline works.
3. EEG encoder forward pass works.
4. Gated fusion forward pass works.
5. Fusion training loop works.
6. Sanity-check generation modes exist.

Do not optimize final accuracy yet.

---

## 4. Hardware and Runtime Assumptions

Default environment:

```text
Single GPU
Target VRAM: 48GB
Preferred dtype: bf16
```

### 4.1 GPU no-idle policy

Do not wait passively on one small alignment run if the GPU is mostly idle.
Low GPU utilization is a scheduling signal: keep useful work queued and launch
safe concurrent jobs instead of leaving the server idle. Jobs do not need to be
lightweight when the server has headroom; heavier Base/Strong alignment runs,
fusion controls, degraded-sanity generation, and cache/precompute jobs are all
acceptable if they are scientifically useful and auditable.

For Day4+ research runs:

- Monitor `nvidia-smi`, GPU memory, GPU utilization, and running Python training processes before launching more work.
- If a valid experiment is already running, do not interrupt it.
- While it runs, prepare the next configs, reports, and launch queue.
- If GPU memory usage is below 10GB and GPU utilization is below 40%, launch another alignment screening, cache, or sanity job when safe.
- If GPU memory usage is below 20GB and GPU utilization is below 60%, allow multiple concurrent useful GPU jobs.
- If the server still has clear headroom, prefer heavier high-value jobs over tiny filler jobs.
- Current heavy-stage override: up to 8 concurrent small non-LLM training jobs are allowed when memory headroom is clear and each job is auditable; run only one heavy pretraining/tri-modal job at a time unless memory is clearly sufficient.
- Do not run fusion or LLM training concurrently with alignment search jobs unless explicitly requested and the current GPU headroom supports it.
- Prefer short screening runs first, then longer runs for the best candidates.
- Keep a structured experiment board so parallel runs remain auditable.
- If no alignment job is running and the GPU is idle, start the next useful fallback in this order:
  CLIP feature/cache comparison, caption target generation, then dataset inspection smoke tests.
- Record every queued, running, completed, failed, or skipped job in the experiment board before launching more work.

Allowed:

- Single-GPU PyTorch.
- Gradient accumulation.
- Frozen CLIP.
- Frozen LLM.
- Optional LoRA after baseline works.
- Optional 4-bit loading after baseline works.

Avoid by default:

- Multi-GPU DDP.
- DeepSpeed.
- FSDP.
- Full LLM finetuning.
- Very long sequence length.
- Large image caches for every augmentation.

Practical memory rules:

- Use Qwen2.5-1.5B-Instruct for debugging.
- Move to Qwen2.5-7B-Instruct only after the pipeline is stable.
- Precompute CLIP embeddings once for real data.
- Generate image corruptions on the fly unless explicitly asked to cache.

---

## 5. Recommended Repository Layout

Use this structure unless the repository already has a better one.

```text
eeg_vision_caption/
  AGENTS.md
  README.md
  requirements.txt

  configs/
    debug.yaml
    real_small.yaml
    train_48gb.yaml

  docs/
    PROJECT_BRIEF.md
    GLOBAL_ROADMAP.md
    INTERFACE_CONTRACTS.md
    DECISIONS.md
    STATUS.md
    RUNBOOK.md
    EXPERIMENTS.md

  data/
    README.md

  src/
    __init__.py

    data/
      __init__.py
      dataset.py
      collate.py
      dummy_data.py
      manifest.py
      corruptions.py

    models/
      __init__.py
      vision_encoder.py
      eeg_encoder.py
      fusion.py
      caption_model.py
      alignment_model.py

    train/
      __init__.py
      train_baseline.py
      train_fusion.py
      train_align.py

    eval/
      __init__.py
      generate.py
      metrics.py
      sanity_check.py
      retrieval.py

    utils/
      __init__.py
      config.py
      seed.py
      logger.py
      checkpoint.py
      tensor.py

  scripts/
    make_dummy_data.py
    precompute_vision.py
    run_smoke.sh
    run_baseline.sh
    run_fusion.sh
    run_align.sh
    run_generate.sh
    run_sanity.sh

  outputs/
    .gitkeep
```

---

## 6. Required Documentation Files

### 6.1 `docs/PROJECT_BRIEF.md`

Must explain:

- The project goal.
- Why this is EEG + vision, not EEG-only.
- The expected input and output.
- The high-level model architecture.
- The good-enough success criteria.

### 6.2 `docs/GLOBAL_ROADMAP.md`

Must list:

- Phase 0 to Phase 5.
- What is complete.
- What is blocked.
- What is next.

### 6.3 `docs/INTERFACE_CONTRACTS.md`

Must define:

- Manifest schema.
- Dataset output.
- Collate output.
- Vision encoder output.
- EEG encoder output.
- Fusion output.
- Caption model input.
- Generation output.

### 6.4 `docs/DECISIONS.md`

Append design decisions in this format:

```text
## YYYY-MM-DD — Decision Title

Decision:
...

Reason:
...

Alternatives considered:
...

Consequences:
...
```

### 6.5 `docs/STATUS.md`

Update after every meaningful task:

```text
# Status

## Done
- ...

## In Progress
- ...

## Blocked
- ...

## Next
- ...
```

### 6.6 `docs/RUNBOOK.md`

Must contain exact commands:

- create dummy data
- run smoke test
- train baseline
- train fusion
- generate captions
- run sanity check
- run alignment
- evaluate metrics

### 6.7 `docs/EXPERIMENTS.md`

Must track experiments in a table:

```text
| ID | Config | Data | Model | Mode | Status | Notes |
```

---

## 7. Interface Contracts

### 7.1 Manifest JSONL schema

Each line must be valid JSON:

```json
{
  "image_id": "000001",
  "image_path": "images/000001.jpg",
  "eeg_path": "eeg/000001.npy",
  "caption": "a photo of a red object",
  "label": 0,
  "subject_id": "S01",
  "split": "train"
}
```

Required fields:

- `image_id`
- `image_path`
- `caption`

Optional fields:

- `eeg_path`
- `label`
- `subject_id`
- `split`
- `metadata`

The loader must tolerate missing optional fields.

### 7.2 Dataset item

```python
{
    "image": FloatTensor[3, 224, 224],
    "eeg": FloatTensor[C, T] | None,
    "caption": str,
    "image_id": str,
    "label": int | None,
    "subject_id": str | None,
}
```

### 7.3 Collate batch

```python
{
    "image": FloatTensor[B, 3, 224, 224],
    "eeg": FloatTensor[B, C, T] | None,
    "caption": list[str],
    "image_id": list[str],
    "label": LongTensor[B] | None,
    "subject_id": list[str] | None,
}
```

### 7.4 Vision encoder

```python
image_emb = vision_encoder(images)
```

Input:

```text
images: FloatTensor[B, 3, 224, 224]
```

Output:

```text
image_emb: FloatTensor[B, D_img]
```

Default:

```text
D_img = 512 for CLIP ViT-B/32
```

### 7.5 EEG encoder

```python
eeg_emb = eeg_encoder(eeg)
```

Input:

```text
eeg: FloatTensor[B, C, T]
```

Default dummy shape:

```text
C = 64
T = 250
```

Output:

```text
eeg_emb: FloatTensor[B, D_eeg]
```

Default:

```text
D_eeg = 512
```

### 7.6 Gated fusion

```python
fused_emb = fusion(image_emb, eeg_emb)
```

Recommended formula:

```python
gate = sigmoid(MLP(concat(image_emb, eeg_emb)))
eeg_delta = MLP(eeg_emb)
fused = image_emb + gate * eeg_delta
```

Shapes:

```text
image_emb: [B, 512]
eeg_emb:   [B, 512]
fused:     [B, 512]
```

### 7.7 Soft prompt projector

```python
soft_prompt = prompt_projector(fused_emb)
```

Shape:

```text
soft_prompt: FloatTensor[B, K, D_llm]
```

Default:

```text
K = 8
```

### 7.8 Caption generation output

Generation script must write JSONL:

```json
{
  "image_id": "000001",
  "mode": "real_eeg",
  "reference": "a photo of a red object",
  "prediction": "a red object is shown"
}
```

Allowed modes:

```text
image_only
real_eeg
shuffled_eeg
random_eeg
eeg_only
```

---

## 8. Model Specification

### 8.1 Vision encoder

Default:

```text
openai/clip-vit-base-patch32
```

Rules:

- Freeze all parameters.
- Use model-provided preprocessing if possible.
- Expose only normalized image embeddings to downstream code.
- For real data, support cached embeddings.

### 8.2 LLM decoder

Debug default:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Later optional:

```text
Qwen/Qwen2.5-7B-Instruct
```

Rules:

- Freeze LLM by default.
- Use tokenizer from the same model.
- Train soft prompt projector first.
- LoRA is optional and only after frozen-LLM baseline works.
- Never full-finetune the LLM unless explicitly requested.

### 8.3 EEG encoder

Default lightweight architecture:

```text
EEG [B, C, T]
→ Conv1d temporal stem
→ BatchNorm
→ GELU
→ temporal downsampling
→ small Transformer Encoder, 2 layers
→ mean pooling
→ MLP projection
→ EEG embedding [B, 512]
```

Rules:

- Keep parameter count small.
- Add shape checks.
- Avoid large attention over raw long time sequences.
- Downsample before Transformer.
- Prefer robust simple architecture over complex EEG foundation models for MVP.

### 8.4 Fusion module

Default:

```text
Gated residual EEG fusion
```

Do not use cross-attention in the first implementation.

Cross-attention can be added later only after:

- image-only baseline works,
- gated fusion works,
- sanity checks run.

---

## 9. Training Recipes

### 9.1 Debug config

Use `configs/debug.yaml`.

Recommended:

```yaml
seed: 42

data:
  train_manifest: "data/train.jsonl"
  val_manifest: "data/val.jsonl"
  image_root: "data"
  eeg_root: "data"
  image_size: 224
  max_caption_len: 64

model:
  vision_encoder: "openai/clip-vit-base-patch32"
  llm_name: "Qwen/Qwen2.5-1.5B-Instruct"
  freeze_vision: true
  freeze_llm: true
  use_4bit: false
  image_embed_dim: 512
  eeg_embed_dim: 512
  num_soft_prompt_tokens: 8

eeg:
  channels: 64
  time_steps: 250
  hidden_dim: 128
  output_dim: 512
  transformer_layers: 2
  dropout: 0.1

train:
  batch_size: 2
  grad_accum_steps: 4
  epochs: 2
  lr_projector: 1.0e-4
  lr_eeg: 1.0e-4
  lr_fusion: 1.0e-4
  weight_decay: 0.01
  bf16: true
  fp16: false
  num_workers: 2
  log_every: 10
  save_every_epoch: true

output:
  dir: "outputs/debug"
```

### 9.2 Baseline training

Pipeline:

```text
image → frozen CLIP → soft prompt projector → frozen LLM → caption CE
```

Trainable:

- soft prompt projector only.

Frozen:

- CLIP
- LLM

Acceptance:

- finite loss
- saved checkpoint
- generated sample captions

### 9.3 Fusion training

Pipeline:

```text
image → frozen CLIP → image_emb
eeg → EEG encoder → eeg_emb
image_emb + eeg_emb → gated fusion → soft prompt projector → frozen LLM → caption CE
```

Trainable:

- EEG encoder
- fusion
- prompt projector

Frozen:

- CLIP
- LLM

Acceptance:

- finite loss
- saved checkpoint
- generated captions in all sanity modes

### 9.4 Alignment training

Pipeline:

```text
eeg → EEG encoder → projector → CLIP image space
```

Loss:

```text
InfoNCE + MSE + optional class CE
```

Acceptance:

- R@1/R@5 computed.
- checkpoint can be loaded into fusion model.

---

## 10. Required Scripts

### 10.1 `scripts/make_dummy_data.py`

Creates:

```text
data/images/*.jpg
data/eeg/*.npy
data/train.jsonl
data/val.jsonl
```

Default dummy EEG:

```text
[64, 250]
```

Default size:

```text
128 train samples
32 val samples
```

### 10.2 `scripts/run_smoke.sh`

Must run quick checks:

```bash
python scripts/make_dummy_data.py --num_train 16 --num_val 4
python -m src.data.dataset --config configs/debug.yaml
python -m src.models.caption_model --smoke_test --config configs/debug.yaml
```

If module-level smoke tests are not implemented, create a simple `tests/smoke_test.py`.

### 10.3 `scripts/run_baseline.sh`

Runs image-only training.

### 10.4 `scripts/run_fusion.sh`

Runs EEG+vision fusion training.

### 10.5 `scripts/run_generate.sh`

Runs caption generation.

### 10.6 `scripts/run_sanity.sh`

Runs:

```text
image_only
real_eeg
shuffled_eeg
random_eeg
```

and writes one JSONL per mode.

---

## 11. Evaluation and Sanity Checks

### 11.1 Minimum metrics

Implement lightweight metrics first:

- exact file output check
- average generated length
- distinct predictions count
- ROUGE-L if available
- BLEU if available

Optional later:

- CIDEr
- METEOR
- BERTScore

### 11.2 Sanity checks

Mandatory checks before claiming any EEG effect:

1. Real EEG mode runs.
2. Shuffled EEG mode runs.
3. Random EEG mode runs.
4. Image-only mode runs.
5. Real EEG is not worse than random EEG in the main degraded setting before making any strong claim.

### 11.3 Degraded vision conditions

Add only after the dummy and real-data basic pipeline work.

Recommended corruptions:

- Gaussian blur
- cutout occlusion
- Gaussian noise
- low-resolution downsample-upsample

Generate corruptions on the fly.

---

## 12. Context Management and Subagent Workflow

If the coding environment supports subagents, use them. If not, simulate subagents by creating separate task sections and reports.

### 12.1 Main agent responsibilities

The main agent must:

- maintain the roadmap,
- assign small tasks,
- merge changes,
- run integration tests,
- update docs,
- write final status.

### 12.2 Subagent A — Documentation and Context

Scope:

```text
docs/*
README.md
AGENTS.md
```

Tasks:

- create/update project docs,
- record decisions,
- keep run commands current,
- summarize status.

### 12.3 Subagent B — Data

Scope:

```text
src/data/*
scripts/make_dummy_data.py
data/README.md
```

Tasks:

- implement dummy data,
- implement manifest loading,
- implement dataset and collate,
- add shape checks.

### 12.4 Subagent C — Models

Scope:

```text
src/models/*
```

Tasks:

- implement CLIP wrapper,
- implement EEG encoder,
- implement gated fusion,
- implement caption model,
- add smoke tests.

### 12.5 Subagent D — Training

Scope:

```text
src/train/*
scripts/run_baseline.sh
scripts/run_fusion.sh
scripts/run_align.sh
```

Tasks:

- implement training loops,
- implement optimizer and dtype handling,
- save checkpoints,
- log sample captions.

### 12.6 Subagent E — Evaluation

Scope:

```text
src/eval/*
scripts/run_generate.sh
scripts/run_sanity.sh
```

Tasks:

- implement generation,
- implement sanity modes,
- implement metrics,
- save JSONL outputs.

### 12.7 Subagent F — Integration QA

Scope:

```text
all scripts and docs
```

Tasks:

- run smoke test,
- run baseline,
- run fusion,
- run generation,
- update report.

### 12.8 Subagent report format

Each subagent should return:

```text
Files changed:
- ...

What works:
- ...

Tests run:
- ...

Known issues:
- ...

Next suggested action:
- ...
```

---

## 13. Coding Standards

### 13.1 Python style

- Use Python 3.10+.
- Use type hints where helpful.
- Use clear function names.
- Keep functions short.
- Add concise Chinese comments for nontrivial project-specific logic.
- Keep API names and model names in English.
- Avoid hidden global state.

### 13.2 Error handling

Provide clear errors for:

- missing manifest,
- missing image,
- missing EEG file,
- wrong EEG shape,
- unsupported generation mode,
- missing checkpoint,
- CUDA out-of-memory.

### 13.3 Tensor checks

Use helper functions such as:

```python
assert_shape(tensor, expected_rank, name)
```

At minimum, check:

- image rank is 4 in model forward,
- EEG rank is 3 in EEG encoder,
- image/eeg batch sizes match,
- soft prompt rank is 3.

### 13.4 Checkpoints

Save:

```text
outputs/<run_name>/checkpoints/
outputs/<run_name>/samples/
outputs/<run_name>/logs/
outputs/<run_name>/config.yaml
```

Do not overwrite previous runs unless explicitly requested.

---

## 14. Testing Requirements

Minimum tests:

```text
1. Dummy dataset generation.
2. Dataset returns one item.
3. Collate returns one batch.
4. CLIP wrapper returns [B, D].
5. EEG encoder returns [B, 512].
6. Fusion returns [B, 512].
7. Caption model returns finite loss.
8. Baseline script runs one mini epoch.
9. Fusion script runs one mini epoch.
10. Generation writes JSONL.
```

Suggested command:

```bash
bash scripts/run_smoke.sh
```

If there is no formal test framework, implement smoke tests with plain Python scripts.

---

## 15. Data and Storage Rules

Do not assume unlimited disk.

Default good-enough storage target:

```text
50GB to 120GB total project storage
```

Rules:

- Do not download full ImageNet unless explicitly requested.
- Do not cache every corrupted image version.
- Precompute CLIP embeddings for real images.
- Store captions as JSONL.
- Store EEG arrays as `.npy`, `.pt`, or memory-mapped arrays.
- Use relative paths in manifests.
- Do not commit large data or checkpoints.

Recommended ignored paths:

```text
data/images/
data/eeg/
data/cache/
outputs/
checkpoints/
*.pt
*.pth
*.safetensors
*.npy
*.npz
*.mmap
```

---

## 16. Commands to Keep Working

These commands should remain valid as the project evolves:

```bash
python scripts/make_dummy_data.py --num_train 128 --num_val 32

bash scripts/run_smoke.sh

bash scripts/run_baseline.sh

bash scripts/run_fusion.sh

bash scripts/run_generate.sh

bash scripts/run_sanity.sh
```

Real-data commands can be added later.

---

## 17. Good-Enough Success Criteria

For the current two-day milestone, success means:

- Dummy data pipeline works.
- Image-only baseline trains.
- Fusion model trains.
- Generation works.
- Sanity modes work.
- Documentation is updated.
- No major architecture rewrite is needed to plug in real data.

Weak captions on dummy data are acceptable.

For the final course project, success means:

- Real data loads.
- EEG encoder trains or loads.
- Fusion model produces captions.
- At least one degraded-vision condition shows a plausible benefit from real EEG over random/shuffled EEG.
- Results are documented with qualitative examples and ablation tables.

---

## 18. Final Report Expectations

The project should eventually produce:

```text
1. Method diagram
2. Dataset description
3. Model architecture
4. Training stages
5. Baseline comparison
6. Sanity checks
7. Degraded-vision experiment
8. Qualitative examples
9. Limitations
10. Reproducible commands
```

The report should not overclaim.

Correct framing:

```text
EEG may provide auxiliary perceptual cues under degraded vision.
```

Avoid:

```text
The system reads thoughts directly.
```

---

## 19. If Blocked

If blocked, do not improvise a large redesign.

Follow this fallback order:

1. Make dummy data smaller.
2. Reduce batch size.
3. Use Qwen2.5-1.5B instead of 7B.
4. Freeze more modules.
5. Disable LoRA.
6. Replace LLM generation with a small causal LM for smoke testing.
7. Replace CLIP with cached random embeddings only for shape tests.
8. Report the blocker in `docs/STATUS.md`.

---

## 20. Final Instruction to Agents

Always preserve the research direction:

```text
EEG + Vision → Caption
```

Always keep the engineering direction:

```text
simple, frozen backbones, small trainable adapters, strong sanity checks
```

Always keep the scientific caution:

```text
no EEG claim without shuffled/random controls
```


从现在开始，不允许只跑 1 epoch / 64 sample / 512 sample 就写“完成”。smoke test 只能用于验证代码，验证通过后必须自动进入 full training。GPU 如果连续 10 分钟低于 8GB 显存占用或低于 30x % 利用率，必须写诊断并启动更大 batch、更大模型或下一个有效训练任务。当前 small-job 并发上限按用户指令提高到 8；heavy pretraining / tri-modal 仍应优先单独作为主 GPU 任务。
