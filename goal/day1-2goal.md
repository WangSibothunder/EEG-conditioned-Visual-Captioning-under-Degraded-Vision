对，你这个判断是对的。**上一版不是完整全局计划，而是全局计划里的 Day1–Day2 MVP 子任务**。更合理的 Codex 工作方式应该是：

> **主 Agent 负责全局决策和集成；Subagent 分别负责数据、模型、训练、评估、文档，避免一个上下文里混太多信息。**

下面这版是我建议你直接复制给 Codex 的新版 Goal。它会要求 Codex 先建立项目上下文文档，再开 subagent 分工做两天任务。

---

````markdown
# GLOBAL GOAL: EEG + Vision → Caption Research Prototype

We are building a research-course prototype for:

Image + EEG → Text Caption

The final project idea is:

EEG-assisted Robust Image Captioning under Visual Degradation

Core claim:
When visual input is degraded or ambiguous, synchronized EEG can provide auxiliary perceptual information and improve caption generation compared with vision-only, random EEG, and shuffled EEG baselines.

This is NOT a pure EEG-to-text project.
This is NOT a full large-scale multimodal foundation model.
This is a good-enough, single-GPU, fast-iteration research prototype.

---

# Important Context

This project has a global roadmap:

## Phase 0: Project skeleton and context management
Build clean repo structure, documentation, interface contracts, dummy data, and runnable scripts.

## Phase 1: Image-only caption baseline
Run:
Image → Frozen CLIP → Trainable Projector → Frozen LLM → Caption

This gives a stable baseline.

## Phase 2: Add EEG encoder and fusion
Run:
Image → Frozen CLIP → image embedding
EEG → EEG Encoder → EEG embedding
image embedding + EEG embedding → Gated Fusion → Frozen LLM → Caption

## Phase 3: Robustness experiment
Test clean image and degraded image:
- clean
- blur
- occlusion
- noise
- low-resolution

Compare:
- vision-only
- EEG-only
- vision + real EEG
- vision + shuffled EEG
- vision + random EEG

## Phase 4: Real dataset migration
Start with dummy data first.
Then migrate to real EEG/image/caption data:
- Thought2Text / CVPR2017 EEG dataset
- THINGS-EEG2 if available
- EEG-ImageNet if available

Current coding goal:
Finish Phase 0, Phase 1, and a minimal Phase 2 skeleton within two days.

---

# Hardware Assumption

Single GPU.
Assume around 48GB VRAM.

Therefore:
- Do not use distributed training.
- Do not use DeepSpeed unless absolutely necessary.
- Do not full finetune the LLM.
- Freeze CLIP.
- Freeze LLM by default.
- Train only small modules first:
  - projector
  - EEG encoder
  - fusion module
  - optional LoRA later

---

# Engineering Philosophy

Prioritize:

1. Runnable pipeline
2. Clear module boundaries
3. Easy debugging
4. Good-enough caption output
5. EEG ablation sanity checks

Do NOT prioritize:
- SOTA accuracy
- complex cross-attention fusion first
- full LLM finetuning
- multi-GPU optimization
- huge EEG foundation models
- complex real-data preprocessing before dummy pipeline works

---

# Subagent Strategy

You should use subagents to manage context.

The main agent should NOT try to implement everything in one context.

Use this structure:

## Main Agent
Responsibilities:
- Maintain global roadmap.
- Decide interfaces.
- Review subagent outputs.
- Merge code.
- Run integration tests.
- Update project status.

## Subagent A: Context / Documentation Agent
Responsibilities:
- Create and maintain documentation files.
- Keep task status updated.
- Record design decisions.
- Prevent context loss.

## Subagent B: Data Agent
Responsibilities:
- Implement dummy data generation.
- Implement manifest format.
- Implement dataset loader.
- Implement collate function.
- Ensure image/eeg/caption samples load correctly.

## Subagent C: Model Agent
Responsibilities:
- Implement frozen CLIP wrapper.
- Implement EEG encoder.
- Implement gated fusion.
- Implement caption model wrapper.
- Ensure all tensor shapes are correct.

## Subagent D: Training Agent
Responsibilities:
- Implement image-only training.
- Implement EEG+vision fusion training.
- Implement checkpointing.
- Implement mixed precision.
- Log loss and sample captions.

## Subagent E: Evaluation / Ablation Agent
Responsibilities:
- Implement generation script.
- Implement simple metrics.
- Implement sanity checks:
  - real EEG
  - shuffled EEG
  - random EEG
  - image-only

## Subagent F: Integration / QA Agent
Responsibilities:
- Run scripts end-to-end.
- Detect broken imports.
- Detect shape mismatch.
- Detect GPU memory risks.
- Produce final run report.

Each subagent should work on a small set of files and produce a short report.

---

# Context Management Rules

Before coding, create these files:

```text
docs/
  PROJECT_BRIEF.md
  GLOBAL_ROADMAP.md
  INTERFACE_CONTRACTS.md
  DECISIONS.md
  STATUS.md
  RUNBOOK.md
````

## PROJECT_BRIEF.md

Must explain:

* Project goal
* What the system inputs and outputs are
* Why this is EEG+vision, not EEG-only
* Main technical route

## GLOBAL_ROADMAP.md

Must contain:

* Phase 0
* Phase 1
* Phase 2
* Phase 3
* Phase 4
* Current two-day milestone

## INTERFACE_CONTRACTS.md

Must define the exact interfaces between modules.

For example:

```python
batch = {
    "image": FloatTensor[B, 3, 224, 224],
    "eeg": FloatTensor[B, C, T],
    "caption": List[str],
    "input_ids": LongTensor[B, L],
    "labels": LongTensor[B, L],
    "image_id": List[str],
    "label": LongTensor[B],
}
```

Vision encoder output:

```python
image_emb: FloatTensor[B, D_img]
```

EEG encoder output:

```python
eeg_emb: FloatTensor[B, D_eeg]
```

Fusion output:

```python
fused_emb: FloatTensor[B, D_fused]
```

Soft prompt output:

```python
soft_prompt: FloatTensor[B, K, D_llm]
```

## DECISIONS.md

Must record important choices, for example:

* Use CLIP ViT-B/32 first, not ViT-L/14.
* Use Qwen2.5-1.5B first, not 7B.
* Use frozen LLM first.
* Use gated fusion first, not cross-attention.
* Use dummy dataset first.

## STATUS.md

Must be updated after each milestone:

* Done
* In progress
* Blocked
* Next action

## RUNBOOK.md

Must contain exact commands:

* create dummy data
* train baseline
* train fusion
* run generation
* run sanity check

---

# Repository Structure

Create this structure:

```text
eeg_vision_caption/
  README.md
  requirements.txt

  docs/
    PROJECT_BRIEF.md
    GLOBAL_ROADMAP.md
    INTERFACE_CONTRACTS.md
    DECISIONS.md
    STATUS.md
    RUNBOOK.md

  configs/
    base.yaml
    debug.yaml

  data/
    README.md

  src/
    __init__.py

    data/
      __init__.py
      dataset.py
      collate.py
      dummy_data.py

    models/
      __init__.py
      vision_encoder.py
      eeg_encoder.py
      fusion.py
      caption_model.py

    train/
      __init__.py
      train_baseline.py
      train_fusion.py

    eval/
      __init__.py
      generate.py
      metrics.py
      sanity_check.py

    utils/
      __init__.py
      seed.py
      logger.py
      checkpoint.py
      config.py

  scripts/
    make_dummy_data.py
    run_baseline.sh
    run_fusion.sh
    run_generate.sh
    run_sanity.sh

  outputs/
    .gitkeep
```

---

# Day 1 Workload

Day 1 goal:
Finish Phase 0 and Phase 1.

That means:

* Repo skeleton works.
* Dummy data works.
* Dataset loader works.
* Image-only caption baseline can train for a few steps.
* It can generate sample captions.

---

## Day 1 — Subagent A: Context / Documentation

Create:

```text
docs/PROJECT_BRIEF.md
docs/GLOBAL_ROADMAP.md
docs/INTERFACE_CONTRACTS.md
docs/DECISIONS.md
docs/STATUS.md
docs/RUNBOOK.md
```

The documentation should be concise but useful.

Important:
The docs must explicitly say Day1-Day2 are only the first part of the global plan.

---

## Day 1 — Subagent B: Data

Implement dummy dataset.

Create:

```text
scripts/make_dummy_data.py
src/data/dummy_data.py
src/data/dataset.py
src/data/collate.py
```

Dummy data format:

```text
data/
  images/
    000001.jpg
    000002.jpg
  eeg/
    000001.npy
    000002.npy
  train.jsonl
  val.jsonl
```

Each JSONL line:

```json
{
  "image_id": "000001",
  "image_path": "images/000001.jpg",
  "eeg_path": "eeg/000001.npy",
  "caption": "a photo of a red object",
  "label": 0
}
```

Dummy EEG shape:

```python
[C, T] = [64, 250]
```

Dataset output:

```python
{
    "image": image_tensor,        # [3, 224, 224]
    "eeg": eeg_tensor,            # [64, 250]
    "caption": caption_string,
    "image_id": image_id,
    "label": label,
}
```

Collate output:

```python
{
    "image": FloatTensor[B, 3, 224, 224],
    "eeg": FloatTensor[B, 64, 250],
    "caption": List[str],
    "image_id": List[str],
    "label": LongTensor[B],
}
```

Acceptance test:
A dataloader should return one valid batch without errors.

---

## Day 1 — Subagent C: Model Baseline

Implement:

```text
src/models/vision_encoder.py
src/models/caption_model.py
src/models/fusion.py
```

### Vision encoder

Use frozen CLIP first.

Recommended:

* `openai/clip-vit-base-patch32`
* Output dim is usually 512.

Interface:

```python
class FrozenCLIPVisionEncoder(nn.Module):
    def forward(self, images) -> torch.Tensor:
        # images: [B, 3, 224, 224]
        # return: [B, D_img]
```

All CLIP weights must be frozen.

### Baseline caption model

Build an image-only caption model:

```text
image_emb → soft prompt projector → frozen LLM → caption loss
```

Recommended LLM for debugging:

* `Qwen/Qwen2.5-1.5B-Instruct`

Do not start with 7B.
Use 1.5B first to debug quickly.

The projector should map image embedding to K soft prompt tokens:

```python
image_emb: [B, D_img]
projector(image_emb): [B, K, D_llm]
```

K = 8 by default.

Then concatenate soft prompt embeddings before text token embeddings.

Acceptance test:
Forward pass returns a scalar loss.

---

## Day 1 — Subagent D: Training Baseline

Implement:

```text
src/train/train_baseline.py
scripts/run_baseline.sh
```

The baseline training script should:

1. Load config.
2. Load dataset.
3. Load frozen CLIP.
4. Load frozen LLM.
5. Train only soft prompt projector.
6. Log loss.
7. Save checkpoint.
8. Generate several validation captions after each epoch.

Use:

* bf16 if available
* batch size 2–4 for debug
* grad accumulation 4–8
* epochs 1–3

Acceptance criteria:

* The script runs end-to-end on dummy data.
* Loss is finite.
* A checkpoint is saved.
* Sample captions are written to output file.

---

# Day 2 Workload

Day 2 goal:
Finish minimal Phase 2.

That means:

* EEG encoder works.
* EEG+vision gated fusion works.
* Fusion training runs.
* Sanity-check modes exist:

  * real EEG
  * shuffled EEG
  * random EEG
  * image-only

---

## Day 2 — Subagent C: EEG Encoder and Fusion

Implement:

```text
src/models/eeg_encoder.py
src/models/fusion.py
```

### EEG encoder

Use a lightweight EEG encoder:

Input:

```python
eeg: [B, C, T] = [B, 64, 250]
```

Architecture:

```text
EEG
→ temporal Conv1d
→ BatchNorm / GELU
→ temporal downsampling
→ small Transformer Encoder, 2 layers
→ mean pooling
→ MLP projector
→ eeg_emb
```

Output:

```python
eeg_emb: [B, 512]
```

Keep the model small.
Do not build a huge EEG Transformer.

### Gated fusion

Use this first:

```python
gate = sigmoid(MLP(concat(image_emb, eeg_emb)))
eeg_delta = MLP(eeg_emb)
fused = image_emb + gate * eeg_delta
```

Shape:

* image_emb: [B, 512]
* eeg_emb: [B, 512]
* fused: [B, 512]

Do not implement cross-attention in the first version.

Acceptance test:
Fusion forward pass works with dummy tensors.

---

## Day 2 — Subagent D: Fusion Training

Implement:

```text
src/train/train_fusion.py
scripts/run_fusion.sh
```

Fusion training pipeline:

```text
image → frozen CLIP → image_emb
eeg → EEG encoder → eeg_emb
image_emb + eeg_emb → gated fusion → fused_emb
fused_emb → soft prompt projector → frozen LLM → caption CE loss
```

Train:

* EEG encoder
* gated fusion
* soft prompt projector

Freeze:

* CLIP
* LLM

Acceptance criteria:

* Training script runs.
* Loss is finite.
* Checkpoint is saved.
* Sample captions are generated.

---

## Day 2 — Subagent E: Evaluation and Sanity Checks

Implement:

```text
src/eval/generate.py
src/eval/sanity_check.py
scripts/run_generate.sh
scripts/run_sanity.sh
```

Generation modes:

```text
--mode image_only
--mode real_eeg
--mode shuffled_eeg
--mode random_eeg
```

Mode behavior:

## image_only

Ignore EEG.

## real_eeg

Use correct EEG paired with image.

## shuffled_eeg

Shuffle EEG inside the batch or across dataset.

## random_eeg

Replace EEG with random Gaussian noise with same shape.

Output should be saved as JSONL:

```json
{
  "image_id": "000001",
  "mode": "real_eeg",
  "reference": "a photo of a red object",
  "prediction": "a red object is shown"
}
```

Acceptance criteria:

* The same checkpoint can generate in all four modes.
* No mode crashes.
* Outputs are saved.

---

## Day 2 — Subagent F: Integration / QA

Run all scripts:

```bash
bash scripts/run_baseline.sh
bash scripts/run_fusion.sh
bash scripts/run_generate.sh
bash scripts/run_sanity.sh
```

Then update:

```text
docs/STATUS.md
docs/RUNBOOK.md
```

Create a final short report:

```text
outputs/two_day_report.md
```

The report should contain:

* What works
* What does not work
* Current model structure
* Current command list
* Current output examples
* Next recommended steps

---

# Config

Create `configs/debug.yaml`:

```yaml
seed: 42

data:
  train_manifest: "data/train.jsonl"
  val_manifest: "data/val.jsonl"
  image_root: "data"
  eeg_root: "data"
  image_size: 224
  max_caption_len: 64
  use_dummy_data: true

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

---

# Good-Enough Stopping Criteria

Do not over-optimize.

After two days, success means:

1. Dummy data pipeline works.
2. Image-only baseline trains.
3. EEG+vision fusion trains.
4. All four generation modes run:

   * image_only
   * real_eeg
   * shuffled_eeg
   * random_eeg
5. There are saved sample captions.
6. The project has clean docs and run commands.
7. The code is ready to replace dummy data with real EEG data.

It is acceptable if caption quality is weak on dummy data.
The current goal is system correctness, not final performance.

---

# After Day 2: Next Steps

Only after the above works:

## Step 1

Replace dummy data with real processed Thought2Text / CVPR2017 data.

## Step 2

Cache CLIP image embeddings.

## Step 3

Train EEG-to-CLIP alignment:

```text
EEG → CLIP image embedding
loss = InfoNCE + MSE + optional class CE
```

## Step 4

Use pretrained EEG encoder inside fusion caption model.

## Step 5

Run degraded-vision experiments:

* clean
* blur
* occlusion
* noise
* low-res

Then compare:

* vision-only
* vision + real EEG
* vision + shuffled EEG
* vision + random EEG

---

# Critical Warnings

Do not skip image-only baseline.
Do not skip shuffled/random EEG sanity checks.
Do not start with 7B unless 1.5B works.
Do not use cross-attention fusion in the first implementation.
Do not spend time on real dataset preprocessing before dummy pipeline works.
Do not claim EEG helps unless shuffled/random EEG controls are implemented.

---

# Final Deliverable After This Goal

At the end of this goal, the repository should contain:

```text
1. Runnable code
2. Dummy data generator
3. Image-only baseline
4. EEG+vision fusion model
5. Training scripts
6. Generation scripts
7. Sanity-check scripts
8. Documentation
9. two_day_report.md
```

The main agent should coordinate subagents, merge their outputs, run integration tests, and update documentation.

```

你可以直接把这版作为 **Codex 总 Goal**。核心变化是：它不再只是“写一个 Day1–Day2 脚本”，而是先建立**全局工程上下文 + subagent 分工 + 接口契约**，再执行前两天任务。这样后面换真实 EEG 数据、加退化视觉实验、接更大 LLM，Codex 不容易迷路。
```
