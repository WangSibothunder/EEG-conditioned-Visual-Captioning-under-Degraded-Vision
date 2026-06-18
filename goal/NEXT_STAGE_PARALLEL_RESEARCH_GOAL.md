# Goal: Parallel Research Upgrade for EEG + Vision → Robust Semantic Captioning

Current project status:

The pipeline is already running. Real Thought2Text data has been loaded. EEG→CLIP alignment is better than random, but still not strong enough for final claims. Free-form Qwen caption generation is unstable and often produces code-like / URL-like / class-token outputs. Gate values do not yet prove selective EEG usage.

Therefore, stop treating this as a simple caption training problem.

New research objective:

```text
Build a credible EEG-assisted robust semantic captioning system under degraded visual conditions.
```

Core claim to test:

```text
Correctly paired EEG provides auxiliary semantic information under degraded visual input,
and this should be measurable through class-level semantic prediction, class-hit captioning,
real-vs-shuffled/random EEG gaps, and robustness gains.
```

Do not rely on open-ended LLM caption generation as the main evidence.

---

# High-Level Strategy

Run multiple research directions in parallel using subagents.

Main directions:

```text
1. Controlled semantic captioning instead of free-form Qwen captioning
2. Stronger EEG encoder / loss search
3. Larger dataset preparation and pretraining
4. Tri-modal EEG-image-text contrastive learning
5. Subject adaptation and same-image cross-subject consistency
6. Time-frequency / spectrogram EEG branch
7. Robustness and real/shuffled/random EEG evaluation
```

The GPU was under-utilized previously. This goal should keep the server active by running independent subagent tasks and experiment queues.

---

# Hard Rules

Follow these strictly:

* Do not full-finetune Qwen.
* Do not use free-form Qwen caption generation as the main metric.
* Do not claim EEG helps unless real EEG beats shuffled/random controls.
* Do not overwrite previous Day3/Day4/Day5 outputs.
* Use image-level split only.
* Do not allow the same image_id to appear in multiple splits.
* Every subagent must produce a report.
* Every training run must save config, logs, metrics, and checkpoint.
* If a run fails, write a failure report and continue with another safe experiment.
* Keep GPU busy with queued jobs, but do not launch LLM fusion concurrently with multiple alignment jobs.

---

# Required Final Outputs

By the end of this goal, create:

```text
outputs/parallel_stage/MASTER_REPORT.md
outputs/parallel_stage/EXPERIMENT_BOARD.csv
outputs/parallel_stage/EXPERIMENT_BOARD.md

outputs/semantic_caption/SEMANTIC_CAPTION_REPORT.md
outputs/semantic_caption/FULL_METRICS.csv
outputs/semantic_caption/FULL_METRICS.md

outputs/alignment_search/SEARCH_SUMMARY.md
outputs/alignment_search/best_overall.pt

outputs/datasets/THINGS_EEG2_STATUS.md
outputs/datasets/EIT1M_STATUS.md

outputs/trimodal/TRIMODAL_STATUS.md
outputs/subject_adaptation/SUBJECT_ADAPTATION_REPORT.md
outputs/spectrogram/SPECTROGRAM_BRANCH_REPORT.md
outputs/robustness/ROBUSTNESS_REPORT.md
```

If some dataset is missing, create a clear missing-data report instead of failing silently.

---

# Subagent Structure

Use multiple subagents or simulate them with separate independent tasks.

## Main Agent

Responsibilities:

* Coordinate all subagents.
* Maintain experiment board.
* Avoid duplicate work.
* Monitor GPU utilization.
* Launch queued training jobs.
* Collect reports.
* Produce final `MASTER_REPORT.md`.

The main agent should update:

```text
outputs/parallel_stage/LIVE_STATUS.md
outputs/parallel_stage/EXPERIMENT_BOARD.csv
outputs/parallel_stage/EXPERIMENT_BOARD.md
```

Experiment board columns:

```text
experiment_id
subagent
task_type
dataset
encoder
loss
seed
status
start_time
end_time
gpu_mem_peak
metric_primary
metric_secondary
checkpoint_path
report_path
notes
```

---

# Subagent A: Controlled Semantic Captioning Agent

## Motivation

Previous free-form Qwen captioning is unstable. For this small 40-class EEG dataset, use constrained semantic captioning first.

## Task A1: Build class/text prototype bank

Create:

```text
src/eval/prototype_captioner.py
scripts/build_text_prototypes.py
```

Inputs:

```text
data/thought2text/train_human_caption.jsonl
data/thought2text/val_human_caption.jsonl
data/thought2text/test_human_caption.jsonl
```

Build:

```text
1. CLIP image prototype per class
2. CLIP text prototype per class
3. class name map
4. optional BLIP caption prototype per class
```

Save:

```text
data/thought2text/cache/class_image_prototypes.npy
data/thought2text/cache/class_text_prototypes.npy
data/thought2text/cache/class_name_map.json
outputs/semantic_caption/prototype_bank_report.md
```

## Task A2: Train semantic fusion classifier

Do not use Qwen here.

Model:

```text
image_emb + eeg_emb
→ gated fusion
→ classifier / prototype matcher
→ 40-way semantic class prediction
```

Training modes:

```text
F0: vision_only
F1: image + real EEG
F2: image + shuffled EEG control
F3: image + random EEG control
F4: eeg_only
```

Loss:

```text
L_total =
  1.0 * Class CE
+ 0.5 * Supervised Contrastive Loss
+ 0.3 * Prototype Alignment
+ 0.2 * Real-vs-Shuffled Margin
+ 0.05 * Gate Regularization
```

Use degraded image embeddings during training:

```text
clean
blur
occlusion
noise
lowres
```

Create:

```text
src/train/train_semantic_fusion.py
scripts/run_semantic_fusion.sh
```

Output:

```text
outputs/semantic_caption/checkpoints/best.pt
outputs/semantic_caption/semantic_fusion_report.md
```

Metrics:

```text
Top-1 class accuracy
Top-5 class accuracy
Class hit
Real EEG - shuffled EEG gap
Real EEG - random EEG gap
Gate mean by corruption
```

## Task A3: Controlled caption generation

After predicting class:

```text
predicted_class_name → "a photo of a {predicted_class_name}"
```

Create:

```text
src/eval/constrained_caption_eval.py
scripts/run_constrained_caption_eval.sh
```

Output:

```text
outputs/semantic_caption/predictions.jsonl
outputs/semantic_caption/FULL_METRICS.csv
outputs/semantic_caption/FULL_METRICS.md
outputs/semantic_caption/qualitative_examples.md
```

Required table:

```text
| Corruption | Mode | Top1 Acc | Top5 Acc | Class Hit | Real-Shuffled Gap | Real-Random Gap | Gate Mean |
```

---

# Subagent B: EEG Encoder and Loss Search Agent

## Motivation

The current EEG encoder is a useful baseline but too small to be the final model. Explore stronger encoder/loss combinations systematically.

## Task B1: Implement encoder families

All encoders output:

```text
z_eeg: [B, 512]
```

Test these encoders:

```text
E0: Tiny current encoder
E1: EEGNet-style temporal + depthwise spatial conv
E2: Multi-scale temporal convolution + TCN
E3: ConvTransformer Base
E4: ConvTransformer Strong
E5: Subject-adaptive ConvTransformer
E6: Same-image multi-subject consistency encoder
E7: Spectrogram-CNN branch
```

### E0: Tiny baseline

Use current encoder unchanged.

Target:

```text
0.5M–1M params
```

### E1: EEGNet-style encoder

Architecture:

```text
Temporal convolution
Depthwise spatial convolution over EEG channels
Separable convolution
BatchNorm
ELU/GELU
Dropout
Global average pooling
MLP to 512
```

Target:

```text
0.5M–2M params
```

### E2: Multi-scale temporal conv

Architecture:

```text
Parallel Conv1D branches with kernels 3, 7, 15, 31
Concatenate
Residual TCN blocks with dilation 1, 2, 4, 8
Attention pooling
MLP to 512
```

Target:

```text
2M–5M params
```

### E3: ConvTransformer Base

Architecture:

```text
Temporal conv stem
Spatial channel mixer
4-layer Transformer
8 heads
hidden_dim 256
FFN dim 1024
attention pooling
projection head
```

Target:

```text
2M–6M params
```

### E4: ConvTransformer Strong

Architecture:

```text
Multi-scale temporal conv stem
Spatial/channel attention
6-layer Transformer
8 heads
hidden_dim 384
FFN dim 1536
attention pooling
projection head
```

Target:

```text
8M–20M params
```

### E5: Subject-adaptive encoder

Use E3 plus:

```text
subject embedding
subject-specific FiLM adapter
subject-specific affine normalization
```

Use only if subject_id exists.

### E6: Same-image multi-subject encoder

Use E3 or E5, but add loss that pulls EEG embeddings from different subjects viewing the same image closer together.

### E7: Spectrogram-CNN branch

Convert EEG to time-frequency representation.

Options:

```text
STFT
Morlet wavelet
bandpower features
```

Architecture:

```text
spectrogram → CNN → pooled feature → projector to 512
```

Also support late fusion:

```text
raw EEG encoder + spectrogram encoder → fusion → 512
```

---

## Task B2: Implement modular losses

Implement losses that can be turned on/off by config:

```text
L0: MSE + Class CE
L1: Symmetric InfoNCE + Cosine + Class CE
L2: Multi-positive InfoNCE
L3: Supervised Contrastive over class labels
L4: Prototype Alignment
L5: Similarity Distillation
L6: EEG Augmentation Consistency
L7: Same-image Cross-subject Consistency
L8: Hard Negative Contrast
L9: Tri-modal EEG-image-text Contrastive
```

Default weights:

```yaml
lambda_infonce: 1.0
lambda_multi_positive: 0.5
lambda_cosine: 0.5
lambda_class_ce: 0.3
lambda_supcon: 0.2
lambda_proto: 0.2
lambda_similarity: 0.2
lambda_aug: 0.1
lambda_same_image_subject: 0.2
lambda_hard_negative: 0.05
lambda_trimodal: 0.5
temperature: 0.07
tau_sim: 0.1
```

## Task B3: Search queue

Create:

```text
scripts/generate_alignment_experiments.py
scripts/launch_alignment_sweep.py
configs/generated_alignment/
outputs/alignment_search/
```

Candidate grid:

```text
T0: E0 + L0
T1: E0 + L1
T2: E1 + L1
T3: E2 + L1
T4: E3 + L1

S1: E3 + L1 + L2 + L4
S2: E3 + L1 + L2 + L3 + L4
S3: E3 + L1 + L2 + L4 + L5
S4: E3 + L1 + L2 + L4 + L5 + L6
S5: E3 + L1 + L2 + L3 + L4 + L5 + L6

X1: E5 + L1 + L2 + L4 + L5
X2: E5 + L1 + L2 + L4 + L5 + L7
X3: E6 + L1 + L2 + L3 + L4 + L7

G1: E4 + L1 + L2 + L4
G2: E4 + L1 + L2 + L4 + L5 + L6

P1: E7 + L1 + L4 + L5
P2: raw E3 + spectrogram E7 + L1 + L4 + L5
```

Use successive halving:

```text
1. Run all candidates for 20 epochs.
2. Keep top 50%.
3. Continue top 50% to 50 epochs.
4. Keep top 4.
5. Train top 4 to early stopping.
6. Run best 1–2 candidates with seeds 42, 123, 2025.
```

Primary ranking score:

```text
score = val_unique_R@5 + 0.5 * class_accuracy - 0.1 * overfit_gap
```

Report:

```text
outputs/alignment_search/SEARCH_SUMMARY.md
outputs/alignment_search/RANKING.csv
outputs/alignment_search/RANKING.md
outputs/alignment_search/best_overall.pt
```

Required metrics:

```text
trial-level R@1/R@5/R@10
unique-image-level R@1/R@5/R@10
class top-1/top-5 accuracy
mean rank
median rank
random baseline
```

---

# Subagent C: Larger Dataset Preparation Agent

## Motivation

Thought2Text is useful but small. Prepare larger datasets for stronger EEG pretraining.

## Task C1: THINGS-EEG2 inspection and schema conversion

If THINGS-EEG2 is available, inspect it.

Create:

```text
scripts/inspect_things_eeg2.py
scripts/build_things_eeg2_manifest.py
outputs/datasets/THINGS_EEG2_STATUS.md
```

Report:

```text
data existence
subjects
trial count
image count
EEG shape
available preprocessed files
image path availability
whether image-level split can be built
whether current dataset loader supports it
```

If possible, build:

```text
data/THINGS-EEG2/train.jsonl
data/THINGS-EEG2/val.jsonl
data/THINGS-EEG2/test.jsonl
```

Run a 1024-sample smoke alignment.

Output:

```text
outputs/datasets/things_eeg2_smoke_report.md
```

## Task C2: EIT-1M inspection and schema conversion

If EIT-1M is available, inspect it.

Create:

```text
scripts/inspect_eit1m.py
scripts/build_eit1m_manifest.py
outputs/datasets/EIT1M_STATUS.md
```

Report:

```text
file list
whether EEG exists
whether image exists
whether text/caption exists
sample count
EEG shape
image format
text format
whether it can be converted to current manifest schema
```

If possible, build a small manifest:

```text
data/EIT-1M/train_small.jsonl
data/EIT-1M/val_small.jsonl
data/EIT-1M/test_small.jsonl
```

Run a 512-sample tri-modal smoke test.

Output:

```text
outputs/datasets/eit1m_smoke_report.md
```

## Task C3: Dataset recommendation

Create:

```text
outputs/datasets/DATASET_RECOMMENDATION.md
```

Answer:

```text
1. Should we keep Thought2Text as main dataset?
2. Can THINGS-EEG2 be used for pretraining?
3. Can EIT-1M be used for EEG-image-text training?
4. Which dataset should be used next?
5. What are blockers?
```

---

# Subagent D: Tri-Modal Contrastive Agent

## Motivation

Our project is EEG + Image + Text. Do not only align EEG to image. Align EEG, image, and text in one semantic space.

## Task D1: Build text embeddings

Use class names and/or captions.

Text sources:

```text
human class caption
BLIP caption
EIT-1M text if available
```

Build CLIP text embeddings:

```text
data/thought2text/cache/text_embeddings.npy
data/thought2text/cache/text_index.json
```

Report:

```text
outputs/trimodal/text_embedding_report.md
```

## Task D2: Train tri-modal alignment

Create:

```text
src/train/train_trimodal_align.py
scripts/run_trimodal_align.sh
```

Loss:

```text
L_total =
  L_eeg_image_contrast
+ L_eeg_text_contrast
+ L_image_text_contrast
+ L_class_CE
+ L_similarity_distillation
+ L_prototype_alignment
```

Train on Thought2Text first. If EIT-1M is available, run smoke on EIT-1M.

Metrics:

```text
EEG→image R@1/R@5/R@10
EEG→text R@1/R@5/R@10
EEG→class top-1/top-5
text prototype classification
```

Output:

```text
outputs/trimodal/TRIMODAL_STATUS.md
outputs/trimodal/checkpoints/best.pt
```

---

# Subagent E: Subject Adaptation and Same-Image Consistency Agent

## Motivation

EEG is subject-dependent. If multiple subjects saw the same image, exploit it.

## Task E1: Analyze subject/image structure

Create:

```text
outputs/subject_adaptation/subject_image_structure.md
```

Report:

```text
number of subjects
number of shared images across subjects
trials per subject
images per subject
whether same-image cross-subject pairs exist
```

## Task E2: Train with subject adapter

Use:

```text
subject embedding
FiLM adapter
subject-specific LayerNorm
```

Compare:

```text
without subject adapter
with subject adapter
```

Output:

```text
outputs/subject_adaptation/SUBJECT_ADAPTATION_REPORT.md
```

## Task E3: Same-image cross-subject consistency

Add loss:

```text
same image_id, different subject_id → embeddings should be close
```

Report:

```text
outputs/subject_adaptation/SAME_IMAGE_CONSISTENCY_REPORT.md
```

Metrics:

```text
R@5
class accuracy
cross-subject retrieval
same-image consistency score
```

---

# Subagent F: Robustness Evaluation Agent

## Motivation

The final claim depends on degraded visual conditions and real/shuffled/random EEG controls.

## Task F1: Standardize robustness metrics

Create:

```text
src/eval/robustness_metrics.py
outputs/robustness/ROBUSTNESS_METRIC_DEFINITION.md
```

Metrics:

```text
Robustness Gain = score(real_eeg, degraded) - score(vision_only, degraded)

EEG Specific Gain =
  score(real_eeg, degraded)
- max(score(shuffled_eeg, degraded), score(random_eeg, degraded))

Win Rate =
  number of corruptions where real_eeg beats both controls / total corruptions

Gate Shift =
  gate(degraded) - gate(clean)

Class Hit Gap =
  class_hit(real_eeg) - class_hit(shuffled_eeg/random_eeg)
```

## Task F2: Full evaluation

Evaluate best checkpoints from:

```text
alignment_search
semantic_caption
trimodal if available
subject_adaptation if available
```

Conditions:

```text
clean
blur
occlusion
noise
lowres
```

Modes:

```text
vision_only
real_eeg
shuffled_eeg
random_eeg
eeg_only
```

Output:

```text
outputs/robustness/ROBUSTNESS_REPORT.md
outputs/robustness/full_metrics.csv
outputs/robustness/full_metrics.md
outputs/robustness/qualitative_examples.md
```

---

# Parallel Execution Policy

The main agent should keep the server busy.

## GPU jobs

GPU jobs include:

```text
alignment search
semantic fusion
tri-modal alignment
spectrogram encoder training
THINGS/EIT smoke alignment
```

Run multiple alignment/small jobs concurrently only if GPU memory allows.

Policy:

```text
If GPU memory < 10GB and GPU util < 40%, launch another small alignment job.
If GPU memory < 20GB and GPU util < 60%, allow 2–3 small jobs.
Do not exceed 4 concurrent GPU jobs.
Do not run Qwen/fusion jobs concurrently with many alignment jobs.
```

## CPU jobs

CPU/data jobs can run in parallel:

```text
dataset inspection
manifest building
caption target building
WordNet label mapping
metric computation
report writing
```

---

# Priority Order

Use this priority order:

```text
1. Controlled semantic captioning
2. Alignment search with stronger encoders/losses
3. THINGS-EEG2 and EIT-1M inspection
4. Tri-modal contrastive on Thought2Text
5. Subject adaptation / same-image consistency
6. Spectrogram branch
7. Full robustness evaluation
8. Final master report
```

Do not delay semantic captioning forever while searching encoders.

Do not delay dataset inspection forever while training Thought2Text.

---

# Completion Criteria

This goal is complete if:

```text
1. Controlled semantic captioning produces stable class-level captions.
2. Real EEG is compared against shuffled/random EEG.
3. At least 8 alignment candidates are tested.
4. At least 3 encoder families are tested.
5. THINGS-EEG2 and EIT-1M availability is reported.
6. Tri-modal contrastive is attempted or clearly blocked.
7. Subject adaptation is attempted if subject_id exists.
8. Spectrogram branch is attempted or clearly blocked.
9. Robustness metrics are computed.
10. MASTER_REPORT.md clearly states what worked and what failed.
```

---

# Scientific Claim Rule

Allowed claim:

```text
Correctly paired EEG improves constrained semantic prediction under some degraded visual conditions compared with shuffled/random EEG.
```

Allowed only if supported by metrics.

Do not claim:

```text
The model reads thoughts.
```

Do not claim:

```text
Open-ended caption generation is solved.
```

Current most defensible framing:

```text
EEG provides measurable auxiliary semantic information for robust visual understanding under degraded image inputs.
```
