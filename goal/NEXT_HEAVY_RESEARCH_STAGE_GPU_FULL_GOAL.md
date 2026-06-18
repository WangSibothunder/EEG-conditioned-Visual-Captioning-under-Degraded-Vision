# Goal: Heavy Research Stage — Use GPU Seriously and Improve Scientific Strength

Current situation:

The project is no longer in the “make it run” phase.
The current Thought2Text pipeline works, constrained semantic captioning works, and paired EEG clearly beats shuffled/random EEG controls. However, the result is not yet strong enough because:

1. Real EEG still does not beat strong vision-only CLIP prototype classification.
2. Alignment search on Thought2Text has saturated.
3. Free-form Qwen captioning is unstable.
4. THINGS-EEG2 and EIT-1M are not yet loader-ready.
5. GPU utilization is low because current jobs are small cached-embedding experiments.

New objective:

```text
Move from small-data smoke experiments to serious EEG representation learning.
```

Main scientific goal:

```text
Train a stronger EEG encoder using larger EEG data, self-supervised pretraining,
tri-modal EEG-image-text contrastive learning, and EEG-specific model architectures.
Then transfer it back to Thought2Text for robust semantic captioning.
```

Do not spend more time on dummy data or weak free-form caption training.

---

# High-Level Priority

Follow this order:

```text
1. Make larger datasets loader-ready: THINGS-EEG2, EIT-1M, optionally EEG-ImageNet / Alljoined1.
2. Run masked EEG reconstruction pretraining on all available EEG.
3. Train tri-modal EEG-image-text contrastive alignment.
4. Implement stronger EEG encoders: DualBranchEEGConformer, Temporal-Spectral-Spatial encoder, Subject-Adaptive Graph encoder.
5. Add neural-aware CLIP adapter / prompt tuning.
6. Transfer best encoder to Thought2Text.
7. Re-run constrained semantic captioning and robust degraded evaluation.
```

---

# Hard Rules

* Do not run more dummy experiments.
* Do not use free-form Qwen generation as primary evidence.
* Do not claim EEG improves visual captioning unless real EEG beats shuffled/random and has meaningful gain under degraded vision.
* Do not let GPU sit idle while only CPU scripts run.
* Do not spend the whole day on 64-sample or 512-sample smoke tests.
* Use smoke tests only to verify code, then immediately run full training.
* Every run must save config, log, metrics, checkpoint, and report.
* If a dataset is unavailable, write a blocker report and move to another useful task.

---

# Required Final Outputs

Create:

```text
outputs/heavy_stage/MASTER_REPORT.md
outputs/heavy_stage/GPU_UTILIZATION_REPORT.md
outputs/heavy_stage/EXPERIMENT_BOARD.csv
outputs/heavy_stage/EXPERIMENT_BOARD.md

outputs/datasets/THINGS_EEG2_READY_REPORT.md
outputs/datasets/EIT1M_READY_REPORT.md
outputs/datasets/DATASET_DECISION_REPORT.md

outputs/pretrain/MASKED_EEG_PRETRAIN_REPORT.md
outputs/pretrain/checkpoints/best_masked_eeg.pt

outputs/trimodal/TRIMODAL_FULL_REPORT.md
outputs/trimodal/checkpoints/best_trimodal.pt

outputs/architectures/ARCHITECTURE_SEARCH_REPORT.md
outputs/architectures/checkpoints/best_encoder.pt

outputs/clip_adapter/CLIP_ADAPTER_REPORT.md

outputs/transfer/TRANSFER_TO_THOUGHT2TEXT_REPORT.md
outputs/transfer/best_transfer_encoder.pt

outputs/final_semantic/FULL_ROBUST_SEMANTIC_REPORT.md
outputs/final_semantic/FULL_METRICS.csv
outputs/final_semantic/FULL_METRICS.md
```

---

# Subagent 1: Dataset Expansion Agent

## Goal

Make larger datasets loader-ready. Thought2Text is too small to be the only training data.

## Dataset A: THINGS-EEG2

Tasks:

1. Check whether `data/THINGS-EEG2` is empty.
2. If empty, download or resume download.
3. Inspect directory structure.
4. Identify preprocessed EEG files.
5. Identify image set.
6. Build current-project-compatible manifest.
7. Run a full loader smoke test.
8. Precompute CLIP/SigLIP image features if images exist.

Expected outputs:

```text
outputs/datasets/THINGS_EEG2_READY_REPORT.md
data/THINGS-EEG2/train.jsonl
data/THINGS-EEG2/val.jsonl
data/THINGS-EEG2/test.jsonl
data/THINGS-EEG2/cache/clip_train.npy
data/THINGS-EEG2/cache/clip_val.npy
data/THINGS-EEG2/cache/clip_test.npy
```

Report must include:

```text
subjects
trial count
unique images
EEG shape
sampling rate if known
split rule
image-level leakage check
loader-ready status
blockers
```

If full THINGS-EEG2 is too large, create a subset:

```text
data/THINGS-EEG2-subset/
  train.jsonl
  val.jsonl
  test.jsonl
```

Target subset:

```text
at least 20k EEG trials if possible
```

---

## Dataset B: EIT-1M

Tasks:

1. Inspect actual downloaded files, not just README.
2. If only metadata/README exists, attempt proper Hugging Face dataset download.
3. Extract EEG/image/text triples.
4. Build manifest.
5. Verify EEG/image/text alignment.
6. Run 512-sample loader smoke.
7. If loader works, build 10k/50k subset for pretraining.

Expected outputs:

```text
outputs/datasets/EIT1M_READY_REPORT.md
data/EIT-1M/train_small.jsonl
data/EIT-1M/val_small.jsonl
data/EIT-1M/test_small.jsonl
```

Report must include:

```text
sample count
EEG shape
image availability
text availability
whether true EEG-image-text pairs exist
whether usable for tri-modal training
blockers
```

---

## Dataset C: Optional EEG-ImageNet / Alljoined1

Do not block main work.

Tasks:

1. Inspect availability.
2. Record download/manual blockers.
3. Check whether images are available.
4. If loader-ready, create small manifest.
5. Do not spend more than 2 hours if blocked.

Output:

```text
outputs/datasets/OPTIONAL_DATASETS_REPORT.md
```

---

# Subagent 2: Heavy Masked EEG Pretraining Agent

## Motivation

Small supervised contrastive training on Thought2Text is saturated. Use self-supervised masked reconstruction to learn EEG spatio-temporal structure before alignment.

Inspired by masked EEG reconstruction approaches in recent EEG-vision-language alignment work.

## Model

Implement:

```text
MaskedEEGAutoencoder
```

Input:

```text
EEG [B, C, T]
```

Mask strategies:

```text
1. Time-span mask
2. Channel mask
3. Time-channel block mask
4. Optional frequency-band mask if spectrogram branch exists
```

Encoder options:

```text
E3 ConvTransformer Base
E4 ConvTransformer Strong
DualBranchEEGConformer
Temporal-Spectral-Spatial Encoder
```

Decoder:

```text
lightweight temporal decoder
reconstruct masked EEG
```

Loss:

```text
L_masked =
  1.0 * MSE on masked regions
+ 0.2 * spectral reconstruction loss
+ 0.1 * smoothness loss
```

If spectrogram features are available:

```text
add STFT magnitude reconstruction
```

## Training

Use all available EEG:

Priority data:

```text
1. THINGS-EEG2 if ready
2. EIT-1M if ready
3. Thought2Text as fallback
```

Recommended config:

```yaml
epochs: 200
batch_size: 256
grad_accum_steps: 1
lr: 1e-4
weight_decay: 0.05
bf16: true
mask_ratio_time: 0.35
mask_ratio_channel: 0.15
num_workers: 8
save_best_by: val_masked_loss
```

If GPU memory allows:

```yaml
batch_size: 512
hidden_dim: 384
layers: 6
```

Outputs:

```text
outputs/pretrain/MASKED_EEG_PRETRAIN_REPORT.md
outputs/pretrain/checkpoints/best_masked_eeg.pt
```

Report:

```text
dataset used
sample count
model parameter count
mask ratios
train/val loss curves
GPU memory peak
time per epoch
checkpoint path
```

---

# Subagent 3: Full Tri-Modal Contrastive Agent

## Motivation

Our task is EEG + Image + Text. Do not only align EEG to images.

Train EEG-image-text shared space:

```text
EEG ↔ Image
EEG ↔ Text
Image ↔ Text
```

Use EIT-1M if available. Otherwise use Thought2Text human captions / BLIP captions. If THINGS-EEG2 has no text, generate class/template/BLIP captions.

## Model

Use pretrained EEG encoder from masked pretraining if available.

Encoders:

```text
EEG encoder: best masked-pretrained encoder
Image encoder: frozen CLIP/SigLIP image encoder or cached embeddings
Text encoder: frozen CLIP/SigLIP text encoder
Projection heads: trainable
```

Loss:

```text
L_total =
  1.0 * EEG-Image symmetric InfoNCE
+ 0.7 * EEG-Text symmetric InfoNCE
+ 0.3 * Image-Text consistency
+ 0.3 * Class CE if labels exist
+ 0.3 * Prototype Alignment
+ 0.2 * Similarity Distillation
+ 0.2 * Same-image Cross-subject Consistency if available
```

Metrics:

```text
EEG→image R@1/R@5/R@10
EEG→text R@1/R@5/R@10
EEG→class top1/top5
unique-image retrieval
trial-level retrieval
random baseline
```

Training config:

```yaml
epochs: 100
batch_size: 256
lr: 1e-4
bf16: true
patience: 15
seeds: [42, 123, 2025] if time
```

Outputs:

```text
outputs/trimodal/TRIMODAL_FULL_REPORT.md
outputs/trimodal/checkpoints/best_trimodal.pt
```

Important:

The previous tri-modal run was only 64 samples / 1 epoch. That is not a valid experiment. This time run full data or the largest available subset.

---

# Subagent 4: EEG-Specific Architecture Agent

## Motivation

The current ConvTransformer is not enough. Try stronger EEG-specific architectures.

Implement and train these architectures under the same alignment protocol.

## Architecture A: DualBranchEEGConformer

Inspired by dual-branch temporal/spatial EEG conformer designs.

Structure:

```text
Temporal branch:
  multi-scale temporal convolution
  temporal conformer / transformer
  long-range temporal modeling

Spatial branch:
  channel attention
  spatial transformer over EEG channels
  optional electrode graph attention

Fusion:
  concat temporal + spatial
  subject adapter
  projection to 512
```

Target parameters:

```text
10M–30M
```

## Architecture B: Temporal-Spectral-Spatial Transformer

Structure:

```text
Raw EEG temporal-spatial branch
+
STFT / wavelet spectrogram branch
+
feature fusion block
+
Transformer
+
projection to 512
```

Use frequency bands:

```text
theta
alpha
beta
gamma
broadband
```

## Architecture C: Subject-Adaptive Graph Encoder

Structure:

```text
channel graph attention
subject-specific FiLM adapter
same-image cross-subject consistency loss
projection to 512
```

## Architecture D: Raw + Spectrogram Late Fusion

Structure:

```text
raw EEG encoder
spectrogram CNN
late fusion
projection
```

## Training

For each architecture:

1. Smoke test 2 epochs.
2. If smoke passes, run full 80–120 epochs.
3. Use same train/val/test split.
4. Compare against current historical best.

Metrics:

```text
R@1/R@5/R@10
class accuracy
mean rank
GPU memory
time per epoch
overfit gap
```

Outputs:

```text
outputs/architectures/ARCHITECTURE_SEARCH_REPORT.md
outputs/architectures/checkpoints/best_encoder.pt
```

Report table:

```text
| Architecture | Params | Dataset | Pretrain | R@1 | R@5 | R@10 | Class Acc | GPU Mem | Notes |
```

---

# Subagent 5: Neural-Aware CLIP Adapter Agent

## Motivation

Current pipeline freezes CLIP as a static feature extractor. This may be suboptimal because EEG and CLIP image embeddings have a physiological-symbolic gap.

Try lightweight CLIP adaptation, not full CLIP finetuning.

## Methods

Try:

```text
1. Visual prompt tokens inserted into CLIP ViT
2. Small MLP adapter after CLIP image embedding
3. Class prototype adapter
4. Image-text prototype calibration
```

Train only small parameters:

```text
visual prompts
adapter MLP
prototype calibration layer
```

Do not full-finetune CLIP.

Loss:

```text
EEG-image contrastive
EEG-text contrastive
class CE
prototype consistency
```

Compare:

```text
frozen CLIP baseline
CLIP adapter
CLIP prompt tokens
```

Outputs:

```text
outputs/clip_adapter/CLIP_ADAPTER_REPORT.md
```

Metrics:

```text
EEG→image retrieval
EEG→class accuracy
semantic caption accuracy
whether adapter improves over frozen CLIP
```

---

# Subagent 6: Robust Semantic Captioning Agent

## Motivation

Final deliverable should not be free-form Qwen caption. Use constrained semantic captioning and robust evaluation.

## Task

Use best encoder from:

```text
1. masked pretraining
2. tri-modal alignment
3. architecture search
4. CLIP adapter
```

Then train robust semantic fusion:

```text
degraded image embedding + EEG embedding
→ reliability-aware gated fusion
→ class prototype / classifier
→ controlled caption
```

## Reliability-aware gate

Implement:

```text
vision_confidence = max softmax over visual class prototypes
degradation_embedding = optional corruption token
gate = sigmoid(MLP([image_emb, eeg_emb, vision_confidence]))
fused = image_emb + gate * eeg_delta
```

Train with degraded image embeddings:

```text
clean
mild blur
strong blur
mild noise
strong noise
occlusion 25%
occlusion 50%
lowres 32
lowres 16
```

Do not only use weak degradation. Make vision-only meaningfully weaker.

## Training modes

Compare:

```text
vision_only
real_eeg
shuffled_eeg
random_eeg
eeg_only
real_eeg + reliability gate
real_eeg + no gate
```

Loss:

```text
L =
  class CE
+ supervised contrastive
+ prototype alignment
+ real-vs-shuffled margin
+ gate regularization
+ degradation-aware gate target
```

Gate target idea:

```text
If visual confidence is high, gate can be low.
If visual confidence is low and real EEG is paired, gate should be allowed to rise.
Do not force random/shuffled EEG gate to rise.
```

Metrics:

```text
Top1 / Top5 class accuracy
Class Hit
Real-Shuffled Gap
Real-Random Gap
Real-Vision Gap under strong degradation
Win Rate over controls
Win Rate over vision
Gate Shift under degradation
```

Outputs:

```text
outputs/final_semantic/FULL_ROBUST_SEMANTIC_REPORT.md
outputs/final_semantic/FULL_METRICS.csv
outputs/final_semantic/FULL_METRICS.md
outputs/final_semantic/GATE_ANALYSIS.md
outputs/final_semantic/QUALITATIVE_EXAMPLES.md
```

---

# Subagent 7: GPU Scheduler Agent

## Goal

Do not let the server idle.

Create:

```text
scripts/heavy_stage_scheduler.py
outputs/heavy_stage/GPU_UTILIZATION_REPORT.md
```

Policy:

```text
1. CPU/data tasks can always run in parallel.
2. Only run multiple GPU jobs if each is small and memory allows.
3. For heavy masked pretraining or tri-modal full training, use one large job with large batch.
4. If GPU memory < 12GB and util < 40% for 10 minutes, launch another queued alignment job.
5. If no large dataset is ready, run masked pretraining on Thought2Text while dataset agent works.
6. If THINGS/EIT becomes ready, switch next queued job to larger dataset pretraining.
```

Log every 5 minutes:

```text
time
running job
GPU utilization
GPU memory
power
batch size
step time
current epoch
```

Write:

```text
outputs/heavy_stage/GPU_UTILIZATION_REPORT.md
```

---

# Execution Plan

Run in this order, but with parallel subagents.

## Immediately

Start these CPU/data tasks:

```text
Dataset Agent: THINGS-EEG2 + EIT-1M readiness
Semantic Agent: robustness report update
Architecture Agent: implement DualBranchEEGConformer and TSST
Scheduler Agent: GPU monitoring
```

Start this GPU task:

```text
Masked EEG pretraining on all currently available EEG
```

If only Thought2Text is available, still run masked pretraining on Thought2Text as a fallback, but switch to THINGS/EIT when ready.

## After masked pretraining starts

Queue:

```text
1. full tri-modal contrastive
2. DualBranchEEGConformer alignment
3. Temporal-Spectral-Spatial alignment
4. CLIP adapter alignment
5. robust semantic fusion
```

## If GPU is still low

Increase:

```text
batch_size
hidden_dim
number of layers
epochs
number of candidates
dataset size
```

But do not increase uselessly if validation degrades.

---

# Minimum Completion Criteria

This stage is complete if:

```text
1. At least one large-data dataset is made loader-ready or clearly blocked.
2. Masked EEG pretraining completes.
3. Full tri-modal contrastive training runs beyond smoke scale.
4. At least two EEG-specific architectures are fully trained.
5. CLIP adapter or prompt tuning is attempted.
6. Best encoder is transferred to robust semantic captioning.
7. Final metrics compare real EEG, shuffled EEG, random EEG, and vision-only.
8. Master report states whether the stronger training improved over current results.
```

---

# Current Baseline to Beat

Current constrained semantic captioning already shows:

```text
real EEG beats shuffled/random controls in all clean/degraded conditions
but real EEG does not beat strong vision-only CLIP prototype baseline.
```

Current alignment bottleneck:

```text
best current/parallel sweep alignment is not consistently better than historical best.
```

The new goal is to beat:

```text
1. Historical best Thought2Text EEG→image R@5
2. Current semantic caption real EEG accuracy
3. Current real-vs-shuffled/random gaps
4. Current inability to beat vision-only under stronger degradation
```

---

# Final Scientific Claim Rule

Only claim what metrics support.

Allowed if supported:

```text
Paired EEG provides control-specific semantic information beyond shuffled/random EEG.
```

Stronger claim allowed only if supported:

```text
Under sufficiently degraded visual inputs, paired EEG improves semantic prediction over vision-only.
```

Not allowed:

```text
The model reads thoughts.
```

Not allowed:

```text
Open-ended captioning is solved.
```
