# Overnight Goal: Real EEG Alignment + Strong Loss + First Caption Fusion Run

The dummy MVP already runs. Stop optimizing dummy-only code.

Tonight's goal is to produce real scientific progress:

1. Use real Thought2Text / CVPR2017 processed EEG data if available.
2. Build or verify real train/val/test manifests.
3. Precompute CLIP image embeddings.
4. Train EEG → CLIP alignment using stronger losses.
5. Evaluate EEG → image retrieval.
6. Run a small EEG+vision caption fusion experiment using the best aligned EEG encoder.
7. Produce an overnight report by the end.

The most important output tomorrow morning is not pretty captions.
The most important output is:

```text
Does EEG align to visual CLIP space better than random?
```

Only after this is true should we trust EEG+vision captioning.

---

# Runtime Assumption

The server now has internet access.

Use a single GPU.

Assume 48GB VRAM.

It is acceptable to run overnight for 8–12 hours.

Do not use multi-GPU or distributed training unless explicitly requested.

Use `tmux`, `nohup`, or a background script.

---

# Tonight's Main Principle

Do not rely on caption CE alone.

Use a multi-objective EEG representation loss inspired by recent EEG-vision decoding work:

```text
L_total =
  L_InfoNCE
+ λ_mse * L_MSE
+ λ_cls * L_object_CE
+ λ_sim * L_similarity_distillation
+ λ_aug * L_EEG_augmentation_consistency
```

Default weights:

```yaml
lambda_mse: 0.5
lambda_cls: 0.2
lambda_sim: 0.2
lambda_aug: 0.1
temperature: 0.07
```

If labels are unavailable, skip `L_object_CE`.

If a loss term causes instability, automatically disable only that term and continue.

---

# Stage 0: Audit Existing Repo

Before training, create:

```text
outputs/overnight/audit_report.md
```

It must include:

1. Current git commit or file timestamp summary.
2. Whether dummy pipeline still runs.
3. Whether real Thought2Text data exists.
4. GPU name and VRAM.
5. Python, torch, CUDA versions.
6. Available model cache paths.
7. Exact command used to start overnight run.

Run:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
bash scripts/run_smoke.sh || true
```

Do not block the whole overnight run if dummy smoke fails, but record the failure.

---

# Stage 1: Real Thought2Text Data Inspection

Expected layout:

```text
data/thought2text/
  block/
    eeg_55_95_std.pth
    block_splits_by_image_all.pth
  images/
```

Run or implement:

```bash
bash scripts/inspect_thought2text.sh
```

Output:

```text
outputs/overnight/thought2text_inspection.md
```

The inspection report must include:

1. Keys inside `eeg_55_95_std.pth`.
2. Keys inside `block_splits_by_image_all.pth`.
3. EEG sample count.
4. EEG tensor shape.
5. Number of labels.
6. Number of images found.
7. Number of missing images.
8. Split sizes.
9. Example 5 samples.

If data is missing, create:

```text
outputs/overnight/missing_data_instructions.md
```

and stop real-data training gracefully.

Do not silently fall back to dummy data for the overnight research run.

---

# Stage 2: Build Manifest

Create or update:

```bash
bash scripts/build_thought2text_manifest.sh
```

It should write:

```text
data/thought2text/train.jsonl
data/thought2text/val.jsonl
data/thought2text/test.jsonl
```

Each line:

```json
{
  "image_id": "...",
  "image_path": "...",
  "eeg_index": 123,
  "caption": "a photo of a {class_name}",
  "label": 5,
  "split": "train",
  "subject_id": null
}
```

If real captions are unavailable, use class-based short captions:

```text
"a photo of a {class_name}"
```

If class names are unavailable:

```text
"a photo of an object from class {label}"
```

Acceptance:

```bash
python -m src.data.dataset \
  --manifest data/thought2text/train.jsonl \
  --root data/thought2text \
  --smoke_test
```

Save batch shape to:

```text
outputs/overnight/manifest_report.md
```

---

# Stage 3: Precompute CLIP Image Features

Use:

```text
openai/clip-vit-base-patch32
```

Implement or run:

```bash
python scripts/precompute_vision.py \
  --manifest data/thought2text/train.jsonl \
  --image_root data/thought2text \
  --out data/thought2text/cache/clip_train.npy \
  --index_out data/thought2text/cache/clip_index_train.json

python scripts/precompute_vision.py \
  --manifest data/thought2text/val.jsonl \
  --image_root data/thought2text \
  --out data/thought2text/cache/clip_val.npy \
  --index_out data/thought2text/cache/clip_index_val.json

python scripts/precompute_vision.py \
  --manifest data/thought2text/test.jsonl \
  --image_root data/thought2text \
  --out data/thought2text/cache/clip_test.npy \
  --index_out data/thought2text/cache/clip_index_test.json
```

Requirements:

* Save fp16 embeddings if possible.
* Normalize embeddings.
* Save index mapping.
* Report missing images.

Output:

```text
outputs/overnight/clip_cache_report.md
```

---

# Stage 4: Implement Strong EEG Alignment Losses

Create or update:

```text
src/losses/
  __init__.py
  contrastive.py
  similarity.py
  eeg_aug.py
```

## 4.1 Symmetric InfoNCE

Use both directions:

```text
EEG → Image
Image → EEG
```

Given normalized EEG embedding `z_eeg` and image embedding `z_img`:

```python
logits = z_eeg @ z_img.T / temperature
labels = arange(batch_size)
loss = 0.5 * CE(logits, labels) + 0.5 * CE(logits.T, labels)
```

## 4.2 MSE Alignment

```python
loss_mse = mse(z_eeg, z_img)
```

## 4.3 Object Classification CE

If labels exist:

```python
loss_cls = cross_entropy(cls_head(eeg_feat), labels)
```

## 4.4 Similarity Distillation Loss

Preserve the image-space similarity structure inside EEG embeddings.

Compute pairwise similarity matrices:

```python
S_img = normalize(z_img) @ normalize(z_img).T
S_eeg = normalize(z_eeg) @ normalize(z_eeg).T
```

Then use KL or MSE on softened similarities:

```python
P_img = softmax(S_img / tau_sim, dim=-1)
P_eeg = log_softmax(S_eeg / tau_sim, dim=-1)
loss_sim = KLDiv(P_eeg, P_img)
```

Default:

```yaml
tau_sim: 0.1
lambda_sim: 0.2
```

This loss is important because EEG should preserve semantic neighborhood structure, not only match one paired image.

## 4.5 EEG Augmentation Consistency

Create two weak augmentations of EEG:

* small Gaussian noise
* random temporal crop or jitter
* channel dropout with small probability
* amplitude scaling

Encode both:

```python
z1 = encoder(aug1(eeg))
z2 = encoder(aug2(eeg))
loss_aug = 1 - cosine_similarity(z1, z2).mean()
```

Default:

```yaml
lambda_aug: 0.1
```

Keep augmentations mild. Do not destroy EEG signal.

---

# Stage 5: Overnight Alignment Training

Create config:

```text
configs/overnight_align.yaml
```

Suggested config:

```yaml
seed: 42

data:
  train_manifest: "data/thought2text/train.jsonl"
  val_manifest: "data/thought2text/val.jsonl"
  test_manifest: "data/thought2text/test.jsonl"
  root: "data/thought2text"
  clip_train_cache: "data/thought2text/cache/clip_train.npy"
  clip_val_cache: "data/thought2text/cache/clip_val.npy"
  clip_test_cache: "data/thought2text/cache/clip_test.npy"

model:
  eeg_embed_dim: 512
  clip_embed_dim: 512
  eeg_channels: 64
  eeg_time_steps: 250
  hidden_dim: 128
  transformer_layers: 2
  dropout: 0.1

loss:
  use_infonce: true
  use_mse: true
  use_cls: true
  use_similarity_distill: true
  use_aug_consistency: true
  temperature: 0.07
  tau_sim: 0.1
  lambda_mse: 0.5
  lambda_cls: 0.2
  lambda_sim: 0.2
  lambda_aug: 0.1

train:
  batch_size: 128
  grad_accum_steps: 1
  epochs: 80
  lr: 1.0e-4
  weight_decay: 0.01
  bf16: true
  num_workers: 4
  log_every: 20
  eval_every_epoch: true
  patience: 12
  save_best_by: "val_recall_at_5"

output:
  dir: "outputs/overnight/align_strong"
```

Run a short smoke test first:

```bash
python -m src.train.train_align \
  --config configs/overnight_align.yaml \
  --max_train_samples 512 \
  --max_val_samples 128 \
  --epochs 2 \
  --output_dir outputs/overnight/align_smoke
```

If smoke test passes, run full overnight:

```bash
python -m src.train.train_align \
  --config configs/overnight_align.yaml \
  --output_dir outputs/overnight/align_strong
```

---

# Stage 6: Loss Ablation Sweep

If time remains overnight, run 3 alignment variants.

## Variant A: Simple Thought2Text-style

```text
InfoNCE disabled
MSE + CE only
```

Output:

```text
outputs/overnight/align_mse_ce
```

## Variant B: Contrastive core

```text
InfoNCE + MSE + CE
```

Output:

```text
outputs/overnight/align_infonce_mse_ce
```

## Variant C: Strong full loss

```text
InfoNCE + MSE + CE + similarity distillation + augmentation consistency
```

Output:

```text
outputs/overnight/align_strong
```

Do not run all variants if time or GPU memory is limited.

Priority:

```text
C first, B second, A third
```

---

# Stage 7: Retrieval Evaluation

After each alignment run, evaluate:

```bash
python -m src.eval.retrieval \
  --manifest data/thought2text/test.jsonl \
  --clip_cache data/thought2text/cache/clip_test.npy \
  --eeg_ckpt outputs/overnight/align_strong/checkpoints/best.pt \
  --out outputs/overnight/align_strong/retrieval_metrics.json
```

Metrics:

```text
R@1
R@5
R@10
mean_rank
median_rank
random_R@1
random_R@5
random_R@10
```

Save:

```text
outputs/overnight/align_strong/retrieval_report.md
```

The report must compare model vs random.

---

# Stage 8: First Real Caption Fusion Run

Only run this after at least one alignment checkpoint exists.

Use Qwen2.5-1.5B first.

Do not use 7B overnight unless everything is already stable.

Pipeline:

```text
cached image CLIP embedding
+ aligned EEG encoder embedding
→ gated fusion
→ soft prompt projector
→ frozen Qwen2.5-1.5B
→ caption
```

Run:

```bash
python -m src.train.train_fusion \
  --train_manifest data/thought2text/train.jsonl \
  --val_manifest data/thought2text/val.jsonl \
  --root data/thought2text \
  --clip_train_cache data/thought2text/cache/clip_train.npy \
  --clip_val_cache data/thought2text/cache/clip_val.npy \
  --eeg_ckpt outputs/overnight/align_strong/checkpoints/best.pt \
  --llm Qwen/Qwen2.5-1.5B-Instruct \
  --freeze_llm true \
  --freeze_eeg_encoder true \
  --epochs 5 \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --bf16 true \
  --output_dir outputs/overnight/fusion_qwen15
```

Caption fusion loss should be:

```text
L_caption_total =
  L_CE_caption
+ 0.1 * L_fused_text_contrast
+ 0.05 * L_gate_regularization
```

## Fused-text contrast

Use CLIP text embedding or sentence embedding if already available.

If not available, skip this term.

The goal:

```text
fused representation should be closer to its own caption than to captions from other samples
```

Use batch-level InfoNCE.

## Gate regularization

Prevent EEG gate from saturating:

```python
loss_gate = mean((gate - 0.5) ** 2)
```

This should be weak.

Do not force gate too strongly.

---

# Stage 9: Mini Sanity Check

Run generation on a small test subset:

```bash
python -m src.eval.sanity_check \
  --manifest data/thought2text/test.jsonl \
  --max_samples 128 \
  --caption_ckpt outputs/overnight/fusion_qwen15/checkpoints/best.pt \
  --eeg_ckpt outputs/overnight/align_strong/checkpoints/best.pt \
  --modes vision_only real_eeg shuffled_eeg random_eeg \
  --corruptions clean blur occlusion \
  --out outputs/overnight/sanity_mini
```

Save one JSONL per mode and corruption.

Then evaluate:

```bash
python -m src.eval.metrics \
  --pred_dir outputs/overnight/sanity_mini \
  --out outputs/overnight/sanity_mini/metrics.md
```

Do not overclaim.

This is only a first sanity check.

---

# Stage 10: Overnight Report

At the end, create:

```text
outputs/overnight/OVERNIGHT_REPORT.md
```

It must include:

1. Start time and end time.
2. GPU info.
3. Dataset status.
4. Manifest statistics.
5. CLIP cache statistics.
6. Alignment loss curves summary.
7. Retrieval metrics for each loss variant.
8. Best checkpoint path.
9. Fusion caption training status.
10. Mini sanity-check results.
11. 10 sample predictions.
12. Known problems.
13. Recommended next commands.

Use this exact table for alignment:

```text
| Run | Loss | R@1 | R@5 | R@10 | Mean Rank | Random R@5 | Notes |
```

Use this exact table for caption sanity:

```text
| Corruption | Mode | BLEU-1 | ROUGE-L | Avg Len | Distinct Ratio | Notes |
```

Important conclusion rule:

Allowed:

```text
Preliminary results show whether EEG embeddings align with visual CLIP space and whether real EEG behaves differently from shuffled/random EEG under degraded visual inputs.
```

Forbidden:

```text
The model reads thoughts.
```

---

# Suggested Background Run Command

Create:

```bash
scripts/run_overnight.sh
```

It should run the stages in order and tee logs:

```bash
mkdir -p outputs/overnight/logs

{
  date
  nvidia-smi

  bash scripts/inspect_thought2text.sh
  bash scripts/build_thought2text_manifest.sh

  python scripts/precompute_vision.py \
    --manifest data/thought2text/train.jsonl \
    --image_root data/thought2text \
    --out data/thought2text/cache/clip_train.npy \
    --index_out data/thought2text/cache/clip_index_train.json

  python scripts/precompute_vision.py \
    --manifest data/thought2text/val.jsonl \
    --image_root data/thought2text \
    --out data/thought2text/cache/clip_val.npy \
    --index_out data/thought2text/cache/clip_index_val.json

  python scripts/precompute_vision.py \
    --manifest data/thought2text/test.jsonl \
    --image_root data/thought2text \
    --out data/thought2text/cache/clip_test.npy \
    --index_out data/thought2text/cache/clip_index_test.json

  python -m src.train.train_align \
    --config configs/overnight_align.yaml \
    --max_train_samples 512 \
    --max_val_samples 128 \
    --epochs 2 \
    --output_dir outputs/overnight/align_smoke

  python -m src.train.train_align \
    --config configs/overnight_align.yaml \
    --output_dir outputs/overnight/align_strong

  python -m src.eval.retrieval \
    --manifest data/thought2text/test.jsonl \
    --clip_cache data/thought2text/cache/clip_test.npy \
    --eeg_ckpt outputs/overnight/align_strong/checkpoints/best.pt \
    --out outputs/overnight/align_strong/retrieval_metrics.json

  python -m src.train.train_fusion \
    --train_manifest data/thought2text/train.jsonl \
    --val_manifest data/thought2text/val.jsonl \
    --root data/thought2text \
    --clip_train_cache data/thought2text/cache/clip_train.npy \
    --clip_val_cache data/thought2text/cache/clip_val.npy \
    --eeg_ckpt outputs/overnight/align_strong/checkpoints/best.pt \
    --llm Qwen/Qwen2.5-1.5B-Instruct \
    --freeze_llm true \
    --freeze_eeg_encoder true \
    --epochs 5 \
    --batch_size 4 \
    --grad_accum_steps 8 \
    --bf16 true \
    --output_dir outputs/overnight/fusion_qwen15

  python -m src.eval.sanity_check \
    --manifest data/thought2text/test.jsonl \
    --max_samples 128 \
    --caption_ckpt outputs/overnight/fusion_qwen15/checkpoints/best.pt \
    --eeg_ckpt outputs/overnight/align_strong/checkpoints/best.pt \
    --modes vision_only real_eeg shuffled_eeg random_eeg \
    --corruptions clean blur occlusion \
    --out outputs/overnight/sanity_mini

  python -m src.eval.metrics \
    --pred_dir outputs/overnight/sanity_mini \
    --out outputs/overnight/sanity_mini/metrics.md

  python scripts/make_overnight_report.py \
    --root outputs/overnight \
    --out outputs/overnight/OVERNIGHT_REPORT.md

  date
} 2>&1 | tee outputs/overnight/logs/overnight.log
```

Run with:

```bash
tmux new -s eegcap
bash scripts/run_overnight.sh
```

or:

```bash
nohup bash scripts/run_overnight.sh > outputs/overnight/nohup.log 2>&1 &
```

---

# If Something Fails

Do not stop without writing a report.

If real data missing:

```text
write outputs/overnight/missing_data_instructions.md
```

If CLIP cache fails:

```text
write outputs/overnight/clip_cache_error.md
```

If alignment gets NaN:

1. lower lr to 3e-5,
2. disable similarity distillation,
3. disable augmentation consistency,
4. reduce batch size,
5. restart from smoke config.

If caption fusion OOM:

1. reduce batch size to 1,
2. increase grad accumulation,
3. use Qwen2.5-0.5B or a tiny LM for smoke,
4. keep alignment result as the main overnight deliverable.

---

# Completion Criteria

The overnight goal is complete if these exist:

```text
outputs/overnight/OVERNIGHT_REPORT.md
outputs/overnight/thought2text_inspection.md
data/thought2text/train.jsonl
data/thought2text/cache/clip_train.npy
outputs/overnight/align_strong/checkpoints/best.pt
outputs/overnight/align_strong/retrieval_metrics.json
```

Fusion outputs are desirable but secondary.

Priority is:

```text
real data > CLIP cache > EEG alignment > retrieval metrics > caption fusion > degraded sanity check
```
