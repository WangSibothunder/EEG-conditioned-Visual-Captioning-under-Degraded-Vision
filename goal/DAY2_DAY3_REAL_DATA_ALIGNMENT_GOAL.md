# Goal: Day2–Day3 Real Data + EEG Alignment + Caption Fusion

This goal should be executed only after Day1 dummy MVP has finished.

Day1 was only a smoke test.
Day2–Day3 should move the project from toy pipeline to real EEG research prototype.

Core project:

```text
Degraded Image + EEG → Caption
```

Core scientific question:

```text
Can correctly paired EEG help caption generation under degraded visual input more than shuffled or random EEG?
```

Before caption training, we must first prove:

```text
EEG embeddings align with visual CLIP embeddings better than random.
```

Therefore, Day2 priority is not LLM caption quality.
Day2 priority is real data audit, clean split, CLIP cache, and EEG→CLIP alignment.

---

# Global Constraints

Follow these strictly:

* Single GPU only.
* Assume around 48GB VRAM.
* Do not use multi-GPU.
* Do not use DeepSpeed/FSDP unless explicitly requested.
* Do not full-finetune LLM.
* Freeze CLIP.
* Freeze LLM by default.
* Use Qwen2.5-1.5B for first fusion run, not 7B.
* Do not continue optimizing dummy data after Day1.
* Do not claim EEG helps without real/shuffled/random EEG controls.
* Do not allow image leakage across train/val/test.

---

# Required Outputs by End of This Goal

The goal is successful only if the following files exist:

```text
outputs/day2/thought2text_inspection.md
outputs/day2/split_leakage_report.md
outputs/day2/clip_cache_report.md

data/thought2text/train.jsonl
data/thought2text/val.jsonl
data/thought2text/test.jsonl

data/thought2text/cache/clip_train.npy
data/thought2text/cache/clip_val.npy
data/thought2text/cache/clip_test.npy

outputs/day2_align/alignment_metrics.json
outputs/day2_align/alignment_report.md
outputs/day2_align/checkpoints/best.pt

outputs/day3/fusion_report.md
outputs/day3/sanity_real/metrics.md
outputs/day3/sanity_real/sample_predictions.jsonl
outputs/day3/DAY3_REPORT.md
```

If Thought2Text data is missing, create:

```text
outputs/day2/missing_data_instructions.md
```

and stop real-data training gracefully.

Do not silently fall back to dummy data for real experiments.

---

# Expected Thought2Text Data Layout

The expected layout is:

```text
data/thought2text/
  block/
    eeg_55_95_std.pth
    block_splits_by_image_all.pth
  images/
```

If the real file names differ, inspect robustly and document the actual structure.

---

# Day2 Morning: Data Audit + Manifest + Split Check + CLIP Cache

## Day2 Morning Goals

Do:

```text
1. Thought2Text data audit
2. manifest construction
3. image-level split leakage check
4. CLIP image feature cache
```

Acceptance:

```text
thought2text_inspection.md
split_leakage_report.md
clip_cache_report.md
```

---

## Task 1: Inspect Thought2Text Data

Create or update:

```text
src/data/inspect_thought2text.py
scripts/inspect_thought2text.sh
```

Run:

```bash
bash scripts/inspect_thought2text.sh
```

The script must inspect:

```text
data/thought2text/block/eeg_55_95_std.pth
data/thought2text/block/block_splits_by_image_all.pth
data/thought2text/images/
```

The report must be saved to:

```text
outputs/day2/thought2text_inspection.md
```

The report must include:

```text
1. Keys inside eeg_55_95_std.pth
2. Keys inside block_splits_by_image_all.pth
3. Number of EEG trials
4. Number of unique images
5. Number of classes
6. Number of subjects if available
7. EEG tensor shape
8. Label shape
9. Split information
10. Number of image files found
11. Number of missing image files
12. Five example samples with image_id, label, eeg_index, image path
```

Important:

Do not assume the `.pth` schema.
Print keys and infer structure carefully.

---

## Task 2: Build Thought2Text Manifest

Create or update:

```text
src/data/build_thought2text_manifest.py
scripts/build_thought2text_manifest.sh
```

Run:

```bash
bash scripts/build_thought2text_manifest.sh
```

Output:

```text
data/thought2text/train.jsonl
data/thought2text/val.jsonl
data/thought2text/test.jsonl
```

Each JSONL line should follow this schema:

```json
{
  "image_id": "...",
  "image_path": "...",
  "eeg_index": 123,
  "caption": "a photo of a {class_name}",
  "label": 5,
  "subject_id": "S01",
  "split": "train"
}
```

If class names are available, use:

```text
"a photo of a {class_name}"
```

If class names are not available, use:

```text
"a photo of an object from class {label}"
```

This is acceptable for the first real-data MVP.

---

## Task 3: Enforce Image-Level Split

This is mandatory.

The same `image_id` must not appear in more than one of:

```text
train
val
test
```

Do not split randomly by EEG trial.

Why:

```text
The same image may have EEG trials from multiple subjects.
If the same image appears in both train and test, results are invalid.
```

Create:

```text
src/data/check_split_leakage.py
scripts/check_split_leakage.sh
```

Run:

```bash
bash scripts/check_split_leakage.sh
```

Save:

```text
outputs/day2/split_leakage_report.md
```

The report must include:

```text
1. Unique image count in train/val/test
2. EEG trial count in train/val/test
3. Number of overlapping image_ids between train and val
4. Number of overlapping image_ids between train and test
5. Number of overlapping image_ids between val and test
6. Whether leakage exists
```

If leakage exists:

```text
Do not train.
Rebuild manifest using image-level split.
```

---

## Task 4: Dataset Loader Smoke Test

Update dataset loader so that it supports:

```text
1. dummy JSONL mode
2. Thought2Text JSONL mode with eeg_index into .pth
3. image-only mode
4. EEG-only mode
```

Required item format:

```python
{
    "image": FloatTensor[3, 224, 224],
    "eeg": FloatTensor[C, T],
    "caption": str,
    "image_id": str,
    "label": int,
    "subject_id": str | None,
}
```

Run:

```bash
python -m src.data.dataset \
  --manifest data/thought2text/train.jsonl \
  --root data/thought2text \
  --smoke_test
```

The smoke test must print:

```text
image shape
eeg shape
caption example
image_id example
label example
batch shape
```

---

## Task 5: Precompute CLIP Image Features

Create or update:

```text
scripts/precompute_vision.py
src/data/clip_cache.py
```

Use first:

```text
openai/clip-vit-base-patch32
```

Run:

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

```text
1. CLIP embeddings must be normalized.
2. Save as fp16 if possible.
3. Save index mapping from image_id/eeg_index to cache row.
4. Report missing images.
5. Do not repeatedly recompute if cache already exists unless --overwrite is set.
```

Save report:

```text
outputs/day2/clip_cache_report.md
```

The report must include:

```text
1. CLIP model name
2. Number of processed samples
3. Number of unique images
4. Embedding shape
5. dtype
6. cache file size
7. number of missing images
8. output paths
```

---

# Day2 Afternoon / Night: EEG → CLIP Alignment

## Day2 Afternoon / Night Goals

Do:

```text
EEG → CLIP alignment
```

First run small scale:

```text
512 train / 128 val / 2 epochs
```

Then run formal training:

```text
all train / early stopping / 3 seeds if time
```

Acceptance:

```text
alignment_metrics.json
alignment_report.md
best.pt
```

---

## Task 6: Implement Alignment Model

Create or update:

```text
src/models/alignment_model.py
src/train/train_align.py
src/eval/retrieval.py
scripts/run_align_smoke.sh
scripts/run_align.sh
```

Pipeline:

```text
EEG → EEG Encoder → Projector → CLIP image space
```

Input:

```text
EEG tensor
label
cached CLIP image embedding
optional subject_id
```

Output:

```text
z_eeg: normalized EEG embedding
z_img: normalized CLIP image embedding
class logits if labels exist
```

Default dimensions:

```text
CLIP image dim = 512
EEG embedding dim = 512
```

---

## Task 7: Use Small-Sample-Aware Alignment Loss

Do not use only MSE.

Use this total loss:

```text
L_total =
  1.0 * L_multi_positive_InfoNCE
+ 0.5 * L_cosine_or_MSE
+ 0.3 * L_class_CE
+ 0.2 * L_similarity_distillation
+ 0.1 * L_EEG_augmentation_consistency
+ 0.2 * L_prototype_alignment
```

If some required information is unavailable, skip only that term and report it.

### 7.1 Multi-Positive InfoNCE

Use same image as strong positive.

If labels are available, same-class samples can be weak positives.

Reason:

```text
This is a small EEG dataset. EEG may capture class-level semantic information more reliably than fine-grained instance identity.
```

### 7.2 Cosine or MSE Alignment

Use cosine loss first:

```text
L_cos = 1 - cosine(z_eeg, z_img)
```

MSE can be optional.

### 7.3 Class CE

If labels exist:

```text
EEG feature → classifier → class CE
```

This stabilizes category-level EEG semantics.

### 7.4 Similarity Distillation

Preserve CLIP image-space neighborhood structure:

```python
S_img = normalize(z_img) @ normalize(z_img).T
S_eeg = normalize(z_eeg) @ normalize(z_eeg).T
L_sim = KLDiv(
    log_softmax(S_eeg / tau_sim),
    softmax(S_img / tau_sim)
)
```

Default:

```yaml
tau_sim: 0.1
```

### 7.5 EEG Augmentation Consistency

Use two weak augmentations of the same EEG:

```text
small Gaussian noise
mild channel dropout
mild temporal jitter
amplitude scaling
```

Do not destroy EEG signal.

Loss:

```text
L_aug = 1 - cosine(z_aug1, z_aug2)
```

### 7.6 Prototype Alignment

Compute class-level prototypes if labels exist:

```text
image class prototype = mean CLIP image embedding for class c
optional text prototype = CLIP text embedding of "a photo of a {class_name}"
```

Encourage EEG embedding to approach its class prototype.

This is important for small-sample learning.

---

## Task 8: Create Alignment Config

Create:

```text
configs/day2_align.yaml
```

Recommended config:

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
  eeg_channels: 64
  eeg_time_steps: 250
  hidden_dim: 128
  transformer_layers: 2
  dropout: 0.1
  eeg_embed_dim: 512
  clip_embed_dim: 512

loss:
  use_multi_positive_infonce: true
  use_cosine: true
  use_mse: false
  use_class_ce: true
  use_similarity_distillation: true
  use_aug_consistency: true
  use_prototype_alignment: true

  temperature: 0.07
  tau_sim: 0.1

  lambda_infonce: 1.0
  lambda_cosine: 0.5
  lambda_mse: 0.0
  lambda_class_ce: 0.3
  lambda_similarity: 0.2
  lambda_aug: 0.1
  lambda_proto: 0.2

train:
  batch_size: 128
  grad_accum_steps: 1
  epochs: 80
  lr: 1.0e-4
  weight_decay: 0.05
  bf16: true
  num_workers: 4
  log_every: 20
  eval_every_epoch: true
  patience: 12
  save_best_by: "val_recall_at_5"

output:
  dir: "outputs/day2_align"
```

If actual EEG shape is not `[64, 250]`, automatically infer shape from data inspection and update config/report.

---

## Task 9: Run Alignment Smoke Test

First run:

```bash
python -m src.train.train_align \
  --config configs/day2_align.yaml \
  --max_train_samples 512 \
  --max_val_samples 128 \
  --epochs 2 \
  --output_dir outputs/day2_align_smoke
```

Acceptance:

```text
1. no crash
2. loss is finite
3. validation retrieval runs
4. checkpoint saved
5. metrics JSON saved
```

If NaN occurs:

```text
1. lower lr to 3e-5
2. disable similarity distillation
3. disable augmentation consistency
4. reduce batch size to 64
5. rerun smoke
```

---

## Task 10: Run Formal Alignment Training

After smoke test passes, run formal training:

```bash
python -m src.train.train_align \
  --config configs/day2_align.yaml \
  --output_dir outputs/day2_align
```

Use early stopping.

Save best checkpoint by:

```text
val R@5
```

Required output:

```text
outputs/day2_align/checkpoints/best.pt
outputs/day2_align/alignment_metrics.json
outputs/day2_align/alignment_report.md
```

The report must include:

```text
1. train/val/test sample count
2. unique image count
3. EEG shape
4. loss terms used
5. best epoch
6. train loss summary
7. val R@1 / R@5 / R@10
8. test R@1 / R@5 / R@10
9. mean rank
10. median rank
11. random baseline R@1 / R@5 / R@10
12. class accuracy if available
13. best checkpoint path
```

---

## Task 11: Run 3 Seeds If Time Allows

If the first formal run finishes and time remains, run:

```text
seed = 42
seed = 123
seed = 2025
```

Output:

```text
outputs/day2_align_seed42/
outputs/day2_align_seed123/
outputs/day2_align_seed2025/
```

Then create:

```text
outputs/day2_align/multiseed_summary.md
```

The table:

```text
| Seed | R@1 | R@5 | R@10 | Class Acc | Best Epoch |
```

If time is limited, seed 42 alone is acceptable.

---

# Day3: Caption Fusion + Degraded Vision Sanity Check

Day3 should be run only after at least one alignment checkpoint exists.

## Day3 Goals

Do:

```text
1. small-scale caption fusion training
2. degraded image feature preparation
3. real/shuffled/random EEG sanity check
4. metric table and qualitative examples
```

Acceptance:

```text
fusion_report.md
sanity_real/metrics.md
sample_predictions.jsonl
DAY3_REPORT.md
```

---

## Task 12: Train Caption Fusion Model

Use aligned EEG encoder checkpoint:

```text
outputs/day2_align/checkpoints/best.pt
```

Pipeline:

```text
image CLIP embedding
+ aligned EEG embedding
→ gated fusion
→ soft prompt projector
→ frozen Qwen2.5-1.5B
→ caption
```

Use:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Do not use 7B yet.

Freeze:

```text
CLIP
LLM
EEG encoder at first
```

Train:

```text
fusion module
soft prompt projector
optional small EEG adapter
```

Run:

```bash
python -m src.train.train_fusion \
  --train_manifest data/thought2text/train.jsonl \
  --val_manifest data/thought2text/val.jsonl \
  --root data/thought2text \
  --clip_train_cache data/thought2text/cache/clip_train.npy \
  --clip_val_cache data/thought2text/cache/clip_val.npy \
  --eeg_ckpt outputs/day2_align/checkpoints/best.pt \
  --llm Qwen/Qwen2.5-1.5B-Instruct \
  --freeze_llm true \
  --freeze_eeg_encoder true \
  --epochs 5 \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --bf16 true \
  --output_dir outputs/day3/fusion_qwen15
```

Caption loss:

```text
L_caption_total =
  L_caption_CE
+ 0.1 * L_fused_caption_contrast_if_available
+ 0.05 * L_gate_regularization
```

Gate regularization should be weak.

Do not force gate to a fixed value too strongly.

Save:

```text
outputs/day3/fusion_report.md
outputs/day3/fusion_qwen15/checkpoints/best.pt
```

The fusion report must include:

```text
1. LLM name
2. trainable parameter count
3. frozen parameter count
4. epoch loss
5. validation loss
6. 10 generated examples
7. average gate value if available
8. checkpoint path
```

---

## Task 13: Prepare Degraded Vision Features

Create or update:

```text
src/data/corruptions.py
scripts/precompute_degraded_vision.py
```

Support corruptions:

```text
clean
blur
occlusion
noise
lowres
```

Because Thought2Text is small, it is acceptable to precompute separate CLIP caches:

```text
data/thought2text/cache/clip_test_clean.npy
data/thought2text/cache/clip_test_blur.npy
data/thought2text/cache/clip_test_occlusion.npy
data/thought2text/cache/clip_test_noise.npy
data/thought2text/cache/clip_test_lowres.npy
```

Run:

```bash
python scripts/precompute_degraded_vision.py \
  --manifest data/thought2text/test.jsonl \
  --image_root data/thought2text \
  --corruptions clean blur occlusion noise lowres \
  --out_dir data/thought2text/cache/degraded_test
```

Save report:

```text
outputs/day3/degraded_clip_cache_report.md
```

---

## Task 14: Run Sanity Check

Run:

```bash
python -m src.eval.sanity_check \
  --manifest data/thought2text/test.jsonl \
  --max_samples 256 \
  --caption_ckpt outputs/day3/fusion_qwen15/checkpoints/best.pt \
  --eeg_ckpt outputs/day2_align/checkpoints/best.pt \
  --modes vision_only real_eeg shuffled_eeg random_eeg eeg_only \
  --corruptions clean blur occlusion noise lowres \
  --out outputs/day3/sanity_real
```

Each output line:

```json
{
  "image_id": "...",
  "corruption": "blur",
  "mode": "real_eeg",
  "reference": "a photo of a dog",
  "prediction": "a photo of a dog",
  "label": 3,
  "gate_mean": 0.42
}
```

Required modes:

```text
vision_only
real_eeg
shuffled_eeg
random_eeg
eeg_only
```

Important:

Do not claim EEG helps unless real EEG is compared with shuffled and random EEG.

---

## Task 15: Compute Caption Metrics and Gate Analysis

Create or update:

```text
src/eval/metrics.py
src/eval/gate_analysis.py
scripts/evaluate_day3.sh
```

Run:

```bash
bash scripts/evaluate_day3.sh
```

Metrics:

```text
BLEU-1
BLEU-4
ROUGE-L
average prediction length
distinct prediction ratio
optional BERTScore
```

Gate analysis:

```text
average gate value for clean
average gate value for blur
average gate value for occlusion
average gate value for noise
average gate value for lowres
```

Save:

```text
outputs/day3/sanity_real/metrics.md
outputs/day3/sanity_real/metrics.csv
outputs/day3/sanity_real/gate_analysis.md
outputs/day3/sanity_real/sample_predictions.jsonl
```

Metric table format:

```text
| Corruption | Mode | BLEU-1 | BLEU-4 | ROUGE-L | Avg Len | Distinct Ratio | Gate Mean |
```

---

## Task 16: Create Day3 Report

Create:

```text
outputs/day3/DAY3_REPORT.md
```

It must include:

```text
1. Dataset summary
2. Alignment checkpoint used
3. Fusion model setting
4. Caption metric table
5. Gate analysis
6. 10 qualitative examples
7. Whether real EEG differs from shuffled/random EEG
8. Current limitations
9. Recommended next steps
```

Conclusion must be cautious.

Allowed:

```text
Preliminary results indicate whether correctly paired EEG provides auxiliary information under degraded visual inputs.
```

Forbidden:

```text
The model reads thoughts.
```

---

# Optional: Prepare Additional Dataset Scripts If Extra Time Remains

Do not block Day2/Day3 core tasks.

If core tasks finish early, create download/preparation scripts for:

```text
THINGS-EEG2
EIT-1M
```

Create:

```text
docs/DATASETS.md
scripts/download_things_eeg2.sh
scripts/download_eit1m.sh
```

Do not start a full migration unless explicitly requested.

## THINGS-EEG2

Use for future stronger EEG encoder pretraining.

The script should include:

```bash
pip install osfclient
osf -p 3jk45 clone THINGS-EEG2
```

## EIT-1M

Use for future EEG-image-text tri-modal training if available.

The script should include Hugging Face download logic:

```bash
huggingface-cli download eit-1m/EIT-1M \
  --repo-type dataset \
  --local-dir data/EIT-1M
```

If credentials or access are missing, print clear instructions and do not crash the main pipeline.

---

# Priority Order If Time Is Limited

Follow this exact priority:

```text
1. Thought2Text inspection
2. manifest construction
3. image-level split leakage check
4. CLIP cache
5. alignment smoke run
6. full alignment run
7. retrieval metrics
8. caption fusion
9. degraded vision sanity check
10. additional dataset scripts
```

If time is short, do not sacrifice alignment metrics for caption training.

Alignment is the key deliverable.

---

# Final Completion Criteria

The goal is complete if:

```text
1. real Thought2Text data is inspected
2. image-level split is verified leak-free
3. CLIP cache exists
4. EEG→CLIP alignment trains successfully
5. retrieval metrics beat random baseline
6. best alignment checkpoint exists
7. caption fusion runs at least once
8. real/shuffled/random EEG sanity check runs
9. reports are saved
```

If retrieval metrics do not beat random, report honestly and do not proceed to strong EEG claims.

The correct scientific attitude is:

```text
No EEG claim without alignment evidence and shuffled/random controls.
```
