# Goal: Day4–Day5 Heavy Research Run

The previous Day3 run successfully produced a real-data prototype, but the results are still preliminary.

Current status:

* Thought2Text data is loaded.
* Image-level split leakage is false.
* CLIP degraded caches exist.
* EEG→CLIP alignment beats random but is weak.
* Caption fusion runs, but generated text quality is poor and code-like.
* Real EEG does not consistently beat shuffled EEG.
* Gate values are small and similar across real/shuffled/random EEG.

Therefore, the next two days must focus on scientific validity and stronger results, not just runnable code.

Main objective:

```text
Turn the current prototype into a credible small-sample EEG + Vision captioning experiment.
```

Core hypothesis:

```text
Correctly paired EEG should provide auxiliary semantic information under degraded visual inputs,
and this should be visible through:
1. stronger EEG→CLIP alignment,
2. real EEG > shuffled/random EEG under degraded conditions,
3. gate values or fusion sensitivity increasing under degradation,
4. better class/caption consistency.
```

---

# Hard Rules

Follow these strictly:

* Do not full-finetune the LLM.
* Do not use 7B LLM before fixing 1.5B generation quality.
* Do not claim EEG helps unless real EEG beats shuffled/random across seeds or conditions.
* Do not rely on caption loss alone.
* Do not continue optimizing dummy data.
* Do not overwrite Day3 outputs.
* Every major run must produce a report.
* Use GPU heavily when training, but do not hide failed runs.
* If a job fails, write an error report and continue with the next safe job.

---

# Required Final Outputs

By the end of this goal, produce:

```text
outputs/day4_alignment/ALIGNMENT_ABLATION_REPORT.md
outputs/day4_alignment/multiseed_summary.md
outputs/day4_alignment/best_overall.pt

outputs/day4_caption_targets/caption_target_report.md
outputs/day4_caption_targets/imagenet_classname_manifest_report.md

outputs/day5_fusion/FUSION_COMPARISON_REPORT.md
outputs/day5_sanity/FULL_SANITY_METRICS.md
outputs/day5_sanity/FULL_SANITY_METRICS.csv
outputs/day5_sanity/gate_analysis.md
outputs/day5_sanity/sample_predictions.jsonl

outputs/day5_final/NEXT_48H_RESEARCH_REPORT.md
```

If additional datasets are available:

```text
outputs/day5_datasets/things_eeg2_status.md
outputs/day5_datasets/eit1m_status.md
```

---

# Phase 0: Audit Current Results and Identify Failure Modes

Create:

```text
outputs/day4_audit/CURRENT_RESULT_AUDIT.md
```

It must summarize:

1. Dataset size:

   * EEG trials
   * unique images
   * classes
   * split sizes
2. Current alignment metrics:

   * R@1
   * R@5
   * R@10
   * random baseline
3. Current fusion metrics:

   * validation loss
   * train loss
   * sample predictions
4. Current gate values.
5. Current failure modes.

Must explicitly state:

```text
Current caption generation quality is weak and code-like.
Current EEG benefit is preliminary because real EEG does not consistently beat shuffled EEG.
Current gate behavior does not yet prove selective EEG usage.
```

Also check if retrieval evaluation has a bug:

```text
R@5 should normally be >= R@1.
If R@1 and R@5 are identical, verify top-k retrieval code and duplicate image handling.
```

Output:

```text
outputs/day4_audit/retrieval_eval_debug.md
```

---

# Phase 1: Fix Caption Targets

The previous captions are too weak:

```text
"a photo of an object from class n03452741"
```

This causes unnatural output and code-like class-token generation.

We need better text targets.

## Task 1.1: Map WordNet IDs to human-readable class names

Implement:

```text
src/data/imagenet_labels.py
scripts/build_human_caption_manifest.py
```

Input:

```text
data/thought2text/train.jsonl
data/thought2text/val.jsonl
data/thought2text/test.jsonl
```

Output:

```text
data/thought2text/train_human_caption.jsonl
data/thought2text/val_human_caption.jsonl
data/thought2text/test_human_caption.jsonl
```

Convert:

```text
n03452741
```

to a human-readable class name if possible.

Caption format:

```text
"a photo of a {human_class_name}"
```

If WordNet lookup fails, keep the original wnid but mark it in report.

Create:

```text
outputs/day4_caption_targets/imagenet_classname_manifest_report.md
```

Report:

1. Number of captions converted.
2. Number of unknown wnids.
3. 20 before/after examples.

---

## Task 1.2: Generate optional BLIP captions

If `Salesforce/blip-image-captioning-base` is available, generate image captions for unique images.

Create:

```text
scripts/generate_blip_captions.py
```

Output:

```text
data/thought2text/blip_captions.jsonl
data/thought2text/train_blip_caption.jsonl
data/thought2text/val_blip_caption.jsonl
data/thought2text/test_blip_caption.jsonl
```

Rules:

* Generate once per unique image.
* Cache results.
* Do not overwrite existing captions unless `--overwrite`.
* Keep captions short.
* If BLIP fails, continue with human class-name captions.

Report:

```text
outputs/day4_caption_targets/blip_caption_report.md
```

---

## Task 1.3: Build three caption target variants

Create three manifest variants:

```text
1. class_wnid_caption
2. human_class_caption
3. blip_caption
```

Write:

```text
outputs/day4_caption_targets/caption_target_report.md
```

Compare 20 examples.

Priority for later fusion training:

```text
blip_caption if available
human_class_caption otherwise
wnid_caption only as fallback
```

---

# Phase 2: Stronger EEG→CLIP Alignment

The current alignment beats random but remains weak.

We now need a heavier alignment run with ablations and multiple seeds.

## Task 2.1: Debug retrieval evaluation

Before new training, verify retrieval metrics.

Check:

1. Are embeddings normalized?
2. Does each EEG sample retrieve among all test images or all test trials?
3. Are duplicate images handled correctly?
4. Is top-k computed correctly?
5. Is random baseline computed over the same candidate set?

Write:

```text
outputs/day4_alignment/retrieval_debug_report.md
```

If duplicate image trials exist, report both:

```text
trial-level retrieval
unique-image-level retrieval
```

---

## Task 2.2: Implement / verify stronger losses

Use:

```text
L_total =
  1.0 * Multi-positive InfoNCE
+ 0.5 * Cosine alignment
+ 0.3 * Class CE
+ 0.2 * Similarity Distillation
+ 0.1 * EEG Augmentation Consistency
+ 0.2 * Prototype Alignment
```

Required loss variants:

```text
A: MSE + Class CE
B: InfoNCE + Cosine + Class CE
C: InfoNCE + Cosine + Class CE + Similarity Distillation
D: InfoNCE + Cosine + Class CE + Similarity Distillation + Aug Consistency + Prototype Alignment
```

Save config files:

```text
configs/day4_align_A_mse_ce.yaml
configs/day4_align_B_contrastive.yaml
configs/day4_align_C_simdistill.yaml
configs/day4_align_D_full.yaml
```

---

## Task 2.3: Run alignment ablation

Run each variant with seed 42 first.

Output directories:

```text
outputs/day4_alignment/A_mse_ce_seed42
outputs/day4_alignment/B_contrastive_seed42
outputs/day4_alignment/C_simdistill_seed42
outputs/day4_alignment/D_full_seed42
```

Each run should:

* use early stopping,
* save best checkpoint by val R@5,
* log loss terms separately,
* evaluate on test set.

Metrics required:

```text
R@1
R@5
R@10
mean rank
median rank
class accuracy
random R@1/R@5/R@10
```

Create:

```text
outputs/day4_alignment/ALIGNMENT_ABLATION_REPORT.md
```

Table format:

```text
| Variant | Loss Terms | R@1 | R@5 | R@10 | Class Acc | Mean Rank | Random R@5 | Best Epoch |
```

---

## Task 2.4: Multi-seed run for best variant

Select the best variant by validation R@5.

Run seeds:

```text
42
123
2025
```

Output:

```text
outputs/day4_alignment/best_seed42
outputs/day4_alignment/best_seed123
outputs/day4_alignment/best_seed2025
```

Create:

```text
outputs/day4_alignment/multiseed_summary.md
```

Table:

```text
| Seed | R@1 | R@5 | R@10 | Class Acc | Mean Rank | Best Epoch |
```

Also compute:

```text
mean ± std
```

Save best overall checkpoint:

```text
outputs/day4_alignment/best_overall.pt
```

If time is limited, run at least two seeds.

---

# Phase 3: Train Better Fusion Models

The previous fusion model overfits and produces code-like captions.

Now train fusion with improved caption targets and better controls.

## Task 3.1: Implement fusion variants

Train these variants:

```text
F0: vision_only
F1: image + aligned real EEG gated fusion
F2: image + frozen random EEG encoder control
F3: image + shuffled EEG training control
```

For the main model:

```text
image embedding + EEG embedding → gated fusion → soft prompt → frozen Qwen2.5-1.5B
```

Freeze:

```text
CLIP
LLM
EEG encoder initially
```

Train:

```text
fusion
soft prompt projector
optional tiny EEG adapter
```

Use improved manifests:

```text
human_class_caption or blip_caption
```

Do not use wnid-only caption unless all else fails.

---

## Task 3.2: Fusion loss

Use:

```text
L_total =
  L_caption_CE
+ 0.1 * L_fused_caption_contrast
+ 0.05 * L_gate_entropy_or_sparsity_regularization
+ 0.05 * L_real_vs_shuffled_margin
```

If CLIP text embeddings are available, implement fused-caption contrast:

```text
fused_emb should match its own caption text embedding better than other captions in the batch
```

For real-vs-shuffled margin:

During training or evaluation, compare real EEG fusion score and shuffled EEG fusion score.

Encourage:

```text
score(real EEG) > score(shuffled EEG) + margin
```

Use a small margin.

If unstable, disable this term and report.

---

## Task 3.3: Train fusion comparison

Run:

```text
outputs/day5_fusion/F0_vision_only
outputs/day5_fusion/F1_real_eeg
outputs/day5_fusion/F2_random_encoder_control
outputs/day5_fusion/F3_shuffled_training_control
```

Suggested settings:

```text
epochs: 10
batch_size: 4
grad_accum_steps: 8
bf16: true
early_stopping: true
```

Use Qwen2.5-1.5B only.

Create:

```text
outputs/day5_fusion/FUSION_COMPARISON_REPORT.md
```

Report:

1. caption target type used,
2. trainable parameter count,
3. best val loss,
4. generated examples,
5. average gate value,
6. overfitting signs,
7. whether text quality improved.

---

# Phase 4: Full Degraded Sanity Evaluation

The previous run used 256 samples and mixed results.

Now run full test evaluation.

Use all test samples unless too slow.

Corruptions:

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

Run:

```bash
python -m src.eval.sanity_check \
  --manifest data/thought2text/test_human_caption.jsonl \
  --max_samples -1 \
  --caption_ckpt outputs/day5_fusion/F1_real_eeg/checkpoints/best.pt \
  --eeg_ckpt outputs/day4_alignment/best_overall.pt \
  --modes vision_only real_eeg shuffled_eeg random_eeg eeg_only \
  --corruptions clean blur occlusion noise lowres \
  --use_degraded_clip_cache true \
  --degraded_cache_dir data/thought2text/cache/degraded_test \
  --out outputs/day5_sanity
```

If BLIP captions are available and used for training, use the BLIP caption test manifest.

Output one JSONL per corruption/mode.

---

## Task 4.1: Better metrics

Compute:

```text
BLEU-1
BLEU-4
ROUGE-L
METEOR if available
BERTScore if available
average prediction length
distinct prediction ratio
class-name hit accuracy
wnid/class consistency accuracy
```

Class-name hit accuracy is important.

For each prediction, check whether it contains the correct human class name or a synonym if available.

Save:

```text
outputs/day5_sanity/FULL_SANITY_METRICS.csv
outputs/day5_sanity/FULL_SANITY_METRICS.md
```

Table:

```text
| Corruption | Mode | BLEU-1 | BLEU-4 | ROUGE-L | Class Hit | Avg Len | Distinct | Gate Mean |
```

---

## Task 4.2: Gate analysis

Create:

```text
outputs/day5_sanity/gate_analysis.md
```

Analyze:

1. gate mean by corruption,
2. gate mean by mode,
3. real vs shuffled gate difference,
4. whether gate increases under stronger degradation.

Expected desirable trend:

```text
clean gate < degraded gate
real EEG gate differs from shuffled/random gate
```

If this trend does not appear, report honestly.

---

## Task 4.3: Qualitative examples

Create:

```text
outputs/day5_sanity/sample_predictions.jsonl
outputs/day5_sanity/qualitative_examples.md
```

Include at least 30 examples:

* 5 clean
* 5 blur
* 5 occlusion
* 5 noise
* 5 lowres
* 5 failure cases

For each, show:

```text
image_id
label / human class
reference
vision_only prediction
real_eeg prediction
shuffled_eeg prediction
random_eeg prediction
gate values
```

---

# Phase 5: Additional Dataset Preparation

If GPU training is running, a second process can prepare datasets.

Do not block main training.

## Task 5.1: THINGS-EEG2 status

If THINGS-EEG2 exists, inspect it.

Create:

```text
scripts/inspect_things_eeg2.py
outputs/day5_datasets/things_eeg2_status.md
```

Report:

1. whether data exists,
2. file size,
3. available subjects,
4. preprocessed EEG files,
5. image set availability,
6. whether it can be adapted to current manifest schema.

Do not fully migrate unless all Day4–Day5 core tasks are done.

## Task 5.2: EIT-1M status

If EIT-1M exists, inspect it.

Create:

```text
scripts/inspect_eit1m.py
outputs/day5_datasets/eit1m_status.md
```

Report:

1. file list,
2. whether EEG exists,
3. whether image exists,
4. whether text/caption exists,
5. sample count,
6. whether it can be converted to current manifest schema.

---

# Phase 6: Final Two-Day Research Report

Create:

```text
outputs/day5_final/NEXT_48H_RESEARCH_REPORT.md
```

It must include:

1. What was wrong with Day3.
2. What was fixed.
3. Alignment ablation results.
4. Multi-seed results.
5. Best checkpoint path.
6. Caption target improvement.
7. Fusion comparison.
8. Full degraded sanity results.
9. Gate analysis.
10. Qualitative examples.
11. Whether real EEG beats shuffled/random.
12. Whether we can claim EEG benefit.
13. Recommended next step:

    * stronger data,
    * better captions,
    * larger CLIP,
    * THINGS-EEG2 pretraining,
    * EIT-1M tri-modal training.

Conclusion must be honest.

Allowed:

```text
The current evidence suggests / does not suggest that paired EEG improves robust captioning under visual degradation.
```

Forbidden:

```text
The model reads thoughts.
```

---

# Runtime Strategy

This goal should keep the server busy.

Recommended order:

```text
1. caption target repair
2. retrieval debug
3. alignment ablation seed42
4. best alignment multi-seed
5. fusion comparison
6. full degraded sanity
7. dataset inspections
8. final report
```

If GPU is idle during data processing, start CPU-side dataset inspections in parallel.

If GPU is idle after alignment, start fusion immediately.

If fusion is too fast, run:

```text
1. more seeds
2. CLIP ViT-L/14 cache and alignment comparison
3. BLIP caption target generation
4. full test sanity without max_samples
```

---

# Optional Heavy Extension If Everything Finishes Early

If the above completes early and disk/model cache is ready:

## Option A: CLIP ViT-L/14 alignment comparison

Precompute:

```text
openai/clip-vit-large-patch14
```

Then rerun best alignment variant.

Output:

```text
outputs/day5_clipL/clipL_alignment_report.md
```

Compare:

```text
CLIP-B/32 vs CLIP-L/14
```

## Option B: THINGS-EEG2 pretraining preparation

Do not train full THINGS-EEG2 unless inspected successfully.

Only prepare manifest and a 1024-sample smoke alignment run.

Output:

```text
outputs/day5_things_smoke/things_smoke_report.md
```

## Option C: EIT-1M tri-modal smoke

If EIT-1M is available, build a small manifest and run 512-sample smoke.

Output:

```text
outputs/day5_eit1m_smoke/eit1m_smoke_report.md
```

---

# Completion Standard

This goal is complete only if:

```text
1. Caption targets are improved beyond wnid-only templates.
2. Alignment ablation is complete.
3. At least one multi-seed alignment summary exists.
4. Fusion comparison includes vision-only and real EEG.
5. Full degraded sanity check is run.
6. Real EEG is compared against shuffled and random EEG.
7. Gate analysis is reported.
8. Final 48h report exists.
```

If results remain weak, that is acceptable.

Weak results are still useful if the reports clearly explain:

```text
what was tested,
what failed,
why the current evidence is insufficient,
and what should be improved next.
```
