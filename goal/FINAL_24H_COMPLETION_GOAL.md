# FINAL 24H COMPLETION GOAL — EEG-Assisted Robust Semantic Captioning

We only have about one day of server time left. This is the final execution goal.

Do not start broad new research directions. Do not spend time on EIT-1M, THINGS event alignment, EEG-ImageNet, new architectures, or free-form Qwen captioning unless those components are already fully loader-ready and directly help the final Thought2Text result.

The only objective now is:

```text
Maximize and finalize the strongest Thought2Text result for EEG-assisted robust semantic captioning under degraded visual conditions.
```

The final defensible claim should be:

```text
Paired real EEG improves constrained semantic prediction under degraded visual conditions compared with shuffled EEG, random EEG, and ideally vision-only baselines.
```

---

## Current Best Directions

The strongest current directions are:

```text
A2: Temporal-Spectral-Spatial semantic fusion
P2: raw EEG + spectrogram alignment encoder
P2A2: P2-aligned encoder + A2 semantic fusion head
```

Do not dilute effort across weak or unproven directions.

---

## Hard Rules

1. No free-form Qwen caption training.
2. No full LLM fine-tuning.
3. No new dataset engineering unless the dataset is already loader-ready.
4. No 64-sample / 512-sample smoke-only experiments.
5. No new graph encoder / hard-negative / EEG-ImageNet large training.
6. No report polishing before final metrics are produced.
7. Every final run must save config, seed, checkpoint, metrics, and logs.
8. Select final model by validation metrics, then report test metrics.
9. Use multi-seed mean ± std, not only best single seed.
10. If GPU is idle, launch another final seed or increase batch size.

---

## Final Output Directory

Create and fill:

```text
outputs/final_results/
```

Required final files:

```text
outputs/final_results/FINAL_MODEL_SELECTION.md
outputs/final_results/FINAL_24H_REPORT.md
outputs/final_results/A2_FINAL_METRICS.csv
outputs/final_results/A2_FINAL_SUMMARY.md
outputs/final_results/P2_ALIGNMENT_FINAL_METRICS.csv
outputs/final_results/P2_ALIGNMENT_FINAL_SUMMARY.md
outputs/final_results/P2A2_FINAL_METRICS.csv
outputs/final_results/P2A2_FINAL_SUMMARY.md
outputs/final_results/STRONG_DEGRADATION_RESULTS.md
outputs/final_results/GATE_ABLATION_REPORT.md
outputs/final_results/FINAL_CHECKPOINT_PATHS.md
outputs/final_results/QUALITATIVE_EXAMPLES.md
outputs/final_results/GPU_USAGE_FINAL_REPORT.md
```

---

# Priority 1 — A2 Final Semantic Fusion Multi-Seed

Train and evaluate:

```text
A2 Temporal-Spectral-Spatial semantic fusion
```

This is the current strongest downstream semantic result. Finish it first.

## Seeds

Run:

```text
42
123
2025
2718
3407
```

If five seeds cannot finish, complete at least:

```text
42
123
2025
```

## Degradation Conditions

Use strong degradation. Include:

```text
clean
lowres16
lowres8 if implemented
mixed
strong_noise
strong_blur
occlusion50
occlusion70 if implemented
```

If some conditions are not implemented, do not stop. Use all implemented strong degradation conditions and report missing ones.

## Evaluation Modes

Evaluate all modes:

```text
vision_only
real_eeg
shuffled_eeg
random_eeg
eeg_only
```

## Metrics

Report:

```text
Top-1 accuracy
Top-5 accuracy
Class Hit
real - vision gap
real - shuffled gap
real - random gap
win rate over vision
win rate over shuffled/random
mean ± std across seeds
```

## Required Outputs

```text
outputs/final_results/A2_FINAL_METRICS.csv
outputs/final_results/A2_FINAL_SUMMARY.md
outputs/final_results/A2_FINAL_EXAMPLES.md
outputs/final_results/checkpoints/a2/
```

The summary must include this table:

```text
| Corruption | Mode | Top1 Mean | Top1 Std | Top5 Mean | Class Hit | Real-Vision | Real-Shuffled | Real-Random |
```

---

# Priority 2 — P2 Raw + Spectrogram Alignment Multi-Seed

Train the strongest alignment encoder:

```text
P2 raw EEG + spectrogram fusion encoder
```

This should be the final EEG alignment baseline/checkpoint.

## Seeds

Run:

```text
42
123
2025
```

If time remains:

```text
2718
3407
```

## Metrics

Report:

```text
EEG→image R@1 / R@5 / R@10
EEG→class top-1 / top-5
class accuracy
mean rank
median rank
random baseline
train / val / test split metrics
```

Use validation unique-image R@5 as the primary selection metric.

## Required Outputs

```text
outputs/final_results/P2_ALIGNMENT_FINAL_METRICS.csv
outputs/final_results/P2_ALIGNMENT_FINAL_SUMMARY.md
outputs/final_results/best_p2_encoder.pt
outputs/final_results/checkpoints/p2/
```

The summary must explicitly state:

```text
Does P2 beat the previous historical best? yes/no
Best seed:
Best checkpoint:
Best val R@5:
Best test R@5:
```

---

# Priority 3 — P2A2 Combined Final Model

After P2 best encoder is available, train the combined model:

```text
P2-aligned raw+spectrogram EEG encoder
+ A2 temporal-spectral-spatial semantic fusion head
```

Name:

```text
P2A2_final
```

This is the most important final improvement attempt.

## Variants

Train two variants:

```text
P2A2_freeze_encoder
P2A2_unfreeze_last2
```

### Variant 1: P2A2_freeze_encoder

```text
Load best P2 encoder.
Freeze EEG encoder.
Train semantic fusion head / classifier / prototype head.
```

### Variant 2: P2A2_unfreeze_last2

```text
Load best P2 encoder.
Freeze early EEG blocks.
Unfreeze last 2 EEG blocks.
Train semantic fusion head / classifier / prototype head.
Use smaller LR for encoder than fusion head.
```

Recommended LR:

```yaml
fusion_lr: 1e-4
encoder_lr: 1e-5
weight_decay: 0.05
```

## Seeds

Run:

```text
42
123
2025
```

## Training Conditions

Use the same degradation conditions as A2:

```text
clean
lowres16
lowres8 if implemented
mixed
strong_noise
strong_blur
occlusion50
occlusion70 if implemented
```

## Evaluation Modes

Evaluate:

```text
vision_only
real_eeg
shuffled_eeg
random_eeg
eeg_only
```

## Metrics

Report:

```text
Top-1 accuracy
Top-5 accuracy
Class Hit
real - vision gap
real - shuffled gap
real - random gap
win rate over vision
win rate over controls
mean ± std across seeds
```

## Required Outputs

```text
outputs/final_results/P2A2_FINAL_METRICS.csv
outputs/final_results/P2A2_FINAL_SUMMARY.md
outputs/final_results/P2A2_FINAL_EXAMPLES.md
outputs/final_results/checkpoints/p2a2/
```

## Selection Rule

If P2A2 beats A2 on strong degradation mean real EEG accuracy and real-vs-control gap, choose P2A2 as final model.

If P2A2 does not beat A2, keep A2 as final model.

---

# Priority 4 — Real-vs-Shuffled / Real-vs-Random Margin Loss

For A2 and P2A2 training, add or verify margin loss:

```text
score(real_eeg, true_class) > score(shuffled_eeg, true_class) + margin
score(real_eeg, true_class) > score(random_eeg, true_class) + margin
```

Recommended margin:

```yaml
margin: 0.1
lambda_real_vs_shuffled_margin: 0.2
lambda_real_vs_random_margin: 0.2
```

If unstable, reduce:

```yaml
lambda_real_vs_shuffled_margin: 0.1
lambda_real_vs_random_margin: 0.1
```

Do not let this loss break training. If it hurts validation, disable and report.

---

# Priority 5 — Minimal Gate Ablation

Only run this after A2/P2/P2A2 final jobs are launched or finished.

Compare:

```text
A2_with_reliability_gate
A2_no_gate
```

If P2A2 is final, also compare:

```text
P2A2_with_reliability_gate
P2A2_no_gate
```

Run one seed first:

```text
42
```

If cheap, run:

```text
42
123
2025
```

## Required Outputs

```text
outputs/final_results/GATE_ABLATION_REPORT.md
outputs/final_results/GATE_ABLATION_METRICS.csv
```

Question to answer:

```text
Does the reliability gate improve real EEG semantic prediction under strong degradation?
```

If yes, keep gate as a contribution.

If no, do not claim gate learns adaptive EEG reliance. Treat gate only as an engineering fusion module.

---

# GPU Usage Rules

Do not let GPU sit idle.

## If GPU memory usage is below 8GB for more than 10 minutes

Do one of:

```text
increase batch size
run another seed concurrently
run A2 and P2 jobs concurrently if memory allows
run P2A2 freeze variant while P2A2 unfreeze waits
```

## Allowed Parallel Jobs

Allowed if memory is safe:

```text
A2 seed42 + A2 seed123
P2 seed42 + P2 seed123
A2 seed + P2 seed
P2A2_freeze seed + A2 seed
```

Do not run free-form Qwen training.

Do not run large new dataset jobs.

## Log GPU Usage

Update:

```text
outputs/final_results/GPU_USAGE_FINAL_REPORT.md
```

Every 30 minutes, log:

```text
timestamp
running jobs
GPU memory
GPU utilization
batch size
step time
current epoch
best metric so far
```

---

# Final Model Selection

Create:

```text
outputs/final_results/FINAL_MODEL_SELECTION.md
```

Compare:

```text
A2_final
P2A2_freeze_encoder
P2A2_unfreeze_last2
```

Selection metrics, in order:

```text
1. real EEG Top-1 accuracy under strong degradation
2. real - shuffled gap
3. real - random gap
4. real - vision gap
5. stability across seeds
6. Top-5 accuracy
7. Class Hit
```

Strong degradation conditions should count more than clean.

Recommended weighted score:

```python
score = 1.0 * mean_real_acc_strong_degradation \
      + 0.5 * mean_real_minus_shuffled \
      + 0.5 * mean_real_minus_random \
      + 0.3 * mean_real_minus_vision \
      - 0.1 * std_across_seeds
```

The final model selection file must state:

```text
Recommended final model:
Recommended checkpoint:
Recommended metrics file:
Recommended qualitative examples file:
Why this model was selected:
What it failed to solve:
```

---

# Qualitative Examples

Create:

```text
outputs/final_results/QUALITATIVE_EXAMPLES.md
```

Include examples for:

```text
clean
lowres16
mixed
strong_noise
strong_blur
occlusion50
```

For each example include:

```text
image_id
true_class
corruption
vision_only prediction
real_eeg prediction
shuffled_eeg prediction
random_eeg prediction
controlled caption
whether real EEG fixed vision-only
```

Focus on examples where:

```text
vision_only is wrong
real_eeg is correct
shuffled/random are wrong
```

These are the strongest qualitative cases.

---

# What Not To Do

Do not spend final time on:

```text
EIT-1M
THINGS event alignment
EEG-ImageNet training
new architecture search
new graph encoder
new hard-negative experiments
free-form Qwen caption training
full LLM finetuning
report-only work before final metrics
small smoke-only experiments
```

If EIT-1M or THINGS is not already loader-ready, ignore them for this final run.

---

# Final Report

At the end, create:

```text
outputs/final_results/FINAL_24H_REPORT.md
```

It must answer:

```text
1. What is the final model?
2. What is the best checkpoint path?
3. Does real EEG beat shuffled EEG?
4. Does real EEG beat random EEG?
5. Does real EEG beat vision-only under strong degradation?
6. Which corruptions show the largest EEG gain?
7. Did P2A2 improve over A2?
8. Did the gate help?
9. What is the best alignment model?
10. What remains unsolved?
```

The final report must include these exact tables:

```text
Table 1: Final alignment results
Table 2: A2 semantic fusion multi-seed results
Table 3: P2A2 semantic fusion multi-seed results
Table 4: Final model comparison
Table 5: Gate ablation
Table 6: Strong degradation gains
```

---

# Final Scientific Claim

Use only claims supported by metrics.

Allowed claim if metrics support it:

```text
Paired real EEG provides semantic information that improves constrained semantic prediction under degraded visual conditions compared with shuffled and random EEG.
```

Stronger claim allowed only if metrics support it:

```text
Under strong visual degradation, paired real EEG improves semantic prediction over vision-only baselines.
```

Do not claim:

```text
open-ended caption generation is solved
```

Do not claim:

```text
the model reads thoughts
```

Do not claim:

```text
EIT-1M / THINGS improved final results
```

unless directly supported by final metrics.

---

# Completion Criteria

This final goal is complete when:

```text
1. A2 final multi-seed metrics exist.
2. P2 alignment final multi-seed metrics exist.
3. P2A2 final metrics exist or a clear failure report exists.
4. Final model selection exists.
5. Strong degradation results exist.
6. Gate ablation exists or is explicitly skipped due to time.
7. Final checkpoint paths are listed.
8. Final 24h report is complete.
```

After this goal is complete, the project should stop training and move to final write-up.
