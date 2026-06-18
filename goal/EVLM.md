# Goal: Improve Final EVLM Results with EEG-Guided Visual Enhancement

Current final model is A2. A2 is currently the strongest model, but we still want to improve the final EVLM result.

Do not work on free-form Qwen captioning.
Do not work on EIT-1M / THINGS / EEG-ImageNet unless they are already loader-ready and directly useful.
Do not start broad unrelated architecture search.

The main goal is:

```text
Improve our EEG-enhanced VLM result by making EEG act as a visual enhancement signal.
```

Scientific target:

```text
Under degraded visual conditions, paired real EEG should improve semantic prediction compared with vision-only, shuffled EEG, and random EEG.
```

Primary metrics:

```text
Top-1 accuracy
Top-5 accuracy
Class Hit
real EEG - vision-only gap
real EEG - shuffled EEG gap
real EEG - random EEG gap
win rate over vision-only
win rate over shuffled/random controls
mean ± std across seeds
```

Main evaluation conditions:

```text
clean
lowres16
mixed
occlusion50
strong_blur
strong_noise
```

Main evaluation modes:

```text
vision_only
real_eeg
shuffled_eeg
random_eeg
eeg_only
```

Baseline to beat:

```text
A2_final
```

---

# Part 1: EEG Visual Residual Adapter

## Motivation

EEG should not directly replace vision. EEG should correct degraded CLIP image embeddings.

This is the most important direction because it matches our project idea:

```text
EEG = visual enhancement signal
```

## Model

Use current A2 EEG encoder and degraded CLIP image embeddings.

Inputs:

```text
image_emb: [B, 512]
eeg_emb: [B, 512]
text_proto: [C, 512]
label: [B]
```

Compute visual confidence:

```python
image_logits = image_emb @ text_proto.T / tau_cls
image_prob = softmax(image_logits, dim=-1)
vision_conf = image_prob.max(dim=-1).values
```

Build residual adapter input:

```python
x = concat([
    image_emb,
    eeg_emb,
    image_emb * eeg_emb,
    image_emb - eeg_emb,
    vision_conf.unsqueeze(-1)
])
```

Predict EEG correction:

```python
delta = delta_mlp(x)
alpha = sigmoid(alpha_mlp(x))
corrected_img = normalize(image_emb + alpha * delta)
```

Classify:

```python
logits = corrected_img @ text_proto.T / tau_cls
```

## Variants

Run:

```text
A2_residual_scalar
A2_residual_vector
A2_residual_vector_margin
```

Definitions:

```text
A2_residual_scalar:
  alpha shape = [B, 1]

A2_residual_vector:
  alpha shape = [B, 512]

A2_residual_vector_margin:
  alpha shape = [B, 512]
  add real-vs-shuffled/random margin loss
```

## Loss

Use:

```text
L = 1.0 * CE_real
  + 0.2 * margin_real_vs_shuffled
  + 0.2 * margin_real_vs_random
  + 0.01 * delta_norm
```

Margin loss:

```python
score_real = logits_real[range(B), y]
score_shuf = logits_shuf[range(B), y]
score_rand = logits_rand[range(B), y]

loss_shuf = relu(margin - (score_real - score_shuf)).mean()
loss_rand = relu(margin - (score_real - score_rand)).mean()
```

Recommended config:

```yaml
margin: 0.1
tau_cls: 0.07
lr: 1e-4
weight_decay: 0.05
batch_size: auto
epochs: 80
patience: 12
seeds: [42, 123, 2025]
```

Required outputs:

```text
outputs/evlm_improve/residual/A2_residual_scalar_summary.md
outputs/evlm_improve/residual/A2_residual_vector_summary.md
outputs/evlm_improve/residual/A2_residual_vector_margin_summary.md
outputs/evlm_improve/residual/metrics.csv
outputs/evlm_improve/residual/checkpoints/
```

---

# Part 2: EEG-Guided Prototype Attention Bias

## Motivation

The task is small-sample semantic prediction. We can use CLIP text prototypes as semantic anchors and let EEG guide prototype selection.

This direction is especially suitable for the 40-class Thought2Text setting.

## Model

Compute image logits:

```python
image_logits = image_emb @ text_proto.T / tau_cls
```

Compute EEG logits:

```python
eeg_logits = eeg_emb @ text_proto.T / tau_cls
```

Compute visual confidence:

```python
vision_conf = softmax(image_logits, dim=-1).max(dim=-1).values
```

Compute EEG contribution gate:

```python
gamma_raw = sigmoid(gamma_mlp(concat([
    image_emb,
    eeg_emb,
    vision_conf.unsqueeze(-1)
])))
gamma = gamma_raw * (1.0 - vision_conf).unsqueeze(-1)
```

Final logits:

```python
final_logits = image_logits + gamma * eeg_logits
```

Interpretation:

```text
If vision is confident, EEG changes little.
If vision is degraded or uncertain, EEG contributes more.
```

## Variants

Run:

```text
A2_proto_bias
A2_proto_bias_margin
```

Loss:

```text
CE_real
+ real-vs-shuffled/random margin
+ optional gamma regularization
```

Gamma regularization:

```text
keep gamma low on clean high-confidence samples
allow gamma higher on degraded low-confidence samples
```

Recommended config:

```yaml
margin: 0.1
tau_cls: 0.07
lr: 1e-4
weight_decay: 0.05
batch_size: auto
epochs: 80
patience: 12
seeds: [42, 123, 2025]
```

Required outputs:

```text
outputs/evlm_improve/proto_bias/A2_proto_bias_summary.md
outputs/evlm_improve/proto_bias/A2_proto_bias_margin_summary.md
outputs/evlm_improve/proto_bias/metrics.csv
outputs/evlm_improve/proto_bias/checkpoints/
```

---

# Part 3: Combined EEG Visual Enhancement Model

After Part 1 and Part 2 finish, run one combined model:

```text
A2_residual_plus_proto_bias
```

Forward:

```python
# residual correction
corrected_img = normalize(image_emb + alpha * delta)

# prototype logits
image_logits = corrected_img @ text_proto.T / tau_cls
eeg_logits = eeg_emb @ text_proto.T / tau_cls

# EEG prototype bias
final_logits = image_logits + gamma * eeg_logits
```

Run seeds:

```text
42
123
2025
```

Required outputs:

```text
outputs/evlm_improve/combined/A2_residual_plus_proto_bias_summary.md
outputs/evlm_improve/combined/metrics.csv
outputs/evlm_improve/combined/checkpoints/
```

---

# Part 4: Model Selection

Compare all models against A2_final:

```text
A2_final
A2_residual_scalar
A2_residual_vector
A2_residual_vector_margin
A2_proto_bias
A2_proto_bias_margin
A2_residual_plus_proto_bias
```

Selection metrics, in order:

```text
1. mean real EEG Top-1 accuracy under strong degradation
2. real EEG - vision-only gap
3. real EEG - shuffled EEG gap
4. real EEG - random EEG gap
5. win rate over vision-only
6. win rate over shuffled/random
7. stability across seeds
```

Strong degradation conditions count more than clean:

```text
lowres16
mixed
occlusion50
strong_blur
strong_noise
```

Recommended score:

```python
score = 1.0 * mean_real_acc_strong_degradation \
      + 0.5 * mean_real_minus_vision \
      + 0.5 * mean_real_minus_shuffled \
      + 0.5 * mean_real_minus_random \
      - 0.1 * std_across_seeds
```

Required output:

```text
outputs/evlm_improve/EVLM_MODEL_SELECTION.md
outputs/evlm_improve/EVLM_MODEL_SELECTION.csv
```

The selection report must state:

```text
Best model:
Best checkpoint:
Best metrics file:
Does it beat A2_final? yes/no
Where does it improve?
Where does it fail?
Recommended final model for report:
```

---

# Part 5: Autonomous Exploration If Main Experiments Finish

If all above experiments finish early, do not stop. Continue with useful autonomous exploration.

The final objective is always:

```text
Improve EVLM metrics and make the final result stronger.
```

Do not run random experiments. Only explore directions that directly target final EVLM performance.

## Exploration Rule

Any autonomous experiment must satisfy at least one of these:

```text
1. It may improve strong-degradation real EEG accuracy.
2. It may improve real-vs-vision gap.
3. It may improve real-vs-shuffled/random gap.
4. It may improve stability across seeds.
5. It may improve interpretability of EEG visual enhancement.
```

Every autonomous experiment must be compared against A2_final and the best new model so far.

---

## Autonomous Direction A: Stronger Vision Confidence Control

Try alternative confidence definitions:

```text
max softmax probability
entropy of image prototype distribution
margin between top1 and top2 image logits
CLIP embedding norm / degradation confidence
```

Experiments:

```text
A2_residual_entropy_conf
A2_residual_top2_margin_conf
A2_proto_bias_entropy_conf
```

Goal:

```text
Better decide when EEG should influence vision.
```

---

## Autonomous Direction B: Stronger Degradation Curriculum

Train with degradation curriculum:

```text
Stage 1: clean + mild degradation
Stage 2: strong degradation
Stage 3: mixed corruption + shuffled/random margin
```

Experiments:

```text
A2_residual_curriculum
A2_proto_bias_curriculum
A2_combined_curriculum
```

Goal:

```text
Improve strong degradation performance without hurting clean.
```

---

## Autonomous Direction C: More Aggressive Control Margin

Try margin values:

```text
0.05
0.1
0.2
0.3
```

Experiments:

```text
A2_residual_margin005
A2_residual_margin02
A2_proto_bias_margin02
A2_combined_margin02
```

Goal:

```text
Increase real EEG advantage over shuffled/random EEG.
```

Reject if real EEG accuracy drops too much.

---

## Autonomous Direction D: Freeze/Unfreeze A2 Encoder

Try:

```text
freeze A2 EEG encoder
unfreeze last 1 block
unfreeze last 2 blocks
unfreeze all EEG encoder with small lr
```

Recommended LR:

```yaml
encoder_lr: 1e-5
head_lr: 1e-4
```

Experiments:

```text
A2_residual_freeze
A2_residual_unfreeze_last1
A2_residual_unfreeze_last2
A2_combined_unfreeze_last2
```

Goal:

```text
Determine whether adapting the EEG encoder helps final semantic prediction.
```

---

## Autonomous Direction E: Test-Time Ensemble

If multiple models are good, try logit ensemble:

```text
final_logits = w1 * A2_logits
             + w2 * residual_logits
             + w3 * proto_bias_logits
```

Try simple weights:

```text
uniform
validation-optimized weights
strong-degradation optimized weights
```

Experiments:

```text
A2_residual_ensemble
A2_residual_proto_ensemble
best3_ensemble
```

Goal:

```text
Improve final metrics without retraining.
```

This is cheap and should be tried if multiple trained models exist.

---

## Autonomous Direction F: Top-K Prototype Smoothing

Instead of hard class prediction, smooth over top-k prototypes.

Try:

```text
top3 prototype smoothing
top5 prototype smoothing
semantic neighbor smoothing
```

Experiments:

```text
A2_proto_top3_smooth
A2_proto_top5_smooth
A2_combined_top5_smooth
```

Goal:

```text
Improve top-5 and class-hit stability.
```

---

# GPU Usage

Do not let GPU idle.

If GPU memory usage is below 8GB for more than 10 minutes:

```text
increase batch size
run another seed concurrently
launch next queued experiment
run ensemble/evaluation while training another seed
```

Allowed parallel jobs if memory allows:

```text
A2_residual seed42 + A2_residual seed123
A2_proto seed42 + A2_proto seed123
residual training + proto_bias training
```

Do not run free-form Qwen training.

Do not run EIT/THINGS/EEG-ImageNet work.

---

# Final Required Report

At the end, create:

```text
outputs/evlm_improve/FINAL_EVLM_IMPROVEMENT_REPORT.md
```

It must answer:

```text
1. Did EEG Visual Residual Adapter improve over A2_final?
2. Did Prototype Attention Bias improve over A2_final?
3. Did combined residual + prototype bias improve over A2_final?
4. Which model has the best strong-degradation performance?
5. Which model has the best real-vs-vision gap?
6. Which model has the best real-vs-shuffled/random gap?
7. Which model is most stable across seeds?
8. Which model should be used as final EVLM model?
9. What autonomous explorations were attempted?
10. Which autonomous exploration helped most?
11. What still did not improve?
```

Required final tables:

```text
Table 1: A2_final baseline
Table 2: Residual Adapter results
Table 3: Prototype Bias results
Table 4: Combined model results
Table 5: Autonomous exploration results
Table 6: Final model selection
```

Final statement must include:

```text
Recommended final EVLM model:
Recommended checkpoint:
Recommended final metrics file:
Whether it beats A2_final:
Best improvement conditions:
Remaining limitations:
```

---

# Completion Criteria

This goal is complete when:

```text
1. Residual Adapter variants finish.
2. Prototype Attention Bias variants finish.
3. Combined model finishes.
4. At least one autonomous exploration is attempted if time remains.
5. Final model selection report exists.
6. Final EVLM improvement report exists.
```

If no new model beats A2_final, report that honestly and keep A2_final as the final model. Negative ablations are acceptable if clearly documented.
