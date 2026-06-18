# Goal: Token-Level EEG-Enhanced VLM and Free-Form Generative EVLM

We still have about one day of server time. The previous experiments mainly explored embedding-level residual/prototype fusion. Now the goal is to explore a more genuine EVLM direction:

```text
Image → CLIP ViT patch tokens
EEG → EEG tokens
visual tokens attend to EEG tokens
enhanced visual tokens
→ classifier and generative caption model
```

This goal has two mandatory parts:

```text
Part A: Token-level EEG-Vision fusion for semantic prediction.
Part B: Free-form generative EVLM captioning using the enhanced visual tokens.
```

The final report must include at least one valid free-form generated caption example that can be shown in the course report.

Do not only run classifier experiments.
Do not only output structured JSON.
Do not only output template captions.
The final objective is to produce a meaningful EEG-enhanced generative EVLM prototype.

---

# 0. Project Context

Current strongest model:

```text
A2_final
```

A2_final remains the main semantic baseline.

The new model should try to improve or at least meaningfully extend A2 by doing token-level EEG-VLM fusion.

Scientific target:

```text
Under degraded visual conditions, paired real EEG should help the VLM understand the image better than vision-only, shuffled EEG, or random EEG.
```

Generative target:

```text
Generate short natural-language image captions using degraded image + EEG.
```

Example desired output:

```text
a dog standing in a grassy field
a grand piano in a room
a small airplane flying in the sky
```

Not desired:

```text
JSON only
class name only
WNID tokens
URLs
code-like output
HTML
chatty explanations
```

---

# 1. Mandatory Model: Token-Level EEG-Vision Fusion

Implement the most promising token-level fusion model first.

Model name:

```text
A2_ViTTokenFusion_ConfidenceMargin
```

## 1.1 Image branch

Use CLIP ViT and return patch tokens instead of only pooled image embedding.

Input:

```text
degraded image
```

Output:

```text
visual_tokens: [B, N, Dv]
```

For ViT-B/32, this may be approximately:

```text
CLS token + patch tokens = [B, 50, 768]
```

Project visual tokens to CLIP semantic dimension:

```python
visual_tokens = visual_proj(visual_tokens)  # [B, N, 512]
```

Freeze CLIP vision encoder initially.

Do not unfreeze CLIP in the first run.

---

## 1.2 EEG branch

Use the existing A2 EEG encoder.

Do not replace A2 unless necessary.

Output either:

```text
eeg_emb: [B, 512]
```

Then convert it into multiple EEG tokens:

```python
eeg_tokens = eeg_tokenizer(eeg_emb)
eeg_tokens = eeg_tokens.reshape(B, M, 512)
```

Start with:

```text
M = 4
```

Later try:

```text
M = 8
M = 16
```

Only if time remains.

---

## 1.3 Cross-attention fusion

Visual tokens should attend to EEG tokens:

```text
Q = visual_tokens
K = eeg_tokens
V = eeg_tokens
```

Compute:

```python
eeg_context = CrossAttention(
    query=visual_tokens,
    key=eeg_tokens,
    value=eeg_tokens
)
```

Use residual fusion:

```python
enhanced_visual_tokens = visual_tokens + beta * eeg_context
```

Use a confidence-aware beta gate:

```python
vision_logits = pooled_visual @ text_proto.T / tau_cls
vision_prob = softmax(vision_logits, dim=-1)
vision_conf = vision_prob.max(dim=-1).values

beta_raw = sigmoid(beta_mlp(concat([
    pooled_visual,
    pooled_eeg,
    vision_conf.unsqueeze(-1)
])))

beta = beta_raw * (1.0 - vision_conf).unsqueeze(-1)
```

Interpretation:

```text
If vision is confident, EEG should have small influence.
If vision is degraded or uncertain, EEG should have larger influence.
```

This is important. Do not let EEG dominate clean images.

---

## 1.4 Pooling

Use CLS + mean pooling:

```python
cls_token = enhanced_visual_tokens[:, 0]
mean_patch = enhanced_visual_tokens[:, 1:].mean(dim=1)

enhanced_img_emb = pool_mlp(concat([cls_token, mean_patch]))
enhanced_img_emb = normalize(enhanced_img_emb)
```

Then classify by CLIP text prototypes:

```python
logits = enhanced_img_emb @ text_proto.T / tau_cls
```

---

# 2. Training Loss for Token Fusion

Use real EEG classification loss:

```text
CE_real
```

Add real-vs-control margin loss:

```python
score_real = logits_real[range(B), y]
score_shuf = logits_shuf[range(B), y]
score_rand = logits_rand[range(B), y]

loss_shuf = relu(margin - (score_real - score_shuf)).mean()
loss_rand = relu(margin - (score_real - score_rand)).mean()
```

Total loss:

```text
L = 1.0 * CE_real
  + 0.2 * margin_real_vs_shuffled
  + 0.2 * margin_real_vs_random
  + 0.01 * beta_regularization
```

Recommended config:

```yaml
margin: 0.1
tau_cls: 0.07
lr: 1e-4
weight_decay: 0.05
epochs: 80
patience: 12
seeds: [42, 123, 2025]
batch_size: auto
```

Trainable:

```text
visual_proj
eeg_tokenizer
cross-attention adapter
beta_mlp
pool_mlp
classification head if needed
```

Frozen initially:

```text
CLIP vision encoder
A2 EEG encoder
CLIP text encoder / text prototypes
```

If performance is weak, later try unfreezing the last A2 EEG block with small LR.

Do not unfreeze CLIP unless all lighter options fail.

---

# 3. Required Token Fusion Variants

Run these in order:

```text
VTF1_basic_M4
VTF2_confidence_beta_M4
VTF3_confidence_beta_margin_M4
VTF4_confidence_beta_margin_M8
```

Definitions:

```text
VTF1_basic_M4:
  M = 4 EEG tokens
  cross-attention
  simple learnable beta
  no margin

VTF2_confidence_beta_M4:
  M = 4
  beta = beta_raw * (1 - vision_conf)
  no margin

VTF3_confidence_beta_margin_M4:
  M = 4
  confidence-aware beta
  real-vs-shuffled/random margin

VTF4_confidence_beta_margin_M8:
  same as VTF3
  M = 8 EEG tokens
```

The most important model is:

```text
VTF3_confidence_beta_margin_M4
```

Run this first.

If time is limited, prioritize:

```text
VTF3 → VTF4 → VTF2 → VTF1
```

---

# 4. Evaluation for Token Fusion

Evaluate on:

```text
clean
lowres16
mixed
occlusion50
strong_blur
strong_noise
```

Modes:

```text
vision_only
real_eeg
shuffled_eeg
random_eeg
eeg_only
```

Compare against:

```text
A2_final
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

Strong degradation conditions are more important:

```text
lowres16
mixed
occlusion50
strong_blur
strong_noise
```

Required outputs:

```text
outputs/token_generative_evlm/token_fusion/VTF_MODEL_SELECTION.md
outputs/token_generative_evlm/token_fusion/VTF_MODEL_SELECTION.csv
outputs/token_generative_evlm/token_fusion/metrics.csv
outputs/token_generative_evlm/token_fusion/checkpoints/
```

The report must state:

```text
Best token fusion model:
Does it beat A2_final:
Where it improves:
Where it fails:
Whether real EEG beats shuffled/random:
Recommended token-fusion checkpoint:
```

---

# 5. Mandatory Free-Form Generative EVLM

After implementing token fusion, build a generative EVLM on top of the enhanced visual tokens.

This part is mandatory even if token fusion does not beat A2_final.

The goal is to produce at least one valid, natural, free-form caption example for the course report.

Do not stop at classification.

---

## 5.1 Generative model name

Use:

```text
GenEVLM_VTF_Prefix
```

and optionally:

```text
GenEVLM_VTF_LoRA
```

---

## 5.2 Inputs to generative EVLM

Use the token-fusion model output:

```text
enhanced_visual_tokens: [B, N, 512]
enhanced_img_emb: [B, 512]
eeg_tokens: [B, M, 512]
A2 logits or token-fusion logits
top-k text prototypes
corruption embedding
vision confidence
EEG mode indicator
```

The generative model should use real EEG for real generation and should also evaluate shuffled/random EEG controls.

---

## 5.3 Language model

Use the smallest available stable LLM already supported in the repo.

Preferred:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Do not full-finetune the LLM.

First try:

```text
frozen LLM + trainable prefix projector
```

Then if time remains:

```text
LoRA on attention layers
```

LoRA config:

```yaml
r: 8
alpha: 16
dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
```

---

## 5.4 Prefix projector

Project enhanced VLM features into LLM soft prefix tokens:

```python
prefix_tokens = PrefixProjector(
    enhanced_visual_tokens,
    enhanced_img_emb,
    eeg_tokens,
    topk_text_prototypes,
    corruption_embedding,
    confidence_features
)
```

Output:

```text
prefix_tokens: [B, prefix_len, llm_hidden_dim]
```

Try first:

```text
prefix_len = 16
```

If time remains:

```text
prefix_len = 8
prefix_len = 32
```

---

## 5.5 Caption targets

Use real natural-language caption targets if available.

If no human captions exist, create pseudo-captions from clean images using an available caption model such as BLIP/BLIP-2 if available in the repo or already installed.

Priority:

```text
1. Existing human captions, if available.
2. BLIP/BLIP-2 pseudo-captions from clean images, if available.
3. Short class-based natural captions as fallback.
```

Fallback examples:

```text
a photo of a dog
a photo of a grand piano
a photo of an airplane
```

Even fallback captions must be normal free-form text, not JSON.

Do not use:

```text
WNID-only strings
URLs
code-like text
HTML
invalid class tokens
```

Create a target report:

```text
outputs/token_generative_evlm/generation/CAPTION_TARGET_REPORT.md
```

It must include:

```text
caption source
number of captions
average caption length
invalid target rate
10 example targets
```

---

# 6. Generative EVLM Variants

Run these in order:

```text
G0_image_only_prefix
G1_vtf_visual_prefix
G2_vtf_visual_eeg_prefix
G3_vtf_visual_eeg_topk_prefix
G4_vtf_visual_eeg_topk_lora
```

Definitions:

```text
G0_image_only_prefix:
  degraded CLIP visual tokens/embedding only
  no EEG
  frozen LLM

G1_vtf_visual_prefix:
  token-fusion enhanced visual features
  no explicit EEG tokens
  frozen LLM

G2_vtf_visual_eeg_prefix:
  enhanced visual features + EEG tokens
  frozen LLM

G3_vtf_visual_eeg_topk_prefix:
  enhanced visual features + EEG tokens + top-k semantic prototypes/logits
  frozen LLM

G4_vtf_visual_eeg_topk_lora:
  same as G3
  LoRA enabled
```

Most important first:

```text
G3_vtf_visual_eeg_topk_prefix
```

If time is limited, run:

```text
G3 first
then G2
then G0 baseline
then G4 LoRA
```

---

# 7. Generation Training

Use teacher forcing.

Loss:

```text
caption cross-entropy loss
```

Optional auxiliary loss:

```text
generated caption should contain or imply the correct class
```

Recommended config:

```yaml
epochs: 20
patience: 5
batch_size: auto
grad_accum_steps: auto
lr_projector: 1e-4
lr_lora: 2e-5
weight_decay: 0.01
bf16: true
max_caption_length: 32
seeds: [42, 123, 2025]
```

For G4 LoRA, first run seed 42 only.
If outputs are valid, continue other seeds.

---

# 8. Generation Decoding

Use short free-form caption decoding.

Prompt:

```text
Write one short natural image caption. Do not mention EEG. Do not output JSON, code, URLs, markdown, or explanations.
```

Decoding config:

```yaml
max_new_tokens: 24
temperature: 0.2
top_p: 0.9
repetition_penalty: 1.1
no_repeat_ngram_size: 3
```

Output should look like:

```text
a dog running on the grass
a piano in a room
a bird sitting on a branch
```

---

# 9. Generation Evaluation

Evaluate on:

```text
clean
lowres16
mixed
occlusion50
strong_blur
strong_noise
```

Modes:

```text
vision_only
real_eeg
shuffled_eeg
random_eeg
eeg_only
```

Report:

```text
caption class hit
caption top-k class hit
real EEG - vision-only class-hit gap
real EEG - shuffled EEG class-hit gap
real EEG - random EEG class-hit gap
valid caption rate
invalid output rate
average caption length
distinct caption count
```

Invalid output if:

```text
empty caption
contains URL
contains code
contains HTML
contains WNID-only string
contains markdown table
contains apology/explanation
too long
repeats same phrase excessively
```

Optional text metrics if available:

```text
BLEU-1
BLEU-4
ROUGE-L
METEOR
BERTScore
CLIPScore between generated caption and image
```

If these libraries are not available, do not spend too much time installing them. Semantic class-hit and valid caption rate are more important.

---

# 10. Required Qualitative Examples

Create:

```text
outputs/token_generative_evlm/generation/QUALITATIVE_EXAMPLES.md
```

It must include at least 30 examples.

Each example must include:

```text
image_id
true class
corruption
mode
generated caption
valid or invalid
class hit or miss
```

Must include a special section:

```text
Best examples for course report
```

This section should include at least 5 examples where possible:

```text
image-only output is wrong or vague
real EEG output is better or correct
shuffled/random EEG output is worse
```

Even if this pattern is rare, find the best available examples and report honestly.

---

# 11. Required Final Outputs

Create:

```text
outputs/token_generative_evlm/
```

Required files:

```text
outputs/token_generative_evlm/TOKEN_GEN_EVLM_FINAL_REPORT.md
outputs/token_generative_evlm/TOKEN_GEN_EVLM_MODEL_SELECTION.md
outputs/token_generative_evlm/TOKEN_GEN_EVLM_MODEL_SELECTION.csv

outputs/token_generative_evlm/token_fusion/VTF_MODEL_SELECTION.md
outputs/token_generative_evlm/token_fusion/metrics.csv

outputs/token_generative_evlm/generation/CAPTION_TARGET_REPORT.md
outputs/token_generative_evlm/generation/GENERATION_MODEL_SELECTION.md
outputs/token_generative_evlm/generation/GENERATION_METRICS.csv
outputs/token_generative_evlm/generation/QUALITATIVE_EXAMPLES.md
outputs/token_generative_evlm/generation/INVALID_OUTPUT_REPORT.md
outputs/token_generative_evlm/generation/checkpoints/
```

Final report must answer:

```text
1. Did token-level EEG-VLM fusion improve over A2_final?
2. Did real EEG beat shuffled/random EEG?
3. Which token fusion variant worked best?
4. Did we produce free-form generated captions?
5. Which generative variant produced the best captions?
6. Did EEG improve generated caption class-hit?
7. What is the invalid output rate?
8. Which examples can be used in the course report?
9. Should A2_final remain the main quantitative result?
10. Should token-level/generative EVLM be included as an exploratory result?
```

---

# 12. AutoSOTA Mode If Time Remains

If all required experiments finish early, do not stop.

Enter AutoSOTA mode.

The goal of AutoSOTA is:

```text
Make the EVLM result stronger, especially free-form generation quality and EEG-specific improvement.
```

Only explore directions that are directly useful for:

```text
better generated captions
higher real EEG caption class-hit
larger real-vs-shuffled/random gap
better strong-degradation performance
lower invalid output rate
better qualitative examples for the report
```

Do not run unrelated experiments.

---

## AutoSOTA Direction A: Token Fusion Improvements

Try:

```text
M = 8 EEG tokens
M = 16 EEG tokens
2 cross-attention layers
visual→EEG attention
EEG→visual attention
bidirectional co-attention
```

Candidate experiments:

```text
VTF_M8_2Layer
VTF_M16_2Layer
VTF_BiCrossAttention
```

Reject if shuffled/random gains as much as real EEG.

---

## AutoSOTA Direction B: Better Beta/Gating

Try:

```text
entropy-based vision uncertainty
top1-top2 logit margin
condition-specific beta
corruption embedding in beta_mlp
```

Candidate experiments:

```text
VTF_entropy_beta
VTF_top2margin_beta
VTF_corruption_aware_beta
```

Goal:

```text
EEG should help more when vision is degraded and less when vision is clean.
```

---

## AutoSOTA Direction C: Generation Stabilization

If generated text is invalid or repetitive, try:

```text
lower temperature
stronger repetition penalty
shorter max_new_tokens
caption-only prompt
filter invalid training captions
train more epochs only on valid captions
```

Candidate experiments:

```text
G3_decode_strict
G3_clean_targets_only
G3_short_caption_only
```

---

## AutoSOTA Direction D: Better Caption Targets

If pseudo-captions are available, compare:

```text
template captions
BLIP pseudo-captions
class + BLIP mixed captions
```

Candidate experiments:

```text
G3_template_target
G3_blip_target
G3_mixed_target
```

Pick the one with best valid caption rate and class-hit.

---

## AutoSOTA Direction E: LoRA Sweep

If G4 LoRA works, try:

```text
r = 4
r = 8
r = 16
```

Candidate experiments:

```text
G4_lora_r4
G4_lora_r8
G4_lora_r16
```

Do not full-finetune LLM.

---

## AutoSOTA Direction F: Ensemble for Generation Guidance

Use best semantic model to guide generation.

Try:

```text
A2_final logits
VTF logits
ensemble logits
top-k prototypes from ensemble
```

Candidate experiment:

```text
G3_ensemble_semantic_guidance
```

Goal:

```text
Use the strongest semantic predictor to guide free-form generation.
```

---

# 13. GPU Usage

Do not let GPU idle.

If GPU memory is below 8GB for more than 10 minutes:

```text
increase batch size
run another seed
increase prefix length
enable LoRA
run evaluation while another model trains
```

This task should use more GPU than lightweight residual/prototype adapter experiments.

Log GPU usage in:

```text
outputs/token_generative_evlm/GPU_USAGE.md
```

---

# 14. Completion Criteria

This goal is complete only when:

```text
1. At least one token-level EEG-VLM fusion model has been trained and evaluated.
2. At least one free-form generative EVLM model has been trained and evaluated.
3. At least 30 qualitative generated caption examples are saved.
4. At least 5 best examples are selected for the course report if possible.
5. A final report clearly states whether the generative EVLM is usable.
6. AutoSOTA was attempted if time remained.
```

Do not mark complete after classifier-only experiments.

Do not mark complete after smoke tests.

Do not mark complete without generated captions.

If the generative captions are poor, report that honestly, but still provide the best examples and explain the limitations.
