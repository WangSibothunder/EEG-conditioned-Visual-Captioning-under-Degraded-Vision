# Goal: Explore the Best Way to Feed EEG-Enhanced Vision Tokens into a Generative VLM

We already have token-level EEG-enhanced vision features from VTF-style models:

```text
Image
  → CLIP ViT
  → visual patch tokens

EEG
  → A2 EEG encoder
  → EEG tokens

visual tokens attend to EEG tokens
  → enhanced vision tokens
```

Now the main objective is to explore the best way to feed these enhanced vision tokens into a generative model and produce free-form natural-language captions.

This goal is not complete unless we obtain real generated captions.

Do not stop at classification.
Do not output only JSON.
Do not output only template captions.
Do not run unrelated datasets.
Do not run EIT-1M / THINGS / EEG-ImageNet unless they are already fully loader-ready and directly useful for this goal.

The final objective is:

```text
Build the strongest possible generative EVLM prototype using EEG-enhanced vision tokens.
```

The final report must include at least:

```text
1. One working generative EVLM pipeline.
2. Free-form caption examples.
3. Comparison between vision-only, real EEG, shuffled EEG, and random EEG.
4. At least 5 good qualitative examples for the course report.
5. Clear statement of which strategy worked best.
```

---

# Priority Ranking

Explore the following strategies in this order.

## Priority 1: Enhanced Vision Tokens → LLM Visual Prefix + Semantic Prompt + LoRA

This is the most likely to work.

Reason:

```text
Enhanced vision tokens alone may be hard for the LLM to interpret.
Adding top-k semantic concepts gives the LLM a useful language anchor.
LoRA helps the LLM adapt to the prefix format without full finetuning.
```

Model name:

```text
EVG1_TokenPrefix_SemanticPrompt_LoRA
```

This should be the first and most important generative EVLM model.

---

## Priority 2: Enhanced Vision Tokens → Perceiver / Q-Former Resampler → LLM Prefix + LoRA

This is the second most promising.

Reason:

```text
Enhanced vision tokens may be too many or too noisy.
A Q-Former/Perceiver-style bridge can compress them into a small set of multimodal query tokens before feeding the LLM.
```

Model name:

```text
EVG2_QFormerBridge_LoRA
```

---

## Priority 3: LLaVA-style Projector if Compatible

Try this if the repo can support it without huge engineering risk.

Reason:

```text
LLaVA-style models are designed to map CLIP visual tokens into LLM token space.
If token dimensions and vision tower outputs are compatible, this is the most VLM-like solution.
```

Model name:

```text
EVG3_LLaVAStyle_Projector
```

---

## Priority 4: BLIP-2 / Q-Former-style Captioning if Available

Try this if BLIP-2 / InstructBLIP / similar model is already installed or easy to load.

Reason:

```text
BLIP-2 already uses a Q-Former bridge between image tokens and language generation.
This may be easier than directly modifying Qwen-VL internals.
```

Model name:

```text
EVG4_BLIP2Style_EnhancedTokens
```

---

## Priority 5: Qwen-VL Internal Token Replacement or Adapter

This is more risky. Try only if time remains and the APIs are accessible.

Reason:

```text
Qwen-VL has its own visual encoder, projector, token layout, and position handling.
Directly replacing its vision tokens may be hard.
```

Model name:

```text
EVG5_QwenVL_InternalAdapter
```

---

## Priority 6: Hybrid AutoSOTA

If all above are completed or blocked, automatically explore combinations that improve generation quality, valid caption rate, and real-vs-control EEG gaps.

Examples:

```text
best enhanced vision tokens + A2 top-k semantic prompt
best enhanced vision tokens + VTF top-k semantic prompt
A2/VTF ensemble semantic prompt
Q-Former bridge + semantic prompt
prefix projector + LoRA rank sweep
caption target filtering
strict decoding
beam search
```

---

# Current Inputs

Use current available artifacts:

```text
A2_final semantic model
VTF token-level EEG-enhanced vision model
CLIP ViT patch tokens
A2 EEG encoder
EEG tokens
CLIP text prototypes
A2 logits
VTF logits
top-k class predictions
corruption type
vision confidence
```

The model should compare:

```text
vision_only
real_eeg
shuffled_eeg
random_eeg
eeg_only
```

Evaluation conditions:

```text
clean
lowres16
mixed
occlusion50
strong_blur
strong_noise
```

Strong degradation conditions are most important:

```text
lowres16
mixed
occlusion50
strong_blur
strong_noise
```

---

# Part 1: Prepare Enhanced Vision Tokens

Use the best currently available token-level fusion model.

Preferred:

```text
VTF3_confidence_beta_margin_M4
```

If newer/better VTF checkpoints exist, choose the best one by:

```text
1. real EEG Top-1 under strong degradation
2. real - shuffled gap
3. real - random gap
4. valid enhanced token pipeline
```

Required output:

```text
outputs/vision_token_gen_evlm/prep/ENHANCED_TOKEN_SOURCE.md
```

This file must state:

```text
which VTF checkpoint is used
token shape
hidden dimension
number of tokens
whether tokens come from real/shuffled/random/vision-only modes
whether CLIP/VTF/A2 modules are frozen
```

---

# Part 2: Caption Targets

Before training generation, prepare clean caption targets.

Target priority:

```text
1. Human captions if available.
2. BLIP/BLIP-2 pseudo-captions generated from clean images.
3. Class-based natural captions as fallback.
4. Mixed target: class caption + BLIP caption if BLIP captions are clean.
```

Fallback target examples:

```text
a photo of a mountain bike
a photo of a grand piano
a photo of a dog
a photo of an airplane
```

These are acceptable fallback captions, but do not call them structured JSON or templates in final examples. They are short natural caption targets.

Filter bad caption targets.

Remove captions that:

```text
contain URLs
contain HTML
contain code
contain WNID tokens
repeat the same word more than 3 times
are empty
are longer than 20 words
are not image captions
contain apology/chatbot/explanation text
```

Required output:

```text
outputs/vision_token_gen_evlm/caption_targets/CAPTION_TARGET_REPORT.md
```

It must include:

```text
caption source
number of captions
invalid target count
average caption length
10 good examples
10 removed bad examples if any
final target strategy
```

---

# Part 3: Priority 1 Model — Token Prefix + Semantic Prompt + LoRA

This is the most important model.

Model name:

```text
EVG1_TokenPrefix_SemanticPrompt_LoRA
```

## Architecture

Use enhanced vision tokens:

```text
enhanced_visual_tokens: [B, N, 512]
```

Pool and project them into LLM prefix space:

```python
prefix_tokens = VisualTokenPrefixProjector(
    enhanced_visual_tokens,
    enhanced_img_emb,
    eeg_tokens,
    topk_text_prototypes,
    confidence_features,
    corruption_embedding
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

Then AutoSOTA can try:

```text
prefix_len = 8
prefix_len = 32
```

Use LLM:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

If this exact model is unavailable, use the smallest available Qwen/Qwen2/LLM already supported in the repo.

Do not full-finetune the LLM.

Train:

```text
visual token prefix projector
optional small resampler
LoRA weights
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

Training mode:

```text
teacher forcing caption cross entropy
```

Prompt format:

```text
Write one short natural image caption.

Candidate visual concepts:
{top5_classes}

Visual condition:
{corruption_type}

Caption:
```

Important:

```text
The candidate concepts are allowed, but the model must generate a normal caption sentence.
Do not output JSON.
Do not output a list.
Do not output only the class name.
```

Implementation hint:

Use `inputs_embeds` if needed:

```text
[prefix embeddings] + [text prompt token embeddings]
```

Labels should be:

```text
-100 for prefix and prompt tokens
normal token labels for caption target
```

## Variants

Run in this order:

```text
EVG1A_frozenLLM_prefix_semantic_prompt
EVG1B_lora_r8_prefix_semantic_prompt
EVG1C_lora_r16_prefix_semantic_prompt
```

If time is limited, prioritize:

```text
EVG1B_lora_r8_prefix_semantic_prompt
```

Seeds:

```text
42 first
then 123 and 2025 if output is valid
```

Required outputs:

```text
outputs/vision_token_gen_evlm/EVG1/EVG1_REPORT.md
outputs/vision_token_gen_evlm/EVG1/EVG1_METRICS.csv
outputs/vision_token_gen_evlm/EVG1/EVG1_QUALITATIVE_EXAMPLES.md
outputs/vision_token_gen_evlm/EVG1/checkpoints/
```

---

# Part 4: Priority 2 Model — Q-Former / Perceiver Bridge + LoRA

Model name:

```text
EVG2_QFormerBridge_LoRA
```

## Motivation

The LLM may not understand raw enhanced visual tokens well. A Q-Former/Perceiver bridge can compress enhanced vision tokens into a small number of trainable query tokens.

## Architecture

Inputs:

```text
enhanced_visual_tokens: [B, N, 512]
eeg_tokens: [B, M, 512]
topk_text_prototypes: [B, K, 512]
```

Create learnable query tokens:

```text
query_tokens: [Q, 512]
```

Recommended:

```text
Q = 8 or 16
```

Bridge:

```text
query tokens attend to enhanced_visual_tokens
query tokens optionally attend to eeg_tokens
query tokens optionally attend to topk_text_prototypes
```

Output:

```text
multimodal_query_tokens: [B, Q, 512]
```

Project to LLM hidden size:

```text
llm_prefix_tokens: [B, Q, llm_hidden_dim]
```

Then use the same caption prompt and LLM as EVG1.

## Variants

Run:

```text
EVG2A_QFormer_visual_only
EVG2B_QFormer_visual_eeg
EVG2C_QFormer_visual_eeg_topk
EVG2D_QFormer_visual_eeg_topk_lora
```

Most important:

```text
EVG2D_QFormer_visual_eeg_topk_lora
```

Use LoRA if possible.

Required outputs:

```text
outputs/vision_token_gen_evlm/EVG2/EVG2_REPORT.md
outputs/vision_token_gen_evlm/EVG2/EVG2_METRICS.csv
outputs/vision_token_gen_evlm/EVG2/EVG2_QUALITATIVE_EXAMPLES.md
outputs/vision_token_gen_evlm/EVG2/checkpoints/
```

---

# Part 5: Priority 3 Model — LLaVA-style Projector

Model name:

```text
EVG3_LLaVAStyle_Projector
```

Try this only if compatible with reasonable engineering effort.

## Goal

Feed EEG-enhanced CLIP visual tokens into a LLaVA-style multimodal projector.

Possible path:

```text
enhanced CLIP visual tokens
→ mm_projector
→ LLM token space
→ LLM decoder
→ caption
```

Check compatibility:

```text
expected CLIP hidden dimension
expected number of visual tokens
expected layer
projector input dimension
projector output dimension
LLM hidden dimension
```

If dimensions mismatch, implement a small adapter:

```text
enhanced_visual_tokens [B, N, 512]
→ dimension adapter
→ projector expected dimension
```

Do not spend excessive time if the interface is too difficult.

## Variants

Run:

```text
EVG3A_projector_frozenLLM
EVG3B_projector_lora
EVG3C_projector_topk_prompt_lora
```

Required outputs:

```text
outputs/vision_token_gen_evlm/EVG3/EVG3_REPORT.md
outputs/vision_token_gen_evlm/EVG3/EVG3_METRICS.csv
outputs/vision_token_gen_evlm/EVG3/EVG3_QUALITATIVE_EXAMPLES.md
```

If blocked, write:

```text
outputs/vision_token_gen_evlm/EVG3/EVG3_BLOCKED_REPORT.md
```

Explain why.

---

# Part 6: Priority 4 Model — BLIP-2 / Q-Former-style Generation

Model name:

```text
EVG4_BLIP2Style_EnhancedTokens
```

Try this if BLIP-2/InstructBLIP or a similar model is already available.

## Goal

Replace or augment the visual token input to BLIP-2/Q-Former with EEG-enhanced vision tokens.

Path:

```text
enhanced vision tokens
→ Q-Former
→ language decoder
→ caption
```

If direct replacement is hard, use enhanced tokens as additional memory tokens to Q-Former.

Variants:

```text
EVG4A_BLIP2_enhanced_tokens
EVG4B_BLIP2_enhanced_tokens_plus_topk
EVG4C_BLIP2_enhanced_tokens_lora_or_adapter
```

Required outputs:

```text
outputs/vision_token_gen_evlm/EVG4/EVG4_REPORT.md
outputs/vision_token_gen_evlm/EVG4/EVG4_METRICS.csv
outputs/vision_token_gen_evlm/EVG4/EVG4_QUALITATIVE_EXAMPLES.md
```

If blocked, write:

```text
outputs/vision_token_gen_evlm/EVG4/EVG4_BLOCKED_REPORT.md
```

---

# Part 7: Priority 5 Model — Qwen-VL Internal Adapter

Model name:

```text
EVG5_QwenVL_InternalAdapter
```

This is risky. Try only after EVG1/EVG2 are done or if the repo already has Qwen-VL support.

## Goal

Inject EEG-enhanced visual features into a Qwen-VL-like visual pathway.

Potential approaches:

```text
1. Replace Qwen-VL visual embeddings after its vision encoder.
2. Add EEG adapter after the Qwen-VL visual projector.
3. Add enhanced CLIP visual tokens as additional visual memory tokens.
4. Use enhanced visual tokens as prefix tokens with Qwen-VL text prompt.
```

Do not spend all remaining time here if the model API is too closed.

Required outputs:

```text
outputs/vision_token_gen_evlm/EVG5/EVG5_REPORT.md
outputs/vision_token_gen_evlm/EVG5/EVG5_METRICS.csv
outputs/vision_token_gen_evlm/EVG5/EVG5_QUALITATIVE_EXAMPLES.md
```

If blocked, write:

```text
outputs/vision_token_gen_evlm/EVG5/EVG5_BLOCKED_REPORT.md
```

---

# Part 8: Evaluation

Evaluate all generative models on:

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

For each model, report:

```text
valid caption rate
invalid output rate
caption class-hit
caption top-k class-hit
real EEG - vision-only gap
real EEG - shuffled EEG gap
real EEG - random EEG gap
average caption length
distinct caption count
repetition rate
```

Invalid output if:

```text
empty
URL
HTML
code
markdown table
WNID-only
only class name
JSON-only
apology or explanation
too long
heavy repetition
```

Optional metrics if already installed:

```text
BLEU-1
BLEU-4
ROUGE-L
METEOR
BERTScore
CLIPScore
```

Do not waste too much time installing optional metrics. Caption class-hit, valid caption rate, and qualitative examples are more important.

---

# Part 9: Decoding

Use short caption decoding.

Default:

```yaml
max_new_tokens: 24
temperature: 0.2
top_p: 0.9
repetition_penalty: 1.1
no_repeat_ngram_size: 3
```

Also try strict decoding if outputs are bad:

```yaml
max_new_tokens: 16
temperature: 0.1
top_p: 0.8
repetition_penalty: 1.2
no_repeat_ngram_size: 3
```

Prompt must ask for one caption only:

```text
Write one short natural image caption. Do not mention EEG. Do not output JSON, code, markdown, URLs, or explanations.
```

Expected output:

```text
a man riding a bike on a rocky trail
a piano in a room
a dog standing in the grass
a bird perched on a branch
```

---

# Part 10: Qualitative Example Mining

This is mandatory.

Create:

```text
outputs/vision_token_gen_evlm/QUALITATIVE_EXAMPLES_ALL.md
outputs/vision_token_gen_evlm/BEST_REPORT_EXAMPLES.md
```

Find at least 30 qualitative examples.

Each example must include:

```text
image_id
true class
corruption
mode
model
generated caption
valid or invalid
class hit or miss
```

Best report examples should prioritize:

```text
1. vision-only wrong/vague, real EEG better/correct
2. real EEG correct, shuffled/random wrong
3. strong degradation condition
4. caption is natural and short
5. caption is not just the class name
```

Need at least 5 good report examples if possible.

If fewer than 5 exist, report honestly and include the best available cases.

---

# Part 11: Model Selection

Compare all models:

```text
Previous GRU decoder baseline if available
EVG1_TokenPrefix_SemanticPrompt_LoRA
EVG2_QFormerBridge_LoRA
EVG3_LLaVAStyle_Projector
EVG4_BLIP2Style_EnhancedTokens
EVG5_QwenVL_InternalAdapter
AutoSOTA variants
```

Selection priority:

```text
1. valid free-form caption quality
2. caption class-hit under strong degradation
3. real EEG - shuffled gap
4. real EEG - random gap
5. real EEG - vision-only gap
6. invalid output rate
7. qualitative examples for report
8. engineering reliability
```

Important:

```text
A2_final may remain the main quantitative semantic classifier.
The best generative EVLM can be used as exploratory generative result even if it does not beat A2_final classification.
```

Required files:

```text
outputs/vision_token_gen_evlm/GEN_MODEL_SELECTION.md
outputs/vision_token_gen_evlm/GEN_MODEL_SELECTION.csv
```

---

# Part 12: AutoSOTA Mode

If required models finish early, continue exploration automatically.

Do not stop while time remains.

AutoSOTA objective:

```text
Improve free-form generative EVLM caption quality and EEG-specific advantage.
```

Allowed AutoSOTA directions:

## A. Prefix and Bridge Sweeps

Try:

```text
prefix_len = 8, 16, 32
Q-Former query count = 8, 16, 32
1-layer vs 2-layer bridge
mean pooling vs attention pooling
CLS+mean pooling
```

## B. LoRA Sweeps

Try:

```text
LoRA r = 4, 8, 16
LoRA alpha = 8, 16, 32
LoRA target modules:
  q_proj/v_proj only
  q_proj/k_proj/v_proj/o_proj
  MLP layers if supported
```

Do not full-finetune LLM unless explicitly necessary and safe.

## C. Semantic Prompt Variants

Try:

```text
top1 class only
top3 class names
top5 class names
top5 + confidence
A2 top-k
VTF top-k
A2/VTF ensemble top-k
```

## D. Caption Target Variants

Try:

```text
template captions
BLIP captions
filtered BLIP captions
mixed template + BLIP
class-aware cleaned captions
```

Reject target sets with high invalid/repetitive captions.

## E. Decoding Variants

Try:

```text
greedy decoding
beam search beam=3
temperature 0.1
temperature 0.2
strict short decoding
```

Select based on valid rate and qualitative examples.

## F. Better EEG Control

Try:

```text
real-vs-shuffled generation contrastive auxiliary objective
caption class-hit auxiliary classifier
semantic consistency loss between generated caption and A2/VTF top-k
strong degradation oversampling
```

## G. Better Enhanced Tokens

If token fusion is weak, try:

```text
M = 8 EEG tokens
M = 16 EEG tokens
2-layer visual-to-EEG cross-attention
bidirectional co-attention
corruption-aware beta
entropy-based vision uncertainty beta
top1-top2 margin beta
```

Use these only if they improve generation or qualitative examples.

---

# Part 13: GPU Usage

Do not leave GPU idle.

If GPU memory usage is below 8GB for more than 10 minutes:

```text
increase batch size
increase prefix length
enable LoRA
run another seed
run Q-Former bridge
run evaluation while another job trains
```

Log usage:

```text
outputs/vision_token_gen_evlm/GPU_USAGE.md
```

---

# Part 14: Final Required Report

Create:

```text
outputs/vision_token_gen_evlm/FINAL_VISION_TOKEN_GENERATIVE_EVLM_REPORT.md
```

It must answer:

```text
1. Did we successfully feed EEG-enhanced vision tokens into a generative model?
2. Which strategy worked best?
3. Did LoRA help?
4. Did Q-Former/Perceiver bridge help?
5. Did LLaVA-style projector work or was it blocked?
6. Did BLIP-2-style generation work or was it blocked?
7. Did Qwen-VL internal adapter work or was it blocked?
8. Did real EEG improve generated captions over shuffled/random EEG?
9. What is the valid caption rate?
10. What is the invalid output rate?
11. Which examples should be used in the course report?
12. Should this be presented as the main result or exploratory EVLM result?
13. What remains the strongest quantitative model?
```

Final report must include:

```text
Best generative model:
Best checkpoint:
Best metrics file:
Best qualitative examples file:
Recommended examples for course report:
Limitations:
Next-step suggestions:
```

---

# Part 15: Completion Criteria

This goal is complete only when:

```text
1. EVG1 is trained and evaluated.
2. At least one of EVG2/EVG3/EVG4/EVG5 is attempted.
3. LoRA is attempted for at least one LLM-based variant.
4. Free-form captions are generated.
5. At least 30 qualitative examples are saved.
6. At least 5 best examples for the course report are selected if possible.
7. A final report compares all attempted strategies.
8. AutoSOTA is attempted if time remains.
```

If a strategy is blocked, write a blocked report and continue to the next strategy.

Do not mark complete if only classification was done.

Do not mark complete without generated caption examples.

Do not mark complete after smoke tests only.
