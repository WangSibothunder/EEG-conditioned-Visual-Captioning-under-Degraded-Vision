# Vision Token Generative EVLM Execution Plan

Goal: execute `goal/VISION_TOKEN_GEN_EVLM.md` end to end, with evidence that `enhanced_visual_tokens [B,N,512]` are fed into a generative model and free-form captions are produced.

## Plan

1. Reuse prior token-fusion artifacts from `outputs/token_generative_evlm` only as inputs.
2. Add tests that fail unless the new generator accepts and uses token sequences rather than pooled embeddings.
3. Implement `scripts/run_vision_token_gen_evlm.py` with:
   - EVG1 direct token-prefix semantic-prompt generator.
   - EVG2 Q-Former-style bridge variant.
   - feature builder that extracts `aux["enhanced_tokens"]` from VTF.
   - reports under `outputs/vision_token_gen_evlm`.
4. Run EVG1 on real train/val/test data across required modes and corruptions.
5. Attempt EVG2 after EVG1. If Qwen LoRA remains blocked by missing `peft` or unstable prior Qwen behavior, document the blocker and continue.
6. Verify required artifacts and tests before claiming completion.

## Completion Checks

- `outputs/vision_token_gen_evlm/prep/ENHANCED_TOKEN_SOURCE.md`
- `outputs/vision_token_gen_evlm/caption_targets/CAPTION_TARGET_REPORT.md`
- `outputs/vision_token_gen_evlm/EVG1/EVG1_REPORT.md`
- `outputs/vision_token_gen_evlm/EVG1/EVG1_METRICS.csv`
- `outputs/vision_token_gen_evlm/EVG1/EVG1_QUALITATIVE_EXAMPLES.md`
- at least one EVG2/EVG3/EVG4/EVG5 attempted or blocked report
- `outputs/vision_token_gen_evlm/FINAL_VISION_TOKEN_GENERATIVE_EVLM_REPORT.md`
