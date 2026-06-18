# Final Deep Generative EVLM Report

## Final Recommendation

- best model: `reranking/Route5_best_of_N_reranking`
- best checkpoint: `outputs/deep_gen_evlm/reranking`
- best metrics file: `outputs/deep_gen_evlm/ALL_DEEP_GEN_METRICS.csv`
- best qualitative examples file: `outputs/deep_gen_evlm/BEST_FINAL_REPORT_EXAMPLES.md`
- full-test: `True`
- full-train: `True`
- LoRA used: `True`
- target strategy: `reranked best checkpoint`

## Answers

1. Full-scale Route5 improved over shallow Route5: `False` (deep best score `0.590696`, shallow Route5 score `0.673385`).
2. Route5 LoRA help is shown by comparing `Route5_Qwen2VL_full_lora_r8` against `Route5_Qwen2VL_full_adapter` in `DEEP_GEN_SELECTION.csv`.
3. Route5 r16 vs r8 is shown in `DEEP_GEN_SELECTION.csv`.
4. Caption target comparison is in `caption_target_ablation/CAPTION_TARGET_REPORT.md`.
5. Best-of-N reranking is in `reranking/RERANKING_REPORT.md`.
6. Full-test real EEG > vision-only for best route: `True` with gap `0.084126`.
7. Full-test real EEG > shuffled/random for best route: `True` with gaps `0.110466` / `0.105959`.
8. Best final generative EVLM route: `reranking/Route5_best_of_N_reranking`.
9. Best course-report examples: `outputs/deep_gen_evlm/BEST_FINAL_REPORT_EXAMPLES.md`.
10. Recommendation: use A2/constrained semantic results as main quantitative evidence and Route5 as pretrained generative EVLM demonstration unless free-form examples are clearly strong enough.
11. Remaining limitations: generation still depends heavily on semantic top-k prompt and should not be presented as pure EEG-to-text mind reading.

## Example Mining

Final examples were re-mined from all full-test JSONL outputs, including best-of-N reranking, with filters for natural length, validity, low repetition, degraded conditions, and cases where real EEG class-hit beats vision/shuffled/random controls.

## Completion Table

See `outputs/deep_gen_evlm/COMPLETION_TABLE.md`.
