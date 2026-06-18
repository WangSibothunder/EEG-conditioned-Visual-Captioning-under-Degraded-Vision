# Caption Target Ablation Report

| Variant | Target | Strong Real Class Hit | Valid Rate | Invalid Rate | Real-Shuffled Gap | Real-Random Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Route5_Qwen2VL_lora_r8_T1_class_only | T1_class_only | 0.370556 | 0.954932 | 0.045068 | 0.095944 | 0.056685 |
| Route5_Qwen2VL_lora_r8_T3_class_plus_blip | T3_class_plus_blip | 0.317476 | 0.983141 | 0.016859 | 0.064096 | 0.001302 |

## Answers

- Class-hit and naturalness should be judged from `TARGET_ABLATION_METRICS.csv` plus the qualitative examples.
- Invalid/repetitive output is tracked by `invalid_output_rate` and `repetition_rate` in the CSV.
- Final target choice is selected in `FINAL_DEEP_GEN_EVLM_REPORT.md` after comparing these rows.
