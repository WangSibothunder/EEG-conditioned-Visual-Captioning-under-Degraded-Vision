# Deep Generative EVLM Selection

| real_strong_class_hit | valid_caption_rate | invalid_output_rate | real_minus_vision | real_minus_shuffled | real_minus_random | score | route | variant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.373560 | 0.977466 | 0.022534 | 0.084126 | 0.110466 | 0.105959 | 0.590696 | reranking | Route5_best_of_N_reranking |
| 0.382574 | 0.918878 | 0.081122 | 0.091037 | 0.103155 | 0.078418 | 0.584507 | route5_qwenvl_adapter | Route5_Qwen2VL_full_lora_r8 |
| 0.370556 | 0.954932 | 0.045068 | 0.084927 | 0.095944 | 0.056685 | 0.576805 | route5_qwenvl_adapter | Route5_Qwen2VL_lora_r8_T1_class_only |
| 0.317476 | 0.983141 | 0.016859 | 0.072809 | 0.064096 | 0.001302 | 0.520644 | route5_qwenvl_adapter | Route5_Qwen2VL_lora_r8_T3_class_plus_blip |
| 0.306560 | 0.921382 | 0.078618 | 0.067501 | 0.083425 | 0.087832 | 0.507962 | route5_qwenvl_adapter | Route5_Qwen2VL_full_adapter |
| 0.310866 | 0.855700 | 0.144300 | 0.065899 | 0.072409 | 0.046970 | 0.493944 | route5_qwenvl_adapter | Route5_Qwen2VL_full_lora_r16 |
| 0.089334 | 0.517944 | 0.482056 | 0.009414 | 0.030846 | 0.024236 | 0.198431 | route1_qwen_inputs_embeds | Route1_QwenLoRA_r8_full_clean_target |
| 0.060791 | 0.456268 | 0.543732 | 0.006710 | 0.022233 | 0.007812 | 0.155049 | route1_qwen_inputs_embeds | Route1_QwenLoRA_r16_full_clean_target |
