# Deep Generative EVLM Artifacts

This directory stores compact artifacts from the final full-scale generative EVLM stage.

## Included

- `reranking/`: full-test generated captions for 6 corruptions x 5 modes, plus reranking metrics/report.
- `checkpoints/Route5_Qwen2VL_full_lora_r8/`: best Route5 source checkpoint used for best-of-N reranking.
- `checkpoints/Route5_Qwen2VL_lora_r8_T1_class_only/`: class-only target ablation checkpoint.
- `checkpoints/Route5_Qwen2VL_full_lora_r16/`: r16 comparison checkpoint.
- `checkpoints/Route1_QwenLoRA_r8_full_clean_target/`: Route1 clean-target comparison checkpoint.

## Not Included

The pretrained base models are not included. Download them separately:

- `Qwen/Qwen2-VL-2B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`

The checkpoint folders contain only trainable prefix/projector weights and PEFT LoRA adapters.
