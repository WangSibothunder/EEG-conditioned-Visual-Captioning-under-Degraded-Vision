# Best-of-N Reranking Report

- source checkpoint: `outputs/deep_gen_evlm/route5_qwenvl_adapter/Route5_Qwen2VL_full_lora_r8/checkpoints`
- N: `3`
- strategies: `greedy, beam3, temp02`
- valid caption rate: `0.977466`
- strong real class-hit: `0.373560`
- real-shuffled gap: `0.110466`
- real-random gap: `0.105959`

Reranking prefers valid short captions with low repetition and semantic/class consistency.
