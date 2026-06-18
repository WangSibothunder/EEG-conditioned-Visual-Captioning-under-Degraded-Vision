# Deep Generative EVLM Completion Table

| item | status | train_samples | val_samples | test_samples | epochs | peak_gpu_memory | runtime | best_metric | reason_if_not_completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Route5 full adapter | completed | full | full | full | 3 | 9.137810 | 145.139402 | 0.507962 |  |
| Route5 LoRA r8 | completed | full | full | full | 3 | 6.508264 | 348.041873 | 0.584507 |  |
| Route5 LoRA r16 | completed | full | full | full | 3 | 6.562549 | 356.771943 | 0.493944 |  |
| Route5 T1 class-only target | completed | full | full | full | 3 | 6.196142 | 355.149477 | 0.576805 |  |
| Route5 T3 class+BLIP target | completed | full | full | full | 3 | 6.538196 | 350.608001 | 0.520644 |  |
| Route5 best-of-N reranking | completed | n/a | n/a | full | 0 |  |  | 0.590696 |  |
| Route1 Qwen-LoRA r8 clean target | completed | full | full | full | 3 | 8.186008 | 200.456470 | 0.198431 |  |
| Route1 Qwen-LoRA r16 clean target if attempted | completed | full | full | full | 3 | 8.214099 | 198.589001 | 0.155049 | optional after r8 |
| AutoSOTA if attempted | not_attempted |  |  |  |  |  |  |  | required work consumed this run |
