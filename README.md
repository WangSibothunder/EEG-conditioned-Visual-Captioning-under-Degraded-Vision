# EEG-conditioned Visual Captioning under Degraded Vision

Research prototype for testing whether paired EEG can provide auxiliary semantic information for image captioning when visual input is degraded.

The project is framed as:

```text
degraded image + synchronized EEG -> semantic visual tokens -> pretrained generator -> caption
```

It is not an EEG-only mind-reading system. The main scientific claim is evaluated against vision-only, shuffled-EEG, and random-EEG controls.

## What Is Included

- `src/`: data loading, EEG encoders, fusion/alignment models, evaluation utilities.
- `scripts/`: dataset preparation, alignment sweeps, semantic evaluation, EVLM generation, GPU scheduling utilities.
- `configs/`: experiment configs used during the project.
- `tests/`: unit and smoke tests for the core pipelines.
- `docs/`: project documentation, runbook, interface contracts, status notes.
- `goal/`: execution goals/specifications used during the research process.
- `results/deep_gen_evlm/`: compact final reports and metrics.
- `artifacts/deep_gen_evlm/`: trainable adapter artifacts and full reranking generation outputs from the final generative EVLM stage.

The repository intentionally does not include raw EEG/image datasets, pretrained Qwen/Qwen2-VL weights, or large intermediate caches.

## Final Generative EVLM Result

The final full-scale run selected:

```text
Route5 best-of-N reranking
source checkpoint: Route5_Qwen2VL_full_lora_r8
base model: Qwen/Qwen2-VL-2B-Instruct
```

Full-test metrics from `results/deep_gen_evlm/DEEP_GEN_SELECTION.csv`:

| Route | Strong real class-hit | Valid caption rate | Real - vision | Real - shuffled | Real - random |
| --- | ---: | ---: | ---: | ---: | ---: |
| Route5 best-of-N reranking | 0.3736 | 0.9775 | +0.0841 | +0.1105 | +0.1060 |

Important caveat: the full-scale deep generative route did not exceed an earlier shallow Route5 score, so it should be presented as a pretrained generative EVLM demonstration. The stronger quantitative evidence should come from constrained semantic / A2-style results where applicable.

Key files:

- Final report: `results/deep_gen_evlm/FINAL_DEEP_GEN_EVLM_REPORT.md`
- Completion table: `results/deep_gen_evlm/COMPLETION_TABLE.md`
- Full metrics: `results/deep_gen_evlm/ALL_DEEP_GEN_METRICS.csv`
- Best examples: `results/deep_gen_evlm/BEST_FINAL_REPORT_EXAMPLES.md`
- 50 qualitative examples: `results/deep_gen_evlm/QUALITATIVE_EXAMPLES_50.md`
- Reranking outputs and metrics: `artifacts/deep_gen_evlm/reranking/`

## Included Model Artifacts

The repository includes only trainable adapter artifacts, not the pretrained base models:

```text
artifacts/deep_gen_evlm/checkpoints/
  Route5_Qwen2VL_full_lora_r8/
  Route5_Qwen2VL_lora_r8_T1_class_only/
  Route5_Qwen2VL_full_lora_r16/
  Route1_QwenLoRA_r8_full_clean_target/
```

Each checkpoint directory contains:

- `prefix_projector.pt`
- optional `lora_adapter/adapter_model.safetensors`
- `lora_adapter/adapter_config.json`
- `config.json`
- `history.json`

To use them, install the dependencies and separately download the matching base model, such as `Qwen/Qwen2-VL-2B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct`, from Hugging Face.

## Quick Start

Create an environment with Python 3.10+ or 3.12 and install:

```bash
pip install -r requirements.txt
pip install -U peft accelerate safetensors einops timm sentencepiece
```

Run the core strict EVLM tests:

```bash
python -m unittest tests/test_strict_vlm_gen.py -v
```

Run a dummy smoke pipeline:

```bash
bash scripts/run_smoke.sh
```

For full real-data training/evaluation commands, see:

- `docs/RUNBOOK.md`
- `results/deep_gen_evlm/FINAL_DEEP_GEN_EVLM_REPORT.md`

## Data

The code expects manifests that follow the schema in `docs/INTERFACE_CONTRACTS.md`. Raw datasets are not included:

- Thought2Text / CVPR2017 EEG visual data
- EEG-ImageNet
- THINGS-EEG2 / EIT-style datasets if available locally

Do not commit raw datasets or pretrained model caches. Use `data/`, `outputs/`, and local Hugging Face caches outside version control.

## Reproducibility Notes

This repository captures the project code and the final compact artifacts/results. Reproducing the exact full training requires:

- local dataset manifests and cached CLIP/Qwen features,
- Hugging Face access to base language/vision-language models,
- a CUDA GPU with sufficient memory,
- the run commands and configs under `scripts/`, `configs/`, and `docs/RUNBOOK.md`.

## Scientific Framing

Supported framing:

```text
Paired EEG provides measurable auxiliary semantic signal under degraded visual inputs compared with shuffled/random EEG controls in this prototype.
```

Avoid overclaiming:

```text
The system reads thoughts directly.
```
