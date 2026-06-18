# Model Artifact Policy

This repository includes small trainable artifacts from the final EVLM experiments:

- PEFT LoRA adapters (`adapter_model.safetensors`)
- local prefix/projector modules (`prefix_projector.pt`)
- config/history files for the corresponding runs

These files are included because each file is below GitHub's 100 MB single-file limit and they are the project-specific trained components.

The repository does not include:

- Qwen/Qwen2-VL pretrained base model weights
- CLIP/BLIP/Qwen local Hugging Face caches
- raw EEG arrays from real datasets
- ImageNet/Thought2Text/THINGS/EIT raw images or archives
- large intermediate feature caches

For larger future checkpoints, prefer Hugging Face Hub or GitHub Releases/LFS rather than committing full model weights directly to the Git repository.
