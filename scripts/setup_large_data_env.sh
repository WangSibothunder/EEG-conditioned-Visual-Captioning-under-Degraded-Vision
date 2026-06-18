#!/usr/bin/env bash
set -euo pipefail

# Source this before large dataset downloads.
export EEG_CAPTION_DATA_ROOT="${EEG_CAPTION_DATA_ROOT:-/cloud/cloud-ssd1/eeg_vision_caption_data}"
export HF_HOME="$EEG_CAPTION_DATA_ROOT/model_cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$EEG_CAPTION_DATA_ROOT/model_cache/torch"
export HF_XET_CACHE="$EEG_CAPTION_DATA_ROOT/model_cache/xet"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. Set it in your shell only if a gated Hugging Face dataset/model requires authentication."
fi

mkdir -p "$EEG_CAPTION_DATA_ROOT" "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" "$HF_XET_CACHE"

echo "EEG_CAPTION_DATA_ROOT=$EEG_CAPTION_DATA_ROOT"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "TORCH_HOME=$TORCH_HOME"
