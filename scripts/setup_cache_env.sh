#!/usr/bin/env bash
set -euo pipefail

# Source this file from the repository root before downloading models.
export HF_HOME="$PWD/data/model_cache/huggingface"
export TRANSFORMERS_CACHE="$PWD/data/model_cache/huggingface"
export HF_HUB_CACHE="$PWD/data/model_cache/huggingface/hub"
export TORCH_HOME="$PWD/data/model_cache/torch"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME"

echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "TORCH_HOME=$TORCH_HOME"
