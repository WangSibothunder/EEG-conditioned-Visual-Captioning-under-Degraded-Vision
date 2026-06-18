# Runbook

Last updated: 2026-06-15

Run commands from the repository root:

```bash
cd /workspace
```

Day1-Day2 are only the first part of the global plan. These commands cover the debug dummy-data pipeline, baseline training, minimal EEG fusion training, generation, and sanity checks.

## Environment

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Use repository-local model/data caches:

```bash
source scripts/setup_cache_env.sh
```

Use the debug config:

```bash
CONFIG=configs/debug.yaml
```

## Download And Verify Required Caches

Install downloader tools:

```bash
python -m pip install -U huggingface_hub hf_transfer gdown osfclient
```

Download core Hugging Face models:

```bash
source scripts/setup_cache_env.sh
hf download openai/clip-vit-base-patch32 --local-dir data/model_cache/openai_clip-vit-base-patch32
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir data/model_cache/Qwen2.5-1.5B-Instruct
hf download Salesforce/blip-image-captioning-base --local-dir data/model_cache/blip-image-captioning-base
```

If the BLIP full download stalls after TensorFlow files, fetch the PyTorch weight explicitly:

```bash
hf download Salesforce/blip-image-captioning-base \
  --local-dir data/model_cache/blip-image-captioning-base \
  pytorch_model.bin config.json preprocessor_config.json tokenizer.json tokenizer_config.json vocab.txt special_tokens_map.json
```

Verify local model loading:

```bash
python - <<'PY'
from pathlib import Path
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

items = [
    ("openai/clip-vit-base-patch32", Path("data/model_cache/openai_clip-vit-base-patch32"), "processor"),
    ("Qwen/Qwen2.5-1.5B-Instruct", Path("data/model_cache/Qwen2.5-1.5B-Instruct"), "tokenizer"),
    ("Salesforce/blip-image-captioning-base", Path("data/model_cache/blip-image-captioning-base"), "processor"),
]
for name, path, kind in items:
    if kind == "processor":
        AutoProcessor.from_pretrained(path, local_files_only=True)
    else:
        AutoTokenizer.from_pretrained(path, local_files_only=True)
        AutoConfig.from_pretrained(path, local_files_only=True)
print("core model cache ok")
PY
```

Download EIT-1M partial:

```bash
source scripts/setup_cache_env.sh
hf download eit-1m/EIT-1M --repo-type dataset --local-dir data/EIT-1M
```

Inspect download reports:

```bash
cat outputs/download_reports/DOWNLOAD_SUMMARY.md
cat outputs/download_reports/model_cache_report.md
cat outputs/download_reports/thought2text_data_report.md
cat outputs/download_reports/eit1m_download_report.md
cat outputs/download_reports/things_eeg2_download_report.md
```

## Create Dummy Data

Exact command:

```bash
python scripts/make_dummy_data.py --config configs/debug.yaml --num-train 8 --num-val 4
```

Expected outputs:

```text
data/images/*.jpg
data/eeg/*.npy
data/train.jsonl
data/val.jsonl
```

## Train Image-Only Baseline

Exact command for only baseline training after data exists:

```bash
python -m src.train.train_baseline --config configs/debug.yaml
```

Wrapper command that creates dummy data first, then trains baseline:

```bash
bash scripts/run_baseline.sh
```

Expected checkpoint path:

```text
outputs/debug/baseline/checkpoint_last.pt
```

## Train EEG+Vision Fusion

Exact command for only fusion training after data exists:

```bash
python -m src.train.train_fusion --config configs/debug.yaml
```

Wrapper command that creates dummy data first, then trains fusion:

```bash
bash scripts/run_fusion.sh
```

Expected checkpoint path:

```text
outputs/debug/fusion/checkpoint_last.pt
```

## Run Generation

Use the fusion checkpoint by default:

```bash
CHECKPOINT=outputs/debug/fusion/checkpoint_last.pt
```

Exact command used by the current wrapper:

```bash
bash scripts/run_generate.sh
```

Equivalent direct command:

```bash
python -m src.eval.generate --config configs/debug.yaml --checkpoint outputs/debug/fusion/checkpoint_last.pt --mode real_eeg
```

Run all required generation modes:

```bash
python -m src.eval.generate --config configs/debug.yaml --checkpoint outputs/debug/fusion/checkpoint_last.pt --mode image_only
python -m src.eval.generate --config configs/debug.yaml --checkpoint outputs/debug/fusion/checkpoint_last.pt --mode real_eeg
python -m src.eval.generate --config configs/debug.yaml --checkpoint outputs/debug/fusion/checkpoint_last.pt --mode shuffled_eeg
python -m src.eval.generate --config configs/debug.yaml --checkpoint outputs/debug/fusion/checkpoint_last.pt --mode random_eeg
```

Expected output format is JSONL records:

```json
{"image_id":"000001","mode":"real_eeg","reference":"a photo of a red object","prediction":"a red object is shown"}
```

## Run Sanity Check

Exact command used by the current wrapper:

```bash
bash scripts/run_sanity.sh
```

Equivalent direct command:

```bash
python -m src.eval.sanity_check --config configs/debug.yaml --checkpoint outputs/debug/fusion/checkpoint_last.pt
```

The sanity check must cover:

```text
image_only
real_eeg
shuffled_eeg
random_eeg
```

## Feishu Hourly Report

Set the webhook locally without committing it:

```bash
export FEISHU_WEBHOOK='https://open.feishu.cn/open-apis/bot/v2/hook/...'
```

Or write it to `.env.feishu` locally:

```bash
printf 'FEISHU_WEBHOOK=%s\n' 'https://open.feishu.cn/open-apis/bot/v2/hook/...' > .env.feishu
```

Send one report immediately:

```bash
bash scripts/send_feishu_report.sh
```

Start the hourly reporter:

```bash
bash scripts/run_hourly_feishu_report.sh
```

Start it in the background and keep the PID:

```bash
bash scripts/start_hourly_feishu_report.sh
```

Stop the background reporter:

```bash
bash scripts/stop_hourly_feishu_report.sh
```

Capture terminal output so the report can include the latest 5 lines:

```bash
bash scripts/run_baseline.sh
bash scripts/run_fusion.sh
bash scripts/run_generate.sh
bash scripts/run_sanity.sh
```

These scripts append to `outputs/monitor/terminal.log`.

## Full Day1-Day2 Debug Sequence

Run after all owning agents have implemented their code paths:

```bash
python scripts/make_dummy_data.py --config configs/debug.yaml --num-train 8 --num-val 4
python -m src.train.train_baseline --config configs/debug.yaml
python -m src.train.train_fusion --config configs/debug.yaml
python -m src.eval.generate --config configs/debug.yaml --checkpoint outputs/debug/fusion/checkpoint_last.pt --mode image_only
python -m src.eval.generate --config configs/debug.yaml --checkpoint outputs/debug/fusion/checkpoint_last.pt --mode real_eeg
python -m src.eval.generate --config configs/debug.yaml --checkpoint outputs/debug/fusion/checkpoint_last.pt --mode shuffled_eeg
python -m src.eval.generate --config configs/debug.yaml --checkpoint outputs/debug/fusion/checkpoint_last.pt --mode random_eeg
python -m src.eval.sanity_check --config configs/debug.yaml --checkpoint outputs/debug/fusion/checkpoint_last.pt
```

If a command fails because a referenced Python module does not exist yet, hand it to the owning code agent. The documentation agent should not edit code files.

## Goal 2 Real-Data Commands

Audit the previous MVP:

```bash
cat outputs/audit_report.md
```

Inspect Thought2Text data:

```bash
bash scripts/inspect_thought2text.sh
```

Build Thought2Text manifests:

```bash
bash scripts/build_thought2text_manifest.sh
```

Smoke-test EEG loading when ImageNet images are not available locally:

```bash
python -m src.data.dataset --manifest data/thought2text/train.jsonl --smoke_test --allow_missing_images
```

This command uses a zero image placeholder only for the explicit smoke test. Normal dataset loading still requires real image files.

Current local status: `data/thought2text/images/*.jpg` resolves through symlinks into KaggleHub cache. Use the missing-image fallback only if those links are absent or broken.

If CLIP cache generation fails because images are missing, inspect:

```bash
cat outputs/missing_vision_images.md
```

If you have the matching ImageNet files in another directory, link them into the Thought2Text layout first:

```bash
bash scripts/link_thought2text_images.sh \
  --source-root /path/to/imagenet \
  --mode symlink
```

Use `--mode copy` instead if the source directory will not remain mounted.

### EEG-ImageNet Image Linking

EEG-ImageNet EEG trials are usable for EEG-only masked pretraining as soon as
`data/EEG-ImageNet/train.jsonl` and `data/EEG-ImageNet/cache/eeg_pretrain_train.npy`
exist. Image+EEG training requires the separate ImageNet JPEG tree.

After the Kaggle ImageNet CLS-LOC download and nested extraction finish, rewrite
the manifest image paths:

```bash
python scripts/link_eeg_imagenet_images.py \
  --manifest data/EEG-ImageNet/train.jsonl \
  --image-root /cloud/cloud-ssd1/eeg_vision_caption_data/ImageNet/kaggle_cls_loc/extracted \
  --out data/EEG-ImageNet/train_image_linked.jsonl \
  --report outputs/datasets/EEG_IMAGENET_IMAGE_LINK_REPORT.md \
  --relative-to /workspace
```

Repeat the same command for `val.jsonl` and `test.jsonl`, changing `--out` to
`val_image_linked.jsonl` and `test_image_linked.jsonl`.

The linker searches standard Kaggle CLS-LOC paths such as:

```text
ILSVRC/Data/CLS-LOC/train/<wnid>/<image_id>.JPEG
```

If the report says `Matched rows: 0`, ImageNet extraction is still incomplete or
the `--image-root` path is wrong.

Precompute vision features:

```bash
python scripts/precompute_vision.py \
  --manifest data/thought2text/train.jsonl \
  --image_root data/thought2text \
  --out data/thought2text/cache/clip_train.npy \
  --index_out data/thought2text/cache/clip_index_train.json
```

This requires actual files under `data/thought2text/images/`; see `outputs/missing_thought2text_images.md` if it fails on missing images.

Run EEG-to-CLIP alignment:

```bash
bash scripts/run_align.sh --max_train_samples 1024 --max_val_samples 256
```

Evaluate sanity JSONL files:

```bash
bash scripts/evaluate_sanity.sh
```
