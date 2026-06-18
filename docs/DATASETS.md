# Datasets

Day1-Day2 uses generated dummy data. Real-data migration starts with the smallest useful EEG+image source that can support the same manifest contract.

## Current Download Policy

Core cache paths are repository-local:

```bash
source scripts/setup_cache_env.sh
```

Current near-term priority:
- Must cache: `openai/clip-vit-base-patch32`, `Qwen/Qwen2.5-1.5B-Instruct`, `Salesforce/blip-image-captioning-base`.
- Must inspect: Thought2Text / CVPR2017 processed EEG data.
- Strongly suggested after core models: EIT-1M partial release and THINGS-EEG2, subject to disk and network budget.
- Optional later: `openai/clip-vit-large-patch14`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct`.
- Do not block the main path on EEG-ImageNet or full ImageNet.

## Priority 1: Thought2Text

Source:
- Repository: `https://github.com/abhijitmishra/Thought2Text`
- Data folder: `https://drive.google.com/drive/folders/1XqV6MMl28iYXkQBMEFHfEXllGmCbqpOu`

Why first:
- It is directly aligned with EEG/image/text experiments.
- It is much smaller and faster to inspect than large raw EEG/image corpora.
- It can bootstrap the real-data manifest adapter before we spend GPU time.

Usage restriction from the upstream README:
- Academic use only.
- Cite the Thought2Text paper and dataset paper when using it.

Download command:

```bash
python -m pip install gdown
python -m gdown --folder 'https://drive.google.com/drive/folders/1XqV6MMl28iYXkQBMEFHfEXllGmCbqpOu' -O data/raw/thought2text
```

Current environment note:
- `drive.google.com` was unreachable from this server on 2026-06-15, so the first real-data download was switched to the Hugging Face CVPR EEG/image dataset below.
- Local `data/thought2text/block/eeg_5_95_std.pth` and `data/thought2text/block/block_splits_by_image_all.pth` exist from a Kaggle fallback.
- Local `data/thought2text/images/*.jpg` currently resolves through valid symlinks into KaggleHub cache. This is usable on the current machine but not portable unless the symlink targets are relinked or copied.
- The official instructions commonly reference `eeg_55_95_std.pth`; the local fallback file is named `eeg_5_95_std.pth`. Keep this naming difference explicit in reports and conversion scripts.

## Priority 2: EEG Image CVPR All Subjects

Source:
- Hugging Face dataset: `https://huggingface.co/datasets/luigi-s/EEG_Image_CVPR_ALL_subj`

Why now:
- Around 1.71 GB, much smaller than THINGS-EEG2.
- Includes EEG/image/caption-style fields according to the dataset card.
- Stored as parquet shards, which are easy to inspect and convert to this repo's manifest format.

Download command:

```bash
python -m pip install huggingface_hub
python scripts/download_hf_dataset.py --repo luigi-s/EEG_Image_CVPR_ALL_subj --dest data/raw/eeg_image_cvpr_all_subj
```

Create a small manifest-compatible sample:

```bash
python -m pip install pyarrow pandas
python scripts/convert_eeg_image_cvpr.py \
  --raw-dir data/raw/eeg_image_cvpr_all_subj \
  --out-dir data/real/eeg_image_cvpr_sample \
  --train-limit 256 \
  --val-limit 64 \
  --test-limit 64
```

Debug-train on the converted sample:

```bash
python -m src.train.train_baseline --config configs/real_debug.yaml
python -m src.train.train_fusion --config configs/real_debug.yaml
```

## Priority 3: THINGS-EEG2

Source:
- Hugging Face dataset: `https://huggingface.co/datasets/AutoLab/THINGS-EEG2`
- OSF project: `3jk45`

Notes:
- The dataset card reports about 27.8 GB.
- Use only after Thought2Text format inspection is complete.
- Do not duplicate or augment the image dataset in-place; create manifests pointing at raw files.
- If using OSF, run `osf -p 3jk45 clone THINGS-EEG2` from `data/` and write partial status if the download is slow or too large.

## Priority 4: EIT-1M Partial

Source:
- Hugging Face dataset: `https://huggingface.co/datasets/eit-1m/EIT-1M`

Why useful:
- It is an EEG-image-text style dataset release, which is close to the target interface for this project.
- The public Hugging Face release is partial, so do not assume the full paper-scale dataset is available locally.

Download command:

```bash
source scripts/setup_cache_env.sh
hf download eit-1m/EIT-1M --repo-type dataset --local-dir data/EIT-1M
```

Report requirements:
- total local size,
- file list,
- whether obvious EEG, image, and text/caption fields exist,
- whether the available files can be converted into this repository's manifest contract.

## Priority 5: EEG-ImageNet / CVPR2017 Visual EEG

Notes:
- EEG-ImageNet data is split into `EEG-ImageNet_1.pth` and `EEG-ImageNet_2.pth`.
- It has EEG-image pairs from 16 participants.
- Due to ImageNet copyright restrictions, it provides ImageNet file index and WNID references rather than original images.
- It is optional for later and is not part of the current Day2-Day3 main path.
- Do not download full ImageNet for this project unless explicitly requested.
