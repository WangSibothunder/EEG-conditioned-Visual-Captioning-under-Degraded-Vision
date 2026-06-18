# Experiments

## Goal 2: Real EEG Research Prototype

Current status:

- Dummy MVP audit completed in `outputs/audit_report.md`.
- Thought2Text Google Drive is unreachable from this server.
- The provided proxy subscription endpoint did not return a usable config from this environment.
- User-provided port forward `7897` was not reachable from this container.
- Kaggle fallback dataset `prithwiserver/eeg-image-net-eegcvpr40things-image` provided EEGCVPR40-compatible EEG and split files.
- Thought2Text manifests were generated: 7970 train / 1998 val / 1997 test.
- Thought2Text image references currently resolve through valid symlinks into KaggleHub cache; this is usable locally but should be copied or relinked for portability.
- Existing Hugging Face real sample `data/real/eeg_image_cvpr_sample` is available for code-path validation.

Validated on available real sample:

```bash
python scripts/precompute_vision.py \
  --manifest data/real/eeg_image_cvpr_sample/train.jsonl \
  --out data/real/eeg_image_cvpr_sample/cache/clip_train.npy \
  --index_out data/real/eeg_image_cvpr_sample/cache/clip_index_train.json \
  --use_tiny_debug_model

python -m src.train.train_align \
  --train_manifest data/real/eeg_image_cvpr_sample/train.jsonl \
  --val_manifest data/real/eeg_image_cvpr_sample/validation.jsonl \
  --train_cache data/real/eeg_image_cvpr_sample/cache/clip_train.npy \
  --val_cache data/real/eeg_image_cvpr_sample/cache/clip_val.npy \
  --train_index data/real/eeg_image_cvpr_sample/cache/clip_index_train.json \
  --val_index data/real/eeg_image_cvpr_sample/cache/clip_index_val.json \
  --out_dir outputs/align_real_debug \
  --epochs 1 \
  --batch_size 8 \
  --max_train_samples 32 \
  --max_val_samples 16
```

Preliminary tiny-debug result:

- alignment loss: `3.9269`
- validation R@1: `0.0625`
- validation R@5: `0.3125`

Do not interpret this as a scientific result because the vision cache used the tiny debug vision encoder.

## Heavy Stage — 2026-06-17

| ID | Config | Data | Model | Mode | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| A2_temporal_spectral_spatial_full | `configs/heavy_architectures/A2_temporal_spectral_spatial_full.yaml` | Thought2Text | Temporal-Spectral-Spatial Transformer | EEG->CLIP alignment | completed | R@5 `0.2943`, below P2 `0.3574`. |
| A2_temporal_spectral_spatial_seed6161_full | `configs/heavy_architectures/A2_temporal_spectral_spatial_seed6161_full.yaml` | Thought2Text | Temporal-Spectral-Spatial Transformer | stability alignment | running | Launched after GPU idle diagnosis. |
| semantic_fusion_A2_temporal_spectral_spatial_full | direct CLI | Thought2Text | A2 checkpoint + semantic classifier | controlled semantic fusion | running | Full 80-epoch run, not smoke. |
| semantic_fusion_controlled_multiseed | direct CLI | Thought2Text test | P2 checkpoint + semantic classifier | strong degradation eval | completed | Real EEG beats shuffled/random controls `18/18`; beats vision-only `11/18`. |
