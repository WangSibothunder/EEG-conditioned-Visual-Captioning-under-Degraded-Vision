# Status

Last updated: 2026-06-17

## 2026-06-17 Heavy Stage A2/SigLIP Active Follow-up Refresh

Done:
- Refreshed `outputs/heavy_stage/MASTER_REPORT.md` from current artifacts.
- Fixed `scripts/materialize_heavy_stage_reports.py` so active M1 status says the M1 Thought2Text transfer watcher is waiting for the formal M1 report/checkpoint, instead of saying a new transfer launch still needs to be manually started.
- Updated regression coverage in `tests/test_heavy_stage_materializer.py`.
- Added a conservative recovery fallback to `scripts/launch_transfer_after_pretrain_artifact.py`: if a monitored pretraining PID exits after producing metrics and a nonempty best checkpoint but without writing the formal report, the watcher can write a recovered report and unblock transfer. It does nothing while the training PID is alive.
- M1 THINGS raw-window pretraining and its Thought2Text transfer completed; transfer R@5 was `0.1201`, so it is preserved as a weak/negative result.
- EEG-ImageNet exact A4 scratch alignment completed with near-random R@5 `0.0036`; it is preserved as a negative result.
- Fixed `scripts/precompute_vision.py` so `google/siglip-base-patch16-224` uses native `SiglipVisionModel` instead of attempting a CLIPVisionModel load.
- Restarted `CLIP_ADAPTER_SIGLIP_PROTOTYPE_CALIBRATION`; it completed on CUDA after the SigLIP loader fix.
- Verified SigLIP cache shapes: train `[7970, 768]`, val `[1998, 768]`, dtype `float16`.
- Fixed `scripts/materialize_heavy_stage_reports.py` so completed THINGS M0/M1/M2 transfer status no longer leaves stale "queued A4/A2/SigLIP" language after A4 and SigLIP are complete.
- Refreshed `outputs/heavy_stage/MASTER_REPORT.md`; it now states that the only remaining exact-linked follow-up is the running A2 scratch tri-modal job.
- EEG-ImageNet exact A2 scratch tri-modal completed and wrote final artifacts:
  - `outputs/trimodal/eeg_imagenet_exact_a2_scratch_full/TRIMODAL_FULL_REPORT.md`
  - `outputs/trimodal/eeg_imagenet_exact_a2_scratch_full/trimodal_metrics.json`
- A2 exact scratch result: image R@1/R@5/R@10 `0.0099 / 0.0406 / 0.0863`, text R@1/R@5/R@10 `0.0080 / 0.0410 / 0.0811`, class accuracy `0.2239`.
- Updated scheduler queue synchronization so completed `running` jobs are written back to `configs/heavy_stage_queue.yaml`.
- Refreshed final heavy-stage reports; `outputs/heavy_stage/MASTER_REPORT.md` now reports `complete_with_mixed_results`.

In Progress:
- None in the heavy-stage queue. The queue is exhausted: `53 completed`, `2 blocked`, `1 partial_ready`, `1 skipped`.

Verification:
- `python -m unittest tests.test_heavy_stage_materializer tests.test_heavy_stage_scheduler tests.test_heavy_stage_scheduler_loop -v`
- `python -m py_compile scripts/materialize_heavy_stage_reports.py scripts/heavy_stage_scheduler.py`
- `python -m unittest tests.test_heavy_stage_materializer.HeavyStageMaterializerTests.test_master_report_names_active_m1_after_m0_m2_transfers_complete -v`
- `python -m unittest tests.test_heavy_stage_materializer -v`
- `python -m unittest tests.test_heavy_stage_scheduler tests.test_heavy_stage_scheduler_loop tests.test_launch_semantic_eval_after_transfer tests.test_final_semantic_materializer tests.test_heavy_stage_materializer -v`
- `python -m py_compile scripts/materialize_heavy_stage_reports.py`
- `python -m unittest tests.test_launch_transfer_after_pretrain_artifact tests.test_heavy_stage_scheduler tests.test_heavy_stage_scheduler_loop -v`
- `python -m py_compile scripts/launch_transfer_after_pretrain_artifact.py`
- `python -m unittest tests.test_precompute_vision -v`
- `python -m py_compile scripts/precompute_vision.py`

Notes:
- `git` is not installed in this runtime, so file status was not available through `git status`.
- The heavy-stage execution is complete with mixed results: constrained semantic EEG evidence is positive, but large-data pretraining/transfer and exact-linked EEG-ImageNet follow-ups did not beat the best Thought2Text alignment.
- The GPU is currently idle because no approved queued/runnable heavy-stage GPU job remains; launching more would require a new research objective.

Day1-Day2 are only the first part of the global plan. This status file tracks the current two-day slice and should be updated after each milestone.

## Done

- Project instructions reviewed from `AGENT.md`.
- Global Day1-Day2 goal reviewed from `goal/day1-2goal.md`.
- Documentation scope initialized for project brief, roadmap, contracts, decisions, status, and runbook.
- Phase 0 project skeleton created.
- Documentation files created:
  - `docs/PROJECT_BRIEF.md`
  - `docs/GLOBAL_ROADMAP.md`
  - `docs/INTERFACE_CONTRACTS.md`
  - `docs/DECISIONS.md`
  - `docs/STATUS.md`
  - `docs/RUNBOOK.md`
- Dummy data generator, dataset loader, and collate function implemented.
- Image-only baseline model and training loop implemented.
- EEG encoder, gated fusion, and fusion training loop implemented.
- Generation and four-mode sanity check implemented.
- Shell entrypoints verified:
  - `scripts/run_baseline.sh`
  - `scripts/run_fusion.sh`
  - `scripts/run_generate.sh`
  - `scripts/run_sanity.sh`
- Current debug config identified at `configs/debug.yaml`.
- Local verification completed:
  - `python scripts/make_dummy_data.py --config configs/debug.yaml --num-train 4 --num-val 2 --smoke-test`
  - model shape/forward smoke test
  - `bash scripts/run_baseline.sh`
  - `bash scripts/run_fusion.sh`
  - `bash scripts/run_generate.sh`
  - `bash scripts/run_sanity.sh`
- Independent QA worker re-ran the full debug sequence and confirmed all requested commands exited `0`.
- First real dataset downloaded:
  - `data/raw/eeg_image_cvpr_all_subj`
  - Source: Hugging Face `luigi-s/EEG_Image_CVPR_ALL_subj`
  - Local size: about 1.6 GB
- Real-data sample conversion implemented and verified:
  - `scripts/convert_eeg_image_cvpr.py`
  - `data/real/eeg_image_cvpr_sample`
  - Current sample size: 256 train / 64 validation / 64 test
  - loader returns `image=(B, 3, 224, 224)` and `eeg=(B, 64, 250)`
- Real-data debug training verified:
  - `python -m src.train.train_baseline --config configs/real_debug.yaml`
  - `python -m src.train.train_fusion --config configs/real_debug.yaml`
- Feishu hourly reporting support added:
  - card payload builder with Chinese labels
  - one-shot sender script
  - background start script for hourly reporting
  - hourly sender script
  - stop script for the background reporter
  - terminal tail capture via `outputs/monitor/terminal.log`
  - run scripts now append output to terminal log
- Feishu hourly background reporter started with PID file at `outputs/monitor/feishu_reporter.pid`.
- Goal 2 audit report created at `outputs/audit_report.md`.
- Thought2Text missing-data scripts added:
  - `scripts/inspect_thought2text.sh`
  - `scripts/build_thought2text_manifest.sh`
  - `src/data/inspect_thought2text.py`
  - `src/data/build_thought2text_manifest.py`
- Dataset loader now supports `.pth` EEG via `eeg_index`.
- CLIP cache and EEG-to-CLIP alignment skeleton implemented:
  - `scripts/precompute_vision.py`
  - `src/data/clip_cache.py`
  - `src/models/alignment_model.py`
  - `src/train/train_align.py`
  - `src/eval/retrieval.py`
  - `scripts/run_align.sh`
- Degradation and metrics skeleton implemented:
  - `src/data/corruptions.py`
  - `src/eval/metrics.py`
  - `scripts/evaluate_sanity.sh`
- Real-sample validation completed using `data/real/eeg_image_cvpr_sample`:
  - tiny vision cache generated
  - one-epoch EEG-to-CLIP alignment smoke run completed
  - `outputs/alignment_report.md` and `outputs/alignment_metrics.json` created
- Kaggle fallback Thought2Text/EEGCVPR40 files downloaded and inspected:
  - `data/thought2text/block/eeg_5_95_std.pth` (3.13 GB)
  - `data/thought2text/block/block_splits_by_image_all.pth`
  - inspection report: `outputs/thought2text_inspection.md`
- Thought2Text manifests generated from the real `.pth` schema:
  - `data/thought2text/train.jsonl` (7970 rows)
  - `data/thought2text/val.jsonl` (1998 rows)
  - `data/thought2text/test.jsonl` (1997 rows)
- `EEGVisionCaptionDataset` now supports Thought2Text `.pth` objects whose root schema is `{"dataset": [...]}`.
- EEG-only Thought2Text loader smoke test passed with explicit missing-image fallback:
  - `python -m src.data.dataset --manifest data/thought2text/train.jsonl --smoke_test --allow_missing_images`
- Subtask reports created:
  - `outputs/data_adapter_report.md`
  - `outputs/vision_cache_report.md`
  - `outputs/fusion_report.md`
  - `outputs/robustness_report.md`
  - `outputs/research_status_report.md`
- Thought2Text CLIP cache command now fails early with a clear missing-image report instead of a DataLoader traceback:
  - `outputs/missing_vision_images.md`
- Download/cache task completed for core local assets:
  - `scripts/setup_cache_env.sh`
  - `data/model_cache/openai_clip-vit-base-patch32`
  - `data/model_cache/Qwen2.5-1.5B-Instruct`
  - `data/model_cache/blip-image-captioning-base`
  - verification report: `outputs/download_reports/model_cache_report.md`
- Thought2Text local data re-verified:
  - `data/thought2text/block/eeg_5_95_std.pth`
  - `data/thought2text/block/block_splits_by_image_all.pth`
  - `data/thought2text/images/*.jpg` are valid symlinks into KaggleHub cache
  - train/val/test manifests have 0 missing image references in the current environment
  - report: `outputs/download_reports/thought2text_data_report.md`
- EIT-1M partial release downloaded:
  - `data/EIT-1M/Participant4_Session1_Visual_Textual.zip`
  - report: `outputs/download_reports/eit1m_download_report.md`
- THINGS-EEG2 access probed without starting a full dataset download:
  - OSF only exposed `experimental_paradigm_movie.mkv`
  - Hugging Face lookup returned unauthenticated 401 for `AutoLab/THINGS-EEG2`
  - report: `outputs/download_reports/things_eeg2_download_report.md`
- Download summary created:
  - `outputs/download_reports/DOWNLOAD_SUMMARY.md`
- Day2 Thought2Text real-data audit completed:
  - `outputs/day2/thought2text_inspection.md`
  - `outputs/day2/split_leakage_report.md`
  - `outputs/day2/clip_cache_report.md`
  - train/val/test image-level leakage is `False`
- Day2 EEG-to-CLIP alignment completed:
  - `outputs/day2_align/checkpoints/best.pt`
  - `outputs/day2_align/alignment_metrics.json`
  - `outputs/day2_align/alignment_report.md`
  - test R@5 `0.0315` vs random R@5 `0.0149`
- Day3 fusion and degraded sanity completed:
  - `outputs/day3/fusion_qwen15/checkpoints/best.pt`
  - `outputs/day3/fusion_report.md`
  - `outputs/day3/degraded_clip_cache_report.md`
  - `outputs/day3/sanity_real/metrics.md`
  - `outputs/day3/sanity_real/gate_analysis.md`
  - `outputs/day3/sanity_real/sample_predictions.jsonl`
  - `outputs/day3/DAY3_REPORT.md`

- Day5 heavy research run is complete:
  - Fusion/control runs completed under `outputs/day5_fusion/F0_vision_only`, `F1_real_eeg`, `F2_random_encoder_control`, and `F3_shuffled_training_control`.
  - Full degraded sanity completed under `outputs/day5_sanity` for clean, blur, occlusion, noise, and lowres conditions.
  - `FULL_SANITY_METRICS.*`, `gate_analysis.md`, `sample_predictions.jsonl`, qualitative examples, and `outputs/day5_final/NEXT_48H_RESEARCH_REPORT.md` are available.
  - Real EEG beats shuffled/random by class-hit on 5/5 evaluated corruptions, but generated text is still weak and the claim should remain cautious.
- Day5 alignment extension results are available:
  - CLIP-L/14, Strong E4, Subject X2, and 6 extra alignment configs completed.
  - None of the Day5 extension runs beat `outputs/day4_alignment/best_overall.pt` by unique-image R@5.
  - The current recommended fusion checkpoint remains `outputs/day4_alignment/best_overall.pt`.
- BLIP caption targets were generated:
  - `data/thought2text/blip_captions.jsonl`
  - `data/thought2text/{train,val,test}_blip_caption.jsonl`
  - `outputs/day4_caption_targets/blip_caption_report.md`
- Day5 engineering fixes completed while training continues:
  - Fusion validation/sample generation now respects `vision_only`, `real_eeg`, `shuffled_eeg`, and `random_eeg` modes.
  - Fusion checkpoints now save compact trainable caption parameters when the LLM is frozen, avoiding repeated multi-GB frozen LLM checkpoints.
  - `scripts/precompute_vision.py` supports native CLIP output dimensions and can require a real model to prevent invalid fallback CLIP-L caches.
  - Day5 sanity can load degraded CLIP caches directly and metrics/gate analysis ignore `sample_predictions.jsonl`.
- Decide whether to make Thought2Text images portable by copying them into `data/thought2text/images/` instead of relying on KaggleHub symlinks.
- Decide whether to rerun Day3 fusion/sanity with multiple seeds before reporting any EEG effect.

## Blocked

- Git status verification is unavailable because `git` is not installed in this environment.
- Thought2Text Google Drive download is blocked by network access to `drive.google.com`.
- Provided proxy subscription endpoint did not return a usable config from this environment.
- User-provided port forward `7897` is not reachable from this container on `127.0.0.1`, common container gateways, SSH client address, or the container host address.
- Thought2Text Google Drive official layout is still not verified; local fallback uses `eeg_5_95_std.pth` rather than the official-style `eeg_55_95_std.pth`.
- Thought2Text image files are currently valid symlinks to `/root/.cache/kagglehub/...`; they are usable on this machine but not portable without relinking or copying.
- THINGS-EEG2 full dataset was not downloaded because the accessible OSF tree did not expose expected data directories and the Hugging Face dataset lookup required authentication or a different repo/source.

## Next action

- Convert a larger subset of `EEG_Image_CVPR_ALL_subj` once storage/time budget is clear.
- Add real-data manifest docs to `docs/INTERFACE_CONTRACTS.md` if the project chooses this as the Phase 3 source.
- Optionally copy Thought2Text symlink targets into project storage if portability matters more than disk savings.
- Run additional seeds for alignment/fusion and compare real EEG against shuffled/random controls before making any stronger claim.

## Heavy Stage Update — 2026-06-17T04:02:00Z

## Done
- A1/A2/A3 named architecture runs now have verified full metrics in `outputs/architectures/ARCHITECTURE_SEARCH_REPORT.md`.
- A2 full run completed with R@1/R@5/R@10 `0.0751 / 0.2943 / 0.5105`, class_acc `0.2954`; it is the best A-series result but below P2.
- Semantic fusion controlled seed evals are complete and summarized in `outputs/final_semantic/SEMANTIC_FUSION_CONTROLLED_MULTISEED_SUMMARY.md`.
- Gate claim boundary is documented in `outputs/final_semantic/GATE_ANALYSIS.md`.
- Materializer tests pass for the A-series reporting fix.

## In Progress
- A2 seed6161 stability run: `outputs/architectures/A2_temporal_spectral_spatial_seed6161_full`.
- A2 downstream semantic fusion full run: `outputs/heavy_stage/semantic_fusion_A2_temporal_spectral_spatial_full`.
- A2 strong-degradation eval watcher: `outputs/final_semantic/semantic_fusion_A2_temporal_spectral_spatial_full_strong_eval`.

## Blocked
- THINGS-EEG2 paired image-trial alignment remains blocked by missing trial/stimulus metadata.
- EIT-1M full tri-modal loader remains blocked by unavailable full paired image/text release.

## Next
- Let A2 semantic fusion finish and evaluate it under strong degradation.
- Let A2 seed6161 finish and refresh architecture stability metrics.
- Update `outputs/heavy_stage/MASTER_REPORT.md` after the A2 downstream/eval artifacts exist.

## Heavy Stage Update — 2026-06-17T04:12:30Z

## Done
- A2 downstream semantic fusion full run completed 80 epochs:
  - `outputs/heavy_stage/semantic_fusion_A2_temporal_spectral_spatial_full/semantic_fusion_train_report.md`
  - final val accuracy `0.9732`.
- A2 strong-degradation eval completed:
  - `outputs/final_semantic/A2_SEMANTIC_FUSION_STRONG_EVAL_SUMMARY.md`
  - real EEG beats shuffled/random and vision-only in `6/6` evaluated conditions.
- `outputs/heavy_stage/MASTER_REPORT.md` now uses the A2 semantic fusion eval as the strongest semantic evidence.

## In Progress
- A2 seed6161 stability alignment remains running under `outputs/architectures/A2_temporal_spectral_spatial_seed6161_full`.

## Heavy Stage Update — 2026-06-17T04:29:00Z

## Done
- A2 semantic fusion seed42/123/2025 full runs and strong-degradation evals completed.
- A2 semantic multi-seed report created:
  - `outputs/final_semantic/A2_SEMANTIC_FUSION_MULTISEED_SUMMARY.md`
  - real EEG beats shuffled/random controls in `18/18` seed-condition pairs.
  - real EEG beats vision-only in `18/18` seed-condition pairs.
- `outputs/heavy_stage/MASTER_REPORT.md` now reflects the A2 multi-seed semantic evidence.

## In Progress
- A2 seed6161 alignment stability run remains active and has reached epoch 32; final alignment metrics are pending.

## 2026-06-17 Heavy Stage Update

Done:
- Added and evaluated reliability-aware gated semantic fusion for A2.
- Generated `outputs/final_semantic/A2_GATE_VS_NOGATE_REPORT.md`.
- A2 gate result: real EEG beats controls/vision in 6/6 strong conditions, but gate variant beats no-gate in only 1/6 and gate_mean does not support a learned reliability mechanism claim.

In Progress:
- M0 ConvTransformer-base-style masked EEG pretraining on THINGS raw EEG windows is running in tmux session `pretrain_things_m0_convtransformer_base`.

Next:
- Let M0 finish or early-stop, then run transfer/alignment comparison if checkpoint is usable.

## 2026-06-17 M2 Masked Pretraining Prep

Done:
- Added `temporal_spectral_spatial` variant support to `MaskedEEGAutoencoder`.
- Added M2 THINGS raw-window masked pretraining config and M2 Thought2Text transfer config.
- Added launcher script `scripts/launch_after_artifacts.py` for artifact-gated queued jobs.

In Progress:
- M0 pretraining continues on GPU.
- M0 transfer watcher waits for formal report + checkpoint.
- M2 launcher waits for M0 transfer metrics before starting.

## 2026-06-17 Heavy Stage Runtime Status

In Progress:
- M0 ConvTransformer-base-style THINGS raw-window masked pretraining is active on GPU, around epoch 14 with batch 1024.
- M0 transfer watcher is waiting for the formal M0 pretraining report before launching Thought2Text alignment.
- M2 pretraining launcher is waiting for M0 transfer metrics before starting.

Notes:
- M2 temporal-spectral-spatial masked autoencoder code/configs are ready and tested, but not launched yet because M0 still occupies the GPU usefully.

## 2026-06-17 Heavy Stage Concurrent Pretraining Update

Done:
- Started M2 Temporal-Spectral-Spatial masked EEG pretraining as a full THINGS raw-window run, not a smoke run:
  - `outputs/pretrain/masked_eeg_things_eeg2_m2_temporal_spectral_spatial/train.stdout.log`
  - config: `configs/pretrain/masked_eeg_things_eeg2_m2_temporal_spectral_spatial.yaml`
- Updated `configs/heavy_stage_queue.yaml` so M2 pretraining is `running` and M2 transfer is `waiting`.
- Started M2 transfer watcher:
  - `outputs/transfer/things_m2_tsst_pretrain_t2t_align/launcher.log`

In Progress:
- M0 ConvTransformer-base-style THINGS raw-window masked pretraining continues.
- M2 Temporal-Spectral-Spatial THINGS raw-window masked pretraining now runs concurrently with M0.
- M0 and M2 transfer watchers wait for formal pretraining reports plus best checkpoints before launching Thought2Text transfer/alignment.

Runtime evidence:
- Fresh GPU sample after M2 launch: ~33GB / 48GB memory, 99% GPU utilization.
- Host memory remains stable with ~85GB available.

## 2026-06-17 GPU Monitor And Coverage Update

Done:
- Added GPU monitor training-status extraction from JSON `train.stdout.log` files.
- Refreshed `outputs/heavy_stage/GPU_UTILIZATION_REPORT.md`; it now includes active job, dataset, epoch, step, batch size, current/best validation metric, and peak GPU memory.
- Verified monitor/scheduler tests:
  - `python -m unittest tests.test_gpu_monitor_training_status tests.test_heavy_stage_scheduler tests.test_heavy_stage_scheduler_loop -v`
  - `python -m py_compile scripts/gpu_monitor.py tests/test_gpu_monitor_training_status.py`

In Progress:
- M0 THINGS raw-window masked pretraining: active full run, batch 1024.
- M2 THINGS raw-window masked pretraining: active full run, batch 1024.
- M0/M2 transfer watchers are waiting for formal pretraining reports plus best checkpoints.

Still Missing Or Partial:
- M1 as a separate new THINGS raw-window pretraining run is queued, not active yet:
  - `configs/pretrain/masked_eeg_things_eeg2_m1_dualbranch_eegconformer.yaml`
  - queue id `MASKED_EEG_PRETRAIN_THINGS_M1_DUALBRANCH_EEGCONFORMER`
- A4 is now materialized as a queued explicit architecture run:
  - `configs/heavy_architectures/A4_raw_spectrogram_late_fusion_full.yaml`
  - queue id `ARCH_A4_RAW_SPECTROGRAM_LATE_FUSION_FULL`
- H1 hard-negative alignment is now implemented and queued:
  - `configs/heavy_architectures/H1_P2_hard_negative.yaml`
  - queue id `H1_P2_HARD_NEGATIVE_ALIGNMENT`
- SigLIP prototype/backbone follow-up is queued as cache-first work:
  - queue id `CLIP_ADAPTER_SIGLIP_PROTOTYPE_CALIBRATION`
  - downstream adapter/alignment should be added only after cache dimensions/model compatibility are verified.

## 2026-06-17 Heavy Stage Queue Coverage Update

Done:
- Implemented batch hard-negative contrast loss for `src.train.train_align` gated by `use_hard_negative`.
- Added a regression test proving `compute_alignment_loss` emits a `hard_negative` term when enabled.
- Added full-run configs for M1, explicit A4, and H1.
- Added queue coverage for M1/A4/H1/SigLIP follow-ups without interrupting active M0/M2 runs.
- M1 is artifact-gated behind M0 and M2 Thought2Text transfer metrics, so it will not start before the current pretraining-to-transfer sequence has produced comparable downstream results.
- Refreshed heavy-stage scheduler board and master report.

Verification:
- `python -m unittest tests.test_heavy_stage_scheduler tests.test_heavy_stage_scheduler_loop tests.test_heavy_stage_materializer tests.test_train_align_loss_hooks tests.test_gpu_monitor_training_status -v`
- `python -m py_compile src/train/train_align.py scripts/materialize_heavy_stage_reports.py scripts/gpu_monitor.py tests/test_heavy_stage_materializer.py tests/test_heavy_stage_scheduler_loop.py tests/test_train_align_loss_hooks.py`

Runtime:
- M0 and M2 THINGS raw-window masked EEG pretraining are still active and using the GPU heavily.
- Scheduler did not auto-launch queued M1/A4/H1/SigLIP jobs because GPU is not idle.

## 2026-06-17 Heavy Stage Model Gap Audit

Done:
- Refreshed heavy-stage monitor, scheduler board, architecture report, transfer report, baseline report, and master report from current artifacts.
- Updated `outputs/heavy_stage/CURRENT_MODEL_COVERAGE.md` so the next jobs reflect the active M0/M2 -> transfer -> M1/A4/H1/SigLIP sequence instead of stale A2 follow-ups.
- Wrote `outputs/heavy_stage/MODEL_TRAINING_GAP_AUDIT.md` summarizing completed, running, and queued model families.
- Verified materializer/scheduler/monitor behavior with:
  - `python -m unittest tests.test_heavy_stage_materializer tests.test_heavy_stage_scheduler tests.test_heavy_stage_scheduler_loop tests.test_gpu_monitor_training_status -v`
  - `python -m py_compile scripts/materialize_heavy_stage_reports.py scripts/heavy_stage_scheduler.py scripts/gpu_monitor.py`

In Progress:
- M0 ConvTransformer-base-style THINGS raw-window masked EEG pretraining is active.
- M2 Temporal-Spectral-Spatial THINGS raw-window masked EEG pretraining is active.
- M0/M2 transfer watchers are waiting for formal pretraining reports and best checkpoints before launching full Thought2Text transfer/alignment.

Runtime Evidence:
- Fresh GPU sample after report refresh: `100%` utilization, `34221 / 49140 MiB` memory, two CUDA processes for M0 and M2.
- Scheduler loop uses `--launch-when-idle`; it did not launch M1/A4/H1/SigLIP because the GPU is not idle.

Queued Next:
- M1 DualBranchEEGConformer THINGS raw-window masked pretraining after M0/M2 transfer metrics exist.
- Explicit A4 raw+spectrogram late-fusion full run.
- H1 hard-negative raw+spectrogram alignment follow-up.
- SigLIP prototype/calibration cache job.

## 2026-06-17 Heavy Stage Report Staleness Fix

Done:
- Fixed `scripts/materialize_heavy_stage_reports.py` so completed exact-linked EEG-ImageNet paired alignment and exact tri-modal runs are treated as negative controls, not as active next work.
- Added regression coverage in `tests/test_heavy_stage_materializer.py` for this stale-report case.
- Regenerated `outputs/heavy_stage/MASTER_REPORT.md`; its next direction now points to the active M0/M2 -> transfer sequence and queued M1/A4/H1/SigLIP follow-ups.

Verification:
- `python -m unittest tests.test_heavy_stage_materializer tests.test_heavy_stage_scheduler tests.test_heavy_stage_scheduler_loop tests.test_gpu_monitor_training_status -v`
- `python -m py_compile scripts/materialize_heavy_stage_reports.py scripts/heavy_stage_scheduler.py scripts/gpu_monitor.py tests/test_heavy_stage_materializer.py`
- Fresh GPU sample: `98%` utilization, `34221 / 49140 MiB` memory, M0 and M2 CUDA processes active.

## 2026-06-17 Evidence Strength Assessment

Done:
- Wrote `outputs/heavy_stage/EVIDENCE_STRENGTH_ASSESSMENT.md` to calibrate what the current data can and cannot support.
- M0 THINGS ConvTransformer-base-style pretraining completed, but its Thought2Text transfer is weak:
  - artifact: `outputs/transfer/things_m0_convtransformer_pretrain_t2t_align/alignment_metrics.json`
  - unique-image R@1/R@5/R@10: `0.0480 / 0.1742 / 0.2943`
  - class_acc: `0.1335`
- The strongest alignment remains P2 raw+spectrogram:
  - artifact: `outputs/alignment_fill8/P2_raw_spectrogram_seed2718_fill8/alignment_metrics.json`
  - unique-image R@1/R@5/R@10: `0.0661 / 0.3574 / 0.5916`
  - class_acc: `0.3323`
- The strongest downstream evidence is A2 semantic fusion multi-seed robustness:
  - artifact: `outputs/final_semantic/A2_SEMANTIC_FUSION_MULTISEED_SUMMARY.csv`
  - real EEG beats shuffled/random and vision-only in `18/18` seed-condition pairs.

In Progress:
- M2 THINGS Temporal-Spectral-Spatial masked pretraining is still running full training with batch/effective batch `1024`.
- M2 transfer watcher remains waiting for the final M2 pretraining report and checkpoint.

Claim Boundary:
- Supported: constrained class-level semantic captioning / robust semantic prediction under degraded vision.
- Not supported: open-ended free-form captioning success or a proven reliability-gate mechanism.

## 2026-06-17 Heavy Stage Current Update

Done:
- M2 THINGS Temporal-Spectral-Spatial masked pretraining completed and transferred to Thought2Text.
  - pretrain report: `outputs/pretrain/masked_eeg_things_eeg2_m2_temporal_spectral_spatial/MASKED_EEG_PRETRAIN_REPORT.md`
  - transfer metrics: `outputs/transfer/things_m2_tsst_pretrain_t2t_align/alignment_metrics.json`
  - unique-image R@1/R@5/R@10: `0.0450 / 0.1742 / 0.3033`
  - class_acc: `0.1400`
- Added A2 grouped/global control audit:
  - report: `outputs/final_semantic/A2_grouped_global_control_eval/A2_GROUPED_GLOBAL_CONTROL_REPORT.md`
  - image-level gap table: `outputs/final_semantic/A2_grouped_global_control_eval/SEMANTIC_GAP_METRICS_IMAGE_LEVEL.csv`
  - result: real EEG beats vision-only, shuffled, global label-mismatched, and random EEG in `6/6` image-level conditions.
- Updated `outputs/heavy_stage/CURRENT_MODEL_COVERAGE.md` to remove stale M2-running text.

In Progress:
- M1 DualBranchEEGConformer masked pretraining on THINGS raw windows is running full training with batch/effective batch `1024`.
- A4 raw+spectrogram late-fusion full alignment is running.
- H1 P2 hard-negative alignment is running.

Dataset Status:
- EEG-ImageNet exact-linked subset is the strongest available image+EEG+text large-data path:
  - train/val/test exact rows: `39379 / 4932 / 4935`
  - cache shapes include `eeg_train_image_exact.npy = (39379, 62, 501)` and `clip_train_image_exact.npy = (39379, 512)`.
- THINGS-EEG2 remains useful for EEG-only masked pretraining, not image+EEG caption/fusion, because trial/image alignment metadata is still missing.
- EIT-1M remains blocked for full training because the local manifest has empty image/eeg paths.

## 2026-06-17 Heavy Stage Continuation After Interruption

Done:
- Fixed stale queue-status regression coverage so active jobs can be `running` instead of falsely requiring `queued`.
- Refreshed `outputs/heavy_stage/LIVE_STATUS.md`, `outputs/heavy_stage/MASTER_REPORT.md`, and the heavy-stage board from current scheduler state.
- Updated `outputs/heavy_stage/MODEL_TRAINING_GAP_AUDIT.md` to reflect current M1/A4/H1 concurrent training and completed M2 transfer.
- Updated large-data progress reporting so ImageNet CLS-LOC is reported as extracted with the original zip cleaned after extraction, not as a failed `0B` zip.
- Updated `outputs/datasets/DATASET_DECISION_REPORT.md` and `outputs/datasets/DATASET_RECOMMENDATION.md` so they no longer say ImageNet is still downloading or EEG-ImageNet paired data is unusable.

In Progress:
- M1 DualBranchEEGConformer THINGS raw-window masked pretraining is still running full training.
- A4 raw+spectrogram late-fusion full alignment is still running.
- H1 P2 hard-negative alignment is still running.

Verification:
- `python -m unittest tests.test_heavy_stage_materializer tests.test_semantic_fusion_classifier_eval tests.test_heavy_stage_scheduler_loop -v`
- `python -m unittest tests.test_large_data_progress_report -v`
- `python -m py_compile tests/test_heavy_stage_scheduler_loop.py scripts/materialize_heavy_stage_reports.py src/eval/semantic_fusion_classifier_eval.py`

Runtime Evidence:
- Fresh GPU sample after refresh: `100%` utilization and about `36.3GB / 48GB` memory used by active M1/A4/H1 jobs.

Next:
- Do not interrupt M1/A4/H1.
- Let the scheduler launch M1 transfer after M1 report/checkpoint exists.
- Let queued EEG-ImageNet exact-linked A4 scratch alignment and A2 scratch tri-modal jobs start when GPU resources free safely.

## 2026-06-17 Scheduler Max-Launch Update

Done:
- Added `--max-launches` support to `scripts/heavy_stage_scheduler.py`.
- Updated `scripts/run_heavy_stage_scheduler_loop.sh` so the idle auto-launch loop defaults to `HEAVY_STAGE_MAX_LAUNCHES:-8`.
- Restarted the scheduler loop without touching active training processes.
- Marked A4 full alignment completed in `configs/heavy_stage_queue.yaml` and recorded its metrics:
  - unique-image R@1/R@5/R@10: `0.0841 / 0.3093 / 0.5165`
  - class_acc: `0.3097`
  - not promotable over P2 R@5 `0.3574`.

Still Running:
- M1 THINGS DualBranchEEGConformer masked EEG pretraining.
- H1 P2 hard-negative alignment.

Verification:
- `python -m unittest tests.test_heavy_stage_scheduler tests.test_heavy_stage_scheduler_loop tests.test_large_data_progress_report -v`
- `python -m py_compile scripts/heavy_stage_scheduler.py scripts/update_large_data_progress_report.py tests/test_heavy_stage_scheduler.py tests/test_heavy_stage_scheduler_loop.py tests/test_large_data_progress_report.py`
- `bash -n scripts/run_heavy_stage_scheduler_loop.sh`

Runtime Evidence:
- Fresh GPU sample after scheduler-loop restart: `100%` utilization and about `35GB / 48GB` memory used by active M1+H1 jobs.

## 2026-06-17 Scheduler Safety Update

Done:
- Added auto-launch safety filtering to `scripts/heavy_stage_scheduler.py`.
- The idle scheduler can still launch up to `8` queued jobs per tick, but if free GPU memory is below `16GB`, it skips direct GPU training/cache commands and only permits lightweight watcher/launcher jobs.
- Restarted the scheduler loop with the safety-filtered code.

Verification:
- `python -m unittest tests.test_heavy_stage_scheduler tests.test_heavy_stage_scheduler_loop tests.test_large_data_progress_report -v`
- `python -m py_compile scripts/heavy_stage_scheduler.py scripts/update_large_data_progress_report.py tests/test_heavy_stage_scheduler.py tests/test_heavy_stage_scheduler_loop.py tests/test_large_data_progress_report.py`
- `bash -n scripts/run_heavy_stage_scheduler_loop.sh`

Runtime Evidence:
- Active training PIDs remain `577333` for M1 masked pretraining and `577371` for H1 hard-negative alignment.
- Fresh GPU sample after loop restart: about `35GB / 48GB` memory used; GPU utilization sampled between `57%` and `100%` while both jobs run.

## 2026-06-17 Scheduler Persistence Fix

Done:
- Replaced fragile background `nohup` scheduler launch with a persistent tmux session:
  - session: `heavy_stage_scheduler`
  - command: `HEAVY_STAGE_MAX_LAUNCHES=8 bash scripts/run_heavy_stage_scheduler_loop.sh`
- Confirmed active scheduler-loop processes:
  - `bash scripts/run_heavy_stage_scheduler_loop.sh`
  - `tee -a outputs/heavy_stage/heavy_stage_scheduler_loop.log`

Runtime Evidence:
- Active training PIDs are still `577333` for M1 and `577371` for H1.
- Scheduler session is independent and will keep refreshing monitor/board/reports plus idle-safe auto-launch.

## 2026-06-17 Heavy Goal Audit Refresh

Done:
- Marked H1 P2 hard-negative alignment completed and preserved it as a negative result:
  - metrics: `outputs/architectures/H1_P2_hard_negative/alignment_metrics.json`
  - unique-image R@1/R@5/R@10: `0.0541 / 0.2763 / 0.5045`
  - class_acc: `0.3097`
  - not promotable over P2 R@5 `0.3574`.
- Added `outputs/heavy_stage/GOAL_COMPLETION_AUDIT.md` with current completion estimates:
  - required artifact surface: about `85%`
  - required training matrix: about `75%`
  - final scientific decision: about `70%`
- Clarified EEG-ImageNet exact-linked coverage:
  - full EEG rows: `63850`
  - exact-linked paired rows: `49246`
  - missing exact stimulus JPEG rows: `14604` (`22.87%`)
  - the missing portion is image JPEG coverage, not EEG coverage.
- Added scheduler safety behavior: `HEAVY_STAGE_MAX_LAUNCHES=8` remains allowed, but each scheduler tick starts at most one direct heavy GPU training job; lightweight watcher/CPU jobs may still launch alongside it.
- Started the M1 transfer watcher in tmux session `m1_transfer_watcher`; it waits for the M1 formal pretraining report and then launches the full Thought2Text transfer automatically.
- Updated `scripts/materialize_final_semantic_report.py` and `scripts/launch_semantic_eval_after_transfer.py` so future transfer-eval materialization keeps A2 multi-seed semantic fusion as the primary evidence and retains weak transfer results as secondary negative/limited evidence.

In Progress:
- M1 DualBranchEEGConformer THINGS raw-window masked pretraining is the active heavy GPU job.
- Latest inspected state: epoch `31`, best val loss at epoch `20`, `11/25` patience epochs consumed.
- M1 is a full 200-epoch/early-stopping run with batch/effective batch `1024`, not a smoke test.

Queued:
- M1 Thought2Text transfer watcher is `waiting` for the M1 pretrain report.
- EEG-ImageNet exact A4 scratch alignment.
- EEG-ImageNet exact A2 scratch tri-modal training.
- SigLIP prototype/cache follow-up.

Verification:
- `python -m unittest tests.test_heavy_stage_scheduler tests.test_heavy_stage_scheduler_loop tests.test_large_data_progress_report -v`
- `python -m py_compile scripts/heavy_stage_scheduler.py scripts/update_heavy_stage_queue_status.py scripts/materialize_heavy_stage_reports.py scripts/gpu_monitor.py tests/test_heavy_stage_scheduler.py tests/test_heavy_stage_scheduler_loop.py tests/test_large_data_progress_report.py`
- `python -m unittest tests.test_launch_semantic_eval_after_transfer tests.test_final_semantic_materializer -v`

Runtime Evidence:
- Fresh GPU sample: `100%` utilization and about `33.6GB / 48GB` memory used by active M1 pretraining.
