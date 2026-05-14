# Project Structure

## Start Here

Open these first if you need the current thesis mainline quickly:

- `CURRENT_MAINLINE.md`
- `tools/current/README.md`
- `tools/current/show_mainline.py`
- `tools/current/check_current_artifacts.py`

## Current Mainline

Current thesis pair-level main model:
- `FINAL360I_struct360b_final_selected`

Current pair-level mainline files:
- config: `configs/final360i_struct360b_final.yaml`
- trainer/selector: `tools/final360i_retrain_and_select.py`
- base model family: `models/struct360b_match_free_coarse_to_fine.py`
- dataset path: `datasets/dset2c_manifest_dataset.py`
- trajectory export/eval: `tools/train360e_sequence_trajectory_export_and_ate_eval.py`

Current pair-level test reference:
- `signed_tdir_mean = 45.264702006380205`
- `anti_parallel_rate = 0.20169893322797314`
- `tmag_median_ratio = 0.8344251368086006`
- `path_ratio = 0.6403519796204528`

Trajectory caveat:
- `TRAIN360E` showed noticeable drift under direct adjacent-pair composition
- `SEQ360B` improved path and SE3 ATE, but over-corrected pair scale and did not improve Sim3 ATE
- `SEQ360A` did not improve trajectory shape
- `STRUCT360C` remains an active challenger, not the promoted mainline

## Runtime Code Layout

Current package-first runtime layout:

- `train360/core/`
  current shared runtime modules used by the DSET2C mainline and current challengers
- `train360/legacy/`
  legacy raw-scan / MVP-era modules preserved for compatibility and audits
- root-level `config.py`, `model.py`, `interaction.py`, `losses.py`, `pose_head.py`, `erp_sampling.py`, `healpix_utils.py`, `transformer_encoder.py`, `train360_pose_losses.py`, `train360d_pose_losses.py`, `depth_branch.py`
  compatibility wrappers that re-export from `train360.core`
- root-level `dataset_pano_only.py`, `train_mvp.py`, `geometry_refine.py`
  compatibility wrappers that re-export from `train360.legacy`

## Active Model Families

- `FINAL360I`: current pair-level thesis mainline
- `STRUCT360A/B/C`: structure-aware challenger family
- `SEQ360A/B`: sequence-level challenger family
- `TRAIN360C/D/H`: earlier or supporting training lines
- `BASE360`: official external baseline and alignment work
- `DEV360`: code-preparation and forward-looking experiment scaffolding

## Where Things Live

- configs: `configs/`
- datasets: `datasets/`
- models: `models/`
- current-index utilities: `tools/current/`
- train/eval/maintenance entrypoints: `tools/`
- reports and metrics: `reports/`
- curated reference archive: `reports_curated/`
- local checkpoints: `checkpoints/`
- raw or prepared data: `data/`
- official baseline outputs and project trajectory exports: `external_baselines/results/`

## Local-Only And Usually Not Committed

- `checkpoints/**/*.pt`
- local trajectory exports under `external_baselines/results/*`
- `pred_tum.txt`
- `adjacent_pair_predictions.jsonl`
- `backup_patches/`
- large logs and caches

## Legacy Vs Runtime-Sensitive Legacy

Legacy or archival code now quarantined under `train360/legacy/`:
- `dataset_pano_only.py`
- `train_mvp.py`
- `geometry_refine.py`

Runtime-sensitive core code now packaged under `train360/core/`:
- `config.py`
- `model.py`
- `interaction.py`
- `losses.py`
- `pose_head.py`
- `erp_sampling.py`
- `healpix_utils.py`
- `transformer_encoder.py`
- `train360_pose_losses.py`
- `train360d_pose_losses.py`
- `depth_branch.py`

The root-level filenames remain import-compatible wrappers so old tools can still import them while current mainline tools and models move toward package imports.
