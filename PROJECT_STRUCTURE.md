# Project Structure

## Current Main Model

The current thesis pair-level main model is `FINAL360I_struct360b_final_selected`.

Current pair-level test reference:
- `signed_tdir_mean = 45.264702006380205`
- `anti_parallel_rate = 0.20169893322797314`
- `tmag_median_ratio = 0.8344251368086006`
- `path_ratio = 0.6403519796204528`

Trajectory caveat:
- `TRAIN360E` showed that direct adjacent-pair composition still produces noticeable drift.
- `SEQ360B` improved path and SE3 ATE, but over-corrected pair-scale magnitude and did not improve Sim3 ATE.
- `SEQ360A` did not improve trajectory shape.
- `STRUCT360C` remains an active challenger family, but not the promoted mainline.

## Active Model Families

- `FINAL360I`: current pair-level mainline and thesis-facing selection.
- `STRUCT360A/B/C`: structure-aware challenger family, with `STRUCT360B` supplying the coarse-to-fine base used by FINAL360I and `STRUCT360C` extending it.
- `SEQ360A/B`: sequence-level drift and composition challengers.
- `TRAIN360C/D/H`: earlier and supporting mainline or challenger training lines.
- `BASE360`: external official baseline and component-alignment work.
- `DEV360`: code-preparation and forward-looking experiment scaffolding.

## Where Things Live

- configs: `configs/`
- datasets: `datasets/`
- active models: `models/`
- entrypoints and utilities: `tools/`
- reports and metrics: `reports/`
- curated reference archive: `reports_curated/`
- local checkpoints: `checkpoints/`
- raw or prepared data: `data/`
- external baseline results and trajectory exports: `external_baselines/results/`

## Local-Only And Usually Not Committed

- `checkpoints/**/*.pt`
- local trajectory exports under `external_baselines/results/*`
- `pred_tum.txt`
- `adjacent_pair_predictions.jsonl`
- `backup_patches/`
- large logs and caches

## Legacy Vs Runtime-Sensitive Legacy

Legacy or archival examples:
- `train_mvp.py`
- `dataset_pano_only.py`
- `geometry_refine.py`
- `depth_branch.py`

Runtime-sensitive legacy-root files that still matter today:
- `config.py`
- `model.py`
- `interaction.py`

Those root-level files look like cleanup candidates, but active FINAL360I / STRUCT360B / STRUCT360C tools still depend on them directly or transitively, so they should only move in a dedicated import-migration step.
