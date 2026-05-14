# Project Structure

## Start Here

- `README.md`
- `CURRENT_MAINLINE.md`
- `tools/current/README.md`

## Core Runtime

- `train360/core/`
  current shared runtime modules used by the kept mainline and `SEQ360B`

## Active Code

- `datasets/`
  kept DSET2C manifest-native datasets
- `models/`
  kept mainline model plus `SEQ360B` scale head
- `tools/`
  kept current entrypoints only
- `configs/`
  kept mainline and `SEQ360B` configs only

## Kept Mainline Files

- `tools/final360i_retrain_and_select.py`
- `tools/train360e_sequence_trajectory_export_and_ate_eval.py`
- `tools/train_struct360b_match_free_coarse_to_fine.py`
- `models/struct360b_match_free_coarse_to_fine.py`
- `configs/final360i_struct360b_final.yaml`
- `configs/struct360b_match_free_coarse_to_fine.yaml`

## Kept Variant

- `tools/train_seq360b_lightweight_scale_smoothing.py`
- `models/seq360b_scale_smoothing_head.py`
- `configs/seq360b_lightweight_scale_smoothing.yaml`

## Data And Results

- `checkpoints/`
  local checkpoint storage; `.pt` files are not committed in this cleanup
- `data/`
  raw/prepared data, untouched
- `external_baselines/results/dset2c_360dvo_canonical/`
  canonical manifests, preserved
- `external_baselines/results/base360_hkust_360dvo_official/`
  official baseline results, preserved
- `reports/`
  reduced to current-mainline-facing reports, key dependencies, and MAINT13 cleanup records

## Removed From Active Tree

- root-level compatibility wrappers
- `train360/legacy/`
- old S5 / scene01 / one-off train/eval scripts
- maintenance-heavy report noise
- failed or non-promoted ablation codepaths

