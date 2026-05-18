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

- `tools/train_final360m_fulltrain_thesis_main.py`
- `tools/final360m_trajectory_thesis_refresh.py`
- `tools/odom360a_lightweight_trajectory_fusion.py`
- `tools/odom360b_local_pose_graph_kstep.py`
- `tools/train_struct360b_match_free_coarse_to_fine.py`
- `models/struct360b_match_free_coarse_to_fine.py`
- `configs/final360m_fulltrain_struct360b_thesis_main_guarded.yaml`
- `configs/struct360b_match_free_coarse_to_fine.yaml`

## Kept Variant

- `tools/train_seq360b_lightweight_scale_smoothing.py`
- `models/seq360b_scale_smoothing_head.py`
- `configs/seq360b_lightweight_scale_smoothing.yaml`

## Visible Comparison Set

- `FINAL360M`: current full-train pair-level thesis main model
- `FINAL360I`: subset-trained candidate only
- `FINAL360M-ODOM360A`: recommended refreshed eval-only trajectory backend
- `TRAIN360E`: old `FINAL360I`-based direct trajectory reference
- `SEQ360B`: retained sequence-scale variant
- `BASE360D`: official external baseline
- `T57b`: legacy external reference

## Status-Only Items

- `SEQ360A`: no_improvement; summary only
- `STRUCT360C`: evaluation_failed; summary only

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
  reduced to current-mainline-facing reports, retained summaries, and cleanup records

## Removed From Active Tree

- root-level compatibility wrappers
- `train360/legacy/`
- old one-off train/eval scripts
- maintenance-heavy report noise
- failed or non-promoted ablation codepaths
