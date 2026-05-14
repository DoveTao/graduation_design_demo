# Current Mainline

## Open These First

- mainline summary: `CURRENT_MAINLINE.md`
- structure guide: `PROJECT_STRUCTURE.md`
- quick terminal entrypoint: `tools/current/show_mainline.py`
- artifact check: `tools/current/check_current_artifacts.py`

## Current Thesis Mainline

- main model: `FINAL360I_struct360b_final_selected`
- pair-level config: `configs/final360i_struct360b_final.yaml`
- pair-level trainer/selector: `tools/final360i_retrain_and_select.py`
- pair-level base model: `models/struct360b_match_free_coarse_to_fine.py`
- dataset path: `datasets/dset2c_manifest_dataset.py`
- trajectory evaluation: `tools/train360e_sequence_trajectory_export_and_ate_eval.py`
- runtime core package: `train360/core/`

## Preserved Variant

- sequence scale variant: `SEQ360B`
- config: `configs/seq360b_lightweight_scale_smoothing.yaml`
- tool: `tools/train_seq360b_lightweight_scale_smoothing.py`
- model head: `models/seq360b_scale_smoothing_head.py`

## Current Pair-Level Reference

- `signed_tdir_mean = 45.264702006380205`
- `anti_parallel_rate = 0.20169893322797314`
- `tmag_median_ratio = 0.8344251368086006`
- `path_ratio = 0.6403519796204528`

## Current Caveat

- direct adjacent-pair trajectory composition still drifts
- `TRAIN360E` is the kept trajectory evaluation path
- `SEQ360B` is retained as the only promoted non-mainline variant
- `SEQ360A` and `STRUCT360C` are no longer active codepaths; only status summaries are kept

