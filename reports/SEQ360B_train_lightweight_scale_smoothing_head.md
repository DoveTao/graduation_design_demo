# SEQ360B train lightweight scale smoothing head

## 1. Executive summary
- training executed: `true`
- checkpoint saved: `true`
- best checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/SEQ360B_lightweight_scale_smoothing/best_val.pt`
- scale smoothing head trained: `true`
- pair test signed / anti / tmag / path: `45.325446` / `0.200158` / `1.697848` / `1.350477`
- trajectory test ATE none / SE3 / Sim3: `139.227175` / `75.946913` / `27.567211`
- trajectory test path ratio: `1.350237`
- classification: `partial`

## 2. Motivation
- FINAL360I pair-level metrics are strong, but TRAIN360E trajectory path ratio is `1.756343083453392`.
- SEQ360B tests whether a small log_tmag correction head can reduce scale/path drift while preserving R/tdir.

## 3. Design
- base model frozen: `true`
- R/tdir changed by head: `false`
- delta_log_tmag clamp: `[-0.3, 0.3]`
- losses: `SmoothL1 log_tmag + path ratio + delta smoothness + delta regularization`
- explicit matching / RANSAC / PnP / BA: `false / false / false / false`

## 4. Data protocol
- DSET2C canonical train/val/test manifests used.
- Train clips from train manifest only; val for selection; test for final evaluation only.
- No random pair split and no direct raw sequence glob split.

## 5. Training setup
- clip_len: `3`
- train/val/test clip counts: `3043` / `1337` / `1267`
- epochs: `1`
- batch size: `4`
- LR: `0.0001`
- optimizer: `AdamW`
- selected epoch: `1`
- best val score: `23.94464857444634`

## 6. Validation results
- clip path ratio: `1.177761`
- tmag median ratio: `2.199424`
- pair metrics: `{'count': 1339, 'coverage': 1.0, 'rot_mean_deg': 0.9819342331976553, 'signed_tdir_mean_deg': 104.52596244804725, 'anti_parallel_rate': 0.6594473487677371, 'tmag_median_ratio': 2.199424098126921, 'path_ratio': 1.1770705298793382, 'path_length_pred': 654.9351881891489, 'path_length_gt': 556.4111678645853, 'nan_inf_count': 0, 'note': 'SEQ360B pair metrics are adjacent trajectory rows with scale-smoothed tmag; R/tdir come from frozen FINAL360I.'}`

## 7. Test pair-level results
- FINAL360I vs SEQ360B pair comparison: `better`
- SEQ360B test pair metrics: `{'count': 1269, 'coverage': 1.0, 'rot_mean_deg': 0.8862303512771328, 'signed_tdir_mean_deg': 45.32544604265589, 'anti_parallel_rate': 0.20015760441292357, 'tmag_median_ratio': 1.6978484631707425, 'path_ratio': 1.3504765883324317, 'path_length_pred': 1006.1652189642191, 'path_length_gt': 745.0445477226908, 'nan_inf_count': 0, 'note': 'SEQ360B pair metrics are adjacent trajectory rows with scale-smoothed tmag; R/tdir come from frozen FINAL360I.'}`

## 8. Test trajectory results
- compared to TRAIN360E FINAL360I trajectory: `better`
- compared to BASE360D trajectory: `partial`
- SEQ360B test trajectory metrics: path_ratio=`1.350237`, ATE Sim3=`27.567211`

## 9. Analysis
- scale smoothing reduced path overestimation: `True`
- ATE improved: `True`
- Pair direction should remain stable because the head never modifies R/tdir.
- FINAL360I remains the pair-level main model unless sequence-scale behavior is preferred.

## 10. Next recommendation
- `proceed_to_SEQ360A_sequence_consistency_scale_drift_stabilization`

## 11. Compliance checklist
- `training_executed = true`
- `fine_tune_executed = true`
- `learned_weights_saved = true`
- `final360i_checkpoint_modified = false`
- `r_tdir_modified_by_head = false`
- `explicit_matching_used = false`
- `match_list_output = false`
- `ransac_used = false`
- `pnp_used = false`
- `bundle_adjustment_used = false`
- `hkust_360dvo_teacher_used = false`
- `base360_outputs_used_as_training_input = false`
- `train_manifest_used = true`
- `val_manifest_used_for_selection_only = true`
- `test_manifest_used_for_final_eval_only = true`
- `dset2c_canonical_split_used = true`
- `random_pair_split_used = false`
- `direct_glob_data_360dvo_sequences = false`
- `s5_locked_metrics_modified = false`
- `large_checkpoints_committed_to_git = false`
