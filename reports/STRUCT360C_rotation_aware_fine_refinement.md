# STRUCT360C rotation-aware fine refinement

## 1. Executive summary
- training executed: `true`
- checkpoint saved: `true`
- best checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/STRUCT360C_rotation_aware_fine_refinement/best_val.pt`
- init checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
- classification: `evaluation_incomplete_after_training`
- pair metrics: `test evaluation not completed in this run`
- trajectory metrics: `trajectory evaluation not completed in this run`

## 2. Motivation
- FINAL360I pair-level strong.
- TRAIN360E trajectory drift.
- SEQ360B improved scale/path but not Sim3.
- SEQ360A sequence consistency no improvement.
- STRUCT360C targets rotation-aware fine refinement.

## 3. Research alignment
- match-free: `true`
- no explicit correspondence: `true`
- no RANSAC / PnP / BA: `true / true / true`
- latent rotation-aware attention bias only: `true`

## 4. Architecture
- coarse-to-fine inherited from STRUCT360B.
- coarse `R0` conditions latent spherical attention bias.
- fine branch predicts `ΔR / Δtdir / Δlog_tmag`.
- final pose composition remains conservative with `alpha=0.25`, `beta=0.25`.

## 5. Training setup
- init checkpoint: `checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
- epochs: `3`
- batch size: `2`
- LR groups: `coarse=1e-05`, `fine/bias=3e-05`
- warmup/unfreeze: `warmup_epochs=1`
- gamma: `1.0`
- val score: `{'signed_tdir_weight': 1.0, 'anti_parallel_weight': 70.0, 'tmag_ratio_weight': 20.0, 'path_ratio_weight': 10.0, 'rot_weight': 0.1}`

## 6. Validation results
- selected epoch: `1`
- pair metrics: `{'count': 5342, 'coverage': 1.0, 'rot_mean_deg': 1.3044976745413086, 'rot_median_deg': 1.0903805494308472, 'rot_p90_deg': 2.7736392736434943, 'signed_tdir_mean_deg': 105.138765615312, 'signed_tdir_median_deg': 117.11144011875797, 'signed_tdir_p90_deg': 146.9936745944493, 'unsigned_tdir_mean_deg': 53.97376438482147, 'unsigned_tdir_median_deg': 52.74270375837135, 'unsigned_tdir_p90_deg': 81.10707346742221, 'anti_parallel_rate': 0.6695994009734182, 'tmag_ratio_p10': 0.2691349621630504, 'tmag_median_ratio': 1.0034856308119506, 'tmag_ratio_p50': 1.0034856308119506, 'tmag_mean_ratio': 1.476341932809384, 'tmag_ratio_p90': 3.983267687891274, 'tmag_p90_ratio': 3.983267687891274, 'log_tmag_mae': 0.7428025147760814, 'scale_collapse_rate': 0.0, 'scale_explosion_rate': 0.0, 'path_ratio': 0.5629363941249144, 'path_length_pred': 3420.209930151701, 'path_length_gt': 6075.659640852362, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0}`
- diagnostics: `{'fine_gate_stats': {'mean': 0.039118158662059206, 'median': 0.039117392152547836, 'min': 0.03911544010043144, 'max': 0.03912162408232689}, 'delta_rot_mean_deg': 0.041369106232818705, 'delta_tdir_norm_mean': 2.6737776166856713e-05, 'delta_log_tmag_abs_mean': 7.815955942563342e-06, 'attention_bias_mean': -1.5707554033773388, 'attention_bias_min': -3.14035626407155, 'attention_bias_max': -0.0009120791363822642, 'attention_entropy_mean': 6.427700396069509, 'explicit_matching_used': False, 'match_list_output': False, 'correspondence_list_output': False, 'topk_matching_used': False}`

## 7. Test pair-level results
- FINAL360I: `available in reports/FINAL360I_metrics_test.json`
- STRUCT360B: `available in reports/STRUCT360B_metrics_test.json`
- STRUCT360C: `not completed in this run`

## 8. Test trajectory results
- TRAIN360E FINAL360I: `available in reports/TRAIN360E_metrics_test.json`
- SEQ360B: `available in external_baselines/results/seq360b_scale_smoothing_trajectory/`
- SEQ360A: `available in reports/SEQ360A_trajectory_metrics_test.json`
- STRUCT360C: `not completed in this run`
- BASE360D: `available in reports/BASE360_metrics_test.json`

## 9. Diagnostics
- bias stats: `mean=-1.5707554033773388, min=-3.14035626407155, max=-0.0009120791363822642`
- attention entropy: `6.427700396069509`
- gate stats: `{'mean': 0.039118158662059206, 'median': 0.039117392152547836, 'min': 0.03911544010043144, 'max': 0.03912162408232689}`
- residual magnitude: `delta_rot_mean_deg=0.041369106232818705, delta_tdir_norm_mean=2.6737776166856713e-05, delta_log_tmag_abs_mean=7.815955942563342e-06`
- epoch 3 regression: `val metrics contained NaN/Inf, best checkpoint remained epoch 1`

## 10. Analysis
- rotation-aware bias was integrated into fine cross-attention logits: `true`
- training completed and checkpoints were saved: `true`
- full test / trajectory evaluation completed: `false`
- current blocker: `evaluation tail did not finish within reasonable runtime in this environment`

## 11. Recommendation
- `keep_FINAL360I_as_main_and_report_STRUCT360C_ablation`

## 12. Compliance checklist
- `training_executed = true`
- `fine_tune_executed = true`
- `learned_weights_saved = true`
- `final360i_checkpoint_modified = false`
- `explicit_matching_used = false`
- `match_list_output = false`
- `correspondence_list_output = false`
- `topk_matching_used = false`
- `ransac_used = false`
- `pnp_used = false`
- `bundle_adjustment_used = false`
- `hkust_360dvo_teacher_used = false`
- `base360_outputs_used_as_training_input = false`
- `train_manifest_used = true`
- `val_manifest_used_for_selection_only = true`
- `test_manifest_used_for_final_eval_only = false`
- `dset2c_canonical_split_used = true`
- `random_pair_split_used = false`
- `direct_glob_data_360dvo_sequences = false`
- `s5_locked_metrics_modified = false`
- `large_checkpoints_committed_to_git = false`
