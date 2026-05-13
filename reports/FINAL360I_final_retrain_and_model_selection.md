# FINAL360I final retrain and model selection

## 1. Executive summary
- final retrain executed true/false: `true`
- number of seeds: `1`
- selected checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
- whether selected model is FINAL360I or fallback STRUCT360B: `FINAL360I`
- final classification: `final_balanced_model`
- main thesis model recommendation: `FINAL360I_struct360b_final_selected`

## 2. Baseline recap
| model | split | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| T57b | reference | 111.96493221327962 | 0.6746724890829694 | 0.1751560082454769 | 0.14878731297064046 |
| TRAIN360C | test | 54.956992341554056 | 0.21572500987751878 | 0.7692854374461574 | 0.5888660831060318 |
| TRAIN360D | test | 44.986174454415895 | 0.19695772421967603 | 0.7572524310356576 | 0.5787942300950458 |
| TRAIN360H | test | 53.73375854133177 | 0.20209403397866457 | 0.8503917514492008 | 0.6493755813576354 |
| STRUCT360A | test | 45.806573800840134 | 0.21197155274595023 | 0.7683920813313971 | 0.5926014164348259 |
| STRUCT360B | test | 45.924701790734524 | 0.2042670881074674 | 0.8509755191001558 | 0.656656775908536 |
| BASE360D | test component | 128.40257804384538 | 0.8148952983010668 | 0.039909732236488166 | 0.04354171873324947 |

## 3. Final retrain setup
- branch: `experiment/final360i-final-retrain-and-model-selection`
- git commit: `c4712f78f5df20dc849a15be706419ccb846a881`
- config: `configs/final360i_struct360b_final.yaml`
- init checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/TRAIN360D_observability_kstep_scale/best_val.pt`
- seeds: `[0]`
- epochs: `5`
- staged training: `same as STRUCT360B`
- no architecture changes: `true`
- no hyperparameter sweep: `true`

## 4. Model selection protocol
- val score formula: `signed_tdir_mean + 60 * anti_parallel_rate + 20 * abs(log(tmag_median_ratio + eps)) + 10 * abs(log(path_ratio + eps))`
- val used for selection: `true`
- test used for selection: `false`
- selected seed/checkpoint: `seed0 -> /home/dovetao/graduation_design_demo/checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`

## 5. Seed robustness
- seed policy: `resource_limited_single_seed`
- per-seed summary: `[{'seed': 0, 'best_epoch': 1, 'val_score': 149.8338157649262, 'val_signed_tdir_mean_deg': 104.46986810553231, 'val_anti_parallel_rate': 0.6593036315986522, 'val_tmag_median_ratio': 1.0006463292736378, 'val_path_ratio': 0.5603012265255422, 'checkpoint': '/home/dovetao/graduation_design_demo/checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt'}]`
- note: `single-seed run due to time/disk resource guardrail; robustness sweep intentionally skipped.`

## 6. Final test results
- final model: `FINAL360I_struct360b_final_selected`
- final test metrics: `{'count': 5062, 'coverage': 1.0, 'rot_mean_deg': 2.330108616583517, 'rot_median_deg': 0.20558422803878784, 'rot_p90_deg': 6.713612318038943, 'signed_tdir_mean_deg': 45.264702006380205, 'signed_tdir_median_deg': 22.704615219747595, 'signed_tdir_p90_deg': 140.27662565571296, 'unsigned_tdir_mean_deg': 25.937378929700255, 'unsigned_tdir_median_deg': 19.432276054381838, 'unsigned_tdir_p90_deg': 58.16424864012134, 'anti_parallel_rate': 0.20169893322797314, 'tmag_ratio_p10': 0.2966194117283973, 'tmag_median_ratio': 0.8344251368086006, 'tmag_ratio_p50': 0.8344251368086006, 'tmag_mean_ratio': 1.1036085340991515, 'tmag_ratio_p90': 2.4225009459219193, 'tmag_p90_ratio': 2.4225009459219193, 'log_tmag_mae': 0.6559796168700758, 'scale_collapse_rate': 0.0019755037534571317, 'scale_explosion_rate': 0.0, 'path_ratio': 0.6403519796204528, 'path_length_pred': 5221.007291018963, 'path_length_gt': 8153.339814946055, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`
- retrain selected candidate test metrics: `{'count': 5062, 'coverage': 1.0, 'rot_mean_deg': 2.330108616583517, 'rot_median_deg': 0.20558422803878784, 'rot_p90_deg': 6.713612318038943, 'signed_tdir_mean_deg': 45.264702006380205, 'signed_tdir_median_deg': 22.704615219747595, 'signed_tdir_p90_deg': 140.27662565571296, 'unsigned_tdir_mean_deg': 25.937378929700255, 'unsigned_tdir_median_deg': 19.432276054381838, 'unsigned_tdir_p90_deg': 58.16424864012134, 'anti_parallel_rate': 0.20169893322797314, 'tmag_ratio_p10': 0.2966194117283973, 'tmag_median_ratio': 0.8344251368086006, 'tmag_ratio_p50': 0.8344251368086006, 'tmag_mean_ratio': 1.1036085340991515, 'tmag_ratio_p90': 2.4225009459219193, 'tmag_p90_ratio': 2.4225009459219193, 'log_tmag_mae': 0.6559796168700758, 'scale_collapse_rate': 0.0019755037534571317, 'scale_explosion_rate': 0.0, 'path_ratio': 0.6403519796204528, 'path_length_pred': 5221.007291018963, 'path_length_gt': 8153.339814946055, 'nan_inf_count': 0, 'nan_count': 0, 'inf_count': 0, 'ate_none': None, 'ate_se3': None, 'ate_sim3': None}`
- comparison to STRUCT360B: `comparable`
- comparison to TRAIN360D: `partial`
- comparison to TRAIN360H: `partial`
- comparison to T57b/BASE360D: `better than both`

## 7. Final model decision
- main model: `FINAL360I_struct360b_final_selected`
- checkpoint path: `/home/dovetao/graduation_design_demo/checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
- reason: `FINAL360I retrain remained balanced and comparable to or better than original STRUCT360B, so the retrained checkpoint is recommended as the thesis main model.`
- fallback if needed: `not needed`

## 8. Thesis-ready claim
- English paragraph: see `reports/FINAL360I_thesis_ready_result_paragraph.md`.
- Chinese paragraph: see `reports/FINAL360I_thesis_ready_result_paragraph.md`.

## 9. Caveats
- pair-level metrics only so far
- trajectory ATE still pending
- BASE360D component metrics are trajectory-derived
- final model selected by val, not test

## 10. Next recommendation
- `prepare_thesis_experiment_section`

## 11. Compliance checklist
- `real_training_executed = true`
- `learned_weights_saved = true`
- `train360c_checkpoint_modified = false`
- `train360d_checkpoint_modified = false`
- `train360h_checkpoint_modified = false`
- `struct360a_checkpoint_modified = false`
- `struct360b_checkpoint_modified = false`
- `architecture_changed = false`
- `hyperparameter_sweep_executed = false`
- `test_used_for_model_selection = false`
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
- `direct_glob_data_360dvo_sequences = false`
- `random_pair_split_used = false`
- `s5_locked_metrics_modified = false`
- `large_checkpoints_committed_to_git = false`
