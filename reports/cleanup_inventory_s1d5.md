# Cleanup Inventory S1d5

## 1. git status --short

```text
 M config.py
 M interaction.py
 M losses.py
 M model.py
 M scripts/eval_checkpoint_trajectory_repro.py
 M scripts/run_c31_anchor_smallk.sh
 M train_mvp.py
?? reports/T57c_final_selection_summary.md
?? reports/current_valid_baselines.md
?? reports/final_s1d5_english_abstract.md
?? reports/final_s1d5_mainline_report.md
?? reports/final_s1d5_results_table.csv
?? reports/final_s1d5_thesis_summary.md
?? scripts/eval_s1d5_clean_policy.sh
?? scripts/run_eight_hour_odom_cycle.sh
?? scripts/run_f1a_fine_dir_refine.sh
?? scripts/run_f1b_fine_geom_refine_eval.sh
?? scripts/run_f1c0_fine_failure_attribution_resume.sh
?? scripts/run_f1d_frozen_coarse_rot_only.sh
?? scripts/run_t57a_clean_no_dt.sh
?? scripts/run_t57b_no_dt_multiscale_tmag_head.sh
?? scripts/run_t57c_ridge_frozen_tmag_head.sh
?? scripts/run_t57c_ridge_init_tmag_head.sh
?? tools/audit_tmag_feature_source.py
?? tools/eval_clean_policy.py
?? tools/f1c0_fine_failure_attribution.py
?? tools/s1_true_multiscale_scale_anchor.py
?? tools/s1c_s1d_followup.py
?? tools/s1d2_learnable_dt_bucket_chain_anchor.py
?? tools/s1d3_rot_plus_dt_bucket_anchor_strength_sweep.py
?? tools/s1d4_train_selected_dt_anchor_plus_rot_policy.py
?? tools/s1d5_finalization_and_visualization.py
?? tools/t57c0d_batch_aligned_ridge_audit.py
?? tools/t57c0e_full_batch_chain_cv_ridge.py
?? tools/t57d_sequence_scale_calibration.py
?? tools/t57e2_sequence_chain_path_ratio_calibration.py
?? tools/t57e_constrained_ridge_calibration_head.py
?? tools/t57e_fit_calib_init.py
?? tools/t59_trajectory_latent_scale_optimization.py
```

## 2. checkpoints/ size summary

```text
64K\tcheckpoints/S1d5_clean_policy_eval
64K\tcheckpoints/S1d5_final_repro
92K\tcheckpoints/BATCH_SHAPE_REPORT_T55A_COMPARE
112K\tcheckpoints/BATCH_SHAPE_REPORT_T53A4_COMPARE
120K\tcheckpoints/BATCH_SHAPE_REPORT_T53B_COMPARE
200K\tcheckpoints/S1d5_final_figures
308K\tcheckpoints/F1a_reval_best_eval_joint
308K\tcheckpoints/F1a_reval_best_joint
308K\tcheckpoints/F1a_reval_eval_upd_0100
308K\tcheckpoints/F1a_reval_final
308K\tcheckpoints/F1a_reval_fine_final
308K\tcheckpoints/F1c0_F1a_final_r0p00_t0p00
308K\tcheckpoints/F1c0_F1a_final_r0p00_t0p10
308K\tcheckpoints/F1c0_F1a_final_r0p00_t0p25
308K\tcheckpoints/F1c0_F1a_final_r0p10_t0p00
308K\tcheckpoints/F1c0_F1a_final_r0p10_t0p10
308K\tcheckpoints/F1c0_F1a_final_r0p25_t0p25
312K\tcheckpoints/F1a_reval_fine_best_eval_joint
312K\tcheckpoints/F1a_reval_fine_best_joint
312K\tcheckpoints/F1a_reval_fine_eval_upd_0100
312K\tcheckpoints/F1c0_F1a_eval_upd_0100_r0p00_t0p00
312K\tcheckpoints/F1c0_F1a_eval_upd_0100_r0p00_t0p05
312K\tcheckpoints/F1c0_F1a_eval_upd_0100_r0p00_t0p10
312K\tcheckpoints/F1c0_F1a_eval_upd_0100_r0p00_t0p25
312K\tcheckpoints/F1c0_F1a_eval_upd_0100_r0p05_t0p00
312K\tcheckpoints/F1c0_F1a_eval_upd_0100_r0p05_t0p05
312K\tcheckpoints/F1c0_F1a_eval_upd_0100_r0p10_t0p00
312K\tcheckpoints/F1c0_F1a_eval_upd_0100_r0p10_t0p10
312K\tcheckpoints/F1c0_F1a_eval_upd_0100_r0p25_t0p00
312K\tcheckpoints/F1c0_F1a_eval_upd_0100_r0p25_t0p25
312K\tcheckpoints/F1c0_F1a_final_r0p00_t0p05
312K\tcheckpoints/F1c0_F1a_final_r0p05_t0p00
312K\tcheckpoints/F1c0_F1a_final_r0p05_t0p05
312K\tcheckpoints/F1c0_F1a_final_r0p25_t0p00
312K\tcheckpoints/F1d_reval_best_eval_joint
312K\tcheckpoints/F1d_reval_best_joint
312K\tcheckpoints/F1d_reval_eval_upd_0100
312K\tcheckpoints/F1d_reval_final
312K\tcheckpoints/F1d_reval_zero_eval_upd_0100
312K\tcheckpoints/F1d_reval_zero_final
388K\tcheckpoints/S1_followup_runs
396K\tcheckpoints/F1c0_T57b_final_r0p50_t0p25
476K\tcheckpoints/T57b_harness_multiscale_k5_pref5
516K\tcheckpoints/S1_runs
544K..8.7G omitted for brevity in report body; see terminal audit used to generate this file.
```

## 3. reports/ file list

```text
reports/T51_final_selection_summary.md
reports/T53_final_selection_summary.md
reports/T54_final_candidate_review.md
reports/T55a_selection_summary.md
reports/T57c_final_selection_summary.md
reports/current_valid_baselines.md
reports/f0_fine_matching_ablation.md
reports/fig_repro_shape_compare_clean.png
reports/fig_t53_shape_compare.png
reports/fig_t55a_shape_compare.png
reports/final_s1d5_english_abstract.md
reports/final_s1d5_mainline_report.md
reports/final_s1d5_results_table.csv
reports/final_s1d5_thesis_summary.md
reports/o49_tmag_shape_diagnosis.md
reports/t51_dt_conditioned_tmag_head_plan.md
```

## 4. scripts/ candidate list

```text
scripts/batch_eval_shape_report.py
scripts/check_triplet_transform_composition.py
scripts/eval_checkpoint_trajectory_repro.py
scripts/eval_s1d5_clean_policy.sh
scripts/inspect_o49_shape_debug.py
scripts/inspect_tmag_buckets.py
scripts/plot_odom_trajectory_debug.py
scripts/run_c31_anchor_smallk.sh
scripts/run_candidate_eval.sh
scripts/run_curriculum_odom.sh
scripts/run_debug_smoke.sh
scripts/run_effect_ablation.sh
scripts/run_eight_hour_odom_cycle.sh
scripts/run_f1a_fine_dir_refine.sh
scripts/run_f1b_fine_geom_refine_eval.sh
scripts/run_f1c0_fine_failure_attribution_resume.sh
scripts/run_f1d_frozen_coarse_rot_only.sh
scripts/run_fine_matching_ablation.sh
scripts/run_fine_stage.sh
scripts/run_o49_seqturn_focus.sh
scripts/run_odom_small_k.sh
scripts/run_six_hour_odom_cycle.sh
scripts/run_t50_tmag_affine_probe.sh
scripts/run_t57a_clean_no_dt.sh
scripts/run_t57b_no_dt_multiscale_tmag_head.sh
scripts/run_t57c_ridge_frozen_tmag_head.sh
scripts/run_t57c_ridge_init_tmag_head.sh
scripts/run_trajectory_debug_eval.sh
scripts/run_wide_baseline.sh
scripts/select_shape_aware_checkpoint.py
scripts/summarize_odom_cycle.py
```

## 5. tools/ candidate list

```text
tools/audit_tmag_feature_source.py
tools/buckets_to_csv.py
tools/eval_clean_policy.py
tools/f1c0_fine_failure_attribution.py
tools/s1_true_multiscale_scale_anchor.py
tools/s1c_s1d_followup.py
tools/s1d2_learnable_dt_bucket_chain_anchor.py
tools/s1d3_rot_plus_dt_bucket_anchor_strength_sweep.py
tools/s1d4_train_selected_dt_anchor_plus_rot_policy.py
tools/s1d5_finalization_and_visualization.py
tools/summarize_experiments.py
tools/t57c0d_batch_aligned_ridge_audit.py
tools/t57c0e_full_batch_chain_cv_ridge.py
tools/t57d_sequence_scale_calibration.py
tools/t57e2_sequence_chain_path_ratio_calibration.py
tools/t57e_constrained_ridge_calibration_head.py
tools/t57e_fit_calib_init.py
tools/t59_trajectory_latent_scale_optimization.py
```

## 6. Classification

### KEEP_FINAL
- checkpoints/S1d5_clean_dt_anchor_policy.json
- checkpoints/S1d5_final_repro
- checkpoints/S1d5_final_figures
- checkpoints/S1d5_final_baseline_comparison.md
- checkpoints/S1d5_final_mainline_summary.md
- checkpoints/S1d5_freeze_clean_dt_anchor_policy_export_report.md
- reports/current_valid_baselines.md
- reports/final_s1d5_mainline_report.md
- reports/final_s1d5_thesis_summary.md
- reports/final_s1d5_english_abstract.md
- reports/final_s1d5_results_table.csv
- tools/eval_clean_policy.py
- scripts/eval_s1d5_clean_policy.sh
- checkpoints/S1d6_minimal_trainable_dt_anchor_head_proposal.md
- checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt
- checkpoints/T57b_no_dt_multiscale_tmag_head_400 (directory metadata kept intact)

### DO_NOT_TOUCH
- config.py
- interaction.py
- losses.py
- model.py
- train_mvp.py
- scripts/eval_checkpoint_trajectory_repro.py
- scripts/run_c31_anchor_smallk.sh

### KEEP_REFERENCE
- checkpoints/T57b_eval_harness_audit_report.md
- checkpoints/T57b_F1c_baseline_relabel_report.md
- checkpoints/F1d_T57b_init_reproduction_audit.md
- checkpoints/F1c_eval_rot_only_fusion_baseline.md
- reports/final_s1d5_* (all final outputs already in KEEP_FINAL)

### ARCHIVE_ONLY
- All `checkpoints/F1a*`, `checkpoints/F1b*`, `checkpoints/F1c*`, `checkpoints/F1d*`
- All `checkpoints/T57c*`, `checkpoints/T57d*`, `checkpoints/T57e*`
- checkpoints/S1_runs
- checkpoints/S1_followup_runs
- checkpoints/S1d2_runs
- checkpoints/S1d3_runs
- checkpoints/S1d4_runs
- checkpoints/S1c0_pairwise_bias_control_report.md
- checkpoints/S1c_S1d_scale_anchor_followup_plan.md
- checkpoints/S1d0_offline_train_chain_scale_report.md
- checkpoints/S1d1_dt_bucket_train_chain_scale_report.md
- checkpoints/S1d2_best_plus_rot_only_eval_report.md
- checkpoints/S1d2_learnable_dt_bucket_chain_anchor_report.md
- checkpoints/S1d3_rot_plus_dt_bucket_anchor_strength_sweep_report.md
- checkpoints/S1_ratio_definition_audit.md
- checkpoints/S1_scale_anchor_candidates.md
- checkpoints/S1_tmag_pathratio_audit.md
- checkpoints/S1_true_multiscale_scale_anchor_plan.md
- checkpoints/S1a_global_train_calib_eval_report.md
- checkpoints/S1e_constant_scale_probe_report.md
- checkpoints/S1d4_train_selected_dt_anchor_plus_rot_policy_report.md
- checkpoints/S1d5_clean_policy_eval
- reports/T51_final_selection_summary.md
- reports/T53_final_selection_summary.md
- reports/T54_final_candidate_review.md
- reports/T55a_selection_summary.md
- reports/T57c_final_selection_summary.md
- reports/f0_fine_matching_ablation.md
- reports/o49_tmag_shape_diagnosis.md
- reports/t51_dt_conditioned_tmag_head_plan.md
- reports/fig_repro_shape_compare_clean.png
- reports/fig_t53_shape_compare.png
- reports/fig_t55a_shape_compare.png
- scripts/run_f1a_fine_dir_refine.sh
- scripts/run_f1b_fine_geom_refine_eval.sh
- scripts/run_f1c0_fine_failure_attribution_resume.sh
- scripts/run_f1d_frozen_coarse_rot_only.sh
- scripts/run_t57a_clean_no_dt.sh
- scripts/run_t57b_no_dt_multiscale_tmag_head.sh
- scripts/run_t57c_ridge_frozen_tmag_head.sh
- scripts/run_t57c_ridge_init_tmag_head.sh
- scripts/run_eight_hour_odom_cycle.sh
- tools/f1c0_fine_failure_attribution.py
- tools/audit_tmag_feature_source.py
- tools/s1_true_multiscale_scale_anchor.py
- tools/s1c_s1d_followup.py
- tools/s1d2_learnable_dt_bucket_chain_anchor.py
- tools/s1d3_rot_plus_dt_bucket_anchor_strength_sweep.py
- tools/s1d4_train_selected_dt_anchor_plus_rot_policy.py
- tools/s1d5_finalization_and_visualization.py
- tools/t57c0d_batch_aligned_ridge_audit.py
- tools/t57c0e_full_batch_chain_cv_ridge.py
- tools/t57d_sequence_scale_calibration.py
- tools/t57e2_sequence_chain_path_ratio_calibration.py
- tools/t57e_constrained_ridge_calibration_head.py
- tools/t57e_fit_calib_init.py
- tools/t59_trajectory_latent_scale_optimization.py

### DELETE_CANDIDATE
- `__pycache__/`
- `scripts/__pycache__/`
- `tools/__pycache__/`
- No permanent deletions proposed in round 1; these are cache-only candidates.

## Proposed archive moves (dry-run)
- archive all F1 history listed above
- archive all T57c/T57d/T57e diagnostic outputs and scripts/tools listed above
- archive S1 intermediate runs, reports, and helper tools, while keeping only S1d5 final artifacts and S1d6 proposal in-place
- archive obsolete/older report figures and selection summaries

## Proposed delete candidates (dry-run)
- cache directories only; no permanent deletion in this round
