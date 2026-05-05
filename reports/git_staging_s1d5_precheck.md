# Git Staging Precheck For S1d5

## `git status --short`
```text
 M .gitignore
R  reports/T51_final_selection_summary.md -> archive/s1d5_cleanup_20260505/reports/T51_final_selection_summary.md
R  reports/T53_final_selection_summary.md -> archive/s1d5_cleanup_20260505/reports/T53_final_selection_summary.md
R  reports/T54_final_candidate_review.md -> archive/s1d5_cleanup_20260505/reports/T54_final_candidate_review.md
R  reports/T55a_selection_summary.md -> archive/s1d5_cleanup_20260505/reports/T55a_selection_summary.md
R  reports/f0_fine_matching_ablation.md -> archive/s1d5_cleanup_20260505/reports/f0_fine_matching_ablation.md
R  reports/fig_repro_shape_compare_clean.png -> archive/s1d5_cleanup_20260505/reports/fig_repro_shape_compare_clean.png
R  reports/fig_t53_shape_compare.png -> archive/s1d5_cleanup_20260505/reports/fig_t53_shape_compare.png
R  reports/fig_t55a_shape_compare.png -> archive/s1d5_cleanup_20260505/reports/fig_t55a_shape_compare.png
R  reports/o49_tmag_shape_diagnosis.md -> archive/s1d5_cleanup_20260505/reports/o49_tmag_shape_diagnosis.md
R  reports/t51_dt_conditioned_tmag_head_plan.md -> archive/s1d5_cleanup_20260505/reports/t51_dt_conditioned_tmag_head_plan.md
 D checkpoints/O45_six_hour_cycle_results.md
 D checkpoints/O45_six_hour_cycle_state.json
 M config.py
 M interaction.py
 M losses.py
 M model.py
 M scripts/eval_checkpoint_trajectory_repro.py
 M scripts/run_c31_anchor_smallk.sh
 M train_mvp.py
?? checkpoints/S1d5_clean_dt_anchor_policy.json
?? checkpoints/S1d5_final_baseline_comparison.md
?? checkpoints/S1d5_final_figures/
?? checkpoints/S1d5_final_mainline_summary.md
?? checkpoints/S1d5_final_repro/
?? checkpoints/S1d5_freeze_clean_dt_anchor_policy_export_report.md
?? checkpoints/S1d6_minimal_trainable_dt_anchor_head_proposal.md
?? reports/
?? scripts/eval_s1d5_clean_policy.sh
?? tools/eval_clean_policy.py
```

## `git status --short --ignored`
```text
 M .gitignore
R  reports/T51_final_selection_summary.md -> archive/s1d5_cleanup_20260505/reports/T51_final_selection_summary.md
R  reports/T53_final_selection_summary.md -> archive/s1d5_cleanup_20260505/reports/T53_final_selection_summary.md
R  reports/T54_final_candidate_review.md -> archive/s1d5_cleanup_20260505/reports/T54_final_candidate_review.md
R  reports/T55a_selection_summary.md -> archive/s1d5_cleanup_20260505/reports/T55a_selection_summary.md
R  reports/f0_fine_matching_ablation.md -> archive/s1d5_cleanup_20260505/reports/f0_fine_matching_ablation.md
R  reports/fig_repro_shape_compare_clean.png -> archive/s1d5_cleanup_20260505/reports/fig_repro_shape_compare_clean.png
R  reports/fig_t53_shape_compare.png -> archive/s1d5_cleanup_20260505/reports/fig_t53_shape_compare.png
R  reports/fig_t55a_shape_compare.png -> archive/s1d5_cleanup_20260505/reports/fig_t55a_shape_compare.png
R  reports/o49_tmag_shape_diagnosis.md -> archive/s1d5_cleanup_20260505/reports/o49_tmag_shape_diagnosis.md
R  reports/t51_dt_conditioned_tmag_head_plan.md -> archive/s1d5_cleanup_20260505/reports/t51_dt_conditioned_tmag_head_plan.md
 D checkpoints/O45_six_hour_cycle_results.md
 D checkpoints/O45_six_hour_cycle_state.json
 M config.py
 M interaction.py
 M losses.py
 M model.py
 M scripts/eval_checkpoint_trajectory_repro.py
 M scripts/run_c31_anchor_smallk.sh
 M train_mvp.py
?? checkpoints/S1d5_clean_dt_anchor_policy.json
?? checkpoints/S1d5_final_baseline_comparison.md
?? checkpoints/S1d5_final_figures/
?? checkpoints/S1d5_final_mainline_summary.md
?? checkpoints/S1d5_final_repro/
?? checkpoints/S1d5_freeze_clean_dt_anchor_policy_export_report.md
?? checkpoints/S1d6_minimal_trainable_dt_anchor_head_proposal.md
?? reports/
?? scripts/eval_s1d5_clean_policy.sh
?? tools/eval_clean_policy.py
!! .codex
!! .vscode/
!! __pycache__/
!! archive/s1d5_cleanup_20260505/checkpoints/
!! archive/s1d5_cleanup_20260505/reports/T57c_final_selection_summary.md
!! archive/s1d5_cleanup_20260505/scripts/
!! archive/s1d5_cleanup_20260505/tools/
!! check_k_dist.py
!! check_labels_pose.py
!! checkpoints/S1d5_final_repro/odom_trajectory_debug_latest.json
!! checkpoints/S1d5_final_repro/odom_trajectory_debug_latest.npz
!! checkpoints/S1d5_final_repro/odom_trajectory_steps_latest.csv
!! checkpoints/T57b_no_dt_multiscale_tmag_head_400/
!! checkpoints_smoke/
!! codex_changes.patch
!! config.md
!! data/
!! logs/
!! scripts/__pycache__/
!! tools/__pycache__/
```

## Required File Existence Check
- Missing count: `0`
- Result: all required S1d5 final/mainline files are present.

## Notes
- No staging was performed before this precheck report was written.
- Base checkpoint weights remain ignored and are not part of the staging whitelist.
