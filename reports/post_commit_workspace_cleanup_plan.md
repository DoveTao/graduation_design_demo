# Post-commit Workspace Cleanup Plan
## Current `git status --short`
```text
 D checkpoints/O45_six_hour_cycle_results.md
 D checkpoints/O45_six_hour_cycle_state.json
 D reports/T51_final_selection_summary.md
 D reports/T53_final_selection_summary.md
 D reports/T54_final_candidate_review.md
 D reports/T55a_selection_summary.md
 D reports/f0_fine_matching_ablation.md
 D reports/fig_repro_shape_compare_clean.png
 D reports/fig_t53_shape_compare.png
 D reports/fig_t55a_shape_compare.png
 D reports/o49_tmag_shape_diagnosis.md
 D reports/t51_dt_conditioned_tmag_head_plan.md
 M scripts/run_c31_anchor_smallk.sh
?? reports/checkpoints_live_cleanup_inventory.md
?? reports/checkpoints_live_cleanup_state.json
?? reports/checkpoints_live_cleanup_summary.md
?? reports/cleanup_actions_s1d5.md
?? reports/cleanup_delete_candidates_s1d5.md
?? reports/cleanup_inventory_s1d5.md
?? reports/cleanup_summary_s1d5.md
?? reports/git_commit_s1d5_summary.md
?? reports/git_staging_s1d5_precheck.md
?? reports/git_staging_s1d5_summary.md
```
## Current `git log --oneline -1`
```text
c2d7146 Finalize S1d5 clean mainline artifacts and gitignore
```
## Current tag check
```text
s1d5-clean-mainline
```
## Classification
### LEGACY_DELETIONS_TO_COMMIT
- `checkpoints/O45_six_hour_cycle_results.md`
- `checkpoints/O45_six_hour_cycle_state.json`
- `reports/T51_final_selection_summary.md`
- `reports/T53_final_selection_summary.md`
- `reports/T54_final_candidate_review.md`
- `reports/T55a_selection_summary.md`
- `reports/f0_fine_matching_ablation.md`
- `reports/fig_repro_shape_compare_clean.png`
- `reports/fig_t53_shape_compare.png`
- `reports/fig_t55a_shape_compare.png`
- `reports/o49_tmag_shape_diagnosis.md`
- `reports/t51_dt_conditioned_tmag_head_plan.md`
### CLEANUP_AUDIT_REPORTS_TO_COMMIT
- `reports/checkpoints_live_cleanup_inventory.md`
- `reports/checkpoints_live_cleanup_state.json`
- `reports/checkpoints_live_cleanup_summary.md`
- `reports/cleanup_actions_s1d5.md`
- `reports/cleanup_delete_candidates_s1d5.md`
- `reports/cleanup_inventory_s1d5.md`
- `reports/cleanup_summary_s1d5.md`
- `reports/git_staging_s1d5_precheck.md`
- `reports/git_staging_s1d5_summary.md`
- `reports/git_commit_s1d5_summary.md`
- `reports/post_commit_workspace_cleanup_plan.md`
### RESTORE_OR_REVIEW
- `scripts/run_c31_anchor_smallk.sh`
### IGNORE_OR_ARCHIVE
- historical experiment content already moved under `archive/s1d5_cleanup_20260505/`
## `scripts/run_c31_anchor_smallk.sh` diff summary
This diff is a large historical experiment expansion adding many old `O40*`/`seqturn` helper runs. It is not required for the locked `S1d5` final mainline and should not be mixed into the post-commit cleanup series.
Action: restore this file to the committed version before cleanup commits.

## Notes
- `S1d5` commit/tag is already complete.
- This task is workspace cleanup only, not a new experiment.
- `F1d` and `S1d6` remain paused and will not be run.
