# Gitignore Cleanup Plan For S1d5 Mainline

## Current `.gitignore` Review
The repository now uses scoped ignore rules instead of a blanket `checkpoints/` ignore. This keeps source, final S1d5 artifacts, and final reports visible to Git while hiding heavy checkpoints, archives, caches, and generated debug files.

## Git Status Snapshot
### Before `.gitignore` cleanup
- `S1d5` final artifacts under `checkpoints/` were being ignored together with all other checkpoint content.
- The base `T57b` checkpoint directory and `S1d5` final policy files were not cleanly separable in `git status`.

### After `.gitignore` cleanup
- `S1d5` final artifacts are visible to Git as untracked files that can be intentionally added.
- Heavy checkpoint payloads remain ignored.
- `archive/` remains ignored.
- `T57b` base checkpoint weights remain ignored, which is preferred for normal Git.

## Current `git status --short`
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

## Checkpoints Size Snapshot
```text
64K  checkpoints/S1d5_final_repro
200K checkpoints/S1d5_final_figures
454M checkpoints/T57b_no_dt_multiscale_tmag_head_400
454M checkpoints
```

## Classification
### TRACK_FINAL
- `.gitignore`
- `checkpoints/S1d5_clean_dt_anchor_policy.json`
- `checkpoints/S1d5_final_baseline_comparison.md`
- `checkpoints/S1d5_final_mainline_summary.md`
- `checkpoints/S1d5_freeze_clean_dt_anchor_policy_export_report.md`
- `checkpoints/S1d6_minimal_trainable_dt_anchor_head_proposal.md`
- `checkpoints/S1d5_final_figures/*.png`
- `checkpoints/S1d5_final_repro/s1d5_policy_eval_summary.json`
- `reports/current_valid_baselines.md`
- `reports/final_s1d5_mainline_report.md`
- `reports/final_s1d5_thesis_summary.md`
- `reports/final_s1d5_english_abstract.md`
- `reports/final_s1d5_results_table.csv`
- `scripts/eval_s1d5_clean_policy.sh`
- `tools/eval_clean_policy.py`

### TRACK_SOURCE
- `config.py`
- `interaction.py`
- `losses.py`
- `model.py`
- `train_mvp.py`
- `scripts/eval_checkpoint_trajectory_repro.py`
- `scripts/run_c31_anchor_smallk.sh`

### IGNORE_GENERATED
- `__pycache__/`
- `.pytest_cache/`
- `.mypy_cache/`
- `.ruff_cache/`
- `logs/`
- `runs/`
- `wandb/`
- `tmp/`
- `temp/`
- `reports/tmp/`
- `reports/drafts/`
- `checkpoints/**/odom_trajectory_debug*.json`
- `checkpoints/**/odom_trajectory_debug*.npz`
- `checkpoints/**/odom_trajectory_steps*.csv`
- `checkpoints/**/eval_history*.json`
- `checkpoints/**/eval_buckets*.json`
- `checkpoints/**/eval_buckets*.csv`
- `checkpoints/**/matching_diag*.json`
- `checkpoints/**/odom_metrics*.json`
- `checkpoints/**/eval_pairs_manifest.json`
- `checkpoints/**/final_summary.json`

### IGNORE_HEAVY
- `archive/`
- `checkpoints/**/*.pt`
- `checkpoints/**/*.pth`
- `checkpoints/**/*.ckpt`
- `checkpoints/**/*.npz`
- `checkpoints_smoke/`
- `data/`
- `PanoramaView/`
- `FisheyeView/`

### REVIEW_MANUAL
- `checkpoints/T57b_no_dt_multiscale_tmag_head_400/`
  - Reason: needed for local reproducibility, but `final.pt` is 91,067,239 bytes and should not go into normal Git.
  - Recommendation: store externally or via Git LFS if the team needs versioned weight transport.
- `reports/checkpoints_live_cleanup_state.json`
  - Reason: generated state artifact; track only if you want repo-level cleanup provenance.
- Archived rename/deletion entries already present in `git status`
  - Reason: this is a repository-history choice, not a `.gitignore` choice.

## Reports Directory Snapshot
- `reports/checkpoints_live_cleanup_inventory.md`
- `reports/checkpoints_live_cleanup_state.json`
- `reports/checkpoints_live_cleanup_summary.md`
- `reports/cleanup_actions_s1d5.md`
- `reports/cleanup_delete_candidates_s1d5.md`
- `reports/cleanup_inventory_s1d5.md`
- `reports/cleanup_summary_s1d5.md`
- `reports/current_valid_baselines.md`
- `reports/final_s1d5_english_abstract.md`
- `reports/final_s1d5_mainline_report.md`
- `reports/final_s1d5_results_table.csv`
- `reports/final_s1d5_thesis_summary.md`

## Scripts Snapshot
- `scripts/batch_eval_shape_report.py`
- `scripts/check_triplet_transform_composition.py`
- `scripts/eval_checkpoint_trajectory_repro.py`
- `scripts/eval_s1d5_clean_policy.sh`
- `scripts/inspect_o49_shape_debug.py`
- `scripts/inspect_tmag_buckets.py`
- `scripts/plot_odom_trajectory_debug.py`
- `scripts/run_c31_anchor_smallk.sh`
- `scripts/run_candidate_eval.sh`
- `scripts/run_curriculum_odom.sh`
- `scripts/run_debug_smoke.sh`
- `scripts/run_effect_ablation.sh`
- `scripts/run_fine_matching_ablation.sh`
- `scripts/run_fine_stage.sh`
- `scripts/run_o49_seqturn_focus.sh`
- `scripts/run_odom_small_k.sh`
- `scripts/run_six_hour_odom_cycle.sh`
- `scripts/run_t50_tmag_affine_probe.sh`
- `scripts/run_trajectory_debug_eval.sh`
- `scripts/run_wide_baseline.sh`
- `scripts/select_shape_aware_checkpoint.py`
- `scripts/summarize_odom_cycle.py`

## Tools Snapshot
- `tools/buckets_to_csv.py`
- `tools/eval_clean_policy.py`
- `tools/summarize_experiments.py`

## Recommendation
- Track source, `S1d5` final artifacts, and final reports.
- Keep the base `T57b` checkpoint outside normal Git; prefer external storage or Git LFS.
- Do not auto-add archived rename/deletion changes unless you intentionally want the repository history rewritten to match the cleanup move.
