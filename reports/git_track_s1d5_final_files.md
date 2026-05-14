# Suggested Git Track List For S1d5 Final Mainline

This is a recommendation only. No `git add` was run automatically.

## Recommended final tracked files
### Repository policy and core source
- `.gitignore`
- `config.py`
- `interaction.py`
- `losses.py`
- `model.py`
- `train_mvp.py`
- `scripts/eval_checkpoint_trajectory_repro.py`
- `scripts/eval_s1d5_clean_policy.sh`
- `tools/eval_clean_policy.py`

### Final reports
- `reports/current_valid_baselines.md`
- `reports/final_s1d5_mainline_report.md`
- `reports/final_s1d5_thesis_summary.md`
- `reports/final_s1d5_english_abstract.md`
- `reports/final_s1d5_results_table.csv`
- `reports/gitignore_cleanup_plan.md`
- `reports/git_track_s1d5_final_files.md`

### Final checkpoint-side documents and policy artifacts
- `checkpoints/S1d5_clean_dt_anchor_policy.json`
- `checkpoints/S1d5_final_baseline_comparison.md`
- `checkpoints/S1d5_final_mainline_summary.md`
- `checkpoints/S1d5_freeze_clean_dt_anchor_policy_export_report.md`
- `checkpoints/S1d6_minimal_trainable_dt_anchor_head_proposal.md`
- `checkpoints/S1d5_final_repro/s1d5_policy_eval_summary.json`

### Final figures
- `checkpoints/S1d5_final_figures/baseline_bar_path_ratio.png`
- `checkpoints/S1d5_final_figures/baseline_bar_ATE_drift.png`
- `checkpoints/S1d5_final_figures/tmag_distribution_s1d5.png`
- `checkpoints/S1d5_final_figures/trajectory_comparison_s1d5.png`
- `checkpoints/S1d5_final_figures/scale_repair_summary.png`

## Recommended hold-out from normal Git
### Base checkpoint weights
- `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- other `.pt` files under `checkpoints/T57b_no_dt_multiscale_tmag_head_400/`

Reason:
- `final.pt` is `91,067,239` bytes.
- These are binary model artifacts, not source.
- They should be stored externally or via Git LFS, not normal Git, unless your repo policy explicitly allows large binaries.

## Suggested command
```bash
git add \
  .gitignore \
  config.py interaction.py losses.py model.py train_mvp.py \
  scripts/eval_checkpoint_trajectory_repro.py \
  scripts/eval_s1d5_clean_policy.sh \
  tools/eval_clean_policy.py \
  reports/current_valid_baselines.md \
  reports/final_s1d5_mainline_report.md \
  reports/final_s1d5_thesis_summary.md \
  reports/final_s1d5_english_abstract.md \
  reports/final_s1d5_results_table.csv \
  reports/gitignore_cleanup_plan.md \
  reports/git_track_s1d5_final_files.md \
  checkpoints/S1d5_clean_dt_anchor_policy.json \
  checkpoints/S1d5_final_baseline_comparison.md \
  checkpoints/S1d5_final_mainline_summary.md \
  checkpoints/S1d5_freeze_clean_dt_anchor_policy_export_report.md \
  checkpoints/S1d6_minimal_trainable_dt_anchor_head_proposal.md \
  checkpoints/S1d5_final_repro/s1d5_policy_eval_summary.json \
  checkpoints/S1d5_final_figures/*.png
```

## Notes
- This recommendation intentionally excludes historical scalar-load baselines.
- This recommendation intentionally excludes `archive/`.
- This recommendation intentionally excludes the large `T57b` weight files from normal Git.
- If the team wants versioned checkpoint transport, use Git LFS for `final.pt` instead of normal Git.
