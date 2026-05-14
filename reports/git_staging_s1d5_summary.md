# Git Staging Summary For S1d5

## Staged File List
```text
.gitignore
checkpoints/S1d5_clean_dt_anchor_policy.json
checkpoints/S1d5_final_baseline_comparison.md
checkpoints/S1d5_final_figures/baseline_bar_ATE_drift.png
checkpoints/S1d5_final_figures/baseline_bar_path_ratio.png
checkpoints/S1d5_final_figures/scale_repair_summary.png
checkpoints/S1d5_final_figures/tmag_distribution_s1d5.png
checkpoints/S1d5_final_figures/trajectory_comparison_s1d5.png
checkpoints/S1d5_final_mainline_summary.md
checkpoints/S1d5_final_repro/s1d5_policy_eval_summary.json
checkpoints/S1d5_freeze_clean_dt_anchor_policy_export_report.md
checkpoints/S1d6_minimal_trainable_dt_anchor_head_proposal.md
config.py
interaction.py
losses.py
model.py
reports/current_valid_baselines.md
reports/final_s1d5_english_abstract.md
reports/final_s1d5_mainline_report.md
reports/final_s1d5_results_table.csv
reports/final_s1d5_thesis_summary.md
reports/git_track_s1d5_final_files.md
reports/gitignore_cleanup_plan.md
scripts/eval_checkpoint_trajectory_repro.py
scripts/eval_s1d5_clean_policy.sh
tools/eval_clean_policy.py
train_mvp.py
```

## Staged File Count
- `27`

## Approximate Staged Size
- Approximate on-disk size of explicitly staged whitelist files: `603830` bytes (`~589.7 KiB`)
- `git diff --cached --stat`:
```text
 .gitignore                                         |  55 +++-
 checkpoints/S1d5_clean_dt_anchor_policy.json       |  35 +++
 checkpoints/S1d5_final_baseline_comparison.md      |  13 +
 .../S1d5_final_figures/baseline_bar_ATE_drift.png  | Bin 0 -> 36095 bytes
 .../S1d5_final_figures/baseline_bar_path_ratio.png | Bin 0 -> 35332 bytes
 .../S1d5_final_figures/scale_repair_summary.png    | Bin 0 -> 43401 bytes
 .../S1d5_final_figures/tmag_distribution_s1d5.png  | Bin 0 -> 41993 bytes
 .../trajectory_comparison_s1d5.png                 | Bin 0 -> 35630 bytes
 checkpoints/S1d5_final_mainline_summary.md         |  61 ++++
 .../S1d5_final_repro/s1d5_policy_eval_summary.json |  34 +++
 ..._freeze_clean_dt_anchor_policy_export_report.md | 111 +++++++
 ...d6_minimal_trainable_dt_anchor_head_proposal.md |  42 +++
 config.py                                          |  30 ++
 interaction.py                                     | 312 ++++++++++++++++++--
 losses.py                                          |  28 +-
 model.py                                           |  41 ++-
 reports/current_valid_baselines.md                 |  15 +
 reports/final_s1d5_english_abstract.md             |   9 +
 reports/final_s1d5_mainline_report.md              | 110 +++++++
 reports/final_s1d5_results_table.csv               |  10 +
 reports/final_s1d5_thesis_summary.md               |   9 +
 reports/git_track_s1d5_final_files.md              |  79 +++++
 reports/gitignore_cleanup_plan.md                  | 175 ++++++++++++
 scripts/eval_checkpoint_trajectory_repro.py        |  87 ++++++
 scripts/eval_s1d5_clean_policy.sh                  |  27 ++
 tools/eval_clean_policy.py                         | 318 +++++++++++++++++++++
 train_mvp.py                                       | 106 ++++++-
 27 files changed, 1664 insertions(+), 43 deletions(-)
```

## Forbidden File Check
- Forbidden staged file count: `0`
- Result: no staged `.pt/.pth/.ckpt/.npz`, no `archive/*`, no forbidden odom/eval diagnostic payloads.

## Forbidden Reset Actions Performed
- Yes
- Unstaged archive paths that had leaked into the index from prior cleanup renames.
- Also unstaged extra historical deletions that were not part of the requested whitelist staging.

## Post-staging `git status --short`
```text
M  .gitignore
 D checkpoints/O45_six_hour_cycle_results.md
 D checkpoints/O45_six_hour_cycle_state.json
A  checkpoints/S1d5_clean_dt_anchor_policy.json
A  checkpoints/S1d5_final_baseline_comparison.md
A  checkpoints/S1d5_final_figures/baseline_bar_ATE_drift.png
A  checkpoints/S1d5_final_figures/baseline_bar_path_ratio.png
A  checkpoints/S1d5_final_figures/scale_repair_summary.png
A  checkpoints/S1d5_final_figures/tmag_distribution_s1d5.png
A  checkpoints/S1d5_final_figures/trajectory_comparison_s1d5.png
A  checkpoints/S1d5_final_mainline_summary.md
A  checkpoints/S1d5_final_repro/s1d5_policy_eval_summary.json
A  checkpoints/S1d5_freeze_clean_dt_anchor_policy_export_report.md
A  checkpoints/S1d6_minimal_trainable_dt_anchor_head_proposal.md
M  config.py
M  interaction.py
M  losses.py
M  model.py
 D reports/T51_final_selection_summary.md
 D reports/T53_final_selection_summary.md
 D reports/T54_final_candidate_review.md
 D reports/T55a_selection_summary.md
A  reports/current_valid_baselines.md
 D reports/f0_fine_matching_ablation.md
 D reports/fig_repro_shape_compare_clean.png
 D reports/fig_t53_shape_compare.png
 D reports/fig_t55a_shape_compare.png
A  reports/final_s1d5_english_abstract.md
A  reports/final_s1d5_mainline_report.md
A  reports/final_s1d5_results_table.csv
A  reports/final_s1d5_thesis_summary.md
A  reports/git_track_s1d5_final_files.md
A  reports/gitignore_cleanup_plan.md
 D reports/o49_tmag_shape_diagnosis.md
 D reports/t51_dt_conditioned_tmag_head_plan.md
M  scripts/eval_checkpoint_trajectory_repro.py
A  scripts/eval_s1d5_clean_policy.sh
 M scripts/run_c31_anchor_smallk.sh
A  tools/eval_clean_policy.py
M  train_mvp.py
?? reports/checkpoints_live_cleanup_inventory.md
?? reports/checkpoints_live_cleanup_state.json
?? reports/checkpoints_live_cleanup_summary.md
?? reports/cleanup_actions_s1d5.md
?? reports/cleanup_delete_candidates_s1d5.md
?? reports/cleanup_inventory_s1d5.md
?? reports/cleanup_summary_s1d5.md
?? reports/git_staging_s1d5_precheck.md
```

## Sanity Check
### `python tools/eval_clean_policy.py --help`
Used interpreter:
- `/home/dovetao/miniconda3/envs/pytorch/bin/python`

Output:
```text
usage: eval_clean_policy.py [-h] --policy POLICY --output-dir OUTPUT_DIR
                            [--eval-variant {default,max_eval_batches_off,explicit_selected_k}]

Evaluate a frozen clean dt-anchor policy.

options:
  -h, --help            show this help message and exit
  --policy POLICY       Path to policy json
  --output-dir OUTPUT_DIR
                        Output directory
  --eval-variant {default,max_eval_batches_off,explicit_selected_k}
```

### `bash -n scripts/eval_s1d5_clean_policy.sh`
- Passed with no syntax errors.

### `bash scripts/eval_s1d5_clean_policy.sh dryrun`
```text
Policy: checkpoints/S1d5_clean_dt_anchor_policy.json
Output: checkpoints/S1d5_clean_policy_eval
Command: /home/dovetao/miniconda3/envs/pytorch/bin/python tools/eval_clean_policy.py --policy checkpoints/S1d5_clean_dt_anchor_policy.json --output-dir checkpoints/S1d5_clean_policy_eval --eval-variant default
[DRYRUN] not executing
```

## Commit Recommendation
No commit was executed.

Suggested command:
```bash
git commit -m "Finalize S1d5 clean mainline artifacts and gitignore"
```
