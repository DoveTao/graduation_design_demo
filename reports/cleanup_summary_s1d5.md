# Cleanup Summary S1d5

## 1. git status --short before

See: [cleanup_inventory_s1d5.md](/home/dovetao/graduation_design_demo/reports/cleanup_inventory_s1d5.md)

## 2. git status --short after

```text
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
 M config.py
 M interaction.py
 M losses.py
 M model.py
 M scripts/eval_checkpoint_trajectory_repro.py
 M scripts/run_c31_anchor_smallk.sh
 M train_mvp.py
?? archive/s1d5_cleanup_20260505/reports/T57c_final_selection_summary.md
?? archive/s1d5_cleanup_20260505/scripts/
?? archive/s1d5_cleanup_20260505/tools/
?? reports/
?? scripts/eval_s1d5_clean_policy.sh
?? tools/eval_clean_policy.py
```

## 3. Archive root

- path: archive/s1d5_cleanup_20260505
- top-level archive counts:
  - checkpoints: 165
  - reports: 11
  - scripts: 9
  - tools: 15
- archived file count: 1763
- archived total size: 1.4G

## 4. Deletions

- deleted file count: 0
- deleted total size: 0
- note: cache entries remain only as delete candidates; nothing was permanently deleted.

## 5. Kept S1d5 final artifacts

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

## 6. S1d5 eval wrapper sanity check

### python help

```text
usage: eval_clean_policy.py [-h] --policy POLICY --output-dir OUTPUT_DIR
                            [--eval-variant {default,max_eval_batches_off,explicit_selected_k}]

Evaluate a frozen clean dt-anchor policy.

options:
  -h, --help            show this help message and exit
  --policy POLICY       Path to policy json
```

### shell dryrun

```text
Policy: checkpoints/S1d5_clean_dt_anchor_policy.json
Output: checkpoints/S1d5_clean_policy_eval
Command: /home/dovetao/miniconda3/envs/pytorch/bin/python tools/eval_clean_policy.py --policy checkpoints/S1d5_clean_dt_anchor_policy.json --output-dir checkpoints/S1d5_clean_policy_eval --eval-variant default
[DRYRUN] not executing
```

- result: PASS

## 7. Missing artifacts

- none detected in the protected S1d5 final whitelist

## 8. Next-step recommendation

- project now enters final reporting state
- S1d5 remains the current clean exported mainline
- F1d remains paused
- S1d6 remains proposal only and should not be run unless internalization is later required
