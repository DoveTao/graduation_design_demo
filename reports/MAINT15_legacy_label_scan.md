# MAINT15 legacy label scan

## Summary
- scan target: current docs, current tools, current configs, kept main reports, and status summaries
- visible comparison set after trim: `FINAL360I`, `TRAIN360E`, `SEQ360B`, `BASE360D`, `T57b`
- status-only items after trim: `SEQ360A`, `STRUCT360C`
- intentionally omitted optional artifacts remain documented, but no longer act like required current comparisons

## Records
| file | line | label/reference | category | action |
| --- | ---: | --- | --- | --- |
| `README.md` | 32 | `SEQ360A` | `status_only_allowed` | `keep` |
| `README.md` | 33 | `STRUCT360C` | `status_only_allowed` | `keep` |
| `CURRENT_MAINLINE.md` | 49 | `SEQ360A` | `status_only_allowed` | `keep` |
| `CURRENT_MAINLINE.md` | 50 | `STRUCT360C` | `status_only_allowed` | `keep` |
| `PROJECT_STRUCTURE.md` | 50 | `SEQ360A` | `status_only_allowed` | `keep` |
| `PROJECT_STRUCTURE.md` | 51 | `STRUCT360C` | `status_only_allowed` | `keep` |
| `tools/current/show_mainline.py` | 23 | `reports/SEQ360A_status_summary.md` | `status_only_allowed` | `keep` |
| `tools/current/show_mainline.py` | 24 | `reports/STRUCT360C_status_summary.md` | `status_only_allowed` | `keep` |
| `tools/current/check_current_artifacts.py` | 39 | `reports/FINAL360I_vs_all_baselines_summary.md` | `stale_comparison` | `mark_intentionally_omitted` |
| `tools/current/check_current_artifacts.py` | 41 | `reports/SEQ360B_metrics_val.json` | `stale_comparison` | `mark_intentionally_omitted` |
| `tools/current/check_current_artifacts.py` | 42 | `reports/SEQ360B_trajectory_metrics_val.json` | `stale_comparison` | `mark_intentionally_omitted` |
| `tools/current/check_current_artifacts.py` | 43 | `reports/SEQ360B_vs_FINAL360I_TRAIN360E_BASE360D_summary.md` | `stale_comparison` | `mark_intentionally_omitted` |
| `configs/final360i_struct360b_final.yaml` | 19 | `checkpoints/TRAIN360D_observability_kstep_scale/best_val.pt` | `historical_note` | `keep` |
| `reports/FINAL360I_final_retrain_and_model_selection.md` | 37 | `reports/SEQ360A_status_summary.md` | `status_only_allowed` | `keep` |
| `reports/FINAL360I_final_retrain_and_model_selection.md` | 38 | `reports/STRUCT360C_status_summary.md` | `status_only_allowed` | `keep` |
| `reports/RESULTS360_main_results_table.md` | 37 | `reports/SEQ360A_status_summary.md` | `status_only_allowed` | `keep` |
| `reports/RESULTS360_main_results_table.md` | 38 | `reports/STRUCT360C_status_summary.md` | `status_only_allowed` | `keep` |
| `reports/SEQ360A_status_summary.md` | 7 | `tools/train_seq360a_sequence_consistency_scale_drift.py` | `historical_note` | `keep` |
| `reports/STRUCT360C_status_summary.md` | 6 | `tools/train_struct360c_rotation_aware_fine_refinement.py` | `historical_note` | `keep` |
| `reports/BASE360D_component_metric_alignment.md` | 16 | `MAINT13 cleanup` | `historical_note` | `keep` |

## Interpretation
- no remaining `SEQ360A` or `STRUCT360C` references act as current runtime dependencies or main comparison candidates
- no deleted comparison artifact remains in the `current_required` set
- the remaining old labels are either:
  - status-summary links kept on purpose
  - historical checkpoint lineage
  - intentionally omitted legacy comparison outputs documented for cleanup clarity
