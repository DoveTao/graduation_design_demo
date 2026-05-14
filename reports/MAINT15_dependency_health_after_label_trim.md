# MAINT15 dependency health after label trim

## Summary
- scanned files: `27`
- total references: `107`
- current_required missing: `0`
- current_optional missing: `12`
- intentional omissions: `12`
- deleted references: `0`
- health status: `pass_with_intentional_omissions`

## Reference table
| file | line | reference | exists_now | category | recommended_action |
| --- | ---: | --- | --- | --- | --- |
| CURRENT_MAINLINE.md | 7 | `tools/current/show_mainline.py` | `true` | `current_required` | `keep` |
| CURRENT_MAINLINE.md | 8 | `tools/current/check_current_artifacts.py` | `true` | `current_required` | `keep` |
| CURRENT_MAINLINE.md | 14 | `configs/final360i_struct360b_final.yaml` | `true` | `current_required` | `keep` |
| CURRENT_MAINLINE.md | 15 | `tools/final360i_retrain_and_select.py` | `true` | `current_required` | `keep` |
| CURRENT_MAINLINE.md | 18 | `tools/train360e_sequence_trajectory_export_and_ate_eval.py` | `true` | `current_required` | `keep` |
| CURRENT_MAINLINE.md | 25 | `configs/seq360b_lightweight_scale_smoothing.yaml` | `true` | `current_required` | `keep` |
| CURRENT_MAINLINE.md | 26 | `tools/train_seq360b_lightweight_scale_smoothing.py` | `true` | `current_required` | `keep` |
| CURRENT_MAINLINE.md | 49 | `SEQ360A` | `false` | `legacy_reference` | `keep` |
| CURRENT_MAINLINE.md | 49 | `reports/SEQ360A_status_summary.md` | `true` | `current_optional` | `keep` |
| CURRENT_MAINLINE.md | 50 | `STRUCT360C` | `false` | `legacy_reference` | `keep` |
| CURRENT_MAINLINE.md | 50 | `reports/STRUCT360C_status_summary.md` | `true` | `current_optional` | `keep` |
| PROJECT_STRUCTURE.md | 7 | `tools/current/README.md` | `true` | `current_required` | `keep` |
| PROJECT_STRUCTURE.md | 27 | `tools/final360i_retrain_and_select.py` | `true` | `current_required` | `keep` |
| PROJECT_STRUCTURE.md | 28 | `tools/train360e_sequence_trajectory_export_and_ate_eval.py` | `true` | `current_required` | `keep` |
| PROJECT_STRUCTURE.md | 29 | `tools/train_struct360b_match_free_coarse_to_fine.py` | `true` | `current_optional` | `keep` |
| PROJECT_STRUCTURE.md | 31 | `configs/final360i_struct360b_final.yaml` | `true` | `current_required` | `keep` |
| PROJECT_STRUCTURE.md | 32 | `configs/struct360b_match_free_coarse_to_fine.yaml` | `true` | `current_required` | `keep` |
| PROJECT_STRUCTURE.md | 36 | `tools/train_seq360b_lightweight_scale_smoothing.py` | `true` | `current_required` | `keep` |
| PROJECT_STRUCTURE.md | 38 | `configs/seq360b_lightweight_scale_smoothing.yaml` | `true` | `current_required` | `keep` |
| PROJECT_STRUCTURE.md | 50 | `SEQ360A` | `false` | `legacy_reference` | `keep` |
| PROJECT_STRUCTURE.md | 51 | `STRUCT360C` | `false` | `legacy_reference` | `keep` |
| PROJECT_STRUCTURE.md | 59 | `external_baselines/results/dset2c_360dvo_canonical/` | `true` | `current_required` | `keep` |
| PROJECT_STRUCTURE.md | 61 | `external_baselines/results/base360_hkust_360dvo_official/` | `true` | `current_optional` | `keep` |
| README.md | 6 | `tools/current/show_mainline.py` | `true` | `current_required` | `keep` |
| README.md | 7 | `tools/current/check_current_artifacts.py` | `true` | `current_required` | `keep` |
| README.md | 32 | `SEQ360A` | `false` | `legacy_reference` | `keep` |
| README.md | 32 | `reports/SEQ360A_status_summary.md` | `true` | `current_optional` | `keep` |
| README.md | 33 | `STRUCT360C` | `false` | `legacy_reference` | `keep` |
| README.md | 33 | `reports/STRUCT360C_status_summary.md` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 13 | `configs/struct360b_match_free_coarse_to_fine.yaml` | `true` | `current_required` | `keep` |
| configs/final360i_struct360b_final.yaml | 14 | `checkpoints/DSET2C_360DVO_dataset_hygiene.json` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 15 | `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl` | `true` | `current_required` | `keep` |
| configs/final360i_struct360b_final.yaml | 16 | `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl` | `true` | `current_required` | `keep` |
| configs/final360i_struct360b_final.yaml | 17 | `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl` | `true` | `current_required` | `keep` |
| configs/final360i_struct360b_final.yaml | 19 | `TRAIN360D` | `false` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 19 | `checkpoints/TRAIN360D_observability_kstep_scale/best_val.pt` | `true` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 20 | `checkpoints/STRUCT360B_match_free_coarse_to_fine/best_val.pt` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 21 | `reports/STRUCT360B_metrics_val.json` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 22 | `reports/STRUCT360B_metrics_test.json` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 23 | `reports/BASE360D_metrics_test.json` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 26 | `checkpoints/FINAL360I_struct360b_final` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 27 | `reports/FINAL360I_final_retrain_and_model_selection.md` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 28 | `reports/FINAL360I_metrics_val.json` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 29 | `reports/FINAL360I_metrics_test.json` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 30 | `reports/FINAL360I_model_selection_table.json` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 31 | `reports/FINAL360I_mainline_comparison_summary.md` | `false` | `current_optional` | `mark_intentionally_omitted` |
| configs/final360i_struct360b_final.yaml | 32 | `reports/FINAL360I_thesis_ready_result_paragraph.md` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 5 | `checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 6 | `checkpoints/DSET2C_360DVO_dataset_hygiene.json` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 7 | `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl` | `true` | `current_required` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 8 | `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl` | `true` | `current_required` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 9 | `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl` | `true` | `current_required` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 10 | `reports/FINAL360I_metrics_val.json` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 11 | `reports/FINAL360I_metrics_test.json` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 12 | `reports/TRAIN360E_metrics_val.json` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 13 | `reports/TRAIN360E_metrics_test.json` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 14 | `reports/BASE360D_metrics_val.json` | `false` | `current_optional` | `mark_intentionally_omitted` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 15 | `reports/BASE360D_metrics_test.json` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 64 | `checkpoints/SEQ360B_lightweight_scale_smoothing` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 65 | `external_baselines/results/seq360b_scale_smoothing_trajectory` | `false` | `current_optional` | `mark_intentionally_omitted` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 66 | `reports/SEQ360B_train_lightweight_scale_smoothing_head.md` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 67 | `reports/SEQ360B_current_variant_metrics_val.json` | `false` | `current_optional` | `mark_intentionally_omitted` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 68 | `reports/SEQ360B_metrics_test.json` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 69 | `reports/SEQ360B_current_variant_trajectory_metrics_val.json` | `false` | `current_optional` | `mark_intentionally_omitted` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 70 | `reports/SEQ360B_trajectory_metrics_test.json` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 71 | `reports/SEQ360B_current_variant_summary.md` | `false` | `current_optional` | `mark_intentionally_omitted` |
| tools/current/README.md | 9 | `tools/final360i_retrain_and_select.py` | `true` | `current_required` | `keep` |
| tools/current/README.md | 10 | `tools/train360e_sequence_trajectory_export_and_ate_eval.py` | `true` | `current_required` | `keep` |
| tools/current/README.md | 11 | `tools/train_struct360b_match_free_coarse_to_fine.py` | `true` | `current_optional` | `keep` |
| tools/current/README.md | 12 | `tools/train_seq360b_lightweight_scale_smoothing.py` | `true` | `current_required` | `keep` |
| tools/current/check_current_artifacts.py | 13 | `configs/final360i_struct360b_final.yaml` | `true` | `current_required` | `keep` |
| tools/current/check_current_artifacts.py | 14 | `configs/struct360b_match_free_coarse_to_fine.yaml` | `true` | `current_required` | `keep` |
| tools/current/check_current_artifacts.py | 15 | `configs/seq360b_lightweight_scale_smoothing.yaml` | `true` | `current_required` | `keep` |
| tools/current/check_current_artifacts.py | 16 | `tools/final360i_retrain_and_select.py` | `true` | `current_required` | `keep` |
| tools/current/check_current_artifacts.py | 17 | `tools/train360e_sequence_trajectory_export_and_ate_eval.py` | `true` | `current_required` | `keep` |
| tools/current/check_current_artifacts.py | 18 | `tools/train_struct360b_match_free_coarse_to_fine.py` | `true` | `current_optional` | `keep` |
| tools/current/check_current_artifacts.py | 19 | `tools/train_seq360b_lightweight_scale_smoothing.py` | `true` | `current_required` | `keep` |
| tools/current/check_current_artifacts.py | 25 | `reports/FINAL360I_metrics_test.json` | `true` | `current_optional` | `keep` |
| tools/current/check_current_artifacts.py | 26 | `reports/TRAIN360E_metrics_test.json` | `true` | `current_optional` | `keep` |
| tools/current/check_current_artifacts.py | 27 | `reports/SEQ360B_metrics_test.json` | `true` | `current_optional` | `keep` |
| tools/current/check_current_artifacts.py | 28 | `reports/BASE360D_component_metric_alignment.md` | `true` | `current_optional` | `keep` |
| tools/current/check_current_artifacts.py | 29 | `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl` | `true` | `current_required` | `keep` |
| tools/current/check_current_artifacts.py | 33 | `SEQ360A` | `false` | `legacy_reference` | `keep` |
| tools/current/check_current_artifacts.py | 33 | `reports/SEQ360A_status_summary.md` | `true` | `current_optional` | `keep` |
| tools/current/check_current_artifacts.py | 34 | `STRUCT360C` | `false` | `legacy_reference` | `keep` |
| tools/current/check_current_artifacts.py | 34 | `reports/STRUCT360C_status_summary.md` | `true` | `current_optional` | `keep` |
| tools/current/check_current_artifacts.py | 38 | `reports/BASE360D_metrics_val.json` | `false` | `current_optional` | `mark_intentionally_omitted` |
| tools/current/check_current_artifacts.py | 39 | `reports/FINAL360I_vs_all_baselines_summary.md` | `false` | `current_optional` | `mark_intentionally_omitted` |
| tools/current/check_current_artifacts.py | 40 | `external_baselines/results/seq360b_scale_smoothing_trajectory` | `false` | `current_optional` | `mark_intentionally_omitted` |
| tools/current/check_current_artifacts.py | 41 | `reports/SEQ360B_metrics_val.json` | `false` | `current_optional` | `mark_intentionally_omitted` |
| tools/current/check_current_artifacts.py | 42 | `reports/SEQ360B_trajectory_metrics_val.json` | `false` | `current_optional` | `mark_intentionally_omitted` |
| tools/current/check_current_artifacts.py | 43 | `reports/SEQ360B_vs_FINAL360I_TRAIN360E_BASE360D_summary.md` | `false` | `current_optional` | `mark_intentionally_omitted` |
| tools/current/show_mainline.py | 10 | `configs/final360i_struct360b_final.yaml` | `true` | `current_required` | `keep` |
| tools/current/show_mainline.py | 11 | `tools/final360i_retrain_and_select.py` | `true` | `current_required` | `keep` |
| tools/current/show_mainline.py | 15 | `tools/train360e_sequence_trajectory_export_and_ate_eval.py` | `true` | `current_required` | `keep` |
| tools/current/show_mainline.py | 17 | `tools/train_seq360b_lightweight_scale_smoothing.py` | `true` | `current_required` | `keep` |
| tools/current/show_mainline.py | 23 | `SEQ360A` | `false` | `legacy_reference` | `keep` |
| tools/current/show_mainline.py | 23 | `reports/SEQ360A_status_summary.md` | `true` | `current_optional` | `keep` |
| tools/current/show_mainline.py | 24 | `STRUCT360C` | `false` | `legacy_reference` | `keep` |
| tools/current/show_mainline.py | 24 | `reports/STRUCT360C_status_summary.md` | `true` | `current_optional` | `keep` |
| tools/final360i_retrain_and_select.py | 189 | `data/360DVO/Sequences/` | `true` | `current_optional` | `keep` |
| tools/final360i_retrain_and_select.py | 616 | `reports/FINAL360I_thesis_ready_result_paragraph.md` | `true` | `current_optional` | `keep` |
| tools/final360i_retrain_and_select.py | 617 | `reports/FINAL360I_thesis_ready_result_paragraph.md` | `true` | `current_optional` | `keep` |
| tools/train360e_sequence_trajectory_export_and_ate_eval.py | 374 | `data/360DVO/Sequences/` | `true` | `current_optional` | `keep` |
| tools/train360e_sequence_trajectory_export_and_ate_eval.py | 1031 | `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_` | `false` | `unknown` | `replace_path` |
| train360/core/model.py | 458 | `S5E15` | `false` | `legacy_reference` | `keep` |
| train360/core/model.py | 683 | `S5E15` | `false` | `legacy_reference` | `keep` |

## Optional report snapshot
- `reports/FINAL360I_final_retrain_and_model_selection.md`: `present`
- `reports/FINAL360I_metrics_val.json`: `present`
- `reports/FINAL360I_metrics_test.json`: `present`
- `reports/SEQ360B_train_lightweight_scale_smoothing_head.md`: `present`
- `reports/SEQ360B_metrics_test.json`: `present`
- `reports/TRAIN360E_sequence_trajectory_export_and_ATE_eval.md`: `present`
- `reports/TRAIN360E_metrics_test.json`: `present`
- `reports/BASE360D_component_metric_alignment.md`: `present`
- `reports/BASE360D_metrics_test.json`: `present`
- `reports/RESULTS360_main_results_table.md`: `present`

## Intentional omissions
- `external_baselines/results/seq360b_scale_smoothing_trajectory`: `intentionally_omitted_after_cleanup`
- `reports/BASE360D_metrics_val.json`: `intentionally_omitted_after_cleanup`
- `reports/FINAL360I_vs_all_baselines_summary.md`: `intentionally_omitted_after_cleanup`
- `reports/SEQ360B_metrics_val.json`: `intentionally_omitted_after_cleanup`
- `reports/SEQ360B_trajectory_metrics_val.json`: `intentionally_omitted_after_cleanup`
- `reports/SEQ360B_vs_FINAL360I_TRAIN360E_BASE360D_summary.md`: `intentionally_omitted_after_cleanup`
