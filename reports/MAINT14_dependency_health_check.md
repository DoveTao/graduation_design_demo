# MAINT14 dependency health check

## Summary
- scanned files: `27`
- total references: `126`
- current_required missing: `0`
- current_optional missing: `7`
- deleted references: `0`

## Reference table
| file | line | reference | exists_now | category | recommended_action |
| --- | ---: | --- | --- | --- | --- |
| CURRENT_MAINLINE.md | 7 | `tools/current/show_mainline.py` | `true` | `current_required` | `keep` |
| CURRENT_MAINLINE.md | 8 | `tools/current/check_current_artifacts.py` | `true` | `current_required` | `keep` |
| CURRENT_MAINLINE.md | 13 | `configs/final360i_struct360b_final.yaml` | `true` | `current_required` | `keep` |
| CURRENT_MAINLINE.md | 14 | `tools/final360i_retrain_and_select.py` | `true` | `current_required` | `keep` |
| CURRENT_MAINLINE.md | 17 | `tools/train360e_sequence_trajectory_export_and_ate_eval.py` | `true` | `current_required` | `keep` |
| CURRENT_MAINLINE.md | 23 | `configs/seq360b_lightweight_scale_smoothing.yaml` | `true` | `current_required` | `keep` |
| CURRENT_MAINLINE.md | 24 | `tools/train_seq360b_lightweight_scale_smoothing.py` | `true` | `current_required` | `keep` |
| CURRENT_MAINLINE.md | 39 | `SEQ360A` | `false` | `current_optional` | `keep` |
| CURRENT_MAINLINE.md | 39 | `STRUCT360C` | `false` | `current_optional` | `keep` |
| PROJECT_STRUCTURE.md | 7 | `tools/current/README.md` | `true` | `current_required` | `keep` |
| PROJECT_STRUCTURE.md | 27 | `tools/final360i_retrain_and_select.py` | `true` | `current_required` | `keep` |
| PROJECT_STRUCTURE.md | 28 | `tools/train360e_sequence_trajectory_export_and_ate_eval.py` | `true` | `current_required` | `keep` |
| PROJECT_STRUCTURE.md | 29 | `tools/train_struct360b_match_free_coarse_to_fine.py` | `true` | `current_optional` | `keep` |
| PROJECT_STRUCTURE.md | 31 | `configs/final360i_struct360b_final.yaml` | `true` | `current_required` | `keep` |
| PROJECT_STRUCTURE.md | 32 | `configs/struct360b_match_free_coarse_to_fine.yaml` | `true` | `current_required` | `keep` |
| PROJECT_STRUCTURE.md | 36 | `tools/train_seq360b_lightweight_scale_smoothing.py` | `true` | `current_required` | `keep` |
| PROJECT_STRUCTURE.md | 38 | `configs/seq360b_lightweight_scale_smoothing.yaml` | `true` | `current_required` | `keep` |
| PROJECT_STRUCTURE.md | 46 | `external_baselines/results/dset2c_360dvo_canonical/` | `true` | `current_required` | `keep` |
| PROJECT_STRUCTURE.md | 48 | `external_baselines/results/base360_hkust_360dvo_official/` | `true` | `current_optional` | `keep` |
| PROJECT_STRUCTURE.md | 51 | `MAINT` | `false` | `legacy_reference` | `keep` |
| PROJECT_STRUCTURE.md | 57 | `scene01` | `false` | `legacy_reference` | `keep` |
| README.md | 6 | `tools/current/show_mainline.py` | `true` | `current_required` | `keep` |
| README.md | 7 | `tools/current/check_current_artifacts.py` | `true` | `current_required` | `keep` |
| README.md | 24 | `SEQ360A` | `false` | `current_optional` | `keep` |
| README.md | 24 | `STRUCT360C` | `false` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 13 | `configs/struct360b_match_free_coarse_to_fine.yaml` | `true` | `current_required` | `keep` |
| configs/final360i_struct360b_final.yaml | 14 | `checkpoints/DSET2C_360DVO_dataset_hygiene.json` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 15 | `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl` | `true` | `current_required` | `keep` |
| configs/final360i_struct360b_final.yaml | 16 | `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl` | `true` | `current_required` | `keep` |
| configs/final360i_struct360b_final.yaml | 17 | `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl` | `true` | `current_required` | `keep` |
| configs/final360i_struct360b_final.yaml | 19 | `TRAIN360D` | `false` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 19 | `checkpoints/TRAIN360D_observability_kstep_scale/best_val.pt` | `true` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 20 | `checkpoints/STRUCT360B_match_free_coarse_to_fine/best_val.pt` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 21 | `TRAIN360C` | `false` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 21 | `reports/TRAIN360C_metrics_val.json` | `true` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 22 | `TRAIN360C` | `false` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 22 | `reports/TRAIN360C_metrics_test.json` | `true` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 23 | `TRAIN360D` | `false` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 23 | `reports/TRAIN360D_metrics_val.json` | `true` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 24 | `TRAIN360D` | `false` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 24 | `reports/TRAIN360D_metrics_test.json` | `true` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 25 | `TRAIN360H` | `false` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 25 | `reports/TRAIN360H_best_config_val.json` | `true` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 26 | `TRAIN360H` | `false` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 26 | `reports/TRAIN360H_best_config_test.json` | `true` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 27 | `STRUCT360A` | `false` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 27 | `reports/STRUCT360A_metrics_val.json` | `true` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 28 | `STRUCT360A` | `false` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 28 | `reports/STRUCT360A_metrics_test.json` | `true` | `legacy_reference` | `keep` |
| configs/final360i_struct360b_final.yaml | 29 | `reports/STRUCT360B_metrics_val.json` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 30 | `reports/STRUCT360B_metrics_test.json` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 31 | `reports/BASE360D_metrics_val.json` | `false` | `current_optional` | `make_optional` |
| configs/final360i_struct360b_final.yaml | 32 | `reports/BASE360D_metrics_test.json` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 35 | `checkpoints/FINAL360I_struct360b_final` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 36 | `reports/FINAL360I_final_retrain_and_model_selection.md` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 37 | `reports/FINAL360I_metrics_val.json` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 38 | `reports/FINAL360I_metrics_test.json` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 39 | `reports/FINAL360I_model_selection_table.json` | `true` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 40 | `reports/FINAL360I_vs_all_baselines_summary.md` | `false` | `current_optional` | `keep` |
| configs/final360i_struct360b_final.yaml | 41 | `reports/FINAL360I_thesis_ready_result_paragraph.md` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 5 | `checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 6 | `checkpoints/DSET2C_360DVO_dataset_hygiene.json` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 7 | `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_train.jsonl` | `true` | `current_required` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 8 | `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl` | `true` | `current_required` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 9 | `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl` | `true` | `current_required` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 10 | `reports/FINAL360I_metrics_val.json` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 11 | `reports/FINAL360I_metrics_test.json` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 12 | `reports/TRAIN360E_metrics_val.json` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 13 | `reports/TRAIN360E_metrics_test.json` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 14 | `reports/BASE360D_metrics_val.json` | `false` | `current_optional` | `make_optional` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 15 | `reports/BASE360D_metrics_test.json` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 64 | `checkpoints/SEQ360B_lightweight_scale_smoothing` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 65 | `external_baselines/results/seq360b_scale_smoothing_trajectory` | `false` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 66 | `reports/SEQ360B_train_lightweight_scale_smoothing_head.md` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 67 | `reports/SEQ360B_metrics_val.json` | `false` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 68 | `reports/SEQ360B_metrics_test.json` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 69 | `reports/SEQ360B_trajectory_metrics_val.json` | `false` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 70 | `reports/SEQ360B_trajectory_metrics_test.json` | `true` | `current_optional` | `keep` |
| configs/seq360b_lightweight_scale_smoothing.yaml | 71 | `reports/SEQ360B_vs_FINAL360I_TRAIN360E_BASE360D_summary.md` | `false` | `current_optional` | `keep` |
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
| tools/current/check_current_artifacts.py | 28 | `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl` | `true` | `current_required` | `keep` |
| tools/current/check_current_artifacts.py | 32 | `SEQ360A` | `false` | `legacy_reference` | `keep` |
| tools/current/check_current_artifacts.py | 32 | `reports/SEQ360A_status_summary.md` | `true` | `current_optional` | `keep` |
| tools/current/check_current_artifacts.py | 33 | `STRUCT360C` | `false` | `legacy_reference` | `keep` |
| tools/current/check_current_artifacts.py | 33 | `reports/STRUCT360C_status_summary.md` | `true` | `current_optional` | `keep` |
| tools/current/check_current_artifacts.py | 34 | `reports/BASE360D_component_metric_alignment.md` | `true` | `current_optional` | `keep` |
| tools/current/show_mainline.py | 9 | `configs/final360i_struct360b_final.yaml` | `true` | `current_required` | `keep` |
| tools/current/show_mainline.py | 10 | `tools/final360i_retrain_and_select.py` | `true` | `current_required` | `keep` |
| tools/current/show_mainline.py | 14 | `tools/train360e_sequence_trajectory_export_and_ate_eval.py` | `true` | `current_required` | `keep` |
| tools/current/show_mainline.py | 16 | `tools/train_seq360b_lightweight_scale_smoothing.py` | `true` | `current_required` | `keep` |
| tools/final360i_retrain_and_select.py | 200 | `data/360DVO/Sequences/` | `true` | `current_optional` | `keep` |
| tools/final360i_retrain_and_select.py | 352 | `TRAIN360C` | `false` | `legacy_reference` | `keep` |
| tools/final360i_retrain_and_select.py | 353 | `TRAIN360D` | `false` | `legacy_reference` | `keep` |
| tools/final360i_retrain_and_select.py | 354 | `TRAIN360H` | `false` | `legacy_reference` | `keep` |
| tools/final360i_retrain_and_select.py | 355 | `STRUCT360A` | `false` | `legacy_reference` | `keep` |
| tools/final360i_retrain_and_select.py | 604 | `TRAIN360C` | `false` | `legacy_reference` | `keep` |
| tools/final360i_retrain_and_select.py | 605 | `TRAIN360D` | `false` | `legacy_reference` | `keep` |
| tools/final360i_retrain_and_select.py | 606 | `TRAIN360H` | `false` | `legacy_reference` | `keep` |
| tools/final360i_retrain_and_select.py | 607 | `STRUCT360A` | `false` | `legacy_reference` | `keep` |
| tools/final360i_retrain_and_select.py | 638 | `TRAIN360D` | `false` | `legacy_reference` | `keep` |
| tools/final360i_retrain_and_select.py | 639 | `TRAIN360H` | `false` | `legacy_reference` | `keep` |
| tools/final360i_retrain_and_select.py | 649 | `reports/FINAL360I_thesis_ready_result_paragraph.md` | `true` | `current_optional` | `keep` |
| tools/final360i_retrain_and_select.py | 650 | `reports/FINAL360I_thesis_ready_result_paragraph.md` | `true` | `current_optional` | `keep` |
| tools/final360i_retrain_and_select.py | 696 | `TRAIN360D` | `false` | `legacy_reference` | `keep` |
| tools/final360i_retrain_and_select.py | 697 | `TRAIN360H` | `false` | `legacy_reference` | `keep` |
| tools/final360i_retrain_and_select.py | 703 | `TRAIN360D` | `false` | `legacy_reference` | `keep` |
| tools/final360i_retrain_and_select.py | 704 | `TRAIN360H` | `false` | `legacy_reference` | `keep` |
| tools/final360i_retrain_and_select.py | 745 | `TRAIN360D` | `false` | `legacy_reference` | `keep` |
| tools/final360i_retrain_and_select.py | 746 | `TRAIN360H` | `false` | `legacy_reference` | `keep` |
| tools/train360e_sequence_trajectory_export_and_ate_eval.py | 375 | `data/360DVO/Sequences/` | `true` | `current_optional` | `keep` |
| tools/train360e_sequence_trajectory_export_and_ate_eval.py | 1032 | `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_` | `false` | `unknown` | `replace_path` |
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
