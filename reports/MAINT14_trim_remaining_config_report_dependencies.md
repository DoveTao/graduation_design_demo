# MAINT14 trim remaining config/report dependencies

## 1. Executive summary
- dependency trim executed: `true`
- files restored count: `0`
- old reports restored count: `0`
- current mainline preserved: `true`
- py_compile result: `pass`
- import smoke result: `pass`
- current artifact check result: `pass`
- dependency health check result: `partial`

## 2. What was checked
- `tools/final360i_retrain_and_select.py`
- `tools/train360e_sequence_trajectory_export_and_ate_eval.py`
- `tools/train_seq360b_lightweight_scale_smoothing.py`
- `models/struct360b_match_free_coarse_to_fine.py`
- `models/seq360b_scale_smoothing_head.py`
- `configs/final360i_struct360b_final.yaml`
- `configs/seq360b_lightweight_scale_smoothing.yaml`
- `tools/current/check_current_artifacts.py`
- `README.md`, `CURRENT_MAINLINE.md`, `PROJECT_STRUCTURE.md`

## 3. Broken references found
| file | reference | category | action |
| --- | --- | --- | --- |
| `tools/train360e_sequence_trajectory_export_and_ate_eval.py` | `reports/BASE360D_metrics_val.json` | `current_optional missing` | made optional in precheck/baseline recap |
| `tools/train360e_sequence_trajectory_export_and_ate_eval.py` | `reports/FINAL360I_vs_all_baselines_summary.md` | `current_optional missing` | removed from required precheck path set |
| `configs/seq360b_lightweight_scale_smoothing.yaml` | `reports/BASE360_metrics_{val,test}.json` | `stale path` | replaced with `reports/BASE360D_metrics_{val,test}.json` |
| `tools/train_seq360b_lightweight_scale_smoothing.py` | hard-coded `proceed_to_SEQ360A...` next-step text | `legacy reference` | replaced with retained-mainline-safe recommendation |
| `tools/final360i_retrain_and_select.py` | old baseline metric inputs | `legacy comparison dependency` | downgraded to optional precheck status instead of required blocker |

## 4. Fixes applied
- Added `tools/mainline_dependency_utils.py` with small missing-safe helpers for optional JSON/text reads and artifact summaries.
- Relaxed `TRAIN360E` precheck so only truly required FINAL360I inputs block execution; BASE360D comparison reports are now optional.
- Added missing-safe `BASE360D` recap handling and fallback empty summary generation when optional artifacts are absent after cleanup.
- Relaxed `FINAL360I` precheck so legacy comparison metrics do not block mainline execution on the cleaned repo.
- Updated `SEQ360B` config to use `BASE360D` report naming instead of removed `BASE360` paths.
- Updated `SEQ360B` reporting text so the next-step recommendation no longer points at removed `SEQ360A` training codepaths.
- Expanded maintenance-branch allowlists for the kept mainline tools/config so the cleaned repo can be smoke-checked without pretending to be on old experiment branches.
- Updated `tools/current/check_current_artifacts.py` to keep status-summary artifacts optional instead of silently assuming old full reports must exist.

## 5. Restored files
- none
- backup source tag available if needed later: `backup/pre-maint13-destructive-cleanup-20260514-174819`

## 6. Remaining optional missing references
- `reports/BASE360D_metrics_val.json`
- `reports/FINAL360I_vs_all_baselines_summary.md`
- `external_baselines/results/seq360b_scale_smoothing_trajectory`
- `reports/SEQ360B_metrics_val.json`
- `reports/SEQ360B_trajectory_metrics_val.json`
- `reports/SEQ360B_vs_FINAL360I_TRAIN360E_BASE360D_summary.md`
- These are safe because they are comparison/output artifacts, not runtime-required inputs for the cleaned mainline smoke path.

## 7. Validation
- `py_compile`: `pass`
- import smoke: `pass`
- real dataset class name used in smoke: `Dset2CCanonicalPairDataset`
- `tools/current/show_mainline.py`: `pass`
- `tools/current/check_current_artifacts.py`: `pass`
- `tools/maint14_dependency_health_check.py`: `partial`
- health-check interpretation: `current_required = 0 missing`, remaining findings are optional/missing-after-cleanup artifacts and preserved legacy labels.

## 8. Compliance
- `training_executed = false`
- `fine_tune_executed = false`
- `checkpoints_modified = false`
- `raw_data_modified = false`
- `locked_metrics_modified = false`
- `s5_locked_metrics_modified = false`
- `old_experiment_bulk_restored = false`
- `large_artifacts_committed = false`
