# MAINT15 reduce legacy comparison labels

## 1. Executive summary
- label cleanup executed: `true`
- metrics modified: `false`
- old files restored: `false`
- current_required missing after cleanup: `0`
- optional omitted count: `12`

## 2. What changed
- `README.md` simplified to emphasize only the kept visible comparison set.
- `CURRENT_MAINLINE.md` simplified around `FINAL360I`, `TRAIN360E`, `SEQ360B`, `BASE360D`, and `T57b`.
- `PROJECT_STRUCTURE.md` simplified to separate visible comparison items from status-only items.
- `tools/current/show_mainline.py` now prints `FINAL360I`, `TRAIN360E`, `SEQ360B`, `BASE360D`, and `T57b`, with `SEQ360A` / `STRUCT360C` moved under archived status summaries.
- `tools/current/check_current_artifacts.py` now exposes intentionally omitted artifacts explicitly instead of making them look like accidental breakage.
- `configs/final360i_struct360b_final.yaml` and `tools/final360i_retrain_and_select.py` no longer treat older comparison reports as part of the current visible comparison set.
- `configs/seq360b_lightweight_scale_smoothing.yaml` and `tools/train_seq360b_lightweight_scale_smoothing.py` now use current-variant naming and no longer point to removed `SEQ360A` progression text.
- kept narrative reports were rewritten to foreground only the current retained mainline.

## 3. Current visible comparison set
- `FINAL360I`
- `TRAIN360E`
- `SEQ360B`
- `BASE360D`
- `T57b`

## 4. Status-only items
- `SEQ360A`
- `STRUCT360C`

## 5. Intentionally omitted artifacts
- `reports/BASE360D_metrics_val.json`
- `reports/FINAL360I_vs_all_baselines_summary.md`
- `external_baselines/results/seq360b_scale_smoothing_trajectory`
- `reports/SEQ360B_metrics_val.json`
- `reports/SEQ360B_trajectory_metrics_val.json`
- `reports/SEQ360B_vs_FINAL360I_TRAIN360E_BASE360D_summary.md`
- reason for each omission: removed by `MAINT13` cleanup, not required for the current kept mainline

## 6. Validation
- `py_compile`: `pass`
- import smoke: `pass`
- current artifact check: `pass`
- dependency health check: `pass_with_intentional_omissions`

## 7. Compliance
- `training_executed = false`
- `fine_tune_executed = false`
- `metrics_modified = false`
- `old_reports_restored = false`
- `checkpoints_modified = false`
- `raw_data_modified = false`
- `locked_metrics_modified = false`
- `s5_locked_metrics_modified = false`
