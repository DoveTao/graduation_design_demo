# FINAL360M NaN root-cause audit

## Summary

- `corrupt FINAL360M run stopped`: `true`
- `stop method`: `already_dead`
- `corrupt run archived`: `true`
- `original checkpoint dir preserved`: `true`
- `NaN root cause identified`: `true`
- `suspected cause`: `AMP mixed-precision path caused nonfinite gradients on the very first update; FP32 path remained finite`
- `dataset finite sanity passed`: `true`
- `bad samples found`: `false`
- `fp32 single batch finite`: `true`
- `amp single batch finite`: `false` for gradients
- `loss finite`: `true` in FP32 and AMP single-batch forward/loss
- `grad finite`: `true` in FP32, `false` in AMP
- `first NaN batch identified`: `true`
- `guarded smoke training executed`: `true`
- `guarded smoke updates`: `100`
- `guarded smoke finite`: `true`
- `guarded full-train restarted`: `true`
- `new PID`: `18450`
- `new log`: `logs/FINAL360M_fulltrain_guarded_20260518_031947.log`
- `new checkpoint dir`: `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded`
- `train subset disabled`: `true`
- `train sample count`: `12154`
- `val sample count`: `5342`
- `test sample count`: `5062`
- `disk free before restart`: `8.6G`
- `metrics modified`: `false`
- `checkpoints committed`: `false`
- `current artifact check`: `pass`
- `py_compile`: `pass`

## A. Corrupt run stop/archive

- original process state when checked: `15106 [python] <defunct>`
- `stopped_nan_run = true`
- `stop_method = already_dead`
- original corrupt checkpoint dir moved to:
  - `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_nan_corrupt_20260518_022354`
- original log preserved:
  - `logs/FINAL360M_fulltrain_20260518_022354.log`
- copied diagnostic log:
  - `logs/FINAL360M_fulltrain_20260518_022354_nan_corrupt.log`

## B. Config and training path audit

Checked:

- `configs/final360m_fulltrain_struct360b_thesis_main.yaml`
- `tools/train_final360m_fulltrain_thesis_main.py`
- `train360/core/train360d_pose_losses.py`
- `datasets/dset2c_manifest_dataset.py`

Findings:

- full train split settings were correct:
  - `train_subset_max: null`
  - `train_subset_count: null`
  - `use_train_subset: false`
  - precheck asserted `train/val/test = 12154/5342/5062`
- YAML parsing was not the problem:
  - `yaml.safe_load` resolved `null` and `false` correctly
- init checkpoint config injection was not the primary NaN source:
  - model load is `non-strict` because STRUCT360B fine/refiner parameters are newly initialized, which is expected
  - no shape mismatch was reported
- original risky numeric choices:
  - `amp: true`
  - `data.tmag_epsilon: 1e-6`
  - `observability.tmag_epsilon: 1e-6`
  - `fine_lr: 1e-4`
  - `coarse_lr: 1e-5`
- the strongest evidence points to AMP instability rather than dataset corruption or manifest misuse.

## C. Dataset sanity

Output file:

- `reports/FINAL360M_dataset_nan_sanity.json`

Results:

- `train_sample_count = 12154`
- `val_sample_count = 5342`
- `test_sample_count = 5062`
- `bad_samples_detected = false`
- `nan_inf_count = 0`
- `zero_tmag_count = 0`
- `negative_tmag_count = 0`
- `tmag_gt_min = 2.2487470105565646e-09`
- `tmag_gt_max = 6.05706417021006`
- `tmag_gt_median = 0.7540582156494824`

Interpretation:

- raw dataset values are finite.
- there are extremely small but still positive `tmag` values, which makes `1e-6` a weak safety margin for log-space losses.
- this supports raising `tmag_epsilon`, but does not by itself explain the immediate NaN; the decisive trigger still appears to be AMP.

## D. Single-batch forward/loss sanity

Original config:

- `configs/final360m_fulltrain_struct360b_thesis_main.yaml`

Single-batch results:

- FP32:
  - `R_pred finite = true`
  - `tdir_pred finite = true`
  - `tmag_pred finite = true`
  - `loss_rot finite = true`
  - `loss_tdir finite = true`
  - `loss_tmag finite = true`
  - `loss_total finite = true`
  - `grad finite = true`
- AMP:
  - `R_pred finite = true`
  - `tdir_pred finite = true`
  - `tmag_pred finite = true`
  - `loss_total finite = true`
  - `grad finite = false`
  - `grad_norm = NaN`

Interpretation:

- forward outputs are not inherently broken.
- loss computation itself is not inherently broken.
- AMP is the point where the path becomes unstable.

## E. First NaN batch

Using original config and actual training-style update path with AMP + GradScaler:

- `first_nan_batch_index = 0`
- `loss_component_that_first_became_nan = gradient_nonfinite`
- first bad sample meta:
  - `sequence = ['city_driving', 'city_driving']`
  - `pair_index = [12012, 11967]`
  - `k = [5, 3]`
  - `dt_world = [0.5, 0.30000000000000004]`
  - `tmag_gt = [3.876643180847168, 2.4688963890075684]`

Interpretation:

- the original run was already invalid on the first optimizer step.
- the later epoch-wide NaN logs are downstream symptoms, not a late-training collapse.

## F. Minimal guarded fixes applied

Added:

- `tools/final360m_nan_sanity_check.py`
- `configs/final360m_fulltrain_struct360b_thesis_main_guarded.yaml`

Changed in trainer:

- `tools/train_final360m_fulltrain_thesis_main.py`

Guarded changes:

- `amp: false`
- `coarse_lr: 5e-6`
- `fine_lr: 1e-5`
- `warmup_fine_lr: 1e-5`
- `data.tmag_epsilon: 1e-3`
- `loss.observability.tmag_epsilon: 1e-3`
- `abort_on_nonfinite_loss: true`
- `abort_on_nonfinite_grad: true`

## G. Guarded smoke training

Guarded config:

- `configs/final360m_fulltrain_struct360b_thesis_main_guarded.yaml`

Results:

- `guarded_smoke_training_executed = true`
- `guarded_smoke_updates = 100`
- `guarded_smoke_finite = true`
- every logged checkpoint at 10-update intervals remained finite
- no nonfinite loss, output, or gradient was detected in the smoke run

Interpretation:

- guarded settings are numerically stable enough to justify a real restart.

## H. Guarded restart

Restarted:

- `guarded full-train restarted = true`
- `new PID = 18450`
- `new log = logs/FINAL360M_fulltrain_guarded_20260518_031947.log`
- `new checkpoint dir = checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded`
- `new protocol = reports/FINAL360M_fulltrain_guarded_protocol.json`

Current protocol confirms:

- `train_subset_disabled = true`
- `train_sample_count = 12154`
- `val_sample_count = 5342`
- `test_sample_count = 5062`

## Final Judgment

- original `FINAL360M` run is invalid and must not be used for any thesis claim
- root cause is sufficiently identified as `AMP-induced nonfinite gradients on step 0`, with `tiny tmag epsilon` as a secondary numerical risk
- guarded restart is justified because:
  - dataset sanity passed
  - FP32 single-batch sanity passed
  - first NaN was reproduced in AMP path
  - guarded 100-update smoke passed

## Recommended next task

- monitor the guarded run through at least the first completed epoch and first mini-val/full-val checkpoint, then execute `FINAL360M_guarded_training_progress_audit`

