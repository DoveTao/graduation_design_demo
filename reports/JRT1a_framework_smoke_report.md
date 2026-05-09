# JRT1a Joint R/tdir Refiner Framework Smoke

## Scope

JRT1a is a framework and smoke-test harness only. It does not select a final candidate, does not run final test, and leaves S5 as the final clean candidate.

## Motivation

S13 oracle evidence isolates a coupled R/tdir bottleneck: oracle_R alone worsened ATE to 10.590301 and drift to 1.516962; oracle_tdir alone remained limited at ATE 7.658319 and drift 1.375470; oracle_R_tdir improved sharply to ATE 0.911024 and drift 0.304305.

## Dataset Builder

- dataset source: `RflyPanoPanoramaPairsEvalFixedKList train split with S5 policy predictions`
- num pairs: `384`
- train/val/test cached counts: `256` / `128` / `0`
- available test pairs counted but not cached for training: `1695`
- feature fields: `['R_pred_00', 'R_pred_01', 'R_pred_02', 'R_pred_10', 'R_pred_11', 'R_pred_12', 'R_pred_20', 'R_pred_21', 'R_pred_22', 'tdir_pred_x', 'tdir_pred_y', 'tdir_pred_z', 'log_tmag_pred', 'dt_world', 'k']`
- label fields: `['R_gt_3x3', 'tdir_gt_3', 'tmag_gt']`
- test GT excluded from training: `True`

## Model

Prediction-space 2-layer MLP residual refiner. Variants are A_s5_no_refiner_reference, B_rot_only_residual_smoke, C_tdir_only_residual_smoke, and D_joint_rtdir_residual_smoke. Loss terms are geodesic rotation loss, translation-direction cosine loss, a multiplicative coupling term, and residual regularization.

## Smoke Training

| variant | status | train_loss_initial | train_loss_final | num_updates | notes |
| --- | --- | --- | --- | --- | --- |
| A_s5_no_refiner_reference | ok |  |  | 0 | smoke harness row |
| B_rot_only_residual_smoke | ok | 0.965613 | 0.233547 | 20 | smoke harness row |
| C_tdir_only_residual_smoke | ok | 1.166280 | 0.464502 | 20 | smoke harness row |
| D_joint_rtdir_residual_smoke | ok | 1.023233 | 0.430529 | 20 | smoke harness row |

## Component Diagnostics

| variant | rot_mean_deg | rot_p90_deg | tdir_mean_deg | tdir_p90_deg | tdir_mean_cosine | tmag_mean_log_error |
| --- | --- | --- | --- | --- | --- | --- |
| A_s5_no_refiner_reference | 20.415323 | 21.848961 | 51.167244 | 163.550644 | 0.500287 | 1.049476 |
| B_rot_only_residual_smoke | 7.142703 | 13.836673 | 51.167244 | 163.550644 | 0.500287 | 1.049476 |
| C_tdir_only_residual_smoke | 20.415323 | 21.848961 | 52.333576 | 165.960968 | 0.495705 | 1.049476 |
| D_joint_rtdir_residual_smoke | 8.765825 | 12.603235 | 51.617874 | 163.679504 | 0.494787 | 1.049476 |

## Smoke Classification

`JRT1A-FRAMEWORK-SMOKE-PASS`

## Next Step

Proceed to JRT1b train-CV variants only if this smoke classification remains pass; otherwise fix the dataset or harness before algorithm claims.

## Caveats

- no final test
- no candidate replacement
- no practical-ready claim
- S5 locked metrics unchanged
- smoke results are not clean final results
