# JRT1b Train-CV Joint R/tdir Refiner

## Scope

JRT1b is train-CV only. It does not run final test, and S5 remains the final candidate.

## JRT1a Recap

JRT1a showed that rotation residual learning works in smoke, while translation direction did not improve. JRT1b therefore adds translation-direction weighted and train-only hardcase-weighted variants.

## Dataset and Folds

- dataset source: `RflyPanoPanoramaPairsEvalFixedKList train split with S5 policy predictions`
- cached train-split pairs: `2810`
- cached test pairs: `0`
- no test GT cached/used: `True`

| fold | train_pairs | val_pairs |
| --- | ---: | ---: |
| 0 | 1024 | 512 |
| 1 | 1024 | 512 |
| 2 | 1024 | 512 |

## Variants

| variant | description | weights | hardcase_alpha |
| --- | --- | --- | ---: |
| A_s5_no_refiner_reference | S5 reference, no refiner training | `{'w_rot': 1.0, 'w_tdir': 1.0, 'w_couple': 0.2, 'w_reg': 0.01}` | 0.0 |
| B_rot_only_residual | rotation residual only | `{'w_rot': 1.0, 'w_tdir': 1.0, 'w_couple': 0.2, 'w_reg': 0.01}` | 0.0 |
| C_tdir_residual_cosine | translation direction residual with cosine loss | `{'w_rot': 0.0, 'w_tdir': 1.0, 'w_couple': 0.0, 'w_reg': 0.01}` | 0.0 |
| D_joint_rtdir_residual | joint rotation and translation direction residual | `{'w_rot': 1.0, 'w_tdir': 1.0, 'w_couple': 0.2, 'w_reg': 0.01}` | 0.0 |
| E_joint_rtdir_tdir_weighted | joint residual with stronger translation direction weight | `{'w_rot': 0.5, 'w_tdir': 2.0, 'w_couple': 0.2, 'w_reg': 0.01}` | 0.0 |
| F_joint_rtdir_tdir_hardcase_weighted | joint residual with train-only hardcase translation direction weighting | `{'w_rot': 0.5, 'w_tdir': 2.0, 'w_couple': 0.2, 'w_reg': 0.01}` | 2.0 |

## Train-CV Component Metrics

| variant | rot_mean mean/std | tdir_mean mean/std | tdir_cos mean/std | tmag_log_err mean/std |
| --- | ---: | ---: | ---: | ---: |
| A_s5_no_refiner_reference | 21.093326 / 0.030379 | 39.315952 / 0.629807 | 0.686719 / 0.005714 | 0.953125 / 0.003950 |
| B_rot_only_residual | 1.217314 / 0.114776 | 39.315952 / 0.629807 | 0.686719 / 0.005714 | 0.953125 / 0.003950 |
| C_tdir_residual_cosine | 21.093326 / 0.030379 | 16.559566 / 1.424819 | 0.904896 / 0.016564 | 0.953125 / 0.003950 |
| D_joint_rtdir_residual | 1.413717 / 0.057736 | 24.310256 / 2.676283 | 0.815147 / 0.038702 | 0.953125 / 0.003950 |
| E_joint_rtdir_tdir_weighted | 1.783210 / 0.089999 | 17.323818 / 0.775487 | 0.896417 / 0.010178 | 0.953125 / 0.003950 |
| F_joint_rtdir_tdir_hardcase_weighted | 2.060442 / 0.016165 | 17.957445 / 0.706060 | 0.899039 / 0.011353 | 0.953125 / 0.003950 |

## Gate Decision

| variant | tdir_improvement | rot_worsening | tdir_cos_improved | trajectory_gate_available | gate_status |
| --- | ---: | ---: | --- | --- | --- |
| A_s5_no_refiner_reference | 0.000000 | 0.000000 | False | False | REFERENCE |
| B_rot_only_residual | 0.000000 | -19.876012 | False | False | FAIL |
| C_tdir_residual_cosine | 22.756386 | 0.000000 | True | False | COMPONENT-PASS-TRAJECTORY-UNAVAILABLE |
| D_joint_rtdir_residual | 15.005697 | -19.679608 | True | False | COMPONENT-PASS-TRAJECTORY-UNAVAILABLE |
| E_joint_rtdir_tdir_weighted | 21.992134 | -19.310116 | True | False | COMPONENT-PASS-TRAJECTORY-UNAVAILABLE |
| F_joint_rtdir_tdir_hardcase_weighted | 21.358507 | -19.032884 | True | False | COMPONENT-PASS-TRAJECTORY-UNAVAILABLE |

## Interpretation

- translation-direction signal present: `True`
- best variant by mean CV tdir error: `C_tdir_residual_cosine`
- joint variants evaluated: `['D_joint_rtdir_residual', 'E_joint_rtdir_tdir_weighted', 'F_joint_rtdir_tdir_hardcase_weighted']`
- trajectory gate available: `False`
Component-only train-CV diagnostics are insufficient for a final clean claim without trajectory gate support.

## Final Classification

`TRAJECTORY-GATE-UNAVAILABLE`

## Next Step

Do not run final test unless a later run has both a component gate pass and an available trajectory gate.

## Caveats

- no final candidate selected
- no practical-ready claim
- S5 unchanged
- small sequence protocol
- component-only signal is insufficient for final clean claim
