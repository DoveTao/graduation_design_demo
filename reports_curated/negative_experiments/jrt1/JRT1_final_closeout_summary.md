# JRT1 Final Closeout Summary

## Scope

JRT1 explored a joint R/tdir residual refiner motivated by S13 oracle diagnostics. It is a new algorithm experiment, not a final clean candidate selection path. It did not produce a final clean candidate, and S5 remains final clean candidate.

## Motivation

S13 oracle diagnostics showed that isolated component fixes were not enough:

| oracle | ATE | drift |
| --- | ---: | ---: |
| oracle_R | 10.590301 | 1.516962 |
| oracle_tdir | 7.658319 | 1.375470 |
| oracle_R_tdir | 0.911024 | 0.304305 |

The evidence suggested that R and tdir must be improved jointly. JRT1 tested whether a lightweight prediction-space residual refiner could capture that coupling.

## JRT1a Summary

Classification: `JRT1A-FRAMEWORK-SMOKE-PASS`

The JRT1a framework and harness worked. Rotation residual learning worked in smoke, but translation-direction correction did not yet improve.

| variant | rot_mean | tdir_mean | interpretation |
| --- | ---: | ---: | --- |
| A_s5_no_refiner_reference | 20.415323 | 51.167244 | S5 smoke reference |
| B_rot_only_residual_smoke | 7.142703 | 51.167244 | rotation improved, tdir unchanged |
| D_joint_rtdir_residual_smoke | 8.765825 | 51.617874 | rotation improved, tdir did not improve |

## JRT1b Summary

Classification: `TRAJECTORY-GATE-UNAVAILABLE`

JRT1b produced component-level positive signal under train-CV. There was no final test and no trajectory gate in this stage.

| variant | rot_mean | tdir_mean | tdir_cos | interpretation |
| --- | ---: | ---: | ---: | --- |
| A_s5_no_refiner_reference | 21.093326 | 39.315952 | 0.686719 | S5 train-CV component reference |
| B_rot_only_residual | 1.217314 | 39.315952 | 0.686719 | strong rotation improvement, no tdir improvement |
| C_tdir_residual_cosine | 21.093326 | 16.559566 | 0.904896 | strongest tdir-only component improvement |
| D_joint_rtdir_residual | 1.413717 | 24.310256 | 0.815147 | joint diagnostic signal |
| E_joint_rtdir_tdir_weighted | 1.783210 | 17.323818 | 0.896417 | balanced joint diagnostic signal |
| F_joint_rtdir_tdir_hardcase_weighted | 2.060442 | 17.957445 | 0.899039 | balanced joint diagnostic signal with hardcase weighting |

C tdir-only gave the strongest translation-direction improvement. E/F gave the most meaningful balanced joint diagnostic signal. But no trajectory proxy was available in JRT1b, so these were diagnostic component results only.

## JRT1c Summary

Classification: `NO_STABLE_JRT1_TRAJECTORY_GAIN`

JRT1c used deterministic `k=1` validation odometry chains from the train-CV folds. No variant passed the trajectory gate, and there is no selected next-stage candidate.

| variant | ATE_proxy | drift_proxy | path_ratio | rot_mean | tdir_mean | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A_s5_no_refiner_reference | 0.174705 | 0.174946 | 1.146411 | 20.844836 | 45.816231 | REFERENCE |
| B_rot_only_residual | 0.178203 | 0.182629 | 1.146411 | 7.695970 | 45.816231 | FAIL |
| C_tdir_residual_cosine | 0.186241 | 0.195738 | 1.146411 | 20.844836 | 46.180848 | FAIL |
| D_joint_rtdir_residual | 0.180036 | 0.179990 | 1.146411 | 4.804207 | 44.890367 | FAIL |
| E_joint_rtdir_tdir_weighted | 0.185991 | 0.187880 | 1.146411 | 5.200944 | 46.576359 | FAIL |
| F_joint_rtdir_tdir_hardcase_weighted | 0.191893 | 0.193065 | 1.146411 | 7.542631 | 48.327126 | FAIL |

The component improvements did not transfer to trajectory-level proxy gains. All variants failed versus the A reference. The path_ratio proxy was outside the safe band for all variants, including A.

## Final Decision

Do not proceed to JRT1d final test. No JRT1 variant is selected over S5. S5 remains final clean candidate.

## Final Classification

`NO_STABLE_JRT1_TRAJECTORY_GAIN`

## Thesis / Report Usage

JRT1 is a negative but informative result. It shows that prediction-space local residual refinement can improve pairwise components, but does not reliably improve trajectory accumulation. This supports the conclusion that solving R/tdir coupling likely requires a deeper multi-frame geometric architecture rather than lightweight pairwise residual correction.

## Caveats

- train-CV only
- no final test
- proxy trajectory is not official final test
- small sequence protocol
- S5 unchanged
- no practical-ready claim
