# JRT1c Train-CV Trajectory Proxy Gate

## Scope

Train-CV trajectory proxy only. No final test is run. S5 remains the final clean candidate.

## JRT1b Recap

JRT1b produced component improvements: C was the strongest translation-direction-only variant, while E/F were balanced joint candidates. JRT1b left trajectory gate unavailable.

## Trajectory Proxy Construction

Option 2 was used. The chain source is the JRT1b train-split cache and reproduced train-CV fold assignment. The proxy filters validation pairs to `k=1`, groups by sequence, orders by timestamp, and composes local odometry chains. Prediction translation uses `tmag_s5 * tdir_refined`; S5 predicted magnitude is preserved and GT magnitude is not used for prediction.

- chain source: `JRT1b train-CV validation folds from train split cache`
- k filtering: `k=1 only`
- tmag policy: `S5 predicted tmag reused for prediction`
- no GT tmag used for prediction: `True`

## Per-Fold Results

| fold | variant | ATE_proxy | drift_proxy | path_ratio_proxy | rot_mean | tdir_mean | tdir_cos | num_chains | num_pairs_used |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | A_s5_no_refiner_reference | 0.180875 | 0.182501 | 1.158655 | 20.856220 | 45.090519 | 0.619649 | 41 | 51 |
| 0 | B_rot_only_residual | 0.185850 | 0.187158 | 1.158655 | 8.372541 | 45.090519 | 0.619649 | 41 | 51 |
| 0 | C_tdir_residual_cosine | 0.189931 | 0.200413 | 1.158655 | 20.856220 | 44.335899 | 0.617479 | 41 | 51 |
| 0 | D_joint_rtdir_residual | 0.187438 | 0.186670 | 1.158655 | 4.112818 | 43.182987 | 0.631436 | 41 | 51 |
| 0 | E_joint_rtdir_tdir_weighted | 0.196913 | 0.199879 | 1.158655 | 4.641282 | 46.340271 | 0.598325 | 41 | 51 |
| 0 | F_joint_rtdir_tdir_hardcase_weighted | 0.207940 | 0.205858 | 1.158654 | 6.766007 | 46.613697 | 0.587050 | 41 | 51 |
| 1 | A_s5_no_refiner_reference | 0.158585 | 0.162308 | 1.139983 | 20.782812 | 46.402454 | 0.608690 | 39 | 42 |
| 1 | B_rot_only_residual | 0.169897 | 0.177449 | 1.139983 | 5.866008 | 46.402454 | 0.608690 | 39 | 42 |
| 1 | C_tdir_residual_cosine | 0.178565 | 0.187714 | 1.139983 | 20.782812 | 47.272194 | 0.579759 | 39 | 42 |
| 1 | D_joint_rtdir_residual | 0.175063 | 0.174740 | 1.139982 | 5.240752 | 43.680927 | 0.625389 | 39 | 42 |
| 1 | E_joint_rtdir_tdir_weighted | 0.185825 | 0.180426 | 1.139982 | 6.080258 | 45.882595 | 0.586910 | 39 | 42 |
| 1 | F_joint_rtdir_tdir_hardcase_weighted | 0.185949 | 0.185241 | 1.139982 | 8.536525 | 46.867649 | 0.577794 | 39 | 42 |
| 2 | A_s5_no_refiner_reference | 0.184654 | 0.180030 | 1.140597 | 20.895475 | 45.955719 | 0.608270 | 40 | 46 |
| 2 | B_rot_only_residual | 0.178861 | 0.183280 | 1.140597 | 8.849360 | 45.955719 | 0.608270 | 40 | 46 |
| 2 | C_tdir_residual_cosine | 0.190226 | 0.199087 | 1.140597 | 20.895475 | 46.934452 | 0.582465 | 40 | 46 |
| 2 | D_joint_rtdir_residual | 0.177609 | 0.178559 | 1.140597 | 5.059052 | 47.807186 | 0.556319 | 40 | 46 |
| 2 | E_joint_rtdir_tdir_weighted | 0.175234 | 0.183333 | 1.140597 | 4.881292 | 47.506210 | 0.585136 | 40 | 46 |
| 2 | F_joint_rtdir_tdir_hardcase_weighted | 0.181790 | 0.188096 | 1.140597 | 7.325359 | 51.500031 | 0.515851 | 40 | 46 |

## Mean CV Results

| variant | mean_ATE_proxy | mean_drift_proxy | mean_path_ratio_proxy | mean_rot | mean_tdir | mean_tdir_cos | gate_status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A_s5_no_refiner_reference | 0.174705 | 0.174946 | 1.146411 | 20.844836 | 45.816231 | 0.612203 | REFERENCE |
| B_rot_only_residual | 0.178203 | 0.182629 | 1.146411 | 7.695970 | 45.816231 | 0.612203 | FAIL |
| C_tdir_residual_cosine | 0.186241 | 0.195738 | 1.146411 | 20.844836 | 46.180848 | 0.593234 | FAIL |
| D_joint_rtdir_residual | 0.180036 | 0.179990 | 1.146411 | 4.804207 | 44.890367 | 0.604381 | FAIL |
| E_joint_rtdir_tdir_weighted | 0.185991 | 0.187880 | 1.146411 | 5.200944 | 46.576359 | 0.590124 | FAIL |
| F_joint_rtdir_tdir_hardcase_weighted | 0.191893 | 0.193065 | 1.146411 | 7.542631 | 48.327126 | 0.560232 | FAIL |

## Gate Decision

- `A_s5_no_refiner_reference`: `REFERENCE`
- `B_rot_only_residual`: `FAIL`
- `C_tdir_residual_cosine`: `FAIL`
- `D_joint_rtdir_residual`: `FAIL`
- `E_joint_rtdir_tdir_weighted`: `FAIL`
- `F_joint_rtdir_tdir_hardcase_weighted`: `FAIL`
- selected candidate for next stage: `None`

## Interpretation

The report compares every evaluated variant against A on the same fold and chain set. Component gains are considered only diagnostic unless the trajectory proxy gate also passes. If C improves tdir but worsens or fails trajectory proxy, it is not a next-stage candidate. E/F must keep balanced rotation behavior and satisfy ATE, drift, and path-ratio proxy criteria.

## Final Classification

`NO_STABLE_JRT1_TRAJECTORY_GAIN`

## Next Step

If the gate passes, prepare a separate JRT1d final-test evaluation plan. Otherwise stop JRT1 as a negative or diagnostic result.

## Caveats

- train-CV only
- no final candidate selected
- no final test
- S5 locked metrics unchanged
- proxy trajectory is not final official test
- small sequence protocol
