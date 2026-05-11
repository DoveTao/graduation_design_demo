# S5D3 Validation and Alignment-aware Component Audit

## Executive summary

Final classification: `S5D3_COMPLETE_VALIDATION_CLEAN`

## Validation status

- verify_final_candidate: `PASS`
- project_health_check: `PASS`
- unittest: `PASS`
- s6_final_clean_candidate_lockdown_audit.py --eval-only: `PASS`
- root cause summary: PASS; no lockdown mismatch found.
- branch clean: `False`

## Why S5D3 is needed

S5D2 raw tdir is near 90-100 deg for both S5 and ORB-SLAM3. ORB-SLAM3 can still have low aligned ATE, so raw local tdir can be confounded by gauge/alignment and tiny-step pairs.

## Protocol

- TUM pose convention: `timestamp tx ty tz qx qy qz qw`
- T_wc convention: translation is world-frame camera position, quaternion order is `qx qy qz qw`
- relative transform: `inverse(T_i) @ T_j`
- alignment modes: `none / se3 / sim3`
- threshold sweep: absolute and `median_gt_step * {0.01,0.05,0.10}`
- matched-pair fairness: same ORB tracked adjacent timestamp pairs; no ORB interpolation
- no official S5 metric replacement

## Results

Recommended threshold: `median_gt_step * 0.05 = 0.000375`

### alignment=none

| set | num valid pairs | valid ratio | rot mean/median/p90 | tdir mean/median/p90 | tdir mean cosine | tmag mean/median/p90 log | tmag mean/median ratio |
|---|---:|---:|---|---|---:|---|---|
| s5_full | 453.000000 | 1.000000 | 20.825151/20.770591/21.356673 | 91.392419/95.679859/139.551260 | -0.028522 | 2.560973/2.856695/4.164398 | 27.174142/17.403920 |
| orbslam3_tracked | 179.000000 | 0.658088 | 0.625448/0.604320/0.790470 | 100.876813/110.038101/140.516282 | -0.172633 | 1.301885/1.185641/2.316543 | 0.909747/0.370701 |
| s5_on_orb_matched_pairs | 272.000000 | 1.000000 | 20.762745/20.762964/20.897836 | 95.811144/98.763815/139.161361 | -0.093894 | 3.439775/3.537678/4.310919 | 41.645272/34.386996 |
| orbslam3_on_matched_pairs | 179.000000 | 0.658088 | 0.625448/0.604320/0.790470 | 100.876813/110.038101/140.516282 | -0.172633 | 1.301885/1.185641/2.316543 | 0.909747/0.370701 |

### alignment=se3

| set | num valid pairs | valid ratio | rot mean/median/p90 | tdir mean/median/p90 | tdir mean cosine | tmag mean/median/p90 log | tmag mean/median ratio |
|---|---:|---:|---|---|---:|---|---|
| s5_full | 453.000000 | 1.000000 | 20.825151/20.770591/21.356673 | 91.392419/95.679859/139.551260 | -0.028522 | 2.560973/2.856695/4.164398 | 27.174142/17.403920 |
| orbslam3_tracked | 179.000000 | 0.658088 | 0.625448/0.604320/0.790470 | 100.876813/110.038101/140.516282 | -0.172633 | 1.301885/1.185641/2.316543 | 0.909747/0.370701 |
| s5_on_orb_matched_pairs | 272.000000 | 1.000000 | 20.762745/20.762964/20.897836 | 95.811144/98.763815/139.161361 | -0.093894 | 3.439775/3.537678/4.310919 | 41.645272/34.386996 |
| orbslam3_on_matched_pairs | 179.000000 | 0.658088 | 0.625448/0.604320/0.790470 | 100.876813/110.038101/140.516282 | -0.172633 | 1.301885/1.185641/2.316543 | 0.909747/0.370701 |

### alignment=sim3

| set | num valid pairs | valid ratio | rot mean/median/p90 | tdir mean/median/p90 | tdir mean cosine | tmag mean/median/p90 log | tmag mean/median ratio |
|---|---:|---:|---|---|---:|---|---|
| s5_full | 453.000000 | 1.000000 | 20.825151/20.770591/21.356673 | 91.392419/95.679859/139.551260 | -0.028522 | 1.287677/1.282280/2.309257 | 2.788342/1.785819 |
| orbslam3_tracked | 254.000000 | 0.933824 | 0.625448/0.604320/0.790470 | 97.598384/108.376509/140.093521 | -0.123146 | 1.381172/1.090739/2.780737 | 4.774020/1.655887 |
| s5_on_orb_matched_pairs | 272.000000 | 1.000000 | 20.762745/20.762964/20.897836 | 95.811144/98.763815/139.161361 | -0.093894 | 1.304635/1.310946/2.103418 | 4.273227/3.528455 |
| orbslam3_on_matched_pairs | 254.000000 | 0.933824 | 0.625448/0.604320/0.790470 | 97.598384/108.376509/140.093521 | -0.123146 | 1.381172/1.090739/2.780737 | 4.774020/1.655887 |

## Interpretation

- S5 rotation error confirmed: `True`
- S5 tmag over-scaling confirmed: `True`
- tdir reliable after threshold/alignment: `False`
- dominant S5 error source: `mixed`

## Recommendations

- Run dense export convention audit on axis/sign/frame consistency.
- Add explicit tmag scale calibration regularization across short windows.
- Strengthen SO(3) geodesic rotation loss and short-window pose consistency.
- If tdir remains high after alignment-aware thresholds, add tdir-focused loss.
- Use ORB-SLAM3 successful segments as distillation targets.

## Caveats

- S5D3 does not replace the official S5 locked result.
- S5 official locked metrics/policy were not changed.
- ORB-SLAM3 remains an external strong baseline.
- ORB-SLAM3 has partial coverage.
- ORB-SLAM3 uses fitted KB8 compatibility calibration.
- same_input_protocol=false.
