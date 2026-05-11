# S5 vs ORB-SLAM3 Component Diagnostics

## Executive Summary

Final classification: `S5D2_COMPONENT_DIAGNOSTICS_COMPLETE`
Dominant S5 error source: `mixed`

## Why component diagnostics are needed

Trajectory-level ATE, drift, and path_ratio describe overall trajectory behavior, but they do not isolate whether the gap is mainly due to rotation, translation direction, translation magnitude / scale, or cumulative composition drift.

## Inputs

- S5 dense trajectory: `/home/dovetao/graduation_design_demo/external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt`
- ORB-SLAM3 trajectory: `/home/dovetao/graduation_design_demo/external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt`
- GT: `external_baselines/dataset/scene01_seq03/groundtruth_tum.txt`
- timestamps: `external_baselines/dataset/scene01_seq03/timestamps.txt`

## Protocol

- TUM format: `timestamp tx ty tz qx qy qz qw`
- T_wc convention: translation is world-frame camera position and quaternion is interpreted as `qx qy qz qw`
- relative transform definition: `inverse(T_i) @ T_j`
- rotation error: geodesic angle of `R_est_rel @ inverse(R_gt_rel)`
- translation direction error: `arccos(dot(normalize(t_est), normalize(t_gt)))`
- translation magnitude log error: `abs(log(norm(t_est) / norm(t_gt)))`
- min translation threshold: `1e-08`
- timestamp matching tolerance: `1e-06`

## Results Table 1: S5 Full Sequence

| num poses | num pairs | rot mean | rot median | rot p90 | tdir mean | tdir median | tdir p90 | tmag mean log err | tmag median log err | tmag p90 log err | tmag mean ratio | tmag median ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 454.000000 | 453.000000 | 20.825151 | 20.770591 | 21.356673 | 91.392419 | 95.679859 | 139.551260 | 2.560973 | 2.856695 | 4.164398 | 27.174142 | 17.403920 |

## Results Table 2: ORB-SLAM3 Tracked Sequence

| num poses | num pairs | rot mean | rot median | rot p90 | tdir mean | tdir median | tdir p90 | tmag mean log err | tmag median log err | tmag p90 log err | tmag mean ratio | tmag median ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 273.000000 | 272.000000 | 0.625448 | 0.604320 | 0.790470 | 96.086118 | 105.884537 | 140.239046 | 2.053944 | 1.591362 | 4.264499 | 0.616085 | 0.208860 |

## Results Table 3: Matched Pair Comparison

| method | num pairs | rot mean | rot median | rot p90 | tdir mean | tdir median | tdir p90 | tdir mean cosine | tmag mean log err | tmag median log err | tmag p90 log err | tmag mean ratio | tmag median ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S5-on-ORB-matched-pairs | 272.000000 | 20.762745 | 20.762964 | 20.897836 | 95.811144 | 98.763815 | 139.161361 | -0.093894 | 3.439775 | 3.537678 | 4.310919 | 41.645272 | 34.386996 |
| ORB-SLAM3-on-same-matched-pairs | 272.000000 | 0.625448 | 0.604320 | 0.790470 | 96.086118 | 105.884537 | 140.239046 | -0.099288 | 2.053944 | 1.591362 | 4.264499 | 0.616085 | 0.208860 |

## Interpretation

- On ORB-matched pairs, S5 mean tmag ratio is substantially above 1, which points to local translation magnitude over-scaling.
- S5 translation-direction error on matched pairs is materially elevated, so local tdir quality also contributes.
- S5 rotation error on matched pairs is high enough that local orientation quality cannot be ignored.
- ORB-SLAM3 remains a high-precision partial-tracking baseline: local component quality is measured only on tracked segments.

## Recommendations

- prioritize translation magnitude calibration under the dense-export / same-evaluator scope
- revisit short-window pose consistency with explicit scale-aware objectives
- add stronger tdir supervision or a direction-focused loss on matched local motions
- consider stronger SO(3) geodesic rotation supervision and short-window rotational consistency
- treat ORB-SLAM3 as a successful-segment teacher or distillation target rather than as a full-coverage replacement

## Caveats

- This diagnostic does not replace the official S5 locked result.
- S5 official locked metrics/policy were not changed.
- ORB-SLAM3 uses raw fisheye cam0 with fitted KB8 compatibility calibration.
- S5 and ORB-SLAM3 are not same-input-protocol.
- ORB-SLAM3 has partial coverage.
