# S5 vs ORB-SLAM3 Same-Evaluator Comparison

## Executive summary

- Experiment: `S5D_same_evaluator_comparison`
- Final classification: `S5D_SAME_EVALUATOR_COMPARISON_COMPLETE`
- Checkpoint: `checkpoints/S5D_same_evaluator_comparison_results.json`
- Comparison type: `same_gt_same_external_evaluator_same_alignment_policy`
- same_input_protocol: `False`

S5 official locked metrics are unchanged and remain distinct from this external same-evaluator comparison.
This external comparison does not replace the official S5 locked result.

## Comparison protocol

Same sequence, same GT, same external evaluator, same alignment modes: `none`, `se3`, `sim3`.

## Input protocol note

S5 uses the project panorama/S5 representation, while ORB-SLAM3 uses raw fisheye cam0. The comparison is fair at the trajectory-evaluator level, not identical at the input-modality level.

## S5 external trajectory summary

- Trajectory: `external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt`
- Estimated poses: `454`
- Input frames: `454`
- Coverage: `1.0`

## ORB-SLAM3 external trajectory summary

- Trajectory: `external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt`
- Estimated poses: `273`
- Input frames: `454`
- Coverage: `0.6013215859030837`
- Calibration caveat: `fitted KB8 compatibility approximation, not native factory KB8`

## Metrics table

| Method | Input | Evaluator | Alignment | ATE | Drift | Path ratio | Estimated poses | Input frames | Coverage | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| S5 dense external export | S5 panorama/project representation | external evaluator | none | 21.681522044975264 | 0.21683137477814862 | 2.777267572676944 | 454 | 454 | 1.0 | S5 diagnostic dense export; official locked result unchanged |
| S5 dense external export | S5 panorama/project representation | external evaluator | se3 | 8.231468716451547 | 0.23274388321733164 | 2.777267572676944 | 454 | 454 | 1.0 | S5 diagnostic dense export; official locked result unchanged |
| S5 dense external export | S5 panorama/project representation | external evaluator | sim3 | 4.07912293550008 | 0.13382808429229665 | 2.777267572676944 | 454 | 454 | 1.0 | S5 diagnostic dense export; official locked result unchanged |
| ORB-SLAM3 fisheye cam0 | raw fisheye cam0 | external evaluator | none | 11.46685113827757 | 0.03737053832593786 | 0.2998258665660257 | 273 | 454 | 0.6013215859030837 | fitted KB8 compatibility YAML; partial tracking coverage |
| ORB-SLAM3 fisheye cam0 | raw fisheye cam0 | external evaluator | se3 | 0.30854441069248173 | 0.0374760642069601 | 0.2998258665660257 | 273 | 454 | 0.6013215859030837 | fitted KB8 compatibility YAML; partial tracking coverage |
| ORB-SLAM3 fisheye cam0 | raw fisheye cam0 | external evaluator | sim3 | 0.224292165986624 | 0.06452708470276013 | 0.2998258665660257 | 273 | 454 | 0.6013215859030837 | fitted KB8 compatibility YAML; partial tracking coverage |

## Coverage table

| Method | Estimated poses | Input frames | Coverage |
| --- | ---: | ---: | ---: |
| S5 dense external export | 454 | 454 | 1.0 |
| ORB-SLAM3 fisheye cam0 | 273 | 454 | 0.6013215859030837 |

## Interpretation

S5 and ORB-SLAM3 are compared here under the same trajectory evaluator and alignment policy, but they do not use identical input modalities. ORB-SLAM3 has partial tracking coverage, while the S5 dense diagnostic export covers the full timestamp stream. S5 official locked result remains unchanged.

## Final classification

`S5D_SAME_EVALUATOR_COMPARISON_COMPLETE`

Allowed classifications: `S5D_SAME_EVALUATOR_COMPARISON_COMPLETE`, `S5D_SAME_EVALUATOR_COMPARISON_PARTIAL`, `S5D_SAME_EVALUATOR_COMPARISON_BLOCKED`, `S5D_SAME_EVALUATOR_COMPARISON_ERROR`.
