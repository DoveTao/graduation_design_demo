# ORB1d ORB-SLAM3 Fisheye Evaluation Report

## Executive summary

- Experiment: `ORB1d_evaluate_orbslam3_fisheye_trajectory`
- Final classification: `ORB1D_EVALUATION_COMPLETE`
- Evaluator: `tools/evaluate_external_baseline_trajectory.py`
- Checkpoint: `checkpoints/ORB1d_orbslam3_fisheye_evaluation_results.json`
- tracking_success_rate: `273 / 454 = 0.6013215859030837`

S5 locked metrics/policy were not changed.
ORB-SLAM3 results are external baseline results.
ORB-SLAM3 results do not replace S5.

## ORB1c trajectory summary

- Trajectory: `external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt`
- Ground truth: `external_baselines/dataset/scene01_seq03/groundtruth_tum.txt`
- Num input frames: `454`
- Num estimated poses: `273`
- Num matched poses used by evaluator: `273`
- Tracking success rate: `0.6013215859030837`

## Calibration caveat

ORB-SLAM3 used a fitted KB8 compatibility approximation, not native factory KB8 calibration. The camera model is recorded as `fitted_KB8_compatibility_approximation`, with `native_factory_kb8=false`.

## Evaluation protocol

The trajectory was evaluated with:

```bash
/home/dovetao/miniconda3/envs/pytorch/bin/python tools/evaluate_external_baseline_trajectory.py --trajectory external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt --groundtruth external_baselines/dataset/scene01_seq03/groundtruth_tum.txt --alignment none
/home/dovetao/miniconda3/envs/pytorch/bin/python tools/evaluate_external_baseline_trajectory.py --trajectory external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt --groundtruth external_baselines/dataset/scene01_seq03/groundtruth_tum.txt --alignment se3
/home/dovetao/miniconda3/envs/pytorch/bin/python tools/evaluate_external_baseline_trajectory.py --trajectory external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt --groundtruth external_baselines/dataset/scene01_seq03/groundtruth_tum.txt --alignment sim3
```

Raw evaluator outputs:

- `external_baselines/results/orbslam3_fisheye_cam0/eval_alignment_none.json`
- `external_baselines/results/orbslam3_fisheye_cam0/eval_alignment_se3.json`
- `external_baselines/results/orbslam3_fisheye_cam0/eval_alignment_sim3.json`

## Results

| alignment | status | ATE | drift | path_ratio | matched poses |
| --- | --- | ---: | ---: | ---: | ---: |
| none | ok | 11.466851 | 0.037371 | 0.299826 | 273 |
| se3 | ok | 0.308544 | 0.037476 | 0.299826 | 273 |
| sim3 | ok | 0.224292 | 0.064527 | 0.299826 | 273 |

## Interpretation of none / se3 / sim3

- `none`: no trajectory alignment; this exposes global frame, origin, and scale disagreement.
- `se3`: rigid alignment only; scale remains fixed, so monocular scale inconsistency is still visible.
- `sim3`: similarity alignment; scale can be corrected, so ATE can improve for monocular trajectories.

## Warning about sparse tracking coverage

ORB-SLAM3 produced 273 estimated poses from 454 input frames. These metrics are computed only on the matched subset, so they describe the tracked portion of the run, not full-sequence dense coverage.

## Comparison caveat against S5

These results should not be inserted into the official S5 main table as same-scope replacement metrics. S5 dense external trajectory export is not available; S5 external TUM export is sparse diagnostic only. `same_evaluator_main_table_allowed=false`.

## Final classification

`ORB1D_EVALUATION_COMPLETE`

Allowed classifications: `ORB1D_EVALUATION_COMPLETE`, `ORB1D_PARTIAL_EVALUATION`, `ORB1D_EVALUATION_FAILED`, `ORB1D_BLOCKED_BY_ORB1C`, `ORB1D_ERROR`.
