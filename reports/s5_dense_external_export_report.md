# S5 Dense External Trajectory Export Report

## Executive summary

- Experiment: `S5D_dense_external_trajectory_export`
- Final classification: `S5D_DENSE_EXPORT_READY`
- Checkpoint: `checkpoints/S5D_dense_external_export.json`
- Trajectory: `external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt`
- Estimated poses: `454` / timestamps `454`
- Coverage: `1.0`

This export does not modify S5 locked metrics or official evaluator.
S5 official locked metrics are unchanged and remain separate from this external diagnostic comparison.

## Why dense export is needed

The previous S5 external TUM export was sparse diagnostic only. A dense diagnostic export is needed before comparing S5 and ORB-SLAM3 with the same GT, same external evaluator, and same alignment modes.

## S5 pose source

- Policy: `checkpoints/S5_clean_tmag_calibration_policy.json`
- Base checkpoint: `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- Pose source: `S5 locked-policy model predictions on adjacent panorama pairs; no GT pose copy`
- Composition convention: `adjacent relative transforms are composed with train_mvp._compose_rel_pose_np; TUM translation is the camera center from train_mvp._camera_center_from_T_c0_np`
- Frame convention: `camera-center translation is written in TUM timestamp tx ty tz qx qy qz qw order`

## Timestamp alignment

- Timestamp file: `external_baselines/dataset/scene01_seq03/timestamps.txt`
- Timestamp match status: `aligned_exact`
- Duplicate timestamps: `0`
- Monotonic timestamps: `True`

## TUM format validation

- Format: `TUM`
- Trajectory file line count: `454`

## Coverage

- num_s5_poses: `454`
- num_timestamps: `454`
- coverage: `1.0`
- dense_or_sparse: `dense`

## GT leakage prevention

- GT poses are not copied into the exported estimate.
- The exporter uses S5 model predictions and the locked policy wrapper.
- GT is read only after export for leakage checks and validation.
- GT leakage check passed: `True`
- Leakage check reason: `not_identical_to_groundtruth`

## Final classification

`S5D_DENSE_EXPORT_READY`

Allowed classifications: `S5D_DENSE_EXPORT_READY`, `S5D_SPARSE_EXPORT_ONLY`, `S5D_DENSE_EXPORT_UNAVAILABLE`, `S5D_EXPORT_REJECTED_GT_LEAKAGE_RISK`, `S5D_EXPORT_ERROR`.
