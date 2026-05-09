# ORB1c ORB-SLAM3 Fisheye Cam0 Run Report

## Executive summary

- Experiment: `ORB1c_run_orbslam3_fisheye_cam0_scene01_seq03`
- Final classification: `ORB1C_RUN_COMPLETE`
- Input frames: `454`
- Estimated poses: `273`
- Tracking success rate: `0.601322`
- Checkpoint: `checkpoints/ORB1c_orbslam3_fisheye_run.json`

This step runs ORB-SLAM3 and exports trajectory only.
It does not evaluate ATE/drift/path_ratio.
It does not modify S5 locked metrics or policy.

## Precheck status

- ORB1a: `RAW_FISHEYE_READY`
- ORB1b: `ORB1B_READY_FITTED_KB8`
- ORB0b: `ORB0_READY_AFTER_REMEDIATION`
- ORB0b ready for ORB1c: `True`
- Passed: `True`

## Input sequence summary

- Raw root: `/home/dovetao/graduation_design_demo/data/FisheyeView`
- Sequence: `scene01/seq03`
- Camera: `cam0`
- Image manifest: `external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_images.txt`
- Timestamp manifest: `external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_timestamps.txt`
- Images: `454`
- Timestamps: `454`

## Camera model warning

YAML is a fitted KB8 compatibility approximation, not original factory KannalaBrandt8 calibration. ORB1c used an OpenCV FileStorage-compatible runtime copy at `external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_orbslam3_runtime.yaml` so the original ORB1b YAML remained unchanged.

## ORB-SLAM3 runtime command

```bash
bash external_baselines/runners/run_orbslam3_fisheye_cam0.sh
```

Runner: `external_baselines/runners/orbslam3_mono_fisheye_runner`

## Runtime result

- Exit code: `0`
- stdout log: `external_baselines/results/orbslam3_fisheye_cam0/orbslam3_stdout.log`
- stderr log: `external_baselines/results/orbslam3_fisheye_cam0/orbslam3_stderr.log`
- Runner summary: `frames=454 success=273 failure=181`

## Trajectory output path

`external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt`

## Tracking summary

- num input frames: `454`
- num estimated poses: `273`
- tracking success rate: `0.601322`

## Baseline metric status

- `baseline_metrics.evaluation_run=false`
- ATE: `null`
- drift: `null`
- path_ratio: `null`

## Final classification

`ORB1C_RUN_COMPLETE`

Allowed classifications: `ORB1C_RUN_COMPLETE`, `ORB1C_TRACKING_FAILED`, `ORB1C_SEQUENCE_EXPORT_FAILED`, `ORB1C_BLOCKED_BY_PRECHECK`, `ORB1C_RUNTIME_ERROR`, `ORB1C_ERROR`.

## Notes

- The run initialized ORB-SLAM3, created maps, and produced a TUM-format estimated trajectory.
- Some local map tracking failures were reported in stdout, but the run completed with nonzero estimated poses.
- S5 policy, S5 locked metrics, final manifest, split, and official evaluator were not modified.
