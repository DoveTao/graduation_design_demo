# External Algorithm Baseline Protocol

## Scope
This protocol defines a fair external baseline comparison framework for the thesis. Only trajectories generated on the same exported sequence under the same evaluation protocol can be used for numeric comparison.
Published results from different datasets are not compared directly against S5.

## Dataset and Protocol
- sequence: `scene01/seq03`
- input modality: `monocular panorama / equirectangular RGB`
- camera intrinsics availability: `camera.yaml` with pinhole-unavailable caveat at `external_baselines/dataset/scene01_seq03/camera.yaml`
- GT trajectory: `external_baselines/dataset/scene01_seq03/groundtruth_tum.txt`
- alignment modes: `none / se3 / sim3`
- metrics: `ATE`, `drift` (RPE-like translation RMSE), `path_ratio`, `tracking_success_rate`
- scale-sensitive path_ratio must be reported in addition to any Sim(3)-aligned ATE.

## Baseline Methods
| method | category | input | loop_closure | learning_based | run_status | output_trajectory_path | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ORB-SLAM2 | feature-based SLAM | monocular / stereo / RGB-D | yes | no | pending_external_run | (missing) | Protocol-compatible only if built and run on the exported sequence. |
| ORB-SLAM3 | feature-based visual / visual-inertial SLAM | monocular / stereo / RGB-D / visual-inertial | yes | no | pending_external_run | (missing) | Protocol-compatible only if built and run on the exported sequence. |
| DSO | direct sparse visual odometry | monocular | no | no | pending_external_run | (missing) | Requires an input mode compatible with the exported panorama sequence or a documented adapter. |
| DROID-SLAM | deep learning SLAM | monocular / stereo / RGB-D | implicit / backend-dependent | yes | pending_external_run | (missing) | Protocol-compatible only after a same-sequence trajectory is generated locally. |
| S5 | proposed method | monocular panorama / equirectangular RGB | no | yes | locked_internal_result | internal clean eval / locked metrics | Final clean candidate under the historical protocol. |

## Literature Context
- ORB-SLAM2 and ORB-SLAM3 represent classical feature-based SLAM baselines.
- DSO represents a direct sparse monocular odometry baseline.
- DROID-SLAM represents a learning-based SLAM baseline.
- Published results from other datasets are used only for method context, not for direct numeric comparison with S5.

## Caveats
- external baselines require matching input modality
- monocular baselines have scale ambiguity
- Sim(3)-aligned ATE and scale-sensitive path_ratio answer different questions
- results are not comparable to published numbers from different datasets
