# ORB-SLAM3 Environment Setup Report

## Scope
This report documents the attempt to prepare `ORB-SLAM3 monocular` as the first external baseline for the exported `scene01/seq03` sequence. It does not change S5, does not add a new candidate, and does not invent any external baseline metrics.

## Repository Target
- upstream checked: `https://github.com/UZ-SLAMLab/ORB_SLAM3.git`
- upstream heads probe: success
- probed master commit: `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`

## Dependency Summary
- compiler toolchain: available
- CMake: available
- OpenCV 4.5.4: available
- Eigen 3.4.0: available
- Boost / GLEW / SuiteSparse: available
- Pangolin:
  - installed through ROS Humble packages under `/opt/ros/humble`
  - not visible through default `pkg-config`
  - likely needs explicit `CMAKE_PREFIX_PATH=/opt/ros/humble`

## Download Attempt
- attempt 1: `git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git external_baselines/third_party/ORB_SLAM3`
  - result: stalled during object transfer
- attempt 2: `curl -L https://codeload.github.com/UZ-SLAMLab/ORB_SLAM3/tar.gz/refs/heads/master`
  - result: partial download reached about `9.97 MB`
  - failure: `curl: (28) Operation timed out after 180000 milliseconds`
- attempt 3: resumed tarball fetch
  - result: failed
  - failure: `curl: (33) HTTP server doesn't seem to support byte ranges`

## Build Status
- build command: not run to completion
- build success: no
- primary setup blocker:
  - third-party source download was not completed under the current network behavior

## Fair-Protocol Compatibility Audit
- exported input is `monocular panorama / equirectangular RGB`
- current `camera.yaml` explicitly records:
  - `camera_model: equirectangular_panorama`
  - `fx/fy/cx/cy: null`
- ORB-SLAM3 monocular expects a calibrated pinhole or fisheye camera model
- fair direct runability on the current export: no

## Artifacts Added
- runner: `external_baselines/runners/run_orbslam3_mono.sh`
- config placeholder: `external_baselines/config/orbslam3_scene01_seq03.yaml`

## Current Outcome
- ORB-SLAM3 build success: `false`
- ORB-SLAM3 runnable on current fair export: `false`
- generated trajectory:
  - `external_baselines/results/orbslam3/scene01_seq03_est_tum.txt`: not generated

## Failure Summary
1. Third-party source retrieval was unreliable and did not finish within the attempted fetch windows.
2. Even with a successful download, the current fair export is not directly compatible with ORB-SLAM3 monocular because it lacks trustworthy pinhole/fisheye calibration.

## Next Fair Steps
- Do not fabricate camera intrinsics.
- If ORB-SLAM3 remains the target baseline, first add a documented projection/calibration pipeline that converts the panorama input into a fair ORB-SLAM3-compatible camera stream.
- After that, retry source download and build with:
  - `CMAKE_PREFIX_PATH=/opt/ros/humble`
  - explicit Pangolin path if needed
