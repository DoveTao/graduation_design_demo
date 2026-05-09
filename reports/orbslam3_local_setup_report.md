# ORB0 ORB-SLAM3 Local Setup Report

## Executive summary

- Experiment: `ORB0_orbslam3_local_setup`
- Final classification: `ORB0_BUILD_FAILED`
- Ready for ORB1c: `False`
- Source was cloned outside the project repository at `/home/dovetao/third_party/ORB_SLAM3`.
- Checkpoint: `checkpoints/ORB0_orbslam3_local_setup.json`

This setup step does not run ORB-SLAM3 on the dataset and does not produce baseline metrics.

## Source location

- Target path: `/home/dovetao/third_party/ORB_SLAM3`
- Exists: `True`
- Repository clean: `True`

## Git commit / branch

- Branch: `master`
- Commit: `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`

## Dependency snapshot

- gcc: `gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0`
- g++: `g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0`
- cmake: `cmake version 3.22.1`
- OpenCV: `4.5.4`
- Eigen: `3.4.0`
- Pangolin: `pkg-config pangolin not found`
- CMAKE_PREFIX_PATH used for build: `/opt/ros/humble`

## Build command

```bash
CMAKE_PREFIX_PATH=/opt/ros/humble ./build.sh
```

Build log: `/home/dovetao/graduation_design_demo/logs/orbslam3_build.log`

## Build result

- Attempted: `True`
- Success: `False`
- Error summary: build failed while compiling `src/System.cc.o`. The compiler included `/opt/ros/humble/include/sigslot/signal.hpp` and reported template/member errors around `cow_read`, `cow_write`, and `m_slots`; `make` exited with code `2`.

## Executables / libraries found

- `lib/libORB_SLAM3.so`: `False`
- Executables found: `[]`

## Whether ready for ORB1c

`False`. ORB1c should remain blocked until ORB-SLAM3 builds successfully and the required monocular executable plus `libORB_SLAM3.so` are present.

## Final classification

`ORB0_BUILD_FAILED`

Allowed classifications: `ORB0_READY`, `ORB0_SOURCE_UNAVAILABLE`, `ORB0_BUILD_FAILED`, `ORB0_ERROR`.

## Notes

- ORB-SLAM3 source was not placed inside tracked project source.
- No baseline run, trajectory, or ORB-SLAM3 metric was produced.
- ORB1a and ORB1b classifications were not changed.
- S5 policy, locked metrics, final manifest, split, and official evaluator were not modified.
