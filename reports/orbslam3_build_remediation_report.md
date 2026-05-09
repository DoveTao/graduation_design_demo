# ORB0b ORB-SLAM3 Build Remediation Report

## Executive summary

- Experiment: `ORB0b_orbslam3_build_remediation_without_running_baseline`
- Final classification: `ORB0_READY_AFTER_REMEDIATION`
- Source path: `/home/dovetao/third_party/ORB_SLAM3`
- Remediation that fixed the build: `cpp14_patch`
- Ready for ORB1c: `True`
- Checkpoint: `checkpoints/ORB0b_orbslam3_build_remediation.json`

This remediation step builds ORB-SLAM3 only.
It does not run ORB-SLAM3 on the dataset.
It does not generate a trajectory.
It does not produce baseline metrics.

## Original ORB0 failure summary

The original ORB0 build used `CMAKE_PREFIX_PATH=/opt/ros/humble ./build.sh` and failed while compiling `src/System.cc.o`. The compiler pulled in `/opt/ros/humble/include/sigslot/signal.hpp`; errors involved `cow_read`, `cow_write`, and `m_slots`, and `make` exited with code `2`. `pkg-config pangolin` was also not found, although the ROS Humble Pangolin libraries were available through `/opt/ros/humble`.

## Remediation attempts

| attempt | attempted | success | log |
| --- | --- | --- | --- |
| cpp14_patch | True | True | `/home/dovetao/graduation_design_demo/logs/orbslam3_build_orb0b_cpp14.log` |
| cpp17_patch | False | False | not attempted because C++14 succeeded |
| isolated_pangolin | False | False | not attempted because C++14 succeeded |

## C++ standard used

The successful build used C++14. Local ORB-SLAM3 patch:

- Added `set(CMAKE_CXX_STANDARD 14)`.
- Added `set(CMAKE_CXX_STANDARD_REQUIRED ON)`.
- Added `set(CMAKE_CXX_EXTENSIONS OFF)`.
- Replaced upstream `-std=c++11` / `-std=c++0x` flags with `-std=c++14`.
- Created local backup `CMakeLists.txt.bak_orb0b_cpp_standard`.

## Pangolin source used

The successful build used Pangolin resolved from the ROS Humble environment with `CMAKE_PREFIX_PATH=/opt/ros/humble`. The isolated `/home/dovetao/third_party/pangolin_install` path was not built or used because the C++14 patch succeeded first.

## Whether ROS Humble sigslot is still present in build

`/opt/ros/humble/include/sigslot/signal.hpp` still appears in generated dependency files under the ORB-SLAM3 build tree. The blocker was remediated by compiling ORB-SLAM3 with C++14; no source-level Pangolin isolation was required.

## Generated library / executable inventory

- `lib/libORB_SLAM3.so`: `True`
- Library path: `/home/dovetao/third_party/ORB_SLAM3/lib/libORB_SLAM3.so`
- Monocular executables:
  - `/home/dovetao/third_party/ORB_SLAM3/Examples/Monocular/mono_euroc`
  - `/home/dovetao/third_party/ORB_SLAM3/Examples/Monocular/mono_kitti`
  - `/home/dovetao/third_party/ORB_SLAM3/Examples/Monocular/mono_realsense_D435i`
  - `/home/dovetao/third_party/ORB_SLAM3/Examples/Monocular/mono_realsense_t265`
  - `/home/dovetao/third_party/ORB_SLAM3/Examples/Monocular/mono_tum`
  - `/home/dovetao/third_party/ORB_SLAM3/Examples/Monocular/mono_tum_vi`

## ldd dependency summary

- Checked executable: `/home/dovetao/third_party/ORB_SLAM3/Examples/Monocular/mono_tum`
- ldd log: `/home/dovetao/graduation_design_demo/logs/orbslam3_mono_tum_ldd_orb0b.log`
- Missing dependencies: `[]`

## ready_for_orb1c

`True`. ORB1c may proceed as a separate step, but this report does not start ORB1c.

## Final classification

`ORB0_READY_AFTER_REMEDIATION`

Allowed classifications: `ORB0_READY_AFTER_REMEDIATION`, `ORB0B_CPP_STANDARD_PATCH_FAILED`, `ORB0B_ISOLATED_PANGOLIN_FAILED`, `ORB0B_ERROR`.

## Notes

- ORB-SLAM3 source remains outside the tracked project repository.
- The local patch lives only in `/home/dovetao/third_party/ORB_SLAM3`.
- S5 policy, locked metrics, final manifest, split, and official evaluator were not modified.
