# External Baseline Environment Audit

## Scope
This audit checks whether the current machine can support third-party external SLAM/VO baselines without modifying the locked S5 policy, metrics, split, or eval convention.

## System Summary
- OS: `Ubuntu 22.04.5 LTS (jammy)`
- compiler: `gcc/g++ 11.4.0`
- build tools: `cmake 3.22.1`, `GNU make 4.3`
- Python:
  - `python` on PATH: unavailable
  - project env: `/home/dovetao/miniconda3/envs/pytorch/bin/python`
  - project env version: `Python 3.12.12`
- GPU:
  - `nvidia-smi`: available
  - GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`
  - driver / CUDA runtime: `580.142 / 13.0`
  - VRAM: `6144 MiB`
- CUDA toolkit:
  - `nvcc`: unavailable

## Library Audit
- OpenCV:
  - `pkg-config opencv4`: `4.5.4`
  - Debian packages: present
- Eigen:
  - `pkg-config eigen3`: `3.4.0`
  - Debian package: `libeigen3-dev` present
- Pangolin:
  - Debian/ROS package present: `ros-humble-pangolin`
  - `pkg-config pangolin`: unavailable on default search path
  - implication: CMake may need explicit `CMAKE_PREFIX_PATH=/opt/ros/humble`
- Additional C++ deps:
  - Boost dev packages: present
  - GLEW dev packages: present
  - SuiteSparse dev packages: present

## Dataset Runability Audit
- exported sequence: `scene01/seq03`
- image list exists: yes
- image paths accessible: yes
- timestamps / GT rows: `454 / 454`
- input modality: `monocular panorama / equirectangular RGB`
- fair camera intrinsics for ORB-SLAM3 monocular: unavailable

## Build Readiness
- C++ build environment: available
- OpenCV: available
- Eigen: available
- Pangolin: partially available
  - headers/libs appear installed under `/opt/ros/humble`
  - additional CMake path setup is likely required
- CUDA/GPU:
  - runtime GPU is available
  - toolkit compiler `nvcc` is not installed
  - current VRAM is only `6 GB`

## Recommendation
- Recommended first baseline: `ORB-SLAM3 monocular`
  - reason: C++ stack is mostly present, while DROID-SLAM has heavier GPU and environment requirements
  - caveat: the current exported data is equirectangular panorama, not a fair native pinhole/fisheye ORB-SLAM3 input
- DROID-SLAM recommendation: `skip for now`
  - reason: `6 GB` VRAM and missing `nvcc` are below a comfortable setup/run margin for a separate DROID-SLAM environment

## Conclusion
- The machine is close to being ORB-SLAM3-build-capable from a system dependency perspective.
- The more important blocker is modality compatibility: the current fair export is panorama/equirectangular and does not provide trustworthy pinhole/fisheye calibration for a direct ORB-SLAM3 monocular run.
- Therefore the next fair step is not to fabricate camera parameters, but to either:
  - add a documented perspective/fisheye projection export with calibration, or
  - use an external baseline that truly supports panorama/equirectangular inputs.
