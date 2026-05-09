#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORB_SLAM3_ROOT=/home/dovetao/third_party/ORB_SLAM3
VOCAB="${ORB_SLAM3_ROOT}/Vocabulary/ORBvoc.txt"
YAML="${REPO_ROOT}/external_baselines/config/orbslam3_fisheye_cam0_scene01_seq03.yaml"
OUT_DIR="${REPO_ROOT}/external_baselines/results/orbslam3_fisheye_cam0"
IMAGE_LIST="${OUT_DIR}/scene01_seq03_images.txt"
TIMESTAMPS="${OUT_DIR}/scene01_seq03_timestamps.txt"
RUNTIME_YAML="${OUT_DIR}/scene01_seq03_orbslam3_runtime.yaml"
EST_TUM="${OUT_DIR}/scene01_seq03_est_tum.txt"
STDOUT_LOG="${OUT_DIR}/orbslam3_stdout.log"
STDERR_LOG="${OUT_DIR}/orbslam3_stderr.log"
RUNNER_SRC="${REPO_ROOT}/external_baselines/runners/orbslam3_mono_fisheye_runner.cc"
RUNNER_BIN="${REPO_ROOT}/external_baselines/runners/orbslam3_mono_fisheye_runner"

# This step runs ORB-SLAM3 and exports trajectory only.
# It does not evaluate ATE/drift/path_ratio.
# It does not modify S5 locked metrics or policy.
# It does not call any trajectory evaluator.

mkdir -p "${OUT_DIR}"
if [ ! -f "${VOCAB}" ] && [ -f "${ORB_SLAM3_ROOT}/Vocabulary/ORBvoc.txt.tar.gz" ]; then
  tar -xf "${ORB_SLAM3_ROOT}/Vocabulary/ORBvoc.txt.tar.gz" -C "${ORB_SLAM3_ROOT}/Vocabulary"
fi

/home/dovetao/miniconda3/envs/pytorch/bin/python "${REPO_ROOT}/tools/export_orbslam3_fisheye_sequence.py" \
  --raw-root "${REPO_ROOT}/data/FisheyeView" \
  --scene scene01 \
  --seq seq03 \
  --settings-yaml "${YAML}" \
  --out-dir "${OUT_DIR}" > "${OUT_DIR}/sequence_export_stdout.json"

g++ -std=c++14 -DGL_GLEXT_PROTOTYPES "${RUNNER_SRC}" -o "${RUNNER_BIN}" \
  -I"${ORB_SLAM3_ROOT}/include" \
  -I"${ORB_SLAM3_ROOT}/include/CameraModels" \
  -I"${ORB_SLAM3_ROOT}" \
  -I"${ORB_SLAM3_ROOT}/Thirdparty/Sophus" \
  -I/opt/ros/humble/include \
  -I/usr/include/eigen3 \
  $(pkg-config --cflags opencv4) \
  -L"${ORB_SLAM3_ROOT}/lib" \
  -L"${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib" \
  -L"${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib" \
  -L/opt/ros/humble/lib \
  -Wl,-rpath,"${ORB_SLAM3_ROOT}/lib" \
  -Wl,-rpath,"${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib" \
  -Wl,-rpath,"${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib" \
  -Wl,-rpath,/opt/ros/humble/lib \
  -lORB_SLAM3 -lDBoW2 -lg2o -lboost_serialization -lcrypto \
  -lpango_display -lpango_opengl -lpango_vars -lpango_core -lGLEW -lGL \
  $(pkg-config --libs opencv4)

"${RUNNER_BIN}" "${VOCAB}" "${RUNTIME_YAML}" "${IMAGE_LIST}" "${TIMESTAMPS}" "${EST_TUM}" \
  > "${STDOUT_LOG}" 2> "${STDERR_LOG}"
