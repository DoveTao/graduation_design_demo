#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_ROOT="${ORB_SLAM3_THIRD_PARTY_ROOT:-/home/dovetao/third_party}"
TARGET_PATH="${TARGET_ROOT}/ORB_SLAM3"
LOG_PATH="${REPO_ROOT}/logs/orbslam3_build.log"

mkdir -p "${TARGET_ROOT}" "${REPO_ROOT}/logs"

if [ ! -d "${TARGET_PATH}" ]; then
  git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git "${TARGET_PATH}"
fi

cd "${TARGET_PATH}"
chmod +x build.sh
CMAKE_PREFIX_PATH=/opt/ros/humble ./build.sh 2>&1 | tee "${LOG_PATH}"
