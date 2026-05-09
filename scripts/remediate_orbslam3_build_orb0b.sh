#!/usr/bin/env bash
set -euo pipefail

# This remediation step builds ORB-SLAM3 only.
# It does not run ORB-SLAM3 on the dataset.
# It does not generate a trajectory.
# It does not produce baseline metrics.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORB_SLAM3_DIR="/home/dovetao/third_party/ORB_SLAM3"
CPP14_LOG="${REPO_ROOT}/logs/orbslam3_build_orb0b_cpp14.log"
CPP17_LOG="${REPO_ROOT}/logs/orbslam3_build_orb0b_cpp17.log"
PANGOLIN_LOG="${REPO_ROOT}/logs/orbslam3_build_orb0b_isolated_pangolin.log"

mkdir -p "${REPO_ROOT}/logs"
cd "${ORB_SLAM3_DIR}"

if [ ! -f CMakeLists.txt.bak_orb0b_cpp_standard ]; then
  cp CMakeLists.txt CMakeLists.txt.bak_orb0b_cpp_standard
fi

perl -0pi -e 's/-std=c\+\+11/-std=c++14/g; s/-std=c\+\+0x/-std=c++14/g; if (!/CMAKE_CXX_STANDARD/) { s/project\(ORB_SLAM3\)/project(ORB_SLAM3)\n\nset(CMAKE_CXX_STANDARD 14)\nset(CMAKE_CXX_STANDARD_REQUIRED ON)\nset(CMAKE_CXX_EXTENSIONS OFF)/ }' CMakeLists.txt
rm -rf build Thirdparty/DBoW2/build Thirdparty/g2o/build
if CMAKE_PREFIX_PATH=/opt/ros/humble ./build.sh 2>&1 | tee "${CPP14_LOG}"; then
  exit 0
fi

perl -0pi -e 's/-std=c\+\+14/-std=c++17/g; s/set\(CMAKE_CXX_STANDARD 14\)/set(CMAKE_CXX_STANDARD 17)/g' CMakeLists.txt
rm -rf build Thirdparty/DBoW2/build Thirdparty/g2o/build
if CMAKE_PREFIX_PATH=/opt/ros/humble ./build.sh 2>&1 | tee "${CPP17_LOG}"; then
  exit 0
fi

mkdir -p /home/dovetao/third_party
cd /home/dovetao/third_party
if [ ! -d Pangolin ]; then
  git clone https://github.com/stevenlovegrove/Pangolin.git
fi
cd Pangolin
git checkout v0.6 2>/dev/null || true
rm -rf build
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/dovetao/third_party/pangolin_install
make -j4
make install

cd "${ORB_SLAM3_DIR}"
rm -rf build Thirdparty/DBoW2/build Thirdparty/g2o/build
env -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH CMAKE_PREFIX_PATH=/home/dovetao/third_party/pangolin_install ./build.sh 2>&1 | tee "${PANGOLIN_LOG}"
