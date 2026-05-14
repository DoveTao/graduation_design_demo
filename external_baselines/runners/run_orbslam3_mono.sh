#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="external_baselines/config/orbslam3_scene01_seq03.yaml"
DATASET_DIR="external_baselines/dataset/scene01_seq03"
RESULT_DIR="external_baselines/results/orbslam3"
TRAJ_OUT="$RESULT_DIR/scene01_seq03_est_tum.txt"
ORB_ROOT="external_baselines/third_party/ORB_SLAM3"
VOCAB="$ORB_ROOT/Vocabulary/ORBvoc.txt"
BIN="$ORB_ROOT/Examples/Monocular/mono_tum"

mkdir -p "$RESULT_DIR"

if [[ ! -f "$CONFIG" ]]; then
  echo "[orbslam3-runner] missing config: $CONFIG"
  exit 1
fi
if [[ ! -f "$DATASET_DIR/image_list.txt" ]]; then
  echo "[orbslam3-runner] missing image list: $DATASET_DIR/image_list.txt"
  exit 1
fi
if [[ ! -f "$DATASET_DIR/groundtruth_tum.txt" ]]; then
  echo "[orbslam3-runner] missing GT: $DATASET_DIR/groundtruth_tum.txt"
  exit 1
fi

if grep -q '^orbslam3_ready: false' "$CONFIG"; then
  echo "[orbslam3-runner] config marks this dataset as not directly runnable by ORB-SLAM3."
  echo "[orbslam3-runner] reason: panorama/equirectangular export lacks fair pinhole/fisheye calibration."
  bash scripts/run_external_baseline_comparison.sh
  exit 0
fi

if [[ ! -x "$BIN" ]]; then
  echo "[orbslam3-runner] missing ORB-SLAM3 binary: $BIN"
  bash scripts/run_external_baseline_comparison.sh
  exit 0
fi
if [[ ! -f "$VOCAB" ]]; then
  echo "[orbslam3-runner] missing ORB vocabulary: $VOCAB"
  bash scripts/run_external_baseline_comparison.sh
  exit 0
fi

echo "[orbslam3-runner] TODO: wire ORB-SLAM3 monocular execution and TUM export when a fair camera model is available."
echo "[orbslam3-runner] expected output path: $TRAJ_OUT"
bash scripts/run_external_baseline_comparison.sh
