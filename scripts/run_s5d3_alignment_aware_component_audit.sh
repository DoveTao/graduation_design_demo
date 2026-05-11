#!/usr/bin/env bash
set -euo pipefail

PYBIN="/home/dovetao/miniconda3/envs/pytorch/bin/python"

"$PYBIN" tools/evaluate_alignment_aware_component_diagnostics.py \
  --groundtruth external_baselines/dataset/scene01_seq03/groundtruth_tum.txt \
  --s5-trajectory external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt \
  --orb-trajectory external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt \
  --timestamps external_baselines/dataset/scene01_seq03/timestamps.txt \
  --out-json checkpoints/S5D3_validation_and_alignment_aware_component_audit.json \
  --out-report reports/s5d3_validation_and_alignment_aware_component_audit.md \
  --out-dir external_baselines/results/component_diagnostics_s5d3 \
  --timestamp-tolerance 1e-6 \
  --thresholds 1e-8,1e-6,1e-5,1e-4,1e-3,1e-2 \
  --relative-thresholds 0.01,0.05,0.10
