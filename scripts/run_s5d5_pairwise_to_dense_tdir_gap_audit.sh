#!/usr/bin/env bash
set -euo pipefail
/home/dovetao/miniconda3/envs/pytorch/bin/python tools/audit_pairwise_to_dense_tdir_gap.py \
  --s5-dense-trajectory external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt \
  --groundtruth external_baselines/dataset/scene01_seq03/groundtruth_tum.txt \
  --timestamps external_baselines/dataset/scene01_seq03/timestamps.txt \
  --s5d2-checkpoint checkpoints/S5D2_component_diagnostics.json \
  --s5d3-checkpoint checkpoints/S5D3_validation_and_alignment_aware_component_audit.json \
  --s5d4-checkpoint checkpoints/S5D4_reconcile_old_tdir_with_dense_tdir.json \
  --out-json checkpoints/S5D5_pairwise_to_dense_tdir_gap_audit.json \
  --out-report reports/s5d5_pairwise_to_dense_tdir_gap_audit.md \
  --out-dir external_baselines/results/pairwise_to_dense_tdir_gap
