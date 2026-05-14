#!/usr/bin/env bash
set -euo pipefail
/home/dovetao/miniconda3/envs/pytorch/bin/python tools/audit_tdir_metric_provenance.py \
  --groundtruth external_baselines/dataset/scene01_seq03/groundtruth_tum.txt \
  --s5-trajectory external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt \
  --timestamps external_baselines/dataset/scene01_seq03/timestamps.txt \
  --s5d2-checkpoint checkpoints/S5D2_component_diagnostics.json \
  --s5d3-checkpoint checkpoints/S5D3_validation_and_alignment_aware_component_audit.json \
  --out-json checkpoints/S5D4_reconcile_old_tdir_with_dense_tdir.json \
  --out-report reports/s5d4_reconcile_old_tdir_with_dense_tdir.md \
  --out-dir external_baselines/results/tdir_reconciliation
