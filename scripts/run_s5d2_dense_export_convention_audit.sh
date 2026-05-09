#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

bash scripts/verify_final_candidate.sh

/home/dovetao/miniconda3/envs/pytorch/bin/python tools/audit_s5_dense_export_convention.py \
  --s5-policy checkpoints/S5_clean_tmag_calibration_policy.json \
  --manifest checkpoints/final_clean_candidate_manifest.json \
  --gt external_baselines/dataset/scene01_seq03/groundtruth_tum.txt \
  --search-root external_baselines/results \
  --output-json checkpoints/S5D2_dense_export_convention_audit.json \
  --output-md reports/S5D2_dense_export_convention_audit.md

echo "[s5d2-dense-export-convention-audit] PASS"
