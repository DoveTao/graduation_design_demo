#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYBIN_TORCH="/home/dovetao/miniconda3/envs/pytorch/bin/python"

bash scripts/verify_final_candidate.sh
bash scripts/run_alignment_consistent_baseline_comparison.sh
"$PYBIN_TORCH" tools/audit_s5_external_export_compatibility.py \
  --s5-trajectory external_baselines/results/s5/scene01_seq03_est_tum.txt \
  --gt external_baselines/dataset/scene01_seq03/groundtruth_tum.txt \
  --sb2-json checkpoints/SB2_alignment_consistent_baseline_results.json \
  --output-json checkpoints/SB2b_s5_export_compatibility_audit.json \
  --output-md reports/s5_external_export_compatibility_audit.md
echo "[s5-export-compatibility-audit] PASS"
