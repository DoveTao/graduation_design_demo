#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYBIN_TORCH="/home/dovetao/miniconda3/envs/pytorch/bin/python"

bash scripts/verify_final_candidate.sh
bash scripts/run_pano_orb_vo_baseline.sh
"$PYBIN_TORCH" tools/export_s5_trajectory_for_external_eval.py \
  --output-json checkpoints/SB2_alignment_consistent_baseline_results.json \
  --output-md reports/alignment_consistent_strong_baseline_comparison.md
echo "[alignment-consistent-baseline-comparison] PASS"
