#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if command -v python3 >/dev/null 2>&1; then
  PYBIN="python3"
elif [[ -n "${PYTHON:-}" ]]; then
  PYBIN="$PYTHON"
else
  PYBIN="/home/dovetao/miniconda3/envs/pytorch/bin/python"
fi

bash scripts/verify_final_candidate.sh
"$PYBIN" tools/run_pano_orb_vo_baseline.py \
  --dataset external_baselines/dataset/scene01_seq03 \
  --output-dir external_baselines/results/pano_orb_vo \
  --output-json checkpoints/SB1_pano_orb_vo_baseline_results.json \
  --output-md reports/pano_orb_vo_baseline_report.md \
  --fov-deg 90 \
  --view-yaws 0,90,180,270 \
  --view-width 640 \
  --view-height 480 \
  "$@"
echo "[pano-orb-vo-baseline] PASS"
