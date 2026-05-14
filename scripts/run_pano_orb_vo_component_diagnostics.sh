#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

bash scripts/verify_final_candidate.sh
bash scripts/run_pano_orb_vo_baseline.sh
python3 tools/evaluate_pairwise_pose_components.py \
  --pred external_baselines/results/pano_orb_vo/scene01_seq03_est_tum.txt \
  --gt external_baselines/dataset/scene01_seq03/groundtruth_tum.txt \
  --pair-diagnostics external_baselines/results/pano_orb_vo/pair_diagnostics.json \
  --output-json checkpoints/SB1b_pano_orb_vo_component_diagnostics.json \
  --output-md reports/pano_orb_vo_component_diagnostics.md
echo "[pano-orb-vo-component-diagnostics] PASS"
