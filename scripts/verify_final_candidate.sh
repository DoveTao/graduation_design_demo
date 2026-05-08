#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYBIN="${PYBIN:-/home/dovetao/miniconda3/envs/pytorch/bin/python}"

"$PYBIN" tools/verify_final_s5_candidate.py \
  --policy checkpoints/S5_clean_tmag_calibration_policy.json \
  --manifest checkpoints/final_clean_candidate_manifest.json \
  --expected-ate 7.352288 \
  --expected-drift 1.327343 \
  --expected-path-ratio 0.932379
