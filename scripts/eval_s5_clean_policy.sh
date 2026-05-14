#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYBIN="${PYBIN:-/home/dovetao/miniconda3/envs/pytorch/bin/python}"
POLICY_PATH="checkpoints/S5_clean_tmag_calibration_policy.json"

if [[ ! -f "$POLICY_PATH" ]]; then
  echo "[S6-eval] missing policy: $POLICY_PATH"
  echo "[S6-eval] generating policy + audit artifacts first..."
  "$PYBIN" tools/s6_final_clean_candidate_lockdown_audit.py
fi

echo "[S6-eval] policy=$POLICY_PATH"
echo "[S6-eval] running fixed selected-candidate evaluation"
"$PYBIN" tools/s6_final_clean_candidate_lockdown_audit.py --eval-only
