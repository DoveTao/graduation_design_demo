#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYBIN="${PYBIN:-/home/dovetao/miniconda3/envs/pytorch/bin/python}"
POLICY_PATH="${1:-checkpoints/S10_chain_level_path_ratio_preserving_smoother_policy.json}"

if [[ ! -f "$POLICY_PATH" ]]; then
  echo "[S10-eval] missing policy: $POLICY_PATH"
  exit 1
fi

echo "[S10-eval] policy=$POLICY_PATH"
"$PYBIN" tools/s10_chain_level_path_ratio_preserving_smoother.py --mode eval_policy --policy "$POLICY_PATH"
