#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYBIN="${PYBIN:-/home/dovetao/miniconda3/envs/pytorch/bin/python}"

echo "[S8-diagnostic] running staged probe mode"
"$PYBIN" tools/s8_fine_spherical_reliability_router.py \
  --resume \
  --stage probe \
  --skip-final-test \
  --no-odometry-route
