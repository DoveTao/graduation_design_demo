#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYBIN="${PYBIN:-/home/dovetao/miniconda3/envs/pytorch/bin/python}"

bash scripts/verify_final_candidate.sh

if [[ ! -f "checkpoints/JRT1b_train_cv_results.json" ]]; then
  bash scripts/run_jrt1b_train_cv.sh
fi

"$PYBIN" tools/jrt1_eval_train_cv_trajectory_proxy.py \
  --config checkpoints/JRT1b_train_cv_config.json \
  --jrt1b-results checkpoints/JRT1b_train_cv_results.json \
  --output-root outputs/jrt1/train_cv_trajectory_proxy \
  --output-json checkpoints/JRT1c_train_cv_trajectory_proxy_results.json \
  --output-md reports/JRT1c_train_cv_trajectory_proxy_report.md

echo "[jrt1c-train-cv-trajectory-proxy] PASS"
