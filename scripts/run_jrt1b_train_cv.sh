#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYBIN="${PYBIN:-/home/dovetao/miniconda3/envs/pytorch/bin/python}"

bash scripts/verify_final_candidate.sh

"$PYBIN" tools/jrt1_train_cv_joint_rtdir_refiner.py \
  --config checkpoints/JRT1b_train_cv_config.json \
  --output-root outputs/jrt1/train_cv \
  --output-json checkpoints/JRT1b_train_cv_results.json

"$PYBIN" tools/jrt1_eval_train_cv_joint_rtdir_refiner.py \
  --config checkpoints/JRT1b_train_cv_config.json \
  --results checkpoints/JRT1b_train_cv_results.json \
  --output-md reports/JRT1b_train_cv_report.md

echo "[jrt1b-train-cv] PASS"
