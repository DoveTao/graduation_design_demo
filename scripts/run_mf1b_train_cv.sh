#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYBIN="${PYBIN:-/home/dovetao/miniconda3/envs/pytorch/bin/python}"

bash scripts/verify_final_candidate.sh

"$PYBIN" tools/mf1_build_chain_dataset.py \
  --config checkpoints/MF1b_train_cv_config.json \
  --output outputs/mf1/train_cv/MF1b_chain_dataset.npz \
  --summary outputs/mf1/train_cv/MF1b_chain_dataset_summary.json

"$PYBIN" tools/mf1_train_cv_chain_refiner.py \
  --config checkpoints/MF1b_train_cv_config.json \
  --dataset outputs/mf1/train_cv/MF1b_chain_dataset.npz \
  --output-root outputs/mf1/train_cv \
  --output-json checkpoints/MF1b_train_cv_results.json

"$PYBIN" tools/mf1_eval_train_cv_chain_refiner.py \
  --config checkpoints/MF1b_train_cv_config.json \
  --results checkpoints/MF1b_train_cv_results.json \
  --output-md reports/MF1b_train_cv_report.md

echo "[mf1b-train-cv] PASS"
