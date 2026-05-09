#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYBIN="${PYBIN:-/home/dovetao/miniconda3/envs/pytorch/bin/python}"

bash scripts/verify_final_candidate.sh

"$PYBIN" tools/jrt1_build_pair_dataset.py \
  --config checkpoints/JRT1_joint_rtdir_refiner_config.json \
  --output outputs/jrt1/JRT1a_pair_dataset_smoke.npz \
  --summary outputs/jrt1/JRT1a_pair_dataset_summary.json

"$PYBIN" tools/jrt1_train_joint_rtdir_refiner.py \
  --config checkpoints/JRT1_joint_rtdir_refiner_config.json \
  --dataset outputs/jrt1/JRT1a_pair_dataset_smoke.npz \
  --output-dir outputs/jrt1/smoke \
  --train-log outputs/jrt1/JRT1a_smoke_train_log.json

"$PYBIN" tools/jrt1_eval_joint_rtdir_refiner.py \
  --config checkpoints/JRT1_joint_rtdir_refiner_config.json \
  --dataset outputs/jrt1/JRT1a_pair_dataset_smoke.npz \
  --train-log outputs/jrt1/JRT1a_smoke_train_log.json \
  --output-json checkpoints/JRT1a_framework_smoke_results.json \
  --output-md reports/JRT1a_framework_smoke_report.md

echo "[jrt1a-framework-smoke] PASS"
