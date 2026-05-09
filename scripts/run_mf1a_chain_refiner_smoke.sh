#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYBIN="${PYBIN:-/home/dovetao/miniconda3/envs/pytorch/bin/python}"

bash scripts/verify_final_candidate.sh

"$PYBIN" tools/mf1_build_chain_dataset.py \
  --config checkpoints/MF1_chain_refiner_config.json \
  --output outputs/mf1/MF1a_chain_dataset_smoke.npz \
  --summary outputs/mf1/MF1a_chain_dataset_summary.json

"$PYBIN" tools/mf1_train_chain_refiner.py \
  --config checkpoints/MF1_chain_refiner_config.json \
  --dataset outputs/mf1/MF1a_chain_dataset_smoke.npz \
  --output-dir outputs/mf1/smoke \
  --train-log outputs/mf1/MF1a_chain_refiner_train_log.json

"$PYBIN" tools/mf1_eval_chain_refiner.py \
  --config checkpoints/MF1_chain_refiner_config.json \
  --dataset outputs/mf1/MF1a_chain_dataset_smoke.npz \
  --train-log outputs/mf1/MF1a_chain_refiner_train_log.json \
  --output-json checkpoints/MF1a_chain_refiner_smoke_results.json \
  --output-md reports/MF1a_chain_refiner_smoke_report.md

echo "[mf1a-chain-refiner-smoke] PASS"
