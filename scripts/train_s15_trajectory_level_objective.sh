#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/dovetao/miniconda3/envs/pytorch/bin/python}"
MODE="${MODE:-cv}"
SMOKE_STEPS="${SMOKE_STEPS:-20}"
CV_STEPS="${CV_STEPS:-100}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

export OMP_NUM_THREADS
export MKL_NUM_THREADS
export OPENBLAS_NUM_THREADS

"$PYTHON_BIN" -m py_compile config.py dataset_pano_only.py losses.py model.py train_mvp.py tools/s15_trajectory_level_training_objective.py
"$PYTHON_BIN" tools/s15_trajectory_level_training_objective.py --mode "$MODE" --smoke-steps "$SMOKE_STEPS" --cv-steps "$CV_STEPS"
