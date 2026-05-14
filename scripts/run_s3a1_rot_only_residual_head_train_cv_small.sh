#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/dovetao/miniconda3/envs/pytorch/bin/python}"

"$PYTHON_BIN" -m py_compile config.py dataset_pano_only.py interaction.py model.py train_mvp.py losses.py tools/eval_clean_policy.py tools/s3a1_train_cv_small_run.py
"$PYTHON_BIN" tools/s3a1_train_cv_small_run.py
