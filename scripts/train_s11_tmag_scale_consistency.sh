#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/dovetao/miniconda3/envs/pytorch/bin/python}"
MODE="${MODE:-smoke}"
MAX_STEPS="${MAX_STEPS:-20}"

"$PYTHON_BIN" -m py_compile config.py losses.py interaction.py model.py train_mvp.py tools/s11_tmag_scale_consistency_training.py
"$PYTHON_BIN" tools/s11_tmag_scale_consistency_training.py --mode "$MODE" --max-steps "$MAX_STEPS"
