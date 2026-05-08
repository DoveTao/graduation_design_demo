#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dovetao/miniconda3/envs/pytorch/bin/python}"
CONFIG_PATH="${1:-$ROOT/configs/s19_geometry_aware_pretraining_feasibility.yaml}"

cd "$ROOT"
"$PYTHON_BIN" tools/s19_geometry_aware_pretraining_feasibility.py --config "$CONFIG_PATH"
