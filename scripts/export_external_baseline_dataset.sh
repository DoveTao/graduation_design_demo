#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  PYBIN="$PYTHON"
elif [[ -x /home/dovetao/miniconda3/envs/pytorch/bin/python ]]; then
  PYBIN="/home/dovetao/miniconda3/envs/pytorch/bin/python"
else
  PYBIN="python"
fi

"$PYBIN" tools/export_external_baseline_dataset.py "$@"
