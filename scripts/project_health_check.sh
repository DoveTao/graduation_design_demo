#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  PYBIN="$PYTHON"
elif [[ -x "/home/dovetao/miniconda3/envs/pytorch/bin/python" ]]; then
  PYBIN="/home/dovetao/miniconda3/envs/pytorch/bin/python"
else
  PYBIN="python"
fi

echo "[project-health-check] python: $PYBIN"
"$PYBIN" --version

bash scripts/verify_final_candidate.sh
"$PYBIN" -m unittest discover -s tests -q

if "$PYBIN" -m pytest --version >/dev/null 2>&1; then
  echo "[project-health-check] pytest is available, optional command: pytest -q"
fi

echo "[project-health-check] PASS"
