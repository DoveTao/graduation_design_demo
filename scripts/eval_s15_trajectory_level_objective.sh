#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REPORT_PATH="checkpoints/S15_trajectory_level_training_objective_report.md"
CANDIDATES_PATH="checkpoints/S15_trajectory_level_training_objective_candidates.json"

echo "[S15-eval] report=$REPORT_PATH"
echo "[S15-eval] candidates=$CANDIDATES_PATH"

if [[ -f "$REPORT_PATH" && -f "$CANDIDATES_PATH" ]]; then
  sed -n '1,220p' "$REPORT_PATH"
else
  echo "[S15-eval] no S15 report found yet"
  exit 1
fi
