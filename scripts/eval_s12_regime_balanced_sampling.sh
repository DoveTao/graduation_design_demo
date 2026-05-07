#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REPORT_PATH="checkpoints/S12_regime_balanced_sampling_report.md"
CANDIDATES_PATH="checkpoints/S12_regime_balanced_sampling_candidates.json"

echo "[S12-eval] report=$REPORT_PATH"
echo "[S12-eval] candidates=$CANDIDATES_PATH"

if [[ -f "$REPORT_PATH" && -f "$CANDIDATES_PATH" ]]; then
  sed -n '1,160p' "$REPORT_PATH"
else
  echo "[S12-eval] no S12 report found yet"
  exit 1
fi
