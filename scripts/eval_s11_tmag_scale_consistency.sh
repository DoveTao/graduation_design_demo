#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REPORT_PATH="checkpoints/S11_tmag_scale_consistency_report.md"
CANDIDATES_PATH="checkpoints/S11_tmag_scale_consistency_candidates.json"

echo "[S11-eval] report=$REPORT_PATH"
echo "[S11-eval] candidates=$CANDIDATES_PATH"

if [[ -f "$CANDIDATES_PATH" ]]; then
  sed -n '1,120p' "$REPORT_PATH"
else
  echo "[S11-eval] no S11 candidate report found yet"
  exit 1
fi
