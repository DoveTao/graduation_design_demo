#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REPORT_PATH="checkpoints/S14_local_window_pose_graph_optimization_report.md"
CANDIDATES_PATH="checkpoints/S14_local_window_pose_graph_optimization_candidates.json"
POLICY_PATH="checkpoints/S14_local_window_pose_graph_optimization_policy.json"

echo "[S14-eval] report=$REPORT_PATH"
echo "[S14-eval] candidates=$CANDIDATES_PATH"
echo "[S14-eval] policy=$POLICY_PATH"

if [[ -f "$REPORT_PATH" ]]; then
  sed -n '1,200p' "$REPORT_PATH"
else
  echo "[S14-eval] no S14 report found yet"
  exit 1
fi
