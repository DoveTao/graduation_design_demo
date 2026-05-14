#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

bash scripts/verify_final_candidate.sh

/home/dovetao/miniconda3/envs/pytorch/bin/python tools/materialize_curated_reports.py \
  --plan-json checkpoints/REPORT3_curated_report_archive_plan.json \
  --output-root reports_curated

echo "[curated-reports-materialization] PASS"
