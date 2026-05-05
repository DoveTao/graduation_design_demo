#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-dryrun}"
OUTDIR="${2:-checkpoints/S1d5_clean_policy_eval}"
VARIANT="${3:-default}"
PYBIN="/home/dovetao/miniconda3/envs/pytorch/bin/python"
POLICY="checkpoints/S1d5_clean_dt_anchor_policy.json"

CMD=(
  "$PYBIN" tools/eval_clean_policy.py
  --policy "$POLICY"
  --output-dir "$OUTDIR"
  --eval-variant "$VARIANT"
)

echo "Policy: $POLICY"
echo "Output: $OUTDIR"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "$MODE" == "run" ]]; then
  "${CMD[@]}"
else
  echo "[DRYRUN] not executing"
fi
