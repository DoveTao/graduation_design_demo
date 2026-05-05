#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
POLICY="checkpoints/S2b_clean_fine_rot_policy.json"
OUTDIR="checkpoints/S2b_final_repro"
CMD=(/home/dovetao/miniconda3/envs/pytorch/bin/python tools/eval_clean_policy.py --policy "$POLICY" --output-dir "$OUTDIR" --eval-variant default)
echo "Policy: $POLICY"
echo "Output: $OUTDIR"
echo "Command: ${CMD[*]}"
if [[ "${1:-}" == "dryrun" ]]; then
  echo "[DRYRUN] not executing"
  exit 0
fi
"${CMD[@]}"
