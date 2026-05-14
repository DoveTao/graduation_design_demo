#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYBIN="${PYBIN:-/home/dovetao/miniconda3/envs/pytorch/bin/python}"
POLICY_PATH="${1:-checkpoints/S9_regime_only_reliability_router_policy.json}"

if [[ ! -f "$POLICY_PATH" ]]; then
  echo "[S9-eval] missing policy: $POLICY_PATH"
  exit 1
fi

echo "[S9-eval] policy=$POLICY_PATH"
echo "[S9-eval] base_policy=$(python3 - "$POLICY_PATH" <<'PY'
import json
import sys
from pathlib import Path
p=Path(sys.argv[1])
obj=json.loads(p.read_text())
print(obj.get("base_policy_path",""))
PY
)"
"$PYBIN" tools/s9_regime_only_reliability_router.py --mode eval_policy --policy "$POLICY_PATH"
