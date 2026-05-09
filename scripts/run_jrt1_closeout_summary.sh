#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYBIN="${PYBIN:-/home/dovetao/miniconda3/envs/pytorch/bin/python}"

bash scripts/verify_final_candidate.sh

"$PYBIN" - <<'PY'
import json
from pathlib import Path

repo = Path.cwd()
report = repo / "reports" / "JRT1_final_closeout_summary.md"
summary = repo / "checkpoints" / "JRT1_final_closeout_summary.json"
if not report.is_file():
    raise SystemExit(f"missing report: {report}")
if not summary.is_file():
    raise SystemExit(f"missing json: {summary}")
payload = json.loads(summary.read_text(encoding="utf-8"))
expected = {"ATE": 7.352288, "drift": 1.327343, "path_ratio": 0.932379}
for key, val in expected.items():
    if abs(float(payload["s5_locked_metrics"][key]) - val) > 1.0e-9:
        raise SystemExit(f"S5 locked metric changed: {key}")
text = report.read_text(encoding="utf-8")
required = [
    "NO_STABLE_JRT1_TRAJECTORY_GAIN",
    "no final test",
    "S5 remains final clean candidate",
]
for item in required:
    if item not in text:
        raise SystemExit(f"missing closeout text: {item}")
if payload.get("selected_candidate") is not None:
    raise SystemExit("JRT1 closeout must not select a candidate")
if not payload.get("no_final_test", False):
    raise SystemExit("JRT1 closeout must record no_final_test=true")
PY

echo "[jrt1-closeout-summary] PASS"
