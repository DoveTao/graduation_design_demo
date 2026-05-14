#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnostic-only S5 dense fill instrumentation stub. It is off by default and never changes S5 policy or evaluator behavior."
    )
    parser.add_argument("--enable-diagnostic-output", action="store_true")
    parser.add_argument("--out-json", default="external_baselines/results/s5d12_dense_fill_path_audit/instrumentation_plan.json")
    args = parser.parse_args()

    payload: Dict[str, Any] = {
        "instrumentation": "s5_dense_export_fill_path",
        "diagnostic_only": True,
        "default_enabled": False,
        "enabled_this_run": bool(args.enable_diagnostic_output),
        "does_not_modify_policy": True,
        "does_not_modify_manifest": True,
        "does_not_modify_train_test_split": True,
        "does_not_modify_official_evaluator": True,
        "suggested_fields": [
            "edge_index",
            "source_type",
            "source_pair_or_component",
            "gap_start_edge",
            "gap_end_edge",
            "gap_num_adjacent_steps",
            "endpoint_delta_norm",
            "emitted_step_delta_norm",
            "dt_seconds",
            "dt_raw_units",
        ],
    }
    if args.enable_diagnostic_output:
        out = _resolve(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
