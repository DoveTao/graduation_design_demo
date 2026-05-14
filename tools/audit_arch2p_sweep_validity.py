#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from s5e2_adjacent_dense_lib import write_json


def run(args: argparse.Namespace) -> Dict[str, Any]:
    results_dir = Path(args.results_dir)
    runs: List[Dict[str, Any]] = []
    invalid_best: List[str] = []
    for path in sorted(results_dir.glob("run_*_summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        real = bool(row.get("real_training_executed"))
        no_op = bool(row.get("no_op_risk"))
        fallback = bool(row.get("fallback_risk"))
        old_metrics_reuse = False
        eval_gt = bool(row.get("uses_eval_gt_for_training"))
        orb = bool(row.get("uses_orbslam3_teacher"))
        valid = real and not no_op and not fallback and not old_metrics_reuse and not eval_gt and not orb
        if not valid:
            invalid_best.append(str(row.get("run_id")))
        runs.append(
            {
                "run_id": row.get("run_id"),
                "real_training_executed": real,
                "no_op_risk": no_op,
                "fallback_risk": fallback,
                "old_metrics_reuse_risk": old_metrics_reuse,
                "uses_eval_gt_for_training": eval_gt,
                "uses_orbslam3_teacher": orb,
                "valid": valid,
            }
        )
    payload = {
        "experiment": "ARCH2P_parameter_sensitivity_sweep",
        "runs": runs,
        "valid_run_count": sum(1 for r in runs if r["valid"]),
        "invalid_run_ids": invalid_best,
        "final_classification": "ARCH2P_SWEEP_INVALID" if not runs or all(not r["valid"] for r in runs) else "ARCH2P_VALIDITY_OK",
    }
    write_json(Path(args.out_json), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
