#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from s5e2_adjacent_dense_lib import write_json


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    train = _read_json(Path(args.candidate) / "training_status.json")
    prov = _read_jsonl(Path(args.results) / "edge_provenance.jsonl")
    metrics = _read_json(Path(args.results) / "edge_component_metrics.json")
    payload = {
        "experiment": "STRUCT1B_scale_guard_repair",
        "hard_scale_guard_enabled": bool(train.get("hard_scale_guard_enabled")),
        "recorded_raw_log_tmag": all("raw_log_tmag" in r for r in prov),
        "recorded_bounded_log_tmag": all("bounded_log_tmag" in r for r in prov),
        "recorded_final_tmag": all("final_tmag" in r for r in prov),
        "export_used_guarded_tmag": all("final_tmag" in r and "bounded_log_tmag" in r for r in prov),
        "uses_eval_gt_for_prediction": any(bool(r.get("uses_eval_gt_for_prediction")) for r in prov),
        "uses_orbslam3_teacher": any(bool(r.get("uses_orbslam3_teacher")) for r in prov),
        "coverage": metrics.get("coverage"),
        "integrity_pass": bool(
            train.get("real_training_executed")
            and train.get("hard_scale_guard_enabled")
            and all("raw_log_tmag" in r for r in prov)
            and metrics.get("coverage", {}).get("all_edges_traceable")
        ),
    }
    write_json(Path(args.out_json), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
