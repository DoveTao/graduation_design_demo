#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, write_json


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick(ck: Dict[str, Any], name: str) -> Dict[str, Any]:
    comp = ck.get("component_metrics", {})
    ext = ck.get("external_eval", {})
    return {
        "method": name,
        "status": ck.get("final_classification"),
        "coverage": ck.get("adjacent_dense_export", {}).get("coverage", 1.0 if "coverage" not in ck else ck["coverage"]),
        "signed_tdir_mean_deg": comp.get("signed_tdir_mean_deg", comp.get("tdir_mean_deg")),
        "tdir_abs_mean_deg": comp.get("tdir_abs_mean_deg"),
        "anti_parallel_rate": comp.get("anti_parallel_rate"),
        "tmag_median_ratio": comp.get("tmag_median_ratio"),
        "tmag_p95_ratio": comp.get("tmag_p95_ratio"),
        "path_ratio": comp.get("path_ratio"),
        "none_ate": ext.get("none", {}).get("ate"),
        "se3_ate": ext.get("se3", {}).get("ate"),
        "sim3_ate": ext.get("sim3", {}).get("ate"),
        "caveat": ck.get("final_classification"),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    s19c = _load(Path(args.s5e19c_checkpoint))
    s19 = _load(Path(args.s5e19_checkpoint))
    s15 = _load(Path(args.s5e15_checkpoint))
    rows: List[Dict[str, Any]] = [
        _pick(s15, "S5E15"),
        _pick(s19, "S5E19"),
        _pick(s19c, "S5E19C"),
        {
            "method": "ORB-SLAM3",
            "status": "external_baseline",
            "coverage": ORBSLAM3_REFERENCE["tracking_success_rate"],
            "signed_tdir_mean_deg": None,
            "tdir_abs_mean_deg": None,
            "anti_parallel_rate": None,
            "tmag_median_ratio": None,
            "tmag_p95_ratio": None,
            "path_ratio": ORBSLAM3_REFERENCE["se3"]["path_ratio"],
            "none_ate": ORBSLAM3_REFERENCE["none"]["ate"],
            "se3_ate": ORBSLAM3_REFERENCE["se3"]["ate"],
            "sim3_ate": ORBSLAM3_REFERENCE["sim3"]["ate"],
            "caveat": "external_baseline",
        },
    ]
    payload = {
        "experiment": "S5E19C_real_direction_training_no_fallback",
        "rows": rows,
        "summary": "S5E19C 的目标不是再做 smoke/no-op，而是证明 real training + no-fallback export 是否真的改变了 direction path。",
    }
    write_json(Path(args.out_json), payload)
    Path(args.out_report).write_text("# S5E19C comparison\n\n## 执行摘要\n" + payload["summary"] + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s5e19c-checkpoint", required=True)
    p.add_argument("--s5e19-checkpoint", required=True)
    p.add_argument("--s5e15-checkpoint", required=True)
    p.add_argument("--orbslam3-eval-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
