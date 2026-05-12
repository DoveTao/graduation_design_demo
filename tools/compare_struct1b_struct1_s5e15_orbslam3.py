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
    if "overall" in comp:
        comp = comp["overall"]
    ext = ck.get("external_eval", {})
    return {
        "method": name,
        "status": ck.get("final_classification"),
        "signed_tdir_mean_deg": comp.get("signed_tdir_mean_deg", comp.get("tdir_mean_deg")),
        "tmag_median_ratio": comp.get("tmag_median_ratio"),
        "path_ratio": comp.get("path_ratio"),
        "sim3_ate": ext.get("sim3", {}).get("ate"),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    struct1b = _load(Path(args.struct1b_checkpoint))
    struct1 = _load(Path(args.struct1_checkpoint))
    s5e15 = _load(Path(args.s5e15_checkpoint))
    rows: List[Dict[str, Any]] = [
        _pick(s5e15, "S5E15"),
        _pick(struct1, "STRUCT1"),
        _pick(struct1b, "STRUCT1B"),
        {
            "method": "ORB-SLAM3",
            "status": "external_baseline",
            "signed_tdir_mean_deg": None,
            "tmag_median_ratio": None,
            "path_ratio": ORBSLAM3_REFERENCE["se3"]["path_ratio"],
            "sim3_ate": ORBSLAM3_REFERENCE["sim3"]["ate"],
        },
    ]
    summary = {
        "scale_repair_helped": bool(struct1b.get("component_metrics", {}).get("tmag_median_ratio", 1e9) < struct1.get("component_metrics", {}).get("tmag_median_ratio", 1e9)),
        "path_repair_helped": bool(struct1b.get("component_metrics", {}).get("path_ratio", 1e9) < struct1.get("component_metrics", {}).get("path_ratio", 1e9)),
        "should_do_struct1c_tdir_only": bool(struct1b.get("final_classification") == "STRUCT1B_SCALE_REPAIR_SUCCESS_TDIR_STILL_BAD"),
        "should_stop_struct1_struct2": bool(struct1b.get("recommendation", {}).get("recommended_next_stage") == "stop_STRUCT1_and_STRUCT2"),
    }
    write_json(Path(args.out_json), {"experiment": "STRUCT1B_scale_guard_repair", "rows": rows, "summary": summary})
    Path(args.out_report).write_text(
        "# STRUCT1B / STRUCT1 / S5E15 / ORB-SLAM3 对比\n\n"
        f"- scale 修复是否有效：{summary['scale_repair_helped']}\n"
        f"- path 修复是否有效：{summary['path_repair_helped']}\n"
        f"- 是否应进入 STRUCT1C_geometry_token_tdir_only：{summary['should_do_struct1c_tdir_only']}\n"
        f"- 是否应停止 STRUCT1/STRUCT2：{summary['should_stop_struct1_struct2']}\n",
        encoding="utf-8",
    )
    return {"rows": rows, "summary": summary}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--struct1b-checkpoint", required=True)
    p.add_argument("--struct1-checkpoint", required=True)
    p.add_argument("--s5e15-checkpoint", required=True)
    p.add_argument("--orbslam3-eval-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
