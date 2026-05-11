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
    audit = ck.get("no_harm_gate_audit", {})
    return {
        "method": name,
        "status": ck.get("final_classification"),
        "coverage": ck.get("adjacent_dense_export", {}).get("coverage", 1.0),
        "signed_tdir_mean_deg": comp.get("signed_tdir_mean_deg", comp.get("tdir_mean_deg")),
        "anti_parallel_rate": comp.get("anti_parallel_rate"),
        "tmag_median_ratio": comp.get("tmag_median_ratio"),
        "path_ratio": comp.get("path_ratio"),
        "modified_edge_count": audit.get("modified_edge_count"),
        "no_harm_pass_rate": audit.get("no_harm_pass_rate"),
        "se3_ate": ext.get("se3", {}).get("ate"),
        "sim3_ate": ext.get("sim3", {}).get("ate"),
        "caveat": ck.get("final_classification"),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    s21 = _load(Path(args.s5e21_checkpoint))
    s20 = _load(Path(args.s5e20_checkpoint))
    s15 = _load(Path(args.s5e15_checkpoint))
    rows: List[Dict[str, Any]] = [
        _pick(s15, "S5E15"),
        _pick(s20, "S5E20"),
        _pick(s21, "S5E21"),
        {
            "method": "ORB-SLAM3",
            "status": "external_baseline",
            "coverage": ORBSLAM3_REFERENCE["tracking_success_rate"],
            "signed_tdir_mean_deg": None,
            "anti_parallel_rate": None,
            "tmag_median_ratio": None,
            "path_ratio": ORBSLAM3_REFERENCE["se3"]["path_ratio"],
            "modified_edge_count": None,
            "no_harm_pass_rate": None,
            "se3_ate": ORBSLAM3_REFERENCE["se3"]["ate"],
            "sim3_ate": ORBSLAM3_REFERENCE["sim3"]["ate"],
            "caveat": "external_baseline",
        },
    ]
    summary = (
        "S5E21 的目标不是再追求全局 direction 进攻，而是验证 observability-gated no-harm refinement 是否至少比 S5E20 更安全。"
        "如果它没有真实超过 S5E15，就不应继续 promote。"
    )
    write_json(Path(args.out_json), {"experiment": "S5E21_no_harm_observability_gated_refinement", "rows": rows, "summary": summary})
    Path(args.out_report).write_text("# S5E21 对比报告\n\n## 执行摘要\n" + summary + "\n", encoding="utf-8")
    return {"rows": rows, "summary": summary}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s5e21-checkpoint", required=True)
    p.add_argument("--s5e20-checkpoint", required=True)
    p.add_argument("--s5e15-checkpoint", required=True)
    p.add_argument("--orbslam3-eval-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
