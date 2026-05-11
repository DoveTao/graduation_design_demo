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
        "coverage": ck.get("coverage", ck.get("adjacent_dense_export", {})).get("coverage", 1.0) if isinstance(ck.get("coverage", {}), dict) else ck.get("adjacent_dense_export", {}).get("coverage", 1.0),
        "signed_tdir_mean_deg": comp.get("signed_tdir_mean_deg", comp.get("tdir_mean_deg")),
        "anti_parallel_rate": comp.get("anti_parallel_rate"),
        "tmag_median_ratio": comp.get("tmag_median_ratio"),
        "path_ratio": comp.get("path_ratio"),
        "sim3_ate": ext.get("sim3", {}).get("ate"),
        "caveat": ck.get("final_classification"),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    arch2 = _load(Path(args.arch2_checkpoint))
    s21 = _load(Path(args.s5e21_checkpoint))
    s20 = _load(Path(args.s5e20_checkpoint))
    s15 = _load(Path(args.s5e15_checkpoint))
    rows: List[Dict[str, Any]] = [
        _pick(s15, "S5E15"),
        _pick(s20, "S5E20"),
        _pick(s21, "S5E21"),
        _pick(arch2, "ARCH2"),
        {
            "method": "ORB-SLAM3",
            "status": "external_baseline",
            "coverage": ORBSLAM3_REFERENCE["tracking_success_rate"],
            "signed_tdir_mean_deg": None,
            "anti_parallel_rate": None,
            "tmag_median_ratio": None,
            "path_ratio": ORBSLAM3_REFERENCE["se3"]["path_ratio"],
            "sim3_ate": ORBSLAM3_REFERENCE["sim3"]["ate"],
            "caveat": "external_baseline",
        },
    ]
    summary = (
        "ARCH2 吸收了 S5E21 的 no-harm 控制，同时把 soft correspondence geometry、rotation compensation 和 true k-step 放回主线。"
        "如果它仍没有全局超过 S5E15，就说明下一步更适合进入 STRUCT1 级别的主结构重构，而不是继续在轻量 residual 上打补丁。"
    )
    write_json(Path(args.out_json), {"experiment": "ARCH2_mainline_rotation_compensated_softcorr_geometry", "rows": rows, "summary": summary})
    Path(args.out_report).write_text("# ARCH2 对比报告\n\n## 执行摘要\n" + summary + "\n", encoding="utf-8")
    return {"rows": rows, "summary": summary}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--arch2-checkpoint", required=True)
    p.add_argument("--s5e21-checkpoint", required=True)
    p.add_argument("--s5e20-checkpoint", required=True)
    p.add_argument("--s5e15-checkpoint", required=True)
    p.add_argument("--orbslam3-eval-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
