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
    overall = comp.get("overall", comp)
    ext = ck.get("external_eval", {})
    return {
        "method": name,
        "status": ck.get("final_classification"),
        "coverage": ck.get("adjacent_dense_export", {}).get("coverage"),
        "rot_mean_deg": overall.get("rot_mean_deg"),
        "signed_tdir_mean_deg": overall.get("signed_tdir_mean_deg", overall.get("tdir_mean_deg")),
        "tdir_abs_mean_deg": overall.get("tdir_abs_mean_deg"),
        "anti_parallel_rate": overall.get("anti_parallel_rate"),
        "tmag_median_ratio": overall.get("tmag_median_ratio"),
        "tmag_p95_ratio": overall.get("tmag_p95_ratio"),
        "path_ratio": overall.get("path_ratio"),
        "none_ate": ext.get("none", {}).get("ate"),
        "se3_ate": ext.get("se3", {}).get("ate"),
        "sim3_ate": ext.get("sim3", {}).get("ate"),
        "caveat": ck.get("final_classification"),
    }


def _report(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# S5E19 comparison report",
        "",
        "## 执行摘要",
        payload["summary"],
        "",
        "## 比较表",
    ]
    for row in payload["rows"]:
        lines.append(str(row))
    lines += [
        "",
        "## 解释",
        "- S5E19 是否保留主线，要看 spherical token / coarse-to-fine / geometry constraint 是否都还在主模型内部。",
        "- 如果没有优于 S5E15，主要 blocker 需要区分训练不足、结构接入失败还是 tdir 信号不足。",
        "- 即便 full coverage 仍然领先 ORB-SLAM3，也不能把它说成 official result replacement。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    s19 = _load(Path(args.s5e19_checkpoint))
    s15 = _load(Path(args.s5e15_checkpoint))
    s18 = _load(Path(args.s5e18_checkpoint))
    rows: List[Dict[str, Any]] = [
        {
            "method": "S5 official locked reference",
            "status": "official_locked",
            "coverage": None,
            "rot_mean_deg": None,
            "signed_tdir_mean_deg": None,
            "tdir_abs_mean_deg": None,
            "anti_parallel_rate": None,
            "tmag_median_ratio": None,
            "tmag_p95_ratio": None,
            "path_ratio": 0.932379,
            "none_ate": None,
            "se3_ate": 7.352288,
            "sim3_ate": 7.352288,
            "caveat": "official_locked",
        },
        _pick(s15, "S5E15"),
        _pick(s18, "S5E18"),
        _pick(s19, "S5E19"),
        {
            "method": "ORB-SLAM3",
            "status": "external_baseline",
            "coverage": ORBSLAM3_REFERENCE["tracking_success_rate"],
            "rot_mean_deg": None,
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
    summary = "S5E19 回到 spherical token + coarse-to-fine + multiframe geometry 主线，但是否真正优于 S5E15，要看 signed tdir、anti_parallel 和 path_ratio 是否一起受益。"
    payload = {"experiment": "S5E19_rotation_compensated_multiframe_geometry_refinement", "rows": rows, "summary": summary}
    write_json(Path(args.out_json), payload)
    _report(Path(args.out_report), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s5e19-checkpoint", required=True)
    p.add_argument("--s5e15-checkpoint", required=True)
    p.add_argument("--s5e18-checkpoint", required=True)
    p.add_argument("--orbslam3-eval-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
