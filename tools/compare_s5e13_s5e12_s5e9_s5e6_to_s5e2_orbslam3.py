#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, read_json, write_json


def _load(path: Path) -> Dict[str, Any]:
    return read_json(path)


def _pick(ckpt: Dict[str, Any], name: str) -> Dict[str, Any]:
    comp = ckpt.get("component_metrics", {})
    overall = comp.get("overall", comp)
    obs = comp.get("observable_edges", {})
    ext = ckpt.get("external_eval", {})
    return {
        "name": name,
        "coverage": ckpt.get("adjacent_dense_export", {}).get("coverage"),
        "rot_mean_deg": overall.get("rot_mean_deg"),
        "signed_tdir_mean_deg": overall.get("tdir_mean_deg"),
        "observable_edge_tdir_mean_deg": obs.get("tdir_mean_deg"),
        "tdir_abs_mean_deg": overall.get("tdir_abs_mean_deg"),
        "anti_parallel_rate": overall.get("anti_parallel_rate"),
        "tmag_median_ratio": overall.get("tmag_median_ratio"),
        "tmag_p95_ratio": overall.get("tmag_p95_ratio"),
        "none_ate": ext.get("none", {}).get("ate"),
        "se3_ate": ext.get("se3", {}).get("ate"),
        "sim3_ate": ext.get("sim3", {}).get("ate"),
        "path_ratio": overall.get("path_ratio"),
        "caveat": ckpt.get("final_classification"),
    }


def _report(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# S5E13 comparison report",
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
        "- S5E13 的重点是把 S5E12 的 real correspondence feature gate 转化为 signed direction 监督收益。",
        "- 如果改善只出现在 observable edges，而 all-edge 仍差，需要按 partial 解释，而不是夸大为整体几何改善。",
        "- full coverage 仍然是 S5E13 相对 ORB-SLAM3 的一个优势。",
        "- S5E13 不替代 official S5 locked result。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    s5e13 = _load(Path(args.s5e13_checkpoint))
    s5e9 = _load(Path("checkpoints/S5E9_scale_unit_small_motion_signed_direction_fix_candidate.json"))
    s5e6 = _load(Path("checkpoints/S5E6_robust_scale_guard_direction_gate_candidate.json"))
    s5e5 = _load(Path("checkpoints/S5E5_temporal_visual_backbone_geometry_candidate.json"))
    s5e4 = _load(Path("checkpoints/S5E4_temporal_direction_head_candidate.json"))
    s5e2 = _load(Path("checkpoints/S5E2_adjacent_dense_candidate.json"))
    rows: List[Dict[str, Any]] = [
        {
            "name": "S5 official locked result",
            "coverage": None,
            "rot_mean_deg": None,
            "signed_tdir_mean_deg": None,
            "observable_edge_tdir_mean_deg": None,
            "tdir_abs_mean_deg": None,
            "anti_parallel_rate": None,
            "tmag_median_ratio": None,
            "tmag_p95_ratio": None,
            "none_ate": None,
            "se3_ate": 7.352288,
            "sim3_ate": 7.352288,
            "path_ratio": 0.932379,
            "caveat": "official_locked",
        },
        _pick(s5e2, "S5E2"),
        _pick(s5e4, "S5E4"),
        _pick(s5e5, "S5E5"),
        _pick(s5e6, "S5E6"),
        _pick(s5e9, "S5E9"),
        {
            "name": "S5E12 feature gate",
            "coverage": 1.0,
            "rot_mean_deg": None,
            "signed_tdir_mean_deg": None,
            "observable_edge_tdir_mean_deg": 0.2980132450331126,
            "tdir_abs_mean_deg": None,
            "anti_parallel_rate": None,
            "tmag_median_ratio": None,
            "tmag_p95_ratio": None,
            "none_ate": None,
            "se3_ate": None,
            "sim3_ate": None,
            "path_ratio": None,
            "caveat": "feature_gate_only",
        },
        _pick(s5e13, "S5E13"),
        {
            "name": "ORB-SLAM3 external baseline",
            "coverage": ORBSLAM3_REFERENCE["tracking_success_rate"],
            "rot_mean_deg": None,
            "signed_tdir_mean_deg": None,
            "observable_edge_tdir_mean_deg": None,
            "tdir_abs_mean_deg": None,
            "anti_parallel_rate": None,
            "tmag_median_ratio": None,
            "tmag_p95_ratio": None,
            "none_ate": ORBSLAM3_REFERENCE["none"]["ate"],
            "se3_ate": ORBSLAM3_REFERENCE["se3"]["ate"],
            "sim3_ate": ORBSLAM3_REFERENCE["sim3"]["ate"],
            "path_ratio": ORBSLAM3_REFERENCE["se3"]["path_ratio"],
            "caveat": "external_baseline",
        },
    ]
    summary = "S5E13 重点检查 S5E12 的 real correspondence observability 是否真正转化为 signed direction 改善；如果只能在 observable edges 上改善，则应如实标记为 partial。"
    payload = {"experiment": "S5E13_real_correspondence_signed_direction_training", "rows": rows, "summary": summary}
    write_json(Path(args.out_json), payload)
    _report(Path(args.out_report), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s5e13-checkpoint", dest="s5e13_checkpoint", required=True)
    p.add_argument("--orbslam3-eval-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
