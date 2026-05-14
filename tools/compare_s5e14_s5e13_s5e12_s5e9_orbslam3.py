#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, write_json


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick(ckpt: Dict[str, Any], name: str) -> Dict[str, Any]:
    comp = ckpt.get("component_metrics", {})
    ov = comp.get("overall", comp)
    obs = comp.get("observable_edges", {})
    rel = comp.get("reliable_original_edges", comp.get("reliable_signed_direction_edges", {}))
    ext = ckpt.get("external_eval", {})
    return {
        "name": name,
        "coverage": ckpt.get("adjacent_dense_export", {}).get("coverage"),
        "rot_mean_deg": ov.get("rot_mean_deg"),
        "signed_tdir_mean_deg": ov.get("signed_tdir_mean_deg", ov.get("tdir_mean_deg")),
        "observable_edge_signed_tdir_mean_deg": obs.get("signed_tdir_mean_deg", obs.get("tdir_mean_deg")),
        "reliable_edge_anti_parallel_rate": rel.get("anti_parallel_rate"),
        "tdir_abs_mean_deg": ov.get("tdir_abs_mean_deg"),
        "anti_parallel_rate": ov.get("anti_parallel_rate"),
        "tmag_median_ratio": ov.get("tmag_median_ratio"),
        "tmag_p95_ratio": ov.get("tmag_p95_ratio"),
        "none_ate": ext.get("none", {}).get("ate"),
        "se3_ate": ext.get("se3", {}).get("ate"),
        "sim3_ate": ext.get("sim3", {}).get("ate"),
        "path_ratio": ov.get("path_ratio"),
        "caveat": ckpt.get("final_classification"),
    }


def _report(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# S5E14 / S5E13 / S5E12 / S5E9 / ORB-SLAM3 comparison",
        "",
        "## 执行摘要",
        payload["summary"],
        "",
        "## 对比表",
    ]
    for r in payload["rows"]:
        lines.append(str(r))
    lines += [
        "",
        "## 解释与结论",
        "- S5E14 的目标是把 S5E13 的 observable-edge 改善做扎实，同时降低 anti_parallel，并缓解 under-scale。",
        "- 即便 S5E14 有局部改善，也不能 claim 替代 official S5 locked result。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    s5e14 = _load(Path(args.s5e14_checkpoint))
    s5e13 = _load(Path(args.s5e13_checkpoint))
    s5e12 = _load(Path(args.s5e12_checkpoint))
    s5e9 = _load(Path("checkpoints/S5E9_scale_unit_small_motion_signed_direction_fix_candidate.json"))

    rows: List[Dict[str, Any]] = [
        {
            "name": "S5 official locked result",
            "coverage": None,
            "rot_mean_deg": None,
            "signed_tdir_mean_deg": None,
            "observable_edge_signed_tdir_mean_deg": None,
            "reliable_edge_anti_parallel_rate": None,
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
        _pick(s5e9, "S5E9 stable scale/path baseline"),
        {
            "name": "S5E12 feature gate",
            "coverage": 1.0,
            "rot_mean_deg": None,
            "signed_tdir_mean_deg": None,
            "observable_edge_signed_tdir_mean_deg": 46.32471099526666,
            "reliable_edge_anti_parallel_rate": None,
            "tdir_abs_mean_deg": None,
            "anti_parallel_rate": None,
            "tmag_median_ratio": None,
            "tmag_p95_ratio": None,
            "none_ate": None,
            "se3_ate": None,
            "sim3_ate": None,
            "path_ratio": None,
            "caveat": s5e12.get("final_classification"),
        },
        _pick(s5e13, "S5E13 correspondence direction candidate"),
        _pick(s5e14, "S5E14 refinement candidate"),
        {
            "name": "ORB-SLAM3 external baseline",
            "coverage": ORBSLAM3_REFERENCE["tracking_success_rate"],
            "rot_mean_deg": None,
            "signed_tdir_mean_deg": None,
            "observable_edge_signed_tdir_mean_deg": None,
            "reliable_edge_anti_parallel_rate": None,
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
    summary = "S5E14 对 S5E13 做 observable-edge direction refinement 与 strict-reliable gating，重点观察 anti_parallel 与 under-scale 是否同步改善；即使有改善，仍是 experimental candidate。"
    payload = {"experiment": "S5E14_observable_edge_direction_refinement", "rows": rows, "summary": summary}
    write_json(Path(args.out_json), payload)
    _report(Path(args.out_report), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s5e14-checkpoint", required=True)
    p.add_argument("--s5e13-checkpoint", required=True)
    p.add_argument("--s5e12-checkpoint", required=True)
    p.add_argument("--orbslam3-eval-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
