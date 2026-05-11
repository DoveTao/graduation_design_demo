#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from s5e2_adjacent_dense_lib import OFFICIAL_S5_LOCKED, ORBSLAM3_REFERENCE, read_json, write_json


def _row(name: str, coverage: str, ckpt: Dict[str, Any], caveat: str) -> Dict[str, Any]:
    c = ckpt.get("component_metrics", {})
    e = ckpt.get("external_eval", {})
    return {
        "method": name,
        "coverage": coverage,
        "rot_mean_deg": c.get("rot_mean_deg"),
        "signed_tdir_mean_deg": c.get("tdir_mean_deg"),
        "tdir_abs_mean_deg": c.get("tdir_abs_mean_deg"),
        "anti_parallel_rate": c.get("anti_parallel_rate"),
        "tmag_median_ratio": c.get("tmag_median_ratio"),
        "tmag_p95_ratio": c.get("tmag_p95_ratio"),
        "none_ate": e.get("none", {}).get("ate"),
        "se3_ate": e.get("se3", {}).get("ate"),
        "sim3_ate": e.get("sim3", {}).get("ate"),
        "path_ratio": c.get("path_ratio") or e.get("se3", {}).get("path_ratio"),
        "caveat": caveat,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    s5e8 = read_json(Path(args.s5e8_checkpoint))
    s5e7 = read_json(Path(args.s5e7_checkpoint))
    s5e6 = read_json(Path(args.s5e6_checkpoint))
    s5e5 = read_json(Path(args.s5e5_checkpoint))
    s5e4 = read_json(Path(args.s5e4_checkpoint))
    s5e3 = read_json(Path(args.s5e3_checkpoint))
    s5e2 = read_json(Path(args.s5e2_checkpoint))
    table = [
        {"method": "S5 official locked result", "coverage": "official evaluator", "rot_mean_deg": None, "signed_tdir_mean_deg": None, "tdir_abs_mean_deg": None, "anti_parallel_rate": None, "tmag_median_ratio": None, "tmag_p95_ratio": None, "none_ate": OFFICIAL_S5_LOCKED["ate"], "se3_ate": None, "sim3_ate": None, "path_ratio": OFFICIAL_S5_LOCKED["path_ratio"], "caveat": "official locked result"},
        _row("S5E2 traceable dense", "454/454", s5e2, "minimal ridge baseline"),
        _row("S5E3 traceable dense", "454/454", s5e3, "scale calibration"),
        _row("S5E4 traceable dense", "454/454", s5e4, "temporal sign prior"),
        _row("S5E5 traceable dense", "454/454", s5e5, "temporal visual backbone smoke"),
        _row("S5E6 traceable dense", "454/454", s5e6, "robust scale guard"),
        _row("S5E7 traceable dense", "454/454", s5e7, "direction-scale factorization but guard dependent"),
        _row("S5E8 traceable dense", "454/454", s5e8, "translation geometry diagnostic ablation"),
        {"method": "ORB-SLAM3 external baseline", "coverage": "273/454", "rot_mean_deg": None, "signed_tdir_mean_deg": None, "tdir_abs_mean_deg": None, "anti_parallel_rate": None, "tmag_median_ratio": None, "tmag_p95_ratio": None, "none_ate": ORBSLAM3_REFERENCE["none"]["ate"], "se3_ate": ORBSLAM3_REFERENCE["se3"]["ate"], "sim3_ate": ORBSLAM3_REFERENCE["sim3"]["ate"], "path_ratio": ORBSLAM3_REFERENCE["se3"]["path_ratio"], "caveat": "partial coverage external baseline"},
    ]
    payload = {
        "experiment": "S5E8_translation_geometry_diagnostic_ablation",
        "comparison_table": table,
        "comparison_summary": s5e8.get("comparison_summary"),
        "comparison_to_orbslam3": s5e8.get("comparison_to_orbslam3"),
        "final_classification": s5e8.get("final_classification"),
        "summary": "S5E8 关注 translation supervision、input geometry 与 small-motion 放大问题，而不是单纯刷 ATE。",
    }
    write_json(Path(args.out_json), payload)
    lines = [
        "# S5E8 / S5E7 / S5E6 / S5E5 / S5E4 / S5E3 / S5E2 / ORB-SLAM3 comparison",
        "",
        "## 执行摘要",
        "S5E8 的重点是诊断为什么 translation direction 和 scale 学不稳，而不是直接把 guarded 指标当作几何进步。",
        "",
        "## comparison table",
        "| method | coverage | rot | signed tdir | tdir_abs | anti_parallel_rate | tmag median | tmag p95 | none ATE | se3 ATE | sim3 ATE | path_ratio | caveat |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in table:
        lines.append(f"| {r['method']} | {r['coverage']} | {r['rot_mean_deg']} | {r['signed_tdir_mean_deg']} | {r['tdir_abs_mean_deg']} | {r['anti_parallel_rate']} | {r['tmag_median_ratio']} | {r['tmag_p95_ratio']} | {r['none_ate']} | {r['se3_ate']} | {r['sim3_ate']} | {r['path_ratio']} | {r['caveat']} |")
    lines += ["", "## comparison summary", str(payload["comparison_summary"]), "", "## comparison vs ORB-SLAM3", str(payload["comparison_to_orbslam3"]), "", "## caveats", "- S5E8 是实验性诊断，不替换官方 S5 locked 结果。", "- ORB-SLAM3 的 edge-level tdir/tmag gap 仍 unavailable。"]
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--s5e8-checkpoint", required=True)
    p.add_argument("--s5e7-checkpoint", required=True)
    p.add_argument("--s5e6-checkpoint", required=True)
    p.add_argument("--s5e5-checkpoint", required=True)
    p.add_argument("--s5e4-checkpoint", required=True)
    p.add_argument("--s5e3-checkpoint", required=True)
    p.add_argument("--s5e2-checkpoint", required=True)
    p.add_argument("--orbslam3-eval-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
