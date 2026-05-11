#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from s5e2_adjacent_dense_lib import OFFICIAL_S5_LOCKED, ORBSLAM3_REFERENCE, read_json, write_json


def _row(name, coverage, ckpt, caveat):
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


def run(args):
    s5e10 = read_json(Path(args.s5e10_checkpoint))
    s5e9 = read_json(Path(args.s5e9_checkpoint))
    s5e8 = read_json(Path(args.s5e8_checkpoint))
    s5e7 = read_json(Path(args.s5e7_checkpoint))
    s5e6 = read_json(Path(args.s5e6_checkpoint))
    s5e5 = read_json(Path(args.s5e5_checkpoint))
    s5e4 = read_json(Path(args.s5e4_checkpoint))
    s5e3 = read_json(Path(args.s5e3_checkpoint))
    s5e2 = read_json(Path(args.s5e2_checkpoint))
    table = [
        {"method": "S5 official locked result", "coverage": "official evaluator", "rot_mean_deg": None, "signed_tdir_mean_deg": None, "tdir_abs_mean_deg": None, "anti_parallel_rate": None, "tmag_median_ratio": None, "tmag_p95_ratio": None, "none_ate": OFFICIAL_S5_LOCKED["ate"], "se3_ate": None, "sim3_ate": None, "path_ratio": OFFICIAL_S5_LOCKED["path_ratio"], "caveat": "official locked result"},
        _row("S5E2", "454/454", s5e2, "minimal ridge"),
        _row("S5E3", "454/454", s5e3, "scale calibration"),
        _row("S5E4", "454/454", s5e4, "signed direction prior"),
        _row("S5E5", "454/454", s5e5, "temporal visual smoke"),
        _row("S5E6", "454/454", s5e6, "robust scale guard"),
        _row("S5E7", "454/454", s5e7, "guard dependent"),
        _row("S5E8", "454/454", s5e8, "translation diagnostic ablation"),
        _row("S5E9", "454/454", s5e9, "scale raw improved"),
        _row("S5E10", "454/454", s5e10, "order-aware signed direction + scale recalibration"),
        {"method": "ORB-SLAM3 external baseline", "coverage": "273/454", "rot_mean_deg": None, "signed_tdir_mean_deg": None, "tdir_abs_mean_deg": None, "anti_parallel_rate": None, "tmag_median_ratio": None, "tmag_p95_ratio": None, "none_ate": ORBSLAM3_REFERENCE["none"]["ate"], "se3_ate": ORBSLAM3_REFERENCE["se3"]["ate"], "sim3_ate": ORBSLAM3_REFERENCE["sim3"]["ate"], "path_ratio": ORBSLAM3_REFERENCE["se3"]["path_ratio"], "caveat": "partial coverage external baseline"},
    ]
    payload = {
        "experiment": "S5E10_order_aware_signed_direction_scale_recalibration",
        "comparison_table": table,
        "comparison_summary": s5e10.get("comparison_summary"),
        "comparison_to_orbslam3": s5e10.get("comparison_to_orbslam3"),
        "final_classification": s5e10.get("final_classification"),
    }
    write_json(Path(args.out_json), payload)
    lines = [
        "# S5E10 / S5E9 / S5E8 / S5E7 / S5E6 / S5E5 / S5E4 / S5E3 / S5E2 / ORB-SLAM3 comparison",
        "",
        "## 执行摘要",
        "S5E10 的重点是确认 signed direction 失败是否来自 pair-order 不可观测，并在不重新引入 scale explosion 的前提下恢复 path ratio。",
        "",
        "## comparison table",
        "| method | coverage | rot | signed tdir | tdir_abs | anti_parallel_rate | tmag median | tmag p95 | none ATE | se3 ATE | sim3 ATE | path_ratio | caveat |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in table:
        lines.append(f"| {r['method']} | {r['coverage']} | {r['rot_mean_deg']} | {r['signed_tdir_mean_deg']} | {r['tdir_abs_mean_deg']} | {r['anti_parallel_rate']} | {r['tmag_median_ratio']} | {r['tmag_p95_ratio']} | {r['none_ate']} | {r['se3_ate']} | {r['sim3_ate']} | {r['path_ratio']} | {r['caveat']} |")
    lines += ["", "## comparison summary", str(payload["comparison_summary"]), "", "## comparison vs ORB-SLAM3", str(payload["comparison_to_orbslam3"]), "", "## caveats", "- S5E10 是 experimental，不替换 official S5 locked result。"]
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--s5e10-checkpoint", required=True)
    p.add_argument("--s5e9-checkpoint", required=True)
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
