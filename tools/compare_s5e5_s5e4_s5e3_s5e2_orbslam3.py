#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from s5e2_adjacent_dense_lib import OFFICIAL_S5_LOCKED, ORBSLAM3_REFERENCE, read_json, write_json


def _row(name: str, coverage: str, ckpt: Dict[str, Any], caveat: str) -> Dict[str, Any]:
    c, e = ckpt.get("component_metrics", {}), ckpt.get("external_eval", {})
    return {
        "method": name,
        "coverage": coverage,
        "rot_mean_deg": c.get("rot_mean_deg"),
        "tdir_mean_deg": c.get("tdir_mean_deg"),
        "tdir_abs_mean_deg": c.get("tdir_abs_mean_deg"),
        "anti_parallel_rate": c.get("anti_parallel_rate"),
        "tmag_median_ratio": c.get("tmag_median_ratio"),
        "none_ate": e.get("none", {}).get("ate"),
        "se3_ate": e.get("se3", {}).get("ate"),
        "sim3_ate": e.get("sim3", {}).get("ate"),
        "path_ratio": c.get("path_ratio") or e.get("se3", {}).get("path_ratio"),
        "caveat": caveat,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    s5e5 = read_json(Path(args.s5e5_checkpoint))
    s5e4 = read_json(Path(args.s5e4_checkpoint))
    s5e3 = read_json(Path(args.s5e3_checkpoint))
    s5e2 = read_json(Path(args.s5e2_checkpoint))
    table = [
        {"method": "S5 official locked result", "coverage": "official evaluator", "rot_mean_deg": None, "tdir_mean_deg": None, "tdir_abs_mean_deg": None, "anti_parallel_rate": None, "tmag_median_ratio": None, "none_ate": OFFICIAL_S5_LOCKED["ate"], "se3_ate": None, "sim3_ate": None, "path_ratio": OFFICIAL_S5_LOCKED["path_ratio"], "caveat": "official locked result"},
        _row("S5E2 traceable dense", "454/454", s5e2, "minimal ridge baseline"),
        _row("S5E3 traceable dense", "454/454", s5e3, "scale calibration"),
        _row("S5E4 traceable dense", "454/454", s5e4, "temporal sign prior"),
        _row("S5E5 traceable dense", "454/454", s5e5, "temporal visual backbone"),
        {"method": "ORB-SLAM3 external baseline", "coverage": "273/454", "rot_mean_deg": None, "tdir_mean_deg": None, "tdir_abs_mean_deg": None, "anti_parallel_rate": None, "tmag_median_ratio": None, "none_ate": ORBSLAM3_REFERENCE["none"]["ate"], "se3_ate": ORBSLAM3_REFERENCE["se3"]["ate"], "sim3_ate": ORBSLAM3_REFERENCE["sim3"]["ate"], "path_ratio": ORBSLAM3_REFERENCE["se3"]["path_ratio"], "caveat": "partial coverage external baseline"},
    ]
    e5 = s5e5.get("external_eval", {})
    se3_gap = None if e5.get("se3", {}).get("ate") is None else e5["se3"]["ate"] - ORBSLAM3_REFERENCE["se3"]["ate"]
    sim3_gap = None if e5.get("sim3", {}).get("ate") is None else e5["sim3"]["ate"] - ORBSLAM3_REFERENCE["sim3"]["ate"]
    payload = {
        "experiment": "S5E5_temporal_visual_backbone_geometry_candidate",
        "comparison_table": table,
        "comparison_to_orbslam3": {
            "coverage": "S5E5 454/454 vs ORB-SLAM3 273/454",
            "se3_ate_gap": se3_gap,
            "sim3_ate_gap": sim3_gap,
            "path_ratio": f"S5E5={s5e5.get('component_metrics', {}).get('path_ratio')}; ORB-SLAM3={ORBSLAM3_REFERENCE['se3']['path_ratio']}",
        },
        "interpretation": "S5E5 is the first ordered image-pair temporal visual candidate; it remains diagnostic and cannot replace official S5.",
        "final_classification": s5e5.get("final_classification"),
    }
    write_json(Path(args.out_json), payload)
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# S5E5 / S5E4 / S5E3 / S5E2 / ORB-SLAM3 comparison",
        "",
        "## 执行摘要",
        "S5E5 使用 ordered image pair temporal visual backbone，目标是同时改进 signed tdir、tdir_abs、tmag 和 path_ratio，同时保留 full traceability。",
        "",
        "## comparison table",
        "| method | coverage | rot | signed tdir | tdir_abs | anti_parallel_rate | tmag | none ATE | se3 ATE | sim3 ATE | path_ratio | caveat |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in table:
        lines.append(f"| {r['method']} | {r['coverage']} | {r['rot_mean_deg']} | {r['tdir_mean_deg']} | {r['tdir_abs_mean_deg']} | {r['anti_parallel_rate']} | {r['tmag_median_ratio']} | {r['none_ate']} | {r['se3_ate']} | {r['sim3_ate']} | {r['path_ratio']} | {r['caveat']} |")
    lines += [
        "",
        "## interpretation",
        payload["interpretation"],
        "",
        "## caveats",
        "- S5E5 是 experimental candidate。",
        "- 不替代 official S5 locked result。",
        "- S5 locked metrics/policy unchanged。",
        "- ORB-SLAM3 是 external strong baseline。",
    ]
    Path(args.out_report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def parse_args():
    p = argparse.ArgumentParser()
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
