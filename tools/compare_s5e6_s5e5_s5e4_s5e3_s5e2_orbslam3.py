#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from s5e2_adjacent_dense_lib import OFFICIAL_S5_LOCKED, ORBSLAM3_REFERENCE, read_json, write_json


def _row(name: str, coverage: str, ckpt: Dict[str, Any], caveat: str) -> Dict[str, Any]:
    c, e = ckpt.get("component_metrics", {}), ckpt.get("external_eval", {})
    return {"method": name, "coverage": coverage, "rot_mean_deg": c.get("rot_mean_deg"), "tdir_mean_deg": c.get("tdir_mean_deg"), "tdir_abs_mean_deg": c.get("tdir_abs_mean_deg"), "anti_parallel_rate": c.get("anti_parallel_rate"), "tmag_median_ratio": c.get("tmag_median_ratio"), "tmag_p95_ratio": c.get("tmag_p95_ratio"), "none_ate": e.get("none", {}).get("ate"), "se3_ate": e.get("se3", {}).get("ate"), "sim3_ate": e.get("sim3", {}).get("ate"), "path_ratio": c.get("path_ratio") or e.get("se3", {}).get("path_ratio"), "caveat": caveat}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    s5e6, s5e5, s5e4, s5e3, s5e2 = [read_json(Path(p)) for p in [args.s5e6_checkpoint, args.s5e5_checkpoint, args.s5e4_checkpoint, args.s5e3_checkpoint, args.s5e2_checkpoint]]
    table = [
        {"method": "S5 official locked result", "coverage": "official evaluator", "rot_mean_deg": None, "tdir_mean_deg": None, "tdir_abs_mean_deg": None, "anti_parallel_rate": None, "tmag_median_ratio": None, "tmag_p95_ratio": None, "none_ate": OFFICIAL_S5_LOCKED["ate"], "se3_ate": None, "sim3_ate": None, "path_ratio": OFFICIAL_S5_LOCKED["path_ratio"], "caveat": "official locked result"},
        _row("S5E2 traceable dense", "454/454", s5e2, "minimal ridge baseline"),
        _row("S5E3 traceable dense", "454/454", s5e3, "scale calibration"),
        _row("S5E4 traceable dense", "454/454", s5e4, "temporal sign prior"),
        _row("S5E5 traceable dense", "454/454", s5e5, "temporal visual backbone smoke"),
        _row("S5E6 traceable dense", "454/454", s5e6, "robust scale guard"),
        {"method": "ORB-SLAM3 external baseline", "coverage": "273/454", "rot_mean_deg": None, "tdir_mean_deg": None, "tdir_abs_mean_deg": None, "anti_parallel_rate": None, "tmag_median_ratio": None, "tmag_p95_ratio": None, "none_ate": ORBSLAM3_REFERENCE["none"]["ate"], "se3_ate": ORBSLAM3_REFERENCE["se3"]["ate"], "sim3_ate": ORBSLAM3_REFERENCE["sim3"]["ate"], "path_ratio": ORBSLAM3_REFERENCE["se3"]["path_ratio"], "caveat": "partial coverage external baseline"},
    ]
    e6 = s5e6.get("external_eval", {})
    payload = {
        "experiment": "S5E6_robust_scale_guard_and_direction_metric_gate_candidate",
        "comparison_table": table,
        "comparison_to_orbslam3": {
            "coverage": "S5E6 454/454 vs ORB-SLAM3 273/454",
            "se3_ate_gap": None if e6.get("se3", {}).get("ate") is None else e6["se3"]["ate"] - ORBSLAM3_REFERENCE["se3"]["ate"],
            "sim3_ate_gap": None if e6.get("sim3", {}).get("ate") is None else e6["sim3"]["ate"] - ORBSLAM3_REFERENCE["sim3"]["ate"],
            "path_ratio": f"S5E6={s5e6.get('component_metrics', {}).get('path_ratio')}; ORB-SLAM3={ORBSLAM3_REFERENCE['se3']['path_ratio']}",
        },
        "interpretation": "S5E6 is a robust scale-guard experiment focused on fixing S5E5 path explosion without giving up full traceability.",
        "final_classification": s5e6.get("final_classification"),
    }
    write_json(Path(args.out_json), payload)
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# S5E6 / S5E5 / S5E4 / S5E3 / S5E2 / ORB-SLAM3 comparison",
        "",
        "## 执行摘要",
        "S5E6 不是去堆更大的模型，而是优先修复 S5E5 的 scale outlier 和 path explosion。",
        "",
        "## comparison table",
        "| method | coverage | rot | signed tdir | tdir_abs | anti_parallel_rate | tmag median | tmag p95 | none ATE | se3 ATE | sim3 ATE | path_ratio | caveat |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in table:
        lines.append(f"| {r['method']} | {r['coverage']} | {r['rot_mean_deg']} | {r['tdir_mean_deg']} | {r['tdir_abs_mean_deg']} | {r['anti_parallel_rate']} | {r['tmag_median_ratio']} | {r['tmag_p95_ratio']} | {r['none_ate']} | {r['se3_ate']} | {r['sim3_ate']} | {r['path_ratio']} | {r['caveat']} |")
    lines += ["", "## interpretation", payload["interpretation"], "", "## caveats", "- S5E6 是 experimental candidate。", "- 不替代 official S5 locked result。", "- S5 locked metrics/policy unchanged。", "- ORB-SLAM3 是 external strong baseline。"]
    Path(args.out_report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def parse_args():
    p = argparse.ArgumentParser()
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
