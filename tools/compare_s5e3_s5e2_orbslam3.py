#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from s5e2_adjacent_dense_lib import OFFICIAL_S5_LOCKED, ORBSLAM3_REFERENCE, read_json, write_json


def _row(name: str, coverage: Any, component: Dict[str, Any], external: Dict[str, Any], caveat: str) -> Dict[str, Any]:
    return {
        "method": name,
        "coverage": coverage,
        "rot_mean_deg": component.get("rot_mean_deg"),
        "tdir_mean_deg": component.get("tdir_mean_deg"),
        "tdir_abs_mean_deg": component.get("tdir_abs_mean_deg"),
        "tmag_median_ratio": component.get("tmag_median_ratio"),
        "none_ate": external.get("none", {}).get("ate"),
        "se3_ate": external.get("se3", {}).get("ate"),
        "sim3_ate": external.get("sim3", {}).get("ate"),
        "path_ratio": component.get("path_ratio") or external.get("se3", {}).get("path_ratio"),
        "caveat": caveat,
    }


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# S5E3 / S5E2 / ORB-SLAM3 comparison",
        "",
        "## 执行摘要",
        "",
        "本报告比较 S5 official locked result、S5E2 traceable dense、S5E3 traceable dense 与 ORB-SLAM3 external baseline。S5E3 仍是 experimental candidate，不替代 official S5 locked result。",
        "",
        "## comparison table",
        "",
        "| method | coverage | rot | tdir | tdir_abs | tmag | none ATE | se3 ATE | sim3 ATE | path_ratio | caveat |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["comparison_table"]:
        lines.append(f"| {row['method']} | {row['coverage']} | {row['rot_mean_deg']} | {row['tdir_mean_deg']} | {row['tdir_abs_mean_deg']} | {row['tmag_median_ratio']} | {row['none_ate']} | {row['se3_ate']} | {row['sim3_ate']} | {row['path_ratio']} | {row['caveat']} |")
    lines += [
        "",
        "## interpretation",
        "",
        payload["interpretation"],
        "",
        "## recommendations",
        "",
        "S5E3 若仍未接近 ORB-SLAM3，应优先升级视觉 backbone 与 direction head；若要使用 ORB-SLAM3 做 teacher，需要另开明确的 distillation 实验。",
        "",
        "## caveats",
        "",
        "- S5E3 是 experimental candidate。",
        "- 不替代 official S5 locked result。",
        "- S5 locked metrics/policy unchanged。",
        "- ORB-SLAM3 是 external strong baseline。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    s5e3 = read_json(Path(args.s5e3_checkpoint))
    s5e2 = read_json(Path(args.s5e2_checkpoint))
    s5e3_comp, s5e2_comp = s5e3.get("component_metrics", {}), s5e2.get("component_metrics", {})
    s5e3_ext, s5e2_ext = s5e3.get("external_eval", {}), s5e2.get("external_eval", {})
    table = [
        {"method": "S5 official locked result", "coverage": "official evaluator", "rot_mean_deg": None, "tdir_mean_deg": None, "tdir_abs_mean_deg": None, "tmag_median_ratio": None, "none_ate": OFFICIAL_S5_LOCKED["ate"], "se3_ate": None, "sim3_ate": None, "path_ratio": OFFICIAL_S5_LOCKED["path_ratio"], "caveat": "official locked result; not external dense"},
        _row("S5E2 traceable dense", "454/454", s5e2_comp, s5e2_ext, "experimental minimal ridge; failed metrics"),
        _row("S5E3 traceable dense", "454/454", s5e3_comp, s5e3_ext, "experimental scale calibrated candidate"),
        {"method": "ORB-SLAM3 external baseline", "coverage": "273/454", "rot_mean_deg": None, "tdir_mean_deg": None, "tdir_abs_mean_deg": None, "tmag_median_ratio": None, "none_ate": ORBSLAM3_REFERENCE["none"]["ate"], "se3_ate": ORBSLAM3_REFERENCE["se3"]["ate"], "sim3_ate": ORBSLAM3_REFERENCE["sim3"]["ate"], "path_ratio": ORBSLAM3_REFERENCE["se3"]["path_ratio"], "caveat": "partial coverage; external strong baseline"},
    ]
    se3_gap = None
    sim3_gap = None
    if s5e3_ext.get("se3", {}).get("ate") is not None:
        se3_gap = s5e3_ext["se3"]["ate"] - ORBSLAM3_REFERENCE["se3"]["ate"]
    if s5e3_ext.get("sim3", {}).get("ate") is not None:
        sim3_gap = s5e3_ext["sim3"]["ate"] - ORBSLAM3_REFERENCE["sim3"]["ate"]
    interpretation = (
        f"S5E3 相对 S5E2 的 improvement flags 为 {s5e3.get('improvement_vs_s5e2')}. "
        f"S5E3 对 ORB-SLAM3 的 se3 ATE gap={se3_gap}, sim3 ATE gap={sim3_gap}. "
        "full coverage 仍是 S5E3 相对 ORB-SLAM3 的优势，但不得声称替代 official S5。"
    )
    payload = {
        "experiment": "S5E3_scale_calibrated_adjacent_dense_candidate",
        "comparison_table": table,
        "comparison_to_orbslam3": {"coverage": "S5E3 454/454 vs ORB-SLAM3 273/454", "se3_ate_gap": se3_gap, "sim3_ate_gap": sim3_gap, "path_ratio": f"S5E3={s5e3_comp.get('path_ratio')}; ORB-SLAM3={ORBSLAM3_REFERENCE['se3']['path_ratio']}"},
        "interpretation": interpretation,
        "final_classification": s5e3.get("final_classification"),
    }
    write_json(Path(args.out_json), payload)
    s5e3["comparison_to_orbslam3"].update({"aligned_ate_gap_to_orbslam3": se3_gap, "summary": interpretation})
    write_json(Path(args.s5e3_checkpoint), s5e3)
    _write_report(Path(args.out_report), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare S5E3 with S5E2 and ORB-SLAM3.")
    parser.add_argument("--s5e3-checkpoint", required=True)
    parser.add_argument("--s5e2-checkpoint", required=True)
    parser.add_argument("--orbslam3-eval-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
