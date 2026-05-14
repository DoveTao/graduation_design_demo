#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, RESTORED_S5_DENSE_REFERENCE, load_or_base_checkpoint, write_json


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# S5E2 与 ORB-SLAM3 external comparison",
        "",
        "## 执行摘要",
        "",
        "S5E2 是 experimental adjacent_dense candidate，目标是解决 S5E1 的 selected-only blocker。它与 ORB-SLAM3 的比较只能作为 diagnostic comparison，不替代 official S5 locked result。",
        "",
        "## S5E2",
        "",
        f"- coverage = {payload['s5e2']['coverage']}",
        f"- none = {payload['s5e2']['none']}",
        f"- se3 = {payload['s5e2']['se3']}",
        f"- sim3 = {payload['s5e2']['sim3']}",
        f"- component_metrics = {payload['s5e2']['component_metrics']}",
        "",
        "## ORB-SLAM3",
        "",
        f"- coverage = {ORBSLAM3_REFERENCE['coverage']}",
        f"- none = {ORBSLAM3_REFERENCE['none']}",
        f"- se3 = {ORBSLAM3_REFERENCE['se3']}",
        f"- sim3 = {ORBSLAM3_REFERENCE['sim3']}",
        "",
        "## interpretation",
        "",
        payload["comparison_to_orbslam3"]["summary"],
        "",
        "## recommendations",
        "",
        "S5E2 若要接近 ORB-SLAM3，需要把 minimal baseline 升级为真正视觉 backbone，并继续保持 full edge provenance、no GT leakage 和 held-out evaluation。",
        "",
        "## caveats",
        "",
        "- S5E2 是 experimental candidate。",
        "- 不替代 official S5 locked result。",
        "- S5 locked metrics/policy unchanged。",
        "- ORB-SLAM3 是 external strong baseline。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    ckpt = load_or_base_checkpoint(Path(args.s5e2_checkpoint))
    export = ckpt.get("adjacent_dense_export", {})
    external = ckpt.get("external_eval", {})
    component = ckpt.get("component_metrics", {})
    se3_ate = external.get("se3", {}).get("ate")
    sim3_ate = external.get("sim3", {}).get("ate")
    path_ratio = component.get("path_ratio")
    coverage_adv = (export.get("num_poses") or 0) >= 454
    ate_gap = None if se3_ate is None else se3_ate - ORBSLAM3_REFERENCE["se3"]["ate"]
    closer_than_restored = None if se3_ate is None else se3_ate < RESTORED_S5_DENSE_REFERENCE["se3_ate"]
    summary = (
        "S5E2 生成了 full coverage traceable dense candidate，并且相对 restored S5 dense diagnostic 更合理。"
        if closer_than_restored
        else "S5E2 已建立 traceable dense pipeline，但 aligned ATE 尚未优于 restored S5 dense diagnostic。"
    )
    if ate_gap is not None:
        summary += f" 与 ORB-SLAM3 se3 ATE 的差距为 {ate_gap}。"
    if path_ratio is not None:
        summary += f" S5E2 path_ratio 为 {path_ratio}，ORB-SLAM3 path_ratio 为 {ORBSLAM3_REFERENCE['se3']['path_ratio']}。"
    comparison = {
        "experiment": "S5E2_real_adjacent_dense_candidate",
        "s5e2_checkpoint": args.s5e2_checkpoint,
        "orbslam3_eval_dir": args.orbslam3_eval_dir,
        "s5e2": {
            "coverage": export.get("coverage"),
            "num_poses": export.get("num_poses"),
            "num_edges": export.get("num_edges"),
            "none": external.get("none"),
            "se3": external.get("se3"),
            "sim3": external.get("sim3"),
            "component_metrics": component,
        },
        "orbslam3": ORBSLAM3_REFERENCE,
        "restored_s5_dense_reference": RESTORED_S5_DENSE_REFERENCE,
        "comparison_to_orbslam3": {
            "orbslam3_coverage": "273/454",
            "s5e2_coverage_advantage": coverage_adv,
            "aligned_ate_gap_to_orbslam3": ate_gap,
            "sim3_ate_gap_to_orbslam3": None if sim3_ate is None else sim3_ate - ORBSLAM3_REFERENCE["sim3"]["ate"],
            "path_ratio_comparison": f"S5E2={path_ratio}; ORB-SLAM3={ORBSLAM3_REFERENCE['se3']['path_ratio']}",
            "closer_to_orbslam3_than_restored_s5_dense": closer_than_restored,
            "summary": summary,
        },
        "final_classification": ckpt.get("final_classification"),
    }
    write_json(Path(args.out_json), comparison)
    ckpt["comparison_to_orbslam3"] = comparison["comparison_to_orbslam3"]
    write_json(Path(args.s5e2_checkpoint), ckpt)
    _write_report(Path(args.out_report), comparison)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare S5E2 experimental candidate with ORB-SLAM3.")
    parser.add_argument("--s5e2-checkpoint", required=True)
    parser.add_argument("--orbslam3-eval-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
