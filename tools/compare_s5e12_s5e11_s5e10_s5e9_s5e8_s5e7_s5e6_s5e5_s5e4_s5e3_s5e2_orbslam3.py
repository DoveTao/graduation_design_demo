#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from s5e2_adjacent_dense_lib import read_json, write_json


def run(args: argparse.Namespace) -> Dict[str, Any]:
    s5e12 = read_json(Path(args.s5e12_checkpoint))
    s5e11 = read_json(Path(args.s5e11_checkpoint))
    rows = {
        "S5E12": {
            "classification": s5e12.get("final_classification"),
            "observable_edge_fraction": s5e12.get("observability_gate", {}).get("observable_edge_fraction"),
            "reliable_signed_direction_fraction": s5e12.get("observability_gate", {}).get("signed_direction_supervision_reliable_fraction"),
            "correspondence_success_fraction": s5e12.get("correspondence_quality_audit", {}).get("correspondence_success_fraction"),
            "strict_essential_geometry_available": s5e12.get("essential_geometry_audit", {}).get("strict_essential_geometry_available"),
        },
        "S5E11": {
            "classification": s5e11.get("final_classification"),
            "observable_edge_fraction": s5e11.get("observability_audit", {}).get("observable_edge_fraction"),
            "reliable_signed_direction_fraction": s5e11.get("observability_audit", {}).get("signed_direction_supervision_reliable_fraction"),
        },
        "S5E10": {"classification": read_json(Path(args.s5e10_checkpoint)).get("final_classification")},
        "S5E9": {"classification": read_json(Path(args.s5e9_checkpoint)).get("final_classification")},
        "S5E8": {"classification": read_json(Path(args.s5e8_checkpoint)).get("final_classification")},
        "S5E7": {"classification": read_json(Path(args.s5e7_checkpoint)).get("final_classification")},
        "S5E6": {"classification": read_json(Path(args.s5e6_checkpoint)).get("final_classification")},
        "S5E5": {"classification": read_json(Path(args.s5e5_checkpoint)).get("final_classification")},
        "S5E4": {"classification": read_json(Path(args.s5e4_checkpoint)).get("final_classification")},
        "S5E3": {"classification": read_json(Path(args.s5e3_checkpoint)).get("final_classification")},
        "S5E2": {"classification": read_json(Path(args.s5e2_checkpoint)).get("final_classification")},
        "ORB-SLAM3": {
            "coverage": "273/454",
            "edge_level_direction_gap": "unavailable",
        },
    }
    payload = {
        "experiment": "S5E12_real_correspondence_feature_gate",
        "rows": rows,
        "summary": "S5E12 重点比较 observable/reliable fraction 是否高于 S5E11，而不是比较最终 trajectory ATE。",
    }
    write_json(Path(args.out_json), payload)
    report = Path(args.out_report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# S5E12 comparison\n\n"
        "## 对比摘要\n"
        "S5E12 关注真实 correspondence feature gate 能否提高 signed direction supervision 的可靠比例，而不是直接替代前序 trajectory 实验。\n\n"
        f"## rows\n{rows}\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s5e12-checkpoint", required=True)
    p.add_argument("--s5e11-checkpoint", required=True)
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
