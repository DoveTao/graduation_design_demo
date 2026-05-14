#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from s5e2_adjacent_dense_lib import read_json, write_json


def _row(name: str, ckpt: Dict[str, Any]) -> Dict[str, Any]:
    raw = ckpt.get("raw_prediction_metrics") or ckpt.get("raw_component_metrics") or {}
    ext = ckpt.get("external_eval", {})
    se3 = ext.get("se3", {})
    sim3 = ext.get("sim3", {})
    return {
        "name": name,
        "classification": ckpt.get("final_classification"),
        "coverage": (ckpt.get("adjacent_dense_export") or {}).get("coverage"),
        "signed_tdir_mean_deg": raw.get("tdir_mean_deg"),
        "tdir_abs_mean_deg": raw.get("tdir_abs_mean_deg"),
        "anti_parallel_rate": raw.get("anti_parallel_rate"),
        "tmag_p95_ratio": raw.get("tmag_p95_ratio"),
        "path_ratio": raw.get("path_ratio"),
        "se3_ate": se3.get("ate"),
        "sim3_ate": sim3.get("ate"),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rows = {
        "S5E11": _row("S5E11", read_json(Path(args.s5e11_checkpoint))),
        "S5E10": _row("S5E10", read_json(Path(args.s5e10_checkpoint))),
        "S5E9": _row("S5E9", read_json(Path(args.s5e9_checkpoint))),
        "S5E8": _row("S5E8", read_json(Path(args.s5e8_checkpoint))),
        "S5E7": _row("S5E7", read_json(Path(args.s5e7_checkpoint))),
        "S5E6": _row("S5E6", read_json(Path(args.s5e6_checkpoint))),
        "S5E5": _row("S5E5", read_json(Path(args.s5e5_checkpoint))),
        "S5E4": _row("S5E4", read_json(Path(args.s5e4_checkpoint))),
        "S5E3": _row("S5E3", read_json(Path(args.s5e3_checkpoint))),
        "S5E2": _row("S5E2", read_json(Path(args.s5e2_checkpoint))),
        "ORB-SLAM3": {
            "coverage": "273/454",
            "se3_ate": 0.30854441069248173,
            "sim3_ate": 0.224292165986624,
            "path_ratio": 0.2998258665660257,
            "signed_tdir_mean_deg": "unavailable",
            "tdir_abs_mean_deg": "unavailable",
            "anti_parallel_rate": "unavailable",
            "tmag_p95_ratio": "unavailable",
        },
    }
    summary = {
        "experiment": "S5E11_correspondence_parallax_translation_geometry",
        "rows": rows,
        "comparison_summary": "S5E11 重点看显式 correspondence/parallax/observability 特征能否在保持 S5E9 scale 稳定的前提下改善 raw signed direction。",
    }
    write_json(Path(args.out_json), summary)
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(
        "# S5E11 comparison\n\n"
        "## 对比摘要\n"
        "S5E11 与 S5E10/S5E9 的核心区别，在于是否能通过显式几何特征而不是继续堆 pair-regression head，来改善 signed direction。\n\n"
        f"## rows\n{rows}\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
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
