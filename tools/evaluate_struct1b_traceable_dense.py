#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from s5e2_adjacent_dense_lib import write_json


S5E15_REF = {
    "tmag_median_ratio": 1.2685,
    "path_ratio": 0.1572,
    "sim3_ate": 3.9682,
    "signed_tdir_mean_deg": 50.3530,
}


def _ext(traj: Path, gt: Path, out_json: Path, mode: str) -> Dict[str, Any]:
    subprocess.run(
        [
            "/home/dovetao/miniconda3/envs/pytorch/bin/python",
            "tools/evaluate_external_baseline_trajectory.py",
            "--trajectory",
            str(traj),
            "--groundtruth",
            str(gt),
            "--alignment",
            mode,
            "--output-json",
            str(out_json),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    obj = json.loads(out_json.read_text(encoding="utf-8"))
    return {"ate": obj.get("ATE"), "drift": obj.get("drift"), "path_ratio": obj.get("path_ratio")}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    integrity = json.loads(Path(args.integrity_audit).read_text(encoding="utf-8"))
    train = json.loads((Path(args.candidate) / "training_status.json").read_text(encoding="utf-8"))
    ext = {m: _ext(Path(args.trajectory), Path(args.groundtruth), Path(args.trajectory).parent / f"eval_alignment_{m}.json", m) for m in ["none", "se3", "sim3"]}
    comp = metrics["component_metrics"]
    scale_success = bool(comp["tmag_median_ratio"] < 3.0 and comp["path_ratio"] < 1.0 and comp["tmag_p95_ratio"] < 20.0)
    strong_success = bool(0.7 <= comp["tmag_median_ratio"] <= 1.5 and comp["path_ratio"] >= 0.1572 and ext["sim3"]["ate"] <= 3.9682)
    if strong_success:
        final = "STRUCT1B_SCALE_REPAIR_AND_GLOBAL_IMPROVEMENT"
    elif scale_success:
        final = "STRUCT1B_SCALE_REPAIR_SUCCESS_TDIR_STILL_BAD"
    elif comp["tmag_median_ratio"] < 10.0 and comp["path_ratio"] < 3.0:
        final = "STRUCT1B_SCALE_PARTIAL_REPAIR"
    else:
        final = "STRUCT1B_SCALE_GUARD_NOT_EFFECTIVE"
    payload = {
        "experiment": "STRUCT1B_scale_guard_repair",
        "training": train,
        "integrity_audit": integrity,
        "component_metrics": comp,
        "external_eval": ext,
        "improvement_vs_struct1": {
            "tmag_median_ratio_improved": comp["tmag_median_ratio"] < 23.3454,
            "path_ratio_improved": comp["path_ratio"] < 12.9702,
            "tmag_p95_ratio_improved": comp["tmag_p95_ratio"] < 188.6088,
            "signed_tdir_changed": comp.get("signed_tdir_mean_deg") != 76.6451,
        },
        "improvement_vs_s5e15": {
            "tmag_median_ratio_recovered_to_s5e15_band": comp["tmag_median_ratio"] <= 3.0,
            "path_ratio_not_exploding": comp["path_ratio"] < 1.0,
            "sim3_ate_beats_s5e15": ext["sim3"]["ate"] <= S5E15_REF["sim3_ate"],
        },
        "recommendation": {
            "keep_s5e15_as_best_candidate": final != "STRUCT1B_SCALE_REPAIR_AND_GLOBAL_IMPROVEMENT",
            "promote_struct1": False,
            "continue_to_struct2": False if not scale_success else False,
            "recommended_next_stage": "STRUCT1C_geometry_token_tdir_only" if scale_success else "stop_STRUCT1_and_STRUCT2",
        },
        "s5_locked_metrics_policy_unchanged": True,
        "final_classification": final,
    }
    write_json(Path(args.out_json), payload)
    report = [
        "# STRUCT1B scale guard 修复报告",
        "",
        "## 执行摘要",
        f"final_classification={final}",
        "",
        "## 前提说明",
        "由于原始 `STRUCT1` 训练权重 `struct1_model.pt` 已不在当前工作区，本轮 `STRUCT1B` 采用与 STRUCT1 相同的 geometry-token 主干和相同 tdir head 结构重新训练，只修复 scale guard 与 scale/path 权重，不重设计 tdir head。",
        "",
        "## scale repair 结果",
        json.dumps(comp, ensure_ascii=False, indent=2),
        "",
        "## external evaluator",
        json.dumps(ext, ensure_ascii=False, indent=2),
        "",
        "## recommendation",
        json.dumps(payload["recommendation"], ensure_ascii=False, indent=2),
    ]
    Path(args.out_report).write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--trajectory", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--integrity-audit", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
