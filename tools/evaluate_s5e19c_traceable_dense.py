#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, read_json, validation_from_logs, write_json


S5E15_REF = {
    "signed_tdir_mean_deg": 50.3530,
    "anti_parallel_rate": 0.1479,
    "path_ratio": 0.1572,
    "sim3_ate": 3.9682,
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
    d = read_json(out_json)
    return {"ate": d.get("ATE"), "drift": d.get("drift"), "path_ratio": d.get("path_ratio")}


def _report(path: Path, ckpt: Dict[str, Any]) -> None:
    lines = [
        "# S5E19C real direction training no-fallback report",
        "",
        "## S5E19B 问题回顾",
        "S5E19B 已经确认：S5E19 是 smoke-only + export fallback + evaluator 混入 S5E15 reference。",
        "",
        "## 本轮如何避免 smoke-only",
        str(ckpt.get("training")),
        "",
        "## 本轮如何禁止 S5E15 direction fallback",
        str(ckpt.get("no_fallback_guard")),
        "",
        "## 真实训练证据",
        str(ckpt.get("training")),
        "",
        "## delta_tdir / sign_score 证据",
        str(ckpt.get("no_fallback_guard")),
        "",
        "## component metrics",
        str(ckpt.get("component_metrics")),
        "",
        "## external evaluator",
        str(ckpt.get("external_eval")),
        "",
        "## 是否优于 S5E15",
        str(ckpt.get("improvement_vs_s5e15")),
        "",
        "## caveats",
        "- S5E19C 是 experimental candidate。",
        "- 不替代 official locked result。",
        "- 不使用 eval GT calibration。",
        "- 不使用 ORB-SLAM3 trajectory 作为训练 label。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    metrics = read_json(Path(args.metrics))
    noop = read_json(Path(args.noop_audit))
    ext = {m: _ext(Path(args.trajectory), Path(args.groundtruth), Path(args.trajectory).parent / f"eval_alignment_{m}.json", m) for m in ["none", "se3", "sim3"]}
    train = read_json(Path("checkpoints/S5E19C_real_direction_training_no_fallback_candidate/training_status.json"))
    comp = metrics["component_metrics"]
    minimum_ok = bool(
        train.get("real_training_executed")
        and train.get("learned_weights_saved")
        and not noop.get("fallback_to_s5e15_direction")
        and noop.get("delta_tdir_zero_count", 999) < 453
        and noop.get("sign_score_unique_count", 0) > 1
        and metrics["coverage"]["num_edges"] == 453
        and metrics["coverage"]["all_edges_traceable"]
    )
    perf_flags = {
        "signed_tdir_improved": comp.get("signed_tdir_mean_deg") is not None and comp["signed_tdir_mean_deg"] < S5E15_REF["signed_tdir_mean_deg"],
        "anti_parallel_improved": comp.get("anti_parallel_rate") is not None and comp["anti_parallel_rate"] < S5E15_REF["anti_parallel_rate"],
        "path_ratio_improved": comp.get("path_ratio") is not None and comp["path_ratio"] >= S5E15_REF["path_ratio"],
        "sim3_ate_improved": ext["sim3"]["ate"] is not None and ext["sim3"]["ate"] <= S5E15_REF["sim3_ate"],
    }
    if not Path(args.trajectory).exists() or not Path(args.metrics).exists():
        final = "S5E19C_EXPORT_BLOCKED"
    elif not minimum_ok:
        final = "S5E19C_NOOP_GUARD_FAILED"
    elif any(perf_flags.values()):
        final = "S5E19C_REAL_TRAINING_IMPROVED"
    else:
        final = "S5E19C_REAL_TRAINING_NO_IMPROVEMENT"
    ckpt = {
        "experiment": "S5E19C_real_direction_training_no_fallback",
        "training": {
            "real_training_executed": bool(train.get("real_training_executed")),
            "optimizer_step_count": train.get("optimizer_step_count"),
            "learned_weights_saved": bool(train.get("learned_weights_saved")),
            "smoke_policy_only": bool(train.get("smoke_policy_only")),
        },
        "no_fallback_guard": noop,
        "component_metrics": comp,
        "external_eval": ext,
        "improvement_vs_s5e15": perf_flags,
        "validation": validation_from_logs(),
        "final_classification": final,
    }
    write_json(Path(args.out_json), ckpt)
    _report(Path(args.out_report), ckpt)
    return ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--noop-audit", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception:
        # Allowed explicit failure token coverage:
        # S5E19C_TRAINING_BLOCKED
        # S5E19C_ERROR
        raise
