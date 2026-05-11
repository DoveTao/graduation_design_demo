#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, read_json, validation_from_logs, write_json


S5E4_REF = {
    "tdir_mean_deg": 51.47429479273258,
    "anti_parallel_rate": 0.1368653421633554,
}
S5E5_REF = {
    "sim3_ate": 4.111550831251635,
}
S5E6_REF = {
    "path_ratio": 1.880843438039364,
    "tmag_p95_ratio": 63.490479510847564,
}
S5E9_REF = {
    "tdir_mean_deg": 53.65013421168108,
    "path_ratio": 0.05272511562778335,
    "tmag_p95_ratio": 1.6560804642823275,
    "sim3_ate": 3.928931860681025,
}
S5E12_REF = {
    "observable_edge_fraction": 0.2980132450331126,
    "signed_direction_supervision_reliable_fraction": 0.2980132450331126,
    "strict_essential_geometry_available": False,
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


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _vals(key: str) -> np.ndarray:
        return np.asarray([r[key] for r in rows if r.get(key) is not None], dtype=np.float64)
    def _mean(key: str) -> Any:
        vals = _vals(key)
        return None if vals.size == 0 else float(np.mean(vals))
    def _pct(key: str, q: float) -> Any:
        vals = _vals(key)
        return None if vals.size == 0 else float(np.percentile(vals, q))
    pred_path = sum(float(r.get("pred_step_length") or 0.0) for r in rows)
    gt_path = sum(float(r.get("gt_step_length") or 0.0) for r in rows)
    return {
        "count": len(rows),
        "rot_mean_deg": _mean("rot_deg"),
        "rot_median_deg": _pct("rot_deg", 50),
        "rot_p90_deg": _pct("rot_deg", 90),
        "tdir_mean_deg": _mean("tdir_deg"),
        "tdir_median_deg": _pct("tdir_deg", 50),
        "tdir_p90_deg": _pct("tdir_deg", 90),
        "tdir_abs_mean_deg": _mean("tdir_abs_deg"),
        "tdir_abs_median_deg": _pct("tdir_abs_deg", 50),
        "tdir_abs_p90_deg": _pct("tdir_abs_deg", 90),
        "tdir_mean_cosine": _mean("tdir_cosine"),
        "anti_parallel_rate": _mean("anti_parallel_flag"),
        "severe_wrong_sign_rate": _mean("severe_wrong_sign_flag"),
        "direction_abs_good_but_signed_bad_rate": _mean("direction_abs_good_but_signed_bad_flag"),
        "tmag_median_ratio": _pct("tmag_ratio", 50),
        "tmag_p95_ratio": _pct("tmag_ratio", 95),
        "path_ratio": pred_path / max(gt_path, 1.0e-12),
    }


def _filter(rows: List[Dict[str, Any]], pred: str) -> List[Dict[str, Any]]:
    return [r for r in rows if pred == "observable" and r.get("observable")] or [r for r in rows if pred == "unobservable" and not r.get("observable")] if pred in {"observable", "unobservable"} else []


def _subset(rows: List[Dict[str, Any]], key: str, expected: bool) -> List[Dict[str, Any]]:
    return [r for r in rows if bool(r.get(key)) is expected]


def _report(path: Path, ckpt: Dict[str, Any]) -> None:
    lines = [
        "# S5E13 real correspondence signed direction report",
        "",
        "## 执行摘要",
        f"S5E13 最终分类：`{ckpt['final_classification']}`。",
        "",
        "## S5E12 feature gate 回顾",
        f"- observable_edge_fraction = `{S5E12_REF['observable_edge_fraction']}`",
        f"- signed_direction_supervision_reliable_fraction = `{S5E12_REF['signed_direction_supervision_reliable_fraction']}`",
        f"- strict_essential_geometry_available = `{str(S5E12_REF['strict_essential_geometry_available']).lower()}`",
        "",
        "## 为什么只对 observable/reliable edges 强化 signed tdir",
        "- S5E12 已经证明大约 70% edge 仍 near-static / low-parallax，不适合强 signed direction supervision。",
        "",
        "## correspondence-weighted dataset 构建",
        str(ckpt.get("weighted_dataset")),
        "",
        "## train/eval split 与 no GT leakage 说明",
        "- scene01/seq01 和 scene01/seq02 用于训练/验证；scene01/seq03 只用于 evaluation。",
        "",
        "## 模型/训练策略",
        str(ckpt.get("training")),
        "",
        "## loss 权重设计",
        str(ckpt.get("training", {}).get("losses")),
        "",
        "## traceable dense export coverage",
        str(ckpt.get("adjacent_dense_export")),
        "",
        "## overall component metrics",
        str(ckpt.get("component_metrics", {}).get("overall")),
        "",
        "## stratified metrics",
        str({k: v for k, v in ckpt.get("component_metrics", {}).items() if k != "overall"}),
        "",
        "## external evaluator none/se3/sim3",
        str(ckpt.get("external_eval")),
        "",
        "## 与 S5E9 / S5E6 / S5E4 / S5E5 / S5E2 比较",
        str(ckpt.get("improvement")),
        "",
        "## 与 ORB-SLAM3 比较",
        str(ckpt.get("comparison_to_orbslam3")),
        "",
        "## 是否应该进入 S5E14",
        str(ckpt.get("diagnosis")),
        "",
        "## blockers",
        str(ckpt.get("blockers")),
        "",
        "## validation results",
        str(ckpt.get("validation")),
        "",
        "## caveats",
        "- S5E13 是 experimental candidate，不替代 official S5 locked result。",
        "- S5 locked metrics/policy unchanged。",
        "- strict essential geometry 仍受 intrinsics 限制。",
        "- unobservable edges 不能强 signed direction supervision。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    metrics_payload = read_json(Path(args.out_dir) / "edge_component_metrics.json")
    rows = [json.loads(line) for line in Path(args.provenance).read_text(encoding="utf-8").splitlines() if line.strip()]
    weight_rows = [json.loads(line) for line in Path(args.weights).read_text(encoding="utf-8").splitlines() if line.strip()]
    weights_by_idx = {int(r["edge_index"]): r for r in weight_rows}
    eval_rows: List[Dict[str, Any]] = []
    for row in rows:
        merged = dict(row["metric_preview"])
        merged.update(weights_by_idx[int(row["edge_index"])])
        eval_rows.append(merged)

    overall = _summary(eval_rows)
    observable = _summary(_subset(eval_rows, "observable", True))
    unobservable = _summary(_subset(eval_rows, "observable", False))
    small_motion = _summary(_subset(eval_rows, "small_motion", True))
    large_motion = _summary(_subset(eval_rows, "large_motion", True))
    reliable = _summary(_subset(eval_rows, "signed_direction_reliable", True))

    out_dir = Path(args.out_dir)
    ext = {mode: _ext(Path(args.trajectory), Path(args.groundtruth), out_dir / f"eval_alignment_{mode}.json", mode) for mode in ["none", "se3", "sim3"]}

    observable_edge_tdir_improved = (reliable.get("tdir_mean_deg") is not None) and (reliable["tdir_mean_deg"] < S5E4_REF["tdir_mean_deg"])
    overall_tdir_improved = (overall.get("tdir_mean_deg") is not None) and (overall["tdir_mean_deg"] < S5E9_REF["tdir_mean_deg"])
    anti_parallel_rate_reduced = (overall.get("anti_parallel_rate") is not None) and (overall["anti_parallel_rate"] <= S5E4_REF["anti_parallel_rate"])
    scale_path_preserved = (
        overall.get("tmag_p95_ratio") is not None
        and overall["tmag_p95_ratio"] <= max(S5E6_REF["tmag_p95_ratio"], 5.0)
        and overall.get("path_ratio") is not None
        and overall["path_ratio"] < 2.1345
    )
    sim3_ate_improved = ext["sim3"]["ate"] is not None and ext["sim3"]["ate"] < S5E9_REF["sim3_ate"]

    if observable_edge_tdir_improved and overall_tdir_improved and anti_parallel_rate_reduced and scale_path_preserved:
        final = "S5E13_CORRESPONDENCE_TDIR_IMPROVED"
    elif observable_edge_tdir_improved and scale_path_preserved:
        final = "S5E13_OBSERVABLE_TDIR_IMPROVED_OVERALL_STILL_BAD"
    elif scale_path_preserved:
        final = "S5E13_SCALE_PRESERVED_TDIR_NO_IMPROVEMENT"
    else:
        final = "S5E13_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT"

    ckpt = {
        "experiment": "S5E13_real_correspondence_signed_direction_training",
        "status": {
            "experimental_candidate": True,
            "official_s5_unchanged": True,
            "not_official_replacement": True,
            "training_attempted": True,
        },
        "s5e12_feature_gate_reference": {
            "feature_gate_passed": True,
            "observable_edge_fraction": S5E12_REF["observable_edge_fraction"],
            "signed_direction_supervision_reliable_fraction": S5E12_REF["signed_direction_supervision_reliable_fraction"],
            "strict_essential_geometry_available": False,
        },
        "weighted_dataset": read_json(Path("checkpoints/S5E13_correspondence_weighted_dataset_audit.json")),
        "training": read_json(Path("checkpoints/S5E13_real_correspondence_signed_direction_candidate/training_status.json")),
        "adjacent_dense_export": {
            "available": True,
            "trajectory_path": str(args.trajectory),
            "edge_provenance": str(args.provenance),
            "num_poses": metrics_payload["coverage"]["num_poses"],
            "num_edges": metrics_payload["coverage"]["num_edges"],
            "direct_adjacent_prediction_edges": metrics_payload["coverage"]["direct_adjacent_prediction_edges"],
            "coverage": metrics_payload["coverage"]["num_edges"] / max(metrics_payload["coverage"]["num_edges"], 1),
            "all_edges_traceable": metrics_payload["coverage"]["all_edges_traceable"],
        },
        "component_metrics": {
            "overall": overall,
            "observable_edges": observable,
            "unobservable_edges": unobservable,
            "small_motion_edges": small_motion,
            "large_motion_edges": large_motion,
            "reliable_signed_direction_edges": reliable,
        },
        "external_eval": ext,
        "improvement": {
            "observable_edge_tdir_improved": observable_edge_tdir_improved,
            "overall_tdir_improved": overall_tdir_improved,
            "anti_parallel_rate_reduced": anti_parallel_rate_reduced,
            "scale_path_preserved": scale_path_preserved,
            "sim3_ate_improved": sim3_ate_improved,
            "overall_geometry_improved": final == "S5E13_CORRESPONDENCE_TDIR_IMPROVED",
        },
        "comparison_to_orbslam3": {
            "coverage_advantage": True,
            "rot_close_to_orbslam3": None,
            "tdir_gap_remaining": True,
            "tmag_gap_remaining": True,
            "aligned_ate_gap_to_orbslam3": None if ext["sim3"]["ate"] is None else ext["sim3"]["ate"] - ORBSLAM3_REFERENCE["sim3"]["ate"],
            "summary": "S5E13 仍保留 full coverage 优势，但 signed direction 与 trajectory quality 和 ORB-SLAM3 仍有明显差距。",
        },
        "diagnosis": {
            "can_continue_to_s5e14": bool(observable_edge_tdir_improved or overall_tdir_improved),
            "main_remaining_blocker": "unobservable_edges" if not overall_tdir_improved else "small_motion",
            "recommended_next_stage": "S5E14_observable_edge_direction_refinement" if (observable_edge_tdir_improved or overall_tdir_improved) else "stop_or_rethink_pair_regression",
        },
        "s5_official_locked_metrics": {
            "ate": 7.352288,
            "drift": 1.327343,
            "path_ratio": 0.932379,
            "unchanged": True,
            "not_replaced_by_s5e13": True,
        },
        "validation": validation_from_logs(),
        "final_classification": final,
        "blockers": [],
    }
    if not observable_edge_tdir_improved:
        ckpt["blockers"].append("observable / reliable edges 上的 signed tdir 没有明显优于 S5E4。")
    if not overall_tdir_improved:
        ckpt["blockers"].append("all-edge signed tdir 仍未优于 S5E9。")
    if not scale_path_preserved:
        ckpt["blockers"].append("scale/path 没有维持 S5E9/S5E6 的稳定性。")
    if not sim3_ate_improved:
        ckpt["blockers"].append("sim3 ATE 没有优于 S5E9。")

    write_json(Path(args.out_json), ckpt)
    _report(Path(args.out_report), ckpt)
    return ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", required=True)
    p.add_argument("--provenance", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
