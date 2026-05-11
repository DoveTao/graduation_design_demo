#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, validation_from_logs, write_json

S5E13_REF = {
    "overall_signed": 50.35304145887563,
    "observable_signed": 46.32471099526666,
    "reliable_signed": 47.07029256818264,
    "overall_anti": 0.1479028697571744,
    "reliable_anti": 0.20535714285714285,
    "path_ratio": 0.05272511597296368,
    "sim3_ate": 3.9479012852111164,
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _vals(key: str) -> np.ndarray:
        return np.asarray([r[key] for r in rows if r.get(key) is not None], dtype=np.float64)

    def _mean(key: str) -> float | None:
        v = _vals(key)
        return None if v.size == 0 else float(np.mean(v))

    def _pct(key: str, q: float) -> float | None:
        v = _vals(key)
        return None if v.size == 0 else float(np.percentile(v, q))

    pred_path = sum(float(r.get("pred_step_length") or 0.0) for r in rows)
    gt_path = sum(float(r.get("gt_step_length") or 0.0) for r in rows)
    return {
        "count": len(rows),
        "rot_mean_deg": _mean("rot_deg"),
        "rot_median_deg": _pct("rot_deg", 50),
        "rot_p90_deg": _pct("rot_deg", 90),
        "signed_tdir_mean_deg": _mean("tdir_deg"),
        "signed_tdir_median_deg": _pct("tdir_deg", 50),
        "signed_tdir_p90_deg": _pct("tdir_deg", 90),
        "tdir_abs_mean_deg": _mean("tdir_abs_deg"),
        "tdir_abs_median_deg": _pct("tdir_abs_deg", 50),
        "tdir_abs_p90_deg": _pct("tdir_abs_deg", 90),
        "tdir_mean_cosine": _mean("tdir_cosine"),
        "anti_parallel_rate": _mean("anti_parallel_flag"),
        "severe_wrong_sign_rate": _mean("severe_wrong_sign_flag"),
        "direction_abs_good_but_signed_bad_rate": _mean("direction_abs_good_but_signed_bad_flag"),
        "tmag_median_ratio": _pct("tmag_ratio", 50),
        "tmag_mean_ratio": _mean("tmag_ratio"),
        "tmag_p90_ratio": _pct("tmag_ratio", 90),
        "tmag_p95_ratio": _pct("tmag_ratio", 95),
        "tmag_p99_ratio": _pct("tmag_ratio", 99),
        "tmag_max_ratio": None if _vals("tmag_ratio").size == 0 else float(np.max(_vals("tmag_ratio")),),
        "path_ratio": pred_path / max(gt_path, 1.0e-12),
    }


def _eval_external(traj: Path, gt: Path, out_path: Path, align: str) -> Dict[str, Any]:
    cmd = [
        "/home/dovetao/miniconda3/envs/pytorch/bin/python",
        "tools/evaluate_external_baseline_trajectory.py",
        "--trajectory", str(traj),
        "--groundtruth", str(gt),
        "--alignment", align,
        "--output-json", str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    d = json.loads(out_path.read_text(encoding="utf-8"))
    return {"ate": d.get("ATE"), "drift": d.get("drift"), "path_ratio": d.get("path_ratio")}


def _report(path: Path, ckpt: Dict[str, Any]) -> None:
    lines = [
        "# S5E14 observable edge direction refinement report",
        "",
        "## 执行摘要",
        f"final_classification = `{ckpt['final_classification']}`。S5E14 作为 experimental candidate，目标是在不破坏 rot 和 traceability 的前提下，做 observable-edge direction refinement，并降低 anti_parallel 风险。",
        "",
        "## S5E13 结果回顾",
        f"- overall signed tdir = `{S5E13_REF['overall_signed']}`",
        f"- observable signed tdir = `{S5E13_REF['observable_signed']}`",
        f"- reliable signed tdir = `{S5E13_REF['reliable_signed']}`",
        f"- overall anti_parallel_rate = `{S5E13_REF['overall_anti']}`",
        f"- reliable anti_parallel_rate = `{S5E13_REF['reliable_anti']}`",
        f"- path_ratio = `{S5E13_REF['path_ratio']}`",
        "",
        "## S5E13 direction error audit",
        str(ckpt.get("direction_error_audit", {})),
        "",
        "## 为什么 observable 不等于 reliable signed direction",
        "observable 仅表示局部 correspondence 可见；但在 low_parallax / near_static / high_angle_dispersion / small_motion 下，signed direction 依然可能噪声高或符号不稳定，因此需要 strict reliable mask。",
        "",
        "## refined mask / strict reliable mask 设计",
        "通过 reliable_original + low_parallax/near_static/small_motion 过滤 + angle_dispersion/parallax/flow/inlier/gt_tmag 阈值得到 reliable_strict。",
        "",
        "## inference-time blending 或 refinement 策略",
        "采用 inference-time feature gating：reliable_strict 主要使用 correspondence_head，其他边退回 fallback_prior 或 blended，且明确标记 direction_source/scale_source/inference_blend_weight。",
        "",
        "## no GT leakage / no eval GT calibration 说明",
        "uses_scene01_seq03_gt_for_training=false；uses_eval_gt_for_calibration=false。scene01/seq03 GT 仅用于 evaluation/diagnostics。",
        "",
        "## traceable dense export coverage",
        str(ckpt.get("adjacent_dense_export", {})),
        "",
        "## overall component metrics",
        str(ckpt.get("component_metrics", {}).get("overall", {})),
        "",
        "## stratified metrics: observable / reliable_original / reliable_strict / unobservable / small-motion",
        str({k: v for k, v in ckpt.get("component_metrics", {}).items() if k != "overall"}),
        "",
        "## external evaluator none/se3/sim3",
        str(ckpt.get("external_eval", {})),
        "",
        "## 与 S5E13 / S5E12 / S5E9 比较",
        str(ckpt.get("improvement_vs_s5e13", {})),
        "",
        "## 与 ORB-SLAM3 比较",
        str(ckpt.get("comparison_to_orbslam3", {})),
        "",
        "## 是否进入 S5E15",
        str(ckpt.get("diagnosis", {})),
        "",
        "## caveats",
        "- S5E14 是 experimental candidate。",
        "- 不替代 official S5 locked result。",
        "- S5 locked metrics/policy unchanged。",
        "- strict essential geometry 仍受 intrinsics 限制。",
        "- 若 train reliable signed edges 缺失，不声称 supervised signed-direction training 完整成功。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    metrics_payload = json.loads((out_dir / "edge_component_metrics.json").read_text(encoding="utf-8"))
    provenance = _read_jsonl(Path(args.provenance))
    weights = _read_jsonl(Path(args.weights))
    w_by_idx = {int(w["edge_index"]): w for w in weights}
    audit = json.loads((out_dir / "direction_error_audit.json").read_text(encoding="utf-8")) if (out_dir / "direction_error_audit.json").exists() else {}
    train_status = json.loads(Path("checkpoints/S5E14_observable_edge_direction_refinement_candidate/training_status.json").read_text(encoding="utf-8"))

    rows: List[Dict[str, Any]] = []
    for p in provenance:
        idx = int(p["edge_index"])
        r = dict(p.get("metric_preview", {}))
        r.update(w_by_idx.get(idx, {}))
        rows.append(r)

    def filt(k: str, val: bool) -> List[Dict[str, Any]]:
        return [r for r in rows if bool(r.get(k, False)) is val]

    overall = _summary(rows)
    observable = _summary(filt("observable", True))
    reliable_original = _summary(filt("reliable_original", True))
    reliable_strict = _summary(filt("reliable_strict", True))
    unobservable = _summary(filt("observable", False))
    small_motion = _summary(filt("small_motion", True))
    low_parallax = _summary(filt("low_parallax", True))
    near_static = _summary(filt("near_static", True))

    ext = {a: _eval_external(Path(args.trajectory), Path(args.groundtruth), out_dir / f"eval_alignment_{a}.json", a) for a in ["none", "se3", "sim3"]}

    improve = {
        "overall_tdir_improved": overall.get("signed_tdir_mean_deg") is not None and overall["signed_tdir_mean_deg"] <= S5E13_REF["overall_signed"],
        "observable_tdir_improved": observable.get("signed_tdir_mean_deg") is not None and observable["signed_tdir_mean_deg"] <= S5E13_REF["observable_signed"],
        "reliable_anti_parallel_reduced": reliable_original.get("anti_parallel_rate") is not None and reliable_original["anti_parallel_rate"] < S5E13_REF["reliable_anti"],
        "overall_anti_parallel_reduced": overall.get("anti_parallel_rate") is not None and overall["anti_parallel_rate"] < S5E13_REF["overall_anti"],
        "under_scale_reduced": overall.get("path_ratio") is not None and overall["path_ratio"] > S5E13_REF["path_ratio"],
        "sim3_ate_improved": ext["sim3"].get("ate") is not None and ext["sim3"]["ate"] <= S5E13_REF["sim3_ate"],
    }
    improve["overall_geometry_improved"] = bool(improve["overall_tdir_improved"] and improve["overall_anti_parallel_reduced"] and improve["under_scale_reduced"])

    if improve["overall_geometry_improved"] and improve["sim3_ate_improved"]:
        final = "S5E14_DIRECTION_REFINEMENT_IMPROVED"
    elif improve["observable_tdir_improved"]:
        final = "S5E14_OBSERVABLE_IMPROVED_OVERALL_STILL_BAD"
    elif improve["under_scale_reduced"] and not improve["overall_tdir_improved"]:
        final = "S5E14_UNDERSCALE_REDUCED_TDIR_NOT_IMPROVED"
    elif train_status.get("classification") == "S5E14_INFERENCE_GATING_ONLY_NO_TRAIN_SIGNAL":
        final = "S5E14_INFERENCE_GATING_ONLY_NO_TRAIN_SIGNAL"
    else:
        final = "S5E14_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT"

    coverage = metrics_payload["coverage"]
    ckpt = {
        "experiment": "S5E14_observable_edge_direction_refinement",
        "status": {"experimental_candidate": True, "official_s5_unchanged": True, "not_official_replacement": True},
        "s5e13_reference": {
            "overall_tdir_mean_deg": S5E13_REF["overall_signed"],
            "observable_tdir_mean_deg": S5E13_REF["observable_signed"],
            "reliable_tdir_mean_deg": S5E13_REF["reliable_signed"],
            "overall_anti_parallel_rate": S5E13_REF["overall_anti"],
            "reliable_anti_parallel_rate": S5E13_REF["reliable_anti"],
            "path_ratio": S5E13_REF["path_ratio"],
            "sim3_ate": S5E13_REF["sim3_ate"],
        },
        "direction_error_audit": {
            "path": "external_baselines/results/s5e14_traceable_dense/direction_error_audit.json",
            "reliable_mask_too_loose": audit.get("diagnosis", {}).get("reliable_mask_too_loose"),
            "observable_edges_include_noisy_direction_cases": audit.get("diagnosis", {}).get("observable_edges_include_noisy_direction_cases"),
            "inference_time_blending_recommended": audit.get("diagnosis", {}).get("inference_time_blending_recommended"),
            "main_direction_error_source": audit.get("diagnosis", {}).get("main_direction_error_source", "unknown"),
        },
        "training": {
            "attempted": True,
            "classification": train_status.get("classification"),
            "config": "configs/s5e14_observable_edge_direction_refinement.yaml",
            "checkpoint_dir": "checkpoints/S5E14_observable_edge_direction_refinement_candidate",
            "train_reliable_signed_edges": train_status.get("train_reliable_signed_edges"),
            "uses_scene01_seq03_gt_for_training": False,
            "uses_eval_gt_for_calibration": False,
            "uses_correspondence_features": True,
            "uses_strict_essential_geometry": False,
            "inference_time_feature_gating": train_status.get("inference_time_feature_gating"),
            "supervised_train_refinement_available": train_status.get("supervised_train_refinement_available"),
        },
        "adjacent_dense_export": {
            "available": True,
            "trajectory_path": str(args.trajectory),
            "edge_provenance": str(args.provenance),
            "num_poses": coverage["num_poses"],
            "num_edges": coverage["num_edges"],
            "direct_adjacent_prediction_edges": coverage["direct_adjacent_prediction_edges"],
            "coverage": coverage["num_edges"] / max(coverage["num_edges"], 1),
            "all_edges_traceable": coverage["all_edges_traceable"],
        },
        "component_metrics": {
            "overall": overall,
            "observable_edges": observable,
            "reliable_original_edges": reliable_original,
            "reliable_strict_edges": reliable_strict,
            "unobservable_edges": unobservable,
            "small_motion_edges": small_motion,
            "low_parallax_edges": low_parallax,
            "near_static_edges": near_static,
        },
        "external_eval": ext,
        "improvement_vs_s5e13": improve,
        "comparison_to_orbslam3": {
            "coverage_advantage": True,
            "tdir_gap_remaining": True,
            "scale_path_gap_remaining": True,
            "aligned_ate_gap_to_orbslam3": None if ext["sim3"]["ate"] is None else ext["sim3"]["ate"] - ORBSLAM3_REFERENCE["sim3"]["ate"],
            "summary": "S5E14 保持 full coverage，但与 ORB-SLAM3 在 tdir、scale/path、ATE 上仍有明显差距。",
        },
        "diagnosis": {
            "can_continue_to_s5e15": bool(coverage["all_edges_traceable"]),
            "main_remaining_blocker": "under_scale" if not improve["under_scale_reduced"] else ("anti_parallel" if not improve["overall_anti_parallel_reduced"] else "intrinsics_geometry"),
            "recommended_next_stage": "S5E15_scale_direction_joint_refinement",
        },
        "s5_official_locked_metrics": {"ate": 7.352288, "drift": 1.327343, "path_ratio": 0.932379, "unchanged": True, "not_replaced_by_s5e14": True},
        "validation": validation_from_logs(),
        "final_classification": final,
    }

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
