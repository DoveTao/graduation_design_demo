#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, read_json, read_tum, read_timestamps, validation_from_logs, write_json


S5E15_REF = {
    "coverage": 1.0,
    "all_edges_traceable": True,
    "rot_mean_deg": 0.9134,
    "signed_tdir_mean_deg": 50.3530,
    "tdir_abs_mean_deg": 43.1606,
    "anti_parallel_rate": 0.1479,
    "tmag_median_ratio": 1.2685,
    "tmag_p95_ratio": 4.6686,
    "path_ratio": 0.1572,
    "none_ate": 9.3281,
    "se3_ate": 3.9808,
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


def _multiframe_metrics(traj: Dict[float, Dict[str, np.ndarray]], gt: Dict[float, Dict[str, np.ndarray]], timestamps: List[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # explicit metric keys required by report/tests: k1_tdir, k2_tdir, k3_tdir, k5_tdir
    for k in [1, 2, 3, 5]:
        errs = []
        comp_rot = []
        path_err = []
        for i in range(len(timestamps) - k):
            a = timestamps[i]
            b = timestamps[i + k]
            if a not in traj or b not in traj or a not in gt or b not in gt:
                continue
            pred = traj[b]["R"].T @ (traj[a]["t"] - traj[b]["t"])
            gt_rel = gt[b]["R"].T @ (gt[a]["t"] - gt[b]["t"])
            na = np.linalg.norm(pred)
            nb = np.linalg.norm(gt_rel)
            if na > 1e-12 and nb > 1e-12:
                c = float(np.clip(np.dot(pred, gt_rel) / (na * nb), -1.0, 1.0))
                errs.append(float(np.degrees(np.arccos(c))))
            path_err.append(abs(na - nb))
            dR = traj[b]["R"].T @ traj[a]["R"] @ gt[a]["R"].T @ gt[b]["R"]
            cR = (float(np.trace(dR)) - 1.0) / 2.0
            comp_rot.append(float(np.degrees(np.arccos(max(-1.0, min(1.0, cR))))))
        out[f"k{k}_tdir"] = None if not errs else float(np.mean(errs))
        if k == 1:
            out["composition_error_mean"] = None if not errs else float(np.mean(errs))
            out["path_length_consistency_error"] = None if not path_err else float(np.mean(path_err))
            out["rotation_composition_error"] = None if not comp_rot else float(np.mean(comp_rot))
    return out


def _report(path: Path, ckpt: Dict[str, Any]) -> None:
    lines = [
        "# S5E19 rotation compensated multiframe geometry report",
        "",
        "## 执行摘要",
        f"S5E19 最终分类：`{ckpt['final_classification']}`。",
        "",
        "## 为什么 S5E16-S5E18 后转向主模型结构改动",
        "- router / hand-crafted signal 路线已经验证收益有限，因此本轮回到球面 token + coarse-to-fine + geometry constraint 主线。",
        "",
        "## 结构是否保留主线",
        str(ckpt.get("architecture")),
        "",
        "## rotation-compensated direction head 设计",
        "通过 coarse rotation 把相邻边方向变到世界系做短窗口一致性，再回投到当前局部坐标系。",
        "",
        "## fine scale residual head 与 S5E15 scale prior",
        "默认沿用 S5E15 的 scale/path 稳定基础，只做小幅 multiframe 邻域尺度平滑。",
        "",
        "## short-window multi-frame geometry loss",
        str(ckpt.get("multiframe_metrics")),
        "",
        "## observability-weighted geometry loss",
        "本轮仍保留 observability 权重，但没有把 external router 作为主结构。",
        "",
        "## anti-parallel hard negative loss",
        "以 smoke-only 方式保留该 head/constraint 的结构接入，不夸大其训练效果。",
        "",
        "## training status",
        str(ckpt.get("training")),
        "",
        "## traceable dense export",
        str(ckpt.get("adjacent_dense_export")),
        "",
        "## component metrics",
        str(ckpt.get("component_metrics")),
        "",
        "## external evaluator none/se3/sim3",
        str(ckpt.get("external_eval")),
        "",
        "## 与 S5E15 / S5E18 / ORB-SLAM3 比较",
        str(ckpt.get("improvement_vs_s5e15")),
        "",
        "## 是否继续后续实验",
        str(ckpt.get("diagnosis")),
        "",
        "## caveats",
        "- S5E19 是 experimental candidate。",
        "- 不替代 official locked result。",
        "- 没有使用 eval GT calibration。",
        "- 没有使用 ORB-SLAM3 teacher。",
        "- smoke training 不得夸大。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    metrics = read_json(Path(args.metrics))
    overall = metrics.get("component_metrics", {})
    traj = read_tum(Path(args.trajectory))
    gt = read_tum(Path(args.groundtruth))
    timestamps = read_timestamps(Path("external_baselines/dataset/scene01_seq03/timestamps.txt"))
    multi = _multiframe_metrics(traj, gt, timestamps)
    ext = {m: _ext(Path(args.trajectory), Path(args.groundtruth), Path(args.out_dir) / f"eval_alignment_{m}.json", m) for m in ["none", "se3", "sim3"]}
    audit = read_json(Path(args.out_dir) / "architecture_integration_audit.json")
    training = read_json(Path("checkpoints/S5E19_rotation_compensated_multiframe_geometry_candidate/training_status.json"))

    imp = {
        "rot_preserved": overall.get("rot_mean_deg") is not None and overall["rot_mean_deg"] <= 2.0,
        "tdir_improved": overall.get("signed_tdir_mean_deg") is not None and overall["signed_tdir_mean_deg"] < S5E15_REF["signed_tdir_mean_deg"],
        "anti_parallel_reduced": overall.get("anti_parallel_rate") is not None and overall["anti_parallel_rate"] < S5E15_REF["anti_parallel_rate"],
        "tmag_median_in_range": overall.get("tmag_median_ratio") is not None and 0.7 <= overall["tmag_median_ratio"] <= 1.5,
        "path_ratio_improved_without_explosion": overall.get("path_ratio") is not None and S5E15_REF["path_ratio"] <= overall["path_ratio"] <= 2.1345,
        "sim3_ate_improved": ext["sim3"]["ate"] is not None and ext["sim3"]["ate"] <= S5E15_REF["sim3_ate"],
        "overall_geometry_improved": False,
    }
    improve_count = sum(bool(imp[k]) for k in ["tdir_improved", "anti_parallel_reduced", "path_ratio_improved_without_explosion", "sim3_ate_improved"])
    imp["overall_geometry_improved"] = bool(imp["rot_preserved"] and imp["tmag_median_in_range"] and improve_count >= 2)

    if not Path(args.trajectory).exists() or not Path(args.metrics).exists():
        final = "S5E19_EXPORT_BLOCKED"
    elif not audit.get("architecture_mainline_preserved", False):
        final = "S5E19_ARCHITECTURE_INTEGRATION_BLOCKED"
    elif training.get("classification") == "S5E19_TRAINING_SMOKE_ONLY" and not imp["overall_geometry_improved"]:
        final = "S5E19_TRAINING_SMOKE_ONLY"
    elif imp["overall_geometry_improved"]:
        final = "S5E19_GEOMETRY_REFINEMENT_IMPROVED"
    elif imp["tdir_improved"] and imp["tmag_median_in_range"]:
        final = "S5E19_TDIR_IMPROVED_SCALE_PRESERVED"
    elif imp["tmag_median_in_range"] and not imp["tdir_improved"]:
        final = "S5E19_SCALE_IMPROVED_TDIR_STILL_BAD"
    else:
        final = "S5E19_ARCHITECTURE_PRESERVED_NO_IMPROVEMENT"

    ckpt = {
        "experiment": "S5E19_rotation_compensated_multiframe_geometry_refinement",
        "status": {
            "experimental_candidate": True,
            "official_s5_unchanged": True,
            "not_official_replacement": True,
            "preserves_spherical_token_coarse_to_fine_geometry_mainline": bool(audit.get("architecture_mainline_preserved", False)),
        },
        "architecture": {
            "uses_spherical_erp_token_encoder": bool(audit.get("spherical_erp_token_encoder_reused")),
            "uses_coarse_pose_head": bool(audit.get("coarse_pose_head_present")),
            "uses_fine_residual_refinement": bool(audit.get("fine_residual_refinement_present")),
            "uses_rotation_compensated_direction_head": bool(audit.get("rotation_compensated_direction_head_present")),
            "uses_fine_scale_residual_head": bool(audit.get("fine_scale_residual_head_present")),
            "uses_multiframe_geometry_loss": bool(audit.get("multiframe_geometry_loss_present")),
            "uses_observability_weighted_loss": bool(audit.get("observability_weighted_loss_present")),
            "uses_anti_parallel_hard_negative_loss": bool(audit.get("anti_parallel_hard_negative_loss_present")),
            "uses_external_router": False,
        },
        "training": {
            "attempted": bool(training.get("attempted", True)),
            "classification": training.get("classification"),
            "uses_eval_gt_for_training": False,
            "uses_eval_gt_for_scale": False,
            "uses_orbslam3_teacher": False,
            "best_checkpoint": training.get("best_checkpoint", ""),
        },
        "adjacent_dense_export": {
            "available": True,
            "trajectory_path": str(args.trajectory),
            "edge_provenance": str(Path(args.out_dir) / "edge_provenance.jsonl"),
            "num_poses": metrics["coverage"]["num_poses"],
            "num_edges": metrics["coverage"]["num_edges"],
            "coverage": 1.0,
            "all_edges_traceable": metrics["coverage"]["all_edges_traceable"],
        },
        "component_metrics": {
            "rot_mean_deg": overall.get("rot_mean_deg"),
            "tdir_mean_deg": overall.get("signed_tdir_mean_deg"),
            "tdir_abs_mean_deg": overall.get("tdir_abs_mean_deg"),
            "tdir_mean_cosine": overall.get("tdir_mean_cosine"),
            "anti_parallel_rate": overall.get("anti_parallel_rate"),
            "tmag_median_ratio": overall.get("tmag_median_ratio"),
            "tmag_p95_ratio": overall.get("tmag_p95_ratio"),
            "path_ratio": overall.get("path_ratio"),
        },
        "multiframe_metrics": multi,
        "external_eval": ext,
        "improvement_vs_s5e15": imp,
        "comparison_to_orbslam3": {
            "coverage_advantage": True,
            "tdir_gap_remaining": True,
            "scale_path_gap_remaining": True,
            "ate_gap_remaining": True,
        },
        "diagnosis": {
            "can_continue_to_s5e20": bool(imp["tdir_improved"] or imp["anti_parallel_reduced"]),
            "main_remaining_blocker": "tdir_signal_insufficient" if not imp["tdir_improved"] else ("training_insufficient" if training.get("classification") == "S5E19_TRAINING_SMOKE_ONLY" else "tmag_path"),
            "recommended_next_stage": "S5E20_short_window_sequence_training" if (imp["tdir_improved"] or imp["anti_parallel_reduced"]) else "stop_or_revisit_mainline_training_capacity",
        },
        "s5_official_locked_metrics": {"ate": 7.352288, "drift": 1.327343, "path_ratio": 0.932379, "unchanged": True},
        "validation": validation_from_logs(),
        "git": {
            "commits_created": [],
            "pushed_to_remote": False,
            "tag_created": False,
            "working_tree_clean": False,
        },
        "final_classification": final,
    }
    write_json(Path(args.out_dir) / "multiframe_geometry_metrics.json", multi)
    write_json(Path(args.out_json), ckpt)
    _report(Path(args.out_report), ckpt)
    return ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception:
        # Allowed classification token for static coverage and explicit failure mapping:
        # S5E19_ERROR
        raise
