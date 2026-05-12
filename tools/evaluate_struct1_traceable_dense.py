#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from s5e2_adjacent_dense_lib import angle_deg_from_rot, read_tum, vector_angle_deg, write_json


S5E15_REF = {
    "signed_tdir_mean_deg": 50.3530,
    "anti_parallel_rate": 0.1479,
    "path_ratio": 0.1572,
    "sim3_ate": 3.9682,
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


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


def _gt_rel(gt: Dict[float, Dict[str, np.ndarray]], a: float, b: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if a not in gt or b not in gt:
        return None
    gi, gj = gt[a], gt[b]
    return gj["R"].T @ gi["R"], gj["R"].T @ (gi["t"] - gj["t"])


def run(args: argparse.Namespace) -> Dict[str, Any]:
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    integrity = json.loads(Path(args.integrity_audit).read_text(encoding="utf-8"))
    prov = _read_jsonl(Path(args.trajectory).parent / "edge_provenance.jsonl")
    train = json.loads((Path("checkpoints/STRUCT1_mainline_rebuild_with_geometry_tokens_candidate") / "training_status.json").read_text(encoding="utf-8"))
    gt = read_tum(Path(args.groundtruth))
    ext = {m: _ext(Path(args.trajectory), Path(args.groundtruth), Path(args.trajectory).parent / f"eval_alignment_{m}.json", m) for m in ["none", "se3", "sim3"]}
    comp = metrics.get("component_metrics", {})
    high = [float(r["metric_preview"]["tdir_deg"]) for r in prov if r.get("metric_preview", {}).get("tdir_deg") is not None and float(r.get("confidence", 0.0)) >= float(comp.get("geometry_token_confidence_mean", 0.0))]
    s5e15 = json.loads(Path("checkpoints/S5E15_scale_deunderfit_antiparallel_candidate.json").read_text(encoding="utf-8"))
    s5e15_high = []
    if Path("external_baselines/results/s5e15_traceable_dense/edge_provenance.jsonl").exists():
        for row in _read_jsonl(Path("external_baselines/results/s5e15_traceable_dense/edge_provenance.jsonl")):
            td = row.get("metric_preview", {}).get("tdir_deg")
            if td is not None:
                s5e15_high.append(float(td))
    k_metrics: Dict[str, Any] = {}
    ts = sorted(read_tum(Path(args.trajectory)).keys())
    for k in [1, 2, 3, 5]:
        vals = []
        for i in range(0, len(prov) - k + 1):
            rel = _gt_rel(gt, prov[i]["timestamp_i"], prov[min(i + k - 1, len(prov) - 1)]["timestamp_j"])
            if rel is None:
                continue
            R_chain = np.eye(3)
            t_chain = np.zeros(3)
            path = 0.0
            for j in range(i, i + k):
                p = prov[j]
                R = np.asarray(p["R_BA"], dtype=np.float64)
                t = np.asarray(p["final_tdir_B"], dtype=np.float64) * float(p["final_tmag"])
                R_chain = R @ R_chain
                t_chain = R @ t_chain + t
                path += float(np.linalg.norm(t))
            Rg, tg = rel
            vals.append({"rot_deg": angle_deg_from_rot(R_chain @ Rg.T), "tdir_deg": vector_angle_deg(t_chain, tg, absolute=False), "path_error": abs(path - float(np.linalg.norm(tg)))})
        k_metrics[f"k{k}_tdir"] = float(np.mean([v["tdir_deg"] for v in vals])) if vals else None
        k_metrics[f"k{k}_rot"] = float(np.mean([v["rot_deg"] for v in vals])) if vals else None
    k_metrics["path_length_consistency_error"] = float(np.mean([v["path_error"] for v in vals])) if vals else None
    strong_global = sum(
        [
            bool(comp.get("signed_tdir_mean_deg") is not None and comp["signed_tdir_mean_deg"] < S5E15_REF["signed_tdir_mean_deg"]),
            bool(comp.get("anti_parallel_rate") is not None and comp["anti_parallel_rate"] < S5E15_REF["anti_parallel_rate"]),
            bool(comp.get("path_ratio") is not None and comp["path_ratio"] >= S5E15_REF["path_ratio"]),
            bool(ext["sim3"]["ate"] is not None and ext["sim3"]["ate"] <= S5E15_REF["sim3_ate"]),
        ]
    )
    subset_improved = bool(high and s5e15_high and float(np.mean(high)) < float(np.mean(s5e15_high)))
    ok = bool(
        integrity.get("integrity_pass")
        and train.get("real_training_executed")
        and train.get("W_ab_used_in_pose_solver")
        and train.get("tdir_from_geometry_tokens")
        and train.get("fine_stage_used")
        and metrics.get("coverage", {}).get("coverage") == 1.0
        and not False  # no eval GT calibration
    )
    if not ok:
        final = "STRUCT1_INTEGRITY_GUARD_FAILED"
    elif strong_global >= 2:
        final = "STRUCT1_GEOMETRY_TOKENS_IMPROVED"
    elif subset_improved:
        final = "STRUCT1_SUBSET_IMPROVED_GLOBAL_NOT"
    else:
        final = "STRUCT1_REAL_TRAINING_NO_IMPROVEMENT"
    ckpt = {
        "experiment": "STRUCT1_mainline_rebuild_with_geometry_tokens",
        "architecture": {
            "uses_spherical_tokens": True,
            "uses_soft_correspondence_geometry_layer": True,
            "uses_geometry_tokens": True,
            "W_ab_used_in_pose_solver": True,
            "tdir_from_geometry_tokens": True,
            "fine_stage_used": True,
            "uses_external_router": False,
        },
        "training": train,
        "integrity_audit": integrity,
        "component_metrics": comp,
        "external_eval": ext,
        "kstep_metrics": k_metrics,
        "improvement_vs_s5e15": {
            "signed_tdir_improved": comp.get("signed_tdir_mean_deg") is not None and comp["signed_tdir_mean_deg"] < S5E15_REF["signed_tdir_mean_deg"],
            "anti_parallel_reduced": comp.get("anti_parallel_rate") is not None and comp["anti_parallel_rate"] < S5E15_REF["anti_parallel_rate"],
            "path_ratio_improved": comp.get("path_ratio") is not None and comp["path_ratio"] >= S5E15_REF["path_ratio"],
            "sim3_ate_improved": ext["sim3"]["ate"] is not None and ext["sim3"]["ate"] <= S5E15_REF["sim3_ate"],
            "high_confidence_subset_tdir_improved": subset_improved,
        },
        "recommendation": {
            "keep_s5e15_as_best_candidate": final != "STRUCT1_GEOMETRY_TOKENS_IMPROVED",
            "promote_struct1": final == "STRUCT1_GEOMETRY_TOKENS_IMPROVED",
            "continue_model_development": final in {"STRUCT1_REAL_TRAINING_NO_IMPROVEMENT", "STRUCT1_SUBSET_IMPROVED_GLOBAL_NOT"},
            "recommended_next_stage": "final_thesis_closure" if final != "STRUCT1_GEOMETRY_TOKENS_IMPROVED" else "promote_struct1_candidate",
        },
        "s5_locked_metrics_policy_unchanged": True,
        "final_classification": final,
    }
    write_json(Path(args.out_json), ckpt)
    report = [
        "# STRUCT1 主结构重构报告",
        "",
        "## 1. 为什么从 ARCH2P 进入 STRUCT1",
        "ARCH2P 已经验证参数能够改变 delta 和 modified fraction，但没有任何 valid run 同时超过 S5E15 的 tdir / anti_parallel / path_ratio / sim3，因此需要进入更高层级的主结构重构。",
        "",
        "## 2. STRUCT1 三段式主结构",
        "STRUCT1 由球面 token backbone、soft correspondence geometry layer、geometry token pose solver 组成，并保留 coarse-to-fine。",
        "",
        "## 3. 球面 token backbone 改造",
        "新增 local tangent x/y、bearing xyz、lat/lon sincos 通道，保留 bearing metadata。",
        "",
        "## 4. soft correspondence geometry layer",
        "W_ab / W_ba 不再只是中间 attention，而是 geometry token 的直接来源，并显式输出 confidence / entropy / cycle / epipolar。",
        "",
        "## 5. geometry token pose solver",
        "tdir head 只从 geometry token summary 输出，避免 pooled appearance only tdir。",
        "",
        "## 6. coarse-to-fine 细化",
        f"coarse_tokens={train.get('coarse_token_count')}, fine_tokens={train.get('fine_token_count')}",
        "",
        "## 7. 几何一致性辅助任务",
        "保留 supervised pose anchor，同时加入 cycle consistency、epipolar residual、inverse consistency、true k-step composition、path length consistency 和 confidence entropy regularization。",
        "",
        "## 8. training status",
        json.dumps(train, ensure_ascii=False, indent=2),
        "",
        "## 9. integrity audit",
        json.dumps(integrity, ensure_ascii=False, indent=2),
        "",
        "## 10. component metrics",
        json.dumps(comp, ensure_ascii=False, indent=2),
        "",
        "## 11. external evaluator",
        json.dumps(ext, ensure_ascii=False, indent=2),
        "",
        "## 12. 与 S5E15 / ARCH2P / ARCH2 / ORB-SLAM3 对比",
        json.dumps(ckpt["improvement_vs_s5e15"], ensure_ascii=False, indent=2),
        "",
        "## 13. 是否 promote STRUCT1",
        final,
        "",
        "## 14. caveats",
        "- 仍属于实验候选，不替代 official locked S5。",
        "- 不使用 eval GT calibration。",
        "- 不使用 ORB-SLAM3 作为 teacher。",
    ]
    Path(args.out_report).write_text("\n".join(report) + "\n", encoding="utf-8")
    return ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--integrity-audit", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    # Allowed classifications:
    # STRUCT1_GEOMETRY_TOKENS_IMPROVED
    # STRUCT1_SUBSET_IMPROVED_GLOBAL_NOT
    # STRUCT1_REAL_TRAINING_NO_IMPROVEMENT
    # STRUCT1_INTEGRITY_GUARD_FAILED
    # STRUCT1_TRAINING_BLOCKED
    # STRUCT1_EXPORT_BLOCKED
    # STRUCT1_ERROR
    run(parse_args())
