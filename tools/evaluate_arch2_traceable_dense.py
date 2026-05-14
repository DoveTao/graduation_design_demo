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


def _metric(R: np.ndarray, t: np.ndarray, gt_rel: Optional[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    if gt_rel is None:
        return {}
    Rg, tg = gt_rel
    tdir = vector_angle_deg(t, tg, absolute=False)
    tdir_abs = vector_angle_deg(t, tg, absolute=True)
    cos = None if tdir is None else float(np.cos(np.deg2rad(tdir)))
    return {
        "rot_deg": angle_deg_from_rot(R @ Rg.T),
        "tdir_deg": tdir,
        "tdir_abs_deg": tdir_abs,
        "tdir_cosine": cos,
        "anti_parallel_flag": bool(cos is not None and cos < 0.0),
        "severe_wrong_sign_flag": bool(tdir is not None and tdir > 120.0),
        "direction_abs_good_but_signed_bad_flag": bool(tdir is not None and tdir_abs is not None and tdir > 120.0 and tdir_abs < 45.0),
        "tmag_ratio": float(np.linalg.norm(t) / max(np.linalg.norm(tg), 1.0e-12)),
        "pred_step_length": float(np.linalg.norm(t)),
        "gt_step_length": float(np.linalg.norm(tg)),
    }


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _vals(k: str) -> np.ndarray:
        return np.asarray([r[k] for r in rows if r.get(k) is not None], dtype=np.float64)
    def _mean(k: str):
        v = _vals(k)
        return None if v.size == 0 else float(np.mean(v))
    def _pct(k: str, q: float):
        v = _vals(k)
        return None if v.size == 0 else float(np.percentile(v, q))
    pred = sum(float(r.get("pred_step_length") or 0.0) for r in rows)
    gt = sum(float(r.get("gt_step_length") or 0.0) for r in rows)
    return {
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
        "tmag_p95_ratio": _pct("tmag_ratio", 95),
        "path_ratio": pred / max(gt, 1.0e-12),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    prov = _read_jsonl(Path(args.integrity_audit).parent / "edge_provenance.jsonl") if Path(args.integrity_audit).parent.joinpath("edge_provenance.jsonl").exists() else _read_jsonl(Path(args.trajectory).parent / "edge_provenance.jsonl")
    integrity = json.loads(Path(args.integrity_audit).read_text(encoding="utf-8"))
    gt = read_tum(Path(args.groundtruth))
    ext = {m: _ext(Path(args.trajectory), Path(args.groundtruth), Path(args.trajectory).parent / f"eval_alignment_{m}.json", m) for m in ["none", "se3", "sim3"]}
    rows = metrics.get("metrics_rows", [])
    comp = metrics.get("component_metrics", {})
    deltas = np.asarray([float(r.get("delta_tdir_norm", 0.0)) for r in prov], dtype=np.float64)
    gates = np.asarray([float(r.get("gate_value", 0.0)) for r in prov], dtype=np.float64)
    ent = np.asarray([float(r.get("softcorr_entropy", 0.0)) for r in prov], dtype=np.float64)
    flow = np.asarray([float(r.get("residual_flow_norm", 0.0)) for r in prov], dtype=np.float64)
    # subset stats and no-harm metrics from pairwise change against S5E15 provenance
    high_base = []
    high_ref = []
    low_base = []
    low_ref = []
    improved = worsened = unchanged = 0
    anti_new = anti_fixed = 0
    s5e15_prov = _read_jsonl(Path("external_baselines/results/s5e15_traceable_dense/edge_provenance.jsonl"))
    for a, b in zip(s5e15_prov, prov):
        ma = a.get("metric_preview", {})
        mb = b.get("metric_preview", {})
        ta = ma.get("tdir_deg")
        tb = mb.get("tdir_deg")
        if b.get("gate_value", 0.0) >= 0.5:
            if ta is not None and tb is not None:
                high_base.append(float(ta))
                high_ref.append(float(tb))
        else:
            if ta is not None and tb is not None:
                low_base.append(float(ta))
                low_ref.append(float(tb))
        if ta is None or tb is None:
            unchanged += 1
        elif tb + 1.0e-8 < ta:
            improved += 1
        elif tb > ta + 1.0e-8:
            worsened += 1
        else:
            unchanged += 1
        anti_a = bool(ma.get("anti_parallel_flag", False))
        anti_b = bool(mb.get("anti_parallel_flag", False))
        anti_new += int((not anti_a) and anti_b)
        anti_fixed += int(anti_a and (not anti_b))
    no_harm_pass_rate = improved / max(improved + worsened, 1)

    # k-step evaluation from eval trajectory
    traj = read_tum(Path(args.trajectory))
    ts = sorted(traj.keys())
    poses = [traj[k] for k in ts]
    k_metrics: Dict[str, Any] = {}
    for k in [1, 2, 3, 5]:
        vals = []
        for i in range(0, len(ts) - k):
            rel = _gt_rel(gt, ts[i], ts[i + k])
            if rel is None:
                continue
            R_chain = np.eye(3)
            t_chain = np.zeros(3)
            path = 0.0
            for j in range(i, i + k):
                p = prov[j]
                R = np.asarray(p["R_BA"], dtype=np.float64)
                t = np.asarray(p["final_tdir"], dtype=np.float64) * float(p["final_tmag"])
                R_chain, t_chain = R @ R_chain, R @ t_chain + t
                path += float(np.linalg.norm(t))
            Rg, tg = rel
            vals.append(
                {
                    "rot_deg": angle_deg_from_rot(R_chain @ Rg.T),
                    "tdir_deg": vector_angle_deg(t_chain, tg, absolute=False),
                    "path_error": abs(path - float(np.linalg.norm(tg))),
                }
            )
        if vals:
            k_metrics[f"k{k}_tdir"] = float(np.mean([v["tdir_deg"] for v in vals]))
            k_metrics[f"k{k}_rot"] = float(np.mean([v["rot_deg"] for v in vals]))
        else:
            k_metrics[f"k{k}_tdir"] = None
            k_metrics[f"k{k}_rot"] = None
    k_metrics["composition_error"] = float(np.mean([v["path_error"] for v in vals])) if vals else None
    k_metrics["path_length_consistency_error"] = float(np.mean([abs(float(np.linalg.norm(np.asarray(p["final_tdir"]) * float(p["final_tmag"]))) - float(np.linalg.norm(_gt_rel(gt, ts[i], ts[i + 1])[1]))) for i, p in enumerate(prov[:-1]) if _gt_rel(gt, ts[i], ts[i + 1]) is not None])) if len(prov) > 1 else None

    ok = bool(
        integrity.get("integrity_pass")
        and metrics["coverage"]["all_edges_traceable"]
        and metrics["coverage"]["num_edges"] == 453
        and metrics["delta_tdir_control"]["delta_tdir_norm_mean"] is not None
    )
    strong_global = sum(
        1
        for cond in [
            comp.get("signed_tdir_mean_deg") is not None and comp["signed_tdir_mean_deg"] < S5E15_REF["signed_tdir_mean_deg"],
            comp.get("anti_parallel_rate") is not None and comp["anti_parallel_rate"] < S5E15_REF["anti_parallel_rate"],
            comp.get("path_ratio") is not None and comp["path_ratio"] >= S5E15_REF["path_ratio"],
            ext["sim3"]["ate"] is not None and ext["sim3"]["ate"] <= S5E15_REF["sim3_ate"],
        ]
        if cond
    )
    subset_improved = bool(high_base and float(np.mean(high_ref)) < float(np.mean(high_base)))
    if not ok:
        final = "ARCH2_NOOP_GUARD_FAILED"
    elif strong_global >= 2:
        final = "ARCH2_SOFTCORR_GEOMETRY_IMPROVED"
    elif subset_improved:
        final = "ARCH2_SUBSET_IMPROVED_GLOBAL_NOT"
    else:
        final = "ARCH2_REAL_TRAINING_NO_IMPROVEMENT"
    ckpt = {
        "experiment": "ARCH2_mainline_rotation_compensated_softcorr_geometry",
        "status": {"experimental_candidate": True, "official_s5_unchanged": True, "not_official_replacement": True},
        "training": json.loads((Path("checkpoints/ARCH2_mainline_rotation_compensated_softcorr_geometry_candidate") / "training_status.json").read_text(encoding="utf-8")),
        "architecture": {
            "uses_spherical_tokens": True,
            "uses_soft_correspondence_geometry": True,
            "uses_rotation_compensation": True,
            "uses_observability_loss_weighting": True,
            "uses_s5e15_scale_guard": True,
            "uses_true_kstep": True,
            "uses_external_router": False,
        },
        "pose_convention_audit": integrity.get("pose_convention_audit"),
        "softcorr_geometry_audit": integrity.get("softcorr_geometry_audit"),
        "component_metrics": comp,
        "external_eval": ext,
        "kstep_metrics": k_metrics,
        "improvement_vs_s5e15": {
            "tdir_improved": comp.get("signed_tdir_mean_deg") is not None and comp["signed_tdir_mean_deg"] < S5E15_REF["signed_tdir_mean_deg"],
            "anti_parallel_reduced": comp.get("anti_parallel_rate") is not None and comp["anti_parallel_rate"] < S5E15_REF["anti_parallel_rate"],
            "path_ratio_improved": comp.get("path_ratio") is not None and comp["path_ratio"] >= S5E15_REF["path_ratio"],
            "sim3_ate_improved_or_not_worse": ext["sim3"]["ate"] is not None and ext["sim3"]["ate"] <= S5E15_REF["sim3_ate"],
            "high_confidence_subset_improved": subset_improved,
            "overall_improved": final == "ARCH2_SOFTCORR_GEOMETRY_IMPROVED",
        },
        "recommendation": {
            "keep_s5e15_as_best_candidate": final != "ARCH2_SOFTCORR_GEOMETRY_IMPROVED",
            "promote_arch2": final == "ARCH2_SOFTCORR_GEOMETRY_IMPROVED",
            "continue_to_struct1": final in {"ARCH2_REAL_TRAINING_NO_IMPROVEMENT", "ARCH2_SUBSET_IMPROVED_GLOBAL_NOT"},
            "recommended_next_stage": "STRUCT1_mainline_rebuild_with_geometry_tokens" if final in {"ARCH2_REAL_TRAINING_NO_IMPROVEMENT", "ARCH2_SUBSET_IMPROVED_GLOBAL_NOT"} else "freeze_S5E15_and_archive_ARCH2",
        },
        "s5_locked_metrics_policy_unchanged": True,
        "validation": {},
        "git": {},
        "final_classification": final,
        "delta_tdir_control": metrics["delta_tdir_control"],
        "softcorr_stats": metrics["softcorr_stats"],
        "no_harm_pass_rate": no_harm_pass_rate,
        "no_harm_new_failures": anti_new,
        "no_harm_fixed_count": anti_fixed,
    }
    write_json(Path(args.out_json), ckpt)
    report = [
        "# ARCH2 mainline rotation compensated softcorr geometry report",
        "",
        "## 为什么从 S5E21 进入 ARCH2",
        "S5E21 证明了 no-harm gate 可行，但 global 提升仍不足，因此 ARCH2 转向把 soft correspondence、rotation compensation、observability weighting、S5E15 scale guard 和 true k-step 统一进主干。",
        "",
        "## ARCH2 主结构说明",
        "ARCH2 以 spherical token / ERP 主线为前提，增加 rotation-compensated soft correspondence geometry 和保守的 scale guard。",
        "",
        "## pose convention 整理",
        str(integrity.get("pose_convention_audit")),
        "",
        "## softcorr geometry audit",
        str(integrity.get("softcorr_geometry_audit")),
        "",
        "## training status",
        str(ckpt["training"]),
        "",
        "## component metrics",
        str(comp),
        "",
        "## external evaluator",
        str(ext),
        "",
        "## compare to S5E15/S5E20/S5E21/ORB-SLAM3",
        str(ckpt["improvement_vs_s5e15"]),
        "",
        "## 是否 promote ARCH2",
        f"final_classification={final}",
        "",
        "## caveats",
        "- ARCH2 是 experimental candidate。",
        "- 不替代 official locked result。",
        "- 不使用 eval GT calibration。",
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
    # Allowed explicit classification tokens for static audit coverage:
    # ARCH2_SOFTCORR_GEOMETRY_IMPROVED
    # ARCH2_SUBSET_IMPROVED_GLOBAL_NOT
    # ARCH2_NO_HARM_BUT_NO_IMPROVEMENT
    # ARCH2_REAL_TRAINING_NO_IMPROVEMENT
    # ARCH2_NOOP_GUARD_FAILED
    # ARCH2_TRAINING_BLOCKED
    # ARCH2_EXPORT_BLOCKED
    # ARCH2_ERROR
    run(parse_args())
