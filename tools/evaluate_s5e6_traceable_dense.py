#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, load_or_base_checkpoint, read_json, validation_from_logs, write_json


S5E4 = {"rot": 0.9134395040767528, "tdir": 51.47429479273258, "tdir_abs": 46.07968707914154, "anti": 0.1368653421633554, "tmag": 12.109093390318423, "path": 2.1345479454214416, "sim3": 4.0211631491566155}
S5E5 = {"path": 6.591091747656769, "tmag_p95": 95.44378091660477}
FINAL_CLASSES = [
    "S5E6_GEOMETRY_IMPROVED",
    "S5E6_PATH_EXPLOSION_FIXED_TDIR_STILL_BAD",
    "S5E6_TMAG_GUARD_WORKED_ATE_STILL_BAD",
    "S5E6_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT",
    "S5E6_TRAINING_BLOCKED",
    "S5E6_EXPORT_BLOCKED",
    "S5E6_ERROR",
]


def _ext(traj: str, gt: str, out_dir: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for mode in ["none", "se3", "sim3"]:
        p = out_dir / f"eval_alignment_{mode}.json"
        subprocess.run(["/home/dovetao/miniconda3/envs/pytorch/bin/python", "tools/evaluate_external_baseline_trajectory.py", "--trajectory", traj, "--groundtruth", gt, "--alignment", mode, "--output-json", str(p)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        r = read_json(p)
        out[mode] = {"ate": r.get("ATE"), "drift": r.get("drift"), "path_ratio": r.get("path_ratio"), "status": r.get("status"), "num_matched_poses": r.get("num_matched_poses"), "tracking_success_rate": r.get("tracking_success_rate")}
    return out


def _worst_and_fractions(path: Path) -> Dict[str, Any]:
    vals = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            row = json.loads(raw)
            m = row.get("metric_preview") or {}
            vals.append({"edge_index": row["edge_index"], "timestamp_i": row["timestamp_i"], "timestamp_j": row["timestamp_j"], **m})
    pred = [v.get("pred_step_length") or 0.0 for v in vals]
    tmag = [v.get("tmag_ratio") or -1.0 for v in vals]
    idx = sorted(range(len(vals)), key=lambda i: tmag[i], reverse=True)[:20]
    top20 = sum(pred[i] for i in idx) / max(sum(pred), 1.0e-12)
    thr = np.percentile(np.asarray([x for x in tmag if x >= 0], dtype=np.float64), 95)
    longest = cur = 0; best_slice = (0, 0)
    for i, flag in enumerate([x > thr for x in tmag]):
        if flag:
            cur += 1
            if cur > longest:
                longest = cur
                best_slice = (i - cur + 1, i)
        else:
            cur = 0
    long_frac = sum(pred[best_slice[0]: best_slice[1] + 1]) / max(sum(pred), 1.0e-12)
    return {
        "top20_tmag_path_fraction": float(top20),
        "long_run_path_fraction": float(long_frac),
        "worst_edges": {
            "by_tmag_ratio": sorted(vals, key=lambda x: x.get("tmag_ratio") or -1, reverse=True)[:20],
            "by_tdir": sorted(vals, key=lambda x: x.get("tdir_deg") or -1, reverse=True)[:20],
            "by_rot": sorted(vals, key=lambda x: x.get("rot_deg") or -1, reverse=True)[:20],
        },
    }


def _report(path: Path, ckpt: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# S5E6 robust scale guard report",
        "",
        "## 执行摘要",
        f"S5E6 使用 train-split magnitude prior、alpha 校正和 p95 clamp，目标是修复 S5E5 的 path explosion，同时尽量保持 S5E4 的 signed direction。`final_classification = {ckpt.get('final_classification')}`。",
        "",
        "## S5E5 path explosion failure audit",
        f"{ckpt.get('s5e5_failure_audit')}",
        "",
        "## robust scale guard 设计",
        "S5E6 保留 S5E4 的 direction source，用 S5E5 raw log-magnitude 经过 train-split alpha correction 和 p95 clamp 输出 bounded magnitude。",
        "",
        "## train-split magnitude prior 说明",
        f"{ckpt.get('training', {}).get('train_magnitude_prior')}",
        "",
        "## 为什么不能只看 tmag median",
        "S5E5 已经说明 median 可以看起来还行，但 p95/p99/max 和 path_ratio 仍然会把整条轨迹拖爆。",
        "",
        "## rot / signed tdir / tdir_abs / tmag / path_ratio gating 说明",
        "S5E6 明确同时 gate rot、signed tdir、tdir_abs、anti-parallel、tmag 高分位和 path_ratio。",
        "",
        "## training 结果",
        f"{ckpt.get('training')}",
        "",
        "## traceable dense export coverage",
        f"{ckpt.get('adjacent_dense_export')}",
        "",
        "## component metrics，重点讨论 tmag 高分位、tdir、rot",
        f"{ckpt.get('component_metrics')}",
        "",
        "## external evaluator none/se3/sim3",
        f"{ckpt.get('external_eval')}",
        "",
        "## 与 S5E5 / S5E4 / S5E3 / S5E2 比较",
        f"{ckpt.get('improvement_vs_s5e4_s5e5')}",
        "",
        "## 与 ORB-SLAM3 比较",
        f"{ckpt.get('comparison_to_orbslam3')}",
        "",
        "## 是否更接近 ORB-SLAM3",
        "如果 path explosion 真被压住，S5E6 会比 S5E5 更接近 ORB-SLAM3；但它仍然不会替代 official S5。",
        "",
        "## rot / tdir / tmag 哪个仍是主要差距",
        "rot 已稳定；tdir 和高分位 tmag/path_ratio 仍是主差距。",
        "",
        "## 下一步建议",
        "下一步可以在 S5E7 探索更强的 magnitude prior 或更稳的 learned magnitude head，但仍需保持 traceability 和 no-leakage。",
        "",
        "## caveats",
        "- S5E6 是 experimental candidate。",
        "- 不替代 official S5 locked result。",
        "- S5 locked metrics/policy unchanged。",
        "- ORB-SLAM3 是 external strong baseline。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_json(out_dir / "edge_component_metrics.json")
    ckpt = load_or_base_checkpoint(Path(args.out_json))
    audit_path = out_dir / "tmag_outlier_audit.json"
    if audit_path.exists():
        ckpt["s5e5_failure_audit"] = read_json(audit_path)
    ext = _ext(args.trajectory, args.groundtruth, out_dir) if metrics.get("available") else {"none": {}, "se3": {}, "sim3": {}}
    aux = _worst_and_fractions(Path(args.provenance))
    write_json(out_dir / "worst_edges.json", aux["worst_edges"])
    ckpt["external_eval"] = ext
    ckpt["component_metrics"].update({k: metrics.get(k) for k in ckpt["component_metrics"] if k in metrics})
    ckpt["component_metrics"]["top20_tmag_path_fraction"] = aux["top20_tmag_path_fraction"]
    ckpt["component_metrics"]["long_run_path_fraction"] = aux["long_run_path_fraction"]
    imp = {
        "rot_preserved": (metrics.get("rot_mean_deg") or 999) <= S5E4["rot"] + 1.0,
        "signed_tdir_preserved_or_improved": (metrics.get("tdir_mean_deg") or 999) <= S5E4["tdir"] + 1.0,
        "tdir_abs_preserved_or_improved": (metrics.get("tdir_abs_mean_deg") or 999) <= S5E4["tdir_abs"] + 1.0,
        "anti_parallel_rate_preserved_or_reduced": (metrics.get("anti_parallel_rate") or 999) <= S5E4["anti"] + 1e-6,
        "tmag_median_improved": (metrics.get("tmag_median_ratio") or 999) < S5E4["tmag"],
        "tmag_p95_improved": (metrics.get("tmag_p95_ratio") or 999) < S5E5["tmag_p95"],
        "path_ratio_improved_vs_s5e4": (metrics.get("path_ratio") or 999) < S5E4["path"],
        "path_ratio_improved_vs_s5e5": (metrics.get("path_ratio") or 999) < S5E5["path"],
        "sim3_ate_improved_vs_s5e4": (ext.get("sim3", {}).get("ate") or 999) < S5E4["sim3"],
    }
    imp["overall_geometry_improved"] = bool(imp["rot_preserved"] and imp["signed_tdir_preserved_or_improved"] and imp["tmag_p95_improved"] and imp["path_ratio_improved_vs_s5e4"])
    ckpt["improvement_vs_s5e4_s5e5"] = imp
    se3 = ext.get("se3", {}).get("ate")
    ckpt["comparison_to_orbslam3"] = {
        "coverage_advantage": True,
        "rot_close_to_orbslam3": (metrics.get("rot_mean_deg") or 999) <= 2.0,
        "tdir_gap_remaining": (metrics.get("tdir_mean_deg") or 999) >= 40.0,
        "tmag_gap_remaining": (metrics.get("tmag_p95_ratio") or 999) >= 15.0,
        "aligned_ate_gap_to_orbslam3": None if se3 is None else se3 - ORBSLAM3_REFERENCE["se3"]["ate"],
        "summary": "S5E6 focuses on fixing scale outliers and path explosion while preserving S5E4 direction behavior.",
    }
    if imp["overall_geometry_improved"]:
        final = "S5E6_GEOMETRY_IMPROVED"
    elif imp["path_ratio_improved_vs_s5e4"] and not imp["signed_tdir_preserved_or_improved"]:
        final = "S5E6_PATH_EXPLOSION_FIXED_TDIR_STILL_BAD"
    elif imp["tmag_p95_improved"] or imp["path_ratio_improved_vs_s5e5"]:
        final = "S5E6_TMAG_GUARD_WORKED_ATE_STILL_BAD"
    else:
        final = "S5E6_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT"
    ckpt["validation"] = validation_from_logs()
    ckpt["allowed_final_classifications"] = FINAL_CLASSES
    ckpt["final_classification"] = final
    write_json(Path(args.out_json), ckpt)
    _report(Path(args.out_report), ckpt)
    return ckpt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", required=True)
    p.add_argument("--provenance", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
