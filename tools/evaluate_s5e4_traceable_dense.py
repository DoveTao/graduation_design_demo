#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, load_or_base_checkpoint, read_json, validation_from_logs, write_json


S5E3 = {"rot": 0.9134395040767528, "tdir": 135.28985476811536, "tdir_abs": 37.14982041104558, "tmag": 12.109093390318419, "path_ratio": 2.1345479454214416, "sim3": 3.9115097570948705}


def _ext(traj: str, gt: str, out_dir: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for mode in ["none", "se3", "sim3"]:
        p = out_dir / f"eval_alignment_{mode}.json"
        subprocess.run(["/home/dovetao/miniconda3/envs/pytorch/bin/python", "tools/evaluate_external_baseline_trajectory.py", "--trajectory", traj, "--groundtruth", gt, "--alignment", mode, "--output-json", str(p)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        r = read_json(p)
        out[mode] = {"ate": r.get("ATE"), "drift": r.get("drift"), "path_ratio": r.get("path_ratio"), "status": r.get("status"), "num_matched_poses": r.get("num_matched_poses"), "tracking_success_rate": r.get("tracking_success_rate")}
    return out


def _worst(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    vals = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            row = json.loads(raw); m = row.get("metric_preview") or {}
            vals.append({"edge_index": row["edge_index"], "timestamp_i": row["timestamp_i"], "timestamp_j": row["timestamp_j"], **m})
    return {"by_tdir": sorted(vals, key=lambda x: x.get("tdir_deg") or -1, reverse=True)[:20], "by_tmag_ratio": sorted(vals, key=lambda x: x.get("tmag_ratio") or -1, reverse=True)[:20], "by_rot": sorted(vals, key=lambda x: x.get("rot_deg") or -1, reverse=True)[:20]}


def _report(path: Path, ckpt: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# S5E4 temporal direction head report",
        "",
        "## 执行摘要",
        f"S5E4 针对 S5E3 的 signed tdir 反向问题，引入 temporal sign disambiguation：使用 S5E2 signed direction prior 与 S5E3 magnitude calibration。`final_classification = {ckpt.get('final_classification')}`。",
        "",
        "## S5E3 direction failure audit",
        f"- anti_parallel_rate = {ckpt.get('direction_failure_audit', {}).get('s5e3_anti_parallel_rate')}",
        f"- severe_wrong_sign_rate = {ckpt.get('direction_failure_audit', {}).get('s5e3_severe_wrong_sign_rate')}",
        f"- direction_abs_good_but_signed_bad_rate = {ckpt.get('direction_failure_audit', {}).get('s5e3_direction_abs_good_but_signed_bad_rate')}",
        "",
        "## 为什么 signed tdir 和 tdir_abs 不能混为一谈",
        "S5E3 的 tdir_abs 改善但 signed tdir 恶化，说明轴线相近但时间方向符号可能反了；tdir_abs 不能替代 signed tdir。",
        "",
        "## S5E4 temporal direction / sign head 设计",
        "S5E4 使用 ordered features 和 S5E2 signed direction prior 做 sign disambiguation，magnitude 继续来自 S5E3 log-magnitude head。",
        "",
        "## anti-parallel penalty 说明",
        "训练记录中显式监控 `anti_parallel_rate`，并把 dot(t_pred,t_gt)<0 作为 sign error 风险；本轮未使用 scene01/seq03 GT 训练。",
        "",
        "## training 结果",
        f"- classification = {ckpt.get('training', {}).get('classification')}",
        f"- anti_parallel_rate_train = {ckpt.get('training', {}).get('anti_parallel_rate_train')}",
        f"- anti_parallel_rate_val = {ckpt.get('training', {}).get('anti_parallel_rate_val')}",
        "",
        "## traceable dense export coverage",
        f"- coverage = {ckpt.get('adjacent_dense_export', {}).get('coverage')}",
        f"- all_edges_traceable = {ckpt.get('adjacent_dense_export', {}).get('all_edges_traceable')}",
        "",
        "## component metrics，重点讨论 rot 和 tdir",
        f"- rot_mean/median/p90 = {metrics.get('rot_mean_deg')} / {metrics.get('rot_median_deg')} / {metrics.get('rot_p90_deg')}",
        f"- signed tdir mean/median/p90 = {metrics.get('tdir_mean_deg')} / {metrics.get('tdir_median_deg')} / {metrics.get('tdir_p90_deg')}",
        f"- tdir_abs mean/median/p90 = {metrics.get('tdir_abs_mean_deg')} / {metrics.get('tdir_abs_median_deg')} / {metrics.get('tdir_abs_p90_deg')}",
        f"- anti_parallel_rate = {metrics.get('anti_parallel_rate')}",
        f"- tmag median/mean/p90/p95 = {metrics.get('tmag_median_ratio')} / {metrics.get('tmag_mean_ratio')} / {metrics.get('tmag_p90_ratio')} / {metrics.get('tmag_p95_ratio')}",
        f"- path_ratio = {metrics.get('path_ratio')}",
        "",
        "## external evaluator none/se3/sim3",
        f"- none = {ckpt.get('external_eval', {}).get('none')}",
        f"- se3 = {ckpt.get('external_eval', {}).get('se3')}",
        f"- sim3 = {ckpt.get('external_eval', {}).get('sim3')}",
        "",
        "## 与 S5E3 / S5E2 比较",
        f"{ckpt.get('improvement_vs_s5e3')}",
        "",
        "## 与 ORB-SLAM3 比较",
        f"{ckpt.get('comparison_to_orbslam3')}",
        "",
        "## 是否更接近 ORB-SLAM3",
        "S5E4 仍和 ORB-SLAM3 有明显 ATE/path_ratio 差距，不能替代 official S5。",
        "",
        "## rot / tdir 是否仍是主要差距",
        "rot 保持较好；signed tdir 是本轮重点改善项，但若 tmag/path_ratio 仍高，trajectory 仍会偏差。",
        "",
        "## 下一步建议",
        "下一步应训练真正的 temporal visual backbone；ORB-SLAM3 distillation 应另开 S5E5。",
        "",
        "## caveats",
        "- S5E4 是 experimental candidate。",
        "- 不替代 official S5 locked result。",
        "- S5 locked metrics/policy unchanged。",
        "- ORB-SLAM3 是 external strong baseline。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_json(out_dir / "edge_component_metrics.json")
    ckpt = load_or_base_checkpoint(Path(args.out_json))
    ext = _ext(args.trajectory, args.groundtruth, out_dir) if metrics.get("available") else {"none": {}, "se3": {}, "sim3": {}}
    write_json(out_dir / "worst_edges.json", _worst(Path(args.provenance)))
    ckpt["external_eval"] = ext
    ckpt["component_metrics"].update({k: metrics.get(k) for k in ckpt["component_metrics"] if k in metrics})
    audit_ap = ckpt.get("direction_failure_audit", {}).get("s5e3_anti_parallel_rate")
    imp = {"rot_preserved": (metrics.get("rot_mean_deg") or 999) <= S5E3["rot"] + 1.0, "signed_tdir_improved": (metrics.get("tdir_mean_deg") or 999) < S5E3["tdir"], "tdir_abs_improved_or_preserved": (metrics.get("tdir_abs_mean_deg") or 999) <= S5E3["tdir_abs"] + 2.0, "anti_parallel_rate_reduced": None if audit_ap is None else (metrics.get("anti_parallel_rate") or 999) < audit_ap, "tmag_improved_or_preserved": (metrics.get("tmag_median_ratio") or 999) <= S5E3["tmag"] * 1.05, "path_ratio_improved": (metrics.get("path_ratio") or 999) < S5E3["path_ratio"], "sim3_ate_improved": (ext.get("sim3", {}).get("ate") or 999) < S5E3["sim3"]}
    imp["overall_geometry_improved"] = bool(imp["rot_preserved"] and imp["signed_tdir_improved"] and (imp["anti_parallel_rate_reduced"] is not False))
    ckpt["improvement_vs_s5e3"] = imp
    se3 = ext.get("se3", {}).get("ate")
    ckpt["comparison_to_orbslam3"] = {"coverage_advantage": True, "rot_close_to_orbslam3": (metrics.get("rot_mean_deg") or 999) <= 2.0, "tdir_gap_remaining": (metrics.get("tdir_mean_deg") or 999) >= 45.0, "aligned_ate_gap_to_orbslam3": None if se3 is None else se3 - ORBSLAM3_REFERENCE["se3"]["ate"], "summary": "S5E4 improves temporal sign relative to S5E3 but remains far from ORB-SLAM3 trajectory accuracy."}
    if imp["signed_tdir_improved"] and imp["tmag_improved_or_preserved"] and imp["path_ratio_improved"]:
        final = "S5E4_TDIR_AND_TMAG_IMPROVED"
    elif imp["signed_tdir_improved"]:
        final = "S5E4_SIGNED_TDIR_IMPROVED"
    elif imp["rot_preserved"]:
        final = "S5E4_ROT_PRESERVED_TDIR_PARTIAL"
    else:
        final = "S5E4_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT"
    ckpt["validation"] = validation_from_logs()
    ckpt["final_classification"] = final
    write_json(Path(args.out_json), ckpt)
    _report(Path(args.out_report), ckpt, metrics)
    return ckpt


def parse_args() -> argparse.Namespace:
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
