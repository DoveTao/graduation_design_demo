#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, load_or_base_checkpoint, read_json, validation_from_logs, write_json


OUT_JSON_DEFAULT = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate.json")
S5E2_BASELINE = {
    "rot_mean_deg": 0.9134395040767528,
    "tdir_mean_deg": 51.47429479273258,
    "tdir_abs_mean_deg": 46.07968707914153,
    "tmag_median_ratio": 32.77577273937451,
    "path_ratio": 3.5557784814144453,
    "sim3_ate": 4.097680633241629,
}


def _external_eval(trajectory: str, groundtruth: str, out_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for mode in ["none", "se3", "sim3"]:
        out_json = out_dir / f"eval_alignment_{mode}.json"
        subprocess.run(
            [
                "/home/dovetao/miniconda3/envs/pytorch/bin/python",
                "tools/evaluate_external_baseline_trajectory.py",
                "--trajectory",
                trajectory,
                "--groundtruth",
                groundtruth,
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
        raw = read_json(out_json)
        out[mode] = {"ate": raw.get("ATE"), "drift": raw.get("drift"), "path_ratio": raw.get("path_ratio"), "status": raw.get("status"), "num_matched_poses": raw.get("num_matched_poses"), "tracking_success_rate": raw.get("tracking_success_rate")}
    return out


def _worst_edges(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    vals = []
    for row in rows:
        m = row.get("metric_preview") or {}
        if m:
            vals.append({"edge_index": row["edge_index"], "timestamp_i": row["timestamp_i"], "timestamp_j": row["timestamp_j"], **m})
    return {
        "by_tmag_ratio": sorted(vals, key=lambda x: x.get("tmag_ratio") or -1, reverse=True)[:20],
        "by_tdir": sorted(vals, key=lambda x: x.get("tdir_deg") or -1, reverse=True)[:20],
        "by_rot": sorted(vals, key=lambda x: x.get("rot_deg") or -1, reverse=True)[:20],
    }


def _write_report(path: Path, ckpt: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imp = ckpt.get("improvement_vs_s5e2", {})
    ext = ckpt.get("external_eval", {})
    lines = [
        "# S5E3 scale calibrated adjacent dense report",
        "",
        "## 执行摘要",
        "",
        f"S5E3 基于 S5E2 的 full traceable adjacent-dense pipeline，新增 direction head 和 log-magnitude calibration head，并继承 S5E2 的低 rot 输出。本轮 `final_classification = {ckpt.get('final_classification')}`。",
        "",
        "## S5E2 failure recap",
        "",
        "S5E2 已有 453 条 direct adjacent predictions，但 `tmag_median_ratio=32.77577273937451`、`path_ratio=3.5557784814144453`。worst 20 edges 不单独主导全部路径，tmag over-scaling 更像是系统性小位移放大问题。",
        "",
        "## S5E3 方法设计",
        "",
        "S5E3 采用 multi-head adjacent regressor：rotation head 继承 S5E2，translation direction head 单独回归单位方向，translation magnitude head 回归 `log(||t||)`，并用 train split 的 magnitude quantile 做保守 clipping。",
        "",
        "## scale calibration / tmag loss 说明",
        "",
        "`tmag_log` 使用 train split 的 adjacent labels 训练；`scene01/seq03` GT 只在本报告的 diagnostics/evaluation 阶段使用。",
        "",
        "## rot / tdir / tmag gating 说明",
        "",
        "S5E3 不以 ATE 单独判定成功；gate 同时检查 rot、tdir、tdir_abs、tmag、path_ratio 和 sim3 ATE。",
        "",
        "## training 结果",
        "",
        f"- classification = {ckpt.get('training', {}).get('classification')}",
        f"- num_train_pairs = {ckpt.get('training', {}).get('num_train_pairs')}",
        f"- num_val_pairs = {ckpt.get('training', {}).get('num_val_pairs')}",
        "",
        "## traceable dense export coverage",
        "",
        f"- coverage = {ckpt.get('adjacent_dense_export', {}).get('coverage')}",
        f"- all_edges_traceable = {ckpt.get('adjacent_dense_export', {}).get('all_edges_traceable')}",
        f"- direct_adjacent_prediction_edges = {ckpt.get('adjacent_dense_export', {}).get('direct_adjacent_prediction_edges')}",
        "",
        "## component metrics",
        "",
        f"- rot_mean/median/p90 = {metrics.get('rot_mean_deg')} / {metrics.get('rot_median_deg')} / {metrics.get('rot_p90_deg')}",
        f"- tdir_mean/median/p90 = {metrics.get('tdir_mean_deg')} / {metrics.get('tdir_median_deg')} / {metrics.get('tdir_p90_deg')}",
        f"- tdir_abs_mean/median/p90 = {metrics.get('tdir_abs_mean_deg')} / {metrics.get('tdir_abs_median_deg')} / {metrics.get('tdir_abs_p90_deg')}",
        f"- tdir_mean_cosine = {metrics.get('tdir_mean_cosine')}",
        f"- tmag_median/mean/p90/p95 = {metrics.get('tmag_median_ratio')} / {metrics.get('tmag_mean_ratio')} / {metrics.get('tmag_p90_ratio')} / {metrics.get('tmag_p95_ratio')}",
        f"- path_ratio = {metrics.get('path_ratio')}",
        "",
        "## external evaluator none/se3/sim3",
        "",
        f"- none = {ext.get('none')}",
        f"- se3 = {ext.get('se3')}",
        f"- sim3 = {ext.get('sim3')}",
        "",
        "## 与 S5E2 比较",
        "",
        f"- improvement_vs_s5e2 = {imp}",
        "",
        "## 与 ORB-SLAM3 比较",
        "",
        "ORB-SLAM3 仍是 external strong baseline。S5E3 的优势是 full coverage；如果 aligned ATE 仍远高于 ORB-SLAM3，则不能声称接近 ORB-SLAM3。",
        "",
        "## 是否更接近 ORB-SLAM3",
        "",
        ckpt.get("comparison_to_orbslam3", {}).get("summary", "comparison pending"),
        "",
        "## rot / tdir 是否仍是主要差距",
        "",
        "rot 由 S5E2 继承，通常保持较好；tdir 若仍高于目标阈值，则仍是主要几何差距。",
        "",
        "## 下一步建议",
        "",
        "1. 用更强视觉 backbone 替换 image-statistics features。",
        "2. 引入 sequence-level path loss，但继续禁止 `scene01/seq03` GT 参与训练。",
        "3. 若要借助 ORB-SLAM3 轨迹做蒸馏，应另开 S5E4 并明确 distillation protocol。",
        "",
        "## caveats",
        "",
        "- S5E3 是 experimental candidate。",
        "- 不替代 official S5 locked result。",
        "- S5 locked metrics/policy unchanged。",
        "- ORB-SLAM3 是 external strong baseline。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_json(out_dir / "edge_component_metrics.json")
    ckpt = load_or_base_checkpoint(Path(args.out_json))
    ext = _external_eval(args.trajectory, args.groundtruth, out_dir) if metrics.get("available") else {"none": {}, "se3": {}, "sim3": {}}
    write_json(out_dir / "worst_edges.json", _worst_edges(Path(args.provenance)))
    ckpt["external_eval"] = ext
    ckpt["component_metrics"].update({k: metrics.get(k) for k in ckpt["component_metrics"].keys() if k in metrics})
    imp = {
        "rot_improved_or_preserved": bool((metrics.get("rot_mean_deg") or 999) <= S5E2_BASELINE["rot_mean_deg"] + 1.0),
        "tdir_improved": bool((metrics.get("tdir_mean_deg") or 999) < S5E2_BASELINE["tdir_mean_deg"]),
        "tdir_abs_improved": bool((metrics.get("tdir_abs_mean_deg") or 999) < S5E2_BASELINE["tdir_abs_mean_deg"]),
        "tmag_improved": bool(abs(np.log(max(metrics.get("tmag_median_ratio") or 1.0, 1e-12))) < abs(np.log(S5E2_BASELINE["tmag_median_ratio"]))),
        "path_ratio_improved": bool(abs((metrics.get("path_ratio") or 999) - 1.0) < abs(S5E2_BASELINE["path_ratio"] - 1.0)),
        "sim3_ate_improved": bool((ext.get("sim3", {}).get("ate") or 999) < S5E2_BASELINE["sim3_ate"]),
    }
    imp["overall_geometry_improved"] = bool(imp["rot_improved_or_preserved"] and (imp["tmag_improved"] or imp["path_ratio_improved"]) and (imp["tdir_improved"] or imp["tdir_abs_improved"] or imp["sim3_ate_improved"]))
    ckpt["improvement_vs_s5e2"] = imp
    se3_ate = ext.get("se3", {}).get("ate")
    ckpt["comparison_to_orbslam3"] = {
        "coverage_advantage": (metrics.get("coverage") or 0) >= 1.0,
        "rot_close_to_orbslam3": (metrics.get("rot_mean_deg") or 999) <= 2.0,
        "tdir_gap_remaining": (metrics.get("tdir_mean_deg") or 999) >= 45.0 and (metrics.get("tdir_abs_mean_deg") or 999) >= 35.0,
        "aligned_ate_gap_to_orbslam3": None if se3_ate is None else se3_ate - ORBSLAM3_REFERENCE["se3"]["ate"],
        "summary": "S5E3 full coverage 保留，但 aligned ATE/tdir 与 ORB-SLAM3 仍有差距。",
    }
    if imp["overall_geometry_improved"] and (metrics.get("tmag_median_ratio") or 999) < 5.0 and (metrics.get("path_ratio") or 999) < 1.5 and (metrics.get("tdir_mean_deg") or 999) < 45.0:
        final = "S5E3_GEOMETRY_IMPROVED"
    elif imp["tmag_improved"] or imp["path_ratio_improved"]:
        final = "S5E3_TMAG_IMPROVED_TDIR_STILL_BAD"
    else:
        final = "S5E3_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT"
    ckpt["validation"] = validation_from_logs()
    ckpt["final_classification"] = final
    write_json(Path(args.out_json), ckpt)
    _write_report(Path(args.out_report), ckpt, metrics)
    return ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate S5E3 traceable dense candidate.")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--groundtruth", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
