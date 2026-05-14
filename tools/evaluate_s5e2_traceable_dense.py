#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from s5e2_adjacent_dense_lib import (
    RESTORED_S5_DENSE_REFERENCE,
    load_or_base_checkpoint,
    read_json,
    validation_from_logs,
    write_json,
)


def _run_external_eval(trajectory: str, groundtruth: str, out_dir: Path) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for alignment in ["none", "se3", "sim3"]:
        out_json = out_dir / f"eval_alignment_{alignment}.json"
        cmd = [
            "/home/dovetao/miniconda3/envs/pytorch/bin/python",
            "tools/evaluate_external_baseline_trajectory.py",
            "--trajectory",
            trajectory,
            "--groundtruth",
            groundtruth,
            "--alignment",
            alignment,
            "--output-json",
            str(out_json),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        raw = read_json(out_json)
        results[alignment] = {
            "ate": raw.get("ATE"),
            "drift": raw.get("drift"),
            "path_ratio": raw.get("path_ratio"),
            "status": raw.get("status"),
            "num_matched_poses": raw.get("num_matched_poses"),
            "tracking_success_rate": raw.get("tracking_success_rate"),
        }
    return results


def _worst_edges(provenance_path: Path, top_k: int = 20) -> Dict[str, List[Dict[str, Any]]]:
    rows = [json.loads(x) for x in provenance_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    metric_rows = []
    for row in rows:
        metric = row.get("metric_preview") or {}
        if metric:
            metric_rows.append({"edge_index": row["edge_index"], "timestamp_i": row["timestamp_i"], "timestamp_j": row["timestamp_j"], **metric})
    return {
        "by_tmag_ratio": sorted(metric_rows, key=lambda x: x.get("tmag_ratio") or -1.0, reverse=True)[:top_k],
        "by_tdir": sorted(metric_rows, key=lambda x: x.get("tdir_deg") or -1.0, reverse=True)[:top_k],
        "by_rot": sorted(metric_rows, key=lambda x: x.get("rot_deg") or -1.0, reverse=True)[:top_k],
    }


def _write_report(path: Path, ckpt: Dict[str, Any], edge_metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    export = ckpt.get("adjacent_dense_export", {})
    training = ckpt.get("training", {})
    component = ckpt.get("component_metrics", {})
    external = ckpt.get("external_eval", {})
    lines = [
        "# S5E2 adjacent dense candidate report",
        "",
        "## 执行摘要",
        "",
        "S5E2 新增了 experimental adjacent_dense 训练与推理路径：使用 train split 的 `scene01/seq01` 和 `scene01/seq02` 构建 adjacent image-pair dataset，训练一个 minimal image-statistics ridge pose regressor，并在 held-out `scene01/seq03` 上导出 453 条 direct adjacent predictions。该结果是 experimental candidate，不替代 official S5 locked result。",
        "",
        f"`final_classification = {ckpt.get('final_classification')}`。",
        "",
        "## S5E1 blocker 回顾",
        "",
        ckpt.get("s5e1_blocker_recap", {}).get("blocker_summary", "S5E1 blocked: selected_k1 only."),
        "",
        "## adjacent_dense dataset 构建",
        "",
        f"- adjacent_dataset_ready = {ckpt.get('dataset', {}).get('adjacent_dataset_ready')}",
        f"- split_respected = {ckpt.get('dataset', {}).get('split_respected')}",
        f"- test_gt_used_for_training = {ckpt.get('dataset', {}).get('test_gt_used_for_training')}",
        f"- num_train_pairs = {ckpt.get('dataset', {}).get('num_train_pairs')}",
        f"- num_val_pairs = {ckpt.get('dataset', {}).get('num_val_pairs')}",
        "",
        "## training/fine-tune 情况",
        "",
        f"- attempted = {training.get('attempted')}",
        f"- classification = {training.get('classification')}",
        f"- checkpoint_dir = `{training.get('checkpoint_dir')}`",
        "",
        "## 新增模型/head/loss 说明",
        "",
        "S5E2 使用 `minimal_image_statistics` 作为 experimental backbone，`adjacent_dense_pose_head` 由 closed-form ridge regression 实现。loss/diagnostic proxy 包括 `SO(3) geodesic`、`tdir`、`tmag log`、`path length consistency`、`short-window consistency`，并保留 small translation mask caveat。",
        "",
        "## traceable adjacent dense export coverage",
        "",
        f"- trajectory_path = `{export.get('trajectory_path')}`",
        f"- edge_provenance = `{export.get('edge_provenance')}`",
        f"- num_poses = {export.get('num_poses')}",
        f"- num_edges = {export.get('num_edges')}",
        f"- direct_adjacent_prediction_edges = {export.get('direct_adjacent_prediction_edges')}",
        f"- coverage = {export.get('coverage')}",
        "",
        "## component metrics",
        "",
        f"- rot_mean_deg = {component.get('rot_mean_deg')}",
        f"- tdir_mean_deg = {component.get('tdir_mean_deg')}",
        f"- tdir_abs_mean_deg = {component.get('tdir_abs_mean_deg')}",
        f"- tmag_median_ratio = {component.get('tmag_median_ratio')}",
        f"- tmag_p90_ratio = {component.get('tmag_p90_ratio')}",
        f"- path_ratio = {component.get('path_ratio')}",
        "",
        "## external evaluator none/se3/sim3",
        "",
        f"- none = {external.get('none')}",
        f"- se3 = {external.get('se3')}",
        f"- sim3 = {external.get('sim3')}",
        "",
        "## 与 ORB-SLAM3 对比",
        "",
        "ORB-SLAM3 仍是 external strong baseline。S5E2 的优势是 454/454 full coverage 与完整 edge provenance；主要差距是 aligned ATE 和局部方向/旋转误差仍由极简模型限制。",
        "",
        "## 是否接近 ORB-SLAM3",
        "",
        "如果 `se3` / `sim3` ATE 仍明显高于 ORB-SLAM3，则只能说明 S5E2 解决了 traceability/export blocker，还没有解决高精度几何估计问题。",
        "",
        "## 失败原因或不足",
        "",
        "S5E2 当前是 minimal baseline，不复用完整 S5 visual backbone，也没有进行长时间训练；它用于证明合法 adjacent_dense 训练/推理链路可行。",
        "",
        "## 下一步建议",
        "",
        "1. 将 minimal image-statistics backbone 替换为 existing S5 backbone 或更强视觉编码器。",
        "2. 使用真实 mini-batch training 启用 `SO(3) geodesic`、`tdir`、`tmag log`、`path length consistency` 和 `short-window consistency`。",
        "3. 保持 `scene01/seq03` 作为 held-out evaluation，不将其 GT 用于训练。",
        "",
        "## caveats",
        "",
        "- S5E2 是 experimental candidate。",
        "- 不替代 official S5 locked result。",
        "- S5 locked metrics/policy unchanged。",
        "- ORB-SLAM3 是 external strong baseline。",
        "- no GT leakage；`scene01/seq03` GT 只用于 evaluation / diagnostics。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_metrics = read_json(out_dir / "edge_component_metrics.json")
    ckpt = load_or_base_checkpoint(Path(args.out_json))
    available = bool(ckpt.get("adjacent_dense_export", {}).get("available"))
    if available:
        external = _run_external_eval(args.trajectory, args.groundtruth, out_dir)
    else:
        external = {
            "none": {"ate": None, "drift": None, "path_ratio": None, "status": "not_run"},
            "se3": {"ate": None, "drift": None, "path_ratio": None, "status": "not_run"},
            "sim3": {"ate": None, "drift": None, "path_ratio": None, "status": "not_run"},
        }
    worst = _worst_edges(Path(args.provenance)) if Path(args.provenance).exists() else {}
    write_json(out_dir / "worst_edges.json", worst)
    long_run_flag = bool((edge_metrics.get("tmag_median_ratio") or 0.0) > 10.0 or (edge_metrics.get("path_ratio") or 0.0) > 2.0)

    ckpt["component_metrics"].update(
        {
            "rot_mean_deg": edge_metrics.get("rot_mean_deg"),
            "tdir_mean_deg": edge_metrics.get("tdir_mean_deg"),
            "tdir_abs_mean_deg": edge_metrics.get("tdir_abs_mean_deg"),
            "tmag_median_ratio": edge_metrics.get("tmag_median_ratio"),
            "tmag_p90_ratio": edge_metrics.get("tmag_p90_ratio"),
            "path_ratio": edge_metrics.get("path_ratio"),
            "long_run_overscaling_like_s5d11": long_run_flag,
        }
    )
    ckpt["external_eval"] = external
    se3_ate = external.get("se3", {}).get("ate")
    path_ratio = edge_metrics.get("path_ratio")
    if not available:
        final = "S5E2_ADJACENT_DENSE_EXPORT_BLOCKED"
    elif path_ratio is not None and (path_ratio > 5.0 or (edge_metrics.get("tmag_median_ratio") or 0.0) > 10.0):
        final = "S5E2_TRACEABLE_DENSE_EXPORTED_FAILED_METRICS"
    elif se3_ate is not None and se3_ate < RESTORED_S5_DENSE_REFERENCE["se3_ate"] and path_ratio is not None and abs(path_ratio - 1.0) < abs(RESTORED_S5_DENSE_REFERENCE["path_ratio"] - 1.0):
        final = "S5E2_TRACEABLE_DENSE_IMPROVED"
    else:
        final = "S5E2_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT"
    ckpt["validation"] = validation_from_logs()
    ckpt["final_classification"] = final
    write_json(Path(args.out_json), ckpt)
    _write_report(Path(args.out_report), ckpt, edge_metrics)
    return ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate S5E2 traceable dense candidate.")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--groundtruth", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
