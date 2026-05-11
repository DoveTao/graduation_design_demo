#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_PAIRWISE = Path("external_baselines/results/s5_pairwise_replay_s5d7/final_s5_pairwise_vectors_scene01_seq03.jsonl")
DEFAULT_OUT_JSON = Path("checkpoints/S5E1_traceable_adjacent_dense_candidate.json")
DEFAULT_REPORT = Path("reports/s5e1_traceable_adjacent_dense_candidate_report.md")

ALLOWED_CLASSIFICATIONS = [
    "S5E1_TRACEABLE_DENSE_IMPROVED",
    "S5E1_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT",
    "S5E1_ADJACENT_DENSE_EXPORT_BLOCKED",
    "S5E1_TRAINING_BLOCKED",
    "S5E1_EXPERIMENT_FAILED",
    "S5E1_ERROR",
]

OFFICIAL_S5_LOCKED = {"ate": 7.352288, "drift": 1.327343, "path_ratio": 0.932379}
ORBSLAM3_REFERENCE = {
    "coverage": "273/454",
    "se3_ate": 0.30854441069248173,
    "sim3_ate": 0.224292165986624,
    "path_ratio": 0.2998258665660257,
}


def _read_timestamps(path: Path) -> List[float]:
    values: List[float] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        values.append(float(line.split()[0]))
    if len(values) < 2:
        raise RuntimeError(f"Need at least two timestamps in {path}")
    return values


def _quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1.0e-12)
    x, y, z, w = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _read_gt(path: Path) -> Dict[float, Dict[str, np.ndarray]]:
    rows: Dict[float, Dict[str, np.ndarray]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        ts = float(parts[0])
        rows[ts] = {
            "t": np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64),
            "R": _quat_xyzw_to_rot(float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])),
        }
    return rows


def _read_pairwise(path: Path) -> Dict[int, Dict[str, Any]]:
    selected: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return selected
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        frame_i = int(row["frame_i"])
        frame_j = int(row["frame_j"])
        if frame_j != frame_i + 1:
            continue
        selected[frame_i] = row
    return selected


def _angle_deg_from_rot(delta: np.ndarray) -> float:
    cos_theta = (float(np.trace(delta)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))


def _vector_angle_deg(a: np.ndarray, b: np.ndarray, absolute: bool = False) -> Optional[float]:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1.0e-12 or nb <= 1.0e-12:
        return None
    cos_theta = float(np.dot(a, b) / (na * nb))
    if absolute:
        cos_theta = abs(cos_theta)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))


def _percentile(values: Iterable[float], q: float) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _edge_metric(row: Dict[str, Any], gt_i: Dict[str, np.ndarray], gt_j: Dict[str, np.ndarray]) -> Dict[str, Any]:
    pred_R = np.asarray(row["rotation"]["value"], dtype=np.float64)
    pred_t = np.asarray(row["translation"]["value"], dtype=np.float64)
    gt_R_rel = gt_i["R"].T @ gt_j["R"]
    gt_t_rel = gt_i["R"].T @ (gt_j["t"] - gt_i["t"])
    pred_norm = float(np.linalg.norm(pred_t))
    gt_norm = float(np.linalg.norm(gt_t_rel))
    tdir = _vector_angle_deg(pred_t, gt_t_rel, absolute=False)
    tdir_abs = _vector_angle_deg(pred_t, gt_t_rel, absolute=True)
    return {
        "rot_deg": _angle_deg_from_rot(pred_R @ gt_R_rel.T),
        "tdir_deg": tdir,
        "tdir_abs_deg": tdir_abs,
        "tmag_ratio": pred_norm / gt_norm if gt_norm > 1.0e-12 else None,
        "pred_step_length": pred_norm,
        "gt_step_length": gt_norm,
    }


def _validation_from_logs() -> Dict[str, str]:
    log_map = {
        "verify_final_candidate": Path("logs/s5d11_verify_final_candidate_serial.log"),
        "project_health_check": Path("logs/s5d11_project_health_check_serial.log"),
        "s6_eval_only": Path("logs/s5d11_s6_eval_only_serial.log"),
        "unittest": Path("logs/s5d11_unittest.log"),
    }
    out: Dict[str, str] = {}
    for key, path in log_map.items():
        if not path.exists():
            out[key] = "not_run"
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        out[key] = "FAIL" if "FAILED" in text or "Traceback" in text or "ERROR" in text else "PASS"
    return out


def _base_checkpoint() -> Dict[str, Any]:
    return {
        "experiment": "S5E1_traceable_adjacent_dense_candidate",
        "status": {
            "official_s5_unchanged": True,
            "experimental_candidate": True,
            "not_official_replacement": True,
        },
        "baseline_reference": {
            "s5_official_locked": OFFICIAL_S5_LOCKED,
            "orbslam3": ORBSLAM3_REFERENCE,
        },
        "adjacent_dense_export": {
            "available": False,
            "trajectory_path": "external_baselines/results/s5e1_traceable_dense/scene01_seq03_s5e1_traceable_dense_tum.txt",
            "edge_provenance": "external_baselines/results/s5e1_traceable_dense/edge_provenance.jsonl",
            "num_poses": None,
            "num_edges": None,
            "coverage": None,
            "all_edges_traceable": False,
        },
        "training_or_finetune": {
            "attempted": False,
            "config": "configs/s5e1_geometry_candidate.yaml",
            "checkpoint_dir": "checkpoints/S5E1_geometry_candidate",
            "losses": {
                "so3_geodesic": True,
                "tdir": True,
                "tmag_log": True,
                "path_length_consistency": True,
                "short_window_consistency": True,
            },
            "notes": [],
        },
        "component_metrics": {
            "rot_mean_deg": None,
            "tdir_mean_deg": None,
            "tdir_abs_mean_deg": None,
            "tmag_median_ratio": None,
            "tmag_p90_ratio": None,
            "path_ratio": None,
        },
        "external_eval": {
            "none": {"ate": None, "drift": None, "path_ratio": None},
            "se3": {"ate": None, "drift": None, "path_ratio": None},
            "sim3": {"ate": None, "drift": None, "path_ratio": None},
        },
        "comparison_to_orbslam3": {
            "closer_to_orbslam3_than_restored_s5_dense": None,
            "coverage_advantage_over_orbslam3": None,
            "aligned_ate_gap_to_orbslam3": None,
            "summary": "",
        },
        "validation": _validation_from_logs(),
        "allowed_classifications": ALLOWED_CLASSIFICATIONS,
        "s5_official_locked_metrics": {
            "ate": 7.352288,
            "drift": 1.327343,
            "path_ratio": 0.932379,
            "unchanged": True,
            "not_replaced_by_s5e1": True,
        },
        "final_classification": "S5E1_ADJACENT_DENSE_EXPORT_BLOCKED",
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_report(path: Path, checkpoint: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# S5E1 traceable adjacent dense candidate 报告",
        "",
        "## 执行摘要",
        "",
        "S5E1 尝试为 final S5 建立 traceable adjacent-dense prediction path。当前合法输入仍只能追溯到 `selected_k1` 的 132 条相邻 pairwise 预测，不能生成 453 条完整 adjacent dense predictions。因此本轮没有输出可作为 trajectory 的新 dense TUM，`final_classification` 为 `S5E1_ADJACENT_DENSE_EXPORT_BLOCKED`。",
        "",
        "本报告只描述 experimental candidate 的可用性诊断，不替代 S5 official locked result，也不修改 S5 policy / locked metrics / final manifest / train-test split / official evaluator。",
        "",
        "## 为什么需要 S5E1",
        "",
        "S5D13 已确认 final S5 当前 traceable 输出是 `selected_k1` selected-only：132 条可追溯预测、321 条 unavailable，不能合法生成 453 条 adjacent dense predictions。S5D11/S5D12 还发现 restored dense artifact 的 non-selected long runs 存在严重 `tmag` over-scaling，因此 S5E1 的目标是启动一个新的实验路径，而不是复用或包装 restored dense artifact。",
        "",
        "## S5D13 selected-only blocker 回顾",
        "",
        "- `direct_adjacent_prediction = 0`",
        "- `selected_prediction = 132`",
        "- `unavailable = 321`",
        "- `can_generate_453_adjacent_edges = false`",
        "",
        "## 新增 adjacent-dense prediction path / training path",
        "",
        "`tools/export_s5e1_adjacent_dense_predictions.py` 会输出 453 条 edge provenance，其中真实 S5 模型输出仅限 selected_k1 预测；其余 adjacent edges 明确标记为 `unavailable`。`tools/train_s5e1_geometry_candidate.py` 定义 experimental geometry candidate 的 loss 组合，但在没有合法 adjacent-dense inference/training harness 时只记录 blocked 状态。",
        "",
        "## 使用的 loss 和 config",
        "",
        "`configs/s5e1_geometry_candidate.yaml` 声明了 `SO(3) geodesic`、`tdir`、`tmag log`、`path length consistency` 和 `short-window consistency`。这些 loss 是下一步训练路径的实验设计，本轮未产生新模型权重。",
        "",
        "## traceable dense export coverage",
        "",
        f"- num_edges = {checkpoint['adjacent_dense_export']['num_edges']}",
        f"- available = {checkpoint['adjacent_dense_export']['available']}",
        f"- coverage = {checkpoint['adjacent_dense_export']['coverage']}",
        f"- all_edges_traceable = {checkpoint['adjacent_dense_export']['all_edges_traceable']}",
        "",
        "## component metrics",
        "",
        f"- selected rot_mean_deg = {metrics.get('selected', {}).get('rot_mean_deg')}",
        f"- selected tdir_mean_deg = {metrics.get('selected', {}).get('tdir_mean_deg')}",
        f"- selected tdir_abs_mean_deg = {metrics.get('selected', {}).get('tdir_abs_mean_deg')}",
        f"- selected tmag_median_ratio = {metrics.get('selected', {}).get('tmag_median_ratio')}",
        "- full dense path_ratio = null（未生成合法完整 dense trajectory）",
        "",
        "## external evaluator none/se3/sim3 结果",
        "",
        "未运行 external evaluator，因为 S5E1 没有合法完整 TUM trajectory。强行对空或不完整 trajectory 计算 ATE / drift / path_ratio 会产生误导性指标。",
        "",
        "## 与 ORB-SLAM3 对比",
        "",
        "ORB-SLAM3 仍作为 verified external strong baseline：coverage 为 273/454，`se3` ATE 为 0.30854441069248173，`sim3` ATE 为 0.224292165986624。S5E1 本轮未生成 traceable dense trajectory，因此不能声称接近或超过 ORB-SLAM3。",
        "",
        "## 是否接近 ORB-SLAM3",
        "",
        "不能判断。当前 blocker 是 adjacent-dense export 不可用，而不是几何指标已经改善或退化。",
        "",
        "## 失败或不足原因",
        "",
        "final S5 的可追溯 artifact 仍是 selected_k1 sparse/disconnected protocol。没有 direct adjacent dense model output，也没有 documented fill logic 可以合法 materialize 453 条相邻边。",
        "",
        "## 下一步建议",
        "",
        "1. 新增独立的 experimental adjacent-dense dataloader/inference head，确保每个相邻 edge 的 prediction 都来自模型输出。",
        "2. 在训练配置中启用 `SO(3) geodesic`、`tdir`、`tmag log`、`path length consistency` 和 `short-window consistency`。",
        "3. 先在不改变 official S5 policy 的实验目录中跑通小规模训练，再进行 external evaluator 对比。",
        "",
        "## Caveats",
        "",
        "- S5E1 是 experimental candidate。",
        "- S5E1 不替代 official S5 locked result。",
        "- S5 locked metrics/policy unchanged。",
        "- ORB-SLAM3 仍是 external strong baseline。",
        "- no GT used for prediction；GT 仅用于 evaluation diagnostics。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    timestamps = _read_timestamps(Path(args.timestamps))
    gt_rows = _read_gt(Path(args.groundtruth))
    selected = _read_pairwise(Path(args.pairwise_jsonl))
    out_prov = Path(args.out_provenance)
    out_tum = Path(args.out_tum)
    out_metrics = Path(args.out_metrics)
    out_prov.parent.mkdir(parents=True, exist_ok=True)
    out_tum.parent.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict[str, Any]] = []
    provenance_rows: List[Dict[str, Any]] = []
    for edge_index in range(len(timestamps) - 1):
        ts_i = timestamps[edge_index]
        ts_j = timestamps[edge_index + 1]
        row = selected.get(edge_index)
        if row is not None:
            edge_metrics = {}
            if ts_i in gt_rows and ts_j in gt_rows:
                edge_metrics = _edge_metric(row, gt_rows[ts_i], gt_rows[ts_j])
                metrics_rows.append({"edge_index": edge_index, **edge_metrics})
            provenance_rows.append(
                {
                    "edge_index": edge_index,
                    "timestamp_i": ts_i,
                    "timestamp_j": ts_j,
                    "source_type": "selected_prediction",
                    "source_model": row.get("candidate", "S5_clean_tmag_calibration_policy"),
                    "prediction_type": "selected_k1_pairwise",
                    "rotation": row.get("rotation"),
                    "rotation_representation": row.get("rotation", {}).get("representation"),
                    "translation_vector": row.get("translation", {}).get("value"),
                    "translation_frame": row.get("translation", {}).get("frame", "local"),
                    "uses_gt_for_prediction": False,
                    "metric_preview": edge_metrics,
                    "notes": ["traceable selected_k1 prediction; not a full adjacent-dense export"],
                }
            )
        else:
            provenance_rows.append(
                {
                    "edge_index": edge_index,
                    "timestamp_i": ts_i,
                    "timestamp_j": ts_j,
                    "source_type": "unavailable",
                    "source_model": None,
                    "prediction_type": None,
                    "rotation": None,
                    "rotation_representation": None,
                    "translation_vector": None,
                    "translation_frame": "unknown",
                    "uses_gt_for_prediction": False,
                    "notes": [
                        "no direct adjacent prediction available",
                        "not filled from restored dense artifact",
                        "not filled from GT",
                    ],
                }
            )

    out_prov.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in provenance_rows) + "\n", encoding="utf-8")
    out_tum.write_text(
        "# S5E1 adjacent dense export unavailable: no legal 453-edge direct prediction source.\n",
        encoding="utf-8",
    )

    tmag_values = [m["tmag_ratio"] for m in metrics_rows if m.get("tmag_ratio") is not None]
    selected_summary = {
        "num_edges": len(metrics_rows),
        "rot_mean_deg": _mean(m["rot_deg"] for m in metrics_rows),
        "rot_median_deg": _percentile((m["rot_deg"] for m in metrics_rows), 50),
        "rot_p90_deg": _percentile((m["rot_deg"] for m in metrics_rows), 90),
        "tdir_mean_deg": _mean(m["tdir_deg"] for m in metrics_rows),
        "tdir_median_deg": _percentile((m["tdir_deg"] for m in metrics_rows), 50),
        "tdir_p90_deg": _percentile((m["tdir_deg"] for m in metrics_rows), 90),
        "tdir_abs_mean_deg": _mean(m["tdir_abs_deg"] for m in metrics_rows),
        "tdir_abs_median_deg": _percentile((m["tdir_abs_deg"] for m in metrics_rows), 50),
        "tdir_abs_p90_deg": _percentile((m["tdir_abs_deg"] for m in metrics_rows), 90),
        "tmag_median_ratio": _percentile(tmag_values, 50),
        "tmag_mean_ratio": _mean(tmag_values),
        "tmag_p90_ratio": _percentile(tmag_values, 90),
        "tmag_p95_ratio": _percentile(tmag_values, 95),
        "tmag_max_ratio": max(tmag_values) if tmag_values else None,
    }
    source_counts: Dict[str, int] = {}
    for row in provenance_rows:
        source_counts[row["source_type"]] = source_counts.get(row["source_type"], 0) + 1
    metrics = {
        "experiment": "S5E1_traceable_adjacent_dense_candidate",
        "can_export_adjacent_dense_now": False,
        "reason": "Current final S5 traceable source is selected_k1 only; no legal direct adjacent-dense prediction source was found.",
        "source_counts": source_counts,
        "selected": selected_summary,
        "all_edges_traceable": False,
        "trajectory_generated": False,
        "no_gt_used_for_prediction": True,
    }
    _write_json(out_metrics, metrics)

    checkpoint = _base_checkpoint()
    existing_checkpoint_path = Path(args.out_json)
    if existing_checkpoint_path.exists():
        existing = json.loads(existing_checkpoint_path.read_text(encoding="utf-8"))
        for key in ["training_or_finetune", "comparison_to_orbslam3"]:
            if key in existing:
                checkpoint[key] = existing[key]
    checkpoint["adjacent_dense_export"].update(
        {
            "available": False,
            "trajectory_path": str(out_tum),
            "edge_provenance": str(out_prov),
            "num_poses": 0,
            "num_edges": len(timestamps) - 1,
            "coverage": len(metrics_rows) / max(len(timestamps) - 1, 1),
            "all_edges_traceable": False,
            "can_export_adjacent_dense_now": False,
            "source_counts": source_counts,
            "blocked_reason": metrics["reason"],
        }
    )
    checkpoint["component_metrics"].update(
        {
            "rot_mean_deg": selected_summary["rot_mean_deg"],
            "tdir_mean_deg": selected_summary["tdir_mean_deg"],
            "tdir_abs_mean_deg": selected_summary["tdir_abs_mean_deg"],
            "tmag_median_ratio": selected_summary["tmag_median_ratio"],
            "tmag_p90_ratio": selected_summary["tmag_p90_ratio"],
            "path_ratio": None,
        }
    )
    _write_json(Path(args.out_json), checkpoint)
    _write_report(Path(args.out_report), checkpoint, metrics)
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export S5E1 traceable adjacent dense predictions when legally available.")
    parser.add_argument("--scene", default="scene01")
    parser.add_argument("--seq", default="seq03")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--timestamps", required=True)
    parser.add_argument("--groundtruth", required=True)
    parser.add_argument("--pairwise-jsonl", default=str(DEFAULT_PAIRWISE))
    parser.add_argument("--out-tum", required=True)
    parser.add_argument("--out-provenance", required=True)
    parser.add_argument("--out-metrics", required=True)
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-report", default=str(DEFAULT_REPORT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
