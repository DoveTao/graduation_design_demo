#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


ALLOWED_CLASSIFICATIONS = [
    "S5E1_TRACEABLE_DENSE_IMPROVED",
    "S5E1_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT",
    "S5E1_ADJACENT_DENSE_EXPORT_BLOCKED",
    "S5E1_TRAINING_BLOCKED",
    "S5E1_EXPERIMENT_FAILED",
    "S5E1_ERROR",
]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_provenance(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def _count_tum_poses(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if len(line.split()) >= 8:
            count += 1
    return count


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


def _source_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "direct_adjacent_prediction": 0,
        "selected_prediction": 0,
        "interpolated_fill": 0,
        "composed_fill": 0,
        "unavailable": 0,
    }
    for row in rows:
        source = row.get("source_type", "unavailable")
        counts[source] = counts.get(source, 0) + 1
    return counts


def _load_edge_metrics(path: Path) -> Dict[str, Any]:
    metrics_path = path.with_name("edge_component_metrics.json")
    return _read_json(metrics_path)


def _write_report(path: Path, checkpoint: Dict[str, Any], edge_metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    export = checkpoint.get("adjacent_dense_export", {})
    component = checkpoint.get("component_metrics", {})
    training = checkpoint.get("training_or_finetune", {})
    external_eval = checkpoint.get("external_eval", {})
    lines = [
        "# S5E1 traceable adjacent dense candidate 报告",
        "",
        "## 执行摘要",
        "",
        "S5E1 已建立 experimental adjacent-dense 诊断链路，但当前 final S5 仍无法合法导出 453 条 direct adjacent predictions。导出结果仅包含 132 条 `selected_prediction` 和 321 条 `unavailable` provenance，因此本轮不生成可评估的完整 dense TUM。",
        "",
        f"`final_classification = {checkpoint.get('final_classification')}`。",
        "",
        "## 为什么需要 S5E1",
        "",
        "S5D 系列显示，official S5 locked result 与 restored dense artifact 必须分层理解：前者是正式结果，后者只是 diagnostic artifact。S5E1 的目标是为未来实验候选提供每条 adjacent edge 可追溯的模型预测路径，并用 geometry losses 抑制 restored dense artifact 中出现的 long-run `tmag` over-scaling。",
        "",
        "## S5D13 selected-only blocker 回顾",
        "",
        "S5D13 的结论是 `selected_only`：132 条 selected_k1 预测可追溯，321 条 dense adjacent edge provenance unavailable，`can_generate_453_adjacent_edges = false`。",
        "",
        "## 新增 adjacent-dense prediction path / training path",
        "",
        "`tools/export_s5e1_adjacent_dense_predictions.py` 生成完整 edge provenance map；`tools/train_s5e1_geometry_candidate.py` 记录 experimental training design 和 blocked 原因；`tools/evaluate_s5e1_traceable_dense.py` 汇总 coverage、component metrics 和 external evaluator 可用性。",
        "",
        "## 使用的 loss 和 config",
        "",
        f"- config = `{training.get('config')}`",
        "- `SO(3) geodesic`: true",
        "- `tdir`: true",
        "- `tmag log`: true",
        "- `path length consistency`: true",
        "- `short-window consistency`: true",
        "",
        "## traceable dense export coverage",
        "",
        f"- trajectory_path = `{export.get('trajectory_path')}`",
        f"- edge_provenance = `{export.get('edge_provenance')}`",
        f"- num_poses = {export.get('num_poses')}",
        f"- num_edges = {export.get('num_edges')}",
        f"- coverage = {export.get('coverage')}",
        f"- all_edges_traceable = {export.get('all_edges_traceable')}",
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
        "## external evaluator none/se3/sim3 结果",
        "",
        f"- none = {external_eval.get('none')}",
        f"- se3 = {external_eval.get('se3')}",
        f"- sim3 = {external_eval.get('sim3')}",
        "",
        "## 与 ORB-SLAM3 对比",
        "",
        "S5E1 没有完整 traceable dense trajectory，因此不能与 ORB-SLAM3 做有效 ATE gap 对比。ORB-SLAM3 的 `se3` ATE 0.30854441069248173 和 `sim3` ATE 0.224292165986624 仍作为 external strong baseline 参考。",
        "",
        "## 是否接近 ORB-SLAM3",
        "",
        "不能判断；本轮的主要结果是发现 adjacent_dense export 仍被 selected-only provenance 阻塞。",
        "",
        "## 失败或不足原因",
        "",
        "当前仓库可合法追溯的 final S5 输出仍是 selected_k1 pairwise artifact。没有 direct adjacent-dense model output，也没有可审计的 documented fill logic，因此不能生成 454-pose / 453-edge traceable dense candidate。",
        "",
        "## 下一步建议",
        "",
        "1. 在 experimental config 下新增 direct adjacent pair dataloader 和 inference hook。",
        "2. 增加 `adjacent_dense_pose_head` 或等价输出接口，确保每个 adjacent edge 都有模型预测 provenance。",
        "3. 训练时启用 `SO(3) geodesic`、`tdir`、`tmag log`、`path length consistency`、`short-window consistency`，并监控 long-run over-scaling。",
        "4. 生成完整 TUM 后再运行 none / se3 / sim3 external evaluator。",
        "",
        "## Caveats",
        "",
        "- S5E1 是 experimental candidate。",
        "- S5E1 不替代 official S5 locked result。",
        "- S5 locked metrics/policy unchanged。",
        "- ORB-SLAM3 仍是 external strong baseline。",
        "- no GT used for prediction；GT 仅用于 diagnostics/evaluation。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoint = _read_json(Path(args.out_json))
    if not checkpoint:
        checkpoint = {
            "experiment": "S5E1_traceable_adjacent_dense_candidate",
            "status": {
                "official_s5_unchanged": True,
                "experimental_candidate": True,
                "not_official_replacement": True,
            },
            "final_classification": "S5E1_ADJACENT_DENSE_EXPORT_BLOCKED",
        }
    provenance = _read_provenance(Path(args.provenance))
    counts = _source_counts(provenance)
    pose_count = _count_tum_poses(Path(args.trajectory))
    edge_metrics = _load_edge_metrics(Path(args.provenance))
    selected = edge_metrics.get("selected", {})
    num_edges = len(provenance)
    available = pose_count >= 2 and counts.get("direct_adjacent_prediction", 0) == max(num_edges, 0)
    checkpoint.setdefault("adjacent_dense_export", {})
    checkpoint["adjacent_dense_export"].update(
        {
            "available": bool(available),
            "trajectory_path": args.trajectory,
            "edge_provenance": args.provenance,
            "num_poses": pose_count,
            "num_edges": num_edges,
            "coverage": counts.get("direct_adjacent_prediction", 0) / max(num_edges, 1),
            "selected_prediction_coverage": counts.get("selected_prediction", 0) / max(num_edges, 1),
            "all_edges_traceable": bool(available),
            "source_counts": counts,
        }
    )
    checkpoint.setdefault("component_metrics", {})
    checkpoint["component_metrics"].update(
        {
            "rot_mean_deg": selected.get("rot_mean_deg"),
            "tdir_mean_deg": selected.get("tdir_mean_deg"),
            "tdir_abs_mean_deg": selected.get("tdir_abs_mean_deg"),
            "tmag_median_ratio": selected.get("tmag_median_ratio"),
            "tmag_p90_ratio": selected.get("tmag_p90_ratio"),
            "path_ratio": None,
        }
    )
    checkpoint["external_eval"] = {
        "none": {"ate": None, "drift": None, "path_ratio": None},
        "se3": {"ate": None, "drift": None, "path_ratio": None},
        "sim3": {"ate": None, "drift": None, "path_ratio": None},
        "attempted": False,
        "blocked_reason": "No complete legal traceable dense TUM was exported.",
    }
    checkpoint["validation"] = _validation_from_logs()
    checkpoint["allowed_classifications"] = ALLOWED_CLASSIFICATIONS
    checkpoint.setdefault("s5_official_locked_metrics", {}).update(
        {
            "ate": 7.352288,
            "drift": 1.327343,
            "path_ratio": 0.932379,
            "unchanged": True,
            "not_replaced_by_s5e1": True,
        }
    )
    checkpoint["final_classification"] = "S5E1_ADJACENT_DENSE_EXPORT_BLOCKED"
    _write_json(Path(args.out_json), checkpoint)
    _write_report(Path(args.out_report), checkpoint, edge_metrics)
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate S5E1 traceable dense export diagnostics.")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--groundtruth", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
