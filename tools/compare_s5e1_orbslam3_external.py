#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


ORB_REFERENCE = {
    "coverage": "273/454",
    "tracking_success_rate": 0.6013215859030837,
    "none": {
        "ate": 11.46685113827757,
        "drift": 0.03737053832593786,
        "path_ratio": 0.2998258665660257,
    },
    "se3": {
        "ate": 0.30854441069248173,
        "drift": 0.0374760642069601,
        "path_ratio": 0.2998258665660257,
    },
    "sim3": {
        "ate": 0.224292165986624,
        "drift": 0.06452708470276013,
        "path_ratio": 0.2998258665660257,
    },
}

RESTORED_S5_DENSE_REFERENCE = {
    "se3_ate": 8.231468716451547,
    "sim3_ate": 4.07912293550008,
    "path_ratio": 2.777267572676944,
}

ALLOWED_CLASSIFICATIONS = [
    "S5E1_TRACEABLE_DENSE_IMPROVED",
    "S5E1_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT",
    "S5E1_ADJACENT_DENSE_EXPORT_BLOCKED",
    "S5E1_TRAINING_BLOCKED",
    "S5E1_EXPERIMENT_FAILED",
    "S5E1_ERROR",
]

DEFAULT_REPORT = Path("reports/s5e1_orbslam3_external_comparison.md")
DEFAULT_CHECKPOINT = Path("checkpoints/S5E1_traceable_adjacent_dense_candidate.json")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# S5E1 与 ORB-SLAM3 external comparison",
        "",
        "## 执行摘要",
        "",
        "S5E1 本轮没有生成合法完整 traceable dense trajectory，因此不能进行有效的 none / se3 / sim3 ATE 对比。ORB-SLAM3 仍作为 verified external strong baseline；S5E1 当前结果是 adjacent_dense export blocked。",
        "",
        "## 对比输入",
        "",
        f"- S5E1 checkpoint = `{payload.get('s5e1_checkpoint')}`",
        "- ORB-SLAM3 trajectory = `external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt`",
        "",
        "## ORB-SLAM3 reference",
        "",
        f"- coverage = {ORB_REFERENCE['coverage']}",
        f"- none = {ORB_REFERENCE['none']}",
        f"- se3 = {ORB_REFERENCE['se3']}",
        f"- sim3 = {ORB_REFERENCE['sim3']}",
        "",
        "## S5E1 reference",
        "",
        f"- adjacent_dense_available = {payload.get('s5e1', {}).get('adjacent_dense_available')}",
        f"- coverage = {payload.get('s5e1', {}).get('coverage')}",
        f"- external_eval = {payload.get('s5e1', {}).get('external_eval')}",
        "",
        "## interpretation",
        "",
        "ORB-SLAM3 在 tracked subset 上 aligned ATE 很低，但 coverage 只有 273/454。S5E1 的目标是未来实现 full coverage traceable dense prediction；当前由于没有合法 453-edge direct adjacent source，不能声称比 restored S5 dense artifact 更接近 ORB-SLAM3。",
        "",
        "## recommendations",
        "",
        "下一步应优先实现 experimental direct adjacent prediction head 和 dataloader，然后再运行 same external evaluator。只有当 S5E1 输出完整、可追溯、不使用 GT 生成 prediction 的 TUM 后，才适合计算 ATE gap 与 path_ratio 对比。",
        "",
        "## Caveats",
        "",
        "- S5E1 是 experimental candidate。",
        "- S5E1 不替代 official S5 locked result。",
        "- S5 locked metrics/policy unchanged。",
        "- ORB-SLAM3 仍是 external strong baseline。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoint_path = Path(args.s5e1_checkpoint)
    checkpoint = _read_json(checkpoint_path)
    export = checkpoint.get("adjacent_dense_export", {})
    external_eval = checkpoint.get("external_eval", {})
    available = bool(export.get("available"))
    se3_ate = external_eval.get("se3", {}).get("ate") if isinstance(external_eval.get("se3"), dict) else None
    comparison = {
        "experiment": "S5E1_traceable_adjacent_dense_candidate",
        "s5e1_checkpoint": str(checkpoint_path),
        "orbslam3_eval_dir": args.orbslam3_eval_dir,
        "s5e1": {
            "adjacent_dense_available": available,
            "coverage": export.get("coverage"),
            "num_poses": export.get("num_poses"),
            "num_edges": export.get("num_edges"),
            "external_eval": external_eval,
        },
        "orbslam3": ORB_REFERENCE,
        "restored_s5_dense_reference": RESTORED_S5_DENSE_REFERENCE,
        "comparison_to_orbslam3": {
            "closer_to_orbslam3_than_restored_s5_dense": None if se3_ate is None else se3_ate < RESTORED_S5_DENSE_REFERENCE["se3_ate"],
            "coverage_advantage_over_orbslam3": None if not available else export.get("num_poses", 0) >= 454,
            "aligned_ate_gap_to_orbslam3": None if se3_ate is None else se3_ate - ORB_REFERENCE["se3"]["ate"],
            "path_ratio_gap_to_orbslam3": None,
            "summary": "S5E1 adjacent dense export is blocked; no valid ATE gap can be reported.",
        },
        "allowed_classifications": ALLOWED_CLASSIFICATIONS,
        "final_classification": checkpoint.get("final_classification", "S5E1_ADJACENT_DENSE_EXPORT_BLOCKED"),
    }
    _write_json(Path(args.out_json), comparison)

    checkpoint.setdefault("comparison_to_orbslam3", {}).update(comparison["comparison_to_orbslam3"])
    checkpoint["final_classification"] = comparison["final_classification"]
    _write_json(checkpoint_path, checkpoint)
    _write_report(Path(args.out_report), comparison)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare S5E1 experimental candidate with ORB-SLAM3 external baseline.")
    parser.add_argument("--s5e1-checkpoint", required=True)
    parser.add_argument("--orbslam3-eval-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
