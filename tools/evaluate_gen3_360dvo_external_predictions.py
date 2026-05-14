#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from miniyaml import load_yaml_like
from s5e2_adjacent_dense_lib import write_json


ALLOWED_FINAL_CLASSIFICATIONS = [
    "GEN3_360DVO_EXTERNAL_EVAL_READY",
    "GEN3_360DVO_COMPONENT_METRICS_READY",
    "GEN3_360DVO_TRAJECTORY_EVAL_READY",
    "GEN3_INFERENCE_BRIDGE_BLOCKED",
    "GEN3_MODEL_WEIGHTS_MISSING",
    "GEN3_SCENE_SPECIFIC_EXPORT_ONLY",
    "GEN3_INPUT_PROTOCOL_UNSUPPORTED",
    "GEN3_EVAL_GT_DEPENDENCY_DETECTED",
    "GEN3_ERROR",
]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_config(path: Path) -> Dict[str, Any]:
    return load_yaml_like(path) or {}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_config(Path(args.config))
    results_dir = Path(args.results_dir)
    availability = _read_json(results_dir / "inference_availability_audit.json")
    export_summary = _read_json(results_dir / "export_summary.json")
    metrics = _read_json(results_dir / "edge_component_metrics.json")
    predictions = _read_jsonl(results_dir / "predictions.jsonl")
    manifest = _read_jsonl(Path(str(cfg.get("dataset", {}).get("manifest"))))
    motion = _read_json(Path(str(cfg.get("dataset", {}).get("motion_audit"))))

    total = len(manifest)
    pred_count = len(predictions)
    adj_total = sum(1 for r in manifest if r.get("pair_type") == "adjacent")
    k_total = sum(1 for r in manifest if r.get("pair_type") == "kstep")
    adj_pred = sum(1 for r in predictions if r.get("pair_type") == "adjacent")
    k_pred = sum(1 for r in predictions if r.get("pair_type") == "kstep")

    bridge_cls = str(availability.get("bridge_classification") or "")
    if bridge_cls == "SCENE_SPECIFIC_EXPORT_ONLY":
        final = "GEN3_SCENE_SPECIFIC_EXPORT_ONLY"
    elif bridge_cls == "MODEL_WEIGHTS_MISSING":
        final = "GEN3_MODEL_WEIGHTS_MISSING"
    elif bridge_cls == "EVAL_GT_DEPENDENCY_DETECTED":
        final = "GEN3_EVAL_GT_DEPENDENCY_DETECTED"
    elif bridge_cls == "INPUT_PROTOCOL_UNSUPPORTED":
        final = "GEN3_INPUT_PROTOCOL_UNSUPPORTED"
    elif export_summary.get("model_eval_available"):
        final = "GEN3_360DVO_EXTERNAL_EVAL_READY"
    else:
        final = "GEN3_INFERENCE_BRIDGE_BLOCKED"

    checkpoint = {
        "experiment": "GEN3_360DVO_external_inference_bridge_eval",
        "dataset": "360DVO",
        "sequence": str(cfg.get("dataset", {}).get("sequence")),
        "source_candidate": str(cfg.get("model", {}).get("source_candidate")),
        "training": {
            "train_new_model": False,
            "fine_tune": False,
            "uses_360dvo_gt_for_calibration": False,
            "uses_orbslam3_teacher": False,
        },
        "inference_availability": availability,
        "export": {
            "model_eval_available": bool(export_summary.get("model_eval_available")),
            "num_pairs_predicted": pred_count,
            "num_pairs_total": total,
            "pair_coverage": float(pred_count / max(total, 1)),
            "adjacent_pair_coverage": float(adj_pred / max(adj_total, 1)),
            "kstep_pair_coverage": float(k_pred / max(k_total, 1)),
            "eval_blocker": export_summary.get("eval_blocker"),
        },
        "component_metrics": metrics.get("component_metrics", {}),
        "trajectory_eval": {
            "t_path": str(results_dir / "scene_field_s5e15_like_est_tum.txt"),
            "none": None,
            "se3": None,
            "sim3": None,
            "path_ratio": None,
        },
        "comparison_to_data2": {
            "gen2b_motion_summary": motion,
            "avoids_small_motion_risk": bool(motion.get("small_motion_fraction") == 0.0),
            "s5e15_scene01_seq03_reference": _read_json(Path("checkpoints/S5E15_scale_deunderfit_antiparallel_candidate.json")).get("component_metrics", {}).get("overall", {}),
        },
        "recommendation": {
            "keep_s5e15_as_best_candidate": True,
            "do_gen4_external_expansion": bool(availability.get("bridge_classification") == "SCENE_SPECIFIC_EXPORT_ONLY"),
            "do_external_training": False,
            "main_next_step": "build_true_image_pair_external_bridge_or_reuse_lower_level_npz_models" if not export_summary.get("model_eval_available") else "expand_external_sequences",
        },
        "s5_locked_metrics_policy_unchanged": True,
        "final_classification": final,
    }
    write_json(Path(args.out_json), checkpoint)

    report = [
        "# GEN3 360DVO external inference bridge eval",
        "",
        "## 1. 为什么做 GEN3",
        "GEN2B 已经证明 360DVO field one-sequence 的数据接入、pose convention 和 motion audit 都可以成立，因此 GEN3 的目标是确认 S5E15-like 路径能否真实对外部 ERP pair 做推理。",
        "",
        "## 2. GEN2B 数据和 motion audit 摘要",
        json.dumps(motion, ensure_ascii=False, indent=2),
        "",
        "## 3. S5E15-like inference availability audit",
        json.dumps(availability, ensure_ascii=False, indent=2),
        "",
        "## 4. 是否能真实预测 360DVO pair",
        json.dumps(checkpoint["export"], ensure_ascii=False, indent=2),
        "",
        "## 5. 如果 blocked，blocker 是什么",
        json.dumps({"bridge_classification": bridge_cls, "eval_blocker": export_summary.get("eval_blocker")}, ensure_ascii=False, indent=2),
        "",
        "## 6. 如果成功，component metrics",
        json.dumps(checkpoint["component_metrics"], ensure_ascii=False, indent=2),
        "",
        "## 7. 如果成功，trajectory / path_ratio / ATE",
        json.dumps(checkpoint["trajectory_eval"], ensure_ascii=False, indent=2),
        "",
        "## 8. 与 DATA2 scene01/seq03 的解释性对比",
        json.dumps(checkpoint["comparison_to_data2"], ensure_ascii=False, indent=2),
        "",
        "## 9. 是否进入 GEN4",
        json.dumps(checkpoint["recommendation"], ensure_ascii=False, indent=2),
        "",
        "## 10. caveats",
        "- 本轮不训练、不 fine-tune、不生成新训练候选。",
        "- 不使用 360DVO GT 做 calibration。",
        "- 如果 inference bridge 不可用，绝不伪造 prediction。",
    ]
    Path(args.out_report).write_text("\n".join(report) + "\n", encoding="utf-8")
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
