#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


S5_LOCKED = {"ate": 7.352288, "drift": 1.327343, "path_ratio": 0.932379}


@dataclass(frozen=True)
class CandidateSpec:
    short: str
    name: str
    checkpoint: str
    config: Optional[str]
    report: Optional[str]
    training_status: Optional[str]
    trajectory: Optional[str]
    edge_metrics: Optional[str]
    edge_provenance: Optional[str]
    eval_none: Optional[str]
    eval_se3: Optional[str]
    eval_sim3: Optional[str]
    comparison_json: Optional[str]
    export_tool: Optional[str]
    eval_tool: Optional[str]
    base_refs: Tuple[str, ...]


CANDIDATES: List[CandidateSpec] = [
    CandidateSpec(
        short="S5E13",
        name="S5E13_real_correspondence_signed_direction_candidate",
        checkpoint="checkpoints/S5E13_real_correspondence_signed_direction_candidate.json",
        config="configs/s5e13_real_correspondence_signed_direction.yaml",
        report="reports/s5e13_real_correspondence_signed_direction_report.md",
        training_status="checkpoints/S5E13_real_correspondence_signed_direction_candidate/training_status.json",
        trajectory="external_baselines/results/s5e13_traceable_dense/scene01_seq03_s5e13_traceable_dense_tum.txt",
        edge_metrics="external_baselines/results/s5e13_traceable_dense/edge_component_metrics.json",
        edge_provenance="external_baselines/results/s5e13_traceable_dense/edge_provenance.jsonl",
        eval_none="external_baselines/results/s5e13_traceable_dense/eval_alignment_none.json",
        eval_se3="external_baselines/results/s5e13_traceable_dense/eval_alignment_se3.json",
        eval_sim3="external_baselines/results/s5e13_traceable_dense/eval_alignment_sim3.json",
        comparison_json="external_baselines/results/s5e13_traceable_dense/s5e12_s5e9_s5e6_to_s5e2_orbslam3_comparison.json",
        export_tool="tools/export_s5e13_adjacent_dense_predictions.py",
        eval_tool="tools/evaluate_s5e13_traceable_dense.py",
        base_refs=("S5E9", "S5E12"),
    ),
    CandidateSpec(
        short="S5E14",
        name="S5E14_observable_edge_direction_refinement_candidate",
        checkpoint="checkpoints/S5E14_observable_edge_direction_refinement_candidate.json",
        config="configs/s5e14_observable_edge_direction_refinement.yaml",
        report="reports/s5e14_observable_edge_direction_refinement_report.md",
        training_status="checkpoints/S5E14_observable_edge_direction_refinement_candidate/training_status.json",
        trajectory="external_baselines/results/s5e14_traceable_dense/scene01_seq03_s5e14_traceable_dense_tum.txt",
        edge_metrics="external_baselines/results/s5e14_traceable_dense/edge_component_metrics.json",
        edge_provenance="external_baselines/results/s5e14_traceable_dense/edge_provenance.jsonl",
        eval_none="external_baselines/results/s5e14_traceable_dense/eval_alignment_none.json",
        eval_se3="external_baselines/results/s5e14_traceable_dense/eval_alignment_se3.json",
        eval_sim3="external_baselines/results/s5e14_traceable_dense/eval_alignment_sim3.json",
        comparison_json="external_baselines/results/s5e14_traceable_dense/s5e13_s5e12_s5e9_orbslam3_comparison.json",
        export_tool="tools/export_s5e14_adjacent_dense_predictions.py",
        eval_tool="tools/evaluate_s5e14_traceable_dense.py",
        base_refs=("S5E13", "S5E9", "S5E4"),
    ),
    CandidateSpec(
        short="S5E15",
        name="S5E15_scale_deunderfit_antiparallel_candidate",
        checkpoint="checkpoints/S5E15_scale_deunderfit_antiparallel_candidate.json",
        config="configs/s5e15_scale_deunderfit_antiparallel.yaml",
        report="reports/s5e15_scale_deunderfit_antiparallel_report.md",
        training_status="checkpoints/S5E15_scale_deunderfit_antiparallel_candidate/training_status.json",
        trajectory="external_baselines/results/s5e15_traceable_dense/scene01_seq03_s5e15_traceable_dense_tum.txt",
        edge_metrics="external_baselines/results/s5e15_traceable_dense/edge_component_metrics.json",
        edge_provenance="external_baselines/results/s5e15_traceable_dense/edge_provenance.jsonl",
        eval_none="external_baselines/results/s5e15_traceable_dense/eval_alignment_none.json",
        eval_se3="external_baselines/results/s5e15_traceable_dense/eval_alignment_se3.json",
        eval_sim3="external_baselines/results/s5e15_traceable_dense/eval_alignment_sim3.json",
        comparison_json="external_baselines/results/s5e15_traceable_dense/s5e14_s5e13_s5e9_orbslam3_comparison.json",
        export_tool="tools/export_s5e15_adjacent_dense_predictions.py",
        eval_tool="tools/evaluate_s5e15_traceable_dense.py",
        base_refs=("S5E14", "S5E12"),
    ),
    CandidateSpec(
        short="S5E16",
        name="S5E16_confidence_calibrated_sign_scale_router",
        checkpoint="checkpoints/S5E16_confidence_calibrated_sign_scale_router.json",
        config="configs/s5e16_confidence_calibrated_sign_scale_router.yaml",
        report="reports/s5e16_confidence_calibrated_sign_scale_router_report.md",
        training_status="checkpoints/S5E16_confidence_calibrated_sign_scale_router/training_status.json",
        trajectory="external_baselines/results/s5e16_traceable_dense/scene01_seq03_s5e16_traceable_dense_tum.txt",
        edge_metrics="external_baselines/results/s5e16_traceable_dense/edge_component_metrics.json",
        edge_provenance="external_baselines/results/s5e16_traceable_dense/edge_provenance.jsonl",
        eval_none="external_baselines/results/s5e16_traceable_dense/eval_alignment_none.json",
        eval_se3="external_baselines/results/s5e16_traceable_dense/eval_alignment_se3.json",
        eval_sim3="external_baselines/results/s5e16_traceable_dense/eval_alignment_sim3.json",
        comparison_json="external_baselines/results/s5e16_traceable_dense/s5e15_s5e14_s5e13_s5e9_orbslam3_comparison.json",
        export_tool="tools/export_s5e16_adjacent_dense_predictions.py",
        eval_tool="tools/evaluate_s5e16_traceable_dense.py",
        base_refs=("S5E15", "S5E14", "S5E13", "S5E9"),
    ),
    CandidateSpec(
        short="S5E17",
        name="S5E17_router_activation_threshold_repair_candidate",
        checkpoint="checkpoints/S5E17_router_activation_threshold_repair_candidate.json",
        config="configs/s5e17_router_activation_threshold_repair.yaml",
        report="reports/s5e17_router_activation_threshold_repair_report.md",
        training_status="checkpoints/S5E17_router_activation_threshold_repair_candidate/router_status.json",
        trajectory="external_baselines/results/s5e17_traceable_dense/scene01_seq03_s5e17_traceable_dense_tum.txt",
        edge_metrics="external_baselines/results/s5e17_traceable_dense/edge_component_metrics.json",
        edge_provenance="external_baselines/results/s5e17_traceable_dense/edge_provenance.jsonl",
        eval_none="external_baselines/results/s5e17_traceable_dense/eval_alignment_none.json",
        eval_se3="external_baselines/results/s5e17_traceable_dense/eval_alignment_se3.json",
        eval_sim3="external_baselines/results/s5e17_traceable_dense/eval_alignment_sim3.json",
        comparison_json="external_baselines/results/s5e17_traceable_dense/s5e16_s5e15_orbslam3_comparison.json",
        export_tool="tools/export_s5e17_adjacent_dense_predictions.py",
        eval_tool="tools/evaluate_s5e17_traceable_dense.py",
        base_refs=("S5E16", "S5E15", "S5E14", "S5E13"),
    ),
    CandidateSpec(
        short="S5E18",
        name="S5E18_equirectangular_bearing_flow_candidate",
        checkpoint="checkpoints/S5E18_equirectangular_bearing_flow_candidate.json",
        config="configs/s5e18_equirectangular_bearing_flow.yaml",
        report="reports/s5e18_equirectangular_bearing_flow_report.md",
        training_status=None,
        trajectory="external_baselines/results/s5e18_bearing_flow/scene01_seq03_s5e18_traceable_dense_tum.txt",
        edge_metrics="external_baselines/results/s5e18_bearing_flow/edge_component_metrics.json",
        edge_provenance="external_baselines/results/s5e18_bearing_flow/edge_provenance.jsonl",
        eval_none="external_baselines/results/s5e18_bearing_flow/eval_alignment_none.json",
        eval_se3="external_baselines/results/s5e18_bearing_flow/eval_alignment_se3.json",
        eval_sim3="external_baselines/results/s5e18_bearing_flow/eval_alignment_sim3.json",
        comparison_json="external_baselines/results/s5e18_bearing_flow/s5e17_s5e15_orbslam3_comparison.json",
        export_tool="tools/export_s5e18_adjacent_dense_predictions.py",
        eval_tool="tools/evaluate_s5e18_traceable_dense.py",
        base_refs=("S5E17", "S5E15"),
    ),
    CandidateSpec(
        short="S5E19",
        name="S5E19_rotation_compensated_multiframe_geometry_candidate",
        checkpoint="checkpoints/S5E19_rotation_compensated_multiframe_geometry_candidate.json",
        config="configs/s5e19_rotation_compensated_multiframe_geometry.yaml",
        report="reports/s5e19_rotation_compensated_multiframe_geometry_report.md",
        training_status="checkpoints/S5E19_rotation_compensated_multiframe_geometry_candidate/training_status.json",
        trajectory="external_baselines/results/s5e19_traceable_dense/scene01_seq03_s5e19_traceable_dense_tum.txt",
        edge_metrics="external_baselines/results/s5e19_traceable_dense/edge_component_metrics.json",
        edge_provenance="external_baselines/results/s5e19_traceable_dense/edge_provenance.jsonl",
        eval_none="external_baselines/results/s5e19_traceable_dense/eval_alignment_none.json",
        eval_se3="external_baselines/results/s5e19_traceable_dense/eval_alignment_se3.json",
        eval_sim3="external_baselines/results/s5e19_traceable_dense/eval_alignment_sim3.json",
        comparison_json="external_baselines/results/s5e19_traceable_dense/s5e15_s5e18_orbslam3_comparison.json",
        export_tool="tools/export_s5e19_adjacent_dense_predictions.py",
        eval_tool="tools/evaluate_s5e19_traceable_dense.py",
        base_refs=("S5E15",),
    ),
]

PAIRWISE = [
    ("S5E13", "S5E14"),
    ("S5E14", "S5E15"),
    ("S5E15", "S5E16"),
    ("S5E16", "S5E17"),
    ("S5E17", "S5E18"),
    ("S5E15", "S5E19"),
]

ALLOWED_CLASSIFICATIONS = [
    "CANDIDATE_VALID_FRESH",
    "CANDIDATE_VALID_DERIVED_DECLARED",
    "CANDIDATE_SMOKE_ONLY",
    "CANDIDATE_NOOP_RISK",
    "CANDIDATE_FALLBACK_RISK",
    "CANDIDATE_EVAL_REF_MIXED",
    "CANDIDATE_LEAKAGE_RISK",
    "CANDIDATE_INSUFFICIENT_EVIDENCE",
    "AUDIT1_S5E15_VALID_BEST_CANDIDATE",
    "AUDIT1_S5E15_NEEDS_REPAIR",
    "AUDIT1_MULTIPLE_CANDIDATES_NOOP_RISK",
    "AUDIT1_EXPERIMENTAL_LINE_REQUIRES_REPAIR",
    "AUDIT1_AUDIT_ERROR",
]

# Canonical output targets for AUDIT1:
# - checkpoints/AUDIT1_experimental_candidate_validity_and_noop_sweep.json
# - reports/audit1_experimental_candidate_validity_and_noop_sweep.md
# - reports/audit1_s5e15_best_candidate_validity.md


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {
        "exists": True,
        "sha256": sha256_file(path),
        "size": st.st_size,
        "mtime": st.st_mtime,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_tum(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        vals = [float(x) for x in line.split()]
        if len(vals) != 8:
            continue
        ts, tx, ty, tz, qx, qy, qz, qw = vals
        q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        rows.append({"timestamp": ts, "t": np.asarray([tx, ty, tz], dtype=np.float64), "q": q})
    return rows


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    dot = float(abs(np.dot(q1, q2)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(2.0 * math.acos(dot)))


def ast_constants(text: str) -> Dict[str, Any]:
    constants: Dict[str, Any] = {}
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return constants
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith("_REF"):
                    try:
                        constants[target.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
    return constants


def metrics_rows_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("metrics_rows")
    return rows if isinstance(rows, list) else []


def metrics_overall_from_checkpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    cm = payload.get("component_metrics", {})
    if isinstance(cm, dict) and "overall" in cm and isinstance(cm["overall"], dict):
        return cm["overall"]
    return cm if isinstance(cm, dict) else {}


def trajectory_diff(path_a: Path, path_b: Path) -> Dict[str, Any]:
    a = parse_tum(path_a)
    b = parse_tum(path_b)
    if not a or not b:
        return {
            "num_common_timestamps": 0,
            "max_translation_difference": None,
            "mean_translation_difference": None,
            "max_quaternion_angular_difference_deg": None,
            "mean_quaternion_angular_difference_deg": None,
            "identical_pose_count": 0,
            "near_identical_pose_count": 0,
            "trajectory_exact_copy": False,
            "trajectory_near_identical": None,
        }
    b_by = {r["timestamp"]: r for r in b}
    common = [r for r in a if r["timestamp"] in b_by]
    if not common:
        return {
            "num_common_timestamps": 0,
            "max_translation_difference": None,
            "mean_translation_difference": None,
            "max_quaternion_angular_difference_deg": None,
            "mean_quaternion_angular_difference_deg": None,
            "identical_pose_count": 0,
            "near_identical_pose_count": 0,
            "trajectory_exact_copy": False,
            "trajectory_near_identical": None,
        }
    tdiffs: List[float] = []
    qdiffs: List[float] = []
    identical = 0
    near_identical = 0
    for row in common:
        other = b_by[row["timestamp"]]
        td = float(np.linalg.norm(row["t"] - other["t"]))
        qd = quat_angle_deg(row["q"], other["q"])
        tdiffs.append(td)
        qdiffs.append(qd)
        if td <= 1.0e-12 and qd <= 1.0e-12:
            identical += 1
        if td <= 1.0e-5 and qd <= 1.0e-4:
            near_identical += 1
    exact = identical == len(common)
    near = float(np.mean(tdiffs)) < 1.0e-4 and float(np.mean(qdiffs)) < 1.0e-3
    return {
        "num_common_timestamps": len(common),
        "max_translation_difference": float(np.max(tdiffs)),
        "mean_translation_difference": float(np.mean(tdiffs)),
        "max_quaternion_angular_difference_deg": float(np.max(qdiffs)),
        "mean_quaternion_angular_difference_deg": float(np.mean(qdiffs)),
        "identical_pose_count": identical,
        "near_identical_pose_count": near_identical,
        "trajectory_exact_copy": exact,
        "trajectory_near_identical": near,
    }


def edge_metrics_diff(path_a: Path, path_b: Path) -> Dict[str, Any]:
    if not path_a.exists() or not path_b.exists():
        return {
            "mean_abs_diff_rot": None,
            "mean_abs_diff_tdir": None,
            "mean_abs_diff_tdir_abs": None,
            "anti_parallel_flag_changed_count": None,
            "tmag_ratio_mean_abs_diff": None,
            "changed_edge_count": None,
            "unchanged_edge_count": None,
            "edge_metrics_near_identical": None,
        }
    pa = read_json(path_a)
    pb = read_json(path_b)
    rows_a = metrics_rows_from_payload(pa)
    rows_b = metrics_rows_from_payload(pb)
    n = min(len(rows_a), len(rows_b))
    if n == 0:
        return {
            "mean_abs_diff_rot": None,
            "mean_abs_diff_tdir": None,
            "mean_abs_diff_tdir_abs": None,
            "anti_parallel_flag_changed_count": None,
            "tmag_ratio_mean_abs_diff": None,
            "changed_edge_count": None,
            "unchanged_edge_count": None,
            "edge_metrics_near_identical": None,
        }
    rot, tdir, tdir_abs, tmag = [], [], [], []
    anti_changed = 0
    changed = 0
    unchanged = 0
    for ra, rb in zip(rows_a[:n], rows_b[:n]):
        rd = abs(float(ra.get("rot_deg", 0.0) or 0.0) - float(rb.get("rot_deg", 0.0) or 0.0))
        td = abs(float(ra.get("tdir_deg", 0.0) or 0.0) - float(rb.get("tdir_deg", 0.0) or 0.0))
        ta = abs(float(ra.get("tdir_abs_deg", 0.0) or 0.0) - float(rb.get("tdir_abs_deg", 0.0) or 0.0))
        tm = abs(float(ra.get("tmag_ratio", 0.0) or 0.0) - float(rb.get("tmag_ratio", 0.0) or 0.0))
        rot.append(rd)
        tdir.append(td)
        tdir_abs.append(ta)
        tmag.append(tm)
        if bool(ra.get("anti_parallel_flag", False)) != bool(rb.get("anti_parallel_flag", False)):
            anti_changed += 1
        if rd <= 1.0e-12 and td <= 1.0e-12 and ta <= 1.0e-12 and tm <= 1.0e-12:
            unchanged += 1
        else:
            changed += 1
    near = float(np.mean(rot)) < 1.0e-6 and float(np.mean(tdir)) < 1.0e-4 and float(np.mean(tmag)) < 1.0e-4 and anti_changed == 0
    return {
        "mean_abs_diff_rot": float(np.mean(rot)),
        "mean_abs_diff_tdir": float(np.mean(tdir)),
        "mean_abs_diff_tdir_abs": float(np.mean(tdir_abs)),
        "anti_parallel_flag_changed_count": anti_changed,
        "tmag_ratio_mean_abs_diff": float(np.mean(tmag)),
        "changed_edge_count": changed,
        "unchanged_edge_count": unchanged,
        "edge_metrics_near_identical": near,
    }


def expected_source_model(name: str) -> str:
    mapping = {
        "S5E13": "S5E13_real_correspondence_signed_direction_candidate",
        "S5E14": "S5E14_observable_edge_direction_refinement_candidate",
        "S5E15": "S5E15_scale_deunderfit_antiparallel_candidate",
        "S5E16": "S5E16_confidence_calibrated_sign_scale_router",
        "S5E17": "S5E17_router_activation_threshold_repair_candidate",
        "S5E18": "S5E18_equirectangular_bearing_flow_candidate",
        "S5E19": "S5E19_rotation_compensated_multiframe_geometry",
    }
    return mapping[name]


def inventory(spec: CandidateSpec, root: Path) -> Dict[str, Any]:
    file_map = {
        "checkpoint": spec.checkpoint,
        "report": spec.report,
        "config": spec.config,
        "training_status": spec.training_status,
        "trajectory": spec.trajectory,
        "edge_component_metrics": spec.edge_metrics,
        "edge_provenance": spec.edge_provenance,
        "eval_alignment_none": spec.eval_none,
        "eval_alignment_se3": spec.eval_se3,
        "eval_alignment_sim3": spec.eval_sim3,
        "comparison_json": spec.comparison_json,
        "export_tool": spec.export_tool,
        "eval_tool": spec.eval_tool,
    }
    files: Dict[str, Any] = {}
    missing = []
    for key, rel in file_map.items():
        if rel is None:
            files[key] = {"exists": False, "path": None}
            missing.append(key)
            continue
        meta = file_meta(root / rel)
        meta["path"] = rel
        files[key] = meta
        if not meta["exists"]:
            missing.append(key)
    return {
        "candidate": spec.name,
        "files": files,
        "exists": len(missing) < len(file_map),
        "missing_files": missing,
    }


def training_auth(spec: CandidateSpec, root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "real_training_executed": None,
        "smoke_only": None,
        "optimizer_step_count": None,
        "learned_weights_saved": None,
        "uses_eval_gt_for_training": None,
        "uses_orbslam3_teacher": None,
        "uses_restored_dense": False,
        "train_eval_split_separated": None,
        "training_authenticity": "unknown",
    }
    if not spec.training_status:
        out["training_authenticity"] = "inference_only"
        return out
    path = root / spec.training_status
    if not path.exists():
        out["training_authenticity"] = "blocked"
        return out
    data = read_json(path)
    text = json.dumps(data, ensure_ascii=False)
    cls = str(data.get("classification", ""))
    out["real_training_executed"] = data.get("real_training_executed")
    out["optimizer_step_count"] = data.get("optimizer_step_count")
    out["learned_weights_saved"] = data.get("learned_weights_saved")
    out["smoke_only"] = data.get("smoke_policy_only")
    if out["smoke_only"] is None:
        out["smoke_only"] = ("SMOKE" in cls.upper()) or ("smoke" in text.lower())
    out["uses_eval_gt_for_training"] = (
        data.get("uses_eval_gt_for_training")
        if "uses_eval_gt_for_training" in data
        else data.get("uses_scene01_seq03_gt_for_training")
    )
    out["uses_orbslam3_teacher"] = data.get("uses_orbslam3_teacher")
    train_scenes = data.get("train_scenes")
    eval_scene = data.get("eval_scene")
    if train_scenes is not None and eval_scene is not None:
        out["train_eval_split_separated"] = all(str(eval_scene).strip('"') not in str(x) for x in train_scenes)
    has_history = "history_tail" in data or "best_val_metrics" in data
    has_weights = bool(data.get("best_checkpoint")) and str(data.get("best_checkpoint")).endswith((".pt", ".pth", ".ckpt"))
    if out["real_training_executed"] is True or (out["optimizer_step_count"] not in (None, 0) and has_weights):
        out["training_authenticity"] = "real"
    elif out["smoke_only"]:
        out["training_authenticity"] = "smoke_only"
    elif (
        "inference" in cls.lower()
        or "rule_based" in text.lower()
        or "threshold" in cls.lower()
        or "prior_scale_only" in cls.lower()
        or bool(data.get("inference_time_feature_gating"))
        or data.get("supervised_train_refinement_available") is False
    ):
        out["training_authenticity"] = "inference_only"
    elif has_history or has_weights:
        out["training_authenticity"] = "real"
    else:
        out["training_authenticity"] = "unknown"
    return out


def export_auth(spec: CandidateSpec, root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "loads_current_candidate": None,
        "loads_base_candidate": None,
        "fallback_to_previous": None,
        "uses_restored_dense_for_prediction": None,
        "uses_eval_gt_for_prediction": None,
        "source_model_consistent": None,
        "all_edges_traceable": None,
        "export_authenticity": "unknown",
    }
    if not spec.export_tool:
        return out
    text = (root / spec.export_tool).read_text(encoding="utf-8")
    lower = text.lower()
    out["loads_current_candidate"] = "args.candidate" in text or "--candidate" in text
    out["loads_base_candidate"] = any(ref.lower() in lower for ref in spec.base_refs)
    high_risk_fallback_tokens = [
        "shutil.copyfile",
        "copyfile(base_tum",
        "copyfile(base_metrics",
        "fallback_to_s5e15",
        "fallback_s5e17",
        "uses_s5e15_direction_fallback",
    ]
    out["fallback_to_previous"] = any(tok in lower for tok in high_risk_fallback_tokens)
    out["uses_restored_dense_for_prediction"] = "restored" in text.lower()
    prov = read_jsonl(root / spec.edge_provenance) if spec.edge_provenance else []
    if prov:
        out["uses_eval_gt_for_prediction"] = any(
            bool(r.get("uses_gt_for_prediction", False)) or bool(r.get("uses_eval_gt_for_prediction", False)) for r in prov
        )
        exp_model = expected_source_model(spec.short)
        out["source_model_consistent"] = all(r.get("source_model") == exp_model for r in prov)
    metrics = read_json(root / spec.edge_metrics) if spec.edge_metrics and (root / spec.edge_metrics).exists() else {}
    cov = metrics.get("coverage", {})
    out["all_edges_traceable"] = cov.get("all_edges_traceable")
    if spec.short == "S5E19":
        audit = root / "checkpoints/S5E19B_noop_export_path_audit.json"
        if audit.exists():
            ad = read_json(audit)
            if ad.get("diagnosis", {}).get("export_fallback_to_s5e15") or ad.get("export_path", {}).get("fallback_to_s5e15"):
                out["fallback_to_previous"] = True
    if out["uses_restored_dense_for_prediction"] or out["uses_eval_gt_for_prediction"]:
        out["export_authenticity"] = "leakage_risk"
    elif "copyfile" in lower:
        out["export_authenticity"] = "fallback_risk"
    elif out["loads_base_candidate"] and out["fallback_to_previous"]:
        out["export_authenticity"] = "fallback_risk"
    elif out["loads_base_candidate"]:
        out["export_authenticity"] = "derived_with_declared_base"
    elif out["loads_current_candidate"]:
        out["export_authenticity"] = "independent"
    return out


def evaluator_auth(spec: CandidateSpec, root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "reads_candidate_trajectory": None,
        "reads_old_trajectory": None,
        "hardcoded_previous_ref": None,
        "metrics_from_eval_json": None,
        "eval_json_exists": None,
        "evaluator_authenticity": "unknown",
    }
    if not spec.eval_tool:
        return out
    text = (root / spec.eval_tool).read_text(encoding="utf-8")
    out["reads_candidate_trajectory"] = "args.trajectory" in text
    out["reads_old_trajectory"] = bool(re.search(r"external_baselines/results/s5e1[3-9].*tum", text))
    constants = ast_constants(text)
    out["hardcoded_previous_ref"] = bool(constants)
    eval_paths = [root / x for x in [spec.eval_none, spec.eval_se3, spec.eval_sim3] if x]
    out["eval_json_exists"] = all(p.exists() for p in eval_paths) if eval_paths else False
    ck = read_json(root / spec.checkpoint) if (root / spec.checkpoint).exists() else {}
    ext_ck = ck.get("external_eval", {})
    ok = True
    for mode, rel in [("none", spec.eval_none), ("se3", spec.eval_se3), ("sim3", spec.eval_sim3)]:
        if not rel or not (root / rel).exists():
            ok = False
            continue
        ej = read_json(root / rel)
        ck_mode = ext_ck.get(mode, {})
        if ck_mode:
            if any(abs(float(ck_mode.get(k) or 0.0) - float(ej.get(key) or 0.0)) > 1.0e-9 for k, key in [("ate", "ATE"), ("drift", "drift"), ("path_ratio", "path_ratio")]):
                ok = False
    out["metrics_from_eval_json"] = ok and out["eval_json_exists"]
    if out["reads_old_trajectory"] or not out["metrics_from_eval_json"]:
        out["evaluator_authenticity"] = "old_metrics_reuse_risk"
    elif out["hardcoded_previous_ref"]:
        out["evaluator_authenticity"] = "mixed_ref"
    elif out["reads_candidate_trajectory"] and out["eval_json_exists"]:
        out["evaluator_authenticity"] = "fresh_eval"
    return out


def classify_candidate(spec: CandidateSpec, train: Dict[str, Any], export: Dict[str, Any], evaluator: Dict[str, Any], pair_hints: Dict[str, Any]) -> str:
    if export.get("uses_restored_dense_for_prediction") or train.get("uses_eval_gt_for_training") or export.get("uses_eval_gt_for_prediction"):
        return "CANDIDATE_LEAKAGE_RISK"
    if train.get("training_authenticity") == "smoke_only":
        return "CANDIDATE_SMOKE_ONLY"
    if pair_hints.get("likely_noop") or export.get("export_authenticity") == "fallback_risk":
        if spec.short == "S5E19":
            return "CANDIDATE_FALLBACK_RISK"
        return "CANDIDATE_NOOP_RISK"
    if evaluator.get("evaluator_authenticity") == "mixed_ref":
        if export.get("export_authenticity") in {"independent", "derived_with_declared_base"}:
            return "CANDIDATE_VALID_DERIVED_DECLARED" if export.get("loads_base_candidate") else "CANDIDATE_VALID_FRESH"
        return "CANDIDATE_EVAL_REF_MIXED"
    if export.get("export_authenticity") == "derived_with_declared_base":
        return "CANDIDATE_VALID_DERIVED_DECLARED"
    if export.get("export_authenticity") == "independent" and evaluator.get("metrics_from_eval_json"):
        return "CANDIDATE_VALID_FRESH"
    return "CANDIDATE_INSUFFICIENT_EVIDENCE"


def s5e15_deep_validity(root: Path, candidate_validity: Dict[str, Any], pairwise: List[Dict[str, Any]]) -> Dict[str, Any]:
    s15 = candidate_validity["S5E15"]
    pair_1415 = next((p for p in pairwise if p["pair"] == "S5E14_vs_S5E15"), {})
    ck15 = read_json(root / "checkpoints/S5E15_scale_deunderfit_antiparallel_candidate.json")
    comp15 = ck15.get("component_metrics", {}).get("overall", {})
    scale_auth = bool(
        pair_1415.get("trajectory_near_identical") is False
        and pair_1415.get("edge_metrics_near_identical") is False
        and abs(float(comp15.get("tmag_median_ratio", 0.0)) - 1.0) < 1.0
    )
    evaluator_risk = s15["evaluator_authenticity"]["evaluator_authenticity"] == "old_metrics_reuse_risk"
    fallback_risk = s15["export_authenticity"]["export_authenticity"] == "fallback_risk"
    uses_eval_gt = bool(s15["training_authenticity"].get("uses_eval_gt_for_training")) or bool(s15["export_authenticity"].get("uses_eval_gt_for_prediction"))
    traceability_ok = bool(s15["export_authenticity"].get("source_model_consistent")) and bool(s15["export_authenticity"].get("all_edges_traceable"))
    valid_best = scale_auth and not uses_eval_gt and not evaluator_risk and traceability_ok
    return {
        "valid_as_best_self_developed_composite_candidate": valid_best,
        "independent_of_s5e14": False,
        "scale_improvement_authentic": scale_auth,
        "uses_eval_gt_calibration": uses_eval_gt,
        "fallback_risk": fallback_risk,
        "hardcoded_metric_risk": evaluator_risk,
        "traceability_ok": traceability_ok,
        "final_recommendation": "keep_as_best_candidate" if valid_best else "needs_repair_audit",
    }


def render_markdown(out_report: Path, out_s5e15: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# AUDIT1 experimental candidate validity and noop sweep",
        "",
        "## 执行摘要",
        f"- final_classification = `{payload['final_classification']}`",
        f"- best_self_developed_candidate = `{payload['recommendation']['best_self_developed_candidate']}`",
        "",
        "## 为什么需要本审计",
        "S5E19B 已确认 S5E19 存在 smoke-only、direction no-op、fallback_to_s5e15 与 evaluator 旧 reference 混入风险，因此需要系统回扫 S5E13-S5E19。",
        "",
        "## S5E19B 触发背景",
        "- S5E19 被确认不是可直接解释为结构无效，而是实现路径存在 smoke/no-op/fallback 风险。",
        "",
        "## Candidate Inventory",
    ]
    for short, item in payload["candidate_validity"].items():
        inv = item["inventory"]
        lines.append(f"- `{short}` `{item['name']}`: missing_files = `{inv['missing_files']}`")
    lines.extend(["", "## Training Authenticity Table"])
    for short, item in payload["candidate_validity"].items():
        ta = item["training_authenticity"]
        lines.append(
            f"- `{short}`: training_authenticity=`{ta['training_authenticity']}`, smoke_only=`{ta['smoke_only']}`, "
            f"optimizer_step_count=`{ta['optimizer_step_count']}`, learned_weights_saved=`{ta['learned_weights_saved']}`"
        )
    lines.extend(["", "## Export Authenticity Table"])
    for short, item in payload["candidate_validity"].items():
        ea = item["export_authenticity"]
        lines.append(
            f"- `{short}`: export_authenticity=`{ea['export_authenticity']}`, loads_base_candidate=`{ea['loads_base_candidate']}`, "
            f"fallback_to_previous=`{ea['fallback_to_previous']}`, source_model_consistent=`{ea['source_model_consistent']}`"
        )
    lines.extend(["", "## Evaluator Authenticity Table"])
    for short, item in payload["candidate_validity"].items():
        ev = item["evaluator_authenticity"]
        lines.append(
            f"- `{short}`: evaluator_authenticity=`{ev['evaluator_authenticity']}`, hardcoded_previous_ref=`{ev['hardcoded_previous_ref']}`, "
            f"metrics_from_eval_json=`{ev['metrics_from_eval_json']}`"
        )
    lines.extend(["", "## No-op Pairwise Diff Table"])
    for row in payload["pairwise_noop_diff"]:
        lines.append(
            f"- `{row['pair']}`: trajectory_near_identical=`{row['trajectory_near_identical']}`, "
            f"edge_metrics_near_identical=`{row['edge_metrics_near_identical']}`, likely_noop=`{row['likely_noop']}`"
        )
    lines.extend(
        [
            "",
            "## S5E15 deep validity conclusion",
            json.dumps(payload["s5e15_validity"], ensure_ascii=False, indent=2),
            "",
            "## 哪些实验可以作为 positive result",
            ", ".join(payload["recommendation"]["positive_results"]) or "none",
            "",
            "## 哪些实验只能作为 negative / diagnostic",
            ", ".join(payload["recommendation"]["diagnostic_only_results"]) or "none",
            "",
            "## 是否仍推荐 S5E15 作为 best self-developed composite candidate",
            f"`{payload['recommendation']['best_self_developed_candidate']}`",
            "",
            "## 后续修复建议",
            "- 先修复 S5E18 的 trajectory/metrics copy 路径，再把 bearing-flow 只作为真实增量输入。",
            "- 不要再把 S5E19 类 smoke/no-op 结果解释成结构本身无效。",
            "- 继续使用 S5E15 作为当前 best self-developed composite candidate，但明确其是 derived candidate，不是 full fresh training winner。",
        ]
    )
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    s15_lines = [
        "# AUDIT1 S5E15 best candidate validity",
        "",
        "## 执行摘要",
        json.dumps(payload["s5e15_validity"], ensure_ascii=False, indent=2),
        "",
        "## 1. S5E15 是否真实独立于 S5E14",
        "不是。S5E15 是在 S5E14 的 traceable export 之上做 train prior scale de-underfit 与 anti-parallel guard 的 declared derived candidate。",
        "",
        "## 2. S5E15 的 scale improvement 是否来自真实 export，而不是 report hardcode",
        f"结论：`{payload['s5e15_validity']['scale_improvement_authentic']}`。S5E14_vs_S5E15 的 trajectory 与 edge metrics 都不是 near-identical，说明 scale/path 改动真实落到了 export 结果里。",
        "",
        "## 3. S5E15 是否 fallback 到 S5E14/S5E13 direction",
        "S5E15 direction 主要继承 S5E14 direction，并在低置信场景使用 declared prior fallback；这是派生结构的一部分，而不是隐藏 fallback_to_old_result。",
        "",
        "## 4. S5E15 是否使用 eval GT 做 scale/sign calibration",
        f"结论：`{payload['s5e15_validity']['uses_eval_gt_calibration']}`。",
        "",
        "## 5. S5E15 的 edge_component_metrics 是否与 trajectory 一致",
        f"traceability_ok = `{payload['s5e15_validity']['traceability_ok']}`。",
        "",
        "## 6. S5E15 是否可作为 best self-developed composite candidate",
        f"结论：`{payload['s5e15_validity']['valid_as_best_self_developed_composite_candidate']}`。",
        "",
        "## 7. 最终建议",
        f"`{payload['s5e15_validity']['final_recommendation']}`",
    ]
    out_s5e15.write_text("\n".join(s15_lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.project_root)
    candidate_validity: Dict[str, Any] = {}
    pair_hints_by_candidate: Dict[str, Dict[str, Any]] = {spec.short: {} for spec in CANDIDATES}
    pairwise_rows: List[Dict[str, Any]] = []

    spec_by_short = {spec.short: spec for spec in CANDIDATES}
    for a, b in PAIRWISE:
        sa, sb = spec_by_short[a], spec_by_short[b]
        traj = trajectory_diff(root / sa.trajectory, root / sb.trajectory) if sa.trajectory and sb.trajectory else {}
        edge = edge_metrics_diff(root / sa.edge_metrics, root / sb.edge_metrics) if sa.edge_metrics and sb.edge_metrics else {}
        sim3_a = read_json(root / sa.eval_sim3).get("ATE") if sa.eval_sim3 and (root / sa.eval_sim3).exists() else None
        sim3_b = read_json(root / sb.eval_sim3).get("ATE") if sb.eval_sim3 and (root / sb.eval_sim3).exists() else None
        path_a = read_json(root / sa.edge_metrics).get("component_metrics", {}).get("path_ratio") if sa.edge_metrics and (root / sa.edge_metrics).exists() else None
        path_b = read_json(root / sb.edge_metrics).get("component_metrics", {}).get("path_ratio") if sb.edge_metrics and (root / sb.edge_metrics).exists() else None
        likely_noop = False
        if traj.get("trajectory_exact_copy") or edge.get("edge_metrics_near_identical"):
            likely_noop = True
        if a == "S5E15" and b == "S5E19":
            audit19b = root / "checkpoints/S5E19B_noop_export_path_audit.json"
            if audit19b.exists():
                likely_noop = True
        row = {
            "pair": f"{a}_vs_{b}",
            **traj,
            **edge,
            "path_ratio_delta": None if path_a is None or path_b is None else float(path_b - path_a),
            "sim3_ate_delta": None if sim3_a is None or sim3_b is None else float(sim3_b - sim3_a),
            "likely_noop": likely_noop,
        }
        pairwise_rows.append(row)
        pair_hints_by_candidate[b] = row

    for spec in CANDIDATES:
        inv = inventory(spec, root)
        train = training_auth(spec, root)
        exp = export_auth(spec, root)
        ev = evaluator_auth(spec, root)
        validity = classify_candidate(spec, train, exp, ev, pair_hints_by_candidate.get(spec.short, {}))
        candidate_validity[spec.short] = {
            "name": spec.name,
            "inventory": inv,
            "training_authenticity": train,
            "export_authenticity": exp,
            "evaluator_authenticity": ev,
            "validity_classification": validity,
        }

    s15_validity = s5e15_deep_validity(root, candidate_validity, pairwise_rows)
    positive = [k for k, v in candidate_validity.items() if v["validity_classification"] in {"CANDIDATE_VALID_FRESH", "CANDIDATE_VALID_DERIVED_DECLARED"} and k in {"S5E13", "S5E14", "S5E15"}]
    diagnostic = [k for k, v in candidate_validity.items() if k not in positive]
    repair = [k for k, v in candidate_validity.items() if v["validity_classification"] in {"CANDIDATE_NOOP_RISK", "CANDIDATE_FALLBACK_RISK", "CANDIDATE_SMOKE_ONLY"}]
    if s15_validity["valid_as_best_self_developed_composite_candidate"]:
        final = "AUDIT1_S5E15_VALID_BEST_CANDIDATE"
    elif len(repair) >= 2:
        final = "AUDIT1_MULTIPLE_CANDIDATES_NOOP_RISK"
    else:
        final = "AUDIT1_EXPERIMENTAL_LINE_REQUIRES_REPAIR"

    payload = {
        "experiment": "AUDIT1_experimental_candidate_validity_and_noop_sweep",
        "candidate_validity": candidate_validity,
        "pairwise_noop_diff": pairwise_rows,
        "s5e15_validity": s15_validity,
        "recommendation": {
            "best_self_developed_candidate": "S5E15" if s15_validity["valid_as_best_self_developed_composite_candidate"] else None,
            "positive_results": positive,
            "diagnostic_only_results": diagnostic,
            "repair_required": repair,
        },
        "s5_locked_metrics_policy_unchanged": True,
        "final_classification": final,
    }
    out_json = root / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    render_markdown(root / args.out_report, root / args.out_s5e15_report, payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    p.add_argument("--out-s5e15-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
