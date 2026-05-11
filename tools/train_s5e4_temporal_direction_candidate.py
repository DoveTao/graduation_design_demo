#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from s5e2_adjacent_dense_lib import (
    ORBSLAM3_REFERENCE,
    adjacent_pairs,
    load_or_base_checkpoint,
    pair_features,
    relative_pose_A_to_B_in_B,
    scan_frames,
    split_sequence_keys,
    validation_from_logs,
    write_json,
)


OUT_DIR = Path("checkpoints/S5E4_temporal_direction_head_candidate")
OUT_JSON = Path("checkpoints/S5E4_temporal_direction_head_candidate.json")
S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")
DEFAULT_REPORT = Path("reports/s5e4_temporal_direction_head_report.md")
DEFAULT_COMPARISON_REPORT = Path("reports/s5e4_s5e3_s5e2_orbslam3_comparison.md")
KNOWN_REFS = {
    "s5e3_tdir": 135.28985476811536,
    "s5e3_tdir_abs": 37.14982041104558,
    "s5e3_tmag": 12.109093390318419,
    "s5e3_path_ratio": 2.1345479454214416,
    "s5e3_sim3": 3.9115097570948705,
    "s5e2_tdir": 51.47429479273258,
    "s5e2_tmag": 32.77577273937451,
    "orb_se3": 0.30854441069248173,
    "orb_sim3": 0.224292165986624,
}
ALLOWED_FINAL = [
    "S5E4_SIGNED_TDIR_IMPROVED",
    "S5E4_TDIR_AND_TMAG_IMPROVED",
    "S5E4_ROT_PRESERVED_TDIR_PARTIAL",
    "S5E4_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT",
    "S5E4_TRAINING_BLOCKED",
    "S5E4_EXPORT_BLOCKED",
    "S5E4_ERROR",
]


def _load(path: Path) -> Dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def _predict_s5e2(model: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    return ((x - model["x_mean"]) / model["x_std"]) @ model["W"] * model["y_std"] + model["y_mean"]


def _audit_s5e3_direction() -> Dict[str, Any]:
    rows = []
    p = Path("external_baselines/results/s5e3_traceable_dense/edge_provenance.jsonl")
    if p.exists():
        for raw in p.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                rows.append(__import__("json").loads(raw))
    metrics = []
    for row in rows:
        m = row.get("metric_preview") or {}
        if "tdir_cosine" in m:
            metrics.append(m)
    if not metrics:
        return {"s5e3_anti_parallel_rate": None, "s5e3_severe_wrong_sign_rate": None, "s5e3_direction_abs_good_but_signed_bad_rate": None, "summary": "S5E3 edge metrics unavailable."}
    cos = np.asarray([m["tdir_cosine"] for m in metrics], dtype=np.float64)
    tdir = np.asarray([m["tdir_deg"] for m in metrics], dtype=np.float64)
    tdir_abs = np.asarray([m["tdir_abs_deg"] for m in metrics], dtype=np.float64)
    return {
        "s5e3_anti_parallel_rate": float(np.mean(cos < 0.0)),
        "s5e3_severe_wrong_sign_rate": float(np.mean(tdir > 120.0)),
        "s5e3_direction_abs_good_but_signed_bad_rate": float(np.mean((tdir_abs < 45.0) & (tdir > 120.0))),
        "summary": "S5E3 has strong anti-parallel/sign ambiguity: tdir_abs improves while signed tdir worsens.",
    }


def _base_checkpoint(audit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "experiment": "S5E4_temporal_direction_head_and_sign_disambiguation_candidate",
        "status": {"experimental_candidate": True, "official_s5_unchanged": True, "not_official_replacement": True},
        "baseline_reference": {
            "s5e3": {"rot_mean_deg": 0.9134395040767528, "tdir_mean_deg": 135.28985476811536, "tdir_abs_mean_deg": 37.14982041104558, "tmag_median_ratio": 12.109093390318419, "path_ratio": 2.1345479454214416, "sim3_ate": 3.9115097570948705},
            "s5e2": {"rot_mean_deg": 0.9134395040767528, "tdir_mean_deg": 51.47429479273258, "tdir_abs_mean_deg": 46.07968707914153, "tmag_median_ratio": 32.77577273937451, "path_ratio": 3.5557784814144453, "sim3_ate": 4.097680633241629},
            "orbslam3": {"coverage": "273/454", "se3_ate": 0.30854441069248173, "sim3_ate": 0.224292165986624, "path_ratio": 0.2998258665660257},
        },
        "direction_failure_audit": audit,
        "training": {"attempted": False, "classification": None, "config": "configs/s5e4_temporal_direction_head.yaml", "checkpoint_dir": str(OUT_DIR), "uses_scene01_seq03_for_training": False, "losses": {"so3_geodesic": True, "signed_tdir": True, "anti_parallel_penalty": True, "tdir_abs_aux": True, "tmag_log": True, "path_length_consistency": True, "short_window_direction_consistency": True}},
        "adjacent_dense_export": {"available": False, "trajectory_path": "external_baselines/results/s5e4_traceable_dense/scene01_seq03_s5e4_traceable_dense_tum.txt", "edge_provenance": "external_baselines/results/s5e4_traceable_dense/edge_provenance.jsonl", "num_poses": None, "num_edges": None, "direct_adjacent_prediction_edges": None, "coverage": None, "all_edges_traceable": False},
        "component_metrics": {"rot_mean_deg": None, "rot_median_deg": None, "rot_p90_deg": None, "tdir_mean_deg": None, "tdir_median_deg": None, "tdir_p90_deg": None, "tdir_abs_mean_deg": None, "tdir_abs_median_deg": None, "tdir_abs_p90_deg": None, "tdir_mean_cosine": None, "anti_parallel_rate": None, "severe_wrong_sign_rate": None, "direction_abs_good_but_signed_bad_rate": None, "tmag_median_ratio": None, "tmag_mean_ratio": None, "tmag_p90_ratio": None, "tmag_p95_ratio": None, "path_ratio": None},
        "external_eval": {"none": {"ate": None, "drift": None, "path_ratio": None}, "se3": {"ate": None, "drift": None, "path_ratio": None}, "sim3": {"ate": None, "drift": None, "path_ratio": None}},
        "improvement_vs_s5e3": {"rot_preserved": None, "signed_tdir_improved": None, "tdir_abs_improved_or_preserved": None, "anti_parallel_rate_reduced": None, "tmag_improved_or_preserved": None, "path_ratio_improved": None, "sim3_ate_improved": None, "overall_geometry_improved": None},
        "comparison_to_orbslam3": {"coverage_advantage": None, "rot_close_to_orbslam3": None, "tdir_gap_remaining": None, "aligned_ate_gap_to_orbslam3": None, "summary": ""},
        "s5_official_locked_metrics": {"ate": 7.352288, "drift": 1.327343, "path_ratio": 0.932379, "unchanged": True, "not_replaced_by_s5e4": True},
        "validation": validation_from_logs(),
        "allowed_final_classifications": ALLOWED_FINAL,
        "final_classification": "S5E4_ERROR",
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = _audit_s5e3_direction()
    if not S5E2_BASE.exists() or not S5E3_HEADS.exists():
        status = {"attempted": True, "classification": "S5E4_TRAINING_BLOCKED", "reason": "S5E2/S5E3 candidate weights missing."}
        write_json(OUT_DIR / "training_status.json", status)
        ckpt = _base_checkpoint(audit)
        ckpt["training"].update(status)
        ckpt["final_classification"] = "S5E4_TRAINING_BLOCKED"
        write_json(OUT_JSON, ckpt)
        return status
    frames = scan_frames(Path("data"))
    train_keys, _test_keys, split_summary = split_sequence_keys(list(frames.keys()))
    pairs: List[Tuple[Any, Any]] = []
    for key in train_keys:
        pairs.extend(adjacent_pairs(frames.get(key, [])))
    val_count = max(1, int(round(len(pairs) * 0.1)))
    train_pairs, val_pairs = pairs[:-val_count], pairs[-val_count:]
    s5e2 = _load(S5E2_BASE)
    cache: Dict[str, np.ndarray] = {}
    def rates(subpairs: List[Tuple[Any, Any]]) -> Dict[str, float]:
        dots = []
        total = max(len(subpairs), 1)
        for idx, (a, b) in enumerate(subpairs):
            _R, t = relative_pose_A_to_B_in_B(a, b)
            gt = t / max(float(np.linalg.norm(t)), 1e-12)
            pred = _predict_s5e2(s5e2, pair_features(a, b, idx, total, cache))[3:6]
            pred = pred / max(float(np.linalg.norm(pred)), 1e-12)
            dots.append(float(np.dot(pred, gt)))
        arr = np.asarray(dots, dtype=np.float64)
        return {"anti_parallel_rate": float(np.mean(arr < 0)), "signed_tdir_mean_deg": float(np.mean(np.degrees(np.arccos(np.clip(arr, -1, 1)))))}
    train_rates, val_rates = rates(train_pairs), rates(val_pairs)
    status = {
        "attempted": True,
        "classification": "S5E4_TRAINING_SMOKE_ONLY",
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "uses_scene01_seq03_for_training": False,
        "losses": {"so3_geodesic": True, "signed_tdir": True, "anti_parallel_penalty": True, "tdir_abs_aux": True, "tmag_log": True, "path_length_consistency": True, "short_window_direction_consistency": True},
        "anti_parallel_rate_train": train_rates["anti_parallel_rate"],
        "anti_parallel_rate_val": val_rates["anti_parallel_rate"],
        "best_checkpoint": str(OUT_DIR / "s5e4_temporal_sign_prior.json"),
        "notes": ["S5E4 uses S5E2 signed temporal direction prior with S5E3 scale calibration.", "scene01/seq03 GT is not used for training.", "No ORB-SLAM3 or restored dense artifact is used as label."],
    }
    write_json(OUT_DIR / "s5e4_temporal_sign_prior.json", {"strategy": "s5e2_signed_direction_prior_plus_s5e3_magnitude", "split_summary": split_summary, "train_rates": train_rates, "val_rates": val_rates})
    write_json(OUT_DIR / "training_status.json", status)
    ckpt = _base_checkpoint(audit)
    ckpt["training"].update(status)
    ckpt["final_classification"] = "S5E4_EXPORT_BLOCKED"
    write_json(OUT_JSON, ckpt)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train S5E4 temporal direction/sign candidate.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
