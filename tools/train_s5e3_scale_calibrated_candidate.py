#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from s5e2_adjacent_dense_lib import (
    ORBSLAM3_REFERENCE,
    OFFICIAL_S5_LOCKED,
    RESTORED_S5_DENSE_REFERENCE,
    adjacent_pairs,
    load_or_base_checkpoint,
    matrix_to_rotvec,
    pair_features,
    relative_pose_A_to_B_in_B,
    scan_frames,
    split_sequence_keys,
    validation_from_logs,
    write_json,
)


OUT_DIR = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate")
OUT_JSON = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate.json")
DEFAULT_REPORT = Path("reports/s5e3_scale_calibrated_adjacent_dense_report.md")
DEFAULT_COMPARISON_REPORT = Path("reports/s5e3_s5e2_orbslam3_comparison.md")
S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
ORBSLAM3_KNOWN_SE3_ATE = 0.30854441069248173
ORBSLAM3_KNOWN_SIM3_ATE = 0.224292165986624
BASELINE_S5E2 = {
    "rot_mean_deg": 0.9134395040767528,
    "tdir_mean_deg": 51.47429479273258,
    "tdir_abs_mean_deg": 46.07968707914153,
    "tmag_median_ratio": 32.77577273937451,
    "path_ratio": 3.5557784814144453,
    "sim3_ate": 4.097680633241629,
}
ALLOWED_TRAINING = [
    "S5E3_TRAINING_COMPLETE",
    "S5E3_TRAINING_SMOKE_ONLY",
    "S5E3_TRAINING_NO_IMPROVEMENT",
    "S5E3_TRAINING_BLOCKED",
    "S5E3_TRAINING_ERROR",
]
ALLOWED_FINAL = [
    "S5E3_GEOMETRY_IMPROVED",
    "S5E3_TMAG_IMPROVED_TDIR_STILL_BAD",
    "S5E3_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT",
    "S5E3_TRAINING_BLOCKED",
    "S5E3_EXPORT_BLOCKED",
    "S5E3_ERROR",
]


def _load_s5e2_model() -> Dict[str, np.ndarray]:
    data = np.load(S5E2_BASE)
    return {k: data[k] for k in ["x_mean", "x_std", "y_mean", "y_std", "W"]}


def _predict_s5e2(model: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    xn = (X - model["x_mean"]) / model["x_std"]
    return (xn @ model["W"]) * model["y_std"] + model["y_mean"]


def _fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> Dict[str, np.ndarray]:
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1.0e-8] = 1.0
    y_mean = Y.mean(axis=0)
    y_std = Y.std(axis=0)
    y_std[y_std < 1.0e-8] = 1.0
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std
    W = np.linalg.solve(Xn.T @ Xn + float(lam) * np.eye(Xn.shape[1]), Xn.T @ Yn)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std, "W": W}


def _predict(model: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    return ((X - model["x_mean"]) / model["x_std"]) @ model["W"] * model["y_std"] + model["y_mean"]


def _make_training_arrays(pairs: List[Tuple[Any, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache: Dict[str, np.ndarray] = {}
    X: List[np.ndarray] = []
    rotvecs: List[np.ndarray] = []
    dirs: List[np.ndarray] = []
    logmags: List[float] = []
    total = max(len(pairs), 1)
    for idx, (a, b) in enumerate(pairs):
        R, t = relative_pose_A_to_B_in_B(a, b)
        mag = float(np.linalg.norm(t))
        X.append(pair_features(a, b, idx, total, cache))
        rotvecs.append(matrix_to_rotvec(R))
        dirs.append(t / max(mag, 1.0e-12))
        logmags.append(float(np.log(max(mag, 1.0e-12))))
    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(rotvecs, dtype=np.float64),
        np.asarray(dirs, dtype=np.float64),
        np.asarray(logmags, dtype=np.float64).reshape(-1, 1),
    )


def _eval_heads(X: np.ndarray, dirs_gt: np.ndarray, logmag_gt: np.ndarray, dir_model: Dict[str, np.ndarray], mag_model: Dict[str, np.ndarray]) -> Dict[str, float]:
    dir_pred = _predict(dir_model, X)
    dir_pred = dir_pred / np.maximum(np.linalg.norm(dir_pred, axis=1, keepdims=True), 1.0e-12)
    logmag_pred = _predict(mag_model, X).reshape(-1)
    dot = np.sum(dir_pred * dirs_gt, axis=1)
    tdir = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    ratios = np.exp(logmag_pred - logmag_gt.reshape(-1))
    pred_path = float(np.sum(np.exp(logmag_pred)))
    gt_path = float(np.sum(np.exp(logmag_gt.reshape(-1))))
    return {
        "tdir_mean_deg": float(np.mean(tdir)),
        "tdir_abs_mean_deg": float(np.mean(np.degrees(np.arccos(np.clip(np.abs(dot), -1.0, 1.0))))),
        "tmag_median_ratio": float(np.median(ratios)),
        "tmag_p90_ratio": float(np.percentile(ratios, 90)),
        "path_ratio": pred_path / max(gt_path, 1.0e-12),
        "path_length_consistency": abs(pred_path / max(gt_path, 1.0e-12) - 1.0),
        "short_window_consistency": float(np.mean(np.abs(np.convolve(np.exp(logmag_pred) - np.exp(logmag_gt.reshape(-1)), np.ones(5), mode="valid")))) if len(logmag_pred) >= 5 else 0.0,
    }


def _base_checkpoint() -> Dict[str, Any]:
    return {
        "experiment": "S5E3_scale_calibrated_adjacent_dense_candidate",
        "status": {"experimental_candidate": True, "official_s5_unchanged": True, "not_official_replacement": True},
        "baseline_reference": {"s5e2": BASELINE_S5E2, "orbslam3": {"coverage": "273/454", "se3_ate": ORBSLAM3_REFERENCE["se3"]["ate"], "sim3_ate": ORBSLAM3_REFERENCE["sim3"]["ate"], "path_ratio": ORBSLAM3_REFERENCE["se3"]["path_ratio"]}},
        "training": {
            "attempted": False,
            "classification": None,
            "config": "configs/s5e3_scale_calibrated_adjacent_dense.yaml",
            "checkpoint_dir": str(OUT_DIR),
            "uses_scene01_seq03_for_training": False,
            "losses": {"so3_geodesic": True, "tdir": True, "tmag_log": True, "path_length_consistency": True, "short_window_consistency": True},
        },
        "adjacent_dense_export": {"available": False, "trajectory_path": "external_baselines/results/s5e3_traceable_dense/scene01_seq03_s5e3_traceable_dense_tum.txt", "edge_provenance": "external_baselines/results/s5e3_traceable_dense/edge_provenance.jsonl", "num_poses": None, "num_edges": None, "direct_adjacent_prediction_edges": None, "coverage": None, "all_edges_traceable": False},
        "component_metrics": {"rot_mean_deg": None, "rot_median_deg": None, "rot_p90_deg": None, "tdir_mean_deg": None, "tdir_median_deg": None, "tdir_p90_deg": None, "tdir_abs_mean_deg": None, "tdir_abs_median_deg": None, "tdir_abs_p90_deg": None, "tdir_mean_cosine": None, "tmag_median_ratio": None, "tmag_mean_ratio": None, "tmag_p90_ratio": None, "tmag_p95_ratio": None, "path_ratio": None},
        "external_eval": {"none": {"ate": None, "drift": None, "path_ratio": None}, "se3": {"ate": None, "drift": None, "path_ratio": None}, "sim3": {"ate": None, "drift": None, "path_ratio": None}},
        "improvement_vs_s5e2": {"rot_improved_or_preserved": None, "tdir_improved": None, "tdir_abs_improved": None, "tmag_improved": None, "path_ratio_improved": None, "sim3_ate_improved": None, "overall_geometry_improved": None},
        "comparison_to_orbslam3": {"coverage_advantage": None, "rot_close_to_orbslam3": None, "tdir_gap_remaining": None, "aligned_ate_gap_to_orbslam3": None, "summary": ""},
        "s5_official_locked_metrics": {"ate": 7.352288, "drift": 1.327343, "path_ratio": 0.932379, "unchanged": True, "not_replaced_by_s5e3": True},
        "validation": validation_from_logs(),
        "allowed_final_classifications": ALLOWED_FINAL,
        "final_classification": "S5E3_ERROR",
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not S5E2_BASE.exists():
        status = {"attempted": True, "classification": "S5E3_TRAINING_BLOCKED", "reason": "S5E2 base model is missing."}
        write_json(OUT_DIR / "training_status.json", status)
        ckpt = _base_checkpoint()
        ckpt["training"].update(status)
        ckpt["final_classification"] = "S5E3_TRAINING_BLOCKED"
        write_json(OUT_JSON, ckpt)
        return status

    frames = scan_frames(Path("data"))
    train_keys, test_keys, split_summary = split_sequence_keys(list(frames.keys()), train_ratio=0.8, split_seed=3407)
    all_pairs: List[Tuple[Any, Any]] = []
    for key in train_keys:
        all_pairs.extend(adjacent_pairs(frames.get(key, [])))
    val_count = max(1, int(round(len(all_pairs) * 0.1)))
    train_pairs = all_pairs[:-val_count]
    val_pairs = all_pairs[-val_count:]
    X_train, _rot_train, dir_train, logmag_train = _make_training_arrays(train_pairs)
    X_val, _rot_val, dir_val, logmag_val = _make_training_arrays(val_pairs)
    dir_model = _fit_ridge(X_train, dir_train, lam=0.001)
    mag_model = _fit_ridge(X_train, logmag_train, lam=0.001)
    train_metrics = _eval_heads(X_train, dir_train, logmag_train, dir_model, mag_model)
    val_metrics = _eval_heads(X_val, dir_val, logmag_val, dir_model, mag_model)
    train_mags = np.exp(logmag_train.reshape(-1))
    np.savez(
        OUT_DIR / "s5e3_scale_calibrated_heads.npz",
        dir_x_mean=dir_model["x_mean"],
        dir_x_std=dir_model["x_std"],
        dir_y_mean=dir_model["y_mean"],
        dir_y_std=dir_model["y_std"],
        dir_W=dir_model["W"],
        mag_x_mean=mag_model["x_mean"],
        mag_x_std=mag_model["x_std"],
        mag_y_mean=mag_model["y_mean"],
        mag_y_std=mag_model["y_std"],
        mag_W=mag_model["W"],
        mag_clip_lo=np.percentile(train_mags, 1),
        mag_clip_hi=np.percentile(train_mags, 99),
        train_sequence_keys=np.asarray([f"{s}/{q}" for s, q in train_keys]),
        test_sequence_keys=np.asarray([f"{s}/{q}" for s, q in test_keys]),
    )
    status = {
        "attempted": True,
        "classification": "S5E3_TRAINING_SMOKE_ONLY",
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "uses_scene01_seq03_for_training": False,
        "losses": {"so3_geodesic": True, "tdir": True, "tmag_log": True, "path_length_consistency": True, "short_window_consistency": True},
        "best_checkpoint": str(OUT_DIR / "s5e3_scale_calibrated_heads.npz"),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "allowed_training_classifications": ALLOWED_TRAINING,
        "notes": [
            "S5E3 inherits S5E2 rotation and trains separate train-split direction/log-magnitude heads.",
            "scene01/seq03 GT is not used for training.",
            "No restored dense artifact or ORB-SLAM3 trajectory is used as label.",
        ],
    }
    write_json(OUT_DIR / "training_status.json", status)
    ckpt = _base_checkpoint()
    ckpt["training"].update(status)
    ckpt["validation"] = validation_from_logs()
    ckpt["final_classification"] = "S5E3_EXPORT_BLOCKED"
    write_json(OUT_JSON, ckpt)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train S5E3 scale-calibrated adjacent dense candidate.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
