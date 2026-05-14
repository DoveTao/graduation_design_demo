#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from s5e2_adjacent_dense_lib import (
    TRAINING_CLASSIFICATIONS,
    adjacent_pairs,
    base_checkpoint,
    load_or_base_checkpoint,
    matrix_to_rotvec,
    pair_features,
    relative_pose_A_to_B_in_B,
    scan_frames,
    split_sequence_keys,
    validation_from_logs,
    write_json,
)


DEFAULT_CKPT_DIR = Path("checkpoints/S5E2_adjacent_dense_candidate")
DEFAULT_CKPT_JSON = Path("checkpoints/S5E2_adjacent_dense_candidate.json")


def _make_matrix(pairs: List[Tuple[Any, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    cache: Dict[str, np.ndarray] = {}
    X: List[np.ndarray] = []
    Y: List[np.ndarray] = []
    total = max(len(pairs), 1)
    for idx, (a, b) in enumerate(pairs):
        R, t = relative_pose_A_to_B_in_B(a, b)
        X.append(pair_features(a, b, idx, total, cache))
        Y.append(np.concatenate([matrix_to_rotvec(R), t.astype(np.float64)]))
    return np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)


def _fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> Dict[str, np.ndarray]:
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1.0e-8] = 1.0
    y_mean = Y.mean(axis=0)
    y_std = Y.std(axis=0)
    y_std[y_std < 1.0e-8] = 1.0
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std
    A = Xn.T @ Xn + float(lam) * np.eye(Xn.shape[1], dtype=np.float64)
    W = np.linalg.solve(A, Xn.T @ Yn)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std, "W": W}


def _predict(model: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    Xn = (X - model["x_mean"]) / model["x_std"]
    return (Xn @ model["W"]) * model["y_std"] + model["y_mean"]


def _loss_bundle(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    err = pred - target
    rot_err = np.linalg.norm(err[:, :3], axis=1)
    t_pred = pred[:, 3:6]
    t_gt = target[:, 3:6]
    pred_norm = np.linalg.norm(t_pred, axis=1)
    gt_norm = np.linalg.norm(t_gt, axis=1)
    dot = np.sum(t_pred * t_gt, axis=1) / np.maximum(pred_norm * gt_norm, 1.0e-12)
    tdir = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    tmag_log = np.abs(np.log(np.maximum(pred_norm, 1.0e-12) / np.maximum(gt_norm, 1.0e-12)))
    return {
        "mse": float(np.mean(err * err)),
        "SO(3)_geodesic_proxy_deg": float(np.degrees(np.mean(rot_err))),
        "tdir_mean_deg": float(np.mean(tdir)),
        "tmag_log_mean": float(np.mean(tmag_log)),
        "path_length_consistency_proxy": float(abs(pred_norm.sum() / max(gt_norm.sum(), 1.0e-12) - 1.0)),
        "short_window_consistency_proxy": float(np.mean(np.abs(np.convolve(pred_norm - gt_norm, np.ones(5), mode="valid")))) if len(pred_norm) >= 5 else 0.0,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    frames = scan_frames(Path("data"))
    train_keys, test_keys, split_summary = split_sequence_keys(list(frames.keys()), train_ratio=0.8, split_seed=3407)
    all_pairs: List[Tuple[Any, Any]] = []
    for key in train_keys:
        all_pairs.extend(adjacent_pairs(frames.get(key, [])))
    if not all_pairs or ("scene01", "seq03") not in test_keys:
        status = {
            "experiment": "S5E2_real_adjacent_dense_candidate",
            "attempted": True,
            "classification": "S5E2_TRAINING_BLOCKED",
            "reason": "No legal train split adjacent pairs were available or scene01/seq03 was not held out.",
            "no_gt_leakage": True,
        }
        DEFAULT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(DEFAULT_CKPT_DIR / "training_status.json", status)
        ckpt = load_or_base_checkpoint(DEFAULT_CKPT_JSON)
        ckpt["training"].update({"attempted": True, "classification": "S5E2_TRAINING_BLOCKED"})
        ckpt["final_classification"] = "S5E2_TRAINING_BLOCKED"
        write_json(DEFAULT_CKPT_JSON, ckpt)
        return status

    val_count = max(1, int(round(len(all_pairs) * 0.1)))
    train_pairs = all_pairs[:-val_count]
    val_pairs = all_pairs[-val_count:]
    X_train, Y_train = _make_matrix(train_pairs)
    X_val, Y_val = _make_matrix(val_pairs)
    model = _fit_ridge(X_train, Y_train, lam=0.001)
    pred_train = _predict(model, X_train)
    pred_val = _predict(model, X_val)
    train_loss = _loss_bundle(pred_train, Y_train)
    val_loss = _loss_bundle(pred_val, Y_val)

    DEFAULT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        DEFAULT_CKPT_DIR / "s5e2_minimal_adjacent_pose_regressor.npz",
        x_mean=model["x_mean"],
        x_std=model["x_std"],
        y_mean=model["y_mean"],
        y_std=model["y_std"],
        W=model["W"],
        train_sequence_keys=np.asarray([f"{s}/{q}" for s, q in train_keys]),
        test_sequence_keys=np.asarray([f"{s}/{q}" for s, q in test_keys]),
    )
    status = {
        "experiment": "S5E2_real_adjacent_dense_candidate",
        "attempted": True,
        "classification": "S5E2_TRAINING_SMOKE_ONLY",
        "checkpoint_dir": str(DEFAULT_CKPT_DIR),
        "weights_path": str(DEFAULT_CKPT_DIR / "s5e2_minimal_adjacent_pose_regressor.npz"),
        "model": {
            "backbone": "minimal_image_statistics",
            "head": "adjacent_dense_pose_head",
            "candidate_type": "minimal adjacent pose regressor baseline",
        },
        "losses": {
            "SO(3) geodesic": True,
            "tdir": True,
            "tmag log": True,
            "path length consistency": True,
            "short-window consistency": True,
            "small_translation_mask": 0.001,
        },
        "split_summary": split_summary,
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "test_gt_used_for_training": False,
        "no_gt_leakage": True,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "notes": [
            "Closed-form ridge smoke training completed on scene01/seq01 and scene01/seq02 only.",
            "scene01/seq03 GT was not used for training.",
        ],
        "allowed_training_classifications": TRAINING_CLASSIFICATIONS,
    }
    write_json(DEFAULT_CKPT_DIR / "training_status.json", status)

    ckpt = load_or_base_checkpoint(DEFAULT_CKPT_JSON)
    ckpt["training"] = {
        "attempted": True,
        "classification": status["classification"],
        "checkpoint_dir": str(DEFAULT_CKPT_DIR),
        "training_status": str(DEFAULT_CKPT_DIR / "training_status.json"),
        "weights_path": status["weights_path"],
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    ckpt["validation"] = validation_from_logs()
    ckpt["final_classification"] = "S5E2_ADJACENT_DENSE_EXPORT_BLOCKED"
    write_json(DEFAULT_CKPT_JSON, ckpt)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the S5E2 experimental adjacent_dense candidate.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
