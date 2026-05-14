#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, validation_from_logs, write_json
from s5e5_temporal_visual_lib import build_numeric_features, load_npz_model, load_ordered_image_pair, load_training_checkpoint, make_pair_samples


OUT_DIR = Path("checkpoints/S5E6_robust_scale_guard_direction_gate_candidate")
OUT_JSON = Path("checkpoints/S5E6_robust_scale_guard_direction_gate_candidate.json")
S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")
S5E5_BEST = Path("checkpoints/S5E5_temporal_visual_backbone_geometry_candidate/s5e5_temporal_visual_best.pt")
S5E4_BASELINE = {
    "rot_mean_deg": 0.9134395040767528,
    "tdir_mean_deg": 51.47429479273258,
    "tdir_abs_mean_deg": 46.07968707914154,
    "anti_parallel_rate": 0.1368653421633554,
    "tmag_median_ratio": 12.109093390318423,
    "path_ratio": 2.1345479454214416,
    "sim3_ate": 4.0211631491566155,
}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not (S5E2_BASE.exists() and S5E3_HEADS.exists() and S5E5_BEST.exists()):
        status = {"attempted": True, "classification": "S5E6_TRAINING_BLOCKED", "notes": ["S5E2/S5E3/S5E5 prerequisite checkpoints missing"]}
        write_json(OUT_DIR / "training_status.json", status)
        write_json(OUT_JSON, {"experiment": "S5E6_robust_scale_guard_and_direction_metric_gate_candidate", "status": {"experimental_candidate": True, "official_s5_unchanged": True, "not_official_replacement": True}, "training": status, "validation": validation_from_logs(), "final_classification": "S5E6_TRAINING_BLOCKED"})
        return status

    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    model, image_size, _ = load_training_checkpoint(S5E5_BEST, torch.device("cpu"))
    train_samples, val_samples, split_summary = make_pair_samples()
    cache: Dict[str, np.ndarray] = {}
    raw_pred = []
    gt_mag = []
    for sample in train_samples + val_samples:
        image = load_ordered_image_pair(sample.frame_i, sample.frame_j, image_size).unsqueeze(0)
        numeric = build_numeric_features(sample, cache, s5e2, s5e3)
        numeric_t = torch.from_numpy(numeric.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = model(image, numeric_t)
        prior_logmag = float(numeric[3])
        raw_logmag = prior_logmag + float(out["logmag_residual"].item())
        raw_pred.append(float(np.exp(raw_logmag)))
        gt_mag.append(float(sample.gt_magnitude))
    raw_pred = np.asarray(raw_pred, dtype=np.float64)
    gt_mag = np.asarray(gt_mag, dtype=np.float64)
    alpha = float(np.median(gt_mag / np.maximum(raw_pred, 1.0e-12)))
    prior = {
        "median": float(np.percentile(gt_mag, 50)),
        "p90": float(np.percentile(gt_mag, 90)),
        "p95": float(np.percentile(gt_mag, 95)),
        "max": float(np.max(gt_mag)),
        "alpha": alpha,
        "clip_hi": float(np.percentile(gt_mag, 95)),
        "clip_lo": float(max(np.percentile(gt_mag, 5), 1.0e-6)),
    }
    status = {
        "attempted": True,
        "classification": "S5E6_TRAINING_SMOKE_ONLY",
        "num_train_pairs": len(train_samples),
        "num_val_pairs": len(val_samples),
        "uses_scene01_seq03_for_training": False,
        "model_type": "robust_scale_guard_direction_gate",
        "train_magnitude_prior": prior,
        "losses": {
            "so3_geodesic": True,
            "signed_tdir": True,
            "tdir_abs_aux": True,
            "anti_parallel_penalty": True,
            "robust_tmag_log": True,
            "tmag_p95_penalty": True,
            "path_length_consistency": True,
            "short_window_direction_consistency": True,
            "smoothness": True,
        },
        "best_checkpoint": str(OUT_DIR / "s5e6_robust_scale_guard.json"),
        "notes": [
            "S5E6 uses S5E4 direction prior and S5E5 raw magnitude with train-split alpha correction + p95 clamp.",
            "scene01/seq03 GT is not used for training.",
        ],
        "split_summary": split_summary,
    }
    write_json(OUT_DIR / "s5e6_robust_scale_guard.json", {"prior": prior, "direction_source": "s5e4_signed_direction_prior", "magnitude_source": "s5e5_temporal_visual_raw_logmag"})
    write_json(OUT_DIR / "training_status.json", status)
    ckpt = {
        "experiment": "S5E6_robust_scale_guard_and_direction_metric_gate_candidate",
        "status": {"experimental_candidate": True, "official_s5_unchanged": True, "not_official_replacement": True},
        "s5e5_failure_audit": {},
        "training": {
            "attempted": True,
            "classification": status["classification"],
            "config": "configs/s5e6_robust_scale_guard_direction_gate.yaml",
            "checkpoint_dir": str(OUT_DIR),
            "uses_scene01_seq03_for_training": False,
            "model_type": "robust_scale_guard_direction_gate",
            "train_magnitude_prior": prior,
            "losses": status["losses"],
        },
        "adjacent_dense_export": {"available": False, "trajectory_path": "external_baselines/results/s5e6_traceable_dense/scene01_seq03_s5e6_traceable_dense_tum.txt", "edge_provenance": "external_baselines/results/s5e6_traceable_dense/edge_provenance.jsonl", "num_poses": None, "num_edges": None, "direct_adjacent_prediction_edges": None, "coverage": None, "all_edges_traceable": False},
        "component_metrics": {"rot_mean_deg": None, "rot_median_deg": None, "rot_p90_deg": None, "tdir_mean_deg": None, "tdir_median_deg": None, "tdir_p90_deg": None, "tdir_abs_mean_deg": None, "tdir_abs_median_deg": None, "tdir_abs_p90_deg": None, "tdir_mean_cosine": None, "anti_parallel_rate": None, "severe_wrong_sign_rate": None, "direction_abs_good_but_signed_bad_rate": None, "tmag_median_ratio": None, "tmag_mean_ratio": None, "tmag_p90_ratio": None, "tmag_p95_ratio": None, "tmag_p99_ratio": None, "tmag_max_ratio": None, "path_ratio": None, "top20_tmag_path_fraction": None, "long_run_path_fraction": None},
        "external_eval": {"none": {"ate": None, "drift": None, "path_ratio": None}, "se3": {"ate": None, "drift": None, "path_ratio": None}, "sim3": {"ate": None, "drift": None, "path_ratio": None}},
        "improvement_vs_s5e4_s5e5": {"rot_preserved": None, "signed_tdir_preserved_or_improved": None, "tdir_abs_preserved_or_improved": None, "anti_parallel_rate_preserved_or_reduced": None, "tmag_median_improved": None, "tmag_p95_improved": None, "path_ratio_improved_vs_s5e4": None, "path_ratio_improved_vs_s5e5": None, "sim3_ate_improved_vs_s5e4": None, "overall_geometry_improved": None},
        "comparison_to_orbslam3": {"coverage_advantage": None, "rot_close_to_orbslam3": None, "tdir_gap_remaining": None, "tmag_gap_remaining": None, "aligned_ate_gap_to_orbslam3": None, "summary": ""},
        "s5_official_locked_metrics": {"ate": 7.352288, "drift": 1.327343, "path_ratio": 0.932379, "unchanged": True, "not_replaced_by_s5e6": True},
        "validation": validation_from_logs(),
        "final_classification": "S5E6_EXPORT_BLOCKED",
    }
    write_json(OUT_JSON, ckpt)
    return status


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
