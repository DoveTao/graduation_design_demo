#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from s5e2_adjacent_dense_lib import image_stats, pair_features, read_json, scan_frames, split_sequence_keys, adjacent_pairs, relative_pose_A_to_B_in_B
from s5e7_direction_scale_lib import load_npz_model, predict_s5e2, predict_s5e3_head, rotation_matrix_to_rotvec
from s5e9_small_motion_geometry_lib import assign_bucket, compute_bucket_stats, make_samples as make_s5e9_samples


CONFIG_PATH = Path("configs/s5e11_correspondence_parallax_translation_geometry.yaml")
RESULT_DIR = Path("external_baselines/results/s5e11_traceable_dense")
CKPT_DIR = Path("checkpoints/S5E11_correspondence_parallax_translation_geometry_candidate")
S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")
S5E9_PROV = Path("external_baselines/results/s5e9_traceable_dense/edge_provenance.jsonl")
S5E9_CKPT = Path("checkpoints/S5E9_scale_unit_small_motion_signed_direction_fix_candidate.json")


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    defaults = {
        "experiment": "S5E11_correspondence_parallax_translation_geometry",
        "report_language": "zh",
        "feature_extraction": {
            "resized_height": 160,
            "resized_width": 320,
            "max_keypoints": 400,
            "quality_level": 0.01,
            "min_distance": 7,
            "optical_flow_win_size": 21,
            "optical_flow_max_level": 3,
            "essential_ransac_threshold": 1.0,
        },
        "observability": {
            "large_motion_prior_quantile": 0.75,
            "low_parallax_flow_quantile": 0.25,
            "low_match_quantile": 0.20,
            "low_inlier_ratio": 0.20,
            "near_static_flow_quantile": 0.10,
            "near_static_diff_quantile": 0.10,
        },
        "direction_mask": {
            "near_static_unobservable": 0.0,
            "low_parallax_unreliable": 0.25,
            "normal_observable": 1.0,
            "large_motion_observable": 1.15,
            "outlier_correspondence": 0.10,
        },
        "candidate_search": {
            "essential_conf_thresholds": [0.15, 0.25, 0.35, 0.50],
            "inlier_ratio_thresholds": [0.10, 0.20, 0.30],
            "parallax_quantiles": [0.25, 0.50, 0.75],
        },
        "scale_path": {
            "default_source": "S5E9_tight_bounded_residual",
            "reuse_s5e10_scale_recalibration": False,
        },
    }
    if yaml is None or not path.exists():
        return defaults
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key, value in defaults.items():
        if key not in loaded:
            loaded[key] = value
        elif isinstance(value, dict):
            merged = dict(value)
            merged.update(loaded[key] or {})
            loaded[key] = merged
    return loaded


def normalize(v: Sequence[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    return arr / max(n, 1.0e-12)


def load_s5e9_provenance() -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    if not S5E9_PROV.exists():
        return rows
    for line in S5E9_PROV.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["edge_index"])] = row
    return rows


def load_gray(path: str, size: Tuple[int, int]) -> np.ndarray:
    from PIL import Image

    h, w = size
    img = Image.open(path).convert("L").resize((w, h))
    return np.asarray(img, dtype=np.uint8)


def _phase_correlation_shift(img_i: np.ndarray, img_j: np.ndarray) -> Tuple[float, float, float]:
    a = img_i.astype(np.float32)
    b = img_j.astype(np.float32)
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    eps = 1.0e-6
    cross = fa * np.conj(fb)
    cross /= np.maximum(np.abs(cross), eps)
    corr = np.fft.ifft2(cross)
    corr_abs = np.abs(corr)
    peak = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
    py, px = peak
    h, w = a.shape
    dy = float(py if py < h / 2 else py - h)
    dx = float(px if px < w / 2 else px - w)
    confidence = float(corr_abs[peak] / max(np.mean(corr_abs), eps))
    return dx, dy, confidence


def _safe_percentile(vals: Sequence[float], q: float) -> Optional[float]:
    arr = np.asarray(list(vals), dtype=np.float64)
    return None if arr.size == 0 else float(np.percentile(arr, q))


def _flow_angle_stats(flow: np.ndarray) -> Tuple[float, float]:
    if flow.size == 0:
        return 0.0, 180.0
    ang = np.rad2deg(np.arctan2(flow[:, 1], flow[:, 0]))
    center = np.angle(np.mean(np.exp(1j * np.deg2rad(ang))))
    center_deg = float(np.rad2deg(center))
    dispersion = float(np.mean(np.abs(((ang - center_deg + 180.0) % 360.0) - 180.0)))
    return center_deg, dispersion


def extract_pair_correspondence_features(frame_i: Any, frame_j: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    fe_cfg = config["feature_extraction"]
    size = (int(fe_cfg["resized_height"]), int(fe_cfg["resized_width"]))
    img_i = load_gray(frame_i.image_path, size)
    img_j = load_gray(frame_j.image_path, size)
    diff = np.abs(img_j.astype(np.float32) - img_i.astype(np.float32)) / 255.0

    pts = None
    if cv2 is not None:
        pts = cv2.goodFeaturesToTrack(
            img_i,
            maxCorners=int(fe_cfg["max_keypoints"]),
            qualityLevel=float(fe_cfg["quality_level"]),
            minDistance=float(fe_cfg["min_distance"]),
        )

    matched_keypoint_count = 0
    inlier_match_count = 0
    inlier_ratio = 0.0
    median_flow_magnitude = 0.0
    p90_flow_magnitude = 0.0
    median_flow_angle = 0.0
    flow_angle_dispersion = 180.0
    forward_backward_flow_consistency = 0.0
    parallax_proxy = 0.0
    essential_matrix_success = False
    essential_inlier_count = 0
    essential_inlier_ratio = 0.0
    essential_translation_axis = [0.0, 0.0, 1.0]
    essential_translation_axis_confidence = 0.0
    fallback_proxy_features_used = False

    if cv2 is None:
        gx = np.diff(img_i.astype(np.float32), axis=1, prepend=img_i[:, :1])
        gy = np.diff(img_i.astype(np.float32), axis=0, prepend=img_i[:1, :])
        grad_mag = np.sqrt(gx * gx + gy * gy)
        matched_keypoint_count = int(np.sum(grad_mag > np.percentile(grad_mag, 90)))
        dx, dy, conf = _phase_correlation_shift(img_i, img_j)
        median_flow_magnitude = float(np.hypot(dx, dy))
        p90_flow_magnitude = median_flow_magnitude
        median_flow_angle = float(np.rad2deg(np.arctan2(dy, dx))) if median_flow_magnitude > 1.0e-12 else 0.0
        flow_angle_dispersion = 180.0 * max(0.0, 1.0 - min(conf / 10.0, 1.0))
        forward_backward_flow_consistency = float(min(conf / 10.0, 1.0))
        parallax_proxy = float(median_flow_magnitude * forward_backward_flow_consistency)
        inlier_match_count = int(matched_keypoint_count * forward_backward_flow_consistency)
        inlier_ratio = float(inlier_match_count / max(matched_keypoint_count, 1))
        essential_translation_axis = normalize([dx, dy, 1.0e-3]).tolist()
        essential_translation_axis_confidence = float(min(conf / 10.0, 1.0) * min(median_flow_magnitude / 5.0, 1.0))
        fallback_proxy_features_used = True
    elif pts is None or len(pts) < 8:
        fallback_proxy_features_used = True
        pts = np.zeros((0, 1, 2), dtype=np.float32)
        flow = np.zeros((0, 2), dtype=np.float32)
    else:
        pts2, st, _err = cv2.calcOpticalFlowPyrLK(
            img_i,
            img_j,
            pts,
            None,
            winSize=(int(fe_cfg["optical_flow_win_size"]), int(fe_cfg["optical_flow_win_size"])),
            maxLevel=int(fe_cfg["optical_flow_max_level"]),
        )
        back, st_back, _ = cv2.calcOpticalFlowPyrLK(
            img_j,
            img_i,
            pts2,
            None,
            winSize=(int(fe_cfg["optical_flow_win_size"]), int(fe_cfg["optical_flow_win_size"])),
            maxLevel=int(fe_cfg["optical_flow_max_level"]),
        )
        good = (st.reshape(-1) > 0) & (st_back.reshape(-1) > 0)
        p0 = pts.reshape(-1, 2)[good]
        p1 = pts2.reshape(-1, 2)[good]
        fb = np.linalg.norm(back.reshape(-1, 2)[good] - p0, axis=1) if np.any(good) else np.zeros(0, dtype=np.float32)
        matched_keypoint_count = int(p0.shape[0])
        flow = (p1 - p0).astype(np.float32)
        if matched_keypoint_count > 0:
            mags = np.linalg.norm(flow, axis=1)
            median_flow_magnitude = float(np.percentile(mags, 50))
            p90_flow_magnitude = float(np.percentile(mags, 90))
            median_flow_angle, flow_angle_dispersion = _flow_angle_stats(flow)
            forward_backward_flow_consistency = float(np.mean(fb < 1.5)) if fb.size else 0.0
            parallax_proxy = float(median_flow_magnitude * max(forward_backward_flow_consistency, 1.0e-6))
            try:
                E, mask = cv2.findEssentialMat(
                    p0,
                    p1,
                    focal=float(size[1]),
                    pp=(float(size[1]) / 2.0, float(size[0]) / 2.0),
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=float(fe_cfg["essential_ransac_threshold"]),
                )
                if E is not None and mask is not None:
                    essential_matrix_success = True
                    essential_inlier_count = int(mask.reshape(-1).sum())
                    essential_inlier_ratio = float(essential_inlier_count / max(matched_keypoint_count, 1))
                    _, _R, t, mask_pose = cv2.recoverPose(E, p0, p1, focal=float(size[1]), pp=(float(size[1]) / 2.0, float(size[0]) / 2.0))
                    essential_translation_axis = normalize(t.reshape(-1)).tolist()
                    pose_inlier = int(mask_pose.reshape(-1).sum()) if mask_pose is not None else essential_inlier_count
                    inlier_match_count = pose_inlier
                    inlier_ratio = float(pose_inlier / max(matched_keypoint_count, 1))
                    essential_translation_axis_confidence = float(
                        essential_inlier_ratio * max(forward_backward_flow_consistency, 1.0e-6) * min(1.0, median_flow_magnitude / 5.0)
                    )
                else:
                    fallback_proxy_features_used = True
            except cv2.error:
                fallback_proxy_features_used = True
        else:
            fallback_proxy_features_used = True

    image_difference_mean = float(np.mean(diff))
    image_difference_p90 = float(np.percentile(diff, 90))
    low_parallax_flag = bool(p90_flow_magnitude < 1.5 or median_flow_magnitude < 0.75)
    near_static_flag = bool(median_flow_magnitude < 0.35 and image_difference_mean < 0.025)

    correspondence_features_available = cv2 is not None
    optical_flow_features_available = cv2 is not None
    essential_matrix_features_available = bool(cv2 is not None and essential_matrix_success)

    return {
        "matched_keypoint_count": matched_keypoint_count,
        "inlier_match_count": inlier_match_count,
        "inlier_ratio": inlier_ratio,
        "median_flow_magnitude": median_flow_magnitude,
        "p90_flow_magnitude": p90_flow_magnitude,
        "median_flow_angle": median_flow_angle,
        "flow_angle_dispersion": flow_angle_dispersion,
        "forward_backward_flow_consistency": forward_backward_flow_consistency,
        "parallax_proxy": parallax_proxy,
        "low_parallax_flag": low_parallax_flag,
        "near_static_flag": near_static_flag,
        "image_difference_mean": image_difference_mean,
        "image_difference_p90": image_difference_p90,
        "essential_matrix_success": essential_matrix_success,
        "essential_inlier_count": essential_inlier_count,
        "essential_inlier_ratio": essential_inlier_ratio,
        "essential_translation_axis": essential_translation_axis,
        "essential_translation_axis_confidence": essential_translation_axis_confidence,
        "correspondence_features_available": correspondence_features_available,
        "optical_flow_features_available": optical_flow_features_available,
        "essential_matrix_features_available": essential_matrix_features_available,
        "fallback_proxy_features_used": bool(fallback_proxy_features_used),
    }


def base_prior_direction(sample: Any, cache: Dict[str, np.ndarray], s5e2: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    x = pair_features(sample.frame_i, sample.frame_j, sample.edge_index, sample.total_edges, cache)
    pred = predict_s5e2(s5e2, x)
    return normalize(pred[3:6]), x


def base_prior_magnitude(sample: Any, x_pair: np.ndarray, s5e3: Dict[str, np.ndarray], bucket_stats: Dict[str, Any]) -> Tuple[float, str]:
    prior_mag = float(np.exp(predict_s5e3_head(s5e3, "mag", x_pair).reshape(-1)[0]))
    bucket_name, _ = assign_bucket(prior_mag, bucket_stats)
    return prior_mag, bucket_name


def compute_observability_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    matched = np.asarray([r["features"]["matched_keypoint_count"] for r in records], dtype=np.float64)
    parallax = np.asarray([r["features"]["parallax_proxy"] for r in records], dtype=np.float64)
    image_diff = np.asarray([r["features"]["image_difference_mean"] for r in records], dtype=np.float64)
    inlier_ratio = np.asarray([r["features"]["inlier_ratio"] for r in records], dtype=np.float64)
    prior_mag = np.asarray([r["prior_mag_est"] for r in records], dtype=np.float64)
    return {
        "matched_keypoint_low_threshold": float(np.percentile(matched, 20)),
        "parallax_low_threshold": float(np.percentile(parallax, 25)),
        "near_static_flow_threshold": float(np.percentile([r["features"]["median_flow_magnitude"] for r in records], 10)),
        "near_static_diff_threshold": float(np.percentile(image_diff, 10)),
        "low_inlier_ratio_threshold": max(0.2, float(np.percentile(inlier_ratio, 25))),
        "large_motion_prior_threshold": float(np.percentile(prior_mag, 75)),
        "small_motion_prior_threshold": float(np.percentile(prior_mag, 25)),
        "essential_conf_threshold": max(0.15, float(np.percentile([r["features"]["essential_translation_axis_confidence"] for r in records], 60))),
    }


def assign_observability_bucket(features: Dict[str, Any], stats: Dict[str, Any], prior_mag_est: float) -> str:
    if features["near_static_flag"] or (
        features["median_flow_magnitude"] <= stats["near_static_flow_threshold"]
        and features["image_difference_mean"] <= stats["near_static_diff_threshold"]
    ):
        return "near_static_unobservable"
    if features["matched_keypoint_count"] < stats["matched_keypoint_low_threshold"] or features["forward_backward_flow_consistency"] < 0.35:
        return "outlier_correspondence"
    if (
        features["low_parallax_flag"]
        or features["parallax_proxy"] <= stats["parallax_low_threshold"]
        or features["inlier_ratio"] < stats["low_inlier_ratio_threshold"]
    ):
        return "low_parallax_unreliable"
    if prior_mag_est >= stats["large_motion_prior_threshold"]:
        return "large_motion_observable"
    return "normal_observable"


def signed_direction_weight(bucket: str, config: Dict[str, Any]) -> float:
    return float(config["direction_mask"].get(bucket, 1.0))


def build_ordered_geometry_feature(
    sample: Any,
    cache: Dict[str, np.ndarray],
    features: Dict[str, Any],
    s5e2: Dict[str, np.ndarray],
    s5e3: Dict[str, np.ndarray],
    bucket_stats: Dict[str, Any],
) -> np.ndarray:
    x = pair_features(sample.frame_i, sample.frame_j, sample.edge_index, sample.total_edges, cache)
    prior = predict_s5e2(s5e2, x)
    prior_dir = normalize(prior[3:6])
    prior_mag = float(np.exp(predict_s5e3_head(s5e3, "mag", x).reshape(-1)[0]))
    bucket_name, bucket_index = assign_bucket(prior_mag, bucket_stats)
    return np.concatenate(
        [
            x.astype(np.float64),
            np.asarray(prior_dir, dtype=np.float64),
            np.asarray([prior_mag, bucket_stats[bucket_name]["median"], bucket_stats[bucket_name]["p95"], float(bucket_index)], dtype=np.float64),
            np.asarray(
                [
                    float(features["matched_keypoint_count"]),
                    float(features["inlier_match_count"]),
                    float(features["inlier_ratio"]),
                    float(features["median_flow_magnitude"]),
                    float(features["p90_flow_magnitude"]),
                    float(features["median_flow_angle"]) / 180.0,
                    float(features["flow_angle_dispersion"]) / 180.0,
                    float(features["forward_backward_flow_consistency"]),
                    float(features["parallax_proxy"]),
                    float(features["image_difference_mean"]),
                    float(features["image_difference_p90"]),
                    float(features["essential_inlier_ratio"]),
                    float(features["essential_translation_axis_confidence"]),
                    1.0 if features["low_parallax_flag"] else 0.0,
                    1.0 if features["near_static_flag"] else 0.0,
                    1.0 if features["essential_matrix_success"] else 0.0,
                ],
                dtype=np.float64,
            ),
        ]
    )


def make_feature_records() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    train_samples, val_samples, split_summary, bucket_stats = make_s5e9_samples()
    cfg = load_config()
    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    cache: Dict[str, np.ndarray] = {}

    def _rows(samples: Sequence[Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for s in samples:
            prior_dir, x_pair = base_prior_direction(s, cache, s5e2)
            prior_mag_est, prior_bucket = base_prior_magnitude(s, x_pair, s5e3, bucket_stats)
            feats = extract_pair_correspondence_features(s.frame_i, s.frame_j, cfg)
            rows.append(
                {
                    "sample": s,
                    "prior_dir": prior_dir.tolist(),
                    "prior_mag_est": prior_mag_est,
                    "prior_bucket": prior_bucket,
                    "features": feats,
                    "gt_direction": s.gt_direction.tolist(),
                    "gt_magnitude": float(s.gt_magnitude),
                }
            )
        return rows

    train_rows = _rows(train_samples)
    val_rows = _rows(val_samples)
    obs_stats = compute_observability_stats(train_rows + val_rows)
    for rows in (train_rows, val_rows):
        for row in rows:
            row["observability_bucket"] = assign_observability_bucket(row["features"], obs_stats, row["prior_mag_est"])
            row["signed_direction_loss_weight"] = signed_direction_weight(row["observability_bucket"], cfg)
    return train_rows, val_rows, split_summary, obs_stats


def direction_metrics(rows: Sequence[Dict[str, Any]], pred_dirs: Sequence[np.ndarray]) -> Dict[str, Any]:
    signed, absvals, cosines, anti, severe, bad = [], [], [], [], [], []
    for row, pred_dir in zip(rows, pred_dirs):
        gt_dir = normalize(row["gt_direction"])
        dot = float(np.clip(np.dot(normalize(pred_dir), gt_dir), -1.0, 1.0))
        ang = float(np.rad2deg(np.arccos(dot)))
        ang_abs = float(np.rad2deg(np.arccos(abs(dot))))
        signed.append(ang)
        absvals.append(ang_abs)
        cosines.append(dot)
        anti.append(1.0 if dot < 0 else 0.0)
        severe.append(1.0 if ang > 120.0 else 0.0)
        bad.append(1.0 if ang > 120.0 and ang_abs < 45.0 else 0.0)
    return {
        "signed_tdir_mean": float(np.mean(signed)),
        "signed_tdir_median": float(np.percentile(signed, 50)),
        "signed_tdir_p90": float(np.percentile(signed, 90)),
        "tdir_abs_mean": float(np.mean(absvals)),
        "tdir_abs_median": float(np.percentile(absvals, 50)),
        "tdir_abs_p90": float(np.percentile(absvals, 90)),
        "tdir_mean_cosine": float(np.mean(cosines)),
        "anti_parallel_rate": float(np.mean(anti)),
        "severe_wrong_sign_rate": float(np.mean(severe)),
        "direction_abs_good_but_signed_bad_rate": float(np.mean(bad)),
    }


def pair_order_observable(flag_cos: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(list(flag_cos), dtype=np.float64)
    if arr.size == 0:
        return {
            "pair_order_dir_flip_cosine_mean": None,
            "pair_order_dir_flip_success_rate": None,
        }
    return {
        "pair_order_dir_flip_cosine_mean": float(np.mean(arr)),
        "pair_order_dir_flip_success_rate": float(np.mean(arr > 0.5)),
    }
