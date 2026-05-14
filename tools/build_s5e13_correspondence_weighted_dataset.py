#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from s5e12_extract_real_correspondence_features import extract_orb_matches, extract_sparse_flow, load_gray
from s5e2_adjacent_dense_lib import adjacent_pairs, pair_features, read_json, relative_pose_A_to_B_in_B, scan_frames, write_json
from s5e7_direction_scale_lib import load_npz_model, predict_s5e2, predict_s5e3_head, rotation_matrix_to_rotvec
from s5e9_small_motion_geometry_lib import assign_bucket, compute_bucket_stats


S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")


def load_config(path: Path) -> Dict[str, Any]:
    def parse_scalar(text: str) -> Any:
        text = text.strip()
        if text.lower() == "true":
            return True
        if text.lower() == "false":
            return False
        try:
            if "." in text:
                return float(text)
            return int(text)
        except Exception:
            return text

    cfg: Dict[str, Any] = {}
    current: str | None = None
    mode: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if not raw.startswith(" "):
            if ":" in raw:
                key, value = raw.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value:
                    cfg[key] = parse_scalar(value)
                    current = None
                    mode = None
                else:
                    cfg[key] = {}
                    current = key
                    mode = "dict"
        else:
            if current is None:
                continue
            if raw.startswith("  - "):
                if not isinstance(cfg[current], list):
                    cfg[current] = []
                cfg[current].append(parse_scalar(raw[4:]))
                mode = "list"
            elif raw.startswith("  ") and ":" in raw:
                if mode == "list":
                    continue
                if not isinstance(cfg[current], dict):
                    cfg[current] = {}
                key, value = raw.strip().split(":", 1)
                cfg[current][key.strip()] = parse_scalar(value.strip())
    return cfg


def _gt_direction_and_mag(a: Any, b: Any) -> Tuple[np.ndarray, float, np.ndarray]:
    R, t = relative_pose_A_to_B_in_B(a, b)
    mag = float(np.linalg.norm(t))
    direction = t / max(mag, 1.0e-12)
    return direction.astype(np.float64), mag, rotation_matrix_to_rotvec(R).astype(np.float64)


def _extract_real_features(frame_i: Any, frame_j: Any, cfg: Dict[str, Any], gt_tmag: float | None) -> Dict[str, Any]:
    w = int(cfg["image"]["resized_width"])
    h = int(cfg["image"]["resized_height"])
    img_i = load_gray(frame_i.image_path, (w, h))
    img_j = load_gray(frame_j.image_path, (w, h))
    orb = extract_orb_matches(
        img_i,
        img_j,
        nfeatures=int(cfg["detector"]["nfeatures"]),
        fast_threshold=int(cfg["detector"]["fastThreshold"]),
        ratio_test=float(cfg["matcher"]["ratio_test"]),
        cross_check=bool(cfg["matcher"]["cross_check"]),
    )
    flow = extract_sparse_flow(
        img_i,
        img_j,
        max_corners=int(cfg["flow"]["max_corners"]),
        quality_level=float(cfg["flow"]["quality_level"]),
        min_distance=float(cfg["flow"]["min_distance"]),
        win_size=int(cfg["flow"]["win_size"]),
        max_level=int(cfg["flow"]["max_level"]),
    )
    diff = np.abs(img_j.astype(np.float32) - img_i.astype(np.float32)) / 255.0
    return {
        "correspondence_success": bool(orb["correspondence_success"]),
        "correspondence_failure_reason": orb["correspondence_failure_reason"],
        "detector_type": "ORB",
        "keypoint_count_i": int(orb["keypoint_count_i"]),
        "keypoint_count_j": int(orb["keypoint_count_j"]),
        "raw_match_count": int(orb["raw_match_count"]),
        "filtered_match_count": int(orb["filtered_match_count"]),
        "inlier_match_count": int(orb["inlier_match_count"]),
        "inlier_ratio": float(orb["inlier_ratio"]),
        "median_match_displacement": float(orb["median_match_displacement"]),
        "p90_match_displacement": float(orb["p90_match_displacement"]),
        "match_angle_median": float(orb["match_angle_median"]),
        "match_angle_dispersion": float(orb["match_angle_dispersion"]),
        "parallax_proxy": float(orb["parallax_proxy"]),
        "low_parallax_flag": bool(orb["low_parallax_flag"]),
        "near_static_flag": bool(
            orb["median_match_displacement"] < float(cfg["observability"]["near_static_displacement"])
            and float(np.mean(diff)) < float(cfg["observability"]["near_static_image_diff_mean"])
        ),
        "image_difference_mean": float(np.mean(diff)),
        "image_difference_p90": float(np.percentile(diff, 90)),
        "flow_success": bool(flow["flow_success"]),
        "flow_failure_reason": flow["flow_failure_reason"],
        "tracked_point_count": int(flow["tracked_point_count"]),
        "valid_flow_count": int(flow["valid_flow_count"]),
        "median_flow_magnitude": float(flow["median_flow_magnitude"]),
        "p90_flow_magnitude": float(flow["p90_flow_magnitude"]),
        "flow_angle_median": float(flow["flow_angle_median"]),
        "flow_angle_dispersion": float(flow["flow_angle_dispersion"]),
        "forward_backward_flow_consistency": flow["forward_backward_flow_consistency"],
        "gt_tmag": None if gt_tmag is None else float(gt_tmag),
    }


def _compute_train_thresholds(train_rows: Sequence[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, float]:
    obs = cfg["observability"]
    filtered = np.asarray([r["features"]["filtered_match_count"] for r in train_rows], dtype=np.float64)
    parallax = np.asarray([r["features"]["parallax_proxy"] for r in train_rows], dtype=np.float64)
    flow = np.asarray([r["features"]["median_flow_magnitude"] for r in train_rows], dtype=np.float64)
    diff = np.asarray([r["features"]["image_difference_mean"] for r in train_rows], dtype=np.float64)
    gt_mag = np.asarray([r["gt_magnitude"] for r in train_rows], dtype=np.float64)
    return {
        "low_match_threshold": max(float(obs["min_filtered_matches"]), float(np.percentile(filtered, 5))),
        "low_inlier_threshold": float(obs["min_inlier_ratio"]),
        "low_parallax_threshold": float(np.percentile(parallax, 100.0 * float(obs["low_parallax_quantile"]))),
        "near_static_flow_threshold": float(np.percentile(flow, 100.0 * float(obs["near_static_flow_quantile"]))),
        "near_static_diff_threshold": float(np.percentile(diff, 100.0 * float(obs["near_static_diff_quantile"]))),
        "small_motion_gt_threshold": float(np.percentile(gt_mag, 100.0 * float(obs["small_motion_gt_quantile"]))),
        "large_motion_gt_threshold": float(np.percentile(gt_mag, 100.0 * float(obs["large_motion_gt_quantile"]))),
    }


def _assign_bucket_and_weights(row: Dict[str, Any], thresholds: Dict[str, float], cfg: Dict[str, Any], use_gt_for_bucket: bool) -> Dict[str, Any]:
    f = row["features"]
    wcfg = cfg["weights"]
    est_mag = float(row["prior_mag_est"])
    small_motion = bool((row["gt_magnitude"] if use_gt_for_bucket else est_mag) <= thresholds["small_motion_gt_threshold"])
    large_motion = bool((row["gt_magnitude"] if use_gt_for_bucket else est_mag) >= thresholds["large_motion_gt_threshold"])
    near_static = bool(
        f["near_static_flag"]
        or (f["median_flow_magnitude"] <= thresholds["near_static_flow_threshold"] and f["image_difference_mean"] <= thresholds["near_static_diff_threshold"])
    )
    low_parallax = bool(
        f["low_parallax_flag"]
        or (f["parallax_proxy"] <= thresholds["low_parallax_threshold"] and f["median_flow_magnitude"] <= float(cfg["observability"]["low_parallax_displacement"]))
        or f["inlier_ratio"] < thresholds["low_inlier_threshold"]
    )
    outlier_correspondence = bool(
        (not f["correspondence_success"])
        or f["filtered_match_count"] < int(thresholds["low_match_threshold"])
        or f["inlier_match_count"] < int(cfg["observability"]["min_inlier_matches"])
    )
    if outlier_correspondence:
        bucket = "outlier_correspondence"
    elif near_static:
        bucket = "near_static_unobservable"
    elif low_parallax:
        bucket = "low_parallax_unreliable"
    elif large_motion:
        bucket = "large_motion_observable"
    else:
        bucket = "normal_observable"
    observable = bucket in {"normal_observable", "large_motion_observable"}
    reliable = observable and not small_motion
    signed_weight = float(wcfg["signed_reliable"] if reliable else (wcfg["signed_partial"] if bucket == "low_parallax_unreliable" else wcfg["signed_unobservable"]))
    if small_motion:
        signed_weight *= float(wcfg["signed_small_motion_multiplier"])
    abs_weight = float(wcfg["tdir_abs_reliable"] if observable else (wcfg["tdir_abs_partial"] if bucket == "low_parallax_unreliable" else wcfg["tdir_abs_unobservable"]))
    tmag_weight = float(wcfg["tmag_near_static"] if near_static else wcfg["tmag_default"])
    reason = bucket
    if small_motion and signed_weight <= 0.26:
        reason = reason + "; small_motion_downweighted"
    row.update(
        {
            "observable": bool(observable),
            "signed_direction_reliable": bool(reliable),
            "small_motion": bool(small_motion),
            "large_motion": bool(large_motion),
            "near_static": bool(near_static),
            "low_parallax": bool(low_parallax),
            "observability_bucket": bucket,
            "signed_tdir_loss_weight": float(signed_weight),
            "tdir_abs_loss_weight": float(abs_weight),
            "tmag_loss_weight": float(tmag_weight),
            "rot_loss_weight": float(wcfg["rot_default"]),
            "reason": reason,
        }
    )
    return row


def _feature_vector(row: Dict[str, Any], bucket_stats: Dict[str, Any]) -> List[float]:
    f = row["features"]
    bucket = bucket_stats[row["prior_bucket"]]
    x = np.asarray(row["pair_feature"], dtype=np.float64)
    vec = np.concatenate(
        [
            x,
            np.asarray(row["prior_dir"], dtype=np.float64),
            np.asarray(
                [
                    row["prior_mag_est"],
                    np.log(max(row["prior_mag_est"], 1.0e-12)),
                    bucket["median"],
                    bucket["p95"],
                    bucket["clip_hi"],
                    1.0 if row["small_motion"] else 0.0,
                    1.0 if row["near_static"] else 0.0,
                    1.0 if row["low_parallax"] else 0.0,
                    1.0 if row["observable"] else 0.0,
                ],
                dtype=np.float64,
            ),
            np.asarray(
                [
                    f["filtered_match_count"],
                    f["inlier_match_count"],
                    f["inlier_ratio"],
                    f["median_match_displacement"],
                    f["p90_match_displacement"],
                    f["match_angle_median"] / 180.0,
                    f["match_angle_dispersion"] / 180.0,
                    f["parallax_proxy"],
                    f["image_difference_mean"],
                    f["image_difference_p90"],
                    f["median_flow_magnitude"],
                    f["p90_flow_magnitude"],
                    f["flow_angle_median"] / 180.0,
                    f["flow_angle_dispersion"] / 180.0,
                    0.0 if f["forward_backward_flow_consistency"] is None else float(f["forward_backward_flow_consistency"]),
                ],
                dtype=np.float64,
            ),
        ]
    )
    return [float(v) for v in vec]


def _make_rows(pairs: Sequence[Tuple[Any, Any]], cfg: Dict[str, Any], s5e2: Dict[str, np.ndarray], s5e3: Dict[str, np.ndarray], bucket_stats: Dict[str, Any], include_gt: bool) -> List[Dict[str, Any]]:
    cache: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    total = max(len(pairs), 1)
    for idx, (a, b) in enumerate(pairs):
        gt_dir, gt_mag, gt_rotvec = _gt_direction_and_mag(a, b)
        x = pair_features(a, b, idx, total, cache)
        prior = predict_s5e2(s5e2, x)
        prior_rotvec = prior[:3]
        prior_dir = prior[3:6]
        prior_dir = prior_dir / max(float(np.linalg.norm(prior_dir)), 1.0e-12)
        prior_mag_est = float(np.exp(predict_s5e3_head(s5e3, "mag", x).reshape(-1)[0]))
        prior_bucket, prior_bucket_index = assign_bucket(prior_mag_est, bucket_stats)
        feats = _extract_real_features(a, b, cfg, gt_mag if include_gt else None)
        rows.append(
            {
                "scene": a.scene,
                "seq": a.seq,
                "edge_index": idx,
                "timestamp_i": float(a.timestamp),
                "timestamp_j": float(b.timestamp),
                "image_i": a.image_path,
                "image_j": b.image_path,
                "pair_feature": [float(v) for v in x.tolist()],
                "prior_rotvec": [float(v) for v in prior_rotvec.tolist()],
                "prior_dir": [float(v) for v in prior_dir.tolist()],
                "prior_mag_est": float(prior_mag_est),
                "prior_bucket": prior_bucket,
                "prior_bucket_index": int(prior_bucket_index),
                "gt_direction": [float(v) for v in gt_dir.tolist()],
                "gt_magnitude": float(gt_mag),
                "gt_rotvec": [float(v) for v in gt_rotvec.tolist()],
                "features": feats,
            }
        )
    return rows


def _load_eval_rows_from_s5e12(feature_dir: Path) -> List[Dict[str, Any]]:
    feat = read_json(feature_dir / "correspondence_features.json")
    return list(feat.get("records", []))


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# S5E13 correspondence weighted dataset audit",
        "",
        "## 执行摘要",
        f"S5E13 weighted dataset 构建状态：`{payload['final_classification']}`。",
        "",
        "## train/eval split 与 no GT leakage",
        "- train scenes 使用 `scene01/seq01` 和 `scene01/seq02` 的 adjacent pairs。",
        "- eval scene 使用 `scene01/seq03`，其 GT 不用于训练。",
        f"- uses_scene01_seq03_gt_for_training = `{str(payload['uses_scene01_seq03_gt_for_training']).lower()}`",
        "",
        "## 权重统计",
        f"- train_weighted_edges = `{payload['train_weighted_edges']}`",
        f"- val_weighted_edges = `{payload['val_weighted_edges']}`",
        f"- eval_edges = `{payload['eval_edges']}`",
        f"- train_correspondence_features_available = `{str(payload['train_correspondence_features_available']).lower()}`",
        f"- eval_correspondence_features_available = `{str(payload['eval_correspondence_features_available']).lower()}`",
        f"- observable_fraction_train = `{payload['observable_fraction_train']}`",
        f"- observable_fraction_eval = `{payload['observable_fraction_eval']}`",
        f"- signed_direction_reliable_fraction_train = `{payload['signed_direction_reliable_fraction_train']}`",
        f"- signed_direction_reliable_fraction_eval = `{payload['signed_direction_reliable_fraction_eval']}`",
        "",
        "## 说明",
        "- observable / reliable edges 会获得更高的 signed tdir 监督权重。",
        "- near-static / low-parallax / small-motion edges 会降低 signed direction loss 权重。",
        "- strict essential geometry 没有用于训练监督。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(Path(args.config))
    feature_dir = Path(args.s5e12_feature_dir)
    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    frames = scan_frames(Path("data"), scene=str(cfg.get("scene", "scene01")))
    train_pairs: List[Tuple[Any, Any]] = []
    for seq in cfg.get("train_seqs", ["seq01", "seq02"]):
        train_pairs.extend(adjacent_pairs(frames[(cfg["scene"], seq)]))
    val_count = 79
    core_train = train_pairs[:-val_count]
    core_val = train_pairs[-val_count:]

    all_train_gt = []
    for a, b in train_pairs:
        _d, m, _r = _gt_direction_and_mag(a, b)
        all_train_gt.append(m)
    bucket_stats = compute_bucket_stats(all_train_gt)

    train_rows = _make_rows(core_train, cfg, s5e2, s5e3, bucket_stats, include_gt=True)
    val_rows = _make_rows(core_val, cfg, s5e2, s5e3, bucket_stats, include_gt=True)
    thresholds = _compute_train_thresholds(train_rows + val_rows, cfg)
    for row in train_rows:
        _assign_bucket_and_weights(row, thresholds, cfg, use_gt_for_bucket=True)
    for row in val_rows:
        _assign_bucket_and_weights(row, thresholds, cfg, use_gt_for_bucket=True)
    for row in train_rows + val_rows:
        row["model_feature_vector"] = _feature_vector(row, bucket_stats)

    s5e12_rows = _load_eval_rows_from_s5e12(feature_dir)
    eval_rows: List[Dict[str, Any]] = []
    for idx, src in enumerate(s5e12_rows):
        x_dummy = np.zeros_like(np.asarray(train_rows[0]["pair_feature"], dtype=np.float64))
        if "image_i" in src and "image_j" in src:
            # rebuild S5E2/S5E3 priors for seq03 using actual frames
            fi = next(fr for fr in frames[(cfg["scene"], cfg["eval_seq"])] if abs(fr.timestamp - float(src["timestamp_i"])) < 1.0e-5)
            fj = next(fr for fr in frames[(cfg["scene"], cfg["eval_seq"])] if abs(fr.timestamp - float(src["timestamp_j"])) < 1.0e-5)
            cache: Dict[str, np.ndarray] = {}
            x_dummy = pair_features(fi, fj, idx, len(s5e12_rows), cache)
            prior = predict_s5e2(s5e2, x_dummy)
            prior_rotvec = prior[:3]
            prior_dir = prior[3:6]
            prior_dir = prior_dir / max(float(np.linalg.norm(prior_dir)), 1.0e-12)
            prior_mag_est = float(np.exp(predict_s5e3_head(s5e3, "mag", x_dummy).reshape(-1)[0]))
            prior_bucket, prior_bucket_index = assign_bucket(prior_mag_est, bucket_stats)
        else:
            prior_rotvec = np.zeros(3, dtype=np.float64)
            prior_dir = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            prior_mag_est = float(bucket_stats["global"]["median"])
            prior_bucket, prior_bucket_index = assign_bucket(prior_mag_est, bucket_stats)
        row = {
            "scene": cfg["scene"],
            "seq": cfg["eval_seq"],
            "edge_index": idx,
            "timestamp_i": float(src["timestamp_i"]),
            "timestamp_j": float(src["timestamp_j"]),
            "image_i": src["image_i"],
            "image_j": src["image_j"],
            "pair_feature": [float(v) for v in x_dummy.tolist()],
            "prior_rotvec": [float(v) for v in prior_rotvec.tolist()],
            "prior_dir": [float(v) for v in prior_dir.tolist()],
            "prior_mag_est": float(prior_mag_est),
            "prior_bucket": prior_bucket,
            "prior_bucket_index": int(prior_bucket_index),
            "gt_direction": None,
            "gt_magnitude": None,
            "gt_rotvec": None,
            "features": src,
        }
        _assign_bucket_and_weights(row, thresholds, cfg, use_gt_for_bucket=False)
        row["model_feature_vector"] = _feature_vector(row, bucket_stats)
        eval_rows.append(row)

    train_obs = float(np.mean([1.0 if r["observable"] else 0.0 for r in train_rows + val_rows]))
    eval_obs = float(np.mean([1.0 if r["observable"] else 0.0 for r in eval_rows]))
    train_rel = float(np.mean([1.0 if r["signed_direction_reliable"] else 0.0 for r in train_rows + val_rows]))
    eval_rel = float(np.mean([1.0 if r["signed_direction_reliable"] else 0.0 for r in eval_rows]))
    final_classification = "S5E13_WEIGHTED_DATASET_READY"
    if train_rel <= 0.0:
        final_classification = "S5E13_TRAIN_CORRESPONDENCE_MISSING_PARTIAL"

    dataset_payload = {
        "experiment": "S5E13_real_correspondence_signed_direction_training",
        "thresholds": thresholds,
        "bucket_stats": bucket_stats,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "eval_rows": eval_rows,
        "train_weighted_edges": len(train_rows),
        "val_weighted_edges": len(val_rows),
        "eval_edges": len(eval_rows),
        "train_correspondence_features_available": True,
        "eval_correspondence_features_available": True,
        "uses_scene01_seq03_gt_for_training": False,
        "observable_fraction_train": train_obs,
        "observable_fraction_eval": eval_obs,
        "signed_direction_reliable_fraction_train": train_rel,
        "signed_direction_reliable_fraction_eval": eval_rel,
        "final_classification": final_classification,
    }

    weighted_dataset_json = Path(cfg["weighted_dataset_json"])
    write_json(weighted_dataset_json, dataset_payload)
    audit = {k: v for k, v in dataset_payload.items() if k not in {"train_rows", "val_rows", "eval_rows", "bucket_stats", "thresholds"}}
    write_json(Path(args.out_json), audit)
    _write_report(Path(args.out_report), audit)
    return dataset_payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--s5e12-feature-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
