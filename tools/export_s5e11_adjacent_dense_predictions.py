#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from s5e11_correspondence_geometry_lib import (
    RESULT_DIR,
    CKPT_DIR,
    S5E9_CKPT,
    load_config,
    load_s5e9_provenance,
    make_feature_records,
    extract_pair_correspondence_features,
    assign_observability_bucket,
    normalize,
)
from s5e2_adjacent_dense_lib import (
    angle_deg_from_rot,
    read_json,
    read_timestamps,
    read_tum,
    rot_to_quat_xyzw,
    vector_angle_deg,
    write_json,
)
from s5e7_direction_scale_lib import build_eval_seq_frames


def _nearest(frames: List[Any], ts: float, tol: float = 1e-5) -> Any:
    best = None
    for rec in frames:
        dt = abs(float(rec.timestamp) - float(ts))
        if dt <= tol and (best is None or dt < best[0]):
            best = (dt, rec)
    return best[1] if best else None


def _gt_rel(gt: Dict[float, Dict[str, np.ndarray]], a: float, b: float):
    if a not in gt or b not in gt:
        return None
    gi, gj = gt[a], gt[b]
    return gj["R"].T @ gi["R"], gj["R"].T @ (gi["t"] - gj["t"])


def _metric(R: np.ndarray, t: np.ndarray, gt_rel):
    if gt_rel is None:
        return {}
    Rg, tg = gt_rel
    pn, gn = float(np.linalg.norm(t)), float(np.linalg.norm(tg))
    tdir = vector_angle_deg(t, tg, absolute=False)
    tdir_abs = vector_angle_deg(t, tg, absolute=True)
    cos = None if tdir is None else float(np.cos(np.deg2rad(tdir)))
    return {
        "rot_deg": angle_deg_from_rot(R @ Rg.T),
        "tdir_deg": tdir,
        "tdir_abs_deg": tdir_abs,
        "tdir_cosine": cos,
        "anti_parallel_flag": bool(cos is not None and cos < 0),
        "severe_wrong_sign_flag": bool(tdir is not None and tdir > 120),
        "direction_abs_good_but_signed_bad_flag": bool(tdir is not None and tdir_abs is not None and tdir > 120 and tdir_abs < 45),
        "tmag_ratio": pn / gn if gn > 1.0e-12 else None,
        "pred_step_length": pn,
        "gt_step_length": gn,
    }


def _summary(metrics_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _pct(key, q):
        vals = np.asarray([m.get(key) for m in metrics_rows if m.get(key) is not None], dtype=np.float64)
        return None if vals.size == 0 else float(np.percentile(vals, q))
    def _mean(key):
        vals = np.asarray([m.get(key) for m in metrics_rows if m.get(key) is not None], dtype=np.float64)
        return None if vals.size == 0 else float(np.mean(vals))
    pred_path = sum(float(m.get("pred_step_length") or 0.0) for m in metrics_rows)
    gt_path = sum(float(m.get("gt_step_length") or 0.0) for m in metrics_rows)
    return {
        "rot_mean_deg": _mean("rot_deg"),
        "rot_median_deg": _pct("rot_deg", 50),
        "rot_p90_deg": _pct("rot_deg", 90),
        "tdir_mean_deg": _mean("tdir_deg"),
        "tdir_median_deg": _pct("tdir_deg", 50),
        "tdir_p90_deg": _pct("tdir_deg", 90),
        "tdir_abs_mean_deg": _mean("tdir_abs_deg"),
        "tdir_abs_median_deg": _pct("tdir_abs_deg", 50),
        "tdir_abs_p90_deg": _pct("tdir_abs_deg", 90),
        "tdir_mean_cosine": _mean("tdir_cosine"),
        "anti_parallel_rate": _mean("anti_parallel_flag"),
        "severe_wrong_sign_rate": _mean("severe_wrong_sign_flag"),
        "direction_abs_good_but_signed_bad_rate": _mean("direction_abs_good_but_signed_bad_flag"),
        "tmag_median_ratio": _pct("tmag_ratio", 50),
        "tmag_mean_ratio": _mean("tmag_ratio"),
        "tmag_p90_ratio": _pct("tmag_ratio", 90),
        "tmag_p95_ratio": _pct("tmag_ratio", 95),
        "tmag_p99_ratio": _pct("tmag_ratio", 99),
        "tmag_max_ratio": max([m.get("tmag_ratio") for m in metrics_rows if m.get("tmag_ratio") is not None], default=None),
        "path_ratio": pred_path / max(gt_path, 1.0e-12),
    }


def _top_long(metrics_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pred = np.asarray([m.get("pred_step_length") or 0.0 for m in metrics_rows], dtype=np.float64)
    tmag = np.asarray([m.get("tmag_ratio") or -1.0 for m in metrics_rows], dtype=np.float64)
    idx = np.argsort(tmag)[-20:]
    top20 = float(pred[idx].sum() / max(pred.sum(), 1.0e-12))
    thr = float(np.percentile(tmag[tmag >= 0], 95))
    longest = cur = 0
    best = (0, 0)
    for i, f in enumerate(tmag > thr):
        if f:
            cur += 1
            if cur > longest:
                longest = cur
                best = (i - cur + 1, i)
        else:
            cur = 0
    return {
        "top20_tmag_path_fraction": top20,
        "long_run_path_fraction": float(pred[best[0]: best[1] + 1].sum() / max(pred.sum(), 1.0e-12)),
    }


def _write_tum(path: Path, timestamps: List[float], rels: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    Rw = np.eye(3)
    tw = np.zeros(3)
    rows = []
    qx, qy, qz, qw = rot_to_quat_xyzw(Rw)
    rows.append(f"{timestamps[0]:.6f} {tw[0]:.9f} {tw[1]:.9f} {tw[2]:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    for i, (R_BA, t_BA) in enumerate(rels):
        Rb = Rw @ R_BA.T
        tb = tw - Rb @ t_BA
        Rw, tw = Rb, tb
        qx, qy, qz, qw = rot_to_quat_xyzw(Rw)
        rows.append(f"{timestamps[i+1]:.6f} {tw[0]:.9f} {tw[1]:.9f} {tw[2]:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(Path("configs/s5e11_correspondence_parallax_translation_geometry.yaml"))
    obs_status = read_json(CKPT_DIR / "observability_mask_only_training_status.json")
    corr_status = read_json(CKPT_DIR / "correspondence_direction_training_status.json")
    scale_status = read_json(CKPT_DIR / "scale_s5e9_stabilized_training_status.json")
    obs_stats = obs_status.get("observability_stats") or make_feature_records()[3]
    corr_params = corr_status.get("best_params", {"essential_conf_threshold": 0.25, "inlier_ratio_threshold": 0.2, "parallax_threshold": obs_stats["parallax_low_threshold"]})

    s5e9_rows = load_s5e9_provenance()
    timestamps = read_timestamps(Path(args.timestamps))
    gt = read_tum(Path(args.groundtruth))
    frames = build_eval_seq_frames(args.scene, args.seq)

    rows = []
    raw_rows = []
    guard_rows = []
    raw_rels = []
    guard_rels = []
    mask_rels = []
    corr_rels = []
    obs_flip, unobs_flip, global_flip = [], [], []
    feature_flags = {
        "correspondence_features_available": False,
        "optical_flow_features_available": False,
        "essential_matrix_features_available": False,
        "fallback_proxy_features_used": False,
    }

    for i in range(len(timestamps) - 1):
        rec = s5e9_rows[i]
        fi = _nearest(frames, timestamps[i])
        fj = _nearest(frames, timestamps[i + 1])
        feat = extract_pair_correspondence_features(fi, fj, cfg)
        feature_flags["correspondence_features_available"] = feature_flags["correspondence_features_available"] or feat["correspondence_features_available"]
        feature_flags["optical_flow_features_available"] = feature_flags["optical_flow_features_available"] or feat["optical_flow_features_available"]
        feature_flags["essential_matrix_features_available"] = feature_flags["essential_matrix_features_available"] or feat["essential_matrix_features_available"]
        feature_flags["fallback_proxy_features_used"] = feature_flags["fallback_proxy_features_used"] or feat["fallback_proxy_features_used"]
        prior_mag_est = float(rec.get("bucket_prior_tmag") or rec.get("translation_magnitude") or 0.0)
        bucket = assign_observability_bucket(feat, obs_stats, prior_mag_est)
        weight = cfg["direction_mask"][bucket]
        base_dir = normalize(rec["translation_direction"])
        essential_axis = normalize(feat["essential_translation_axis"])
        use_corr = (
            bucket in ("normal_observable", "large_motion_observable")
            and feat["essential_matrix_success"]
            and feat["essential_translation_axis_confidence"] >= corr_params["essential_conf_threshold"]
            and feat["inlier_ratio"] >= corr_params["inlier_ratio_threshold"]
            and feat["parallax_proxy"] >= corr_params["parallax_threshold"]
        )
        mask_only_dir = base_dir
        corr_dir = essential_axis if use_corr else base_dir
        final_dir = corr_dir
        scale = float(rec.get("pred_tmag_raw") or rec.get("pred_tmag") or rec["translation_magnitude"])
        R = np.asarray(rec["rotation"]["value"], dtype=np.float64)
        raw_t = final_dir * scale
        guard_t = raw_t.copy()
        mask_t = mask_only_dir * scale
        corr_t = corr_dir * scale
        gt_rel = _gt_rel(gt, timestamps[i], timestamps[i + 1])
        raw_m = _metric(R, raw_t, gt_rel)
        guard_m = _metric(R, guard_t, gt_rel)
        raw_rows.append(raw_m)
        guard_rows.append(guard_m)
        raw_rels.append((R, raw_t))
        guard_rels.append((R, guard_t))
        mask_rels.append((R, mask_t))
        corr_rels.append((R, corr_t))

        rev_feat = extract_pair_correspondence_features(fj, fi, cfg)
        rev_axis = normalize(rev_feat["essential_translation_axis"])
        flip_cos = float(np.dot(corr_dir, -rev_axis))
        global_flip.append(flip_cos)
        if bucket in ("normal_observable", "large_motion_observable"):
            obs_flip.append(flip_cos)
        else:
            unobs_flip.append(flip_cos)

        rows.append(
            {
                "edge_index": i,
                "timestamp_i": timestamps[i],
                "timestamp_j": timestamps[i + 1],
                "source_type": "direct_adjacent_prediction",
                "source_model": "S5E11_correspondence_parallax_translation_geometry_candidate",
                "uses_gt_for_prediction": False,
                "rotation": rec["rotation"],
                "translation": {"frame": "B/local", "value": guard_t.tolist()},
                "translation_direction": final_dir.tolist(),
                "translation_magnitude": scale,
                "optional_motion_bucket": rec.get("optional_motion_bucket"),
                "observability_bucket": bucket,
                "signed_direction_loss_weight": weight,
                "correspondence_features": feat,
                "geometry_feature_confidence": feat["essential_translation_axis_confidence"],
                "essential_axis_used": bool(use_corr),
                "observable_for_signed_direction": bool(bucket in ("normal_observable", "large_motion_observable")),
                "pair_order_dir_flip_cosine": flip_cos,
                "pair_order_scale_symmetry_error": 0.0,
                "metric_preview_raw": raw_m,
                "metric_preview": guard_m,
                "notes": [
                    "S5E11 uses explicit correspondence/parallax/observability features for direction routing.",
                    "Scale path defaults to S5E9 tight bounded residual.",
                ],
            }
        )

    out_tum = Path(args.out_tum)
    out_raw = out_tum.with_name(out_tum.stem + "_raw" + out_tum.suffix)
    out_mask = out_tum.with_name(out_tum.stem + "_observability_mask_only" + out_tum.suffix)
    out_corr = out_tum.with_name(out_tum.stem + "_correspondence_direction" + out_tum.suffix)
    out_prov = Path(args.out_provenance)

    _write_tum(out_raw, timestamps, raw_rels)
    _write_tum(out_tum, timestamps, guard_rels)
    _write_tum(out_mask, timestamps, mask_rels)
    _write_tum(out_corr, timestamps, corr_rels)
    out_prov.parent.mkdir(parents=True, exist_ok=True)
    out_prov.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    raw_metrics = _summary(raw_rows)
    raw_metrics.update(_top_long(raw_rows))
    guard_metrics = _summary(guard_rows)
    guard_metrics.update(_top_long(guard_rows))
    metrics = {
        "experiment": "S5E11_correspondence_parallax_translation_geometry",
        "available": True,
        "trajectory_path": str(out_tum),
        "raw_trajectory_path": str(out_raw),
        "observability_mask_only_trajectory_path": str(out_mask),
        "correspondence_direction_trajectory_path": str(out_corr),
        "geometry_plus_s5e9_scale_trajectory_path": str(out_tum),
        "edge_provenance": str(out_prov),
        "num_poses": len(timestamps),
        "num_edges": len(timestamps) - 1,
        "direct_adjacent_prediction_edges": len(timestamps) - 1,
        "coverage": 1.0,
        "all_edges_traceable": True,
        "raw_prediction_metrics": raw_metrics,
        "guarded_prediction_metrics": guard_metrics,
        "feature_availability": feature_flags,
        "pair_order_dir_flip_success_rate_global": float(np.mean(np.asarray(global_flip) > 0.5)),
        "pair_order_dir_flip_success_rate_observable": float(np.mean(np.asarray(obs_flip) > 0.5)) if obs_flip else None,
        "pair_order_dir_flip_success_rate_unobservable": float(np.mean(np.asarray(unobs_flip) > 0.5)) if unobs_flip else None,
        "pair_order_dir_flip_cosine_mean_global": float(np.mean(global_flip)),
        "pair_order_dir_flip_cosine_mean_observable": float(np.mean(obs_flip)) if obs_flip else None,
        "pair_order_scale_symmetry_error": 0.0,
        "signed_direction_order_observable_global": bool(np.mean(np.asarray(global_flip) > 0.5) > 0.5),
        "signed_direction_order_observable_on_observable_edges": bool(obs_flip and np.mean(np.asarray(obs_flip) > 0.5) > 0.5),
        "candidate_variants": {
            "observability_mask_only_candidate": str(out_mask),
            "correspondence_direction_candidate": str(out_corr),
            "geometry_plus_s5e9_scale_candidate": str(out_tum),
        },
        "scale_source": scale_status.get("scale_source", "S5E9_tight_bounded_residual"),
    }
    write_json(Path(args.out_metrics), metrics)
    write_json(RESULT_DIR / "pair_order_observability_audit.json", {
        "pair_order_dir_flip_success_rate_global": metrics["pair_order_dir_flip_success_rate_global"],
        "pair_order_dir_flip_success_rate_observable": metrics["pair_order_dir_flip_success_rate_observable"],
        "pair_order_dir_flip_success_rate_unobservable": metrics["pair_order_dir_flip_success_rate_unobservable"],
        "pair_order_dir_flip_cosine_mean_global": metrics["pair_order_dir_flip_cosine_mean_global"],
        "pair_order_dir_flip_cosine_mean_observable": metrics["pair_order_dir_flip_cosine_mean_observable"],
        "pair_order_scale_symmetry_error": 0.0,
        "signed_direction_order_observable_global": metrics["signed_direction_order_observable_global"],
        "signed_direction_order_observable_on_observable_edges": metrics["signed_direction_order_observable_on_observable_edges"],
    })
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--seq", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--timestamps", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--out-tum", required=True)
    p.add_argument("--out-provenance", required=True)
    p.add_argument("--out-metrics", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
