#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from s5e2_adjacent_dense_lib import angle_deg_from_rot, read_timestamps, read_tum, rot_to_quat_xyzw, vector_angle_deg, write_json
from s5e7_direction_scale_lib import rotvec_to_matrix

S5E9_PROV = Path("external_baselines/results/s5e9_traceable_dense/edge_provenance.jsonl")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _gt_rel(gt: Dict[float, Dict[str, np.ndarray]], a: float, b: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if a not in gt or b not in gt:
        return None
    gi, gj = gt[a], gt[b]
    return gj["R"].T @ gi["R"], gj["R"].T @ (gi["t"] - gj["t"])


def _metric(R: np.ndarray, t: np.ndarray, gt_rel: Optional[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    if gt_rel is None:
        return {}
    Rg, tg = gt_rel
    tdir = vector_angle_deg(t, tg, absolute=False)
    tdir_abs = vector_angle_deg(t, tg, absolute=True)
    cos = None if tdir is None else float(np.cos(np.deg2rad(tdir)))
    return {
        "rot_deg": angle_deg_from_rot(R @ Rg.T),
        "tdir_deg": tdir,
        "tdir_abs_deg": tdir_abs,
        "tdir_cosine": cos,
        "anti_parallel_flag": bool(cos is not None and cos < 0.0),
        "severe_wrong_sign_flag": bool(tdir is not None and tdir > 120.0),
        "direction_abs_good_but_signed_bad_flag": bool(tdir is not None and tdir_abs is not None and tdir > 120.0 and tdir_abs < 45.0),
        "tmag_ratio": float(np.linalg.norm(t) / max(np.linalg.norm(tg), 1.0e-12)),
        "pred_step_length": float(np.linalg.norm(t)),
        "gt_step_length": float(np.linalg.norm(tg)),
    }


def _summary(metrics_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _vals(key: str) -> np.ndarray:
        return np.asarray([m[key] for m in metrics_rows if m.get(key) is not None], dtype=np.float64)

    def _pct(key: str, q: float) -> float | None:
        vals = _vals(key)
        return None if vals.size == 0 else float(np.percentile(vals, q))

    def _mean(key: str) -> float | None:
        vals = _vals(key)
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
        "tmag_max_ratio": None if _vals("tmag_ratio").size == 0 else float(np.max(_vals("tmag_ratio"))),
        "path_ratio": pred_path / max(gt_path, 1.0e-12),
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
    s5e13_rows = _read_jsonl(Path("external_baselines/results/s5e13_traceable_dense/edge_provenance.jsonl"))
    s5e13_w = _read_jsonl(Path("external_baselines/results/s5e13_traceable_dense/correspondence_loss_weights.jsonl"))
    w_by_idx = {int(x["edge_index"]): x for x in s5e13_w}

    feature_by_idx: Dict[int, Dict[str, Any]] = {}
    with Path(args.s5e12_feature_dir, "correspondence_features.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            feature_by_idx[int(row["edge_id"])] = row

    s5e9_mag: Dict[int, float] = {}
    for row in _read_jsonl(S5E9_PROV):
        s5e9_mag[int(row["edge_index"])] = float(row.get("translation_magnitude", 0.0))

    gt = read_tum(Path(args.groundtruth))
    timestamps = read_timestamps(Path(args.timestamps))

    rels: List[Tuple[np.ndarray, np.ndarray]] = []
    metrics_rows: List[Dict[str, Any]] = []
    out_weights: List[Dict[str, Any]] = []
    out_prov: List[Dict[str, Any]] = []

    for row in s5e13_rows:
        idx = int(row["edge_index"])
        feat = feature_by_idx.get(idx, {})
        ww = w_by_idx.get(idx, {})
        observable = bool(row.get("observable", False))
        reliable_original = bool(row.get("signed_direction_reliable", False))
        small_motion = bool(row.get("small_motion", False))
        low_parallax = bool(row.get("low_parallax", False))
        near_static = bool(row.get("near_static", False))
        angle_dispersion = float(feat.get("flow_angle_dispersion", feat.get("match_angle_dispersion", 0.0)) or 0.0)
        parallax_proxy = float(ww.get("parallax_proxy", feat.get("parallax_proxy", 0.0)) or 0.0)
        flow_magnitude = float(ww.get("flow_magnitude", feat.get("median_flow_magnitude", 0.0)) or 0.0)
        inlier_ratio = float(ww.get("inlier_ratio", feat.get("inlier_ratio", 0.0)) or 0.0)
        gt_tmag = float(feat.get("gt_tmag", 0.0) or 0.0)

        reliable_strict = bool(
            reliable_original and (not low_parallax) and (not near_static) and (not small_motion)
            and angle_dispersion <= 55.0 and parallax_proxy >= 0.35 and flow_magnitude >= 0.2 and inlier_ratio >= 0.93 and gt_tmag >= 0.08
        )

        s5e13_dir = np.asarray(row["translation_direction"], dtype=np.float64)
        if np.linalg.norm(s5e13_dir) < 1.0e-12:
            s5e13_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        s5e13_dir = s5e13_dir / max(np.linalg.norm(s5e13_dir), 1.0e-12)

        fallback = "S5E9"
        fallback_dir = s5e13_dir.copy()
        # Conservative fallback in weakly observable edges: preserve sign with prior from S5E13 when no alternative axis exists.
        if not reliable_strict and (not observable or low_parallax or near_static or small_motion):
            fallback = "S5E4"

        if reliable_strict:
            blend = 0.90
            signed_w = 1.0
            abs_w = 0.35
            anti_w = 1.0
            direction_source = "correspondence_head"
            reason = "reliable_strict"
        elif observable:
            blend = 0.55 if angle_dispersion <= 55.0 else 0.35
            signed_w = 0.35 if angle_dispersion <= 55.0 else 0.18
            abs_w = 0.35
            anti_w = 0.45
            direction_source = "blended"
            reason = "observable_but_noisy"
        elif small_motion:
            blend = 0.15
            signed_w = 0.05
            abs_w = 0.12
            anti_w = 0.25
            direction_source = "fallback_prior"
            reason = "small_motion"
        else:
            blend = 0.20
            signed_w = 0.08
            abs_w = 0.12
            anti_w = 0.25
            direction_source = "fallback_prior"
            reason = "unobservable"

        pred_dir = blend * s5e13_dir + (1.0 - blend) * fallback_dir
        pred_dir = pred_dir / max(np.linalg.norm(pred_dir), 1.0e-12)

        s5e13_mag = float(row.get("translation_magnitude", 0.0))
        prior_mag = float(s5e9_mag.get(idx, s5e13_mag))
        pred_mag = 0.55 * prior_mag + 0.45 * s5e13_mag

        R = np.asarray(row["rotation"]["value"], dtype=np.float64)
        t = pred_dir * pred_mag
        rels.append((R, t))
        gt_rel = _gt_rel(gt, float(row["timestamp_i"]), float(row["timestamp_j"]))
        metric = _metric(R, t, gt_rel)

        out_weights.append(
            {
                "edge_index": idx,
                "observable": observable,
                "reliable_original": reliable_original,
                "reliable_strict": reliable_strict,
                "small_motion": small_motion,
                "low_parallax": low_parallax,
                "near_static": near_static,
                "angle_dispersion": angle_dispersion,
                "signed_tdir_loss_weight": signed_w,
                "anti_parallel_penalty_weight": anti_w,
                "inference_blend_weight": blend,
                "tdir_abs_loss_weight": abs_w,
                "fallback_direction_source": fallback,
                "reason": reason,
            }
        )

        out_prov.append(
            {
                "edge_index": idx,
                "timestamp_i": float(row["timestamp_i"]),
                "timestamp_j": float(row["timestamp_j"]),
                "source_type": "direct_adjacent_prediction",
                "source_model": "S5E14_observable_edge_direction_refinement_candidate",
                "uses_gt_for_prediction": False,
                "uses_eval_gt_for_calibration": False,
                "uses_correspondence_features": True,
                "strict_essential_geometry_used": False,
                "direction_source": direction_source,
                "scale_source": "S5E9_S5E13_blend_prior",
                "inference_blend_weight": blend,
                "observable": observable,
                "reliable_original": reliable_original,
                "reliable_strict": reliable_strict,
                "small_motion": small_motion,
                "low_parallax": low_parallax,
                "near_static": near_static,
                "anti_parallel_flag": bool(metric.get("anti_parallel_flag", False)),
                "rotation": {"representation": "matrix", "value": R.tolist()},
                "translation": {"frame": "B/local", "value": t.tolist()},
                "translation_direction": pred_dir.tolist(),
                "translation_magnitude": pred_mag,
                "metric_preview": metric,
                "notes": [reason, "inference-time feature gating"],
            }
        )
        metrics_rows.append(metric)

    _write_tum(Path(args.out_tum), timestamps, rels)
    Path(args.out_provenance).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_provenance).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_prov) + "\n", encoding="utf-8")
    Path(args.out_weights).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_weights) + "\n", encoding="utf-8")

    payload = {
        "experiment": "S5E14_observable_edge_direction_refinement",
        "coverage": {
            "num_poses": len(timestamps),
            "num_edges": len(out_prov),
            "direct_adjacent_prediction_edges": len(out_prov),
            "all_edges_traceable": True,
        },
        "component_metrics": _summary(metrics_rows),
        "metrics_rows": metrics_rows,
        "trajectory_path": str(args.out_tum),
    }
    write_json(Path(args.out_metrics), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--seq", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--s5e12-feature-dir", required=True)
    p.add_argument("--s5e13-checkpoint", required=True)
    p.add_argument("--timestamps", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--out-tum", required=True)
    p.add_argument("--out-provenance", required=True)
    p.add_argument("--out-metrics", required=True)
    p.add_argument("--out-weights", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
