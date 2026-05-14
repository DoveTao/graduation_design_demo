#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from s5e2_adjacent_dense_lib import angle_deg_from_rot, load_or_base_checkpoint, pair_features, read_timestamps, read_tum, repo_rel, rot_to_quat_xyzw, validation_from_logs, write_json
from s5e8_translation_geometry_lib import (
    build_eval_seq_frames,
    build_numeric_features,
    compose_direction,
    load_direction_checkpoint,
    load_npz_model,
    load_ordered_image_pair,
    predict_s5e2,
    rotvec_to_matrix,
)


OUT_JSON = Path("checkpoints/S5E8_translation_geometry_diagnostic_ablation_candidate.json")
S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")


def _nearest(records: List[Any], ts: float, tol: float = 1e-5) -> Optional[Any]:
    best = None
    for rec in records:
        dt = abs(float(rec.timestamp) - float(ts))
        if dt <= tol and (best is None or dt < best[0]):
            best = (dt, rec)
    return best[1] if best else None


def _gt_rel(gt: Dict[float, Dict[str, np.ndarray]], a: float, b: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if a not in gt or b not in gt:
        return None
    gi, gj = gt[a], gt[b]
    return gj["R"].T @ gi["R"], gj["R"].T @ (gi["t"] - gj["t"])


def _vector_angle(a: np.ndarray, b: np.ndarray, absolute: bool) -> Optional[float]:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < 1.0e-12 or bn < 1.0e-12:
        return None
    dot = float(np.dot(a, b) / (an * bn))
    if absolute:
        dot = abs(dot)
    dot = max(-1.0, min(1.0, dot))
    return float(np.degrees(np.arccos(dot)))


def _metric(R: np.ndarray, t: np.ndarray, gt_rel: Optional[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    if gt_rel is None:
        return {}
    Rg, tg = gt_rel
    pn, gn = float(np.linalg.norm(t)), float(np.linalg.norm(tg))
    tdir = _vector_angle(t, tg, absolute=False)
    tdir_abs = _vector_angle(t, tg, absolute=True)
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


def run(args: argparse.Namespace) -> Dict[str, Any]:
    direction_ckpt = Path(args.candidate) / "s5e8_direction_only_best.pt"
    model, _image_size, _numeric_dim, bucket_stats, _model_type = load_direction_checkpoint(direction_ckpt, torch.device("cpu"))
    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    timestamps = read_timestamps(Path(args.timestamps))
    gt = read_tum(Path(args.groundtruth))
    frames = build_eval_seq_frames(args.scene, args.seq)
    cache: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    raw_rels: List[Tuple[np.ndarray, np.ndarray]] = []
    guarded_rels: List[Tuple[np.ndarray, np.ndarray]] = []
    raw_metrics_rows: List[Dict[str, Any]] = []
    guarded_metrics_rows: List[Dict[str, Any]] = []

    for i in range(len(timestamps) - 1):
        fi = _nearest(frames, timestamps[i]); fj = _nearest(frames, timestamps[i + 1])
        class S: pass
        s = S()
        s.frame_i = fi
        s.frame_j = fj
        s.edge_index = i
        s.total_edges = len(timestamps) - 1
        s.bucket_name = "normal_motion"
        s.bucket_index = 1
        x_pair = pair_features(fi, fj, i, len(timestamps) - 1, cache)
        base = predict_s5e2(s5e2, x_pair)
        R = rotvec_to_matrix(base[:3])
        prior_dir = base[3:6]
        prior_dir = prior_dir / max(float(np.linalg.norm(prior_dir)), 1.0e-12)
        numeric = build_numeric_features(s, cache, s5e2, s5e3, bucket_stats)
        image = load_ordered_image_pair(fi, fj, _image_size).unsqueeze(0)
        numeric_t = torch.from_numpy(numeric.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = model(image, numeric_t)
            pred_dir = compose_direction(torch.from_numpy(prior_dir.astype(np.float32)).unsqueeze(0), out).view(-1).cpu().numpy()
            bucket_probs = torch.softmax(out["motion_bucket_logits"], dim=1).view(-1).cpu().numpy()
        bucket_idx = int(np.argmax(bucket_probs))
        bucket_name = ["small_motion", "normal_motion", "large_motion"][bucket_idx]
        raw_tmag = float(np.exp(numeric[3]))
        guarded_tmag = float(np.clip(raw_tmag, bucket_stats[bucket_name]["clip_lo"], bucket_stats[bucket_name]["clip_hi"]))
        raw_t = pred_dir * raw_tmag
        guarded_t = pred_dir * guarded_tmag
        gt_rel = _gt_rel(gt, timestamps[i], timestamps[i + 1])
        raw_m = _metric(R, raw_t, gt_rel)
        guarded_m = _metric(R, guarded_t, gt_rel)
        raw_metrics_rows.append(raw_m)
        guarded_metrics_rows.append(guarded_m)
        raw_rels.append((R, raw_t))
        guarded_rels.append((R, guarded_t))
        rows.append({
            "edge_index": i,
            "timestamp_i": timestamps[i],
            "timestamp_j": timestamps[i + 1],
            "source_type": "direct_adjacent_prediction",
            "source_model": "S5E8_translation_geometry_diagnostic_ablation_candidate",
            "uses_gt_for_prediction": False,
            "rotation": {"representation": "matrix", "value": R.tolist()},
            "translation": {"frame": "B/local", "value": guarded_t.tolist()},
            "translation_direction": pred_dir.tolist(),
            "translation_magnitude": guarded_tmag,
            "pred_dir_unit": pred_dir.tolist(),
            "pred_log_scale_delta": None,
            "pred_tmag": guarded_tmag,
            "pred_t": guarded_t.tolist(),
            "optional_confidence": float(out["confidence"].view(-1)[0].item()),
            "optional_motion_bucket": bucket_name,
            "motion_bucket_probs": bucket_probs.tolist(),
            "raw_tmag_from_prior": raw_tmag,
            "guarded_tmag": guarded_tmag,
            "scale_guard_applied": bool(abs(raw_tmag - guarded_tmag) > 1.0e-12),
            "direction_source": "S5E8_direction_only_candidate",
            "anti_parallel_flag": guarded_m.get("anti_parallel_flag"),
            "metric_preview_raw": raw_m,
            "metric_preview": guarded_m,
            "notes": ["raw 使用 numeric magnitude prior", "guarded 使用 train bucket clip", "direction 由 direction-only candidate 提供"],
        })

    out_tum = Path(args.out_tum)
    out_raw_tum = out_tum.with_name(out_tum.stem + "_raw" + out_tum.suffix)
    out_prov = Path(args.out_provenance)
    out_metrics = Path(args.out_metrics)
    out_prov.parent.mkdir(parents=True, exist_ok=True)
    out_prov.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    _write_tum(out_tum, timestamps, guarded_rels)
    _write_tum(out_raw_tum, timestamps, raw_rels)
    metrics = {
        "experiment": "S5E8_translation_geometry_diagnostic_ablation",
        "available": True,
        "trajectory_path": repo_rel(out_tum),
        "raw_trajectory_path": repo_rel(out_raw_tum),
        "edge_provenance": repo_rel(out_prov),
        "num_poses": len(timestamps),
        "num_edges": len(timestamps) - 1,
        "direct_adjacent_prediction_edges": len(rows),
        "coverage": 1.0,
        "all_edges_traceable": True,
        "raw_prediction_metrics": _summary(raw_metrics_rows),
        "guarded_prediction_metrics": _summary(guarded_metrics_rows),
    }
    write_json(out_metrics, metrics)
    ckpt = load_or_base_checkpoint(OUT_JSON)
    ckpt["adjacent_dense_export"] = {
        "available": True,
        "coverage": 1.0,
        "all_edges_traceable": True,
        "num_poses": len(timestamps),
        "num_edges": len(timestamps) - 1,
        "direct_adjacent_prediction_edges": len(rows),
        "trajectory_path": repo_rel(out_tum),
        "raw_trajectory_path": repo_rel(out_raw_tum),
        "edge_provenance": repo_rel(out_prov),
    }
    ckpt["raw_prediction_metrics"] = metrics["raw_prediction_metrics"]
    ckpt["guarded_prediction_metrics"] = metrics["guarded_prediction_metrics"]
    ckpt["component_metrics"] = metrics["guarded_prediction_metrics"]
    ckpt["validation"] = validation_from_logs()
    write_json(OUT_JSON, ckpt)
    return metrics


def parse_args():
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
