#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from s5e2_adjacent_dense_lib import angle_deg_from_rot, load_or_base_checkpoint, pair_features, read_timestamps, read_tum, repo_rel, rot_to_quat_xyzw, scan_frames, validation_from_logs, vector_angle_deg, write_json
from s5e5_temporal_visual_lib import build_eval_seq_frames, build_numeric_features, load_npz_model, load_ordered_image_pair, load_training_checkpoint, predict_s5e2, rotvec_to_matrix


OUT_JSON = Path("checkpoints/S5E5_temporal_visual_backbone_geometry_candidate.json")
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


def _metric(R: np.ndarray, t: np.ndarray, gt_rel: Optional[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
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


def _write_tum(path: Path, timestamps: List[float], rels: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    Rw = np.eye(3, dtype=np.float64)
    tw = np.zeros(3, dtype=np.float64)
    rows = []
    qx, qy, qz, qw = rot_to_quat_xyzw(Rw)
    rows.append(f"{timestamps[0]:.6f} {tw[0]:.9f} {tw[1]:.9f} {tw[2]:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    for idx, (R_BA, t_BA) in enumerate(rels):
        Rb = Rw @ R_BA.T
        tb = tw - Rb @ t_BA
        Rw, tw = Rb, tb
        qx, qy, qz, qw = rot_to_quat_xyzw(Rw)
        rows.append(f"{timestamps[idx + 1]:.6f} {tw[0]:.9f} {tw[1]:.9f} {tw[2]:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _pct(values, q):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    return None if arr.size == 0 else float(np.percentile(arr, q))


def _mean(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    return None if arr.size == 0 else float(np.mean(arr))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    ckpt_path = Path(args.candidate) / "s5e5_temporal_visual_best.pt"
    device = torch.device("cpu")
    model, image_size, _numeric_dim = load_training_checkpoint(ckpt_path, device)
    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    timestamps = read_timestamps(Path(args.timestamps))
    gt = read_tum(Path(args.groundtruth))
    frames = build_eval_seq_frames(args.scene, args.seq)
    cache: Dict[str, np.ndarray] = {}
    provenance, metrics_rows, rels = [], [], []

    for edge_index in range(len(timestamps) - 1):
        ts_i, ts_j = timestamps[edge_index], timestamps[edge_index + 1]
        fi = _nearest(frames, ts_i)
        fj = _nearest(frames, ts_j)
        image = load_ordered_image_pair(fi, fj, image_size).unsqueeze(0).to(device)
        class S:
            pass
        sample = S()
        sample.frame_i = fi
        sample.frame_j = fj
        sample.edge_index = edge_index
        sample.total_edges = len(timestamps) - 1
        numeric = build_numeric_features(sample, cache, s5e2, s5e3)
        numeric_t = torch.from_numpy(numeric.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(image, numeric_t)
        prior_dir = numeric[:3]
        prior_dir = prior_dir / max(float(np.linalg.norm(prior_dir)), 1.0e-12)
        prior_logmag = float(numeric[3])
        x_pair = pair_features(fi, fj, edge_index, len(timestamps) - 1, cache)
        base_pred = predict_s5e2(s5e2, x_pair)
        sign_score = torch.tanh(out["sign_logit"]).cpu().item()
        dir_raw = out["dir_raw"].cpu().numpy().reshape(-1)
        final_dir_raw = dir_raw + prior_dir * sign_score
        final_dir = final_dir_raw / max(float(np.linalg.norm(final_dir_raw)), 1.0e-12)
        final_logmag = prior_logmag + float(out["logmag_residual"].cpu().item())
        final_mag = float(np.exp(final_logmag))
        R = rotvec_to_matrix(base_pred[:3])
        t = final_dir * final_mag
        rels.append((R, t))
        m = _metric(R, t, _gt_rel(gt, ts_i, ts_j))
        metrics_rows.append(m)
        provenance.append(
            {
                "edge_index": edge_index,
                "timestamp_i": ts_i,
                "timestamp_j": ts_j,
                "source_type": "direct_adjacent_prediction",
                "source_model": "S5E5_temporal_visual_backbone_geometry_candidate",
                "uses_gt_for_prediction": False,
                "rotation": {"representation": "matrix", "value": R.tolist(), "source": "inherited_s5e2_rotation"},
                "translation": {"frame": "B/local", "value": t.tolist()},
                "translation_direction": final_dir.tolist(),
                "translation_magnitude": final_mag,
                "direction_sign_score": sign_score,
                "anti_parallel_flag": m.get("anti_parallel_flag"),
                "visual_backbone_used": True,
                "metric_preview": m,
                "notes": ["ordered image pair temporal visual backbone", "hybrid visual+numeric prior", "no GT scale correction"],
            }
        )

    out_tum = Path(args.out_tum)
    out_prov = Path(args.out_provenance)
    out_metrics = Path(args.out_metrics)
    out_prov.parent.mkdir(parents=True, exist_ok=True)
    out_prov.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in provenance) + "\n", encoding="utf-8")
    _write_tum(out_tum, timestamps, rels)

    vals = {k: [m.get(k) for m in metrics_rows] for k in ["rot_deg", "tdir_deg", "tdir_abs_deg", "tdir_cosine", "tmag_ratio"]}
    pred_path = sum(float(m.get("pred_step_length") or 0.0) for m in metrics_rows)
    gt_path = sum(float(m.get("gt_step_length") or 0.0) for m in metrics_rows)
    metrics = {
        "experiment": "S5E5_temporal_visual_backbone_geometry_candidate",
        "available": True,
        "trajectory_path": repo_rel(out_tum),
        "edge_provenance": repo_rel(out_prov),
        "num_poses": len(timestamps),
        "num_edges": len(timestamps) - 1,
        "direct_adjacent_prediction_edges": len(provenance),
        "coverage": 1.0,
        "all_edges_traceable": True,
        "rot_mean_deg": _mean(vals["rot_deg"]),
        "rot_median_deg": _pct(vals["rot_deg"], 50),
        "rot_p90_deg": _pct(vals["rot_deg"], 90),
        "tdir_mean_deg": _mean(vals["tdir_deg"]),
        "tdir_median_deg": _pct(vals["tdir_deg"], 50),
        "tdir_p90_deg": _pct(vals["tdir_deg"], 90),
        "tdir_abs_mean_deg": _mean(vals["tdir_abs_deg"]),
        "tdir_abs_median_deg": _pct(vals["tdir_abs_deg"], 50),
        "tdir_abs_p90_deg": _pct(vals["tdir_abs_deg"], 90),
        "tdir_mean_cosine": _mean(vals["tdir_cosine"]),
        "anti_parallel_rate": _mean([1.0 if m.get("anti_parallel_flag") else 0.0 for m in metrics_rows]),
        "severe_wrong_sign_rate": _mean([1.0 if m.get("severe_wrong_sign_flag") else 0.0 for m in metrics_rows]),
        "direction_abs_good_but_signed_bad_rate": _mean([1.0 if m.get("direction_abs_good_but_signed_bad_flag") else 0.0 for m in metrics_rows]),
        "tmag_median_ratio": _pct(vals["tmag_ratio"], 50),
        "tmag_mean_ratio": _mean(vals["tmag_ratio"]),
        "tmag_p90_ratio": _pct(vals["tmag_ratio"], 90),
        "tmag_p95_ratio": _pct(vals["tmag_ratio"], 95),
        "tmag_max_ratio": max([x for x in vals["tmag_ratio"] if x is not None], default=None),
        "path_ratio": pred_path / max(gt_path, 1.0e-12),
    }
    write_json(out_metrics, metrics)
    ckpt = load_or_base_checkpoint(OUT_JSON)
    ckpt["adjacent_dense_export"] = {"available": True, "trajectory_path": repo_rel(out_tum), "edge_provenance": repo_rel(out_prov), "num_poses": len(timestamps), "num_edges": len(timestamps) - 1, "direct_adjacent_prediction_edges": len(provenance), "coverage": 1.0, "all_edges_traceable": True}
    ckpt["component_metrics"].update({k: metrics.get(k) for k in ckpt["component_metrics"] if k in metrics})
    ckpt["validation"] = validation_from_logs()
    ckpt["final_classification"] = "S5E5_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT"
    write_json(OUT_JSON, ckpt)
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Export S5E5 temporal visual candidate.")
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
