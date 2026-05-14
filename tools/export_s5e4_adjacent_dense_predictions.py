#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from s5e2_adjacent_dense_lib import angle_deg_from_rot, load_or_base_checkpoint, pair_features, read_timestamps, read_tum, repo_rel, rot_to_quat_xyzw, rotvec_to_matrix, scan_frames, validation_from_logs, vector_angle_deg, write_json


OUT_JSON = Path("checkpoints/S5E4_temporal_direction_head_candidate.json")
S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")


def _load(path: Path) -> Dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def _pred_s5e2(m: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    return ((x - m["x_mean"]) / m["x_std"]) @ m["W"] * m["y_std"] + m["y_mean"]


def _pred_head(m: Dict[str, np.ndarray], p: str, x: np.ndarray) -> np.ndarray:
    return ((x - m[f"{p}_x_mean"]) / m[f"{p}_x_std"]) @ m[f"{p}_W"] * m[f"{p}_y_std"] + m[f"{p}_y_mean"]


def _nearest(records: List[Any], ts: float, tol: float = 1e-5) -> Optional[Any]:
    best = None
    for r in records:
        dt = abs(r.timestamp - ts)
        if dt <= tol and (best is None or dt < best[0]):
            best = (dt, r)
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
    cos = None if tdir is None else float(np.cos(np.deg2rad(tdir)))
    tdir_abs = vector_angle_deg(t, tg, absolute=True)
    return {"rot_deg": angle_deg_from_rot(R @ Rg.T), "tdir_deg": tdir, "tdir_abs_deg": tdir_abs, "tdir_cosine": cos, "anti_parallel_flag": bool(cos is not None and cos < 0), "severe_wrong_sign_flag": bool(tdir is not None and tdir > 120), "direction_abs_good_but_signed_bad_flag": bool(tdir_abs is not None and tdir is not None and tdir_abs < 45 and tdir > 120), "tmag_ratio": pn / gn if gn > 1e-12 else None, "pred_step_length": pn, "gt_step_length": gn}


def _pct(v, q):
    arr = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=np.float64)
    return None if arr.size == 0 else float(np.percentile(arr, q))


def _mean(v):
    arr = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=np.float64)
    return None if arr.size == 0 else float(np.mean(arr))


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
    ts = read_timestamps(Path(args.timestamps))
    frames = scan_frames(Path("data"), scene=args.scene, seq=args.seq).get((args.scene, args.seq), [])
    gt = read_tum(Path(args.groundtruth))
    s5e2, s5e3 = _load(S5E2_BASE), _load(S5E3_HEADS)
    cache: Dict[str, np.ndarray] = {}
    rows, metrics_rows, rels = [], [], []
    for i in range(len(ts) - 1):
        ri, rj = _nearest(frames, ts[i]), _nearest(frames, ts[i + 1])
        x = pair_features(ri, rj, i, len(ts) - 1, cache)
        base = _pred_s5e2(s5e2, x)
        R = rotvec_to_matrix(base[:3])
        d = base[3:6]
        d = d / max(float(np.linalg.norm(d)), 1e-12)
        logmag = float(_pred_head(s5e3, "mag", x).reshape(-1)[0])
        mag = float(np.clip(np.exp(logmag), float(s5e3["mag_clip_lo"]), float(s5e3["mag_clip_hi"])))
        t = d * mag
        rels.append((R, t))
        m = _metric(R, t, _gt_rel(gt, ts[i], ts[i + 1]))
        metrics_rows.append(m)
        rows.append({"edge_index": i, "timestamp_i": ts[i], "timestamp_j": ts[i + 1], "source_type": "direct_adjacent_prediction", "source_model": "S5E4_temporal_direction_head_candidate", "uses_gt_for_prediction": False, "rotation": {"representation": "matrix", "value": R.tolist(), "source": "inherited_s5e2_rotation"}, "translation": {"frame": "B/local", "value": t.tolist()}, "translation_direction": d.tolist(), "translation_magnitude": mag, "direction_sign_score": float(np.linalg.norm(base[3:6])), "anti_parallel_flag": m.get("anti_parallel_flag"), "metric_preview": m, "notes": ["temporal sign from S5E2 signed direction prior", "magnitude from S5E3 log-magnitude head", "no GT scale correction"]})
    out_tum, out_prov, out_metrics = Path(args.out_tum), Path(args.out_provenance), Path(args.out_metrics)
    out_prov.parent.mkdir(parents=True, exist_ok=True)
    out_prov.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    _write_tum(out_tum, ts, rels)
    vals = {k: [m.get(k) for m in metrics_rows] for k in ["rot_deg", "tdir_deg", "tdir_abs_deg", "tdir_cosine", "tmag_ratio"]}
    pred_path = sum(float(m.get("pred_step_length") or 0) for m in metrics_rows)
    gt_path = sum(float(m.get("gt_step_length") or 0) for m in metrics_rows)
    metrics = {"experiment": "S5E4_temporal_direction_head_and_sign_disambiguation_candidate", "available": True, "trajectory_path": repo_rel(out_tum), "edge_provenance": repo_rel(out_prov), "num_poses": len(ts), "num_edges": len(ts) - 1, "direct_adjacent_prediction_edges": len(rows), "coverage": 1.0, "all_edges_traceable": True, "rot_mean_deg": _mean(vals["rot_deg"]), "rot_median_deg": _pct(vals["rot_deg"], 50), "rot_p90_deg": _pct(vals["rot_deg"], 90), "tdir_mean_deg": _mean(vals["tdir_deg"]), "tdir_median_deg": _pct(vals["tdir_deg"], 50), "tdir_p90_deg": _pct(vals["tdir_deg"], 90), "tdir_abs_mean_deg": _mean(vals["tdir_abs_deg"]), "tdir_abs_median_deg": _pct(vals["tdir_abs_deg"], 50), "tdir_abs_p90_deg": _pct(vals["tdir_abs_deg"], 90), "tdir_mean_cosine": _mean(vals["tdir_cosine"]), "anti_parallel_rate": _mean([1.0 if m.get("anti_parallel_flag") else 0.0 for m in metrics_rows]), "severe_wrong_sign_rate": _mean([1.0 if m.get("severe_wrong_sign_flag") else 0.0 for m in metrics_rows]), "direction_abs_good_but_signed_bad_rate": _mean([1.0 if m.get("direction_abs_good_but_signed_bad_flag") else 0.0 for m in metrics_rows]), "tmag_median_ratio": _pct(vals["tmag_ratio"], 50), "tmag_mean_ratio": _mean(vals["tmag_ratio"]), "tmag_p90_ratio": _pct(vals["tmag_ratio"], 90), "tmag_p95_ratio": _pct(vals["tmag_ratio"], 95), "tmag_max_ratio": max([x for x in vals["tmag_ratio"] if x is not None], default=None), "path_ratio": pred_path / max(gt_path, 1e-12)}
    write_json(out_metrics, metrics)
    ckpt = load_or_base_checkpoint(OUT_JSON)
    ckpt["adjacent_dense_export"] = {"available": True, "trajectory_path": repo_rel(out_tum), "edge_provenance": repo_rel(out_prov), "num_poses": len(ts), "num_edges": len(ts) - 1, "direct_adjacent_prediction_edges": len(rows), "coverage": 1.0, "all_edges_traceable": True}
    ckpt["component_metrics"].update({k: v for k, v in metrics.items() if k in ckpt["component_metrics"]})
    ckpt["validation"] = validation_from_logs()
    ckpt["final_classification"] = "S5E4_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT"
    write_json(OUT_JSON, ckpt)
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
