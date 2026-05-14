#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from s5e2_adjacent_dense_lib import (
    angle_deg_from_rot,
    load_or_base_checkpoint,
    pair_features,
    read_timestamps,
    read_tum,
    repo_rel,
    rot_to_quat_xyzw,
    rotvec_to_matrix,
    scan_frames,
    validation_from_logs,
    vector_angle_deg,
    write_json,
)


OUT_JSON = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate.json")
S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")


def _nearest(records: List[Any], timestamp: float, tol: float = 1.0e-5) -> Optional[Any]:
    best = None
    for rec in records:
        dt = abs(float(rec.timestamp) - float(timestamp))
        if dt <= tol and (best is None or dt < best[0]):
            best = (dt, rec)
    return best[1] if best else None


def _load_npz_model(path: Path) -> Dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def _predict_s5e2(model: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    return ((x - model["x_mean"]) / model["x_std"]) @ model["W"] * model["y_std"] + model["y_mean"]


def _predict_head(model: Dict[str, np.ndarray], prefix: str, x: np.ndarray) -> np.ndarray:
    return ((x - model[f"{prefix}_x_mean"]) / model[f"{prefix}_x_std"]) @ model[f"{prefix}_W"] * model[f"{prefix}_y_std"] + model[f"{prefix}_y_mean"]


def _relative_gt(gt: Dict[float, Dict[str, np.ndarray]], ts_i: float, ts_j: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if ts_i not in gt or ts_j not in gt:
        return None
    gi, gj = gt[ts_i], gt[ts_j]
    return gj["R"].T @ gi["R"], gj["R"].T @ (gi["t"] - gj["t"])


def _metric(R_pred: np.ndarray, t_pred: np.ndarray, gt_rel: Optional[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    if gt_rel is None:
        return {}
    R_gt, t_gt = gt_rel
    pn = float(np.linalg.norm(t_pred))
    gn = float(np.linalg.norm(t_gt))
    tdir = vector_angle_deg(t_pred, t_gt, absolute=False)
    return {
        "rot_deg": angle_deg_from_rot(R_pred @ R_gt.T),
        "tdir_deg": tdir,
        "tdir_abs_deg": vector_angle_deg(t_pred, t_gt, absolute=True),
        "tdir_cosine": None if tdir is None else float(np.cos(np.deg2rad(tdir))),
        "tmag_ratio": pn / gn if gn > 1.0e-12 else None,
        "pred_step_length": pn,
        "gt_step_length": gn,
    }


def _pct(vals: List[Optional[float]], q: float) -> Optional[float]:
    arr = np.asarray([float(v) for v in vals if v is not None and np.isfinite(float(v))], dtype=np.float64)
    return None if arr.size == 0 else float(np.percentile(arr, q))


def _mean(vals: List[Optional[float]]) -> Optional[float]:
    arr = np.asarray([float(v) for v in vals if v is not None and np.isfinite(float(v))], dtype=np.float64)
    return None if arr.size == 0 else float(np.mean(arr))


def _write_tum(path: Path, timestamps: List[float], rels: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    R_w = np.eye(3, dtype=np.float64)
    t_w = np.zeros(3, dtype=np.float64)
    rows = []
    qx, qy, qz, qw = rot_to_quat_xyzw(R_w)
    rows.append(f"{timestamps[0]:.6f} {t_w[0]:.9f} {t_w[1]:.9f} {t_w[2]:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    for idx, (R_BA, t_BA) in enumerate(rels):
        R_wB = R_w @ R_BA.T
        t_wB = t_w - R_wB @ t_BA
        R_w, t_w = R_wB, t_wB
        qx, qy, qz, qw = rot_to_quat_xyzw(R_w)
        rows.append(f"{timestamps[idx + 1]:.6f} {t_w[0]:.9f} {t_w[1]:.9f} {t_w[2]:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    candidate = Path(args.candidate) / "s5e3_scale_calibrated_heads.npz"
    out_tum = Path(args.out_tum)
    out_prov = Path(args.out_provenance)
    out_metrics = Path(args.out_metrics)
    out_prov.parent.mkdir(parents=True, exist_ok=True)
    timestamps = read_timestamps(Path(args.timestamps))
    frames = scan_frames(Path("data"), scene=args.scene, seq=args.seq).get((args.scene, args.seq), [])
    gt = read_tum(Path(args.groundtruth))

    if not candidate.exists() or not S5E2_BASE.exists():
        rows = []
        for i in range(len(timestamps) - 1):
            rows.append({"edge_index": i, "timestamp_i": timestamps[i], "timestamp_j": timestamps[i + 1], "source_type": "unavailable", "source_model": "S5E3_scale_calibrated_adjacent_dense_candidate", "uses_gt_for_prediction": False, "rotation": None, "translation": None, "notes": ["candidate missing"]})
        out_prov.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
        out_tum.write_text("# S5E3 export blocked: candidate missing.\n", encoding="utf-8")
        metrics = {"available": False, "coverage": 0.0}
        write_json(out_metrics, metrics)
        return metrics

    s5e2 = _load_npz_model(S5E2_BASE)
    heads = _load_npz_model(candidate)
    cache: Dict[str, np.ndarray] = {}
    provenance: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    rels: List[Tuple[np.ndarray, np.ndarray]] = []
    for edge_index in range(len(timestamps) - 1):
        ts_i, ts_j = timestamps[edge_index], timestamps[edge_index + 1]
        rec_i, rec_j = _nearest(frames, ts_i), _nearest(frames, ts_j)
        if rec_i is None or rec_j is None:
            provenance.append({"edge_index": edge_index, "timestamp_i": ts_i, "timestamp_j": ts_j, "source_type": "unavailable", "source_model": "S5E3_scale_calibrated_adjacent_dense_candidate", "uses_gt_for_prediction": False, "notes": ["image pair missing"]})
            continue
        x = pair_features(rec_i, rec_j, edge_index, len(timestamps) - 1, cache)
        base_pred = _predict_s5e2(s5e2, x)
        R_pred = rotvec_to_matrix(base_pred[:3])
        direction = _predict_head(heads, "dir", x).reshape(-1)
        direction = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
        logmag = float(_predict_head(heads, "mag", x).reshape(-1)[0])
        mag = float(np.exp(logmag))
        mag = float(np.clip(mag, float(heads["mag_clip_lo"]), float(heads["mag_clip_hi"])))
        t_pred = direction * mag
        rels.append((R_pred, t_pred))
        edge_metric = _metric(R_pred, t_pred, _relative_gt(gt, ts_i, ts_j))
        metric_rows.append({"edge_index": edge_index, **edge_metric})
        provenance.append(
            {
                "edge_index": edge_index,
                "timestamp_i": ts_i,
                "timestamp_j": ts_j,
                "source_type": "direct_adjacent_prediction",
                "source_model": "S5E3_scale_calibrated_adjacent_dense_candidate",
                "uses_gt_for_prediction": False,
                "rotation": {"representation": "matrix", "value": R_pred.tolist(), "source": "inherited_s5e2_rotation_head"},
                "translation": {"frame": "B/local", "value": t_pred.tolist()},
                "translation_direction": direction.tolist(),
                "translation_magnitude": mag,
                "metric_preview": edge_metric,
                "notes": ["scale-calibrated direct adjacent prediction", "no restored dense artifact", "no GT scale correction"],
            }
        )
    out_prov.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in provenance) + "\n", encoding="utf-8")
    _write_tum(out_tum, timestamps, rels)
    rot = [m.get("rot_deg") for m in metric_rows]
    tdir = [m.get("tdir_deg") for m in metric_rows]
    tdir_abs = [m.get("tdir_abs_deg") for m in metric_rows]
    tdir_cos = [m.get("tdir_cosine") for m in metric_rows]
    tmag = [m.get("tmag_ratio") for m in metric_rows]
    pred_path = sum(float(m.get("pred_step_length") or 0.0) for m in metric_rows)
    gt_path = sum(float(m.get("gt_step_length") or 0.0) for m in metric_rows)
    direct = sum(1 for x in provenance if x.get("source_type") == "direct_adjacent_prediction")
    metrics = {
        "experiment": "S5E3_scale_calibrated_adjacent_dense_candidate",
        "available": direct == len(timestamps) - 1,
        "trajectory_path": repo_rel(out_tum),
        "edge_provenance": repo_rel(out_prov),
        "num_poses": len(timestamps),
        "num_edges": len(timestamps) - 1,
        "direct_adjacent_prediction_edges": direct,
        "coverage": direct / max(len(timestamps) - 1, 1),
        "all_edges_traceable": direct == len(timestamps) - 1,
        "rot_mean_deg": _mean(rot),
        "rot_median_deg": _pct(rot, 50),
        "rot_p90_deg": _pct(rot, 90),
        "tdir_mean_deg": _mean(tdir),
        "tdir_median_deg": _pct(tdir, 50),
        "tdir_p90_deg": _pct(tdir, 90),
        "tdir_abs_mean_deg": _mean(tdir_abs),
        "tdir_abs_median_deg": _pct(tdir_abs, 50),
        "tdir_abs_p90_deg": _pct(tdir_abs, 90),
        "tdir_mean_cosine": _mean(tdir_cos),
        "tmag_median_ratio": _pct(tmag, 50),
        "tmag_mean_ratio": _mean(tmag),
        "tmag_p90_ratio": _pct(tmag, 90),
        "tmag_p95_ratio": _pct(tmag, 95),
        "tmag_max_ratio": max([x for x in tmag if x is not None], default=None),
        "path_ratio": pred_path / max(gt_path, 1.0e-12),
    }
    write_json(out_metrics, metrics)
    ckpt = load_or_base_checkpoint(OUT_JSON)
    ckpt["adjacent_dense_export"] = {"available": metrics["available"], "trajectory_path": repo_rel(out_tum), "edge_provenance": repo_rel(out_prov), "num_poses": metrics["num_poses"], "num_edges": metrics["num_edges"], "direct_adjacent_prediction_edges": direct, "coverage": metrics["coverage"], "all_edges_traceable": metrics["all_edges_traceable"]}
    for key in ["rot_mean_deg", "rot_median_deg", "rot_p90_deg", "tdir_mean_deg", "tdir_median_deg", "tdir_p90_deg", "tdir_abs_mean_deg", "tdir_abs_median_deg", "tdir_abs_p90_deg", "tdir_mean_cosine", "tmag_median_ratio", "tmag_mean_ratio", "tmag_p90_ratio", "tmag_p95_ratio", "path_ratio"]:
        ckpt["component_metrics"][key] = metrics.get(key)
    ckpt["validation"] = validation_from_logs()
    ckpt["final_classification"] = "S5E3_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT"
    write_json(OUT_JSON, ckpt)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export S5E3 scale-calibrated adjacent dense predictions.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--timestamps", required=True)
    parser.add_argument("--groundtruth", required=True)
    parser.add_argument("--out-tum", required=True)
    parser.add_argument("--out-provenance", required=True)
    parser.add_argument("--out-metrics", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
