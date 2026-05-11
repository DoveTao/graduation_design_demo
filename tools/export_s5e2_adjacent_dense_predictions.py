#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from s5e2_adjacent_dense_lib import (
    angle_deg_from_rot,
    adjacent_pairs,
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


DEFAULT_CKPT_JSON = Path("checkpoints/S5E2_adjacent_dense_candidate.json")


def _nearest_frame(records: List[Any], timestamp: float, tol: float = 1.0e-5) -> Optional[Any]:
    best = None
    for rec in records:
        dt = abs(float(rec.timestamp) - float(timestamp))
        if dt <= tol and (best is None or dt < best[0]):
            best = (dt, rec)
    return best[1] if best else None


def _load_model(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in ["x_mean", "x_std", "y_mean", "y_std", "W"]}


def _predict(model: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    xn = (x - model["x_mean"]) / model["x_std"]
    return (xn @ model["W"]) * model["y_std"] + model["y_mean"]


def _relative_gt(gt: Dict[float, Dict[str, np.ndarray]], ts_i: float, ts_j: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if ts_i not in gt or ts_j not in gt:
        return None
    gi = gt[ts_i]
    gj = gt[ts_j]
    R = gj["R"].T @ gi["R"]
    t = gj["R"].T @ (gi["t"] - gj["t"])
    return R, t


def _edge_metric(R_pred: np.ndarray, t_pred: np.ndarray, gt_rel: Optional[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    if gt_rel is None:
        return {}
    R_gt, t_gt = gt_rel
    pred_norm = float(np.linalg.norm(t_pred))
    gt_norm = float(np.linalg.norm(t_gt))
    return {
        "rot_deg": angle_deg_from_rot(R_pred @ R_gt.T),
        "tdir_deg": vector_angle_deg(t_pred, t_gt, absolute=False),
        "tdir_abs_deg": vector_angle_deg(t_pred, t_gt, absolute=True),
        "tmag_ratio": pred_norm / gt_norm if gt_norm > 1.0e-12 else None,
        "pred_step_length": pred_norm,
        "gt_step_length": gt_norm,
    }


def _percentile(values: List[float], q: float) -> Optional[float]:
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return None
    return float(np.percentile(vals, q))


def _mean(values: List[float]) -> Optional[float]:
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return None
    return float(np.mean(vals))


def _write_tum(path: Path, timestamps: List[float], rels: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    R_w = np.eye(3, dtype=np.float64)
    t_w = np.zeros(3, dtype=np.float64)
    rows: List[str] = []
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
    candidate_dir = Path(args.candidate)
    weights_path = candidate_dir / "s5e2_minimal_adjacent_pose_regressor.npz"
    timestamps = read_timestamps(Path(args.timestamps))
    out_prov = Path(args.out_provenance)
    out_tum = Path(args.out_tum)
    out_metrics = Path(args.out_metrics)
    out_prov.parent.mkdir(parents=True, exist_ok=True)

    frames = scan_frames(Path("data"), scene=args.scene, seq=args.seq).get((args.scene, args.seq), [])
    gt = read_tum(Path(args.groundtruth))
    cache: Dict[str, np.ndarray] = {}
    provenance: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []
    rels: List[Tuple[np.ndarray, np.ndarray]] = []

    if not weights_path.exists():
        for i in range(len(timestamps) - 1):
            provenance.append(
                {
                    "edge_index": i,
                    "timestamp_i": timestamps[i],
                    "timestamp_j": timestamps[i + 1],
                    "source_type": "unavailable",
                    "source_model": "S5E2_adjacent_dense_candidate",
                    "uses_gt_for_prediction": False,
                    "rotation": None,
                    "translation": None,
                    "notes": ["candidate weights missing; no prediction fabricated"],
                }
            )
        out_prov.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in provenance) + "\n", encoding="utf-8")
        out_tum.write_text("# S5E2 export blocked: candidate weights missing.\n", encoding="utf-8")
        metrics = {"available": False, "reason": "candidate weights missing", "source_counts": {"unavailable": len(provenance)}}
        write_json(out_metrics, metrics)
        return metrics

    model = _load_model(weights_path)
    all_direct = True
    for edge_index in range(len(timestamps) - 1):
        ts_i = timestamps[edge_index]
        ts_j = timestamps[edge_index + 1]
        rec_i = _nearest_frame(frames, ts_i)
        rec_j = _nearest_frame(frames, ts_j)
        if rec_i is None or rec_j is None:
            all_direct = False
            provenance.append(
                {
                    "edge_index": edge_index,
                    "timestamp_i": ts_i,
                    "timestamp_j": ts_j,
                    "source_type": "unavailable",
                    "source_model": "S5E2_adjacent_dense_candidate",
                    "uses_gt_for_prediction": False,
                    "rotation": None,
                    "translation": None,
                    "notes": ["image pair unavailable in data root; no restored dense artifact or GT fill used"],
                }
            )
            continue
        x = pair_features(rec_i, rec_j, edge_index, len(timestamps) - 1, cache)
        pred = _predict(model, x)
        R_pred = rotvec_to_matrix(pred[:3])
        t_pred = pred[3:6].astype(np.float64)
        rels.append((R_pred, t_pred))
        edge_metrics = _edge_metric(R_pred, t_pred, _relative_gt(gt, ts_i, ts_j))
        if edge_metrics:
            metrics_rows.append({"edge_index": edge_index, **edge_metrics})
        provenance.append(
            {
                "edge_index": edge_index,
                "timestamp_i": ts_i,
                "timestamp_j": ts_j,
                "source_type": "direct_adjacent_prediction",
                "source_model": "S5E2_adjacent_dense_candidate",
                "prediction_type": "minimal_image_statistics_adjacent_pose_regressor",
                "uses_gt_for_prediction": False,
                "rotation": {"representation": "matrix", "value": R_pred.tolist()},
                "translation": {"frame": "B/local", "value": t_pred.tolist(), "magnitude": float(np.linalg.norm(t_pred))},
                "metric_preview": edge_metrics,
                "notes": ["experimental direct adjacent prediction; no restored dense artifact; no GT scale correction"],
            }
        )

    out_prov.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in provenance) + "\n", encoding="utf-8")
    if all_direct and len(rels) == len(timestamps) - 1:
        _write_tum(out_tum, timestamps, rels)
    else:
        out_tum.write_text("# S5E2 export incomplete: unavailable edges present.\n", encoding="utf-8")

    rot = [m.get("rot_deg") for m in metrics_rows]
    tdir = [m.get("tdir_deg") for m in metrics_rows]
    tdir_abs = [m.get("tdir_abs_deg") for m in metrics_rows]
    tmag = [m.get("tmag_ratio") for m in metrics_rows]
    pred_path = sum(float(m.get("pred_step_length") or 0.0) for m in metrics_rows)
    gt_path = sum(float(m.get("gt_step_length") or 0.0) for m in metrics_rows)
    source_counts: Dict[str, int] = {}
    for row in provenance:
        source_counts[row["source_type"]] = source_counts.get(row["source_type"], 0) + 1
    metrics = {
        "experiment": "S5E2_real_adjacent_dense_candidate",
        "available": bool(all_direct and len(rels) == len(timestamps) - 1),
        "trajectory_path": repo_rel(out_tum),
        "edge_provenance": repo_rel(out_prov),
        "source_counts": source_counts,
        "num_poses": len(timestamps) if all_direct else 0,
        "num_edges": len(timestamps) - 1,
        "direct_adjacent_prediction_edges": source_counts.get("direct_adjacent_prediction", 0),
        "coverage": source_counts.get("direct_adjacent_prediction", 0) / max(len(timestamps) - 1, 1),
        "rot_mean_deg": _mean(rot),
        "rot_median_deg": _percentile(rot, 50),
        "rot_p90_deg": _percentile(rot, 90),
        "tdir_mean_deg": _mean(tdir),
        "tdir_median_deg": _percentile(tdir, 50),
        "tdir_p90_deg": _percentile(tdir, 90),
        "tdir_abs_mean_deg": _mean(tdir_abs),
        "tdir_abs_median_deg": _percentile(tdir_abs, 50),
        "tdir_abs_p90_deg": _percentile(tdir_abs, 90),
        "tmag_median_ratio": _percentile(tmag, 50),
        "tmag_mean_ratio": _mean(tmag),
        "tmag_p90_ratio": _percentile(tmag, 90),
        "tmag_p95_ratio": _percentile(tmag, 95),
        "tmag_max_ratio": max([x for x in tmag if x is not None], default=None),
        "path_ratio": pred_path / max(gt_path, 1.0e-12),
        "no_gt_used_for_prediction": True,
        "no_restored_dense_artifact_for_prediction": True,
    }
    write_json(out_metrics, metrics)

    ckpt = load_or_base_checkpoint(DEFAULT_CKPT_JSON)
    ckpt["adjacent_dense_export"] = {
        "available": metrics["available"],
        "trajectory_path": repo_rel(out_tum),
        "edge_provenance": repo_rel(out_prov),
        "num_poses": metrics["num_poses"],
        "num_edges": metrics["num_edges"],
        "direct_adjacent_prediction_edges": metrics["direct_adjacent_prediction_edges"],
        "coverage": metrics["coverage"],
        "all_edges_traceable": metrics["available"],
        "source_counts": source_counts,
    }
    ckpt["component_metrics"].update(
        {
            "rot_mean_deg": metrics["rot_mean_deg"],
            "tdir_mean_deg": metrics["tdir_mean_deg"],
            "tdir_abs_mean_deg": metrics["tdir_abs_mean_deg"],
            "tmag_median_ratio": metrics["tmag_median_ratio"],
            "tmag_p90_ratio": metrics["tmag_p90_ratio"],
            "path_ratio": metrics["path_ratio"],
        }
    )
    ckpt["validation"] = validation_from_logs()
    ckpt["final_classification"] = "S5E2_TRACEABLE_DENSE_EXPORTED_NO_IMPROVEMENT" if metrics["available"] else "S5E2_ADJACENT_DENSE_EXPORT_BLOCKED"
    write_json(DEFAULT_CKPT_JSON, ckpt)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export S5E2 direct adjacent dense predictions.")
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
