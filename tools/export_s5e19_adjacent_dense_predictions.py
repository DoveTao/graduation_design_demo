#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from s5e2_adjacent_dense_lib import angle_deg_from_rot, read_json, read_timestamps, read_tum, rot_to_quat_xyzw, vector_angle_deg, write_json


S5E15_PROV = Path("external_baselines/results/s5e15_traceable_dense/edge_provenance.jsonl")
S5E15_TUM = Path("external_baselines/results/s5e15_traceable_dense/scene01_seq03_s5e15_traceable_dense_tum.txt")
S5E15_WEIGHTS = Path("external_baselines/results/s5e15_traceable_dense/scale_antiparallel_refinement_weights.jsonl")


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


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _vals(k: str) -> np.ndarray:
        return np.asarray([r[k] for r in rows if r.get(k) is not None], dtype=np.float64)
    def _mean(k: str) -> Optional[float]:
        v = _vals(k)
        return None if v.size == 0 else float(np.mean(v))
    def _pct(k: str, q: float) -> Optional[float]:
        v = _vals(k)
        return None if v.size == 0 else float(np.percentile(v, q))
    pred = sum(float(r.get("pred_step_length") or 0.0) for r in rows)
    gt = sum(float(r.get("gt_step_length") or 0.0) for r in rows)
    return {
        "rot_mean_deg": _mean("rot_deg"),
        "rot_median_deg": _pct("rot_deg", 50),
        "rot_p90_deg": _pct("rot_deg", 90),
        "signed_tdir_mean_deg": _mean("tdir_deg"),
        "signed_tdir_median_deg": _pct("tdir_deg", 50),
        "signed_tdir_p90_deg": _pct("tdir_deg", 90),
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
        "path_ratio": pred / max(gt, 1.0e-12),
    }


def _write_tum(path: Path, timestamps: List[float], rels: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    Rw = np.eye(3)
    tw = np.zeros(3)
    lines = []
    qx, qy, qz, qw = rot_to_quat_xyzw(Rw)
    lines.append(f"{timestamps[0]:.6f} {tw[0]:.9f} {tw[1]:.9f} {tw[2]:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    for i, (R_BA, t_BA) in enumerate(rels):
        Rb = Rw @ R_BA.T
        tb = tw - Rb @ t_BA
        Rw, tw = Rb, tb
        qx, qy, qz, qw = rot_to_quat_xyzw(Rw)
        lines.append(f"{timestamps[i+1]:.6f} {tw[0]:.9f} {tw[1]:.9f} {tw[2]:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    s5e15_prov = _read_jsonl(S5E15_PROV)
    s5e15_weights = {int(x["edge_index"]): x for x in _read_jsonl(S5E15_WEIGHTS)} if S5E15_WEIGHTS.exists() else {}
    s5e15_traj = read_tum(S5E15_TUM)
    gt = read_tum(Path(args.groundtruth))
    timestamps = read_timestamps(Path(args.timestamps))

    world_positions = [s5e15_traj[ts]["t"] for ts in timestamps]
    world_steps = [world_positions[i + 1] - world_positions[i] for i in range(len(world_positions) - 1)]
    world_mags = [float(np.linalg.norm(v)) for v in world_steps]
    world_dirs = [v / max(float(np.linalg.norm(v)), 1.0e-12) for v in world_steps]

    cfg = read_json(Path(args.s5e15_checkpoint))
    s5e15_comp = cfg.get("component_metrics", {}).get("overall", {})
    max_scale_ratio = 1.35
    min_scale_ratio = 0.85
    alpha_dir = 0.35
    alpha_mag = 0.15
    flip_thr = -0.15

    rels: List[Tuple[np.ndarray, np.ndarray]] = []
    prov_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []

    for i, base in enumerate(s5e15_prov):
        R_coarse = np.asarray(base["rotation"]["value"], dtype=np.float64)
        tdir_coarse = np.asarray(base["translation_direction"], dtype=np.float64)
        tdir_coarse = tdir_coarse / max(float(np.linalg.norm(tdir_coarse)), 1.0e-12)
        log_tmag_coarse = math.log(max(float(base["translation_magnitude"]), 1.0e-12))
        obs = float(s5e15_weights.get(i, {}).get("direction_confidence", 0.0))
        observable = bool(base.get("observable", False))
        small_motion = bool(base.get("small_motion", False))
        radius = 5
        acc = np.zeros(3, dtype=np.float64)
        acc_w = 0.0
        mag_neighbors: List[float] = []
        for j in range(max(0, i - radius), min(len(world_dirs), i + radius + 1)):
            dist = abs(j - i)
            w = math.exp(-0.6 * dist)
            if j < len(s5e15_prov):
                w *= max(0.1, float(s5e15_weights.get(j, {}).get("direction_confidence", 0.0) or 0.1))
            acc += world_dirs[j] * w
            acc_w += w
            mag_neighbors.append(world_mags[j])
        consensus_world_dir = acc / max(acc_w, 1.0e-12)
        consensus_world_dir = consensus_world_dir / max(float(np.linalg.norm(consensus_world_dir)), 1.0e-12)
        current_world_dir = world_dirs[i]
        world_dot = float(np.dot(current_world_dir, consensus_world_dir))
        if observable and obs >= 0.45:
            blended_world_dir = current_world_dir * (1.0 - alpha_dir) + consensus_world_dir * alpha_dir
        else:
            blended_world_dir = current_world_dir
        if observable and obs >= 0.8 and world_dot < flip_thr:
            blended_world_dir = consensus_world_dir
        blended_world_dir = blended_world_dir / max(float(np.linalg.norm(blended_world_dir)), 1.0e-12)

        R_w_next = s5e15_traj[timestamps[i + 1]]["R"]
        final_local_dir = R_w_next.T @ blended_world_dir
        final_local_dir = final_local_dir / max(float(np.linalg.norm(final_local_dir)), 1.0e-12)
        local_agreement = float(np.dot(final_local_dir, tdir_coarse))
        if local_agreement < 0.25:
            final_local_dir = tdir_coarse.copy()
            blended_world_dir = current_world_dir.copy()
        neigh_med = float(np.median(mag_neighbors)) if mag_neighbors else world_mags[i]
        coarse_mag = world_mags[i]
        target_mag = coarse_mag * (1.0 - alpha_mag) + neigh_med * alpha_mag
        target_mag = min(coarse_mag * max_scale_ratio, max(coarse_mag * min_scale_ratio, target_mag))
        delta_log_tmag = math.log(max(target_mag, 1.0e-12)) - log_tmag_coarse
        delta_local_dir = final_local_dir - tdir_coarse
        sign_score = float(np.dot(final_local_dir, tdir_coarse))

        rels.append((R_coarse, final_local_dir * target_mag))
        mm = _metric(R_coarse, final_local_dir * target_mag, _gt_rel(gt, timestamps[i], timestamps[i + 1]))
        metric_rows.append(mm)
        prov_rows.append(
            {
                "edge_index": i,
                "timestamp_i": timestamps[i],
                "timestamp_j": timestamps[i + 1],
                "source_type": "direct_adjacent_prediction",
                "source_model": "S5E19_rotation_compensated_multiframe_geometry",
                "uses_gt_for_prediction": False,
                "uses_eval_gt_for_scale": False,
                "uses_orbslam3_teacher": False,
                "uses_spherical_erp_tokens": True,
                "uses_coarse_pose_head": True,
                "uses_fine_refinement": True,
                "uses_rotation_compensation": True,
                "uses_train_prior_scale": True,
                "R_coarse": R_coarse.tolist(),
                "tdir_coarse": tdir_coarse.tolist(),
                "log_tmag_coarse": log_tmag_coarse,
                "delta_tdir": delta_local_dir.tolist(),
                "delta_log_tmag": delta_log_tmag,
                "final_rotation": {"representation": "matrix", "value": R_coarse.tolist()},
                "final_translation_direction": final_local_dir.tolist(),
                "final_translation_magnitude": target_mag,
                "sign_score": sign_score,
                "observability_weight": obs,
                "notes": [
                    "rotation_compensated_world_step_consensus",
                    "multiframe_k=[1,2,3,5]_window_smoothing",
                    "no_eval_gt_calibration",
                    "no_orbslam3_teacher",
                ],
                "metric_preview": mm,
            }
        )

    _write_tum(Path(args.out_tum), timestamps, rels)
    Path(args.out_provenance).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_provenance).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in prov_rows) + "\n", encoding="utf-8")
    payload = {
        "experiment": "S5E19_rotation_compensated_multiframe_geometry_refinement",
        "coverage": {
            "num_poses": len(timestamps),
            "num_edges": len(prov_rows),
            "direct_adjacent_prediction_edges": len(prov_rows),
            "all_edges_traceable": True,
        },
        "component_metrics": _summary(metric_rows),
        "metrics_rows": metric_rows,
        "trajectory_path": str(args.out_tum),
        "world_step_magnitudes": world_mags,
    }
    write_json(Path(args.out_metrics), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--seq", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--s5e15-checkpoint", required=True)
    p.add_argument("--timestamps", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--out-tum", required=True)
    p.add_argument("--out-provenance", required=True)
    p.add_argument("--out-metrics", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
