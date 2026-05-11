#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from s5e12_extract_real_correspondence_features import extract_orb_matches, extract_sparse_flow, load_gray
from s5e2_adjacent_dense_lib import angle_deg_from_rot, pair_features, read_timestamps, read_tum, rot_to_quat_xyzw, scan_frames, vector_angle_deg, write_json
from s5e7_direction_scale_lib import load_npz_model, predict_s5e2, predict_s5e3_head
from train_s5e19c_real_direction_model import DirectionRefineMLP


S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")
S5E15_PROV = Path("external_baselines/results/s5e15_traceable_dense/edge_provenance.jsonl")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _norm(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1.0e-12)


def _extract_corr(frame_i: Any, frame_j: Any) -> List[float]:
    img_i = load_gray(frame_i.image_path, (640, 320))
    img_j = load_gray(frame_j.image_path, (640, 320))
    orb = extract_orb_matches(img_i, img_j, 1200, 12, 0.75, False)
    flow = extract_sparse_flow(img_i, img_j, 600, 0.01, 7.0, 21, 3)
    return [
        float(orb["filtered_match_count"]),
        float(orb["inlier_ratio"]),
        float(orb["median_match_displacement"]),
        float(orb["p90_match_displacement"]),
        float(orb["parallax_proxy"]),
        1.0 if orb["low_parallax_flag"] else 0.0,
        float(flow["median_flow_magnitude"]),
        float(flow["p90_flow_magnitude"]),
        float(flow["flow_angle_dispersion"]) / 180.0,
        0.0 if flow["forward_backward_flow_consistency"] is None else float(flow["forward_backward_flow_consistency"]),
    ]


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
    def _mean(k: str):
        v = _vals(k)
        return None if v.size == 0 else float(np.mean(v))
    def _pct(k: str, q: float):
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
        "tmag_p95_ratio": _pct("tmag_ratio", 95),
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
    payload = torch.load(Path(args.candidate) / "s5e19c_model.pt", map_location="cpu")
    model = DirectionRefineMLP(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        delta_tdir_scale=float(payload["delta_tdir_scale"]),
        delta_log_tmag_clip=float(payload["delta_log_tmag_clip"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    s5e15_by = {int(r["edge_index"]): r for r in _read_jsonl(S5E15_PROV)}
    gt = read_tum(Path(args.groundtruth))
    timestamps = read_timestamps(Path(args.timestamps))
    frames = scan_frames(Path("data"), scene=args.scene, seq=args.seq)[(args.scene, args.seq)]
    frame_by_ts = {round(float(fr.timestamp), 6): fr for fr in frames}
    cache: Dict[str, np.ndarray] = {}

    rels: List[Tuple[np.ndarray, np.ndarray]] = []
    prov_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    for i in range(len(timestamps) - 1):
        fi = frame_by_ts[round(timestamps[i], 6)]
        fj = frame_by_ts[round(timestamps[i + 1], 6)]
        x = pair_features(fi, fj, i, len(timestamps) - 1, cache)
        coarse = predict_s5e2(s5e2, x)
        coarse_dir = _norm(coarse[3:6])
        coarse_mag = float(s5e15_by[i]["translation_magnitude"])
        corr = _extract_corr(fi, fj)
        feat = np.concatenate([x.astype(np.float32), coarse_dir.astype(np.float32), np.asarray([coarse_mag, math.log(max(coarse_mag, 1.0e-12))], dtype=np.float32), np.asarray(corr, dtype=np.float32)])
        feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        coarse_dir_t = F.normalize(torch.tensor(coarse_dir, dtype=torch.float32).unsqueeze(0), dim=1)
        coarse_mag_t = torch.tensor([coarse_mag], dtype=torch.float32)
        with torch.no_grad():
            pred = model(feat_t, coarse_dir_t, coarse_mag_t)
        delta_tdir = pred["delta_tdir"].squeeze(0).cpu().numpy()
        final_tdir = pred["final_tdir"].squeeze(0).cpu().numpy()
        sign_score = float(pred["sign_score"].item())
        delta_log_tmag = float(pred["delta_log_tmag"].item())
        final_tmag = float(pred["final_tmag"].item())
        R = np.asarray(s5e15_by[i]["rotation"]["value"], dtype=np.float64)
        t = final_tdir * final_tmag
        rels.append((R, t))
        mm = _metric(R, t, _gt_rel(gt, timestamps[i], timestamps[i + 1]))
        metric_rows.append(mm)
        prov_rows.append(
            {
                "edge_index": i,
                "source_model": "S5E19C_real_direction_training_no_fallback",
                "uses_s5e15_direction_fallback": False,
                "uses_s5e15_scale_prior_reference": True,
                "tdir_coarse": coarse_dir.tolist(),
                "delta_tdir": delta_tdir.tolist(),
                "delta_tdir_norm": float(np.linalg.norm(delta_tdir)),
                "final_tdir": final_tdir.tolist(),
                "sign_score": sign_score,
                "delta_log_tmag": delta_log_tmag,
                "final_tmag": final_tmag,
                "uses_eval_gt_for_prediction": False,
                "rotation": {"representation": "matrix", "value": R.tolist()},
                "translation": {"frame": "B/local", "value": t.tolist()},
                "metric_preview": mm,
            }
        )

    _write_tum(Path(args.out_tum), timestamps, rels)
    Path(args.out_provenance).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_provenance).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in prov_rows) + "\n", encoding="utf-8")
    payload_out = {
        "experiment": "S5E19C_real_direction_training_no_fallback",
        "coverage": {"num_poses": len(timestamps), "num_edges": len(prov_rows), "direct_adjacent_prediction_edges": len(prov_rows), "all_edges_traceable": True},
        "component_metrics": _summary(metric_rows),
        "metrics_rows": metric_rows,
        "trajectory_path": str(args.out_tum),
    }
    write_json(Path(args.out_metrics), payload_out)
    return payload_out


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
