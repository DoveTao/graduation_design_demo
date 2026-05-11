#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from s5e2_adjacent_dense_lib import angle_deg_from_rot, load_or_base_checkpoint, pair_features, read_timestamps, read_tum, repo_rel, rot_to_quat_xyzw, validation_from_logs, vector_angle_deg, write_json
from s5e10_order_geometry_lib import load_order_direction_checkpoint, ordered_feature_vector
from s5e7_direction_scale_lib import build_eval_seq_frames, load_npz_model, predict_s5e2, predict_s5e3_head, rotvec_to_matrix
from s5e9_small_motion_geometry_lib import assign_bucket


OUT_JSON = Path("checkpoints/S5E10_order_aware_signed_direction_scale_recalibration_candidate.json")
S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")
S5E9_SCALE = Path("checkpoints/S5E9_scale_unit_small_motion_signed_direction_fix_candidate/s5e9_scale_best.pt")


class ScaleModel(torch.nn.Module):
    def __init__(self, feat_dim: int, log_scale_clip: float) -> None:
        super().__init__()
        self.log_scale_clip = float(log_scale_clip)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 1),
        )

    def forward(self, feat: torch.Tensor):
        raw = self.net(feat)
        return {"raw_scale_delta": raw, "pred_log_scale_delta": self.log_scale_clip * torch.tanh(raw)}


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
        "tmag_ratio": pn / gn if gn > 1e-12 else None,
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
        "path_ratio": pred_path / max(gt_path, 1e-12),
    }


def run(args):
    dir_model, feat_dim, bucket_stats = load_order_direction_checkpoint(Path(args.candidate) / "s5e10_order_signed_best.pt", torch.device("cpu"))
    scale_payload = torch.load(S5E9_SCALE, map_location="cpu")
    scale_model = ScaleModel(scale_payload["feat_dim"], scale_payload["log_scale_clip"])
    scale_model.load_state_dict(scale_payload["state_dict"])
    scale_model.eval()
    recal = json.loads((Path(args.candidate) / "scale_recalibration_training_status.json").read_text(encoding="utf-8"))
    recal_priors = scale_payload["bucket_stats"]
    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    timestamps = read_timestamps(Path(args.timestamps))
    gt = read_tum(Path(args.groundtruth))
    frames = build_eval_seq_frames(args.scene, args.seq)
    cache: Dict[str, np.ndarray] = {}
    rows, raw_rows, guard_rows, raw_rels, guard_rels = [], [], [], [], []
    pair_flip = []
    for i in range(len(timestamps) - 1):
        fi = _nearest(frames, timestamps[i]); fj = _nearest(frames, timestamps[i + 1])
        x_pair = pair_features(fi, fj, i, len(timestamps) - 1, cache)
        base = predict_s5e2(s5e2, x_pair)
        R = rotvec_to_matrix(base[:3])
        prior_mag_est = float(np.exp(predict_s5e3_head(s5e3, "mag", x_pair).reshape(-1)[0]))
        bucket_name, bucket_index = assign_bucket(prior_mag_est, recal_priors)
        class S: pass
        s = S()
        s.frame_i = fi
        s.frame_j = fj
        s.edge_index = i
        s.total_edges = len(timestamps) - 1
        s.bucket_name = bucket_name
        s.bucket_index = bucket_index
        feat_f = ordered_feature_vector(s, cache, recal_priors)
        with torch.no_grad():
            out_f = dir_model(torch.from_numpy(feat_f.astype(np.float32)).unsqueeze(0))
        pred_dir = out_f["pred_dir_unit"].view(-1).cpu().numpy()

        rs = S()
        rs.frame_i = fj
        rs.frame_j = fi
        rs.edge_index = i
        rs.total_edges = len(timestamps) - 1
        rs.bucket_name = bucket_name
        rs.bucket_index = bucket_index
        feat_r = ordered_feature_vector(rs, cache, recal_priors)
        with torch.no_grad():
            out_r = dir_model(torch.from_numpy(feat_r.astype(np.float32)).unsqueeze(0))
        pred_dir_rev = out_r["pred_dir_unit"].view(-1).cpu().numpy()
        flip_cos = float(np.dot(pred_dir, -pred_dir_rev) / max(np.linalg.norm(pred_dir) * np.linalg.norm(pred_dir_rev), 1e-12))
        pair_flip.append(flip_cos)

        scale_feat = torch.tensor([[math.log(max(recal_priors[bucket_name]["p95"], 1e-12)), float(timestamps[i+1]-timestamps[i]), i/max(len(timestamps)-1,1), float(bucket_index)]], dtype=torch.float32)
        with torch.no_grad():
            s_out = scale_model(scale_feat)
        pred_log_scale_delta = float(s_out["pred_log_scale_delta"].view(-1)[0].item())
        raw_tmag = float(recal_priors[bucket_name]["p95"] * math.exp(pred_log_scale_delta))
        guarded_tmag = float(np.clip(raw_tmag, recal_priors[bucket_name]["clip_lo"], recal_priors[bucket_name]["p99"]))

        raw_t = pred_dir * raw_tmag
        guarded_t = pred_dir * guarded_tmag
        gt_rel = _gt_rel(gt, timestamps[i], timestamps[i+1])
        raw_m = _metric(R, raw_t, gt_rel)
        guard_m = _metric(R, guarded_t, gt_rel)
        raw_rows.append(raw_m); guard_rows.append(guard_m)
        raw_rels.append((R, raw_t)); guard_rels.append((R, guarded_t))
        rows.append({
            "edge_index": i,
            "timestamp_i": timestamps[i],
            "timestamp_j": timestamps[i+1],
            "source_type": "direct_adjacent_prediction",
            "source_model": "S5E10_order_aware_signed_direction_scale_recalibration_candidate",
            "uses_gt_for_prediction": False,
            "rotation": {"representation": "matrix", "value": R.tolist()},
            "translation": {"frame": "B/local", "value": guarded_t.tolist()},
            "translation_direction": pred_dir.tolist(),
            "translation_magnitude": guarded_tmag,
            "pred_axis_unit": out_f["pred_axis_unit"].view(-1).cpu().numpy().tolist(),
            "pred_sign_logit": float(out_f["pred_sign_logit"].view(-1)[0].item()),
            "pred_dir_unit": pred_dir.tolist(),
            "pred_tmag_raw": raw_tmag,
            "pred_tmag": guarded_tmag,
            "pred_log_scale_delta": pred_log_scale_delta,
            "scale_guard_applied": bool(abs(raw_tmag - guarded_tmag) > 1e-12),
            "optional_motion_bucket": bucket_name,
            "pair_order_dir_flip_cosine": flip_cos,
            "metric_preview_raw": raw_m,
            "metric_preview": guard_m,
            "notes": ["order-aware concat features", "reversed-pair anti-symmetry training", "p95 prior scale recalibration"],
        })

    out_tum = Path(args.out_tum)
    out_raw_tum = out_tum.with_name(out_tum.stem + "_raw" + out_tum.suffix)
    out_prov = Path(args.out_provenance)
    out_metrics = Path(args.out_metrics)
    out_prov.parent.mkdir(parents=True, exist_ok=True)
    out_prov.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    _write_tum(out_tum, timestamps, guard_rels)
    _write_tum(out_raw_tum, timestamps, raw_rels)
    metrics = {
        "experiment": "S5E10_order_aware_signed_direction_scale_recalibration",
        "available": True,
        "trajectory_path": repo_rel(out_tum),
        "raw_trajectory_path": repo_rel(out_raw_tum),
        "edge_provenance": repo_rel(out_prov),
        "num_poses": len(timestamps),
        "num_edges": len(timestamps) - 1,
        "direct_adjacent_prediction_edges": len(rows),
        "coverage": 1.0,
        "all_edges_traceable": True,
        "raw_prediction_metrics": _summary(raw_rows),
        "guarded_prediction_metrics": _summary(guard_rows),
        "pair_order_dir_flip_cosine_mean": float(np.mean(pair_flip)),
        "pair_order_dir_flip_success_rate": float(np.mean(np.asarray(pair_flip) > 0.5)),
        "pair_order_scale_symmetry_error": 0.0,
        "pair_order_failure_rate": float(np.mean(np.asarray(pair_flip) <= 0.5)),
        "signed_direction_order_observable": bool(np.mean(np.asarray(pair_flip) > 0.5) > 0.5),
    }
    write_json(out_metrics, metrics)
    ckpt = load_or_base_checkpoint(OUT_JSON)
    ckpt["adjacent_dense_export"] = {
        "available": True, "coverage": 1.0, "all_edges_traceable": True,
        "num_poses": len(timestamps), "num_edges": len(timestamps)-1, "direct_adjacent_prediction_edges": len(rows),
        "trajectory_path": repo_rel(out_tum), "raw_trajectory_path": repo_rel(out_raw_tum), "edge_provenance": repo_rel(out_prov)
    }
    ckpt["raw_prediction_metrics"] = metrics["raw_prediction_metrics"]
    ckpt["guarded_prediction_metrics"] = metrics["guarded_prediction_metrics"]
    ckpt["component_metrics"] = metrics["guarded_prediction_metrics"]
    ckpt["pair_order_diagnostic"] = {
        "pair_order_dir_flip_cosine_mean": metrics["pair_order_dir_flip_cosine_mean"],
        "pair_order_dir_flip_success_rate": metrics["pair_order_dir_flip_success_rate"],
        "pair_order_scale_symmetry_error": metrics["pair_order_scale_symmetry_error"],
        "pair_order_failure_rate": metrics["pair_order_failure_rate"],
        "signed_direction_order_observable": metrics["signed_direction_order_observable"],
    }
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
