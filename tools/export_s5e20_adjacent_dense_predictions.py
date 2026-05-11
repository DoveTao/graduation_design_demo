#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from s5e12_extract_real_correspondence_features import extract_orb_matches, extract_sparse_flow, load_gray
from s5e2_adjacent_dense_lib import angle_deg_from_rot, read_timestamps, read_tum, rot_to_quat_xyzw, scan_frames, vector_angle_deg, write_json
from s5e7_direction_scale_lib import load_npz_model, predict_s5e2, predict_s5e3_head
from s5e2_adjacent_dense_lib import pair_features
from train_s5e20_true_kstep_composition import ConservativeDirectionResidual, _parse_cfg


S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _load_model(path: Path) -> Tuple[ConservativeDirectionResidual, Dict[str, Any]]:
    blob = torch.load(path, map_location="cpu")
    model = ConservativeDirectionResidual(
        input_dim=int(blob["input_dim"]),
        hidden_dim=int(blob["hidden_dim"]),
        delta_scale=float(blob["delta_tdir_scale"]),
        delta_log_tmag_clip=float(blob["delta_log_tmag_clip"]),
    )
    model.load_state_dict(blob["model_state"])
    model.eval()
    return model, blob


def _extract_corr(frame_i: Any, frame_j: Any) -> Dict[str, float]:
    img_i = load_gray(frame_i.image_path, (640, 320))
    img_j = load_gray(frame_j.image_path, (640, 320))
    orb = extract_orb_matches(img_i, img_j, 1200, 12, 0.75, False)
    flow = extract_sparse_flow(img_i, img_j, 600, 0.01, 7.0, 21, 3)
    return {
        "filtered_match_count": float(orb["filtered_match_count"]),
        "inlier_ratio": float(orb["inlier_ratio"]),
        "median_match_displacement": float(orb["median_match_displacement"]),
        "p90_match_displacement": float(orb["p90_match_displacement"]),
        "parallax_proxy": float(orb["parallax_proxy"]),
        "low_parallax_flag": 1.0 if orb["low_parallax_flag"] else 0.0,
        "median_flow_magnitude": float(flow["median_flow_magnitude"]),
        "p90_flow_magnitude": float(flow["p90_flow_magnitude"]),
        "flow_angle_dispersion": float(flow["flow_angle_dispersion"]) / 180.0,
        "forward_backward_flow_consistency": 0.0 if flow["forward_backward_flow_consistency"] is None else float(flow["forward_backward_flow_consistency"]),
    }


def _obs_weight(corr: Dict[str, float], gt_proxy_mag: float) -> float:
    score = 0.50 * min(1.0, float(corr["parallax_proxy"]) / 0.30) + 0.35 * min(1.0, float(corr["inlier_ratio"])) + 0.15 * min(1.0, float(corr["median_flow_magnitude"]) / 8.0)
    if float(corr["low_parallax_flag"]) >= 0.5 or gt_proxy_mag < 0.01:
        score *= 0.5
    return float(0.05 + (1.0 - 0.05) * min(1.0, max(0.0, score)))


def _gt_rel(gt: Dict[float, Dict[str, np.ndarray]], a: float, b: float) -> Tuple[np.ndarray, np.ndarray] | None:
    if a not in gt or b not in gt:
        return None
    gi, gj = gt[a], gt[b]
    return gj["R"].T @ gi["R"], gj["R"].T @ (gi["t"] - gj["t"])


def _metric(R: np.ndarray, t: np.ndarray, gt_rel: Tuple[np.ndarray, np.ndarray] | None) -> Dict[str, Any]:
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
    cfg = _parse_cfg(Path("configs/s5e20_true_kstep_composition_supervision.yaml"))
    model, blob = _load_model(Path(args.candidate) / "s5e20_model.pt")
    s5e15_prov = _read_jsonl(Path("external_baselines/results/s5e15_traceable_dense/edge_provenance.jsonl"))
    ts = read_timestamps(Path(args.timestamps))
    gt = read_tum(Path(args.groundtruth))
    frames = scan_frames(Path("data"), scene=args.scene, seq=args.seq)[(args.scene, args.seq)]
    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    cache_img: Dict[str, np.ndarray] = {}
    alpha = float(blob.get("export_delta_scale_alpha", cfg["model"]["export_delta_scale_alpha"]))

    rels: List[Tuple[np.ndarray, np.ndarray]] = []
    out_prov: List[Dict[str, Any]] = []
    mrows: List[Dict[str, Any]] = []
    delta_norms: List[float] = []
    for idx, base in enumerate(s5e15_prov):
        a, b = frames[idx], frames[idx + 1]
        x = pair_features(a, b, idx, max(len(frames) - 1, 1), cache_img)
        corr = _extract_corr(a, b)
        base_dir = np.asarray(base["translation_direction"], dtype=np.float32)
        base_dir = base_dir / max(float(np.linalg.norm(base_dir)), 1.0e-12)
        base_mag = float(base["translation_magnitude"])
        obs = _obs_weight(corr, base_mag)
        feat = np.concatenate(
            [
                x.astype(np.float32),
                base_dir.astype(np.float32),
                np.asarray([base_mag, np.log(max(base_mag, 1.0e-12))], dtype=np.float32),
                np.asarray(list(corr.values()), dtype=np.float32),
                np.asarray([obs], dtype=np.float32),
            ]
        )
        with torch.no_grad():
            pred = model(
                torch.tensor(feat, dtype=torch.float32).unsqueeze(0),
                F.normalize(torch.tensor(base_dir, dtype=torch.float32).unsqueeze(0), dim=1),
                torch.tensor([base_mag], dtype=torch.float32),
            )
        raw_delta = pred["delta_tdir"][0].cpu().numpy().astype(np.float64)
        delta = alpha * raw_delta
        delta_norm = float(np.linalg.norm(delta))
        final_dir = base_dir.astype(np.float64) + delta
        final_dir = final_dir / max(float(np.linalg.norm(final_dir)), 1.0e-12)
        delta_log_tmag = float(pred["delta_log_tmag"][0].item())
        final_tmag = float(pred["final_tmag"][0].item())
        R = np.asarray(base["rotation"]["value"], dtype=np.float64)
        t = final_dir * final_tmag
        rels.append((R, t))
        gt_rel = _gt_rel(gt, float(base["timestamp_i"]), float(base["timestamp_j"]))
        mm = _metric(R, t, gt_rel)
        mrows.append(mm)
        delta_norms.append(delta_norm)
        out_prov.append(
            {
                "edge_index": idx,
                "timestamp_i": float(base["timestamp_i"]),
                "timestamp_j": float(base["timestamp_j"]),
                "source_type": "direct_adjacent_prediction",
                "source_model": "S5E20_true_kstep_composition_supervision",
                "base_candidate": "S5E15",
                "uses_s5e15_base_direction": True,
                "uses_learned_delta_tdir": True,
                "delta_tdir": delta.tolist(),
                "delta_tdir_norm": delta_norm,
                "delta_scale_alpha": alpha,
                "final_tdir": final_dir.tolist(),
                "final_tmag": final_tmag,
                "delta_log_tmag": delta_log_tmag,
                "kstep_training_used": True,
                "uses_eval_gt_for_prediction": False,
                "uses_orbslam3_teacher": False,
                "rotation": {"representation": "matrix", "value": R.tolist()},
                "translation": {"frame": "B/local", "value": t.tolist()},
                "metric_preview": mm,
                "observability_weight": obs,
                "risk_flags": {
                    "noop_risk": bool(delta_norm <= 1.0e-8),
                    "aggressive_risk": bool(delta_norm > 0.20),
                },
            }
        )

    _write_tum(Path(args.out_tum), ts, rels)
    Path(args.out_provenance).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_provenance).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_prov) + "\n", encoding="utf-8")
    payload = {
        "experiment": "S5E20_true_kstep_composition_supervision",
        "coverage": {
            "num_poses": len(ts),
            "num_edges": len(out_prov),
            "direct_adjacent_prediction_edges": len(out_prov),
            "all_edges_traceable": True,
        },
        "component_metrics": _summary(mrows),
        "delta_tdir_control": {
            "delta_tdir_norm_mean": float(np.mean(delta_norms)) if delta_norms else None,
            "delta_tdir_norm_p90": float(np.percentile(delta_norms, 90)) if delta_norms else None,
            "delta_tdir_norm_max": float(np.max(delta_norms)) if delta_norms else None,
        },
        "metrics_rows": mrows,
        "trajectory_path": str(args.out_tum),
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
