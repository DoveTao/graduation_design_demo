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
from train_s5e21_no_harm_refinement import NoHarmRefiner, _parse_cfg


S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _corr(frame_i: Any, frame_j: Any) -> Dict[str, float]:
    img_i = load_gray(frame_i.image_path, (640, 320))
    img_j = load_gray(frame_j.image_path, (640, 320))
    orb = extract_orb_matches(img_i, img_j, 1200, 12, 0.75, False)
    flow = extract_sparse_flow(img_i, img_j, 600, 0.01, 7.0, 21, 3)
    return {
        "inlier_ratio": float(orb["inlier_ratio"]),
        "parallax_proxy": float(orb["parallax_proxy"]),
        "median_flow_magnitude": float(flow["median_flow_magnitude"]),
        "low_parallax_flag": 1.0 if orb["low_parallax_flag"] else 0.0,
    }


def _gate(score: float, low_thr: float, high_thr: float, motion_mag: float) -> Tuple[float, bool, bool]:
    is_low = motion_mag < 0.01 or score <= low_thr
    is_high = score >= high_thr
    if is_low:
        return 0.0, is_high, True
    return min(1.0, max(0.0, (score - low_thr) / max(high_thr - low_thr, 1.0e-6))), is_high, False


def _obs_score(corr: Dict[str, float], motion_mag: float) -> float:
    score = 0.45 * min(1.0, corr["inlier_ratio"]) + 0.35 * min(1.0, corr["parallax_proxy"] / 0.30) + 0.20 * min(1.0, corr["median_flow_magnitude"] / 8.0)
    if corr["low_parallax_flag"] >= 0.5:
        score *= 0.55
    if motion_mag < 0.01:
        score *= 0.35
    return float(score)


def _load_model(path: Path) -> Tuple[NoHarmRefiner, Dict[str, Any]]:
    blob = torch.load(path, map_location="cpu")
    model = NoHarmRefiner(int(blob["input_dim"]), int(blob["hidden_dim"]))
    model.load_state_dict(blob["model_state"])
    model.eval()
    return model, blob


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
    cfg = _parse_cfg(Path("configs/s5e21_no_harm_observability_gated_refinement.yaml"))
    model, blob = _load_model(Path(args.candidate) / "s5e21_model.pt")
    s5e15_prov = _read_jsonl(Path("external_baselines/results/s5e15_traceable_dense/edge_provenance.jsonl"))
    frames = scan_frames(Path("data"), scene=args.scene, seq=args.seq)[(args.scene, args.seq)]
    gt = read_tum(Path(args.groundtruth))
    ts = read_timestamps(Path(args.timestamps))
    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    cache_img: Dict[str, np.ndarray] = {}
    rels: List[Tuple[np.ndarray, np.ndarray]] = []
    rows: List[Dict[str, Any]] = []
    prov: List[Dict[str, Any]] = []
    alpha = float(blob["max_alpha"])
    low_thr = float(blob["low_threshold"])
    high_thr = float(blob["high_threshold"])
    delta_norms: List[float] = []
    for idx, base in enumerate(s5e15_prov):
        a, b = frames[idx], frames[idx + 1]
        x = pair_features(a, b, idx, max(len(frames) - 1, 1), cache_img)
        corr = _corr(a, b)
        base_dir = np.asarray(base["translation_direction"], dtype=np.float32)
        base_mag = float(base["translation_magnitude"])
        obs_score = _obs_score(corr, base_mag)
        gate_value, is_high, is_low = _gate(obs_score, low_thr, high_thr, base_mag)
        feat = np.concatenate(
            [
                x.astype(np.float32),
                base_dir.astype(np.float32),
                np.asarray([base_mag], dtype=np.float32),
                np.asarray(list(corr.values()), dtype=np.float32),
                np.asarray([obs_score, gate_value], dtype=np.float32),
            ]
        )
        with torch.no_grad():
            raw = model(torch.tensor(feat, dtype=torch.float32).unsqueeze(0))[0]
        raw_np = raw.cpu().numpy().astype(np.float64)
        raw_norm = float(np.linalg.norm(raw_np))
        if raw_norm > float(blob["hard_clip_norm"]):
            raw_np = raw_np * (float(blob["hard_clip_norm"]) / max(raw_norm, 1.0e-12))
        effective = raw_np * gate_value * alpha
        delta_norm = float(np.linalg.norm(effective))
        final_dir = base_dir.astype(np.float64) + effective
        final_dir = final_dir / max(float(np.linalg.norm(final_dir)), 1.0e-12)
        final_tmag = base_mag
        R = np.asarray(base["rotation"]["value"], dtype=np.float64)
        t = final_dir * final_tmag
        rels.append((R, t))
        gt_rel = _gt_rel(gt, float(base["timestamp_i"]), float(base["timestamp_j"]))
        mm = _metric(R, t, gt_rel)
        rows.append(mm | {"is_high_confidence": is_high, "is_low_signal": is_low, "modified": delta_norm > 1.0e-8})
        delta_norms.append(delta_norm)
        prov.append(
            {
                "edge_index": idx,
                "timestamp_i": float(base["timestamp_i"]),
                "timestamp_j": float(base["timestamp_j"]),
                "base_candidate": "S5E15",
                "base_tdir": base["translation_direction"],
                "delta_tdir": effective.tolist(),
                "delta_tdir_norm": delta_norm,
                "gate_value": gate_value,
                "alpha": alpha,
                "final_tdir": final_dir.tolist(),
                "final_tmag": final_tmag,
                "modified": bool(delta_norm > 1.0e-8),
                "is_high_confidence": bool(is_high),
                "is_low_signal": bool(is_low),
                "no_harm_gate_reason": "low_signal_zero_gate" if is_low else ("high_confidence_refine" if is_high else "mid_signal_soft_gate"),
                "uses_eval_gt_for_prediction": False,
                "uses_orbslam3_teacher": False,
                "metric_preview": mm,
            }
        )
    _write_tum(Path(args.out_tum), ts, rels)
    Path(args.out_provenance).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_provenance).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in prov) + "\n", encoding="utf-8")
    payload = {
        "experiment": "S5E21_no_harm_observability_gated_refinement",
        "coverage": {"num_poses": len(ts), "num_edges": len(prov), "direct_adjacent_prediction_edges": len(prov), "all_edges_traceable": True},
        "component_metrics": _summary(rows),
        "delta_tdir_control": {
            "delta_tdir_norm_mean": float(np.mean(delta_norms)) if delta_norms else None,
            "delta_tdir_norm_p90": float(np.percentile(delta_norms, 90)) if delta_norms else None,
            "delta_tdir_norm_max": float(np.max(delta_norms)) if delta_norms else None,
        },
        "metrics_rows": rows,
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
