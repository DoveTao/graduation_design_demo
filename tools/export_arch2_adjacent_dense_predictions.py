#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s5e2_adjacent_dense_lib import angle_deg_from_rot, read_timestamps, read_tum, rot_to_quat_xyzw, scan_frames, vector_angle_deg, write_json
from train_arch2_mainline_softcorr_geometry import Arch2SoftcorrGeometryModel, _edge_softcorr, _parse_cfg


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _load_model(path: Path) -> Tuple[Arch2SoftcorrGeometryModel, Dict[str, Any]]:
    blob = torch.load(path, map_location="cpu")
    model = Arch2SoftcorrGeometryModel(
        token_in_dim=int(blob["token_in_dim"]),
        hidden_dim=int(blob["hidden_dim"]),
        max_alpha=float(blob["max_alpha"]),
        scale_delta_clip=float(blob["scale_delta_clip"]),
    )
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
    _ = _parse_cfg(Path("configs/arch2_mainline_rotation_compensated_softcorr_geometry.yaml"))
    model, blob = _load_model(Path(args.candidate) / "arch2_model.pt")
    s5e15_prov = _read_jsonl(Path("external_baselines/results/s5e15_traceable_dense/edge_provenance.jsonl"))
    frames = scan_frames(Path("data"), scene=args.scene, seq=args.seq)[(args.scene, args.seq)]
    gt = read_tum(Path(args.groundtruth))
    ts = read_timestamps(Path(args.timestamps))
    low_thr = float(blob["low_threshold"])
    high_thr = float(blob["high_threshold"])
    max_matches = 64
    rels: List[Tuple[np.ndarray, np.ndarray]] = []
    prov: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    entropy_vals: List[float] = []
    flow_vals: List[float] = []
    delta_norms: List[float] = []
    for idx, base in enumerate(s5e15_prov):
        a, b = frames[idx], frames[idx + 1]
        R_coarse = np.asarray(base["rotation"]["value"], dtype=np.float32)
        base_tdir = np.asarray(base["translation_direction"], dtype=np.float32)
        base_tmag = float(base["translation_magnitude"])
        base_log_tmag = float(np.log(max(base_tmag, 1.0e-6)))
        soft = _edge_softcorr(a, b, R_coarse, max_matches=max_matches)
        obs = float(soft["observability_score"])
        if obs <= low_thr:
            gate_floor = 0.0
            level = "low"
        elif obs >= high_thr:
            gate_floor = 1.0
            level = "high"
        else:
            gate_floor = (obs - low_thr) / max(high_thr - low_thr, 1.0e-6)
            level = "mid"
        with torch.no_grad():
            out = model(
                torch.tensor(soft["local_features"], dtype=torch.float32).unsqueeze(0),
                torch.tensor(soft["bearing_a"], dtype=torch.float32).unsqueeze(0),
                torch.tensor(soft["bearing_b"], dtype=torch.float32).unsqueeze(0),
                torch.tensor(soft["W_ab"], dtype=torch.float32).unsqueeze(0),
                torch.tensor(soft["W_ba"], dtype=torch.float32).unsqueeze(0),
                torch.tensor(R_coarse, dtype=torch.float32).unsqueeze(0),
                torch.tensor(base_tdir, dtype=torch.float32).unsqueeze(0),
                torch.tensor([base_log_tmag], dtype=torch.float32),
                torch.tensor([gate_floor], dtype=torch.float32),
            )
        final_dir = out["final_tdir"][0].cpu().numpy().astype(np.float64)
        final_mag = float(out["final_tmag"][0].cpu().item())
        delta = out["delta_effective"][0].cpu().numpy().astype(np.float64)
        gate = float(out["gate_value"][0].cpu().item())
        delta_norm = float(np.linalg.norm(delta))
        entropy = float(out["entropy"][0].mean().cpu().item()) if out["entropy"].ndim > 1 else float(out["entropy"][0].cpu().item())
        residual_flow = float(out["residual_flow"][0].norm(dim=-1).mean().cpu().item())
        t = final_dir * final_mag
        gt_rel = _gt_rel(gt, float(base["timestamp_i"]), float(base["timestamp_j"]))
        mm = _metric(R_coarse.astype(np.float64), t, gt_rel)
        rows.append(mm | {"observability_score": obs, "gate_value": gate, "subset": level})
        entropy_vals.append(entropy)
        flow_vals.append(residual_flow)
        delta_norms.append(delta_norm)
        rels.append((R_coarse.astype(np.float64), t.astype(np.float64)))
        prov.append(
            {
                "edge_index": idx,
                "timestamp_i": float(base["timestamp_i"]),
                "timestamp_j": float(base["timestamp_j"]),
                "source_type": "direct_adjacent_prediction",
                "source_model": "ARCH2_mainline_rotation_compensated_softcorr_geometry",
                "base_candidate": "S5E15",
                "uses_softcorr_geometry": True,
                "uses_rotation_compensation": True,
                "uses_eval_gt_for_prediction": False,
                "uses_orbslam3_teacher": False,
                "uses_s5e15_base_direction": True,
                "R_BA": R_coarse.tolist(),
                "tdir_B_base": base_tdir.tolist(),
                "delta_tdir": delta.tolist(),
                "delta_tdir_norm": delta_norm,
                "observability_score": obs,
                "gate_value": gate,
                "final_tdir": final_dir.tolist(),
                "final_tmag": final_mag,
                "softcorr_entropy": entropy,
                "residual_flow_norm": residual_flow,
                "no_harm_gate_reason": f"{level}_observability",
                "metric_preview": mm,
            }
        )
    _write_tum(Path(args.out_tum), ts, rels)
    Path(args.out_provenance).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_provenance).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in prov) + "\n", encoding="utf-8")
    payload = {
        "experiment": "ARCH2_mainline_rotation_compensated_softcorr_geometry",
        "coverage": {"num_poses": len(ts), "num_edges": len(prov), "direct_adjacent_prediction_edges": len(prov), "all_edges_traceable": True},
        "component_metrics": _summary(rows),
        "delta_tdir_control": {
            "delta_tdir_norm_mean": float(np.mean(delta_norms)) if delta_norms else None,
            "delta_tdir_norm_p90": float(np.percentile(delta_norms, 90)) if delta_norms else None,
            "delta_tdir_norm_max": float(np.max(delta_norms)) if delta_norms else None,
        },
        "softcorr_stats": {
            "entropy_mean": float(np.mean(entropy_vals)) if entropy_vals else None,
            "entropy_p90": float(np.percentile(entropy_vals, 90)) if entropy_vals else None,
            "residual_flow_norm_mean": float(np.mean(flow_vals)) if flow_vals else None,
            "residual_flow_norm_p90": float(np.percentile(flow_vals, 90)) if flow_vals else None,
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
