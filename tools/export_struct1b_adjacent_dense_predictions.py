#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s5e2_adjacent_dense_lib import angle_deg_from_rot, read_timestamps, read_tum, rot_to_quat_xyzw, scan_frames, vector_angle_deg, write_json
from s5e7_direction_scale_lib import load_npz_model, predict_s5e2, predict_s5e3_head, rotvec_to_matrix
from train_struct1_geometry_token_pose_solver import S5E15_POLICY, S5E2_BASE, S5E3_HEADS, _edge_softcorr, _norm
from train_struct1b_scale_guard_repair import load_trained_struct1b
from s5e2_adjacent_dense_lib import pair_features


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
        "signed_tdir_mean_deg": _mean("tdir_deg"),
        "anti_parallel_rate": _mean("anti_parallel_flag"),
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
    model, blob = load_trained_struct1b(Path(args.candidate) / "struct1b_scale_repair.pt")
    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    scale_factor = float(json.loads(S5E15_POLICY.read_text(encoding="utf-8"))["scale_factor"])
    frames = scan_frames(Path("data"), scene=args.scene, seq=args.seq)[(args.scene, args.seq)]
    gt = read_tum(Path(args.groundtruth))
    ts = read_timestamps(Path(args.timestamps))
    cache_img: Dict[str, np.ndarray] = {}
    rels: List[Tuple[np.ndarray, np.ndarray]] = []
    prov: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for idx in range(len(frames) - 1):
        a, b = frames[idx], frames[idx + 1]
        x = pair_features(a, b, idx, max(len(frames) - 1, 1), cache_img)
        prior = predict_s5e2(s5e2, x)
        R_coarse = rotvec_to_matrix(prior[:3]).astype(np.float32)
        base_tdir = _norm(prior[3:6]).astype(np.float32)
        base_log_tmag = float(predict_s5e3_head(s5e3, "mag", x).reshape(-1)[0]) + math.log(scale_factor)
        soft = _edge_softcorr(a, b, R_coarse, max_matches=int(blob["config"]["soft_correspondence_geometry_layer"]["max_matches"]), temperature=float(blob["config"]["soft_correspondence_geometry_layer"]["temperature"]))
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
            )
        pred_R = out["R_BA"][0].cpu().numpy().astype(np.float64)
        final_dir = out["tdir_B"][0].cpu().numpy().astype(np.float64)
        final_mag = float(out["tmag"][0].cpu().item())
        t = final_dir * final_mag
        gt_rel = _gt_rel(gt, float(a.timestamp), float(b.timestamp))
        mm = _metric(pred_R, t, gt_rel)
        rows.append(mm)
        rels.append((pred_R, t.astype(np.float64)))
        prov.append(
            {
                "edge_index": idx,
                "source_model": "STRUCT1B",
                "hard_scale_guard_enabled": True,
                "raw_log_tmag": float(out["raw_log_tmag"][0].cpu().item()),
                "bounded_log_tmag": float(out["bounded_log_tmag"][0].cpu().item()),
                "final_tmag": final_mag,
                "tdir_from_geometry_tokens": True,
                "uses_eval_gt_for_prediction": False,
                "uses_orbslam3_teacher": False,
                "metric_preview": mm,
            }
        )
    _write_tum(Path(args.out_tum), ts, rels)
    Path(args.out_provenance).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_provenance).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in prov) + "\n", encoding="utf-8")
    payload = {
        "experiment": "STRUCT1B_scale_guard_repair",
        "coverage": {"num_poses": len(ts), "num_edges": len(prov), "all_edges_traceable": True, "coverage": 1.0},
        "component_metrics": _summary(rows),
        "metrics_rows": rows,
    }
    write_json(Path(args.out_metrics), payload)
    return payload


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
