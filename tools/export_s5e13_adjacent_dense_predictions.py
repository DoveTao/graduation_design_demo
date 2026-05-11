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

from s5e2_adjacent_dense_lib import angle_deg_from_rot, read_timestamps, read_tum, rot_to_quat_xyzw, vector_angle_deg, write_json
from s5e7_direction_scale_lib import rotvec_to_matrix
from train_s5e13_real_correspondence_direction_candidate import S5E13Model

S5E9_PROV = Path("external_baselines/results/s5e9_traceable_dense/edge_provenance.jsonl")


def load_config(path: Path) -> Dict[str, Any]:
    def parse_scalar(text: str) -> Any:
        text = text.strip()
        if text.lower() == "true":
            return True
        if text.lower() == "false":
            return False
        try:
            if "." in text:
                return float(text)
            return int(text)
        except Exception:
            return text

    cfg: Dict[str, Any] = {}
    current: str | None = None
    mode: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if not raw.startswith(" "):
            if ":" in raw:
                key, value = raw.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value:
                    cfg[key] = parse_scalar(value)
                    current = None
                    mode = None
                else:
                    cfg[key] = {}
                    current = key
                    mode = "dict"
        else:
            if current is None:
                continue
            if raw.startswith("  - "):
                if not isinstance(cfg[current], list):
                    cfg[current] = []
                cfg[current].append(parse_scalar(raw[4:]))
                mode = "list"
            elif raw.startswith("  ") and ":" in raw:
                if mode == "list":
                    continue
                if not isinstance(cfg[current], dict):
                    cfg[current] = {}
                key, value = raw.strip().split(":", 1)
                cfg[current][key.strip()] = parse_scalar(value.strip())
    return cfg


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


def _summary(metrics_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _vals(key: str) -> np.ndarray:
        return np.asarray([m[key] for m in metrics_rows if m.get(key) is not None], dtype=np.float64)
    def _pct(key: str, q: float) -> Optional[float]:
        vals = _vals(key)
        return None if vals.size == 0 else float(np.percentile(vals, q))
    def _mean(key: str) -> Optional[float]:
        vals = _vals(key)
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
        "tmag_max_ratio": None if _vals("tmag_ratio").size == 0 else float(np.max(_vals("tmag_ratio"))),
        "path_ratio": pred_path / max(gt_path, 1.0e-12),
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


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(Path("configs/s5e13_real_correspondence_signed_direction.yaml"))
    weighted = json.loads(Path(cfg["weighted_dataset_json"]).read_text(encoding="utf-8"))
    ckpt = torch.load(Path(args.candidate) / "s5e13_best.pt", map_location="cpu")
    model_cfg = ckpt["config"]
    model = S5E13Model(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(model_cfg["model"]["hidden_dim"]),
        log_scale_clip=float(model_cfg["model"]["log_scale_clip"]),
        rot_delta_scale=float(model_cfg["model"]["rot_delta_scale"]),
        dir_delta_scale=float(model_cfg["model"]["dir_delta_scale"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    gt = read_tum(Path(args.groundtruth))
    timestamps = read_timestamps(Path(args.timestamps))
    eval_rows = weighted["eval_rows"]
    s5e9_mags: Dict[int, float] = {}
    if S5E9_PROV.exists():
        for line in S5E9_PROV.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            s5e9_mags[int(row["edge_index"])] = float(row.get("translation_magnitude", 0.0))

    feat = torch.tensor([r["model_feature_vector"] for r in eval_rows], dtype=torch.float32)
    prior_rot = torch.tensor([r["prior_rotvec"] for r in eval_rows], dtype=torch.float32)
    prior_dir = F.normalize(torch.tensor([r["prior_dir"] for r in eval_rows], dtype=torch.float32), dim=1)
    bucket_prior = torch.tensor([weighted["bucket_stats"][r["prior_bucket"]]["median"] for r in eval_rows], dtype=torch.float32)

    with torch.no_grad():
        out = model(feat, prior_rot, prior_dir, bucket_prior)

    metrics_rows: List[Dict[str, Any]] = []
    provenance_rows: List[Dict[str, Any]] = []
    weight_rows: List[Dict[str, Any]] = []
    rels: List[Tuple[np.ndarray, np.ndarray]] = []

    for idx, row in enumerate(eval_rows):
        # Keep S5E9-style stable rotation/scale path, and only trust learned direction on observable edges.
        R = rotvec_to_matrix(np.asarray(row["prior_rotvec"], dtype=np.float64))
        model_dir = out["pred_dir"][idx].cpu().numpy()
        prior_dir_np = np.asarray(row["prior_dir"], dtype=np.float64)
        if row["signed_direction_reliable"]:
            pred_dir = model_dir
        elif row["observable"]:
            pred_dir = prior_dir_np + 0.35 * model_dir
            pred_dir = pred_dir / max(float(np.linalg.norm(pred_dir)), 1.0e-12)
        else:
            pred_dir = prior_dir_np
        pred_tmag = float(s5e9_mags.get(int(row["edge_index"]), weighted["bucket_stats"][row["prior_bucket"]]["median"]))
        t = pred_dir * pred_tmag
        rels.append((R, t))
        gt_rel = _gt_rel(gt, float(row["timestamp_i"]), float(row["timestamp_j"]))
        metric = _metric(R, t, gt_rel)
        metrics_rows.append(metric)
        weight_row = {
            "edge_index": int(row["edge_index"]),
            "observable": bool(row["observable"]),
            "signed_direction_reliable": bool(row["signed_direction_reliable"]),
            "small_motion": bool(row["small_motion"]),
            "near_static": bool(row["near_static"]),
            "low_parallax": bool(row["low_parallax"]),
            "parallax_proxy": float(row["features"]["parallax_proxy"]),
            "flow_magnitude": float(row["features"]["median_flow_magnitude"]),
            "match_count": int(row["features"]["filtered_match_count"]),
            "inlier_ratio": float(row["features"]["inlier_ratio"]),
            "signed_tdir_loss_weight": float(row["signed_tdir_loss_weight"]),
            "tdir_abs_loss_weight": float(row["tdir_abs_loss_weight"]),
            "tmag_loss_weight": float(row["tmag_loss_weight"]),
            "rot_loss_weight": float(row["rot_loss_weight"]),
            "reason": row["reason"],
        }
        weight_rows.append(weight_row)
        provenance_rows.append(
            {
                "edge_index": int(row["edge_index"]),
                "timestamp_i": float(row["timestamp_i"]),
                "timestamp_j": float(row["timestamp_j"]),
                "source_type": "direct_adjacent_prediction",
                "source_model": "S5E13_real_correspondence_signed_direction_candidate",
                "uses_gt_for_prediction": False,
                "uses_correspondence_features": True,
                "strict_essential_geometry_used": False,
                "rotation": {"representation": "matrix", "value": R.tolist()},
                "translation": {"frame": "B/local", "value": t.tolist()},
                "translation_direction": pred_dir.tolist(),
                "translation_magnitude": pred_tmag,
                "signed_tdir_loss_weight": float(row["signed_tdir_loss_weight"]),
                "observable": bool(row["observable"]),
                "signed_direction_reliable": bool(row["signed_direction_reliable"]),
                "small_motion": bool(row["small_motion"]),
                "large_motion": bool(row["large_motion"]),
                "low_parallax": bool(row["low_parallax"]),
                "near_static": bool(row["near_static"]),
                "anti_parallel_flag": bool(metric.get("anti_parallel_flag", False)),
                "metric_preview": metric,
                "notes": [row["reason"], "S5E12 correspondence feature gate mask reused for S5E13 signed direction weighting"],
            }
        )

    _write_tum(Path(args.out_tum), timestamps, rels)
    Path(args.out_provenance).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_provenance).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in provenance_rows) + "\n", encoding="utf-8")
    Path(args.out_weights).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in weight_rows) + "\n", encoding="utf-8")

    payload = {
        "experiment": "S5E13_real_correspondence_signed_direction_training",
        "coverage": {
            "num_poses": len(timestamps),
            "num_edges": len(provenance_rows),
            "direct_adjacent_prediction_edges": len(provenance_rows),
            "all_edges_traceable": True,
        },
        "component_metrics": _summary(metrics_rows),
        "metrics_rows": metrics_rows,
        "trajectory_path": str(args.out_tum),
    }
    write_json(Path(args.out_metrics), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--seq", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--s5e12-feature-dir", required=True)
    p.add_argument("--timestamps", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--out-tum", required=True)
    p.add_argument("--out-provenance", required=True)
    p.add_argument("--out-metrics", required=True)
    p.add_argument("--out-weights", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
