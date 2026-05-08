#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1.0e-12)
    x, y, z, w = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _read_tum(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        ts = float(parts[0])
        tx, ty, tz = (float(parts[1]), float(parts[2]), float(parts[3]))
        qx, qy, qz, qw = (float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]))
        rows.append(
            {
                "timestamp": ts,
                "t": np.asarray([tx, ty, tz], dtype=np.float64),
                "R": _quat_xyzw_to_rot(qx, qy, qz, qw),
            }
        )
    if not rows:
        raise RuntimeError(f"No valid TUM poses found in {path}")
    rows.sort(key=lambda x: x["timestamp"])
    return rows


def _match_rows(gt_rows: List[Dict[str, Any]], est_rows: List[Dict[str, Any]], tol: float) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    matches: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    j = 0
    for gt in gt_rows:
        best = None
        while j < len(est_rows) and est_rows[j]["timestamp"] < gt["timestamp"] - tol:
            j += 1
        for cand_idx in (j - 1, j, j + 1):
            if cand_idx < 0 or cand_idx >= len(est_rows):
                continue
            est = est_rows[cand_idx]
            dt = abs(est["timestamp"] - gt["timestamp"])
            if dt <= tol and (best is None or dt < best[0]):
                best = (dt, est)
        if best is not None:
            matches.append((gt, best[1]))
    return matches


def _umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    if src.shape != dst.shape:
        raise ValueError("src and dst must have the same shape")
    dim, n = src.shape
    mean_src = src.mean(axis=1, keepdims=True)
    mean_dst = dst.mean(axis=1, keepdims=True)
    src_centered = src - mean_src
    dst_centered = dst - mean_dst
    cov = (dst_centered @ src_centered.T) / float(n)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_src = np.sum(src_centered * src_centered) / float(n)
        scale = float(np.trace(np.diag(D) @ S) / max(var_src, 1.0e-12))
    else:
        scale = 1.0
    t = (mean_dst - scale * R @ mean_src).reshape(dim)
    return scale, R, t


def _apply_alignment(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (scale * (R @ points.T)).T + t.reshape(1, 3)


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diffs = points[1:] - points[:-1]
    return float(np.linalg.norm(diffs, axis=1).sum())


def evaluate_external_baseline_trajectory(
    *,
    gt_path: Path,
    est_path: Path,
    alignment: str = "se3",
    match_tolerance: float = 1.0e-3,
) -> Dict[str, Any]:
    gt_rows = _read_tum(gt_path)
    est_rows = _read_tum(est_path)
    matches = _match_rows(gt_rows, est_rows, match_tolerance)
    if len(matches) < 2:
        return {
            "status": "insufficient_matches",
            "alignment_mode": alignment,
            "num_est_poses": len(est_rows),
            "num_gt_poses": len(gt_rows),
            "num_matched_poses": len(matches),
            "tracking_success_rate": float(len(matches) / max(len(gt_rows), 1)),
        }

    gt_pts = np.asarray([gt["t"] for gt, _ in matches], dtype=np.float64)
    est_pts_raw = np.asarray([est["t"] for _, est in matches], dtype=np.float64)

    if alignment == "none":
        scale = 1.0
        R_align = np.eye(3, dtype=np.float64)
        t_align = np.zeros(3, dtype=np.float64)
    elif alignment == "se3":
        scale, R_align, t_align = _umeyama(est_pts_raw.T, gt_pts.T, with_scale=False)
    elif alignment == "sim3":
        scale, R_align, t_align = _umeyama(est_pts_raw.T, gt_pts.T, with_scale=True)
    else:
        raise ValueError(f"Unsupported alignment={alignment!r}")

    est_pts_aligned = _apply_alignment(est_pts_raw, scale, R_align, t_align)
    ate = float(np.sqrt(np.mean(np.sum((est_pts_aligned - gt_pts) ** 2, axis=1))))

    gt_step = gt_pts[1:] - gt_pts[:-1]
    est_step = est_pts_aligned[1:] - est_pts_aligned[:-1]
    drift = float(np.sqrt(np.mean(np.sum((est_step - gt_step) ** 2, axis=1))))

    gt_path_len = _path_length(gt_pts)
    est_path_len_raw = _path_length(est_pts_raw)
    path_ratio = float(est_path_len_raw / max(gt_path_len, 1.0e-12))

    return {
        "status": "ok",
        "alignment_mode": alignment,
        "ATE": ate,
        "drift": drift,
        "path_ratio": path_ratio,
        "num_est_poses": len(est_rows),
        "num_gt_poses": len(gt_rows),
        "num_matched_poses": len(matches),
        "tracking_success_rate": float(len(matches) / max(len(gt_rows), 1)),
        "gt_path_length": gt_path_len,
        "est_path_length_raw": est_path_len_raw,
        "metric_notes": {
            "ATE": "RMSE position error after requested alignment.",
            "drift": "RPE-like translation RMSE on consecutive matched steps after requested alignment.",
            "path_ratio": "Raw estimated path length divided by GT path length; scale-sensitive and not Sim(3)-normalized.",
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an external baseline trajectory against exported GT in TUM format.")
    p.add_argument("--gt", required=True, help="Ground-truth TUM trajectory path.")
    p.add_argument("--est", required=True, help="Estimated TUM trajectory path.")
    p.add_argument("--alignment", default="se3", choices=["none", "se3", "sim3"], help="Alignment mode.")
    p.add_argument("--match-tolerance", type=float, default=1.0e-3, help="Timestamp matching tolerance in seconds.")
    p.add_argument("--output-json", default="", help="Optional output json path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_external_baseline_trajectory(
        gt_path=Path(args.gt),
        est_path=Path(args.est),
        alignment=args.alignment,
        match_tolerance=float(args.match_tolerance),
    )
    text = json.dumps(result, indent=2, ensure_ascii=True)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
