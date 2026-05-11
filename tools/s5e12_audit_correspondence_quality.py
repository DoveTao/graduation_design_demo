#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import read_json, write_json


def corr(a: List[float], b: List[float]) -> Any:
    xa = np.asarray(a, dtype=np.float64)
    xb = np.asarray(b, dtype=np.float64)
    if xa.size < 2 or xb.size < 2 or np.std(xa) < 1e-12 or np.std(xb) < 1e-12:
        return "unavailable"
    return float(np.corrcoef(xa, xb)[0, 1])


def run(args: argparse.Namespace) -> Dict[str, Any]:
    payload = read_json(Path(args.features_json))
    rows = payload.get("records", [])
    success = [r for r in rows if r.get("correspondence_success")]
    gt = [r["gt_tmag"] for r in rows if r.get("gt_tmag") is not None]
    parallax = [r["parallax_proxy"] for r in rows if r.get("gt_tmag") is not None]
    inlier = [r["inlier_ratio"] for r in rows if r.get("gt_tmag") is not None]
    match_counts = [r["filtered_match_count"] for r in rows if r.get("gt_tmag") is not None]
    flow_mag = [r["median_flow_magnitude"] for r in rows if r.get("gt_tmag") is not None]
    small_thr = float(np.percentile(np.asarray(gt, dtype=np.float64), 25)) if gt else 0.0
    out = {
        "experiment": "S5E12_real_correspondence_feature_gate",
        "edge_count": len(rows),
        "correspondence_success_count": len(success),
        "correspondence_success_fraction": len(success) / max(len(rows), 1),
        "filtered_match_count_median": float(np.percentile([r["filtered_match_count"] for r in rows], 50)) if rows else None,
        "inlier_match_count_median": float(np.percentile([r["inlier_match_count"] for r in rows], 50)) if rows else None,
        "inlier_ratio_median": float(np.percentile([r["inlier_ratio"] for r in rows], 50)) if rows else None,
        "parallax_proxy_median": float(np.percentile([r["parallax_proxy"] for r in rows], 50)) if rows else None,
        "low_parallax_fraction": float(np.mean([1.0 if r["low_parallax_flag"] else 0.0 for r in rows])) if rows else None,
        "near_static_fraction": float(np.mean([1.0 if r["near_static_flag"] else 0.0 for r in rows])) if rows else None,
        "parallax_gt_tmag_correlation": corr(parallax, gt),
        "inlier_ratio_observability_correlation": corr(inlier, gt),
        "matched_keypoint_count_observability_correlation": corr(match_counts, gt),
        "low_parallax_flag_vs_small_motion": float(np.mean([1.0 if (r["low_parallax_flag"] and (r.get("gt_tmag") or 0.0) <= small_thr) else 0.0 for r in rows])) if rows else None,
        "near_static_flag_vs_small_motion": float(np.mean([1.0 if (r["near_static_flag"] and (r.get("gt_tmag") or 0.0) <= small_thr) else 0.0 for r in rows])) if rows else None,
        "flow_magnitude_vs_gt_tmag_correlation": corr(flow_mag, gt),
        "correspondence_too_sparse": bool(len(success) / max(len(rows), 1) < 0.35),
    }
    write_json(Path(args.out_json), out)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features-json", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
