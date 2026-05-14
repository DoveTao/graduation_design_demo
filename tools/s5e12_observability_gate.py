#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import read_json, write_json


S5E11_BASELINE = {
    "observable_edge_fraction": 0.026490066225165563,
    "signed_direction_supervision_reliable_fraction": 0.026490066225165563,
    "small_motion_observable_fraction": 0.0,
}


def load_cfg_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_section_value(text: str, section: str, key: str, default: Any) -> Any:
    lines = text.splitlines()
    inside = False
    prefix = section + ":"
    for raw in lines:
        if not raw.strip():
            continue
        if not raw.startswith(" ") and raw.strip().endswith(":"):
            inside = raw.strip() == prefix
            continue
        if inside and raw.startswith("  ") and ":" in raw:
            k, v = raw.strip().split(":", 1)
            if k == key:
                val = v.strip()
                try:
                    if "." in val:
                        return float(val)
                    return int(val)
                except Exception:
                    return val
    return default


def bucket_row(row: Dict[str, Any], essential_limited: bool, small_thr: float, large_thr: float, cfg_text: str) -> str:
    if row.get("near_static_flag"):
        return "near_static_unobservable"
    if not row.get("correspondence_success"):
        return "outlier_correspondence"
    if row.get("low_parallax_flag") or row.get("inlier_ratio", 0.0) < parse_section_value(cfg_text, "observability", "min_inlier_ratio", 0.2):
        return "low_parallax_unreliable"
    if essential_limited:
        return "essential_geometry_limited"
    if (row.get("gt_tmag") or 0.0) >= large_thr:
        return "large_motion_observable"
    return "normal_observable"


def summarize_bucket(rows: List[Dict[str, Any]], bucket: str) -> Dict[str, Any]:
    sub = [r for r in rows if r["observability_bucket"] == bucket]
    if not sub:
        return {"count": 0, "fraction": 0.0}
    gt = np.asarray([r.get("gt_tmag") or 0.0 for r in sub], dtype=np.float64)
    return {
        "count": len(sub),
        "fraction": len(sub) / max(len(rows), 1),
        "gt_tmag_median": float(np.percentile(gt, 50)),
        "gt_tmag_p95": float(np.percentile(gt, 95)),
        "matched_keypoint_count_median": float(np.percentile([r.get("filtered_match_count", 0) for r in sub], 50)),
        "inlier_ratio_median": float(np.percentile([r.get("inlier_ratio", 0.0) for r in sub], 50)),
        "parallax_proxy_median": float(np.percentile([r.get("parallax_proxy", 0.0) for r in sub], 50)),
        "path_fraction": float(np.sum([r.get("gt_tmag") or 0.0 for r in sub]) / max(np.sum([r.get("gt_tmag") or 0.0 for r in rows]), 1e-12)),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    features = read_json(Path(args.features_json))
    essential = read_json(Path(args.essential_json))
    cfg_text = load_cfg_text(Path(args.config))
    rows = features.get("records", [])
    gt = np.asarray([r.get("gt_tmag") or 0.0 for r in rows], dtype=np.float64)
    small_thr = float(np.percentile(gt, 25)) if gt.size else 0.0
    large_thr = float(np.percentile(gt, 75)) if gt.size else 0.0
    essential_limited = not bool(essential.get("strict_essential_geometry_available"))

    for row in rows:
        row["observability_bucket"] = bucket_row(row, essential_limited, small_thr, large_thr, cfg_text)
        row["signed_direction_reliable"] = row["observability_bucket"] in {"normal_observable", "large_motion_observable", "essential_geometry_limited"}

    buckets = [
        "near_static_unobservable",
        "low_parallax_unreliable",
        "normal_observable",
        "large_motion_observable",
        "outlier_correspondence",
        "essential_geometry_limited",
    ]
    bucket_stats = {b: summarize_bucket(rows, b) for b in buckets}
    observable_count = sum(bucket_stats[b]["count"] for b in ["normal_observable", "large_motion_observable", "essential_geometry_limited"])
    unobservable_count = len(rows) - observable_count
    small_rows = [r for r in rows if (r.get("gt_tmag") or 0.0) <= small_thr]
    small_observable_count = sum(1 for r in small_rows if r["signed_direction_reliable"])
    reliable_fraction = observable_count / max(len(rows), 1)
    out = {
        "experiment": "S5E12_real_correspondence_feature_gate",
        "bucket_stats": bucket_stats,
        "observable_edge_count": observable_count,
        "observable_edge_fraction": observable_count / max(len(rows), 1),
        "unobservable_edge_count": unobservable_count,
        "unobservable_edge_fraction": unobservable_count / max(len(rows), 1),
        "small_motion_observable_count": small_observable_count,
        "small_motion_unobservable_count": len(small_rows) - small_observable_count,
        "small_motion_observable_fraction": small_observable_count / max(len(small_rows), 1),
        "signed_direction_supervision_reliable_fraction": reliable_fraction,
        "observable_edge_fraction_delta_vs_s5e11": observable_count / max(len(rows), 1) - S5E11_BASELINE["observable_edge_fraction"],
        "reliable_supervision_delta_vs_s5e11": reliable_fraction - S5E11_BASELINE["signed_direction_supervision_reliable_fraction"],
        "small_motion_observable_fraction_delta_vs_s5e11": small_observable_count / max(len(small_rows), 1) - S5E11_BASELINE["small_motion_observable_fraction"],
        "rows": rows,
    }
    write_json(Path(args.out_json), out)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--features-json", required=True)
    p.add_argument("--essential-json", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
