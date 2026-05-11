#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e11_correspondence_geometry_lib import RESULT_DIR, CKPT_DIR, CONFIG_PATH, load_config, make_feature_records
from s5e2_adjacent_dense_lib import write_json


def _bucket_summary(rows: List[Dict[str, Any]], bucket: str) -> Dict[str, Any]:
    sub = [r for r in rows if r["observability_bucket"] == bucket]
    if not sub:
        return {"count": 0, "fraction": 0.0}
    gt = np.asarray([r["gt_magnitude"] for r in sub], dtype=np.float64)
    match = np.asarray([r["features"]["matched_keypoint_count"] for r in sub], dtype=np.float64)
    inlier = np.asarray([r["features"]["inlier_ratio"] for r in sub], dtype=np.float64)
    para = np.asarray([r["features"]["parallax_proxy"] for r in sub], dtype=np.float64)
    return {
        "count": len(sub),
        "fraction": len(sub) / max(len(rows), 1),
        "gt_tmag_median": float(np.percentile(gt, 50)),
        "gt_tmag_p95": float(np.percentile(gt, 95)),
        "matched_keypoint_count_median": float(np.percentile(match, 50)),
        "inlier_ratio_median": float(np.percentile(inlier, 50)),
        "parallax_proxy_median": float(np.percentile(para, 50)),
    }


def _report(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# S5E11 geometric observability audit",
        "",
        "## 几何特征可用性",
        str(payload["feature_availability"]),
        "",
        "## observability audit",
        str(payload["global_observability"]),
        "",
        "## bucket summary",
        str(payload["bucket_summary"]),
        "",
        "## 说明",
        "S5E11 先检查显式 correspondence / parallax / essential proxy 特征是否足以支撑 signed direction 监督。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(Path(args.config))
    train_rows, val_rows, split_summary, obs_stats = make_feature_records()
    rows = train_rows + val_rows
    feature_availability = {
        "correspondence_features_available": bool(np.mean([1.0 if r["features"]["correspondence_features_available"] else 0.0 for r in rows]) > 0.5),
        "optical_flow_features_available": bool(np.mean([1.0 if r["features"]["optical_flow_features_available"] else 0.0 for r in rows]) > 0.5),
        "essential_matrix_features_available": bool(np.mean([1.0 if r["features"]["essential_matrix_features_available"] else 0.0 for r in rows]) > 0.1),
        "fallback_proxy_features_used": bool(np.mean([1.0 if r["features"]["fallback_proxy_features_used"] else 0.0 for r in rows]) > 0.1),
    }
    observable = [r for r in rows if r["observability_bucket"] in ("normal_observable", "large_motion_observable")]
    unobservable = [r for r in rows if r["observability_bucket"] in ("near_static_unobservable", "low_parallax_unreliable", "outlier_correspondence")]
    small_rows = [r for r in rows if r["sample"].bucket_name in ("near_static", "small_motion")]
    payload = {
        "experiment": "S5E11_correspondence_parallax_translation_geometry",
        "feature_availability": feature_availability,
        "observability_stats": obs_stats,
        "global_observability": {
            "observable_edge_count": len(observable),
            "observable_edge_fraction": len(observable) / max(len(rows), 1),
            "unobservable_edge_count": len(unobservable),
            "unobservable_edge_fraction": len(unobservable) / max(len(rows), 1),
            "small_motion_observable_fraction": (sum(1 for r in small_rows if r["observability_bucket"] in ("normal_observable", "large_motion_observable")) / max(len(small_rows), 1)),
            "small_motion_unobservable_fraction": (sum(1 for r in small_rows if r["observability_bucket"] not in ("normal_observable", "large_motion_observable")) / max(len(small_rows), 1)),
            "signed_direction_supervision_reliable_fraction": float(np.mean([r["signed_direction_loss_weight"] >= 1.0 for r in rows])),
            "signed_direction_loss_weight_mean": float(np.mean([r["signed_direction_loss_weight"] for r in rows])),
            "excluded_direction_edge_count": int(sum(r["signed_direction_loss_weight"] == 0.0 for r in rows)),
            "excluded_direction_edge_fraction": float(np.mean([r["signed_direction_loss_weight"] == 0.0 for r in rows])),
        },
        "bucket_summary": {
            k: _bucket_summary(rows, k)
            for k in [
                "near_static_unobservable",
                "low_parallax_unreliable",
                "normal_observable",
                "large_motion_observable",
                "outlier_correspondence",
            ]
        },
        "split_summary": split_summary,
        "notes": [
            "S5E11 默认把 signed direction loss 放在 normal_observable / large_motion_observable 上。",
            "near_static_unobservable 与部分 low_parallax_unreliable 只保留弱监督或不参与 signed direction 监督。",
        ],
    }
    write_json(Path(args.out_json), payload)
    _report(Path(args.out_report), payload)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--out-json", default=str(RESULT_DIR / "geometric_observability_audit.json"))
    p.add_argument("--out-report", default=str(CKPT_DIR / "geometric_observability_audit.md"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
