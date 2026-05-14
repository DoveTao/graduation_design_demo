#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import write_json
from s5e8_translation_geometry_lib import make_pair_samples_with_buckets


def run(args: argparse.Namespace) -> Dict[str, Any]:
    train_samples, val_samples, _split_summary, bucket_stats = make_pair_samples_with_buckets()
    all_samples = train_samples + val_samples
    mags = np.asarray([s.gt_magnitude for s in all_samples], dtype=np.float64)
    small_name = "small_motion"
    small = np.asarray([s.bucket_name == small_name for s in all_samples], dtype=bool)
    near_static = mags < np.percentile(mags, 10)
    repeated = mags < 1.0e-4
    label_outliers = sorted(
        [
            {
                "edge_index": s.edge_index,
                "scene": s.frame_i.scene,
                "seq": s.frame_i.seq,
                "timestamp_i": s.frame_i.timestamp,
                "timestamp_j": s.frame_j.timestamp,
                "gt_magnitude": s.gt_magnitude,
                "bucket": s.bucket_name,
            }
            for s in all_samples
        ],
        key=lambda x: x["gt_magnitude"],
        reverse=True,
    )[:20]

    # train-label reliability proxy: small-motion direction changes are more sensitive to annotation noise
    def _bucket_rows(name: str) -> List[Any]:
        return [s for s in all_samples if s.bucket_name == name]

    def _dir_stability(rows: List[Any]) -> float:
        if len(rows) < 2:
            return 0.0
        dirs = np.asarray([s.gt_direction for s in rows], dtype=np.float64)
        dots = np.sum(dirs[1:] * dirs[:-1], axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        return float(np.mean(np.degrees(np.arccos(dots))))

    out = {
        "experiment": "S5E8_translation_geometry_diagnostic_ablation",
        "gt_tmag_median": float(np.percentile(mags, 50)),
        "gt_tmag_p90": float(np.percentile(mags, 90)),
        "gt_tmag_p95": float(np.percentile(mags, 95)),
        "gt_tmag_p99": float(np.percentile(mags, 99)),
        "gt_tmag_max": float(np.max(mags)),
        "small_motion_count": int(np.sum(small)),
        "small_motion_fraction": float(np.mean(small)),
        "gt_direction_stability_by_bucket": {
            "small_motion": _dir_stability(_bucket_rows("small_motion")),
            "normal_motion": _dir_stability(_bucket_rows("normal_motion")),
            "large_motion": _dir_stability(_bucket_rows("large_motion")),
        },
        "small_motion_tdir_label_reliability_proxy": float(_dir_stability(_bucket_rows("small_motion"))),
        "repeated_or_near_static_edge_count": int(np.sum(repeated)),
        "low_parallax_or_low_displacement_edge_count": int(np.sum(near_static)),
        "label_outlier_candidates": label_outliers,
        "train_bucket_stats": bucket_stats,
        "summary": "S5E8 先检查监督本身是否已经对小位移和平移方向学习构成不利条件。",
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, out)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
