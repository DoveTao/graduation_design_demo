#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from s5e2_adjacent_dense_lib import validation_from_logs, write_json
from s5e9_small_motion_geometry_lib import make_samples


OUT_DIR = Path("checkpoints/S5E10_order_aware_signed_direction_scale_recalibration_candidate")


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_samples, val_samples, split_summary, bucket_stats = make_samples()
    payload = {
        "attempted": True,
        "classification": "S5E10_SCALE_RECALIBRATION_SMOKE",
        "num_train_pairs": len(train_samples),
        "num_val_pairs": len(val_samples),
        "uses_scene01_seq03_for_training": False,
        "bucket_prior_choice": "p95",
        "bucket_priors": {
            k: {
                "median": v["median"],
                "p90": v["p90"],
                "p95": v["p95"],
                "p99": v["p99"],
                "max": v["max"],
                "clip_lo": v["clip_lo"],
                "clip_hi": v["clip_hi"],
            }
            for k, v in bucket_stats.items()
            if isinstance(v, dict) and "median" in v
        },
        "notes": [
            "S5E10 scale recalibration 改用更宽但仍受控的 bucket p95 prior。",
            "目标是在不重新引入 raw scale explosion 的前提下，把 S5E9 过度保守的 raw path_ratio 拉回一些。",
        ],
        "split_summary": split_summary,
        "validation_snapshot": validation_from_logs(),
    }
    write_json(OUT_DIR / "scale_recalibration_training_status.json", payload)
    return payload


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
