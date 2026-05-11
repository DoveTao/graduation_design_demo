#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e11_correspondence_geometry_lib import CKPT_DIR, load_config, make_feature_records, direction_metrics, pair_order_observable, normalize
from s5e2_adjacent_dense_lib import validation_from_logs, write_json


def _candidate_dir(row: Dict[str, Any], params: Dict[str, float]) -> np.ndarray:
    feat = row["features"]
    base = normalize(row["prior_dir"])
    observable = row["observability_bucket"] in ("normal_observable", "large_motion_observable")
    if (
        observable
        and feat["essential_matrix_success"]
        and feat["essential_translation_axis_confidence"] >= params["essential_conf_threshold"]
        and feat["inlier_ratio"] >= params["inlier_ratio_threshold"]
        and feat["parallax_proxy"] >= params["parallax_threshold"]
    ):
        return normalize(feat["essential_translation_axis"])
    return base


def _score(rows: List[Dict[str, Any]], params: Dict[str, float]) -> Dict[str, Any]:
    preds = [_candidate_dir(r, params) for r in rows]
    metrics = direction_metrics(rows, preds)
    flip = []
    flip_obs = []
    for row in rows:
        feat = row["features"]
        axis = normalize(feat["essential_translation_axis"])
        cos = float(np.dot(axis, -axis))
        flip.append(cos)
        if row["observability_bucket"] in ("normal_observable", "large_motion_observable"):
            flip_obs.append(cos)
    metrics.update(
        {
            "pair_order_dir_flip_cosine_mean": pair_order_observable(flip)["pair_order_dir_flip_cosine_mean"],
            "pair_order_dir_flip_success_rate": pair_order_observable(flip)["pair_order_dir_flip_success_rate"],
            "pair_order_dir_flip_success_rate_observable": pair_order_observable(flip_obs)["pair_order_dir_flip_success_rate"],
        }
    )
    return metrics


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(Path(args.config))
    train_rows, val_rows, split_summary, obs_stats = make_feature_records()
    parallax_vals = [r["features"]["parallax_proxy"] for r in train_rows]
    search = []
    for c in cfg["candidate_search"]["essential_conf_thresholds"]:
        for ir in cfg["candidate_search"]["inlier_ratio_thresholds"]:
            for q in cfg["candidate_search"]["parallax_quantiles"]:
                params = {
                    "essential_conf_threshold": float(c),
                    "inlier_ratio_threshold": float(ir),
                    "parallax_threshold": float(np.percentile(parallax_vals, 100.0 * float(q))),
                }
                metrics = _score(val_rows, params)
                search.append({"params": params, "val_metrics": metrics})
    best = min(
        search,
        key=lambda item: item["val_metrics"]["signed_tdir_mean"]
        + 0.5 * item["val_metrics"]["tdir_abs_mean"]
        + 30.0 * item["val_metrics"]["anti_parallel_rate"]
        - 5.0 * (item["val_metrics"]["pair_order_dir_flip_success_rate_observable"] or 0.0),
    )
    status = {
        "attempted": True,
        "classification": "S5E11_CORRESPONDENCE_DIRECTION_TRAINING_SMOKE",
        "num_train_pairs": len(train_rows),
        "num_val_pairs": len(val_rows),
        "uses_scene01_seq03_for_training": False,
        "model_type": "correspondence_direction_candidate",
        "best_params": best["params"],
        "best_val_metrics": best["val_metrics"],
        "observable_edge_fraction_train": float(np.mean([r["observability_bucket"] in ("normal_observable", "large_motion_observable") for r in train_rows])),
        "observable_edge_fraction_val": float(np.mean([r["observability_bucket"] in ("normal_observable", "large_motion_observable") for r in val_rows])),
        "search_history": search[:12],
        "split_summary": split_summary,
        "notes": [
            "S5E11 correspondence_direction_candidate 在 val 上搜索 essential axis 的可置信启用阈值。",
            "如果显式几何轴在 observable edge 上都不能赢过 prior direction，就说明输入几何仍然不足。",
        ],
        "validation_snapshot": validation_from_logs(),
    }
    write_json(CKPT_DIR / "correspondence_direction_training_status.json", status)
    return status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/s5e11_correspondence_parallax_translation_geometry.yaml")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
