#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from s5e11_correspondence_geometry_lib import CKPT_DIR, load_config, make_feature_records
from s5e2_adjacent_dense_lib import validation_from_logs, write_json


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(Path(args.config))
    train_rows, val_rows, split_summary, obs_stats = make_feature_records()
    status = {
        "attempted": True,
        "classification": "S5E11_OBSERVABILITY_MASK_TRAINING_SMOKE",
        "num_train_pairs": len(train_rows),
        "num_val_pairs": len(val_rows),
        "uses_scene01_seq03_for_training": False,
        "model_type": "observability_mask_only_candidate",
        "signed_direction_loss_weight_mean": sum(r["signed_direction_loss_weight"] for r in train_rows + val_rows) / max(len(train_rows) + len(val_rows), 1),
        "signed_direction_loss_weight_by_bucket": {
            bucket: cfg["direction_mask"][bucket]
            for bucket in cfg["direction_mask"]
        },
        "excluded_direction_edge_count": int(sum(r["signed_direction_loss_weight"] == 0.0 for r in train_rows + val_rows)),
        "excluded_direction_edge_fraction": float(sum(r["signed_direction_loss_weight"] == 0.0 for r in train_rows + val_rows) / max(len(train_rows) + len(val_rows), 1)),
        "observability_stats": obs_stats,
        "split_summary": split_summary,
        "notes": [
            "S5E11 observability_mask_only_candidate 不改变 S5E9 scale 路径，只改变 signed direction supervision 的权重分配。",
            "训练目的主要是判断 unobservable edge 是否在污染 direction 学习。",
        ],
        "validation_snapshot": validation_from_logs(),
    }
    out = CKPT_DIR / "observability_mask_only_training_status.json"
    write_json(out, status)
    return status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/s5e11_correspondence_parallax_translation_geometry.yaml")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
