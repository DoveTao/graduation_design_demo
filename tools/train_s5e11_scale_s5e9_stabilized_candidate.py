#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from s5e11_correspondence_geometry_lib import CKPT_DIR, S5E9_CKPT
from s5e2_adjacent_dense_lib import read_json, validation_from_logs, write_json


def run(_args: argparse.Namespace) -> Dict[str, Any]:
    s5e9 = read_json(S5E9_CKPT)
    status = {
        "attempted": True,
        "classification": "S5E11_SCALE_S5E9_STABILIZED",
        "model_type": "scale_s5e9_stabilized_candidate",
        "scale_source": "S5E9_tight_bounded_residual",
        "scale_pipeline_audit": s5e9.get("scale_pipeline_audit", {}),
        "notes": [
            "S5E11 默认继承 S5E9 的 tight bounded residual scale 路径。",
            "不会把 S5E10 的 scale recalibration 爆炸结果作为默认 scale 方案。",
        ],
        "validation_snapshot": validation_from_logs(),
    }
    write_json(CKPT_DIR / "scale_s5e9_stabilized_training_status.json", status)
    return status


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser().parse_args()


if __name__ == "__main__":
    run(parse_args())
