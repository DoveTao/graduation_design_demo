#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from s5e2_adjacent_dense_lib import write_json


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    gen3 = _read_json(Path(args.gen3_checkpoint))
    data2 = _read_json(Path(args.data2_checkpoint))
    gen2b = _read_json(Path(args.gen2b_checkpoint))
    s5e15 = _read_json(Path(args.s5e15_checkpoint))
    data2_seq03 = data2.get("gt_motion_distribution", {}).get("scene01_seq03", {})
    gen2b_motion = gen2b.get("motion_distribution", {})

    payload = {
        "comparison": {
            "data2_seq03_motion": data2_seq03,
            "gen2b_field_motion": gen2b_motion,
            "s5e15_scene01_seq03_metrics": s5e15.get("component_metrics", {}).get("overall", {}),
            "gen3_export": gen3.get("export", {}),
        },
        "answers": {
            "avoids_small_motion_risk": bool(gen2b_motion.get("small_motion_fraction") == 0.0),
            "gen3_metrics_available": bool(gen3.get("export", {}).get("num_pairs_predicted", 0) > 0),
            "better_than_scene01_seq03_if_available": None,
            "blocked_due_to_model_inference_path_not_data": bool(gen3.get("final_classification") in {"GEN3_SCENE_SPECIFIC_EXPORT_ONLY", "GEN3_INFERENCE_BRIDGE_BLOCKED"}),
            "worth_gen4_external_adapter_expansion": bool(gen3.get("recommendation", {}).get("do_gen4_external_expansion")),
        },
        "summary": {
            "main_takeaway": "360DVO field 避开了 DATA2 的 small-motion 风险，但当前阻塞点是 S5E15-like inference bridge 仍是 scene-specific export path，而不是外部数据本身。"
        },
    }
    write_json(Path(args.out_json), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen3-checkpoint", required=True)
    parser.add_argument("--data2-checkpoint", required=True)
    parser.add_argument("--gen2b-checkpoint", required=True)
    parser.add_argument("--s5e15-checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
