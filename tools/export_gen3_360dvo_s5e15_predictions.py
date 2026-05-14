#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from miniyaml import load_yaml_like
from s5e2_adjacent_dense_lib import write_json


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_config(path: Path) -> Dict[str, Any]:
    return load_yaml_like(path) or {}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_config(Path(args.config))
    availability = _read_json(Path(args.availability))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_jsonl(Path(str(cfg.get("dataset", {}).get("manifest"))))

    pred_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "edge_component_metrics.json"
    tum_path = out_dir / "scene_field_s5e15_like_est_tum.txt"
    summary_path = out_dir / "export_summary.json"

    if not availability.get("inference_bridge_ready"):
        pred_path.write_text("", encoding="utf-8")
        blocked = {
            "model_eval_available": False,
            "num_pairs_total": len(manifest),
            "num_pairs_predicted": 0,
            "pair_coverage": 0.0,
            "adjacent_pair_coverage": 0.0,
            "kstep_pair_coverage": 0.0,
            "blocked": True,
            "eval_blocker": availability.get("bridge_classification"),
            "prediction_fabricated": False,
            "uses_gt_for_prediction": False,
            "uses_gt_scale_calibration": False,
            "uses_orbslam3_teacher": False,
        }
        write_json(summary_path, blocked)
        write_json(metrics_path, {"component_metrics_available": False, "metrics_rows": [], "component_metrics": {}, "coverage": blocked})
        tum_path.write_text("# GEN3 export blocked\n", encoding="utf-8")
        return blocked

    raise RuntimeError("GEN3 ready-path export is intentionally not implemented until a real S5E15 image-pair forward bridge exists.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--availability", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
