#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import write_json


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rot_deg(R_rows: List[List[float]]) -> float:
    R = np.asarray(R_rows, dtype=np.float64)
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) * 0.5))
    return float(np.rad2deg(np.arccos(c)))


def _empty(reason: str) -> Dict[str, Any]:
    return {
        "num_pairs": 0,
        "gt_step_median": None,
        "gt_step_mean": None,
        "gt_step_p10": None,
        "gt_step_p25": None,
        "gt_step_p50": None,
        "gt_step_p75": None,
        "gt_step_p90": None,
        "gt_step_p95": None,
        "small_motion_fraction": None,
        "very_small_motion_fraction": None,
        "path_length": None,
        "rotation_step_mean": None,
        "rotation_step_p90": None,
        "train_eval_sequence_split_feasibility": False,
        "vs_data2_seq03": {},
        "larger_or_more_stable_than_data2_seq03": False,
        "suitable_for_tdir_generalization_eval": False,
        "small_motion_risk_like_data2": None,
        "blockers": [reason],
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rows = _read_jsonl(Path(args.manifest))
    data2 = _read_json(Path(args.data2_checkpoint))
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        payload = _empty("EMPTY_MANIFEST")
        write_json(out_path, payload)
        return payload

    valid = [r for r in rows if r.get("valid_pose")]
    if not valid:
        payload = _empty("NO_VALID_POSE_ROWS")
        write_json(out_path, payload)
        return payload

    gt_steps = np.asarray([float(r["tmag"]) for r in valid], dtype=np.float64)
    rots = np.asarray([_rot_deg(r["R_BA"]) for r in valid], dtype=np.float64)
    seq_ids = sorted({str(r["seq_id"]) for r in valid})
    seq03 = data2["gt_motion_distribution"]["scene01_seq03"]
    gt_step_median = float(np.median(gt_steps))
    small_motion_fraction = float(np.mean(gt_steps < 0.005))
    path_length = float(np.sum(gt_steps))
    payload = {
        "num_pairs": int(len(valid)),
        "gt_step_median": gt_step_median,
        "gt_step_mean": float(np.mean(gt_steps)),
        "gt_step_p10": float(np.percentile(gt_steps, 10)),
        "gt_step_p25": float(np.percentile(gt_steps, 25)),
        "gt_step_p50": gt_step_median,
        "gt_step_p75": float(np.percentile(gt_steps, 75)),
        "gt_step_p90": float(np.percentile(gt_steps, 90)),
        "gt_step_p95": float(np.percentile(gt_steps, 95)),
        "small_motion_fraction": small_motion_fraction,
        "very_small_motion_fraction": float(np.mean(gt_steps < 0.002)),
        "path_length": path_length,
        "rotation_step_mean": float(np.mean(rots)),
        "rotation_step_p90": float(np.percentile(rots, 90)),
        "train_eval_sequence_split_feasibility": bool(len(seq_ids) >= 2),
        "vs_data2_seq03": {
            "data2_seq03_gt_step_median": float(seq03["gt_step_median"]),
            "data2_seq03_small_motion_fraction": float(seq03["small_motion_fraction"]),
            "data2_seq03_path_length": float(seq03["path_length"]),
            "gt_step_median_ratio_over_data2_seq03": float(gt_step_median / max(float(seq03["gt_step_median"]), 1.0e-12)),
            "small_motion_fraction_delta_vs_data2_seq03": float(small_motion_fraction - float(seq03["small_motion_fraction"])),
            "path_length_ratio_over_data2_seq03": float(path_length / max(float(seq03["path_length"]), 1.0e-12)),
        },
        "larger_or_more_stable_than_data2_seq03": bool(
            gt_step_median > float(seq03["gt_step_median"]) and small_motion_fraction < float(seq03["small_motion_fraction"])
        ),
        "suitable_for_tdir_generalization_eval": bool(
            gt_step_median > float(seq03["gt_step_median"]) * 1.1 and small_motion_fraction < 0.35 and len(seq_ids) >= 2
        ),
        "small_motion_risk_like_data2": bool(small_motion_fraction >= float(seq03["small_motion_fraction"]) * 0.9),
        "blockers": [],
    }
    write_json(out_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data2-checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
