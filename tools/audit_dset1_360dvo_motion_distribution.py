#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import write_json


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def _rot_deg(R_rows: List[List[float]]) -> float:
    R = np.asarray(R_rows, dtype=np.float64)
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) * 0.5))
    return float(np.rad2deg(np.arccos(c)))


def _stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"num_pairs": 0}
    gt_steps = np.asarray([float(r["tmag"]) for r in rows], dtype=np.float64)
    rots = np.asarray([_rot_deg(r["R_BA"]) for r in rows], dtype=np.float64)
    k_vals = sorted(set(int(r["k"]) for r in rows))
    tmag_by_k = {
        str(k): {
            "count": sum(1 for r in rows if int(r["k"]) == k),
            "median": float(np.median([float(r["tmag"]) for r in rows if int(r["k"]) == k])),
            "p90": float(np.percentile([float(r["tmag"]) for r in rows if int(r["k"]) == k], 90)),
        }
        for k in k_vals
    }
    return {
        "num_pairs": len(rows),
        "num_adjacent_pairs": sum(1 for r in rows if r["pair_type"] == "adjacent"),
        "num_kstep_pairs": sum(1 for r in rows if r["pair_type"] == "kstep"),
        "gt_step_median": float(np.median(gt_steps)),
        "gt_step_mean": float(np.mean(gt_steps)),
        "gt_step_p10": float(np.percentile(gt_steps, 10)),
        "gt_step_p25": float(np.percentile(gt_steps, 25)),
        "gt_step_p50": float(np.percentile(gt_steps, 50)),
        "gt_step_p75": float(np.percentile(gt_steps, 75)),
        "gt_step_p90": float(np.percentile(gt_steps, 90)),
        "gt_step_p95": float(np.percentile(gt_steps, 95)),
        "small_motion_fraction": float(np.mean(gt_steps < 0.005)),
        "very_small_motion_fraction": float(np.mean(gt_steps < 0.002)),
        "path_length": float(np.sum(gt_steps)),
        "rotation_step_mean": float(np.mean(rots)),
        "rotation_step_p90": float(np.percentile(rots, 90)),
        "tmag_distribution_by_k": tmag_by_k,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    summary = _read_json(Path(args.manifest_summary))
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.manifest_summary).parent
    data2 = _read_json(Path(args.data2_checkpoint))
    manifests = {
        "all": _read_jsonl(base_dir / "pair_manifest_all.jsonl"),
        "train": _read_jsonl(base_dir / "pair_manifest_train.jsonl"),
        "val": _read_jsonl(base_dir / "pair_manifest_val.jsonl"),
        "test": _read_jsonl(base_dir / "pair_manifest_test.jsonl"),
    }
    seq_stats = {}
    for seq in summary.get("selected_sequences", []):
        seq_rows = [r for r in manifests["all"] if r["seq_id"] == seq]
        seq_stats[seq] = _stats(seq_rows)
    split_stats = {name: _stats(rows) for name, rows in manifests.items()}
    data2_seq03 = data2.get("gt_motion_distribution", {}).get("scene01_seq03", {})
    overall = split_stats["all"]
    payload = {
        "dataset": "360DVO",
        "summary": summary,
        "split_stats": split_stats,
        "sequence_stats": seq_stats,
        "compared_to_data2_seq03": {
            "gt_step_median_ratio": float(overall.get("gt_step_median", 0.0) / max(float(data2_seq03.get("gt_step_median", 1.0)), 1.0e-12)) if overall.get("num_pairs", 0) else None,
            "small_motion_fraction_delta": float(overall.get("small_motion_fraction", 0.0) - float(data2_seq03.get("small_motion_fraction", 0.0))) if overall.get("num_pairs", 0) else None,
            "path_length_ratio": float(overall.get("path_length", 0.0) / max(float(data2_seq03.get("path_length", 1.0)), 1.0e-12)) if overall.get("num_pairs", 0) else None,
        },
        "answers": {
            "reduces_small_motion_risk": bool(overall.get("num_pairs", 0) and overall["small_motion_fraction"] < float(data2_seq03.get("small_motion_fraction", 1.0))),
            "suitable_as_main_train_eval_dataset": bool(overall.get("num_pairs", 0) and overall["small_motion_fraction"] < 0.1 and not summary.get("insufficient_sequences")),
            "multi_sequence_split_feasible": bool(len(summary.get("split", {}).get("train", [])) > 0 and len(summary.get("split", {}).get("test", [])) > 0),
            "enter_train360": bool(overall.get("num_pairs", 0) and len(summary.get("selected_sequences", [])) >= 3),
        },
    }
    write_json(out_json, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-summary", required=True)
    parser.add_argument("--data2-checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
