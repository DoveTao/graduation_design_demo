#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from s5e2_adjacent_dense_lib import write_json


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"num_pairs": 0}
    tmag = np.asarray([float(r["tmag"]) for r in rows], dtype=np.float64)
    rots = []
    for r in rows:
        R = np.asarray(r["R_BA"], dtype=np.float64)
        c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) * 0.5))
        rots.append(float(np.rad2deg(np.arccos(c))))
    k_vals = sorted({int(r["k"]) for r in rows})
    tmag_by_k = {}
    for k in k_vals:
        vals = np.asarray([float(r["tmag"]) for r in rows if int(r["k"]) == k], dtype=np.float64)
        tmag_by_k[str(k)] = {
            "count": int(vals.size),
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
        }
    return {
        "num_pairs": len(rows),
        "num_adjacent_pairs": sum(1 for r in rows if r["pair_type"] == "adjacent"),
        "num_kstep_pairs": sum(1 for r in rows if r["pair_type"] == "kstep"),
        "gt_step_mean": float(np.mean(tmag)),
        "gt_step_median": float(np.median(tmag)),
        "gt_step_p10": float(np.percentile(tmag, 10)),
        "gt_step_p25": float(np.percentile(tmag, 25)),
        "gt_step_p50": float(np.percentile(tmag, 50)),
        "gt_step_p75": float(np.percentile(tmag, 75)),
        "gt_step_p90": float(np.percentile(tmag, 90)),
        "gt_step_p95": float(np.percentile(tmag, 95)),
        "small_motion_fraction": float(np.mean(tmag < 0.005)),
        "very_small_motion_fraction": float(np.mean(tmag < 0.002)),
        "path_length": float(np.sum(tmag)),
        "rotation_step_mean": float(np.mean(rots)),
        "rotation_step_p90": float(np.percentile(rots, 90)),
        "tmag_distribution_by_k": tmag_by_k,
    }


def _similarity(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    if not a.get("num_pairs") or not b.get("num_pairs"):
        return {"available": False}
    return {
        "available": True,
        "median_ratio": float(a["gt_step_median"] / max(float(b["gt_step_median"]), 1.0e-12)),
        "small_motion_fraction_delta": float(a["small_motion_fraction"] - b["small_motion_fraction"]),
        "rotation_mean_ratio": float(a["rotation_step_mean"] / max(float(b["rotation_step_mean"]), 1.0e-12)),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    summary = _read_json(Path(args.manifest_summary))
    all_rows = _read_jsonl(Path(args.manifest_all))
    train_rows = _read_jsonl(Path(args.manifest_train))
    val_rows = _read_jsonl(Path(args.manifest_val))
    test_rows = _read_jsonl(Path(args.manifest_test))
    dset1 = _read_json(Path(args.dset1_checkpoint))
    data2 = _read_json(Path(args.data2_checkpoint))

    split_stats = {
        "all": _stats(all_rows),
        "train": _stats(train_rows),
        "val": _stats(val_rows),
        "test": _stats(test_rows),
    }
    seq_stats = {}
    for seq in summary.get("sequences_all", []):
        seq_stats[seq] = _stats([r for r in all_rows if r["seq_id"] == seq])

    data2_seq03 = data2.get("gt_motion_distribution", {}).get("scene01_seq03", {})
    dset1_motion = dset1.get("motion_distribution", {}).get("split_stats", {}).get("all", {})
    compared = {
        "vs_dset1": _similarity(split_stats["all"], dset1_motion),
        "vs_data2_seq03": {
            "gt_step_median_ratio": float(split_stats["all"]["gt_step_median"] / max(float(data2_seq03.get("gt_step_median", 1.0)), 1.0e-12)) if split_stats["all"].get("num_pairs") else None,
            "small_motion_fraction_delta": float(split_stats["all"]["small_motion_fraction"] - float(data2_seq03.get("small_motion_fraction", 0.0))) if split_stats["all"].get("num_pairs") else None,
            "path_length_ratio": float(split_stats["all"]["path_length"] / max(float(data2_seq03.get("path_length", 1.0)), 1.0e-12)) if split_stats["all"].get("num_pairs") else None,
        },
        "train_val_similarity": _similarity(split_stats["train"], split_stats["val"]),
        "train_test_similarity": _similarity(split_stats["train"], split_stats["test"]),
        "val_test_similarity": _similarity(split_stats["val"], split_stats["test"]),
    }

    readiness = {
        "continues_to_avoid_small_motion_risk": bool(split_stats["all"].get("small_motion_fraction", 1.0) <= 0.05),
        "train_val_test_motion_relatively_balanced": bool(
            compared["train_val_similarity"].get("available")
            and compared["train_test_similarity"].get("available")
            and compared["val_test_similarity"].get("available")
            and max(
                abs(float(compared["train_val_similarity"]["small_motion_fraction_delta"])),
                abs(float(compared["train_test_similarity"]["small_motion_fraction_delta"])),
                abs(float(compared["val_test_similarity"]["small_motion_fraction_delta"])),
            ) <= 0.05
        ),
        "meets_train360_readiness_threshold": False,
        "need_more_sequences": False,
    }
    readiness["meets_train360_readiness_threshold"] = bool(
        summary.get("num_sequences", 0) >= 7
        and split_stats["train"].get("num_pairs", 0) >= 2000
        and split_stats["val"].get("num_pairs", 0) >= 300
        and split_stats["test"].get("num_pairs", 0) >= 300
        and readiness["continues_to_avoid_small_motion_risk"]
    )
    readiness["need_more_sequences"] = not readiness["meets_train360_readiness_threshold"]

    payload = {
        "manifest_summary": summary,
        "split_stats": split_stats,
        "sequence_stats": seq_stats,
        "comparison": compared,
        "answers": readiness,
    }
    write_json(Path(args.out_json), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-summary", required=True)
    parser.add_argument("--manifest-all", required=True)
    parser.add_argument("--manifest-train", required=True)
    parser.add_argument("--manifest-val", required=True)
    parser.add_argument("--manifest-test", required=True)
    parser.add_argument("--dset1-checkpoint", required=True)
    parser.add_argument("--data2-checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
