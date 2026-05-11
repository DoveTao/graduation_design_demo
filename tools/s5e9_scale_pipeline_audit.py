#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import read_tum, write_json


def _rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _adjacent_step_lengths(tum_path: Path) -> np.ndarray:
    rows = []
    tum = read_tum(tum_path)
    keys = sorted(tum.keys())
    for a, b in zip(keys[:-1], keys[1:]):
        rows.append(float(np.linalg.norm(tum[b]["t"] - tum[a]["t"])))
    return np.asarray(rows, dtype=np.float64)


def _stats(arr: np.ndarray, include_p99: bool = True) -> Dict[str, float]:
    out = {
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }
    if include_p99:
        out["p99"] = float(np.percentile(arr, 99))
    return out


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rows = _rows(Path(args.provenance))
    gt_tmag = np.asarray([r["metric_preview_raw"]["gt_step_length"] for r in rows], dtype=np.float64)
    raw_tmag = np.asarray([r.get("raw_tmag_from_prior", r.get("pred_tmag_raw", 0.0)) for r in rows], dtype=np.float64)
    exported_tmag = np.asarray([r.get("pred_tmag", r.get("translation_magnitude", 0.0)) for r in rows], dtype=np.float64)
    evaluator_tmag = _adjacent_step_lengths(Path(args.raw_trajectory))
    prior_tmag = np.asarray([r.get("bucket_prior_tmag", r.get("raw_tmag_from_prior", 0.0)) for r in rows], dtype=np.float64)
    pred_delta = np.asarray([r.get("pred_log_scale_delta", 0.0) or 0.0 for r in rows], dtype=np.float64)
    ratio = raw_tmag / np.maximum(gt_tmag, 1e-12)
    export_eval_consistency = bool(
        abs(float(np.median(exported_tmag)) - float(np.median(evaluator_tmag))) < 1e-6
        or np.allclose(exported_tmag[: min(len(exported_tmag), len(evaluator_tmag))], evaluator_tmag[: min(len(exported_tmag), len(evaluator_tmag))], atol=1e-6)
    )
    unit_mismatch_suspected = bool(np.percentile(ratio, 50) > 50.0 or np.percentile(ratio, 95) > 50.0)
    out = {
        "experiment": "S5E9_scale_unit_small_motion_signed_direction_fix",
        "gt_tmag": _stats(gt_tmag),
        "prior_tmag": _stats(prior_tmag, include_p99=False),
        "pred_log_scale_delta": _stats(pred_delta),
        "pred_tmag_pre_export": _stats(raw_tmag, include_p99=False),
        "pred_tmag_exported": _stats(exported_tmag, include_p99=False),
        "pred_tmag_evaluator": _stats(evaluator_tmag, include_p99=False),
        "export_eval_tmag_consistency": export_eval_consistency,
        "unit_mismatch_suspected": unit_mismatch_suspected,
        "raw_tmag_to_gt_tmag_median_ratio": float(np.percentile(ratio, 50)),
        "raw_tmag_to_gt_tmag_p95_ratio": float(np.percentile(ratio, 95)),
        "summary": "S5E9 首先检查 raw scale explosion 是不是来自单位、导出或字段使用错误。",
    }
    write_json(Path(args.out_json), out)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--provenance", required=True)
    p.add_argument("--raw-trajectory", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
