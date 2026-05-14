#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rows = _read_rows(Path(args.s5e5_provenance))
    tmag = np.asarray([r["metric_preview"]["tmag_ratio"] for r in rows], dtype=np.float64)
    pred = np.asarray([r["metric_preview"]["pred_step_length"] for r in rows], dtype=np.float64)
    gt = np.asarray([r["metric_preview"]["gt_step_length"] for r in rows], dtype=np.float64)
    tdir = np.asarray([r["metric_preview"]["tdir_deg"] for r in rows], dtype=np.float64)
    tdir_abs = np.asarray([r["metric_preview"]["tdir_abs_deg"] for r in rows], dtype=np.float64)
    top_fracs = {}
    for k in [10, 20, 50]:
        idx = np.argsort(tmag)[-k:]
        top_fracs[f"top{k}_tmag_path_fraction"] = float(pred[idx].sum() / max(pred.sum(), 1.0e-12))
    p90, p95, p99 = (float(x) for x in np.percentile(tmag, [90, 95, 99]))
    max_ratio = float(np.max(tmag))
    hi = tmag > p95
    longest = cur = 0
    long_run_pred = 0.0
    cur_pred = 0.0
    for i, flag in enumerate(hi):
        if flag:
            cur += 1
            cur_pred += float(pred[i])
            if cur > longest:
                longest = cur
                long_run_pred = cur_pred
        else:
            cur = 0
            cur_pred = 0.0
    out = {
        "experiment": "S5E6_robust_scale_guard_and_direction_metric_gate_candidate",
        "top10_tmag_path_fraction": top_fracs["top10_tmag_path_fraction"],
        "top20_tmag_path_fraction": top_fracs["top20_tmag_path_fraction"],
        "top50_tmag_path_fraction": top_fracs["top50_tmag_path_fraction"],
        "tmag_ratio_p90": p90,
        "tmag_ratio_p95": p95,
        "tmag_ratio_p99": p99,
        "tmag_ratio_max": max_ratio,
        "corr_tmag_gt_step": float(np.corrcoef(tmag, gt)[0, 1]),
        "corr_tmag_signed_tdir": float(np.corrcoef(tmag, tdir)[0, 1]),
        "corr_tmag_tdir_abs": float(np.corrcoef(tmag, tdir_abs)[0, 1]),
        "longest_run_over_p95": int(longest),
        "long_run_path_fraction": float(long_run_pred / max(pred.sum(), 1.0e-12)),
        "small_motion_high_tmag_rate": float(np.mean((gt < np.percentile(gt, 25)) & (tmag > p90))),
        "outlier_dominated": bool(top_fracs["top20_tmag_path_fraction"] > 0.5),
        "long_run_dominated": bool(longest >= 10 and (long_run_pred / max(pred.sum(), 1.0e-12)) > 0.1),
        "small_motion_amplification": bool(np.mean((gt < np.percentile(gt, 25)) & (tmag > p90)) > 0.03),
        "tmag_head_unbounded": bool(max_ratio > 1000 or p99 > 500),
        "recommended_guard": "train_split_p95_clamp_with_median_logmag_correction",
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--s5e5-metrics", required=True)
    p.add_argument("--s5e5-provenance", required=True)
    p.add_argument("--s5e5-trajectory", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
