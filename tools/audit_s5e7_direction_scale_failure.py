#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _rows(path: Path) -> List[Dict[str, Any]]:
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            out.append(json.loads(line))
    return out


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rows = _rows(Path(args.provenance))
    guarded = [r.get("metric_preview") or {} for r in rows]
    gt_step = np.asarray([m.get("gt_step_length") or 0.0 for m in guarded], dtype=np.float64)
    tmag = np.asarray([m.get("tmag_ratio") or 0.0 for m in guarded], dtype=np.float64)
    tdir = np.asarray([m.get("tdir_deg") or 0.0 for m in guarded], dtype=np.float64)
    tdir_abs = np.asarray([m.get("tdir_abs_deg") or 0.0 for m in guarded], dtype=np.float64)
    pred_step = np.asarray([m.get("pred_step_length") or 0.0 for m in guarded], dtype=np.float64)
    small_thr = float(np.percentile(gt_step, 25))
    small = gt_step <= small_thr
    out = {
        "experiment": "S5E7_direction_scale_calibrated_geometry",
        "small_motion_threshold": small_thr,
        "small_motion_count": int(np.sum(small)),
        "small_motion_fraction": float(np.mean(small)),
        "small_motion_tdir_mean": float(np.mean(tdir[small])) if np.any(small) else None,
        "small_motion_tmag_p95": float(np.percentile(tmag[small], 95)) if np.any(small) else None,
        "small_motion_path_fraction": float(np.sum(pred_step[small]) / max(np.sum(pred_step), 1.0e-12)) if np.any(small) else None,
        "small_motion_amplification": bool(np.mean((small) & (tmag > np.percentile(tmag, 90))) > 0.03),
        "corr_tdir_tmag": float(np.corrcoef(tdir, tmag)[0, 1]),
        "corr_tdir_abs_tmag": float(np.corrcoef(tdir_abs, tmag)[0, 1]),
        "corr_tdir_gt_step": float(np.corrcoef(tdir, gt_step)[0, 1]),
        "corr_tmag_gt_step": float(np.corrcoef(tmag, gt_step)[0, 1]),
        "summary": "S5E7 需要同时解决方向误差与尺度高分位问题，不能只把路径长度压回去。",
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--provenance", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
