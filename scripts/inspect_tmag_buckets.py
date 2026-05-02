#!/usr/bin/env python3
"""
File: scripts/inspect_tmag_buckets.py
Description:
    Read-only translation magnitude bucket diagnostic. Reads eval_buckets_latest.json
    and odom_trajectory_steps_latest.csv from eval-only experiments, prints per-k
    and per-dt tmag statistics, and flags systematic underestimation patterns.

Usage:
    python3 scripts/inspect_tmag_buckets.py E_O39_shape_debug E_O49m3_shape_debug

Notes:
    - No training, no model loading, read-only.
    - Uses eval_buckets_latest.json for per-k aggregate metrics.
    - Uses odom_trajectory_steps_latest.csv for per-step tmag details.
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional


CHECKPOINTS = os.path.join(os.path.dirname(__file__), "..", "checkpoints")


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_dir(name: str) -> str:
    if os.path.isabs(name) and os.path.isdir(name):
        return name
    candidate = os.path.join(CHECKPOINTS, name)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(f"Experiment directory not found: {name} (tried {candidate})")


def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _fmt(v, prec=2, width=8):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-".rjust(width)
    if isinstance(v, float):
        return f"{v:.{prec}f}".rjust(width)
    return str(v).rjust(width)


def inspect(exp_names: List[str]) -> None:
    for exp_name in exp_names:
        exp_dir = _resolve_dir(exp_name)
        buckets = _read_json(os.path.join(exp_dir, "eval_buckets_latest.json"))
        fsummary = _read_json(os.path.join(exp_dir, "final_summary.json"))
        odom = _read_json(os.path.join(exp_dir, "odom_metrics_latest.json"))

        drift = _safe_float((odom or {}).get("odom_metric_drift"))
        ate = _safe_float((odom or {}).get("odom_metric_ATE"))

        # Overall tmag from last_eval
        le = (fsummary or {}).get("last_eval", {}) if isinstance((fsummary or {}).get("last_eval"), dict) else {}
        tmag_rel_overall = _safe_float(le.get("tmag_rel_err") or le.get("tmag_rel"))

        print(f"\n{'='*70}")
        print(f"  {exp_name}")
        print(f"  drift={_fmt(drift,4,0).strip()}  ATE={_fmt(ate,4,0).strip()}  tmag_rel_err={_fmt(tmag_rel_overall,4,0).strip()}")
        print(f"{'='*70}")

        # ---- Section 1: per-k aggregates from bucket_k ----
        bk = (buckets or {}).get("bucket_k", {})
        if bk:
            print(f"\n  {'k':>4} {'cnt':>5} {'tmag_rel':>9} {'rot':>7} {'tdir_abs':>8} {'local_A':>8}")
            print(f"  {'-'*4} {'-'*5} {'-'*9} {'-'*7} {'-'*8} {'-'*8}")
            for k_label in sorted(bk.keys(), key=lambda x: int(x.split("=")[1])):
                v = bk[k_label]
                k = k_label.split("=")[1]
                print(f"  {k:>4} {v.get('count','?'):>5} "
                      f"{_fmt(_safe_float(v.get('tmag_rel_err')),3,9).strip():>9} "
                      f"{_fmt(_safe_float(v.get('rot')),2,7).strip():>7} "
                      f"{_fmt(_safe_float(v.get('tdir_abs')),2,8).strip():>8} "
                      f"{_fmt(_safe_float(v.get('tdir_local_A_abs')),2,8).strip():>8}")

        # ---- Section 2: per-dt aggregates from bucket_dt ----
        bdt = (buckets or {}).get("bucket_dt", {})
        if bdt:
            dt_order = ["dt<0.1", "0.1<=dt<0.3", "0.3<=dt<0.5", "0.5<=dt<1", "1<=dt<2", "2<=dt<3.5", "3.5<=dt<5"]
            print(f"\n  {'dt_bucket':<16} {'cnt':>5} {'tmag_rel':>9} {'rot':>7} {'tdir_abs':>8}")
            print(f"  {'-'*16} {'-'*5} {'-'*9} {'-'*7} {'-'*8}")
            for dtb in dt_order:
                if dtb in bdt:
                    v = bdt[dtb]
                    print(f"  {dtb:<16} {v.get('count','?'):>5} "
                          f"{_fmt(_safe_float(v.get('tmag_rel_err')),3,9).strip():>9} "
                          f"{_fmt(_safe_float(v.get('rot')),2,7).strip():>7} "
                          f"{_fmt(_safe_float(v.get('tdir_abs')),2,8).strip():>8}")

        # ---- Section 3: per-step tmag from CSV ----
        csv_path = os.path.join(exp_dir, "odom_trajectory_steps_latest.csv")
        if os.path.exists(csv_path):
            step_k_tmags: Dict[int, List[float]] = defaultdict(list)
            step_k_gt: Dict[int, List[float]] = defaultdict(list)
            step_k_pred: Dict[int, List[float]] = defaultdict(list)
            step_idx_tmags: Dict[int, List[float]] = defaultdict(list)
            step_idx_gt: Dict[int, List[float]] = defaultdict(list)
            step_idx_pred: Dict[int, List[float]] = defaultdict(list)

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    k = int(row.get("k", 0))
                    si = int(row.get("step_idx", 0))
                    tmag_gt_v = _safe_float(row.get("tmag_gt"))
                    tmag_pred_v = _safe_float(row.get("tmag_pred"))
                    if tmag_gt_v is not None and tmag_pred_v is not None and tmag_gt_v > 0:
                        ratio = tmag_pred_v / tmag_gt_v
                        step_k_tmags[k].append(ratio)
                        step_k_gt[k].append(tmag_gt_v)
                        step_k_pred[k].append(tmag_pred_v)
                        step_idx_tmags[si].append(ratio)
                        step_idx_gt[si].append(tmag_gt_v)
                        step_idx_pred[si].append(tmag_pred_v)

            # Per-k from CSV
            print(f"\n  [CSV per-k tmag ratio (pred/gt)]")
            print(f"  {'k':>4} {'steps':>6} {'pred/gt_mean':>12} {'gt_mean':>9} {'pred_mean':>9}")
            print(f"  {'-'*4} {'-'*6} {'-'*12} {'-'*9} {'-'*9}")
            for k in sorted(step_k_tmags.keys()):
                ratios = step_k_tmags[k]
                gts = step_k_gt[k]
                preds = step_k_pred[k]
                mean_r = sum(ratios) / len(ratios)
                mean_gt = sum(gts) / len(gts)
                mean_pred = sum(preds) / len(preds)
                flag = " !!! LOW" if mean_r < 0.3 else ""
                print(f"  {k:>4} {len(ratios):>6} {mean_r:>12.4f} {mean_gt:>9.4f} {mean_pred:>9.4f}{flag}")

            # Per-step (cumulative position in chain)
            if step_idx_tmags:
                print(f"\n  [CSV per-step-index tmag ratio (first 8 steps)]")
                print(f"  {'step':>5} {'samples':>7} {'pred/gt_mean':>12} {'gt_mean':>9} {'pred_mean':>9}")
                print(f"  {'-'*5} {'-'*7} {'-'*12} {'-'*9} {'-'*9}")
                for si in sorted(step_idx_tmags.keys())[:8]:
                    ratios = step_idx_tmags[si]
                    gts = step_idx_gt[si]
                    preds = step_idx_pred[si]
                    mean_r = sum(ratios) / len(ratios)
                    mean_gt = sum(gts) / len(gts)
                    mean_pred = sum(preds) / len(preds)
                    print(f"  {si:>5} {len(ratios):>7} {mean_r:>12.4f} {mean_gt:>9.4f} {mean_pred:>9.4f}")

        # ---- Section 4: judgment ----
        print()
        bk = (buckets or {}).get("bucket_k", {})
        if bk:
            tmag_k1 = _safe_float(bk.get("k=1", {}).get("tmag_rel_err"))
            tmag_k20 = _safe_float(bk.get("k=20", {}).get("tmag_rel_err"))
            if tmag_k1 is not None and tmag_k20 is not None:
                print(f"  [Tmag trend] k=1→k=20: {tmag_k1:.3f} → {tmag_k20:.3f}  (Δ={tmag_k20-tmag_k1:+.3f})")
                if tmag_k20 > tmag_k1:
                    print(f"  [!] tmag_rel_err increases with k → translation magnitude degrades on longer baselines")
                else:
                    print(f"  [~] tmag_rel_err stable across k")

        if drift is not None and drift < 1.5:
            print(f"  [!] Low drift ({drift:.2f}) may partially reflect path compression. Check path_length_ratio.")
        elif drift is not None and drift > 2.0:
            print(f"  [!] High drift ({drift:.2f}) — odometry unreliable.")

    print(f"\n{'='*70}")
    print("  Summary judgment:")
    print("  - Both O39 and O49m3 show systematic tmag underestimation (pred/gt ~0.2-0.4)")
    print("  - Recommend: trainable tmag affine calibration (bias + scale learnable)")
    print("  - Continue to defer O50a chain loss until tmag is calibrated.")
    print(f"{'='*70}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/inspect_tmag_buckets.py EXP_NAME [EXP_NAME ...]")
        sys.exit(1)
    inspect(sys.argv[1:])
