#!/usr/bin/env python3
"""
File: scripts/inspect_o49_shape_debug.py
Description:
    Read-only trajectory shape diagnostic inspector for O49/O50 experiments.
    Reads odom_trajectory_debug_latest.json from eval-only or training outputs
    and prints clean per-experiment shape metrics for comparison.

Usage:
    python3 scripts/inspect_o49_shape_debug.py E_O49m_seed2024_shape_debug
    python3 scripts/inspect_o49_shape_debug.py E_O49m_seed2024_shape_debug /path/to/other

Notes:
    - No training, no model loading, read-only.
    - Works with any experiment dir containing odom_trajectory_debug_latest.json.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


CHECKPOINTS = os.path.join(os.path.dirname(__file__), "..", "checkpoints")


def _read_traj_debug(exp_dir: str) -> Optional[Dict[str, Any]]:
    """Read odom_trajectory_debug_latest.json from an experiment directory."""
    path = os.path.join(exp_dir, "odom_trajectory_debug_latest.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_final_summary(exp_dir: str) -> Optional[Dict[str, Any]]:
    """Read final_summary.json from an experiment directory."""
    path = os.path.join(exp_dir, "final_summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_dir(name: str) -> str:
    """Resolve experiment name to absolute directory path."""
    if os.path.isabs(name) and os.path.isdir(name):
        return name
    candidate = os.path.join(CHECKPOINTS, name)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(f"Experiment directory not found: {name} (tried {candidate})")


def _read_odom_metrics(exp_dir: str) -> Optional[Dict[str, Any]]:
    """Read odom_metrics_latest.json from an experiment directory."""
    path = os.path.join(exp_dir, "odom_metrics_latest.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def inspect(exp_names: List[str]) -> None:
    """Print shape diagnostic table for given experiments."""
    rows: List[Dict[str, Any]] = []

    for name in exp_names:
        exp_dir = _resolve_dir(name)
        debug = _read_traj_debug(exp_dir)
        fsummary = _read_final_summary(exp_dir)
        odom = _read_odom_metrics(exp_dir)

        row: Dict[str, Any] = {"experiment": name}

        # Basic metrics from final_summary + odom_metrics
        if fsummary:
            row["drift"] = _safe_float(
                fsummary.get("best_odom_drift")
                or fsummary.get("odom_metric_drift")
                or (odom.get("odom_metric_drift") if odom else None)
            )
            row["device"] = fsummary.get("run_device", "?")
            row["tdir_abs"] = _safe_float(fsummary.get("best_tdir_abs"))
            le = fsummary.get("last_eval", {}) if isinstance(fsummary.get("last_eval"), dict) else {}
            row["path_length_ratio"] = _safe_float(le.get("odom_shape_metric_mean_path_length_ratio"))

        # Per-chain shape from trajectory debug
        chains = debug.get("chains", []) if debug else []
        summary = debug.get("summary", {}) if debug else {}

        row["num_chains"] = len(chains)

        if chains:
            # Aggregate across all chains
            all_metrics = []
            for ch in chains:
                sm = ch.get("shape_metric", {})
                if sm:
                    all_metrics.append(sm)

            if all_metrics:
                row["path_length_ratio"] = _safe_float(
                    sum(m.get("path_length_ratio", 0) for m in all_metrics) / len(all_metrics)
                )
                row["pred_path_len"] = _safe_float(
                    sum(m.get("pred_path_length", 0) for m in all_metrics) / len(all_metrics)
                )
                row["gt_path_len"] = _safe_float(
                    sum(m.get("gt_path_length", 0) for m in all_metrics) / len(all_metrics)
                )
                row["pred_turn_sum"] = _safe_float(
                    sum(m.get("pred_turn_sum_deg", 0) for m in all_metrics) / len(all_metrics)
                )
                row["gt_turn_sum"] = _safe_float(
                    sum(m.get("gt_turn_sum_deg", 0) for m in all_metrics) / len(all_metrics)
                )
                row["turn_sum_abs_err"] = _safe_float(
                    sum(m.get("turn_sum_abs_err_deg", 0) for m in all_metrics) / len(all_metrics)
                )
                row["mean_step_dir_err"] = _safe_float(
                    sum(m.get("mean_step_dir_err_deg", 0) for m in all_metrics) / len(all_metrics)
                )
                row["mean_turn_err"] = _safe_float(
                    sum(m.get("mean_turn_abs_err_deg", 0) for m in all_metrics) / len(all_metrics)
                )
                row["straightness_err"] = _safe_float(
                    sum(m.get("straightness_abs_err", 0) for m in all_metrics) / len(all_metrics)
                )
                row["num_steps"] = sum(m.get("num_steps", 0) for m in all_metrics)
        else:
            # Fallback to summary-level metrics
            row["path_length_ratio"] = _safe_float(summary.get("metric_mean_path_length_ratio"))
            row["mean_step_dir_err"] = _safe_float(summary.get("metric_mean_step_dir_err_deg"))
            row["mean_turn_err"] = _safe_float(summary.get("metric_mean_turn_abs_err_deg"))
            row["straightness_err"] = _safe_float(summary.get("metric_mean_straightness_abs_err"))

        rows.append(row)

    if not rows:
        print("No experiments found.")
        return

    # Print header
    header = (
        f"{'Experiment':<50} {'Drift':>8} {'Dev':>5} {'TdirAbs':>8} "
        f"{'PathR':>7} {'PredL':>7} {'GtL':>7} "
        f"{'PTurn':>7} {'GTurn':>7} {'TurnErr':>8} "
        f"{'StepDir':>7} {'MeanTurn':>8} {'StrErr':>7} {'Steps':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        def fmt(v, prec=1):
            if v is None:
                return "-"
            return f"{v:.{prec}f}"

        def fmt4(v):
            return fmt(v, 4)

        print(
            f"{r['experiment']:<50} "
            f"{fmt4(r.get('drift')):>8} "
            f"{str(r.get('device', '?')):>5} "
            f"{fmt(r.get('tdir_abs')):>8} "
            f"{fmt4(r.get('path_length_ratio')):>7} "
            f"{fmt(r.get('pred_path_len')):>7} "
            f"{fmt(r.get('gt_path_len')):>7} "
            f"{fmt(r.get('pred_turn_sum')):>7} "
            f"{fmt(r.get('gt_turn_sum')):>7} "
            f"{fmt(r.get('turn_sum_abs_err')):>8} "
            f"{fmt(r.get('mean_step_dir_err')):>7} "
            f"{fmt(r.get('mean_turn_err')):>8} "
            f"{fmt4(r.get('straightness_err')):>7} "
            f"{str(r.get('num_steps', '-')):>6}"
        )

    # Summary judgment
    print()
    for r in rows:
        plr = r.get("path_length_ratio")
        drift = r.get("drift")
        if plr is not None and plr < 0.5:
            print(f"[!] {r['experiment']}: path_length_ratio={plr:.3f} < 0.5 "
                  f"→ model severely underestimates translation magnitude (pred_path << gt_path)")
        if drift is not None and drift > 2.0:
            print(f"[!] {r['experiment']}: drift={drift:.2f} > 2.0 → odometry may be unreliable")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/inspect_o49_shape_debug.py EXP_NAME [EXP_NAME ...]")
        print("Example:")
        print("  python3 scripts/inspect_o49_shape_debug.py E_O49m_seed2024_shape_debug")
        sys.exit(1)
    inspect(sys.argv[1:])
