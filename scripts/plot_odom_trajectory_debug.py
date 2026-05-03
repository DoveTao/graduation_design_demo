#!/usr/bin/env python3
"""Plot GT/pred trajectories from odom_trajectory_debug outputs.

Usage:
  python3 scripts/plot_odom_trajectory_debug.py --exp-dir DIR --out OUT
  python3 scripts/plot_odom_trajectory_debug.py --compare --exp-dir DIR1 DIR2 ... --out OUT
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_metrics(exp_dir: str) -> Tuple[float, float, float, float, float]:
    fs = os.path.join(exp_dir, "final_summary.json")
    if not os.path.exists(fs):
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    with open(fs, "r", encoding="utf-8") as f:
        data = json.load(f)

    last = data.get("last_eval", {}) if isinstance(data, dict) else {}
    drift = float(last.get("odom_metric_drift", float("nan"))) if isinstance(last, dict) else float("nan")
    ate = float(last.get("odom_metric_ATE", float("nan"))) if isinstance(last, dict) else float("nan")
    path_ratio = float(
        last.get("odom_shape_metric_mean_path_length_ratio", float("nan"))
        if isinstance(last, dict)
        else float("nan")
    )
    step_dir = float(
        last.get("odom_shape_metric_mean_step_dir_err_deg", float("nan"))
        if isinstance(last, dict)
        else float("nan")
    )
    mean_tmag_ratio = float(
        last.get("odom_debug_mean_tmag_ratio", float("nan"))
        if isinstance(last, dict)
        else float("nan")
    )
    return drift, ate, path_ratio, step_dir, mean_tmag_ratio


def _exp_display_name(exp_dir: str) -> str:
    name = os.path.basename(os.path.abspath(exp_dir))
    if name == "TRAJ_REPRO_T51a3b_upd400":
        return "T51a3b upd400"
    if name == "TRAJ_REPRO_T51a3b_upd500":
        return "T51a3b upd500"
    if name == "TRAJ_REPRO_T51a4_upd100":
        return "T51a4 upd100"
    if name == "TRAJ_REPRO_T51a4_final":
        return "T51a4 final"
    if name.startswith("TRAJ_REPRO_"):
        return name.replace("TRAJ_REPRO_", "", 1)
    return name


def _format_title(name: str, drift: float, ate: float, path_ratio: float) -> str:
    if any(np.isnan([drift, ate, path_ratio])):
        return name
    return f"{name} | drift={drift:.3f} | ATE={ate:.3f} | path={path_ratio:.3f}"



def _load_trajectory(exp_dir: str):
    npz_path = os.path.join(exp_dir, "odom_trajectory_debug_latest.npz")
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    gt = data["gt"]  # (T,3)
    pred = data["metric"]  # (T,3)
    if gt.ndim != 2 or pred.ndim != 2:
        return None
    return gt[:, :2], pred[:, :2]


def plot_one(exp_dir: str, out_path: str) -> None:
    traj = _load_trajectory(exp_dir)
    if traj is None:
        raise RuntimeError(f"missing/invalid trajectory in {exp_dir}")

    gt_xy, pred_xy = traj
    drift, ate, path_ratio, _step_dir, _mean_tmag_ratio = _load_metrics(exp_dir)

    plt.figure(figsize=(6, 6))
    plt.plot(gt_xy[:, 0], gt_xy[:, 1], label="GT", linewidth=2)
    plt.plot(pred_xy[:, 0], pred_xy[:, 1], label="Pred", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    name = _exp_display_name(exp_dir)
    plt.title(
        _format_title(name, drift, ate, path_ratio),
        fontsize=12,
    )
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_compare(exp_dirs: List[str], out_path: str) -> None:
    if len(exp_dirs) <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=False)
        axes = axes.reshape(-1)

        traj_data: List[Dict[str, object]] = []
        for exp_dir in exp_dirs:
            traj = _load_trajectory(exp_dir)
            if traj is None:
                traj_data.append({})
                continue
            gt_xy, pred_xy = traj
            drift, ate, path_ratio, _step_dir, _mean_tmag_ratio = _load_metrics(exp_dir)
            x_min = float(np.min([gt_xy[:, 0].min(), pred_xy[:, 0].min()]))
            x_max = float(np.max([gt_xy[:, 0].max(), pred_xy[:, 0].max()]))
            y_min = float(np.min([gt_xy[:, 1].min(), pred_xy[:, 1].min()]))
            y_max = float(np.max([gt_xy[:, 1].max(), pred_xy[:, 1].max()]))
            title = _format_title(_exp_display_name(exp_dir), drift, ate, path_ratio)
            traj_data.append({
                "ax_data": traj,
                "title": title,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            })

        valid = [t for t in traj_data if t]
        if not valid:
            raise RuntimeError("no valid trajectory data in compare set")

        g_xmin = min(t["x_min"] for t in valid)  # type: ignore[index]
        g_xmax = max(t["x_max"] for t in valid)  # type: ignore[index]
        g_ymin = min(t["y_min"] for t in valid)  # type: ignore[index]
        g_ymax = max(t["y_max"] for t in valid)  # type: ignore[index]
        x_pad = 0.05 * (g_xmax - g_xmin) if g_xmax > g_xmin else 1.0
        y_pad = 0.05 * (g_ymax - g_ymin) if g_ymax > g_ymin else 1.0

        handles = None
        labels = None
        for ax, data in zip(axes, traj_data):
            if not data:
                ax.axis("off")
                continue
            gt_xy, pred_xy = data["ax_data"]  # type: ignore[assignment]
            ax.plot(gt_xy[:, 0], gt_xy[:, 1], linestyle="--", alpha=0.6, label="GT")
            ax.plot(pred_xy[:, 0], pred_xy[:, 1], linewidth=2, label="Pred")
            ax.set_title(str(data["title"]), fontsize=11)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True, alpha=0.25)
            ax.axis("equal")
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
            ax.set_xlim(g_xmin - x_pad, g_xmax + x_pad)
            ax.set_ylim(g_ymin - y_pad, g_ymax + y_pad)
        for i in range(len(exp_dirs), len(axes)):
            axes[i].axis("off")
        if handles is not None and labels is not None:
            fig.legend(
                handles,
                labels,
                loc="upper right",
                bbox_to_anchor=(0.98, 0.98),
                ncol=2,
                frameon=False,
                fontsize=9,
            )
        fig.tight_layout(rect=(0, 0, 0.97, 0.97))
    else:
        plt.figure(figsize=(7, 7))
        for exp_dir in exp_dirs:
            traj = _load_trajectory(exp_dir)
            if traj is None:
                continue
            gt_xy, pred_xy = traj
            name = os.path.basename(exp_dir)
            plt.plot(gt_xy[:, 0], gt_xy[:, 1], linestyle="--", alpha=0.6, label=f"{name} GT")
            plt.plot(pred_xy[:, 0], pred_xy[:, 1], linewidth=2, label=f"{name} Pred")
        plt.title("Trajectory shape compare")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--exp-dir",
        nargs="+",
        required=True,
        help="One or more experiment dirs",
    )
    p.add_argument("--out", required=True, help="Output png path")
    p.add_argument("--compare", action="store_true", help="Overlay all provided dirs in one chart")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    exp_dirs = [d for d in args.exp_dir if d]
    if not exp_dirs:
        raise RuntimeError("at least one --exp-dir is required")

    if args.compare:
        plot_compare(exp_dirs, args.out)
    elif len(exp_dirs) == 1:
        plot_one(exp_dirs[0], args.out)
    else:
        raise RuntimeError("use --compare when providing multiple --exp-dir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
