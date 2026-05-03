#!/usr/bin/env python3
"""Offline selector for shape-aware checkpoint ranking.

Inputs:
  - One or more experiment directories.
  - Reads <exp_dir>/eval_history.json (expects dict with key 'history').

Outputs:
  - checkpoints/shape_aware_selection_summary.csv
  - checkpoints/shape_aware_selection_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional


CSV_PATH_DEFAULT = os.path.join("checkpoints", "shape_aware_selection_summary.csv")
MD_PATH_DEFAULT = os.path.join("checkpoints", "shape_aware_selection_summary.md")

FIELDS = [
    "exp",
    "upd",
    "checkpoint",
    "drift",
    "length_norm_drift",
    "ATE",
    "tdir_abs",
    "tmag_rel_err",
    "path_ratio",
    "step_dir_err",
    "odom_debug_mean_tmag_ratio",
    "score_drift",
    "score_shape045_l1",
    "score_shape055_l2",
    "score_balanced",
    "eligible_shape",
]


@dataclass
class Row:
    exp: str
    upd: int
    checkpoint: str
    drift: Optional[float]
    length_norm_drift: Optional[float]
    ate: Optional[float]
    tdir_abs: Optional[float]
    tmag_rel_err: Optional[float]
    path_ratio: Optional[float]
    step_dir_err: Optional[float]
    odom_debug_mean_tmag_ratio: Optional[float]
    score_drift: Optional[float]
    score_shape045_l1: Optional[float]
    score_shape055_l2: Optional[float]
    score_balanced: Optional[float]
    eligible_shape: bool


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isfinite(f):
        return f
    return None


def _normalize_exp_name(exp_name: str) -> str:
    if exp_name.startswith("TRAJ_REPRO_"):
        return exp_name[len("TRAJ_REPRO_"):]
    return exp_name


def _infer_exp_meta(exp_name: str) -> tuple[str, Optional[int], str]:
    normalized = _normalize_exp_name(exp_name)
    m = re.match(r"^(T51a(?:3b|4))_(upd(\d+)|final)$", normalized)
    if m:
        branch = m.group(1)
        upd_tag = m.group(2)
        if upd_tag == "final":
            return branch, None, upd_tag
        return branch, int(upd_tag[3:]), upd_tag
    return normalized, None, ""


def _fmt(v: Optional[float], prec: int = 6) -> str:
    if v is None:
        return ""
    return f"{v:.{prec}f}".rstrip("0").rstrip(".")


def _resolve_exp_dir(path: str) -> str:
    if os.path.isabs(path):
        return path
    candidate = os.path.join("checkpoints", path)
    if os.path.isdir(candidate):
        return candidate
    return path


def _select_ckpt(exp_dir: str, upd: int) -> str:
    candidate = os.path.join(exp_dir, f"eval_upd_{upd:04d}.pt")
    if os.path.exists(candidate):
        return candidate
    return ""


def _extract_rows(exp_dir: str) -> List[Row]:
    eval_json = os.path.join(exp_dir, "eval_history.json")
    if not os.path.exists(eval_json):
        return []

    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history") if isinstance(data, dict) else None
    if not isinstance(history, list):
        return []

    exp_name = os.path.basename(os.path.abspath(exp_dir))
    _, parsed_upd, _ = _infer_exp_meta(exp_name)
    rows: List[Row] = []

    for entry in history:
        if not isinstance(entry, dict):
            continue

        upd = entry.get("upd")
        if not isinstance(upd, int):
            continue
        final_upd = upd
        if parsed_upd is not None and parsed_upd > 0:
            final_upd = parsed_upd

        drift = _safe_float(entry.get("odom_metric_drift"))
        length_norm_drift = _safe_float(entry.get("odom_metric_length_normalized_drift"))
        ate = _safe_float(entry.get("odom_metric_ATE"))
        tdir_abs = _safe_float(entry.get("tdir_abs"))
        tmag_rel_err = _safe_float(entry.get("tmag_rel_err"))

        path_ratio = _safe_float(entry.get("odom_shape_metric_mean_path_length_ratio"))
        if path_ratio is None:
            path_ratio = _safe_float(entry.get("odom_shape_metric_path_weighted_path_length_ratio"))

        step_dir_err = _safe_float(entry.get("odom_shape_metric_mean_step_dir_err_deg"))
        if step_dir_err is None:
            step_dir_err = _safe_float(entry.get("odom_shape_metric_path_weighted_step_dir_err_deg"))

        odom_debug_mean_tmag_ratio = _safe_float(entry.get("odom_debug_mean_tmag_ratio"))

        eligible = (
            path_ratio is not None
            and path_ratio >= 0.35
            and tdir_abs is not None
            and tdir_abs <= 25.0
            and tmag_rel_err is not None
            and tmag_rel_err <= 1.0
            and drift is not None
            and drift <= 1.6
        )

        if drift is not None and path_ratio is not None and ate is not None and step_dir_err is not None:
            score_drift = drift
            score_shape045_l1 = drift + 0.1 * ate + max(0.0, 0.45 - path_ratio)
            score_shape055_l2 = drift + 0.1 * ate + 2.0 * max(0.0, 0.55 - path_ratio)
            score_balanced = drift + 0.1 * ate + abs(path_ratio - 0.55) + 0.01 * step_dir_err
        else:
            score_drift = drift
            score_shape045_l1 = None
            score_shape055_l2 = None
            score_balanced = None

        rows.append(
            Row(
                exp=exp_name,
                upd=final_upd,
                checkpoint=_select_ckpt(exp_dir, upd),
                drift=drift,
                length_norm_drift=length_norm_drift,
                ate=ate,
                tdir_abs=tdir_abs,
                tmag_rel_err=tmag_rel_err,
                path_ratio=path_ratio,
                step_dir_err=step_dir_err,
                odom_debug_mean_tmag_ratio=odom_debug_mean_tmag_ratio,
                score_drift=score_drift,
                score_shape045_l1=score_shape045_l1,
                score_shape055_l2=score_shape055_l2,
                score_balanced=score_balanced,
                eligible_shape=bool(eligible),
            )
        )

    return rows


def _score_sort_key(row: Row, metric: str) -> float:
    val = getattr(row, metric)
    if val is None:
        return float("inf")
    return val


def _to_summary_row(r: Row) -> str:
    chk = os.path.basename(r.checkpoint) if r.checkpoint else "-"
    return (
        f"| {r.exp} | {r.upd} | `{chk}` | {_fmt(r.drift)} | {_fmt(r.length_norm_drift)} | {_fmt(r.ate)} | "
        f"{_fmt(r.tdir_abs)} | {_fmt(r.tmag_rel_err)} | {_fmt(r.path_ratio)} | {_fmt(r.step_dir_err)} | {_fmt(r.odom_debug_mean_tmag_ratio)} | "
        f"{_fmt(r.score_drift)} | {_fmt(r.score_shape045_l1)} | {_fmt(r.score_shape055_l2)} | {_fmt(r.score_balanced)} | {str(r.eligible_shape)} |"
    )


def generate_outputs(rows: List[Row], csv_path: str, md_path: str) -> None:
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    rows_sorted = sorted(rows, key=lambda x: (x.exp, x.upd))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for r in rows_sorted:
            w.writerow(
                [
                    r.exp,
                    r.upd,
                    r.checkpoint,
                    _fmt(r.drift),
                    _fmt(r.length_norm_drift),
                    _fmt(r.ate),
                    _fmt(r.tdir_abs),
                    _fmt(r.tmag_rel_err),
                    _fmt(r.path_ratio),
                    _fmt(r.step_dir_err),
                    _fmt(r.odom_debug_mean_tmag_ratio),
                    _fmt(r.score_drift),
                    _fmt(r.score_shape045_l1),
                    _fmt(r.score_shape055_l2),
                    _fmt(r.score_balanced),
                    int(r.eligible_shape),
                ]
            )

    best_drift = sorted(rows_sorted, key=lambda x: _score_sort_key(x, "score_drift"))[:3]
    best_shape = sorted(
        [r for r in rows_sorted if r.eligible_shape],
        key=lambda x: _score_sort_key(x, "score_shape055_l2"),
    )[:3]

    recommended_mainline = next(
        (r for r in rows_sorted if _infer_exp_meta(r.exp)[0].startswith("T51a3b") and r.upd == 400),
        None,
    )
    recommended_drift = next(
        (r for r in rows_sorted if _infer_exp_meta(r.exp)[0].startswith("T51a3b") and r.upd == 500),
        None,
    )
    if recommended_drift is None:
        recommended_drift = best_drift[0] if best_drift else None
    recommended_non = next((r for r in rows_sorted if _infer_exp_meta(r.exp)[0].startswith("T51a4")), None)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Shape-aware selection summary\n\n")
        f.write("## 输入\n")
        f.write("- 离线读取各实验目录下 `eval_history.json` 并对每个 eval 点打分。\n\n")

        f.write("## 全量评估点\n\n")
        f.write("| exp | upd | checkpoint | drift | length_norm_drift | ATE | tdir_abs | tmag_rel_err | path_ratio | step_dir_err | odom_debug_mean_tmag_ratio | score_drift | score_shape045_l1 | score_shape055_l2 | score_balanced | eligible_shape |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows_sorted:
            f.write(_to_summary_row(r) + "\n")

        f.write("\n## best drift top-3\n\n")
        f.write("| exp | upd | drift | score_drift |\n")
        f.write("|---|---:|---:|---:|\n")
        for r in best_drift:
            f.write(f"| {r.exp} | {r.upd} | {_fmt(r.drift)} | {_fmt(r.score_drift)} |\n")

        f.write("\n## shape-aware eligible top-3 (score_shape055_l2)\n\n")
        f.write("| exp | upd | path_ratio | score_shape055_l2 | score_shape045_l1 | score_balanced |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in best_shape:
            f.write(
                f"| {r.exp} | {r.upd} | {_fmt(r.path_ratio)} | {_fmt(r.score_shape055_l2)} | {_fmt(r.score_shape045_l1)} | {_fmt(r.score_balanced)} |\n"
            )

        f.write("\n## 推荐\n\n")
        if recommended_mainline is not None:
            f.write(
                f"- 主线 checkpoint（shape-aware）：`{recommended_mainline.exp}` upd={recommended_mainline.upd}"
                f" (`{recommended_mainline.checkpoint}`)\n"
            )
        if recommended_drift is not None:
            f.write(
                f"- best-drift 对照：`{recommended_drift.exp}` upd={recommended_drift.upd}"
                f" (`{recommended_drift.checkpoint}`)\n"
            )
        if recommended_non is not None:
            f.write(
                f"- 不推荐：`{recommended_non.exp}` upd={recommended_non.upd}（`{os.path.basename(recommended_non.checkpoint or '')}`）\n"
                "  - 该分支 `path_ratio` 持续不足（小于 0.35），轨迹长度保真退化，作为主线或对照均不优先。\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="offline shape-aware checkpoint selector")
    parser.add_argument(
        "experiments",
        nargs="+",
        help="experiment dir(s), e.g. checkpoints/T51a3b_from_... checkpoints/T51a4_from_...",
    )
    parser.add_argument("--csv", default=CSV_PATH_DEFAULT, help="output CSV path")
    parser.add_argument("--md", default=MD_PATH_DEFAULT, help="output markdown path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_rows: List[Row] = []
    for exp in args.experiments:
        exp_dir = _resolve_exp_dir(exp)
        all_rows.extend(_extract_rows(exp_dir))

    generate_outputs(all_rows, args.csv, args.md)

    print(f"Wrote CSV: {args.csv}")
    print(f"Wrote MD: {args.md}")
    print(f"Rows: {len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
