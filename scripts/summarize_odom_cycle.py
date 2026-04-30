#!/usr/bin/env python3
"""Summarize and rank the O45 six-hour odometry cycle."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINTS = os.path.join(ROOT, "checkpoints")

STAGE1_EXPERIMENTS = [
    "O45a_scale_affine_probe_150",
    "O45b_seqturn_pair_probe_150",
    "O45c_seqturn_chain_probe_150",
    "O45d_curriculum_k1_probe_150",
]


@dataclass
class Thresholds:
    max_tdir_abs: float = 25.0
    max_local_a_abs: float = 25.0
    max_smallk_tdir_regress: float = 0.5
    min_path_delta: float = 0.005
    max_tmag_rel_delta: float = 0.03
    max_turn_abs_err_delta: float = 0.2
    min_drift_delta: float = 0.005


def _json_load(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _finite(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def _get_nested(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _first_float(*values: Any) -> Optional[float]:
    for value in values:
        finite = _finite(value)
        if finite is not None:
            return finite
    return None


def _best_ckpt(exp_name: str) -> str:
    exp_dir = os.path.join(CHECKPOINTS, exp_name)
    for filename in ("best_smallk_odom.pt", "best_joint_local_A_abs.pt", "best_joint.pt"):
        path = os.path.join(exp_dir, filename)
        if os.path.exists(path):
            return path
    return ""


def _arm_for(exp_name: str) -> str:
    if exp_name.startswith("O45a_"):
        return "scale_affine"
    if exp_name.startswith("O45b_"):
        return "seqturn_pair"
    if exp_name.startswith("O45c_"):
        return "seqturn_chain"
    if exp_name.startswith("O45d_"):
        return "k1_curriculum"
    return "unknown"


def _debug_summary_path(exp_name: str) -> str:
    return os.path.join(CHECKPOINTS, f"E_{exp_name}_trajectory_debug", "final_summary.json")


def _experiment_names() -> List[str]:
    names = list(STAGE1_EXPERIMENTS)
    for path in sorted(glob.glob(os.path.join(CHECKPOINTS, "O45*_cont250"))):
        if os.path.isdir(path):
            names.append(os.path.basename(path))
    for path in sorted(glob.glob(os.path.join(CHECKPOINTS, "O45_winner_cont300*"))):
        if os.path.isdir(path):
            names.append(os.path.basename(path))
    return list(dict.fromkeys(names))


def _metrics_from(train: Dict[str, Any], debug: Dict[str, Any]) -> Dict[str, Any]:
    last_eval = train.get("last_eval") if isinstance(train.get("last_eval"), dict) else {}
    debug_last = debug.get("last_eval") if isinstance(debug.get("last_eval"), dict) else {}
    source_eval = debug_last or last_eval

    return {
        "tdir_abs": _first_float(source_eval.get("tdir_abs"), train.get("best_tdir_abs")),
        "local_A_abs": _first_float(source_eval.get("tdir_local_A_abs"), train.get("best_tdir_local_A_abs")),
        "best_smallk_odom_tdir_abs": _first_float(
            train.get("best_smallk_odom_tdir_abs"),
            source_eval.get("tdir_abs"),
            train.get("best_tdir_abs"),
        ),
        "best_smallk_odom_tmag_rel_err": _first_float(
            train.get("best_smallk_odom_tmag_rel_err"),
            source_eval.get("tmag_rel_err"),
        ),
        "tmag_rel_err": _first_float(
            source_eval.get("tmag_rel_err"),
            source_eval.get("odom_debug_mean_tmag_rel_err"),
            _get_nested(debug, ("summary", "mean_tmag_rel_err")),
        ),
        "drift": _first_float(source_eval.get("odom_metric_drift")),
        "scale_fit_drift": _first_float(source_eval.get("odom_metric_scale_fit_drift")),
        "dtcalib_drift": _first_float(source_eval.get("odom_metric_dtcalib_drift")),
        "path_length_ratio": _first_float(
            source_eval.get("odom_shape_metric_mean_path_length_ratio"),
            _get_nested(debug, ("summary", "metric_mean_path_length_ratio")),
        ),
        "path_weighted_ratio": _first_float(
            source_eval.get("odom_shape_metric_path_weighted_path_length_ratio"),
            _get_nested(debug, ("summary", "metric_path_weighted_path_length_ratio")),
        ),
        "turn_sum_abs_err": _first_float(
            source_eval.get("odom_shape_metric_mean_turn_abs_err_deg"),
            _get_nested(debug, ("summary", "metric_mean_turn_abs_err_deg")),
        ),
        "ate": _first_float(source_eval.get("odom_metric_ATE")),
        "checkpoint": "",
    }


def _row(exp_name: str) -> Dict[str, Any]:
    train = _json_load(os.path.join(CHECKPOINTS, exp_name, "final_summary.json"))
    debug = _json_load(_debug_summary_path(exp_name))
    metrics = _metrics_from(train, debug)
    metrics.update(
        {
            "experiment": exp_name,
            "arm": _arm_for(exp_name),
            "train_summary": bool(train),
            "debug_summary": bool(debug),
            "checkpoint": _best_ckpt(exp_name),
        }
    )
    return metrics


def _baseline_row() -> Dict[str, Any]:
    train = _json_load(os.path.join(CHECKPOINTS, "O39_c31_smallk_anchor3_tmag_nodetach_1200", "final_summary.json"))
    debug = _json_load(os.path.join(CHECKPOINTS, "E_O39_trajectory_debug", "final_summary.json"))
    metrics = _metrics_from(train, debug)
    metrics.update(
        {
            "experiment": "O39_baseline",
            "arm": "baseline",
            "train_summary": bool(train),
            "debug_summary": bool(debug),
            "checkpoint": os.path.join(
                "checkpoints",
                "O39_c31_smallk_anchor3_tmag_nodetach_1200",
                "best_smallk_odom.pt",
            ),
        }
    )
    return metrics


def _gap(row: Dict[str, Any]) -> Optional[float]:
    drift = row.get("drift")
    alternatives = [row.get("scale_fit_drift"), row.get("dtcalib_drift")]
    alternatives = [v for v in alternatives if isinstance(v, float)]
    if not isinstance(drift, float) or not alternatives:
        return None
    return min(abs(drift - alt) for alt in alternatives)


def _evaluate(row: Dict[str, Any], baseline: Dict[str, Any], thresholds: Thresholds) -> Dict[str, Any]:
    base_smallk = baseline.get("best_smallk_odom_tdir_abs")
    if not isinstance(base_smallk, float):
        base_smallk = baseline.get("tdir_abs")

    base_path = baseline.get("path_length_ratio")
    base_tmag = baseline.get("tmag_rel_err")
    base_turn = baseline.get("turn_sum_abs_err")
    base_drift = baseline.get("drift")
    base_gap = _gap(baseline)
    row_gap = _gap(row)

    direction_ok = (
        isinstance(row.get("tdir_abs"), float)
        and row["tdir_abs"] <= thresholds.max_tdir_abs
        and isinstance(row.get("local_A_abs"), float)
        and row["local_A_abs"] <= thresholds.max_local_a_abs
        and isinstance(row.get("best_smallk_odom_tdir_abs"), float)
        and isinstance(base_smallk, float)
        and row["best_smallk_odom_tdir_abs"] <= base_smallk + thresholds.max_smallk_tdir_regress
    )
    path_up = (
        isinstance(row.get("path_length_ratio"), float)
        and isinstance(base_path, float)
        and row["path_length_ratio"] >= base_path + thresholds.min_path_delta
    )
    tmag_ok = (
        isinstance(row.get("tmag_rel_err"), float)
        and isinstance(base_tmag, float)
        and row["tmag_rel_err"] <= base_tmag + thresholds.max_tmag_rel_delta
    )

    score = 0
    reasons: List[str] = []
    if direction_ok:
        score += 1
        reasons.append("direction_ok")
    else:
        reasons.append("direction_fail")
    if path_up:
        score += 3
        reasons.append("path_up")
    else:
        reasons.append("path_not_up")
    if isinstance(row.get("drift"), float) and isinstance(base_drift, float) and row["drift"] <= base_drift - thresholds.min_drift_delta:
        score += 2
        reasons.append("drift_down")
    if isinstance(row_gap, float) and isinstance(base_gap, float) and row_gap <= base_gap - thresholds.min_drift_delta:
        score += 1
        reasons.append("scale_gap_shrink")
    if not tmag_ok:
        score -= 2
        reasons.append("tmag_worse_or_missing")
    if (
        isinstance(row.get("turn_sum_abs_err"), float)
        and isinstance(base_turn, float)
        and row["turn_sum_abs_err"] > base_turn + thresholds.max_turn_abs_err_delta
    ):
        score -= 3
        reasons.append("turn_worse")

    eligible = bool(direction_ok and path_up and tmag_ok)
    row.update(
        {
            "direction_ok": direction_ok,
            "path_up": path_up,
            "tmag_ok": tmag_ok,
            "eligible": eligible,
            "score": score,
            "decision": "continue" if eligible else "reject",
            "reasons": ",".join(reasons),
        }
    )
    return row


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    text = str(value)
    return text if text else "-"


def _write_markdown(path: str, baseline: Dict[str, Any], rows: List[Dict[str, Any]], selected: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    headers = [
        "experiment",
        "arm",
        "score",
        "decision",
        "tdir_abs",
        "local_A_abs",
        "best_smallk_odom_tdir_abs",
        "drift",
        "scale_fit_drift",
        "dtcalib_drift",
        "path_length_ratio",
        "turn_sum_abs_err",
        "tmag_rel_err",
        "reasons",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# O45 Six-Hour Odometry Cycle Results\n\n")
        f.write("## Baseline\n\n")
        f.write("| metric | value |\n|---|---:|\n")
        for key in headers[4:13]:
            f.write(f"| {key} | {_fmt(baseline.get(key))} |\n")
        f.write("\n## Ranking\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(_fmt(row.get(h)) for h in headers) + " |\n")
        f.write("\n## Selected\n\n")
        if selected:
            for idx, row in enumerate(selected, 1):
                f.write(f"{idx}. `{row['experiment']}` ({row['arm']}), score={row['score']}, ckpt=`{row.get('checkpoint') or '-'}`\n")
        else:
            f.write("No branch passed the gates. Keep O39 as the mainline.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-md", default=os.path.join(CHECKPOINTS, "O45_six_hour_cycle_results.md"))
    parser.add_argument("--state-json", default=os.path.join(CHECKPOINTS, "O45_six_hour_cycle_state.json"))
    parser.add_argument("--selection-txt", default=os.path.join(CHECKPOINTS, "O45_six_hour_cycle_selection.txt"))
    parser.add_argument("--top-n", type=int, default=2)
    args = parser.parse_args()

    thresholds = Thresholds()
    baseline = _baseline_row()
    rows = [_evaluate(_row(name), baseline, thresholds) for name in _experiment_names()]
    rows = [r for r in rows if r["train_summary"] or r["debug_summary"]]
    rows.sort(key=lambda r: (r["eligible"], r["score"], r.get("path_length_ratio") or -1.0), reverse=True)

    selected = [r for r in rows if r["eligible"] and r.get("checkpoint")][: args.top_n]
    _write_markdown(args.output_md, baseline, rows, selected)

    os.makedirs(os.path.dirname(args.state_json), exist_ok=True)
    state = {
        "baseline": baseline,
        "rows": rows,
        "selected": selected,
        "commit_recommended": bool(selected),
        "output_md": args.output_md,
    }
    with open(args.state_json, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, allow_nan=True)
        f.write("\n")

    with open(args.selection_txt, "w", encoding="utf-8") as f:
        for row in selected:
            f.write(f"{row['experiment']}\t{row['arm']}\t{row.get('checkpoint') or ''}\n")

    print(f"Wrote {args.output_md}")
    if selected:
        print("Selected:")
        for row in selected:
            print(f"  {row['experiment']} score={row['score']} ckpt={row.get('checkpoint')}")
    else:
        print("No branch passed gates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
