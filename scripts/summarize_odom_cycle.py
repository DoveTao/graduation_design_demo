#!/usr/bin/env python3
"""Summarize and rank odometry experiment cycles."""

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

O39_EXP = "O39_c31_smallk_anchor3_tmag_nodetach_1200"
O39_CKPT = os.path.join("checkpoints", O39_EXP, "best_smallk_odom.pt")

O45_STAGE1_EXPERIMENTS = [
    "O45a_scale_affine_probe_150",
    "O45b_seqturn_pair_probe_150",
    "O45c_seqturn_chain_probe_150",
    "O45d_curriculum_k1_probe_150",
]

O46_SCALES = (1.03, 1.05, 1.08, 1.10)
O46_BIASES = (0.00, 0.02, 0.03, 0.05)
O49_FOCUSED_EXPS = (
    "O49m",
    "O49m2",
    "O49m3",
    "O49s",
    "O49t",
    "O49u",
    "O49v",
    "O49h",
    "O49h2",
    "O49n",
    "O49o",
    "O49p",
    "O49q",
    "O49r",
    "O49f",
    "O49g",
    "O49i",
)


@dataclass
class Thresholds:
    max_tdir_abs: float = 25.0
    max_local_a_abs: float = 25.0
    max_smallk_tdir_regress: float = 0.5
    min_path_delta: float = 0.005
    min_path_ratio: float = 0.0
    max_tmag_rel_delta: float = 0.03
    max_turn_abs_err_delta: float = 0.2
    min_drift_delta: float = 0.005
    max_drift_regress: float = 0.08


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


def _tag_float(value: float) -> str:
    return f"{value:.2f}".replace(".", "")


def _affine_grid_name(scale: float, bias: float) -> str:
    return f"O46g_affine_s{_tag_float(scale)}_b{_tag_float(bias)}"


def _arm_for(exp_name: str) -> str:
    if exp_name.startswith("O45a_"):
        return "scale_affine"
    if exp_name.startswith("O45b_"):
        return "seqturn_pair"
    if exp_name.startswith("O45c_"):
        return "seqturn_chain"
    if exp_name.startswith("O45d_"):
        return "k1_curriculum"
    if exp_name.startswith("O46g_affine_"):
        return "affine_grid_eval"
    if exp_name.startswith("O46t_affine_"):
        return "affine_short_train"
    if exp_name.startswith("O47a_"):
        return "affine_anchor4_guard"
    if exp_name.startswith("O47b_"):
        return "affine_lrhalf_guard"
    if exp_name.startswith("O48a_"):
        return "affine_pair_turn"
    if exp_name.startswith("O48b_"):
        return "affine_chain_guard"
    if exp_name.startswith("O49m"):
        return "seqturn_pair_main"
    if exp_name.startswith("O49s"):
        return "seqturn_pair_robust3"
    if exp_name.startswith("O49t"):
        return "seqturn_pair_avgpool"
    if exp_name.startswith("O49u"):
        return "seqturn_pair_nomemattn"
    if exp_name.startswith("O49v"):
        return "seqturn_pair_avgpool_nomemattn"
    if exp_name.startswith("O49h"):
        return "seqturn_pair_higher_w"
    if exp_name.startswith("O49n"):
        return "seqturn_pair_lower_w"
    if exp_name.startswith("O49o"):
        return "seqturn_pair_clamp2deg"
    if exp_name.startswith("O49p"):
        return "acos_eps_5e-5"
    if exp_name.startswith("O49q"):
        return "acos_eps_2e-4"
    if exp_name.startswith("O49r"):
        return "deterministic"
    if exp_name.startswith("O49f"):
        return "seqturn_soft_hitrate"
    if exp_name.startswith("O49g"):
        return "seqturn_low_w"
    if exp_name.startswith("O49i"):
        return "seqturn_late_start"
    return "unknown"


def _stage_for(exp_name: str) -> str:
    if exp_name.startswith("O45"):
        return "O45"
    if exp_name.startswith("O46g_affine_"):
        return "O46_grid"
    if exp_name.startswith("O46t_affine_"):
        return "O46_short_train"
    if exp_name.startswith("O47"):
        return "O47_guard"
    if exp_name.startswith("O48"):
        return "O48_structure"
    if exp_name.startswith("O49"):
        return "O49_seqturn"
    return "unknown"


def _affine_params_for(exp_name: str) -> Dict[str, Optional[float]]:
    parts = exp_name.split("_")
    scale = None
    bias = None
    for part in parts:
        if part.startswith("s") and len(part) == 4 and part[1:].isdigit():
            scale = float(f"{part[1]}.{part[2:]}")
        if part.startswith("b") and len(part) == 4 and part[1:].isdigit():
            bias = float(f"{part[1]}.{part[2:]}")
    return {"affine_scale": scale, "affine_bias": bias}


def _debug_summary_path(exp_name: str) -> str:
    return os.path.join(CHECKPOINTS, f"E_{exp_name}_trajectory_debug", "final_summary.json")


def _experiment_names(cycle: str) -> List[str]:
    if cycle == "O45":
        names = list(O45_STAGE1_EXPERIMENTS)
        for path in sorted(glob.glob(os.path.join(CHECKPOINTS, "O45*_cont250"))):
            if os.path.isdir(path):
                names.append(os.path.basename(path))
        for path in sorted(glob.glob(os.path.join(CHECKPOINTS, "O45_winner_cont300*"))):
            if os.path.isdir(path):
                names.append(os.path.basename(path))
        return list(dict.fromkeys(names))
    if cycle == "O49":
        names = []
        for path in sorted(glob.glob(os.path.join(CHECKPOINTS, "O49*"))):
            if not os.path.isdir(path):
                continue
            name = os.path.basename(path)
            if "_trajectory_debug" in name:
                continue
            if name.startswith("O49"):
                names.append(name)
        known = []
        for prefix in O49_FOCUSED_EXPS:
            for name in names:
                if name.startswith(prefix):
                    known.append(name)
        names = known + [name for name in names if name not in known]
        return list(dict.fromkeys(names))

    names = [_affine_grid_name(scale, bias) for scale in O46_SCALES for bias in O46_BIASES]
    for pattern in (
        "O46t_affine_*_100",
        "O47a_affine_anchor4_150",
        "O47b_affine_anchor3_lrhalf_150",
        "O48a_affine_plus_pair_turn_120",
        "O48b_affine_plus_chain_guard_120",
    ):
        for path in sorted(glob.glob(os.path.join(CHECKPOINTS, pattern))):
            if os.path.isdir(path):
                names.append(os.path.basename(path))
    for pattern in ("E_O46g_affine_*_trajectory_debug", "E_O46t_affine_*_trajectory_debug"):
        for path in sorted(glob.glob(os.path.join(CHECKPOINTS, pattern))):
            if os.path.isdir(path):
                name = os.path.basename(path)
                names.append(name[2:-17])
    return list(dict.fromkeys(names))


def _experiment_names_old() -> List[str]:
    names = list(O45_STAGE1_EXPERIMENTS)
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
        "run_device": str(train.get("run_device", "unknown")),
        "run_device_name": str(train.get("run_device_name", "unknown")),
        "skip_updates": _first_float(train.get("skip_updates")),
        "total_updates": _first_float(train.get("total_updates")),
        "best_odom_drift": _first_float(
            train.get("best_odom_drift"),
            source_eval.get("odom_metric_drift"),
        ),
        "best_odom_tdir_abs": _first_float(
            train.get("best_odom_tdir_abs"),
            source_eval.get("tdir_abs"),
            train.get("best_tdir_abs"),
        ),
        "best_odom_tmag_rel_err": _first_float(
            train.get("best_odom_tmag_rel_err"),
            source_eval.get("tmag_rel_err"),
        ),
        "best_odom_select_score": _first_float(train.get("best_odom_select_score")),
        "best_odom_select_points": _first_float(train.get("best_odom_select_points")),
        "best_odom_update": _first_float(train.get("best_odom_upd")),
        "best_odom_checkpoint": train.get("best_odom_checkpoint", ""),
        "best_smallk_odom_tdir_abs": _first_float(
            train.get("best_smallk_odom_tdir_abs"),
            source_eval.get("tdir_abs"),
            train.get("best_tdir_abs"),
        ),
        "best_smallk_odom_drift": _first_float(
            train.get("best_smallk_odom_drift"),
            source_eval.get("odom_metric_drift"),
        ),
        "best_smallk_odom_tmag_rel_err": _first_float(
            train.get("best_smallk_odom_tmag_rel_err"),
            source_eval.get("tmag_rel_err"),
        ),
        "best_smallk_odom_select_score": _first_float(train.get("best_smallk_odom_select_score")),
        "best_smallk_odom_select_points": _first_float(train.get("best_smallk_odom_select_points")),
        "best_smallk_odom_update": _first_float(train.get("best_smallk_odom_upd")),
        "best_smallk_odom_checkpoint": train.get("best_smallk_odom_checkpoint", ""),
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
    params = _affine_params_for(exp_name)
    checkpoint = _best_ckpt(exp_name)
    for key in ("best_smallk_odom_checkpoint", "best_odom_checkpoint", "best_joint_local_A_abs.pt", "best_joint.pt"):
        if isinstance(train, dict):
            candidate = str(train.get(key, "")).strip()
            if candidate:
                checkpoint = os.path.join(CHECKPOINTS, exp_name, candidate)
                break
    if exp_name.startswith("O46g_affine_") and (train or debug):
        checkpoint = O39_CKPT
    metrics.update(
        {
            "experiment": exp_name,
            "arm": _arm_for(exp_name),
            "stage": _stage_for(exp_name),
            "train_summary": bool(train),
            "debug_summary": bool(debug),
            "checkpoint": checkpoint,
            **params,
        }
    )
    return metrics


def _baseline_row() -> Dict[str, Any]:
    train = _json_load(os.path.join(CHECKPOINTS, O39_EXP, "final_summary.json"))
    debug = _json_load(os.path.join(CHECKPOINTS, "E_O39_trajectory_debug", "final_summary.json"))
    metrics = _metrics_from(train, debug)
    metrics.update(
        {
            "experiment": "O39_baseline",
            "arm": "baseline",
            "stage": "baseline",
            "train_summary": bool(train),
            "debug_summary": bool(debug),
            "checkpoint": O39_CKPT,
            "affine_scale": None,
            "affine_bias": None,
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
    path_up = isinstance(row.get("path_length_ratio"), float) and (
        (
            isinstance(base_path, float)
            and row["path_length_ratio"] >= base_path + thresholds.min_path_delta
        )
        or (
            thresholds.min_path_ratio > 0.0
            and row["path_length_ratio"] >= thresholds.min_path_ratio
        )
    )
    tmag_ok = (
        isinstance(row.get("tmag_rel_err"), float)
        and isinstance(base_tmag, float)
        and row["tmag_rel_err"] <= base_tmag + thresholds.max_tmag_rel_delta
    )
    drift_ok = not (
        isinstance(row.get("drift"), float)
        and isinstance(base_drift, float)
        and row["drift"] > base_drift + thresholds.max_drift_regress
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
    if not drift_ok:
        score -= 3
        reasons.append("drift_regress")
    if (
        isinstance(row.get("turn_sum_abs_err"), float)
        and isinstance(base_turn, float)
        and row["turn_sum_abs_err"] > base_turn + thresholds.max_turn_abs_err_delta
    ):
        score -= 3
        reasons.append("turn_worse")

    eligible = bool(direction_ok and path_up and tmag_ok and drift_ok)
    row.update(
        {
            "direction_ok": direction_ok,
            "path_up": path_up,
            "tmag_ok": tmag_ok,
            "drift_ok": drift_ok,
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
        "stage",
        "arm",
        "run_device",
        "affine_scale",
        "affine_bias",
        "score",
        "decision",
        "best_odom_drift",
        "best_smallk_odom_drift",
        "best_odom_select_score",
        "best_smallk_odom_select_score",
        "best_odom_select_points",
        "best_smallk_odom_select_points",
        "best_odom_update",
        "best_smallk_odom_update",
        "skip_updates",
        "total_updates",
        "tdir_abs",
        "local_A_abs",
        "best_smallk_odom_tdir_abs",
        "best_odom_tdir_abs",
        "drift",
        "scale_fit_drift",
        "dtcalib_drift",
        "path_length_ratio",
        "turn_sum_abs_err",
        "tmag_rel_err",
        "reasons",
    ]
    with open(path, "w", encoding="utf-8") as f:
        if "O49" in os.path.basename(path):
            title = "O49 Seq-Turn Coarse-stage Comparison"
        elif "O46" in os.path.basename(path):
            title = "O46/O47/O48 Eight-Hour Odometry Cycle Results"
        else:
            title = "O45 Six-Hour Odometry Cycle Results"
        f.write(f"# {title}\n\n")
        f.write("## Baseline\n\n")
        f.write("| metric | value |\n|---|---:|\n")
        for key in (
            "run_device",
            "tdir_abs",
            "local_A_abs",
            "best_odom_drift",
            "best_smallk_odom_drift",
            "best_odom_select_score",
            "best_smallk_odom_select_score",
            "best_smallk_odom_tdir_abs",
            "drift",
            "scale_fit_drift",
            "dtcalib_drift",
            "path_length_ratio",
            "turn_sum_abs_err",
            "tmag_rel_err",
        ):
            f.write(f"| {key} | {_fmt(baseline.get(key))} |\n")
        f.write("\n## Ranking\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(_fmt(row.get(h)) for h in headers) + " |\n")
        f.write("\n## Selected\n\n")
        if selected:
            for idx, row in enumerate(selected, 1):
                f.write(f"{idx}. `{row['experiment']}` ({row['stage']}/{row['arm']}), score={row['score']}, ckpt=`{row.get('checkpoint') or '-'}`\n")
        else:
            f.write("No branch passed the gates. Keep O39 as the mainline.\n")
        f.write("\n## CPU/GPU Comparison\n\n")
        by_device: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            key = str(row.get("run_device", "unknown"))
            by_device.setdefault(key, []).append(row)
        f.write("| device | count | best_odom_drift(min) | best_smallk_odom_drift(min) |\n")
        f.write("|---|---:|---:|---:|\n")
        for key, group in sorted(by_device.items()):
            best_odom = min(
                (r.get("best_odom_drift") for r in group if isinstance(r.get("best_odom_drift"), (int, float))),
                default=float("nan"),
            )
            best_smallk = min(
                (r.get("best_smallk_odom_drift") for r in group if isinstance(r.get("best_smallk_odom_drift"), (int, float))),
                default=float("nan"),
            )
            f.write(f"| {key} | {len(group)} | {_fmt(best_odom)} | {_fmt(best_smallk)} |\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", choices=("O45", "O46_O48", "O49"), default="O45")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--state-json", default="")
    parser.add_argument("--selection-txt", default="")
    parser.add_argument(
        "--selection-format",
        choices=("legacy", "rich"),
        default="legacy",
        help="legacy: experiment/arm/checkpoint (default), rich: append extra columns at end",
    )
    parser.add_argument("--top-n", type=int, default=2)
    args = parser.parse_args()

    if not args.output_md:
        if args.cycle == "O49":
            args.output_md = os.path.join(CHECKPOINTS, "O49_seqturn_cycle_results.md")
        elif args.cycle == "O46_O48":
            args.output_md = os.path.join(CHECKPOINTS, "O46_O48_eight_hour_cycle_results.md")
        else:
            args.output_md = os.path.join(CHECKPOINTS, "O45_six_hour_cycle_results.md")

    if not args.state_json:
        if args.cycle == "O49":
            args.state_json = os.path.join(CHECKPOINTS, "O49_seqturn_cycle_state.json")
        elif args.cycle == "O46_O48":
            args.state_json = os.path.join(CHECKPOINTS, "O46_O48_eight_hour_cycle_state.json")
        else:
            args.state_json = os.path.join(CHECKPOINTS, "O45_six_hour_cycle_state.json")

    if not args.selection_txt:
        if args.cycle == "O49":
            args.selection_txt = os.path.join(CHECKPOINTS, "O49_seqturn_cycle_selection.txt")
        elif args.cycle == "O46_O48":
            args.selection_txt = os.path.join(CHECKPOINTS, "O46_O48_eight_hour_cycle_selection.txt")
        else:
            args.selection_txt = os.path.join(CHECKPOINTS, "O45_six_hour_cycle_selection.txt")

    thresholds = Thresholds()
    if args.cycle == "O46_O48":
        thresholds.min_path_ratio = 1.30
    baseline = _baseline_row()
    rows = [_evaluate(_row(name), baseline, thresholds) for name in _experiment_names(args.cycle)]
    rows = [r for r in rows if r["train_summary"] or r["debug_summary"]]
    rows.sort(key=lambda r: (r["eligible"], r["score"], r.get("path_length_ratio") or -1.0), reverse=True)

    selected = [r for r in rows if r["eligible"] and r.get("checkpoint")][: args.top_n]
    _write_markdown(args.output_md, baseline, rows, selected)

    os.makedirs(os.path.dirname(args.state_json), exist_ok=True)
    state = {
        "cycle": args.cycle,
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
            if args.selection_format == "rich":
                f.write(
                    f"{row['experiment']}\t{row['arm']}\t{row.get('checkpoint') or ''}"
                    f"\t{row['stage']}\t{_fmt(row.get('affine_scale'))}\t{_fmt(row.get('affine_bias'))}\n"
                )
            else:
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
