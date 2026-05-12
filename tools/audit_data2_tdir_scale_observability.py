#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from s5e2_adjacent_dense_lib import relative_pose_A_to_B_in_B, scan_frames, write_json


ALLOWED_CLASSIFICATIONS = [
    "DATA2_LOW_MOTION_TDIR_UNOBSERVABLE",
    "DATA2_TRAIN_EVAL_MOTION_DISTRIBUTION_SHIFT",
    "DATA2_SCALE_ERROR_SMALL_MOTION_COUPLED",
    "DATA2_GEOMETRY_CONFIDENCE_NOT_RELIABLE",
    "DATA2_TDIR_AND_SCALE_OBSERVABILITY_LIMITED",
    "DATA2_NO_MAJOR_DATA_ISSUE_FOUND",
    "DATA2_INSUFFICIENT_ARTIFACTS",
    "DATA2_AUDIT_ERROR",
]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def _rot_angle_deg(R: np.ndarray) -> float:
    tr = float(np.trace(R))
    c = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    return float(np.rad2deg(np.arccos(c)))


def _seq_gt_motion(scene: str, seq: str) -> Dict[str, Any]:
    frames = scan_frames(Path("data"), scene=scene, seq=seq)[(scene, seq)]
    steps: List[float] = []
    rots: List[float] = []
    for i in range(len(frames) - 1):
        R, t = relative_pose_A_to_B_in_B(frames[i], frames[i + 1])
        steps.append(float(np.linalg.norm(t)))
        rots.append(_rot_angle_deg(R))
    per = _percentiles(steps)
    return {
        "num_poses": len(frames),
        "num_adjacent_edges": len(steps),
        "gt_step_median": float(np.median(steps)),
        "gt_step_mean": float(np.mean(steps)),
        "gt_step_p10": per["p10"],
        "gt_step_p25": per["p25"],
        "gt_step_p50": per["p50"],
        "gt_step_p75": per["p75"],
        "gt_step_p90": per["p90"],
        "gt_step_p95": per["p95"],
        "very_small_motion_fraction": float(np.mean(np.asarray(steps) < 0.002)),
        "small_motion_fraction": float(np.mean(np.asarray(steps) < 0.005)),
        "medium_motion_fraction": float(np.mean(np.asarray(steps) < 0.02)),
        "rotation_step_mean": float(np.mean(rots)),
        "rotation_step_p90": float(np.percentile(rots, 90)),
        "translation_rotation_ratio": float(np.mean(steps) / max(float(np.mean(rots)), 1.0e-12)),
        "path_length": float(np.sum(steps)),
        "gt_steps": steps,
    }


def _bin_name(step: float) -> str:
    if step < 0.002:
        return "lt_0p002"
    if step < 0.005:
        return "0p002_0p005"
    if step < 0.01:
        return "0p005_0p01"
    if step < 0.02:
        return "0p01_0p02"
    return "gte_0p02"


def _audit_by_bins(gt_steps: Sequence[float], rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    bins: Dict[str, List[Dict[str, Any]]] = {
        "lt_0p002": [],
        "0p002_0p005": [],
        "0p005_0p01": [],
        "0p01_0p02": [],
        "gte_0p02": [],
    }
    for step, row in zip(gt_steps, rows):
        bins[_bin_name(float(step))].append(row)
    out: Dict[str, Any] = {}
    for name, items in bins.items():
        if not items:
            out[name] = {"count": 0}
            continue
        tdir = np.asarray([float(x["tdir_deg"]) for x in items], dtype=np.float64)
        anti = np.asarray([1.0 if x["anti_parallel_flag"] else 0.0 for x in items], dtype=np.float64)
        tmag = np.asarray([float(x["tmag_ratio"]) for x in items], dtype=np.float64)
        pred = np.asarray([float(x["pred_step_length"]) for x in items], dtype=np.float64)
        gt = np.asarray([float(x["gt_step_length"]) for x in items], dtype=np.float64)
        out[name] = {
            "count": len(items),
            "tdir_mean_deg": float(np.mean(tdir)),
            "anti_parallel_rate": float(np.mean(anti)),
            "tmag_median_ratio": float(np.median(tmag)),
            "path_contribution_fraction": float(np.sum(pred) / max(float(np.sum([float(r["pred_step_length"]) for r in rows])), 1.0e-12)),
            "pred_step_mean": float(np.mean(pred)),
            "gt_step_mean": float(np.mean(gt)),
        }
    return out


def _top_edges(rows: Sequence[Dict[str, Any]], topk: int = 20, key: str = "pred_step_length") -> List[Dict[str, Any]]:
    pairs = sorted(enumerate(rows), key=lambda kv: float(kv[1][key]), reverse=True)[:topk]
    return [
        {
            "edge_index": int(i),
            key: float(r[key]),
            "gt_step_length": float(r["gt_step_length"]),
            "tmag_ratio": float(r["tmag_ratio"]),
            "tdir_deg": float(r["tdir_deg"]),
            "anti_parallel_flag": bool(r["anti_parallel_flag"]),
        }
        for i, r in pairs
    ]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        s5e15_ck = _read_json(Path(args.s5e15_checkpoint))
        s5e15_res = _read_json(Path(args.s5e15_results) / "edge_component_metrics.json")
        struct1b_ck = _read_json(Path(args.struct1b_checkpoint))
        struct1b_res = _read_json(Path(args.struct1b_results) / "edge_component_metrics.json")
        struct1_failure = _read_json(Path("checkpoints/STRUCT1_failure_mode_audit.json")) if Path("checkpoints/STRUCT1_failure_mode_audit.json").exists() else {}
    except Exception as exc:
        payload = {
            "experiment": "DATA2_tdir_scale_observability_and_split_shift_audit",
            "final_classification": "DATA2_AUDIT_ERROR",
            "error": str(exc),
            "s5_locked_metrics_policy_unchanged": True,
        }
        write_json(Path(args.out_json), payload)
        Path(args.out_report).write_text("# DATA2 审计失败\n", encoding="utf-8")
        return payload

    seq01 = _seq_gt_motion("scene01", "seq01")
    seq02 = _seq_gt_motion("scene01", "seq02")
    seq03 = _seq_gt_motion("scene01", "seq03")
    train_steps = seq01["gt_steps"] + seq02["gt_steps"]
    train_summary = {
        "gt_step_median": float(np.median(train_steps)),
        "small_motion_fraction": float(np.mean(np.asarray(train_steps) < 0.005)),
        "path_length": float(seq01["path_length"] + seq02["path_length"]),
    }
    gt_motion_distribution = {
        "scene01_seq01": {k: v for k, v in seq01.items() if k != "gt_steps"},
        "scene01_seq02": {k: v for k, v in seq02.items() if k != "gt_steps"},
        "scene01_seq03": {k: v for k, v in seq03.items() if k != "gt_steps"},
        "train_vs_eval_shift": {
            "gt_step_median_ratio_eval_over_train": float(seq03["gt_step_median"] / max(train_summary["gt_step_median"], 1.0e-12)),
            "small_motion_fraction_delta": float(seq03["small_motion_fraction"] - train_summary["small_motion_fraction"]),
            "path_length_ratio_eval_over_train": float(seq03["path_length"] / max(train_summary["path_length"], 1.0e-12)),
            "distribution_shift_suspected": bool(
                seq03["gt_step_median"] / max(train_summary["gt_step_median"], 1.0e-12) < 0.7
                or (seq03["small_motion_fraction"] - train_summary["small_motion_fraction"]) > 0.10
            ),
        },
    }

    s5_rows = s5e15_res["metrics_rows"]
    s1b_rows = struct1b_res["metrics_rows"]
    gt_steps_eval = seq03["gt_steps"]
    s5_bins = _audit_by_bins(gt_steps_eval, s5_rows)
    s1b_bins = _audit_by_bins(gt_steps_eval, s1b_rows)

    low_motion_tdir_unstable = bool(
        s5_bins["lt_0p002"].get("count", 0) > 0
        and s5_bins["lt_0p002"].get("tdir_mean_deg", 0.0) > s5_bins["gte_0p02"].get("tdir_mean_deg", 0.0) + 10.0
    ) or bool(
        s1b_bins["lt_0p002"].get("count", 0) > 0
        and s1b_bins["lt_0p002"].get("tdir_mean_deg", 0.0) > s1b_bins["gte_0p02"].get("tdir_mean_deg", 0.0) + 10.0
    )
    tdir_label_noise_suspected = bool(low_motion_tdir_unstable and gt_motion_distribution["train_vs_eval_shift"]["distribution_shift_suspected"])
    tdir_observability_by_gt_step = {
        "bins": {
            name: {
                "s5e15": s5_bins[name],
                "struct1b": s1b_bins[name],
            }
            for name in s5_bins
        },
        "low_motion_tdir_unstable": low_motion_tdir_unstable,
        "tdir_label_noise_suspected": tdir_label_noise_suspected,
    }

    struct1b_over_scale_small_motion_coupled = bool(
        s1b_bins["lt_0p002"].get("tmag_median_ratio", 0.0) > s1b_bins["gte_0p02"].get("tmag_median_ratio", 0.0) * 2.0
        or s1b_bins["0p002_0p005"].get("tmag_median_ratio", 0.0) > s1b_bins["gte_0p02"].get("tmag_median_ratio", 0.0) * 2.0
    )
    struct1b_pred = np.asarray([float(x["pred_step_length"]) for x in s1b_rows], dtype=np.float64)
    outlier_contrib = float(np.sum(np.sort(struct1b_pred)[-20:]) / max(float(np.sum(struct1b_pred)), 1.0e-12))
    path_error_outlier_dominated = bool(outlier_contrib > 0.75)
    global_scale_bias_suspected = bool(np.median([float(x["tmag_ratio"]) for x in s1b_rows]) > 3.0 and not path_error_outlier_dominated)
    scale_path_error_audit = {
        "s5e15": {
            "overall": s5e15_res["component_metrics"],
            "by_gt_step_bin": s5_bins,
            "top20_path_contributing_edges": _top_edges(s5_rows, key="pred_step_length"),
            "top20_tmag_ratio_edges": _top_edges(s5_rows, key="tmag_ratio"),
        },
        "struct1b": {
            "overall": struct1b_res["component_metrics"],
            "by_gt_step_bin": s1b_bins,
            "top20_path_contributing_edges": _top_edges(s1b_rows, key="pred_step_length"),
            "top20_tmag_ratio_edges": _top_edges(s1b_rows, key="tmag_ratio"),
            "top20_path_contribution_fraction": outlier_contrib,
        },
        "struct1b_over_scale_small_motion_coupled": struct1b_over_scale_small_motion_coupled,
        "path_error_outlier_dominated": path_error_outlier_dominated,
        "global_scale_bias_suspected": global_scale_bias_suspected,
    }

    train_best = struct1b_ck["training"]["best_run"]
    heldout = struct1b_ck["component_metrics"]
    likely_reason = "unknown"
    if gt_motion_distribution["train_vs_eval_shift"]["distribution_shift_suspected"]:
        likely_reason = "motion_distribution_shift"
    elif struct1b_over_scale_small_motion_coupled:
        likely_reason = "guard_too_narrow"
    train_eval_scale_gap = {
        "train_best": train_best,
        "heldout": {
            "tmag_median_ratio": heldout["tmag_median_ratio"],
            "tmag_p95_ratio": heldout["tmag_p95_ratio"],
            "path_ratio": heldout["path_ratio"],
            "signed_tdir_mean_deg": heldout["signed_tdir_mean_deg"],
        },
        "scale_generalization_failed": bool(
            heldout["tmag_median_ratio"] > train_best["tmag_median_ratio"] * 3.0
            or heldout["path_ratio"] > train_best["path_ratio"] * 2.0
        ),
        "likely_reason": likely_reason,
    }

    conf_audit_source = struct1_failure.get("direction_failure_audit", {})
    geometry_confidence_audit = {
        "confidence_available": bool(conf_audit_source),
        "confidence_correlates_with_tdir": None,
        "confidence_correlates_with_tmag": None,
        "high_confidence_reliable": None,
        "confidence_not_reliable": None,
    }
    if conf_audit_source:
        c_tdir = conf_audit_source.get("confidence_vs_tdir_spearman")
        c_tmag = conf_audit_source.get("confidence_vs_tmag_spearman")
        high_anti = conf_audit_source.get("high_conf_anti_parallel_rate")
        low_anti = conf_audit_source.get("low_conf_anti_parallel_rate")
        geometry_confidence_audit = {
            "confidence_available": True,
            "confidence_correlates_with_tdir": bool(c_tdir is not None and abs(float(c_tdir)) > 0.20),
            "confidence_correlates_with_tmag": bool(c_tmag is not None and abs(float(c_tmag)) > 0.20),
            "high_confidence_reliable": bool(high_anti is not None and low_anti is not None and float(high_anti) < float(low_anti)),
            "confidence_not_reliable": bool(high_anti is not None and low_anti is not None and float(high_anti) >= float(low_anti)),
        }

    if tdir_label_noise_suspected and struct1b_over_scale_small_motion_coupled:
        final = "DATA2_TDIR_AND_SCALE_OBSERVABILITY_LIMITED"
    elif gt_motion_distribution["train_vs_eval_shift"]["distribution_shift_suspected"]:
        final = "DATA2_TRAIN_EVAL_MOTION_DISTRIBUTION_SHIFT"
    elif low_motion_tdir_unstable:
        final = "DATA2_LOW_MOTION_TDIR_UNOBSERVABLE"
    elif struct1b_over_scale_small_motion_coupled:
        final = "DATA2_SCALE_ERROR_SMALL_MOTION_COUPLED"
    elif geometry_confidence_audit["confidence_not_reliable"]:
        final = "DATA2_GEOMETRY_CONFIDENCE_NOT_RELIABLE"
    else:
        final = "DATA2_NO_MAJOR_DATA_ISSUE_FOUND"

    recommendation = {
        "keep_s5e15_as_best_candidate": True,
        "continue_struct_line": False if final in {
            "DATA2_LOW_MOTION_TDIR_UNOBSERVABLE",
            "DATA2_TRAIN_EVAL_MOTION_DISTRIBUTION_SHIFT",
            "DATA2_SCALE_ERROR_SMALL_MOTION_COUPLED",
            "DATA2_GEOMETRY_CONFIDENCE_NOT_RELIABLE",
            "DATA2_TDIR_AND_SCALE_OBSERVABILITY_LIMITED",
        } else True,
        "do_struct1c": False if final != "DATA2_NO_MAJOR_DATA_ISSUE_FOUND" else True,
        "do_gen1_external_dataset": True if final in {
            "DATA2_LOW_MOTION_TDIR_UNOBSERVABLE",
            "DATA2_TRAIN_EVAL_MOTION_DISTRIBUTION_SHIFT",
            "DATA2_TDIR_AND_SCALE_OBSERVABILITY_LIMITED",
        } else False,
        "do_final8_freeze": True if final in {
            "DATA2_LOW_MOTION_TDIR_UNOBSERVABLE",
            "DATA2_SCALE_ERROR_SMALL_MOTION_COUPLED",
            "DATA2_GEOMETRY_CONFIDENCE_NOT_RELIABLE",
            "DATA2_TDIR_AND_SCALE_OBSERVABILITY_LIMITED",
        } else False,
        "main_next_step": (
            "GEN1_external_dataset_feasibility_or_final8"
            if final in {"DATA2_LOW_MOTION_TDIR_UNOBSERVABLE", "DATA2_TDIR_AND_SCALE_OBSERVABILITY_LIMITED"}
            else "GEN1_external_dataset_feasibility"
            if final == "DATA2_TRAIN_EVAL_MOTION_DISTRIBUTION_SHIFT"
            else "final8_freeze_and_write_negative_result"
            if final in {"DATA2_SCALE_ERROR_SMALL_MOTION_COUPLED", "DATA2_GEOMETRY_CONFIDENCE_NOT_RELIABLE"}
            else "consider_STRUCT1C_only_if_new_signal_source_exists"
        ),
    }

    payload = {
        "experiment": "DATA2_tdir_scale_observability_and_split_shift_audit",
        "gt_motion_distribution": gt_motion_distribution,
        "tdir_observability_by_gt_step": tdir_observability_by_gt_step,
        "scale_path_error_audit": scale_path_error_audit,
        "train_eval_scale_gap": train_eval_scale_gap,
        "geometry_confidence_audit": geometry_confidence_audit,
        "recommendation": recommendation,
        "s5_locked_metrics_policy_unchanged": True,
        "final_classification": final,
    }
    write_json(Path(args.out_json), payload)

    report = [
        "# DATA2 tdir / scale / observability / split shift 审计",
        "",
        "## 1. 为什么做 DATA2",
        "STRUCT1B 已证明 hard scale guard 能在 train-side 压住一部分 scale 爆炸，但 heldout seq03 仍明显失败，因此需要判断问题更像数据可观测性限制，还是模型本身仍有空间。",
        "",
        "## 2. train/eval GT motion distribution",
        json.dumps(gt_motion_distribution, ensure_ascii=False, indent=2),
        "",
        "## 3. tdir label stability by gt step",
        json.dumps(tdir_observability_by_gt_step, ensure_ascii=False, indent=2),
        "",
        "## 4. S5E15 vs STRUCT1B scale/path error",
        json.dumps(scale_path_error_audit, ensure_ascii=False, indent=2),
        "",
        "## 5. STRUCT1B train-side 与 heldout gap",
        json.dumps(train_eval_scale_gap, ensure_ascii=False, indent=2),
        "",
        "## 6. confidence / entropy 是否可靠",
        json.dumps(geometry_confidence_audit, ensure_ascii=False, indent=2),
        "",
        "## 7. 数据可观测性对 tdir/path_ratio 的影响",
        f"final_classification={final}",
        "",
        "## 8. 是否继续 STRUCT 路线",
        json.dumps(recommendation, ensure_ascii=False, indent=2),
        "",
        "## 9. 是否需要外部数据集泛化",
        "如果 low-motion / split shift 被确认是主因，则更值得做 GEN1 external dataset feasibility，而不是继续在 adjacent tdir 上细调主干。",
        "",
        "## 10. caveats",
        "- 本审计不训练新模型。",
        "- 本审计不刷新 official S5 locked metrics。",
        "- confidence 相关结论主要来自已存在的 STRUCT1A 审计产物；STRUCT1B 当前提交产物中不包含 raw provenance。",
    ]
    Path(args.out_report).write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    p.add_argument("--s5e15-checkpoint", required=True)
    p.add_argument("--s5e15-results", required=True)
    p.add_argument("--struct1b-checkpoint", required=True)
    p.add_argument("--struct1b-results", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
