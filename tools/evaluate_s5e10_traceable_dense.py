#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, load_or_base_checkpoint, read_json, validation_from_logs, write_json


S5E9 = {
    "signed_tdir_mean": 53.65013421168108,
    "anti_parallel_rate": 0.10596026490066225,
    "tdir_abs_mean": 48.92733872256603,
    "tmag_p95": 1.6560804642823275,
    "tmag_max": 7.078435706785601,
    "path_ratio": 0.05272511562778335,
    "pair_order_dir_flip_success_rate": 0.24061810154525387,
}


def _ext(traj: Path, gt: Path, out_json: Path, mode: str) -> Dict[str, Any]:
    subprocess.run(
        [
            "/home/dovetao/miniconda3/envs/pytorch/bin/python",
            "tools/evaluate_external_baseline_trajectory.py",
            "--trajectory",
            str(traj),
            "--groundtruth",
            str(gt),
            "--alignment",
            mode,
            "--output-json",
            str(out_json),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    d = read_json(out_json)
    return {"ate": d.get("ATE"), "drift": d.get("drift"), "path_ratio": d.get("path_ratio")}


def _top_and_long(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    pred = np.asarray([r[field].get("pred_step_length") or 0.0 for r in rows], dtype=np.float64)
    tmag = np.asarray([r[field].get("tmag_ratio") or -1.0 for r in rows], dtype=np.float64)
    idx = np.argsort(tmag)[-20:]
    top20 = float(pred[idx].sum() / max(pred.sum(), 1.0e-12))
    thr = float(np.percentile(tmag[tmag >= 0], 95))
    longest = cur = 0
    best = (0, 0)
    for i, f in enumerate(tmag > thr):
        if f:
            cur += 1
            if cur > longest:
                longest = cur
                best = (i - cur + 1, i)
        else:
            cur = 0
    return {
        "top20_tmag_path_fraction": top20,
        "long_run_path_fraction": float(pred[best[0]: best[1] + 1].sum() / max(pred.sum(), 1.0e-12)),
    }


def _small_motion(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    gt = np.asarray([r[field].get("gt_step_length") or 0.0 for r in rows], dtype=np.float64)
    tmag = np.asarray([r[field].get("tmag_ratio") or 0.0 for r in rows], dtype=np.float64)
    tdir = np.asarray([r[field].get("tdir_deg") or 0.0 for r in rows], dtype=np.float64)
    pred = np.asarray([r[field].get("pred_step_length") or 0.0 for r in rows], dtype=np.float64)
    thr = float(np.percentile(gt, 25))
    small = gt <= thr
    return {
        "small_motion_count": int(np.sum(small)),
        "small_motion_fraction": float(np.mean(small)),
        "small_motion_gt_tmag_median": float(np.percentile(gt[small], 50)) if np.any(small) else None,
        "small_motion_gt_tmag_p95": float(np.percentile(gt[small], 95)) if np.any(small) else None,
        "small_motion_pred_tmag_median": float(np.percentile(tmag[small], 50)) if np.any(small) else None,
        "small_motion_pred_tmag_p95": float(np.percentile(tmag[small], 95)) if np.any(small) else None,
        "small_motion_tdir_mean": float(np.mean(tdir[small])) if np.any(small) else None,
        "small_motion_tdir_p90": float(np.percentile(tdir[small], 90)) if np.any(small) else None,
        "small_motion_signed_tdir_mean": float(np.mean(tdir[small])) if np.any(small) else None,
        "small_motion_signed_tdir_p90": float(np.percentile(tdir[small], 90)) if np.any(small) else None,
        "small_motion_anti_parallel_rate": float(np.mean([1.0 if r[field].get("anti_parallel_flag") else 0.0 for r in rows if (r[field].get("gt_step_length") or 0.0) <= thr])) if np.any(small) else None,
        "small_motion_path_fraction": float(np.sum(pred[small]) / max(np.sum(pred), 1.0e-12)) if np.any(small) else None,
        "small_motion_error_contribution": float(np.mean(np.abs(tdir[small])) / max(np.mean(np.abs(tdir)), 1.0e-12)) if np.any(small) else None,
        "small_motion_amplification": bool(np.mean(small & (tmag > np.percentile(tmag, 90))) > 0.03),
    }


def _report(path: Path, ckpt: Dict[str, Any]) -> None:
    lines = [
        "# S5E10 order-aware signed-direction scale recalibration report",
        "",
        "## 最终分类",
        str(ckpt.get("final_classification")),
        "",
        "## 训练状态",
        str(ckpt.get("training")),
        "",
        "## 与 S5E9 的关系",
        "S5E10 在 S5E9 已修复 raw scale explosion 的基础上，继续检查 signed direction 的 pair-order 可观测性，并尝试恢复过度保守的 scale/path。",
        "",
        "## pair-order architecture audit",
        str(ckpt.get("pair_order_architecture_audit")),
        "",
        "## 是否发现 order-invariant / label flip / augmentation / export bug",
        str({
            "order_invariant_path_suspected": ckpt.get("pair_order_architecture_audit", {}).get("order_invariant_path_suspected"),
            "pair_order_label_flip_consistent": ckpt.get("pair_order_architecture_audit", {}).get("pair_order_label_flip_consistent"),
            "reversed_pair_export_consistent": ckpt.get("pair_order_architecture_audit", {}).get("reversed_pair_export_consistent"),
        }),
        "",
        "## order-aware signed direction candidate",
        str(ckpt.get("order_aware_signed_direction_candidate")),
        "",
        "## reversed-pair diagnostic",
        str(ckpt.get("pair_order_diagnostic")),
        "",
        "## signed direction reliability mask",
        "S5E10 对 near_static 和 small_motion 降低 signed direction loss 权重，对 normal_motion / large_motion 保留完整 signed loss。",
        "",
        "## axis + sign decomposition 结果",
        str(ckpt.get("signed_direction_candidate_metrics")),
        "",
        "## scale recalibration 结果",
        str(ckpt.get("scale_recalibrated_candidate")),
        "",
        "## small-motion / near-static bucket audit",
        str(ckpt.get("small_motion_audit")),
        "",
        "## raw prediction metrics",
        str(ckpt.get("raw_prediction_metrics")),
        "",
        "## guarded prediction metrics",
        str(ckpt.get("guarded_prediction_metrics")),
        "",
        "## raw vs guarded gap",
        str(ckpt.get("raw_vs_guarded_gap")),
        "",
        "## external evaluator results",
        str({"guarded": ckpt.get("external_eval"), "raw": ckpt.get("raw_external_eval")}),
        "",
        "## comparison vs S5E9/S5E8/S5E7/S5E6/S5E5/S5E4/S5E3/S5E2",
        str(ckpt.get("comparison_summary")),
        "",
        "## comparison vs ORB-SLAM3",
        str(ckpt.get("comparison_to_orbslam3")),
        "",
        "## blockers",
        str(ckpt.get("blockers")),
        "",
        "## validation results",
        str(ckpt.get("validation")),
        "",
        "## caveats",
        "- S5E10 是实验性结果，不替换 S5 locked。",
        "- 只有 raw signed direction 和 raw scale/path 同时改善，且 raw vs guarded gap 保持很小，才可称为真实几何改善。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    out_dir = Path(args.out_dir)
    metrics = read_json(out_dir / "edge_component_metrics.json")
    rows = [json.loads(line) for line in Path(args.provenance).read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_rows = [{"metric_preview": r.get("metric_preview_raw") or {}} for r in rows]
    guard_rows = [{"metric_preview": r.get("metric_preview") or {}} for r in rows]
    raw = dict(metrics.get("raw_prediction_metrics", {})); raw.update(_top_and_long(raw_rows, "metric_preview"))
    guard = dict(metrics.get("guarded_prediction_metrics", {})); guard.update(_top_and_long(guard_rows, "metric_preview"))
    ckpt = load_or_base_checkpoint(Path(args.out_json))
    ckpt["raw_prediction_metrics"] = raw
    ckpt["guarded_prediction_metrics"] = guard
    ckpt["raw_component_metrics"] = raw
    ckpt["guarded_component_metrics"] = guard
    ckpt["component_metrics"] = guard
    ckpt["small_motion_audit"] = _small_motion(guard_rows, "metric_preview")
    audit_path = out_dir / "pair_order_observability_audit.json"
    if audit_path.exists():
        ckpt["pair_order_architecture_audit"] = read_json(audit_path)
    scale_status = Path("checkpoints/S5E10_order_aware_signed_direction_scale_recalibration_candidate/scale_recalibration_training_status.json")
    if scale_status.exists():
        ckpt["scale_recalibrated_candidate"] = read_json(scale_status)
    signed_status = Path("checkpoints/S5E10_order_aware_signed_direction_scale_recalibration_candidate/order_signed_direction_training_status.json")
    if signed_status.exists():
        ckpt["order_aware_signed_direction_candidate"] = read_json(signed_status)
        if ckpt["order_aware_signed_direction_candidate"].get("history"):
            ckpt["signed_direction_candidate_metrics"] = ckpt["order_aware_signed_direction_candidate"]["history"][-1]["val_metrics"]
    scale_payload = ckpt.get("scale_recalibrated_candidate") or {}
    signed_payload = ckpt.get("order_aware_signed_direction_candidate") or {}
    ckpt["training"] = {
        "attempted": bool(scale_payload.get("attempted") or signed_payload.get("attempted")),
        "classification": " / ".join(
            [x for x in [scale_payload.get("classification"), signed_payload.get("classification")] if x]
        ) or None,
        "checkpoint_dir": "checkpoints/S5E10_order_aware_signed_direction_scale_recalibration_candidate",
        "training_status": {
            "scale_recalibrated_candidate": str(scale_status),
            "order_aware_signed_direction_candidate": str(signed_status),
        },
    }
    scale_audit = Path(out_dir / "scale_pipeline_audit.json")
    if scale_audit.exists():
        ckpt["scale_pipeline_audit"] = read_json(scale_audit)
    ckpt["pair_order_diagnostic"] = {
        "pair_order_dir_flip_cosine_mean": metrics.get("pair_order_dir_flip_cosine_mean"),
        "pair_order_dir_flip_success_rate": metrics.get("pair_order_dir_flip_success_rate"),
        "pair_order_scale_symmetry_error": metrics.get("pair_order_scale_symmetry_error"),
        "pair_order_failure_rate": metrics.get("pair_order_failure_rate"),
        "signed_direction_order_observable": metrics.get("signed_direction_order_observable"),
    }

    ext, raw_ext = {}, {}
    for mode in ["none", "se3", "sim3"]:
        ext[mode] = _ext(Path(args.trajectory), Path(args.groundtruth), out_dir / f"eval_alignment_{mode}.json", mode)
        raw_ext[mode] = _ext(Path(metrics["raw_trajectory_path"]), Path(args.groundtruth), out_dir / f"eval_alignment_raw_{mode}.json", mode)
    ckpt["external_eval"] = ext
    ckpt["raw_external_eval"] = raw_ext
    ckpt["raw_vs_guarded_gap"] = {
        "tdir_gap": None if raw.get("tdir_mean_deg") is None else raw.get("tdir_mean_deg") - guard.get("tdir_mean_deg"),
        "tdir_abs_gap": None if raw.get("tdir_abs_mean_deg") is None else raw.get("tdir_abs_mean_deg") - guard.get("tdir_abs_mean_deg"),
        "tmag_p95_gap": None if raw.get("tmag_p95_ratio") is None else raw.get("tmag_p95_ratio") - guard.get("tmag_p95_ratio"),
        "tmag_max_gap": None if raw.get("tmag_max_ratio") is None else raw.get("tmag_max_ratio") - guard.get("tmag_max_ratio"),
        "path_ratio_gap": None if raw.get("path_ratio") is None else raw.get("path_ratio") - guard.get("path_ratio"),
        "sim3_ATE_gap": None if raw_ext["sim3"]["ate"] is None else raw_ext["sim3"]["ate"] - ext["sim3"]["ate"],
    }

    comp = {
        "pair_order_dir_flip_success_rate_improved": (metrics.get("pair_order_dir_flip_success_rate") or 0.0) > S5E9["pair_order_dir_flip_success_rate"] + 0.1,
        "signed_tdir_mean_improved_vs_s5e9": (raw.get("tdir_mean_deg") or 999) < S5E9["signed_tdir_mean"] - 2.0,
        "anti_parallel_rate_improved_vs_s5e9": (raw.get("anti_parallel_rate") or 999) < S5E9["anti_parallel_rate"] - 0.01,
        "scale_recalibrated_vs_s5e9": (raw.get("path_ratio") or 999) > S5E9["path_ratio"] + 0.05 and (raw.get("tmag_p95_ratio") or 999) < 10.0,
        "raw_vs_guarded_gap_small": abs((ckpt["raw_vs_guarded_gap"]["path_ratio_gap"] or 0.0)) < 0.5 and abs((ckpt["raw_vs_guarded_gap"]["tmag_p95_gap"] or 0.0)) < 5.0,
    }
    ckpt["comparison_summary"] = comp
    ckpt["comparison_to_orbslam3"] = {
        "coverage": "S5E10 454/454 vs ORB-SLAM3 273/454",
        "edge_level_gap": "unavailable",
        "se3_ate_gap": None if ext["se3"]["ate"] is None else ext["se3"]["ate"] - ORBSLAM3_REFERENCE["se3"]["ate"],
        "sim3_ate_gap": None if ext["sim3"]["ate"] is None else ext["sim3"]["ate"] - ORBSLAM3_REFERENCE["sim3"]["ate"],
        "path_ratio": f"S5E10={guard.get('path_ratio')}; ORB-SLAM3={ORBSLAM3_REFERENCE['se3']['path_ratio']}",
    }
    blockers = []
    if not comp["pair_order_dir_flip_success_rate_improved"]:
        blockers.append("pair-order 可观测性仍然不足，dir flip success rate 没有明显起来。")
    if not comp["signed_tdir_mean_improved_vs_s5e9"]:
        blockers.append("raw signed direction 没有明显优于 S5E9。")
    if not comp["scale_recalibrated_vs_s5e9"]:
        blockers.append("raw path_ratio 虽有变化，但没有形成理想的 scale recalibration。")
    if not comp["raw_vs_guarded_gap_small"]:
        blockers.append("raw vs guarded gap 又被拉大了。")
    ckpt["blockers"] = blockers

    audit = ckpt.get("pair_order_architecture_audit", {})
    if audit.get("order_invariant_path_suspected") or not audit.get("pair_order_label_flip_consistent", True) or not audit.get("reversed_pair_export_consistent", True):
        final = "S5E10_ORDER_INVARIANT_BUG_FOUND"
    elif comp["pair_order_dir_flip_success_rate_improved"] and metrics.get("signed_direction_order_observable"):
        if comp["signed_tdir_mean_improved_vs_s5e9"] and comp["anti_parallel_rate_improved_vs_s5e9"] and comp["scale_recalibrated_vs_s5e9"] and comp["raw_vs_guarded_gap_small"]:
            final = "S5E10_RAW_GEOMETRY_IMPROVED"
        else:
            final = "S5E10_PAIR_ORDER_OBSERVABLE"
    elif comp["signed_tdir_mean_improved_vs_s5e9"] and comp["anti_parallel_rate_improved_vs_s5e9"]:
        final = "S5E10_SIGNED_DIRECTION_IMPROVED"
    elif comp["scale_recalibrated_vs_s5e9"] and comp["raw_vs_guarded_gap_small"]:
        final = "S5E10_SCALE_RECALIBRATED"
    elif ckpt.get("small_motion_audit", {}).get("small_motion_amplification") is False:
        final = "S5E10_SMALL_MOTION_IMPROVED"
    elif not comp["raw_vs_guarded_gap_small"]:
        final = "S5E10_GUARD_DEPENDENT"
    else:
        final = "S5E10_INPUT_GEOMETRY_INSUFFICIENT"
    ckpt["validation"] = validation_from_logs()
    ckpt["final_classification"] = final
    write_json(Path(args.out_json), ckpt)
    _report(Path(args.out_report), ckpt)
    return ckpt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", required=True)
    p.add_argument("--provenance", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
