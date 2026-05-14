#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, load_or_base_checkpoint, read_json, validation_from_logs, write_json


S5E8_RAW = {
    "tdir_abs_mean": 46.17838867792688,
    "anti": 0.1368653421633554,
    "tmag_p95": 95.93377590609806,
    "path_ratio": 6.465477764096885,
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
    for i, flag in enumerate(tmag > thr):
        if flag:
            cur += 1
            if cur > longest:
                longest = cur
                best = (i - cur + 1, i)
        else:
            cur = 0
    return {
        "top20_tmag_path_fraction": top20,
        "long_run_path_fraction": float(pred[best[0] : best[1] + 1].sum() / max(pred.sum(), 1.0e-12)),
    }


def _bucket_audit(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
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
        "# S5E9 scale-unit, small-motion, signed-direction fix report",
        "",
        "## 最终分类",
        str(ckpt.get("final_classification")),
        "",
        "## 训练状态",
        str(ckpt.get("training")),
        "",
        "## 与 S5E8 的关系",
        "S5E9 重点检查 raw scale explosion 到底来自字段/单位链路问题，还是模型真的学坏了。",
        "",
        "## scale pipeline audit",
        str(ckpt.get("scale_pipeline_audit")),
        "",
        "## 是否发现 unit/export/eval mismatch",
        str({"unit_mismatch_suspected": ckpt.get("scale_pipeline_audit", {}).get("unit_mismatch_suspected"), "export_eval_tmag_consistency": ckpt.get("scale_pipeline_audit", {}).get("export_eval_tmag_consistency")}),
        "",
        "## tight bounded scale residual 结果",
        str(ckpt.get("scale_candidate")),
        "",
        "## small-motion / near-static bucket audit",
        str(ckpt.get("small_motion_audit")),
        "",
        "## pair-order / signed direction diagnostic",
        str(ckpt.get("pair_order_diagnostic")),
        "",
        "## scale-first candidate metrics",
        str(ckpt.get("scale_candidate_metrics")),
        "",
        "## signed-direction candidate metrics",
        str(ckpt.get("signed_direction_candidate_metrics")),
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
        "## comparison vs S5E8/S5E7/S5E6/S5E5/S5E4/S5E3/S5E2",
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
        "- S5E9 是实验性结果，不替换 S5 locked。",
        "- 只有 raw scale/path 脱离 guard 后明显改善，才可以认定真实几何改善。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    metrics = read_json(out_dir / "edge_component_metrics.json")
    rows = [json.loads(line) for line in Path(args.provenance).read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_rows = [{"metric_preview": r.get("metric_preview_raw") or {}} for r in rows]
    guard_rows = [{"metric_preview": r.get("metric_preview") or {}} for r in rows]
    raw = dict(metrics.get("raw_prediction_metrics", {})); raw.update(_top_and_long(raw_rows, "metric_preview"))
    guarded = dict(metrics.get("guarded_prediction_metrics", {})); guarded.update(_top_and_long(guard_rows, "metric_preview"))
    ckpt = load_or_base_checkpoint(Path(args.out_json))
    ckpt["raw_prediction_metrics"] = raw
    ckpt["guarded_prediction_metrics"] = guarded
    ckpt["component_metrics"] = guarded
    ckpt["pair_order_diagnostic"] = {
        "pair_order_dir_flip_cosine_mean": metrics.get("pair_order_dir_flip_cosine_mean"),
        "pair_order_dir_flip_success_rate": metrics.get("pair_order_dir_flip_success_rate"),
        "pair_order_scale_symmetry_error": metrics.get("pair_order_scale_symmetry_error"),
        "pair_order_failure_rate": metrics.get("pair_order_failure_rate"),
        "signed_direction_order_observable": metrics.get("signed_direction_order_observable"),
    }
    ckpt["small_motion_audit"] = _bucket_audit(guard_rows, "metric_preview")
    audit_path = out_dir / "scale_pipeline_audit.json"
    if audit_path.exists():
        ckpt["scale_pipeline_audit"] = read_json(audit_path)
    scale_status = Path("checkpoints/S5E9_scale_unit_small_motion_signed_direction_fix_candidate/scale_training_status.json")
    if scale_status.exists():
        ckpt["scale_candidate"] = read_json(scale_status)
    sign_status = Path("checkpoints/S5E9_scale_unit_small_motion_signed_direction_fix_candidate/signed_direction_training_status.json")
    if sign_status.exists():
        ckpt["signed_direction_candidate"] = read_json(sign_status)
        if ckpt["signed_direction_candidate"].get("history"):
            ckpt["signed_direction_candidate_metrics"] = ckpt["signed_direction_candidate"]["history"][-1]["val_metrics"]
    if ckpt.get("scale_candidate", {}).get("history"):
        ckpt["scale_candidate_metrics"] = ckpt["scale_candidate"]["history"][-1]

    ext, raw_ext = {}, {}
    for mode in ["none", "se3", "sim3"]:
        ext[mode] = _ext(Path(args.trajectory), Path(args.groundtruth), out_dir / f"eval_alignment_{mode}.json", mode)
        raw_ext[mode] = _ext(Path(metrics["raw_trajectory_path"]), Path(args.groundtruth), out_dir / f"eval_alignment_raw_{mode}.json", mode)
    ckpt["external_eval"] = ext
    ckpt["raw_external_eval"] = raw_ext
    ckpt["raw_vs_guarded_gap"] = {
        "tdir_gap": None if raw.get("tdir_mean_deg") is None else raw.get("tdir_mean_deg") - guarded.get("tdir_mean_deg"),
        "tdir_abs_gap": None if raw.get("tdir_abs_mean_deg") is None else raw.get("tdir_abs_mean_deg") - guarded.get("tdir_abs_mean_deg"),
        "tmag_p95_gap": None if raw.get("tmag_p95_ratio") is None else raw.get("tmag_p95_ratio") - guarded.get("tmag_p95_ratio"),
        "tmag_max_gap": None if raw.get("tmag_max_ratio") is None else raw.get("tmag_max_ratio") - guarded.get("tmag_max_ratio"),
        "path_ratio_gap": None if raw.get("path_ratio") is None else raw.get("path_ratio") - guarded.get("path_ratio"),
        "sim3_ATE_gap": None if raw_ext["sim3"]["ate"] is None else raw_ext["sim3"]["ate"] - ext["sim3"]["ate"],
    }
    comparison = {
        "raw_tmag_p95_improved_vs_s5e8": (raw.get("tmag_p95_ratio") or 999) < S5E8_RAW["tmag_p95"] - 5.0,
        "raw_tmag_max_improved_vs_s5e8": (raw.get("tmag_max_ratio") or 999) < 1000.0,
        "raw_path_ratio_improved_vs_s5e8": (raw.get("path_ratio") or 999) < S5E8_RAW["path_ratio"] - 1.0,
        "raw_tdir_abs_mean_improved_vs_s5e8": (raw.get("tdir_abs_mean_deg") or 999) < S5E8_RAW["tdir_abs_mean"] - 0.5,
        "anti_parallel_not_regressed": (raw.get("anti_parallel_rate") or 999) <= S5E8_RAW["anti"] + 0.01,
        "guard_dependency": (raw.get("path_ratio") or 999) - (guarded.get("path_ratio") or 999) > 1.0 or (raw.get("tmag_p95_ratio") or 999) - (guarded.get("tmag_p95_ratio") or 999) > 20.0,
    }
    ckpt["comparison_summary"] = comparison
    ckpt["comparison_to_orbslam3"] = {
        "coverage": "S5E9 454/454 vs ORB-SLAM3 273/454",
        "edge_level_gap": "unavailable",
        "se3_ate_gap": None if ext["se3"]["ate"] is None else ext["se3"]["ate"] - ORBSLAM3_REFERENCE["se3"]["ate"],
        "sim3_ate_gap": None if ext["sim3"]["ate"] is None else ext["sim3"]["ate"] - ORBSLAM3_REFERENCE["sim3"]["ate"],
        "path_ratio": f"S5E9={guarded.get('path_ratio')}; ORB-SLAM3={ORBSLAM3_REFERENCE['se3']['path_ratio']}",
    }
    blockers = []
    if ckpt.get("scale_pipeline_audit", {}).get("unit_mismatch_suspected"):
        blockers.append("raw scale 与 GT 在高分位上仍有显著量级偏差。")
    if comparison["guard_dependency"]:
        blockers.append("guarded trajectory 仍明显优于 raw trajectory。")
    if not comparison["raw_tdir_abs_mean_improved_vs_s5e8"]:
        blockers.append("raw signed direction 没有稳定优于 S5E8。")
    if ckpt.get("small_motion_audit", {}).get("small_motion_amplification"):
        blockers.append("small-motion amplification 仍存在。")
    ckpt["blockers"] = blockers

    if ckpt.get("scale_pipeline_audit", {}).get("unit_mismatch_suspected") and not ckpt.get("scale_pipeline_audit", {}).get("export_eval_tmag_consistency") is False:
        final = "S5E9_SCALE_UNIT_BUG_FOUND"
    elif comparison["raw_tmag_p95_improved_vs_s5e8"] and comparison["raw_tmag_max_improved_vs_s5e8"] and comparison["raw_path_ratio_improved_vs_s5e8"] and not comparison["guard_dependency"] and comparison["raw_tdir_abs_mean_improved_vs_s5e8"]:
        final = "S5E9_RAW_GEOMETRY_IMPROVED"
    elif comparison["raw_tmag_p95_improved_vs_s5e8"] and comparison["raw_path_ratio_improved_vs_s5e8"] and not comparison["guard_dependency"]:
        final = "S5E9_SCALE_RAW_IMPROVED"
    elif ckpt.get("pair_order_diagnostic", {}).get("pair_order_dir_flip_success_rate", 0.0) > 0.6 and comparison["raw_tdir_abs_mean_improved_vs_s5e8"]:
        final = "S5E9_SIGNED_DIRECTION_IMPROVED"
    elif ckpt.get("small_motion_audit", {}).get("small_motion_amplification") is False:
        final = "S5E9_SMALL_MOTION_IMPROVED"
    elif comparison["guard_dependency"]:
        final = "S5E9_GUARD_DEPENDENT"
    elif raw.get("tdir_abs_mean_deg") and raw.get("tdir_abs_mean_deg") >= 46.0 and raw.get("tmag_p95_ratio") and raw.get("tmag_p95_ratio") >= 90.0:
        final = "S5E9_INPUT_GEOMETRY_INSUFFICIENT"
    else:
        final = "S5E9_NO_IMPROVEMENT"
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
