#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, load_or_base_checkpoint, read_json, validation_from_logs, write_json


S5E7_RAW = {
    "tdir_abs_mean": 48.649060747032244,
    "tdir_abs_p90": 80.07196258057124,
    "anti_parallel_rate": 0.1545253863134658,
    "tmag_p95": 94.28468533280407,
    "path_ratio": 6.690489734209743,
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
    data = read_json(out_json)
    return {"ate": data.get("ATE"), "drift": data.get("drift"), "path_ratio": data.get("path_ratio")}


def _top_and_long(rows: List[Dict[str, Any]], field: str) -> Dict[str, float]:
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
    long_frac = float(pred[best[0] : best[1] + 1].sum() / max(pred.sum(), 1.0e-12))
    return {"top20_tmag_path_fraction": top20, "long_run_path_fraction": long_frac}


def _bucket_eval(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    out = {}
    for name in ["small_motion", "normal_motion", "large_motion"]:
        sub = [r for r in rows if r.get("optional_motion_bucket") == name]
        if not sub:
            out[name] = {}
            continue
        vals = [r[field] for r in sub]
        pred = np.asarray([v.get("pred_step_length") or 0.0 for v in vals], dtype=np.float64)
        total_pred = np.asarray([r[field].get("pred_step_length") or 0.0 for r in rows], dtype=np.float64)
        tdir = np.asarray([v.get("tdir_deg") or 0.0 for v in vals], dtype=np.float64)
        tdir_abs = np.asarray([v.get("tdir_abs_deg") or 0.0 for v in vals], dtype=np.float64)
        anti = np.asarray([1.0 if v.get("anti_parallel_flag") else 0.0 for v in vals], dtype=np.float64)
        tmag = np.asarray([v.get("tmag_ratio") or 0.0 for v in vals], dtype=np.float64)
        out[name] = {
            "count": len(sub),
            "fraction": len(sub) / max(len(rows), 1),
            "signed_tdir_mean": float(np.mean(tdir)),
            "tdir_abs_mean": float(np.mean(tdir_abs)),
            "tdir_abs_p90": float(np.percentile(tdir_abs, 90)),
            "anti_parallel_rate": float(np.mean(anti)),
            "tmag_median": float(np.percentile(tmag, 50)),
            "tmag_p95": float(np.percentile(tmag, 95)),
            "path_fraction": float(np.sum(pred) / max(np.sum(total_pred), 1.0e-12)),
        }
    return out


def _motion_bucket_audit(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    gt = np.asarray([r[field].get("gt_step_length") or 0.0 for r in rows], dtype=np.float64)
    tdir = np.asarray([r[field].get("tdir_deg") or 0.0 for r in rows], dtype=np.float64)
    pred = np.asarray([r[field].get("pred_step_length") or 0.0 for r in rows], dtype=np.float64)
    tmag = np.asarray([r[field].get("tmag_ratio") or 0.0 for r in rows], dtype=np.float64)
    thr = float(np.percentile(gt, 25))
    small = gt <= thr
    high_tmag = tmag > np.percentile(tmag, 90)
    return {
        "small_motion_count": int(np.sum(small)),
        "small_motion_fraction": float(np.mean(small)),
        "small_motion_tdir_mean": float(np.mean(tdir[small])) if np.any(small) else None,
        "small_motion_tdir_p90": float(np.percentile(tdir[small], 90)) if np.any(small) else None,
        "small_motion_tmag_p95": float(np.percentile(tmag[small], 95)) if np.any(small) else None,
        "small_motion_path_fraction": float(np.sum(pred[small]) / max(np.sum(pred), 1.0e-12)) if np.any(small) else None,
        "small_motion_amplification": bool(np.mean(small & high_tmag) > 0.03),
        "small_motion_error_contribution": float(np.mean(np.abs(tdir[small])) / max(np.mean(np.abs(tdir)), 1.0e-12)) if np.any(small) else None,
    }


def _report(path: Path, ckpt: Dict[str, Any]) -> None:
    lines = [
        "# S5E8 translation geometry diagnostic ablation report",
        "",
        "## 最终分类",
        str(ckpt.get("final_classification")),
        "",
        "## 训练状态",
        str(ckpt.get("training")),
        "",
        "## 实验目的",
        "S5E8 的目标不是刷新 ATE，而是定位 translation direction 与 scale 学不好的根因。",
        "",
        "## 与 S5E7 的关系",
        "S5E7 已经说明收益主要来自 guard，因此 S5E8 重点转向 direction-only、oracle、prior-only 和 numeric-only 诊断。",
        "",
        "## oracle direction ablation",
        str(ckpt.get("oracle_ablation", {}).get("oracle_direction")),
        "",
        "## oracle scale ablation",
        str(ckpt.get("oracle_ablation", {}).get("oracle_scale")),
        "",
        "## direction-only candidate",
        str(ckpt.get("direction_only_candidate")),
        "",
        "## prior-only baseline",
        str(ckpt.get("prior_only_baselines")),
        "",
        "## no-image / numeric-only control",
        str(ckpt.get("numeric_only_control")),
        "",
        "## supervision quality audit",
        str(ckpt.get("translation_supervision_quality")),
        "",
        "## motion bucket audit",
        str(ckpt.get("motion_bucket_audit")),
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
        str({"guarded": ckpt.get("external_eval"), "raw": ckpt.get("raw_external_eval"), "oracle": ckpt.get("oracle_ablation")}),
        "",
        "## comparison vs S5E7/S5E6/S5E5/S5E4/S5E3/S5E2",
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
        "- S5E8 是实验性诊断，不替换 S5 locked 结果。",
        "- 只有 raw prediction 在 direction 和 high-percentile scale 上同时优于 S5E7，才可称为真实几何改善。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_json(out_dir / "edge_component_metrics.json")
    rows = [json.loads(line) for line in Path(args.provenance).read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_rows = [{"metric_preview": r.get("metric_preview_raw") or {}, "optional_motion_bucket": r.get("optional_motion_bucket")} for r in rows]
    guarded_rows = [{"metric_preview": r.get("metric_preview") or {}, "optional_motion_bucket": r.get("optional_motion_bucket")} for r in rows]
    raw = dict(metrics.get("raw_prediction_metrics", {}))
    guarded = dict(metrics.get("guarded_prediction_metrics", {}))
    raw.update(_top_and_long(raw_rows, "metric_preview"))
    guarded.update(_top_and_long(guarded_rows, "metric_preview"))

    ckpt = load_or_base_checkpoint(Path(args.out_json))
    ckpt["raw_prediction_metrics"] = raw
    ckpt["guarded_prediction_metrics"] = guarded
    ckpt["component_metrics"] = guarded
    ckpt["motion_bucket_audit"] = _motion_bucket_audit(guarded_rows, "metric_preview")
    ckpt["bucketed_evaluation"] = _bucket_eval(guarded_rows, "metric_preview")
    q_path = out_dir / "translation_supervision_quality.json"
    if q_path.exists():
        ckpt["translation_supervision_quality"] = read_json(q_path)
    oracle_path = out_dir / "oracle_ablation.json"
    if oracle_path.exists():
        ckpt["oracle_ablation"] = read_json(oracle_path)
    prior_path = Path("checkpoints/S5E8_translation_geometry_diagnostic_ablation_candidate/prior_only_baselines.json")
    if prior_path.exists():
        payload = read_json(prior_path)
        ckpt["prior_only_baselines"] = payload.get("prior_only_baselines")
        ckpt["numeric_only_control"] = payload.get("prior_only_baselines", {}).get("numeric_prior_only_model")
    dir_train = Path("checkpoints/S5E8_translation_geometry_diagnostic_ablation_candidate/direction_only_training_status.json")
    if dir_train.exists():
        ckpt["direction_only_candidate"] = read_json(dir_train)

    ext = {}
    raw_ext = {}
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
        "raw_tdir_abs_mean_improved_vs_s5e7": (raw.get("tdir_abs_mean_deg") or 999) < S5E7_RAW["tdir_abs_mean"] - 1.0,
        "raw_tdir_abs_p90_improved_vs_s5e7": (raw.get("tdir_abs_p90_deg") or 999) < S5E7_RAW["tdir_abs_p90"] - 1.0,
        "raw_anti_parallel_improved_vs_s5e7": (raw.get("anti_parallel_rate") or 999) < S5E7_RAW["anti_parallel_rate"] - 0.01,
        "raw_tmag_p95_improved_vs_s5e7": (raw.get("tmag_p95_ratio") or 999) < S5E7_RAW["tmag_p95"] - 5.0,
        "raw_path_ratio_improved_vs_s5e7": (raw.get("path_ratio") or 999) < S5E7_RAW["path_ratio"] - 0.2,
        "guard_dependency": (raw_ext["sim3"]["ate"] or 999) - (ext["sim3"]["ate"] or 999) > 0.2 or (raw.get("path_ratio") or 999) - (guarded.get("path_ratio") or 999) > 1.0,
    }
    ckpt["comparison_summary"] = comparison
    ckpt["comparison_to_orbslam3"] = {
        "coverage": "S5E8 454/454 vs ORB-SLAM3 273/454",
        "edge_level_gap": "unavailable",
        "se3_ate_gap": None if ext["se3"]["ate"] is None else ext["se3"]["ate"] - ORBSLAM3_REFERENCE["se3"]["ate"],
        "sim3_ate_gap": None if ext["sim3"]["ate"] is None else ext["sim3"]["ate"] - ORBSLAM3_REFERENCE["sim3"]["ate"],
        "path_ratio": f"S5E8={guarded.get('path_ratio')}; ORB-SLAM3={ORBSLAM3_REFERENCE['se3']['path_ratio']}",
    }

    blockers: List[str] = []
    if comparison["guard_dependency"]:
        blockers.append("guarded 指标明显优于 raw 指标，仍然存在 guard dependency。")
    if not comparison["raw_tdir_abs_mean_improved_vs_s5e7"]:
        blockers.append("raw direction 指标没有稳定优于 S5E7。")
    if ckpt["motion_bucket_audit"].get("small_motion_amplification"):
        blockers.append("small-motion amplification 仍然存在。")
    if ckpt.get("prior_only_baselines"):
        learned = raw.get("tdir_abs_mean_deg") or 999
        prior = ckpt["prior_only_baselines"]["numeric_prior_only_model"]["tdir_abs_mean"]
        if learned >= prior - 0.5:
            blockers.append("learned image model 没有明显超越 numeric/prior-only baseline。")
    ckpt["blockers"] = blockers

    if comparison["raw_tdir_abs_mean_improved_vs_s5e7"] and comparison["raw_tdir_abs_p90_improved_vs_s5e7"] and comparison["raw_anti_parallel_improved_vs_s5e7"] and comparison["raw_tmag_p95_improved_vs_s5e7"] and comparison["raw_path_ratio_improved_vs_s5e7"] and not comparison["guard_dependency"]:
        final = "S5E8_RAW_GEOMETRY_IMPROVED"
    elif ckpt["motion_bucket_audit"].get("small_motion_amplification") and (ckpt["motion_bucket_audit"].get("small_motion_error_contribution") or 0.0) > 1.1:
        final = "S5E8_SMALL_MOTION_DOMINATED"
    elif comparison["guard_dependency"]:
        final = "S5E8_GUARD_DEPENDENT"
    elif ckpt.get("oracle_ablation", {}).get("oracle_direction", {}).get("sim3", {}).get("ate", 999) + 0.5 < (ext["sim3"]["ate"] or 999):
        final = "S5E8_DIRECTION_SUPERVISION_LIMITED"
    elif ckpt.get("oracle_ablation", {}).get("oracle_scale", {}).get("sim3", {}).get("ate", 999) + 0.5 < (ext["sim3"]["ate"] or 999):
        final = "S5E8_SCALE_SUPERVISION_LIMITED"
    elif ckpt.get("prior_only_baselines") and (raw.get("tdir_abs_mean_deg") or 999) >= ckpt["prior_only_baselines"]["numeric_prior_only_model"]["tdir_abs_mean"] - 0.5:
        final = "S5E8_INPUT_GEOMETRY_INSUFFICIENT"
    elif comparison["raw_tdir_abs_mean_improved_vs_s5e7"] or comparison["raw_tdir_abs_p90_improved_vs_s5e7"]:
        final = "S5E8_DIRECTION_ONLY_IMPROVED"
    elif (raw.get("tdir_abs_mean_deg") or 999) > S5E7_RAW["tdir_abs_mean"] + 3.0 and (raw.get("tmag_p95_ratio") or 999) > S5E7_RAW["tmag_p95"] + 10.0:
        final = "S5E8_REGRESSION"
    else:
        final = "S5E8_NO_IMPROVEMENT"
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
