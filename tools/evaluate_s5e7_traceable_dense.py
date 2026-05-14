#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, load_or_base_checkpoint, read_json, validation_from_logs, write_json


S5E6 = {
    "rot": 0.9134395040767528,
    "tdir": 51.47429479273258,
    "tdir_abs": 46.07968707914154,
    "anti": 0.1368653421633554,
    "tmag_p95": 63.490479510847564,
    "path": 1.880843438039364,
    "sim3": 4.012221895981248,
}


def _ext(traj: str, gt: str, out_path: Path) -> Dict[str, Any]:
    subprocess.run(
        [
            "/home/dovetao/miniconda3/envs/pytorch/bin/python",
            "tools/evaluate_external_baseline_trajectory.py",
            "--trajectory",
            traj,
            "--groundtruth",
            gt,
            "--alignment",
            out_path.stem.split("_")[-1],
            "--output-json",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    data = read_json(out_path)
    return {
        "ate": data.get("ATE"),
        "drift": data.get("drift"),
        "path_ratio": data.get("path_ratio"),
        "status": data.get("status"),
        "num_matched_poses": data.get("num_matched_poses"),
        "tracking_success_rate": data.get("tracking_success_rate"),
    }


def _top_fracs(rows: List[Dict[str, Any]], field: str) -> Dict[str, float]:
    pred = np.asarray([r[field].get("pred_step_length") or 0.0 for r in rows], dtype=np.float64)
    tmag = np.asarray([r[field].get("tmag_ratio") or -1.0 for r in rows], dtype=np.float64)
    idx = np.argsort(tmag)[-20:]
    top20 = float(pred[idx].sum() / max(pred.sum(), 1.0e-12))
    thr = float(np.percentile(tmag[tmag >= 0], 95))
    flags = tmag > thr
    longest = cur = 0
    best = (0, 0)
    for i, f in enumerate(flags):
        if f:
            cur += 1
            if cur > longest:
                longest = cur
                best = (i - cur + 1, i)
        else:
            cur = 0
    long_frac = float(pred[best[0] : best[1] + 1].sum() / max(pred.sum(), 1.0e-12))
    return {"top20_tmag_path_fraction": top20, "long_run_path_fraction": long_frac}


def _motion_bucket_audit(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    gt = np.asarray([r[field].get("gt_step_length") or 0.0 for r in rows], dtype=np.float64)
    tdir = np.asarray([r[field].get("tdir_deg") or 0.0 for r in rows], dtype=np.float64)
    tmag = np.asarray([r[field].get("tmag_ratio") or 0.0 for r in rows], dtype=np.float64)
    pred = np.asarray([r[field].get("pred_step_length") or 0.0 for r in rows], dtype=np.float64)
    thr = float(np.percentile(gt, 25))
    small = gt <= thr
    return {
        "small_motion_count": int(np.sum(small)),
        "small_motion_fraction": float(np.mean(small)),
        "small_motion_tdir_mean": float(np.mean(tdir[small])) if np.any(small) else None,
        "small_motion_tmag_p95": float(np.percentile(tmag[small], 95)) if np.any(small) else None,
        "small_motion_path_fraction": float(np.sum(pred[small]) / max(np.sum(pred), 1.0e-12)) if np.any(small) else None,
        "small_motion_amplification": bool(np.mean((small) & (tmag > np.percentile(tmag, 90))) > 0.03),
    }


def _compare(raw: Dict[str, Any], guarded: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "raw_vs_guarded_tdir_gap": None if raw.get("tdir_mean_deg") is None else raw.get("tdir_mean_deg") - guarded.get("tdir_mean_deg"),
        "raw_vs_guarded_tdir_abs_gap": None if raw.get("tdir_abs_mean_deg") is None else raw.get("tdir_abs_mean_deg") - guarded.get("tdir_abs_mean_deg"),
        "raw_vs_guarded_tmag_p95_gap": None if raw.get("tmag_p95_ratio") is None else raw.get("tmag_p95_ratio") - guarded.get("tmag_p95_ratio"),
        "raw_vs_guarded_path_ratio_gap": None if raw.get("path_ratio") is None else raw.get("path_ratio") - guarded.get("path_ratio"),
    }


def _report(path: Path, ckpt: Dict[str, Any]) -> None:
    lines = [
        "# S5E7 direction-scale calibrated geometry report",
        "",
        "## 执行摘要",
        f"S5E7 的目标是同时改进方向误差与尺度高分位误差。`final_classification = {ckpt.get('final_classification')}`。",
        "",
        "## training status",
        str(ckpt.get("training")),
        "",
        "## architecture summary",
        "S5E7 使用 direction-scale factorization：`pred_t = pred_dir_unit * pred_tmag`，其中 `pred_tmag = prior_tmag * exp(clamped_log_scale_delta)`。",
        "",
        "## train magnitude prior",
        str(ckpt.get("training", {}).get("train_magnitude_prior")),
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
        "## component metrics",
        str(ckpt.get("component_metrics")),
        "",
        "## external evaluator results",
        str(ckpt.get("external_eval")),
        "",
        "## comparison vs S5E6/S5E5/S5E4/S5E3/S5E2",
        str(ckpt.get("comparison_vs_s5e6")),
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
        "- S5E7 是 experimental candidate。",
        "- 不替代 official S5 locked result。",
        "- guard-only improvement 必须单独标记，不能当作真实几何学习成功。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_json(out_dir / "edge_component_metrics.json")
    rows = []
    for raw in Path(args.provenance).read_text(encoding="utf-8").splitlines():
        if raw.strip():
            rows.append(json.loads(raw))
    raw_rows = [{"metric_preview": r.get("metric_preview_raw") or {}} for r in rows]
    guarded_rows = [{"metric_preview": r.get("metric_preview") or {}} for r in rows]
    raw_fracs = _top_fracs(raw_rows, "metric_preview")
    guarded_fracs = _top_fracs(guarded_rows, "metric_preview")
    raw_metrics = dict(metrics.get("raw_prediction_metrics", {}))
    guarded_metrics = dict(metrics.get("guarded_prediction_metrics", {}))
    raw_metrics.update(raw_fracs)
    guarded_metrics.update(guarded_fracs)
    ckpt = load_or_base_checkpoint(Path(args.out_json))
    ckpt["raw_prediction_metrics"] = raw_metrics
    ckpt["guarded_prediction_metrics"] = guarded_metrics
    ckpt["component_metrics"] = guarded_metrics
    ckpt["motion_bucket_audit"] = _motion_bucket_audit(guarded_rows, "metric_preview")
    audit_path = out_dir / "direction_scale_failure_audit.json"
    if audit_path.exists():
        ckpt["direction_scale_failure_audit"] = read_json(audit_path)

    ext = {}
    raw_ext = {}
    for mode in ["none", "se3", "sim3"]:
        ext[mode] = _ext(args.trajectory, args.groundtruth, out_dir / f"eval_alignment_{mode}.json")
        raw_ext[mode] = _ext(str(Path(args.trajectory).with_name(Path(args.trajectory).stem + "_raw" + Path(args.trajectory).suffix)), args.groundtruth, out_dir / f"eval_alignment_raw_{mode}.json")
    ckpt["external_eval"] = ext
    ckpt["raw_external_eval"] = raw_ext

    direction_improved = (guarded_metrics.get("tdir_mean_deg") or 999) < 45.0 or (guarded_metrics.get("tdir_abs_mean_deg") or 999) < 40.0
    scale_improved = (guarded_metrics.get("tmag_p95_ratio") or 999) < 50.0 and (guarded_metrics.get("path_ratio") or 999) < 1.5
    rot_preserved = (guarded_metrics.get("rot_mean_deg") or 999) <= S5E6["rot"] + 0.3
    anti_good = (guarded_metrics.get("anti_parallel_rate") or 999) < 0.10
    severe_good = (guarded_metrics.get("severe_wrong_sign_rate") or 999) < 0.025
    raw_good_enough = (
        (raw_metrics.get("path_ratio") or 999) < 2.2
        and (raw_metrics.get("tmag_p95_ratio") or 999) < 75.0
        and (raw_metrics.get("tdir_abs_mean_deg") or 999) < 50.0
    )
    guard_dependent = not raw_good_enough and (
        (guarded_metrics.get("path_ratio") or 999) + 0.2 < (raw_metrics.get("path_ratio") or 999)
        or (guarded_metrics.get("tmag_p95_ratio") or 999) + 5.0 < (raw_metrics.get("tmag_p95_ratio") or 999)
    )

    comp = {
        "rot_preserved": rot_preserved,
        "signed_tdir_improved_vs_s5e6": (guarded_metrics.get("tdir_mean_deg") or 999) < S5E6["tdir"] - 1.0,
        "tdir_abs_improved_vs_s5e6": (guarded_metrics.get("tdir_abs_mean_deg") or 999) < S5E6["tdir_abs"] - 1.0,
        "anti_parallel_improved_vs_s5e6": (guarded_metrics.get("anti_parallel_rate") or 999) < S5E6["anti"] - 0.01,
        "tmag_p95_improved_vs_s5e6": (guarded_metrics.get("tmag_p95_ratio") or 999) < S5E6["tmag_p95"] - 5.0,
        "path_ratio_improved_vs_s5e6": (guarded_metrics.get("path_ratio") or 999) < S5E6["path"] - 0.1,
        "sim3_ate_improved_vs_s5e6": (ext.get("sim3", {}).get("ate") or 999) < S5E6["sim3"] - 0.05,
        "raw_vs_guarded": _compare(raw_metrics, guarded_metrics),
    }
    ckpt["comparison_vs_s5e6"] = comp
    ckpt["comparison_to_orbslam3"] = {
        "coverage": "S5E7 454/454 vs ORB-SLAM3 273/454",
        "rot_gap": "unavailable",
        "tdir_gap": "unavailable",
        "tmag_gap": "unavailable",
        "se3_ate_gap": None if ext["se3"]["ate"] is None else ext["se3"]["ate"] - ORBSLAM3_REFERENCE["se3"]["ate"],
        "sim3_ate_gap": None if ext["sim3"]["ate"] is None else ext["sim3"]["ate"] - ORBSLAM3_REFERENCE["sim3"]["ate"],
        "path_ratio": f"S5E7={guarded_metrics.get('path_ratio')}; ORB-SLAM3={ORBSLAM3_REFERENCE['se3']['path_ratio']}",
        "summary": "ORB-SLAM3 没有可比的 edge-level 指标存档，因此这里只比较轨迹层面的覆盖率与 ATE/path_ratio。",
    }

    blockers: List[str] = []
    if not rot_preserved:
        blockers.append("rotation 出现了明显劣化。")
    if not direction_improved:
        blockers.append("translation direction 仍未形成明确改善。")
    if not scale_improved:
        blockers.append("高分位尺度误差或 path_ratio 仍未达到预期。")
    if guard_dependent:
        blockers.append("主要改善来自 fallback guard，而不是 raw direction/scale 学习质量。")
    ckpt["blockers"] = blockers

    if not ckpt["adjacent_dense_export"].get("all_edges_traceable") or ckpt["adjacent_dense_export"].get("coverage") != 1.0:
        final = "S5E7_REGRESSION"
    elif rot_preserved and direction_improved and scale_improved and anti_good and severe_good and (ext["sim3"]["ate"] or 999) < 3.80:
        final = "S5E7_DIRECTION_SCALE_IMPROVED"
    elif guard_dependent:
        final = "S5E7_GUARD_DEPENDENT"
    elif comp["signed_tdir_improved_vs_s5e6"] or comp["tdir_abs_improved_vs_s5e6"]:
        final = "S5E7_DIRECTION_ONLY_IMPROVED"
    elif comp["tmag_p95_improved_vs_s5e6"] or comp["path_ratio_improved_vs_s5e6"]:
        final = "S5E7_SCALE_ONLY_IMPROVED"
    elif (guarded_metrics.get("tdir_mean_deg") or 999) > S5E6["tdir"] + 5.0 and (guarded_metrics.get("tmag_p95_ratio") or 999) > S5E6["tmag_p95"] + 10.0:
        final = "S5E7_REGRESSION"
    else:
        final = "S5E7_NO_GEOMETRY_IMPROVEMENT"
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
