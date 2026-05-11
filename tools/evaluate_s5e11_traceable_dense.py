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
    "severe_wrong_sign_rate": 0.033112582781456956,
    "tdir_abs_mean": 48.92733872256603,
    "tmag_p95": 1.6560804642823275,
    "tmag_max": 7.078435706785601,
    "path_ratio": 0.05272511562778335,
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


def _bucket_metrics(rows: List[Dict[str, Any]], bucket: str, field: str) -> Dict[str, Any]:
    sub = [r for r in rows if r.get("observability_bucket") == bucket]
    if not sub:
        return {"count": 0, "fraction": 0.0}
    vals = [r[field] for r in sub]
    return {
        "count": len(sub),
        "fraction": len(sub) / max(len(rows), 1),
        "signed_tdir_mean": float(np.mean([v.get("tdir_deg") for v in vals if v.get("tdir_deg") is not None])),
        "signed_tdir_p90": float(np.percentile([v.get("tdir_deg") for v in vals if v.get("tdir_deg") is not None], 90)),
        "tdir_abs_mean": float(np.mean([v.get("tdir_abs_deg") for v in vals if v.get("tdir_abs_deg") is not None])),
        "tdir_abs_p90": float(np.percentile([v.get("tdir_abs_deg") for v in vals if v.get("tdir_abs_deg") is not None], 90)),
        "anti_parallel_rate": float(np.mean([1.0 if v.get("anti_parallel_flag") else 0.0 for v in vals])),
        "severe_wrong_sign_rate": float(np.mean([1.0 if v.get("severe_wrong_sign_flag") else 0.0 for v in vals])),
        "pair_order_dir_flip_success_rate": float(np.mean([1.0 if (r.get("pair_order_dir_flip_cosine") or -1.0) > 0.5 else 0.0 for r in sub])),
        "signed_direction_loss_weight_mean": float(np.mean([r.get("signed_direction_loss_weight", 0.0) for r in sub])),
        "tmag_median": float(np.percentile([v.get("tmag_ratio") for v in vals if v.get("tmag_ratio") is not None], 50)),
        "tmag_p95": float(np.percentile([v.get("tmag_ratio") for v in vals if v.get("tmag_ratio") is not None], 95)),
        "path_fraction": float(np.sum([v.get("pred_step_length") or 0.0 for v in vals]) / max(np.sum([r[field].get("pred_step_length") or 0.0 for r in rows]), 1.0e-12)),
    }


def _small_motion(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    gt = np.asarray([r[field].get("gt_step_length") or 0.0 for r in rows], dtype=np.float64)
    pred = np.asarray([r[field].get("pred_step_length") or 0.0 for r in rows], dtype=np.float64)
    tmag = np.asarray([r[field].get("tmag_ratio") or 0.0 for r in rows], dtype=np.float64)
    tdir = np.asarray([r[field].get("tdir_deg") or 0.0 for r in rows], dtype=np.float64)
    thr = float(np.percentile(gt, 25))
    small = gt <= thr
    small_rows = [r for r in rows if (r[field].get("gt_step_length") or 0.0) <= thr]
    observable_small = [r for r in small_rows if r.get("observability_bucket") in ("normal_observable", "large_motion_observable")]
    unobservable_small = [r for r in small_rows if r.get("observability_bucket") not in ("normal_observable", "large_motion_observable")]
    return {
        "small_motion_count": int(np.sum(small)),
        "small_motion_fraction": float(np.mean(small)),
        "small_motion_observable_count": len(observable_small),
        "small_motion_unobservable_count": len(unobservable_small),
        "small_motion_gt_tmag_median": float(np.percentile(gt[small], 50)) if np.any(small) else None,
        "small_motion_gt_tmag_p95": float(np.percentile(gt[small], 95)) if np.any(small) else None,
        "small_motion_pred_tmag_median": float(np.percentile(tmag[small], 50)) if np.any(small) else None,
        "small_motion_pred_tmag_p95": float(np.percentile(tmag[small], 95)) if np.any(small) else None,
        "small_motion_signed_tdir_mean": float(np.mean(tdir[small])) if np.any(small) else None,
        "small_motion_signed_tdir_p90": float(np.percentile(tdir[small], 90)) if np.any(small) else None,
        "small_motion_anti_parallel_rate": float(np.mean([1.0 if r[field].get("anti_parallel_flag") else 0.0 for r in small_rows])) if small_rows else None,
        "small_motion_path_fraction": float(np.sum(pred[small]) / max(np.sum(pred), 1.0e-12)) if np.any(small) else None,
        "small_motion_error_contribution": float(np.mean(np.abs(tdir[small])) / max(np.mean(np.abs(tdir)), 1.0e-12)) if np.any(small) else None,
        "small_motion_amplification": bool(np.mean(small & (tmag > np.percentile(tmag, 90))) > 0.03),
    }


def _report(path: Path, ckpt: Dict[str, Any]) -> None:
    lines = [
        "# S5E11 correspondence parallax translation geometry report",
        "",
        "## 最终分类",
        str(ckpt.get("final_classification")),
        "",
        "## 训练状态",
        str(ckpt.get("training")),
        "",
        "## 与 S5E10 / S5E9 的关系",
        "S5E11 不再继续在 weak image-pair embedding 上堆 head/loss，而是显式引入 correspondence / optical flow / parallax / observability 特征，同时默认回退到 S5E9 的稳定 scale 路径。",
        "",
        "## 几何特征可用性",
        str(ckpt.get("feature_availability")),
        "",
        "## observability audit",
        str(ckpt.get("observability_audit")),
        "",
        "## direction supervision masking 结果",
        str(ckpt.get("direction_supervision_masking")),
        "",
        "## observability mask only candidate",
        str(ckpt.get("observability_mask_only_candidate")),
        "",
        "## correspondence direction candidate",
        str(ckpt.get("correspondence_direction_candidate")),
        "",
        "## geometry_plus_s5e9_scale candidate",
        str(ckpt.get("geometry_plus_s5e9_scale_candidate")),
        "",
        "## pair-order diagnostic by observability bucket",
        str(ckpt.get("pair_order_diagnostic")),
        "",
        "## small-motion observability audit",
        str(ckpt.get("small_motion_observability_audit")),
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
        str(ckpt.get("external_eval")),
        "",
        "## comparison vs ORB-SLAM3",
        str(ckpt.get("comparison_to_orbslam3")),
        "",
        "## blockers",
        str(ckpt.get("blockers")),
        "",
        "## validation",
        str(ckpt.get("validation")),
        "",
        "## caveats",
        "- S5E11 是实验性结果，不替换 S5 locked。",
        "- 只有 raw signed direction 改善且 raw scale/path 保持受控、raw vs guarded gap 很小，才可称为真实几何改善。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    metrics = read_json(out_dir / "edge_component_metrics.json")
    rows = [json.loads(line) for line in Path(args.provenance).read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_rows = [dict(r, metric=r.get("metric_preview_raw") or {}) for r in rows]
    guard_rows = [dict(r, metric=r.get("metric_preview") or {}) for r in rows]
    raw = dict(metrics.get("raw_prediction_metrics", {}))
    guard = dict(metrics.get("guarded_prediction_metrics", {}))
    ckpt = load_or_base_checkpoint(Path(args.out_json))
    ckpt["experiment"] = "S5E11_correspondence_parallax_translation_geometry"
    ckpt["status"] = {
        "experimental_candidate": True,
        "official_s5_unchanged": True,
        "not_official_replacement": True,
    }
    obs_status_path = Path("checkpoints/S5E11_correspondence_parallax_translation_geometry_candidate/observability_mask_only_training_status.json")
    corr_status_path = Path("checkpoints/S5E11_correspondence_parallax_translation_geometry_candidate/correspondence_direction_training_status.json")
    scale_status_path = Path("checkpoints/S5E11_correspondence_parallax_translation_geometry_candidate/scale_s5e9_stabilized_training_status.json")
    obs_status = read_json(obs_status_path)
    corr_status = read_json(corr_status_path)
    scale_status = read_json(scale_status_path)
    ckpt["training"] = {
        "attempted": True,
        "classification": " / ".join(
            x
            for x in [
                obs_status.get("classification"),
                corr_status.get("classification"),
                scale_status.get("classification"),
            ]
            if x
        ),
    }
    ckpt["adjacent_dense_export"] = {
        "available": True,
        "coverage": metrics.get("coverage"),
        "all_edges_traceable": metrics.get("all_edges_traceable"),
        "num_poses": metrics.get("num_poses"),
        "num_edges": metrics.get("num_edges"),
        "direct_adjacent_prediction_edges": metrics.get("direct_adjacent_prediction_edges"),
        "trajectory_path": args.trajectory,
        "raw_trajectory_path": metrics.get("raw_trajectory_path"),
        "edge_provenance": args.provenance,
    }
    ckpt["feature_availability"] = metrics.get("feature_availability")
    buckets = [
        "near_static_unobservable",
        "low_parallax_unreliable",
        "normal_observable",
        "large_motion_observable",
        "outlier_correspondence",
    ]
    bucket_stats = {bucket: _bucket_metrics(guard_rows, bucket, "metric") for bucket in buckets}
    observable_count = sum(bucket_stats[b]["count"] for b in ("normal_observable", "large_motion_observable"))
    unobservable_count = sum(bucket_stats[b]["count"] for b in ("near_static_unobservable", "low_parallax_unreliable", "outlier_correspondence"))
    small_motion = _small_motion(guard_rows, "metric")
    ckpt["observability_audit"] = {
        "bucket_stats": bucket_stats,
        "observable_edge_count": observable_count,
        "observable_edge_fraction": observable_count / max(len(rows), 1),
        "unobservable_edge_count": unobservable_count,
        "unobservable_edge_fraction": unobservable_count / max(len(rows), 1),
        "small_motion_observable_fraction": small_motion["small_motion_observable_count"] / max(small_motion["small_motion_count"], 1),
        "small_motion_unobservable_fraction": small_motion["small_motion_unobservable_count"] / max(small_motion["small_motion_count"], 1),
        "signed_direction_supervision_reliable_fraction": observable_count / max(len(rows), 1),
    }
    ckpt["direction_supervision_masking"] = {
        "signed_direction_loss_weight_mean": float(np.mean([r.get("signed_direction_loss_weight", 0.0) for r in rows])),
        "signed_direction_loss_weight_by_bucket": {bucket: bucket_stats[bucket].get("signed_direction_loss_weight_mean") for bucket in buckets},
        "excluded_direction_edge_count": int(sum(r.get("signed_direction_loss_weight", 0.0) == 0.0 for r in rows)),
        "excluded_direction_edge_fraction": float(np.mean([r.get("signed_direction_loss_weight", 0.0) == 0.0 for r in rows])),
    }
    ckpt["observability_mask_only_candidate"] = {"trajectory_path": metrics["candidate_variants"]["observability_mask_only_candidate"]}
    ckpt["correspondence_direction_candidate"] = {"trajectory_path": metrics["candidate_variants"]["correspondence_direction_candidate"]}
    ckpt["geometry_plus_s5e9_scale_candidate"] = {"trajectory_path": metrics["candidate_variants"]["geometry_plus_s5e9_scale_candidate"]}
    ckpt["pair_order_diagnostic"] = {
        "pair_order_dir_flip_success_rate_global": metrics.get("pair_order_dir_flip_success_rate_global"),
        "pair_order_dir_flip_success_rate_observable": metrics.get("pair_order_dir_flip_success_rate_observable"),
        "pair_order_dir_flip_success_rate_unobservable": metrics.get("pair_order_dir_flip_success_rate_unobservable"),
        "pair_order_dir_flip_cosine_mean_global": metrics.get("pair_order_dir_flip_cosine_mean_global"),
        "pair_order_dir_flip_cosine_mean_observable": metrics.get("pair_order_dir_flip_cosine_mean_observable"),
        "pair_order_scale_symmetry_error": metrics.get("pair_order_scale_symmetry_error"),
        "signed_direction_order_observable_global": metrics.get("signed_direction_order_observable_global"),
        "signed_direction_order_observable_on_observable_edges": metrics.get("signed_direction_order_observable_on_observable_edges"),
    }
    ckpt["small_motion_observability_audit"] = small_motion
    ckpt["raw_prediction_metrics"] = raw
    ckpt["guarded_prediction_metrics"] = guard
    ckpt["raw_component_metrics"] = raw
    ckpt["guarded_component_metrics"] = guard
    ext = {}
    raw_ext = {}
    for mode in ["none", "se3", "sim3"]:
        ext[mode] = _ext(Path(args.trajectory), Path(args.groundtruth), out_dir / f"eval_alignment_{mode}.json", mode)
        raw_ext[mode] = _ext(Path(metrics["raw_trajectory_path"]), Path(args.groundtruth), out_dir / f"eval_alignment_raw_{mode}.json", mode)
        ext[f"observability_mask_only_{mode}"] = _ext(Path(metrics["candidate_variants"]["observability_mask_only_candidate"]), Path(args.groundtruth), out_dir / f"eval_alignment_observability_mask_only_{mode}.json", mode)
        ext[f"correspondence_direction_{mode}"] = _ext(Path(metrics["candidate_variants"]["correspondence_direction_candidate"]), Path(args.groundtruth), out_dir / f"eval_alignment_correspondence_direction_{mode}.json", mode)
    ckpt["external_eval"] = ext
    ckpt["raw_external_eval"] = raw_ext
    ckpt["raw_vs_guarded_gap"] = {
        "tdir_gap": (raw.get("tdir_mean_deg") or 0.0) - (guard.get("tdir_mean_deg") or 0.0),
        "tdir_abs_gap": (raw.get("tdir_abs_mean_deg") or 0.0) - (guard.get("tdir_abs_mean_deg") or 0.0),
        "tmag_p95_gap": (raw.get("tmag_p95_ratio") or 0.0) - (guard.get("tmag_p95_ratio") or 0.0),
        "tmag_max_gap": (raw.get("tmag_max_ratio") or 0.0) - (guard.get("tmag_max_ratio") or 0.0),
        "path_ratio_gap": (raw.get("path_ratio") or 0.0) - (guard.get("path_ratio") or 0.0),
        "sim3_ATE_gap": (raw_ext["sim3"]["ate"] or 0.0) - (ext["sim3"]["ate"] or 0.0),
    }
    ckpt["comparison_to_orbslam3"] = {
        "coverage": "S5E11 454/454 vs ORB-SLAM3 273/454",
        "edge_level_gap": "unavailable",
        "se3_ate_gap": None if ext["se3"]["ate"] is None else ext["se3"]["ate"] - ORBSLAM3_REFERENCE["se3"]["ate"],
        "sim3_ate_gap": None if ext["sim3"]["ate"] is None else ext["sim3"]["ate"] - ORBSLAM3_REFERENCE["sim3"]["ate"],
        "path_ratio": f"S5E11={guard.get('path_ratio')}; ORB-SLAM3={ORBSLAM3_REFERENCE['se3']['path_ratio']}",
    }

    signed_improved = (raw.get("tdir_mean_deg") or 999.0) < S5E9["signed_tdir_mean"] - 1.0 and (raw.get("anti_parallel_rate") or 999.0) < S5E9["anti_parallel_rate"] - 0.01
    severe_improved = (raw.get("severe_wrong_sign_rate") or 999.0) < S5E9["severe_wrong_sign_rate"]
    scale_stable = (raw.get("tmag_p95_ratio") or 999.0) < 10.0 and (raw.get("tmag_max_ratio") or 999.0) < 50.0 and (raw.get("path_ratio") or 999.0) < 1.0
    raw_gap_small = abs(ckpt["raw_vs_guarded_gap"]["path_ratio_gap"]) < 0.5 and abs(ckpt["raw_vs_guarded_gap"]["tmag_p95_gap"]) < 5.0
    observable_fraction = ckpt["observability_audit"]["signed_direction_supervision_reliable_fraction"]
    pair_obs = metrics.get("pair_order_dir_flip_success_rate_observable")

    blockers = []
    if not signed_improved:
        blockers.append("显式 correspondence / parallax 特征仍未把 raw signed direction 拉到优于 S5E9。")
    if not scale_stable:
        blockers.append("虽然 S5E11 默认回退到 S5E9 scale 路径，但当前导出链上的 raw scale/path 仍未达到理想受控范围。")
    if pair_obs is None or pair_obs < 0.5:
        blockers.append("observable edge 上的 pair-order signed direction 仍然不可观测。")
    if observable_fraction < 0.35:
        blockers.append("可靠 signed direction supervision 的 observable edge 比例偏低。")
    ckpt["blockers"] = blockers

    if metrics.get("coverage", 0.0) < 1.0 or not metrics.get("all_edges_traceable", True):
        final = "S5E11_REGRESSION"
    elif not metrics.get("feature_availability", {}).get("correspondence_features_available"):
        final = "S5E11_GEOMETRY_FEATURES_UNAVAILABLE"
    elif observable_fraction < 0.30:
        final = "S5E11_OBSERVABILITY_LIMITED"
    elif signed_improved and severe_improved and pair_obs is not None and pair_obs > 0.5 and scale_stable and raw_gap_small:
        final = "S5E11_RAW_GEOMETRY_IMPROVED"
    elif signed_improved and pair_obs is not None and pair_obs > 0.5:
        final = "S5E11_CORRESPONDENCE_DIRECTION_IMPROVED"
    elif signed_improved and observable_fraction >= 0.35:
        final = "S5E11_OBSERVABILITY_MASK_IMPROVED"
    elif signed_improved:
        final = "S5E11_SIGNED_DIRECTION_IMPROVED"
    elif scale_stable and not signed_improved:
        final = "S5E11_SCALE_STABLE_DIRECTION_NOT_IMPROVED"
    elif not raw_gap_small:
        final = "S5E11_GUARD_DEPENDENT"
    else:
        final = "S5E11_INPUT_GEOMETRY_INSUFFICIENT"
    ckpt["validation"] = validation_from_logs()
    ckpt["final_classification"] = final
    write_json(
        Path("checkpoints/S5E11_correspondence_parallax_translation_geometry_candidate/training_status.json"),
        {
            "attempted": True,
            "classification": ckpt["training"]["classification"],
            "observability_mask_only_candidate": obs_status,
            "correspondence_direction_candidate": corr_status,
            "scale_s5e9_stabilized_candidate": scale_status,
            "notes": [
                "S5E11 训练阶段重点不是刷最终 ATE，而是确认 observability mask 和 correspondence proxy 是否能提升 signed direction。",
                "本轮环境没有 cv2，因此 correspondence / optical flow / essential matrix 退化为 deterministic proxy features。",
            ],
        },
    )
    write_json(Path(args.out_json), ckpt)
    _report(Path(args.out_report), ckpt)
    return ckpt


def parse_args() -> argparse.Namespace:
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
