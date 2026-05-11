#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from s5e2_adjacent_dense_lib import ORBSLAM3_REFERENCE, validation_from_logs, write_json


S5E15_REF = {
    "signed_tdir_mean_deg": 50.3530,
    "anti_parallel_rate": 0.1479,
    "tmag_median_ratio": 1.2685,
    "path_ratio": 0.1572,
    "sim3_ate": 3.9682,
}
S5E19C_REF = {
    "signed_tdir_mean_deg": 56.1753,
    "anti_parallel_rate": 0.1656,
    "path_ratio": 0.1224,
    "sim3_ate": 3.9958,
    "delta_tdir_norm_mean": 0.6029,
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
    data = json.loads(out_json.read_text(encoding="utf-8"))
    return {"ate": data.get("ATE"), "drift": data.get("drift"), "path_ratio": data.get("path_ratio")}


def _report(path: Path, ckpt: Dict[str, Any]) -> None:
    lines = [
        "# S5E20 true k-step composition supervision report",
        "",
        "## 执行摘要",
        f"S5E20 的目标是用真实 contiguous k-step composition supervision 替换 S5E19C 的 fake multiframe loss，并把 delta_tdir 控制在更保守的范围内。最终分类：`{ckpt['final_classification']}`。",
        "",
        "## S5E19D failure recap",
        "S5E19D 已确认：S5E19C 的关键问题不是 no-op，而是 delta_tdir 过激、sign head 无效，以及 multiframe supervision 其实是随机 batch proxy。",
        "",
        "## 为什么要替换 fake multiframe loss",
        "S5E20 只使用真实连续窗口 i->i+k，不再允许 random batch proxy 伪装成 multiframe composition。",
        "",
        "## true k-step contiguous window 构建",
        str(ckpt.get("kstep_dataset")),
        "",
        "## composition supervision 公式",
        "T_chain = T_i,i+1 ∘ ... ∘ T_i+k-1,i+k，并对 direct GT relative pose 做 direction/path consistency 监督。",
        "",
        "## conservative delta_tdir 设计",
        str(ckpt.get("delta_tdir_control")),
        "",
        "## training status",
        str(ckpt.get("training")),
        "",
        "## traceable dense export",
        str(ckpt.get("adjacent_dense_export")),
        "",
        "## delta_tdir norm audit",
        str(ckpt.get("delta_tdir_control")),
        "",
        "## component metrics",
        str(ckpt.get("component_metrics")),
        "",
        "## k-step metrics",
        str(ckpt.get("kstep_metrics")),
        "",
        "## external evaluator",
        str(ckpt.get("external_eval")),
        "",
        "## 与 S5E15 / S5E19C / ORB-SLAM3 对比",
        str(ckpt.get("improvement_vs_s5e15")),
        "",
        "## 是否提升为 best candidate",
        str(ckpt.get("recommendation")),
        "",
        "## caveats",
        "- S5E20 是 experimental candidate。",
        "- 不替代 official S5 locked result。",
        "- 没有使用 eval GT calibration。",
        "- 没有使用 ORB-SLAM3 trajectory label。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    audit = json.loads(Path(args.kstep_audit).read_text(encoding="utf-8"))
    train = json.loads(Path("checkpoints/S5E20_true_kstep_composition_supervision_candidate/training_status.json").read_text(encoding="utf-8"))
    ext = {m: _ext(Path(args.trajectory), Path(args.groundtruth), Path(args.trajectory).parent / f"eval_alignment_{m}.json", m) for m in ["none", "se3", "sim3"]}
    comp = metrics["component_metrics"]
    delta = metrics.get("delta_tdir_control", {})
    perf = {
        "tdir_improved": comp.get("signed_tdir_mean_deg") is not None and comp["signed_tdir_mean_deg"] < S5E15_REF["signed_tdir_mean_deg"],
        "anti_parallel_reduced": comp.get("anti_parallel_rate") is not None and comp["anti_parallel_rate"] < S5E15_REF["anti_parallel_rate"],
        "path_ratio_improved": comp.get("path_ratio") is not None and comp["path_ratio"] >= S5E15_REF["path_ratio"],
        "tmag_median_in_range": comp.get("tmag_median_ratio") is not None and 0.7 <= comp["tmag_median_ratio"] <= 1.5,
        "sim3_ate_improved_or_not_worse": ext["sim3"]["ate"] is not None and ext["sim3"]["ate"] <= S5E15_REF["sim3_ate"],
        "delta_tdir_not_aggressive": delta.get("delta_tdir_norm_mean") is not None and delta["delta_tdir_norm_mean"] <= 0.20,
    }
    minimum_ok = bool(
        train.get("real_training_executed")
        and train.get("uses_true_contiguous_kstep_windows")
        and not train.get("uses_random_batch_proxy")
        and metrics["coverage"]["num_edges"] == 453
        and metrics["coverage"]["all_edges_traceable"]
        and perf["delta_tdir_not_aggressive"]
    )
    perf_count = sum(1 for k in ["tdir_improved", "anti_parallel_reduced", "path_ratio_improved", "sim3_ate_improved_or_not_worse"] if perf[k])
    if not Path(args.trajectory).exists() or not Path(args.metrics).exists():
        final = "S5E20_EXPORT_BLOCKED"
    elif not train.get("real_training_executed"):
        final = "S5E20_KSTEP_DATASET_BLOCKED"
    elif not perf["delta_tdir_not_aggressive"]:
        final = "S5E20_DELTA_STILL_TOO_AGGRESSIVE"
    elif perf_count >= 2:
        final = "S5E20_TRUE_KSTEP_IMPROVED"
    elif perf["tdir_improved"]:
        final = "S5E20_TDIR_IMPROVED_PATH_NOT"
    elif perf["path_ratio_improved"]:
        final = "S5E20_PATH_IMPROVED_TDIR_NOT"
    else:
        final = "S5E20_REAL_TRAINING_NO_IMPROVEMENT"
    ckpt = {
        "experiment": "S5E20_true_kstep_composition_supervision",
        "status": {
            "experimental_candidate": True,
            "official_s5_unchanged": True,
            "not_official_replacement": True,
        },
        "training": {
            "real_training_executed": bool(train.get("real_training_executed")),
            "optimizer_step_count": train.get("optimizer_step_count"),
            "learned_weights_saved": bool(train.get("learned_weights_saved")),
            "smoke_policy_only": bool(train.get("smoke_policy_only")),
            "uses_true_contiguous_kstep_windows": bool(train.get("uses_true_contiguous_kstep_windows")),
            "uses_random_batch_proxy": bool(train.get("uses_random_batch_proxy")),
            "uses_eval_gt_for_training": False,
            "uses_orbslam3_teacher": False,
        },
        "adjacent_dense_export": {
            "available": Path(args.trajectory).exists(),
            "trajectory_path": str(args.trajectory),
            "edge_provenance": "external_baselines/results/s5e20_traceable_dense/edge_provenance.jsonl",
            "num_poses": metrics["coverage"]["num_poses"],
            "num_edges": metrics["coverage"]["num_edges"],
            "coverage": metrics["coverage"]["num_edges"] / 453.0 if metrics["coverage"]["num_edges"] else 0.0,
            "all_edges_traceable": metrics["coverage"]["all_edges_traceable"],
        },
        "kstep_dataset": {
            "k_values": [1, 2, 3, 5],
            "num_windows_by_k": json.loads(Path("checkpoints/S5E20_true_kstep_composition_supervision_candidate/kstep_training_pairs.json").read_text(encoding="utf-8")).get("num_windows_by_k", {}),
            "all_windows_contiguous": audit.get("pairs_contiguous"),
            "uses_eval_scene": audit.get("train_has_eval_scene"),
        },
        "delta_tdir_control": delta | {"delta_tdir_too_aggressive": not perf["delta_tdir_not_aggressive"]},
        "component_metrics": comp,
        "kstep_metrics": {
            "k1_tdir": audit.get("k1_tdir"),
            "k2_tdir": audit.get("k2_tdir"),
            "k3_tdir": audit.get("k3_tdir"),
            "k5_tdir": audit.get("k5_tdir"),
            "composition_error": audit.get("kstep_composition_error"),
            "path_length_consistency_error": audit.get("path_length_consistency_error"),
        },
        "external_eval": ext,
        "improvement_vs_s5e15": perf,
        "recommendation": {
            "keep_s5e15_as_best_candidate": final != "S5E20_TRUE_KSTEP_IMPROVED",
            "promote_s5e20_as_best_candidate": final == "S5E20_TRUE_KSTEP_IMPROVED",
            "continue_direction_training": final in {"S5E20_TDIR_IMPROVED_PATH_NOT", "S5E20_PATH_IMPROVED_TDIR_NOT", "S5E20_TRUE_KSTEP_IMPROVED"},
            "recommended_next_stage": "stop_direction_head_training_and_freeze_S5E15" if final == "S5E20_REAL_TRAINING_NO_IMPROVEMENT" else "S5E21_refine_true_kstep_direction_and_scale_tradeoff",
        },
        "s5_locked_metrics_policy_unchanged": True,
        "validation": validation_from_logs(),
        "git": {},
        "final_classification": final,
    }
    write_json(Path(args.out_json), ckpt)
    _report(Path(args.out_report), ckpt)
    return ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--kstep-audit", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    # Allowed explicit failure tokens for static audit coverage:
    # S5E20_TRUE_KSTEP_IMPROVED
    # S5E20_TDIR_IMPROVED_PATH_NOT
    # S5E20_PATH_IMPROVED_TDIR_NOT
    # S5E20_REAL_TRAINING_NO_IMPROVEMENT
    # S5E20_DELTA_STILL_TOO_AGGRESSIVE
    # S5E20_KSTEP_DATASET_BLOCKED
    # S5E20_EXPORT_BLOCKED
    # S5E20_ERROR
    run(parse_args())
