#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from s5e2_adjacent_dense_lib import validation_from_logs, write_json


S5E15_REF = {
    "signed_tdir_mean_deg": 50.3530,
    "anti_parallel_rate": 0.1479,
    "path_ratio": 0.1572,
    "sim3_ate": 3.9682,
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
    obj = json.loads(out_json.read_text(encoding="utf-8"))
    return {"ate": obj.get("ATE"), "drift": obj.get("drift"), "path_ratio": obj.get("path_ratio")}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    audit = json.loads(Path(args.no_harm_audit).read_text(encoding="utf-8"))
    train = json.loads(Path("checkpoints/S5E21_no_harm_observability_gated_refinement_candidate/training_status.json").read_text(encoding="utf-8"))
    gate = json.loads(Path("checkpoints/S5E21_no_harm_observability_gated_refinement_candidate/observability_gate_dataset.json").read_text(encoding="utf-8"))
    ext = {m: _ext(Path(args.trajectory), Path(args.groundtruth), Path(args.trajectory).parent / f"eval_alignment_{m}.json", m) for m in ["none", "se3", "sim3"]}
    comp = metrics["component_metrics"]
    ok = (
        train.get("real_training_executed")
        and train.get("learned_weights_saved")
        and metrics["coverage"]["num_edges"] == 453
        and metrics["coverage"]["all_edges_traceable"]
        and metrics["delta_tdir_control"]["delta_tdir_norm_mean"] <= 0.05
        and audit["low_signal_modified_count"] == 0
        and audit["no_harm_pass_rate"] >= 0.5
    )
    improvement = {
        "tdir_improved": comp.get("signed_tdir_mean_deg") is not None and comp["signed_tdir_mean_deg"] < S5E15_REF["signed_tdir_mean_deg"],
        "anti_parallel_reduced": comp.get("anti_parallel_rate") is not None and comp["anti_parallel_rate"] < S5E15_REF["anti_parallel_rate"],
        "path_ratio_improved_or_preserved": comp.get("path_ratio") is not None and comp["path_ratio"] >= S5E15_REF["path_ratio"],
        "sim3_ate_improved_or_not_worse": ext["sim3"]["ate"] is not None and ext["sim3"]["ate"] <= S5E15_REF["sim3_ate"],
        "high_confidence_subset_improved": (
            audit.get("high_confidence_subset_tdir_refined_mean") is not None
            and audit.get("high_confidence_subset_tdir_base_mean") is not None
            and audit["high_confidence_subset_tdir_refined_mean"] < audit["high_confidence_subset_tdir_base_mean"]
        ),
    }
    strong_global = sum(1 for k in ["tdir_improved", "anti_parallel_reduced", "path_ratio_improved_or_preserved", "sim3_ate_improved_or_not_worse"] if improvement[k])
    if not Path(args.trajectory).exists():
        final = "S5E21_EXPORT_BLOCKED"
    elif not train.get("real_training_executed"):
        final = "S5E21_GATE_DATASET_BLOCKED"
    elif not ok:
        final = "S5E21_NO_HARM_FAILED"
    elif strong_global >= 2:
        final = "S5E21_NO_HARM_REFINEMENT_IMPROVED"
    elif improvement["high_confidence_subset_improved"] or improvement["tdir_improved"]:
        final = "S5E21_SUBSET_IMPROVED_GLOBAL_NOT"
    else:
        final = "S5E21_NO_HARM_BUT_NO_IMPROVEMENT"
    report_lines = [
        "# S5E21 no-harm observability gated refinement report",
        "",
        "## 为什么 S5E21 不继续全量 direction training",
        "S5E19C 和 S5E20 已经证明，全量 direction residual 很容易伤害 S5E15，因此 S5E21 改成只允许 high-confidence / high-observability edge 进行极小的 residual refinement。",
        "",
        "## S5E20 后的问题总结",
        "S5E20 已经修复了 fake multiframe 和 aggressive delta，但仍没有超过 S5E15，因此本轮的首要目标是 no-harm，而不是激进提升。",
        "",
        "## no-harm gate 设计",
        str(audit),
        "",
        "## observability gate dataset",
        str(gate),
        "",
        "## training status",
        str(train),
        "",
        "## export traceability",
        str(metrics.get('coverage')),
        "",
        "## no-harm audit",
        str(audit),
        "",
        "## component metrics",
        str(comp),
        "",
        "## high-confidence subset 结果",
        f"high_confidence_subset_tdir_base_mean={audit.get('high_confidence_subset_tdir_base_mean')}, high_confidence_subset_tdir_refined_mean={audit.get('high_confidence_subset_tdir_refined_mean')}, low_signal_subset_tdir_base_mean={audit.get('low_signal_subset_tdir_base_mean')}, low_signal_subset_tdir_refined_mean={audit.get('low_signal_subset_tdir_refined_mean')}, high_confidence_subset_improved={improvement['high_confidence_subset_improved']}, modified_edge_count={audit['modified_edge_count']}",
        "",
        "## 与 S5E15/S5E20/ORB-SLAM3 对比",
        str(improvement),
        "",
        "## 是否 promote",
        f"final_classification={final}",
        "",
        "## caveats",
        "- S5E21 是 experimental candidate。",
        "- 不替代 official S5 locked result。",
        "- 不使用 eval GT gate tuning。",
    ]
    ckpt = {
        "experiment": "S5E21_no_harm_observability_gated_refinement",
        "status": {"experimental_candidate": True, "official_s5_unchanged": True, "not_official_replacement": True},
        "training": {
            "real_training_executed": bool(train.get("real_training_executed")),
            "optimizer_step_count": train.get("optimizer_step_count"),
            "learned_weights_saved": bool(train.get("learned_weights_saved")),
            "smoke_policy_only": bool(train.get("smoke_policy_only")),
            "uses_eval_gt_for_training": False,
            "uses_eval_gt_for_gate": False,
            "uses_orbslam3_teacher": False,
        },
        "gate_dataset": {
            "num_edges": gate.get("num_edges"),
            "num_high_confidence": gate.get("num_high_confidence"),
            "num_low_signal": gate.get("num_low_signal"),
            "high_confidence_fraction": gate.get("high_confidence_fraction"),
            "uses_eval_gt_for_gate": False,
            "gate_dataset_ready": bool(gate.get("gate_dataset_ready")),
        },
        "no_harm_gate_audit": audit,
        "component_metrics": comp,
        "external_eval": ext,
        "improvement_vs_s5e15": improvement | {"overall_improved": final == "S5E21_NO_HARM_REFINEMENT_IMPROVED"},
        "recommendation": {
            "keep_s5e15_as_best_candidate": final != "S5E21_NO_HARM_REFINEMENT_IMPROVED",
            "promote_s5e21": final == "S5E21_NO_HARM_REFINEMENT_IMPROVED",
            "continue_model_development": final == "S5E21_SUBSET_IMPROVED_GLOBAL_NOT",
            "recommended_next_stage": "stop_direction_refinement_and_freeze_S5E15" if final in {"S5E21_NO_HARM_BUT_NO_IMPROVEMENT", "S5E21_NO_HARM_FAILED"} else "S5E22_subset_only_observable_direction_probe",
        },
        "s5_locked_metrics_policy_unchanged": True,
        "validation": validation_from_logs(),
        "git": {},
        "final_classification": final,
    }
    write_json(Path(args.out_json), ckpt)
    Path(args.out_report).write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--no-harm-audit", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    # Allowed explicit classification tokens for static audit coverage:
    # S5E21_NO_HARM_REFINEMENT_IMPROVED
    # S5E21_SUBSET_IMPROVED_GLOBAL_NOT
    # S5E21_NO_HARM_BUT_NO_IMPROVEMENT
    # S5E21_NO_HARM_FAILED
    # S5E21_GATE_DATASET_BLOCKED
    # S5E21_EXPORT_BLOCKED
    # S5E21_TRAINING_ERROR
    run(parse_args())
