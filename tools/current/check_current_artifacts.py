#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED = {
    "readme": "README.md",
    "current_mainline_doc": "CURRENT_MAINLINE.md",
    "pair_main_config": "configs/final360m_fulltrain_struct360b_thesis_main_guarded.yaml",
    "pair_base_config": "configs/struct360b_match_free_coarse_to_fine.yaml",
    "seq360b_config": "configs/seq360b_lightweight_scale_smoothing.yaml",
    "pair_main_tool": "tools/train_final360m_fulltrain_thesis_main.py",
    "trajectory_eval_tool": "tools/final360m_trajectory_thesis_refresh.py",
    "odom360a_eval_tool": "tools/odom360a_lightweight_trajectory_fusion.py",
    "odom360b_eval_tool": "tools/odom360b_local_pose_graph_kstep.py",
    "struct360b_tool": "tools/train_struct360b_match_free_coarse_to_fine.py",
    "seq360b_tool": "tools/train_seq360b_lightweight_scale_smoothing.py",
    "pair_main_model": "models/struct360b_match_free_coarse_to_fine.py",
    "seq360b_model": "models/seq360b_scale_smoothing_head.py",
    "core_model": "train360/core/model.py",
    "manifest_dataset": "datasets/dset2c_manifest_dataset.py",
    "sequence_dataset": "datasets/dset2c_sequence_clip_dataset.py",
    "final360m_report": "reports/FINAL360M_metrics_test.json",
    "final360m_trajectory_report": "reports/FINAL360M_trajectory_metrics_test.json",
    "train360e_report": "reports/TRAIN360E_metrics_test.json",
    "seq360b_report": "reports/SEQ360B_metrics_test.json",
    "base360d_report": "reports/BASE360D_component_metric_alignment.md",
    "canonical_manifest": "external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl",
}

OPTIONAL = {
    "seq360a_status_summary": "reports/SEQ360A_status_summary.md",
    "struct360c_status_summary": "reports/STRUCT360C_status_summary.md",
    "final360i_subset_candidate_report": "reports/FINAL360I_metrics_test.json",
    "final360m_mainline_promotion_report": "reports/CURRENT_MAINLINE_FINAL360M_promotion.md",
}

INTENTIONALLY_OMITTED_AFTER_CLEANUP = {
    "base360d_val_metrics": "reports/BASE360D_metrics_val.json",
    "final360i_previous_summary": "reports/FINAL360I_vs_all_baselines_summary.md",
    "seq360b_trajectory_dir": "external_baselines/results/seq360b_scale_smoothing_trajectory",
    "seq360b_previous_val_metrics": "reports/SEQ360B_metrics_val.json",
    "seq360b_previous_trajectory_val_metrics": "reports/SEQ360B_trajectory_metrics_val.json",
    "seq360b_previous_summary": "reports/SEQ360B_vs_FINAL360I_TRAIN360E_BASE360D_summary.md",
}


def main() -> None:
    payload = {}
    for key, rel in REQUIRED.items():
        path = REPO_ROOT / rel
        payload[key] = {"path": rel, "exists": path.exists()}
    payload["all_present"] = all(v["exists"] for v in payload.values())
    payload["optional"] = {}
    for key, rel in OPTIONAL.items():
        path = REPO_ROOT / rel
        payload["optional"][key] = {"path": rel, "exists": path.exists()}
    payload["intentionally_omitted_after_cleanup"] = {}
    for key, rel in INTENTIONALLY_OMITTED_AFTER_CLEANUP.items():
        path = REPO_ROOT / rel
        payload["intentionally_omitted_after_cleanup"][key] = {
            "path": rel,
            "exists": path.exists(),
            "status": "intentionally_omitted_after_cleanup" if not path.exists() else "present",
        }
    payload["mainline_status"] = {
        "pair_main_model": "FINAL360M_fulltrain_struct360b_thesis_main_guarded",
        "old_final360i_status": "subset-trained candidate only",
        "trajectory_backend_refresh": "FINAL360M-ODOM360A recommended, but weaker than old subset-model-based ODOM360A",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
