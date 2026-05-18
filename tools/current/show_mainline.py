#!/usr/bin/env python3
from __future__ import annotations


def main() -> None:
    lines = [
        "Current thesis pair-level mainline:",
        "  model: FINAL360M_fulltrain_struct360b_thesis_main_guarded",
        "  structure lineage: STRUCT360B_match_free_coarse_to_fine",
        "  config: configs/final360m_fulltrain_struct360b_thesis_main_guarded.yaml",
        "  trainer: tools/train_final360m_fulltrain_thesis_main.py",
        "  selected checkpoint: checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded/best_full_val.pt",
        "  selection protocol: full-train, selected by full val only, test not used for selection",
        "  base model: models/struct360b_match_free_coarse_to_fine.py",
        "  dataset: datasets/dset2c_manifest_dataset.py",
        "Trajectory evaluation path:",
        "  recommended refreshed backend: FINAL360M-ODOM360A lightweight fusion",
        "  eval-only tools: tools/odom360a_lightweight_trajectory_fusion.py, tools/odom360b_local_pose_graph_kstep.py",
        "  summary report: reports/FINAL360M_trajectory_level_evaluation.md",
        "Current caveats:",
        "  FINAL360M is protocol-qualified, but not uniformly better than old FINAL360I on pair metrics",
        "  FINAL360M direct trajectory export is weak; lightweight fusion improves it",
        "  FINAL360M trajectory backend is still weaker than old subset-model-based ODOM360A",
        "Old model status:",
        "  FINAL360I_struct360b_final_selected: subset-trained candidate only (train subset = 256)",
        "Retained variant:",
        "  SEQ360B: tools/train_seq360b_lightweight_scale_smoothing.py",
        "  head: models/seq360b_scale_smoothing_head.py",
        "External baselines:",
        "  BASE360D: HKUST official 360DVO trajectory/component baseline",
        "  T57b: legacy recovered ERP pair baseline reference",
        "Archived status summaries:",
        "  SEQ360A: reports/SEQ360A_status_summary.md",
        "  STRUCT360C: reports/STRUCT360C_status_summary.md",
        "Core package:",
        "  train360/core/",
        "Quick docs:",
        "  README.md",
        "  CURRENT_MAINLINE.md",
        "  PROJECT_STRUCTURE.md",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
