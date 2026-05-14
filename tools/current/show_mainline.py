#!/usr/bin/env python3
from __future__ import annotations


def main() -> None:
    lines = [
        "Current thesis pair-level mainline:",
        "  model: FINAL360I_struct360b_final_selected",
        "  structure lineage: STRUCT360B_match_free_coarse_to_fine",
        "  config: configs/final360i_struct360b_final.yaml",
        "  trainer: tools/final360i_retrain_and_select.py",
        "  base model: models/struct360b_match_free_coarse_to_fine.py",
        "  dataset: datasets/dset2c_manifest_dataset.py",
        "Trajectory evaluation path:",
        "  tool: tools/train360e_sequence_trajectory_export_and_ate_eval.py",
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
