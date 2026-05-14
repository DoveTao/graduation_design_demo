#!/usr/bin/env python3
from __future__ import annotations


def main() -> None:
    lines = [
        "Current thesis pair-level mainline:",
        "  model: FINAL360I_struct360b_final_selected",
        "  config: configs/final360i_struct360b_final.yaml",
        "  trainer: tools/final360i_retrain_and_select.py",
        "  base model: models/struct360b_match_free_coarse_to_fine.py",
        "  dataset: datasets/dset2c_manifest_dataset.py",
        "Trajectory evaluation path:",
        "  tool: tools/train360e_sequence_trajectory_export_and_ate_eval.py",
        "Retained variant:",
        "  SEQ360B: tools/train_seq360b_lightweight_scale_smoothing.py",
        "  head: models/seq360b_scale_smoothing_head.py",
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

