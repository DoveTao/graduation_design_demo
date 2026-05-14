#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED = {
    "readme": "README.md",
    "current_mainline_doc": "CURRENT_MAINLINE.md",
    "pair_main_config": "configs/final360i_struct360b_final.yaml",
    "pair_base_config": "configs/struct360b_match_free_coarse_to_fine.yaml",
    "seq360b_config": "configs/seq360b_lightweight_scale_smoothing.yaml",
    "pair_main_tool": "tools/final360i_retrain_and_select.py",
    "trajectory_eval_tool": "tools/train360e_sequence_trajectory_export_and_ate_eval.py",
    "struct360b_tool": "tools/train_struct360b_match_free_coarse_to_fine.py",
    "seq360b_tool": "tools/train_seq360b_lightweight_scale_smoothing.py",
    "pair_main_model": "models/struct360b_match_free_coarse_to_fine.py",
    "seq360b_model": "models/seq360b_scale_smoothing_head.py",
    "core_model": "train360/core/model.py",
    "manifest_dataset": "datasets/dset2c_manifest_dataset.py",
    "sequence_dataset": "datasets/dset2c_sequence_clip_dataset.py",
    "final360i_report": "reports/FINAL360I_metrics_test.json",
    "train360e_report": "reports/TRAIN360E_metrics_test.json",
    "seq360b_report": "reports/SEQ360B_metrics_test.json",
    "canonical_manifest": "external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl",
}


def main() -> None:
    payload = {}
    for key, rel in REQUIRED.items():
        path = REPO_ROOT / rel
        payload[key] = {"path": rel, "exists": path.exists()}
    payload["all_present"] = all(v["exists"] for v in payload.values())
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

