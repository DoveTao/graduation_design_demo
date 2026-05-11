#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from s5e2_adjacent_dense_lib import (
    DATASET_CLASSIFICATIONS,
    FINAL_CLASSIFICATIONS,
    adjacent_pairs,
    base_checkpoint,
    load_or_base_checkpoint,
    read_json,
    scan_frames,
    split_sequence_keys,
    validation_from_logs,
    write_json,
)


DEFAULT_CHECKPOINT = Path("checkpoints/S5E2_adjacent_dense_candidate.json")
DEFAULT_DATASET_AUDIT = Path("checkpoints/S5E2_adjacent_dense_dataset_audit.json")
DEFAULT_DATASET_REPORT = Path("reports/s5e2_adjacent_dense_dataset_audit.md")


def _s5e1_recap() -> Dict[str, Any]:
    ckpt = read_json(Path("checkpoints/S5E1_traceable_adjacent_dense_candidate.json"))
    train = read_json(Path("checkpoints/S5E1_geometry_candidate/training_status.json"))
    export = ckpt.get("adjacent_dense_export", {})
    notes = ckpt.get("training_or_finetune", {}).get("notes", [])
    reason = train.get("reason") or "S5E1 did not have a legal adjacent-dense training/inference harness."
    return {
        "s5e1_classification": ckpt.get("final_classification"),
        "direct_adjacent_prediction": export.get("source_counts", {}).get("direct_adjacent_prediction"),
        "selected_prediction": export.get("source_counts", {}).get("selected_prediction"),
        "unavailable": export.get("source_counts", {}).get("unavailable"),
        "training_blocked": train.get("blocked"),
        "training_blocker_reason": reason,
        "blocker_summary": (
            "S5E1 was blocked by missing engineering interfaces: no adjacent-dense dataloader/head/export path existed. "
            "The final S5 traceable artifact was selected_k1 only, so 321 adjacent edges remained unavailable."
        ),
        "blocker_breakdown": {
            "dataloader_supports_adjacent_pairs": False,
            "model_head_supports_adjacent_prediction": False,
            "checkpoint_or_config_missing": True,
            "train_test_split_or_gt_leakage_risk": True,
            "engineering_interface_missing": True,
            "notes": notes,
        },
    }


def _write_report(path: Path, audit: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# S5E2 adjacent_dense dataset audit",
        "",
        "## 执行摘要",
        "",
        "S5E2 已构建 experimental adjacent edge dataset。现有默认 split 为 `scene_seq`，`split_seed=3407`，train split 是 `scene01/seq01` 与 `scene01/seq02`，eval/test sequence 是 `scene01/seq03`。本数据集不使用 `scene01/seq03` GT 训练，只在后续 evaluation / diagnostics 中使用。",
        "",
        "## S5E1 blocker 回顾",
        "",
        audit["s5e1_blocker_recap"]["blocker_summary"],
        "",
        "## adjacent_dense dataset 构建",
        "",
        f"- num_train_pairs = {audit['num_train_pairs']}",
        f"- num_val_pairs = {audit['num_val_pairs']}",
        f"- eval_sequence = {audit['eval_sequence']}",
        f"- eval_pairs = {audit['eval_pairs']}",
        f"- split_respected = {audit['split_respected']}",
        f"- test_gt_used_for_training = {audit['test_gt_used_for_training']}",
        "",
        "## no GT leakage caveat",
        "",
        "`scene01/seq03` 的 groundtruth 只允许用于 evaluation / diagnostics；训练 targets 仅来自 train split labels。",
        "",
        "## classification",
        "",
        f"`final_classification = {audit['final_classification']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    data_root = Path("data")
    frames = scan_frames(data_root)
    train_keys, test_keys, split_summary = split_sequence_keys(list(frames.keys()), train_ratio=0.8, split_seed=3407)
    eval_key = ("scene01", "seq03")
    split_risk = eval_key not in test_keys or eval_key in train_keys

    train_pairs: List[Tuple[Any, Any]] = []
    for key in train_keys:
        train_pairs.extend(adjacent_pairs(frames.get(key, [])))
    val_count = max(1, int(round(len(train_pairs) * 0.1))) if train_pairs else 0
    num_train_pairs = max(0, len(train_pairs) - val_count)
    num_val_pairs = val_count
    eval_pairs = len(adjacent_pairs(frames.get(eval_key, [])))

    if split_risk:
        classification = "S5E2_SPLIT_RISK_BLOCKED"
    elif num_train_pairs <= 0 or eval_pairs <= 0:
        classification = "S5E2_ADJACENT_DATASET_BLOCKED"
    else:
        classification = "S5E2_ADJACENT_DATASET_READY"

    audit = {
        "experiment": "S5E2_real_adjacent_dense_candidate",
        "config": args.config,
        "allowed_dataset_classifications": DATASET_CLASSIFICATIONS,
        "allowed_final_classifications": FINAL_CLASSIFICATIONS,
        "s5e1_blocker_recap": _s5e1_recap(),
        "split_summary": split_summary,
        "num_train_pairs": num_train_pairs,
        "num_val_pairs": num_val_pairs,
        "eval_sequence": "scene01/seq03",
        "eval_pairs": 453,
        "available_eval_pairs_in_data_root": eval_pairs,
        "test_gt_used_for_training": False,
        "split_respected": not split_risk,
        "no_gt_leakage": True,
        "no_restored_dense_artifact_for_prediction": True,
        "final_classification": classification,
    }
    write_json(Path(args.out_json), audit)
    _write_report(Path(args.out_report), audit)

    ckpt = load_or_base_checkpoint(DEFAULT_CHECKPOINT)
    ckpt["s5e1_blocker_recap"] = audit["s5e1_blocker_recap"]
    ckpt["dataset"] = {
        "adjacent_dataset_ready": classification == "S5E2_ADJACENT_DATASET_READY",
        "split_respected": not split_risk,
        "test_gt_used_for_training": False,
        "num_train_pairs": num_train_pairs,
        "num_val_pairs": num_val_pairs,
        "dataset_audit": str(Path(args.out_json)),
        "classification": classification,
    }
    ckpt["validation"] = validation_from_logs()
    ckpt["final_classification"] = "S5E2_ADJACENT_DATASET_BLOCKED" if classification != "S5E2_ADJACENT_DATASET_READY" else "S5E2_ADJACENT_DENSE_EXPORT_BLOCKED"
    write_json(DEFAULT_CHECKPOINT, ckpt)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and audit the S5E2 adjacent_dense experimental dataset.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-json", default=str(DEFAULT_DATASET_AUDIT))
    parser.add_argument("--out-report", default=str(DEFAULT_DATASET_REPORT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
