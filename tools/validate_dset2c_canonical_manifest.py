#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from miniyaml import load_yaml_like
from s5e2_adjacent_dense_lib import write_json


ALLOWED_CLASSIFICATIONS = [
    "DSET2C_CANONICAL_MANIFEST_READY",
    "DSET2C_TRAIN360_READY_AFTER_HYGIENE",
    "DSET2C_TRAIN360_NOT_READY_AFTER_CLEANING",
    "DSET2C_DSET2B_MANIFEST_REFERENCES_BAD_SEQUENCES",
    "DSET2C_LOCAL_TREE_DIRTY_BUT_CANONICAL_READY",
    "DSET2C_DATASET_HYGIENE_ERROR",
]
QUARANTINE_PREFIX = "QUARANTINE_"


def _load_cfg(path: Path) -> Dict[str, Any]:
    return load_yaml_like(path) or {}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"num_pairs": 0}
    tmag = np.asarray([float(row["tmag"]) for row in rows], dtype=np.float64)
    return {
        "num_pairs": len(rows),
        "num_adjacent_pairs": sum(1 for row in rows if row.get("pair_type") == "adjacent"),
        "num_kstep_pairs": sum(1 for row in rows if row.get("pair_type") == "kstep"),
        "small_motion_fraction": float(np.mean(tmag < 0.005)),
        "very_small_motion_fraction": float(np.mean(tmag < 0.002)),
    }


def _report_lines(
    audit: Dict[str, Any],
    summary: Dict[str, Any],
    excluded: Dict[str, Any],
    readiness: Dict[str, Any],
    recommendation: Dict[str, Any],
    final_classification: str,
) -> List[str]:
    status_counts = audit.get("status_counts", {})
    included = summary.get("sequences_all", [])
    lines = [
        "# DSET2C 360DVO dataset hygiene report",
        "",
        "## 1. 为什么做 DSET2C",
        "DSET2B 已经达到 TRAIN360_READY，但本地 data/360DVO 仍混有 partial download、空图像目录、pose-only 元数据和未进入训练的旁支 sequence，因此 DSET2C 的目标是把可用数据与 quarantine 元数据化，并固化 canonical manifest。",
        "",
        "## 2. 本地 data/360DVO 为什么看起来乱",
        "本地目录同时承载了多轮下载、partial 补齐、HF cache 和不同长度 sequence，因此仅看文件树会同时出现完整序列、半下载序列、纯 pose/timestamp 残留和空目录。",
        "",
        "## 3. 哪些乱是正常的：sequence 长度不同",
        "不同 sequence 本来就有不同长度；只要 image/pose/timestamp 可以形成稳定 pair，并且不直接依赖原始目录 glob 训练，这种长度差异不是问题。",
        "",
        "## 4. 哪些乱是风险：空图像目录、pose-only、image-only、低 pair 数",
        "风险主要来自 QUARANTINE_* sequence：空图像目录、图像缺失、pose/timestamp 缺失、以及 pair 数不足以稳定进入 TRAIN360 的序列。",
        "",
        "## 5. 每类 sequence 的数量",
        f"- status_counts: {json.dumps(status_counts, ensure_ascii=False)}",
        "",
        "## 6. 被纳入 canonical manifest 的 sequence",
        f"- included_sequences: {included}",
        "",
        "## 7. 被排除的 sequence 及原因",
        f"- excluded_sequences: {json.dumps(excluded, ensure_ascii=False)}",
        "",
        "## 8. canonical train/val/test pair 数",
        f"- num_pairs_train: {summary.get('num_pairs_train')}",
        f"- num_pairs_val: {summary.get('num_pairs_val')}",
        f"- num_pairs_test: {summary.get('num_pairs_test')}",
        f"- num_adjacent_pairs_train: {summary.get('num_adjacent_pairs_train')}",
        f"- num_adjacent_pairs_val: {summary.get('num_adjacent_pairs_val')}",
        f"- num_adjacent_pairs_test: {summary.get('num_adjacent_pairs_test')}",
        "",
        "## 9. 是否仍 TRAIN360_READY",
        f"- readiness: {json.dumps(readiness, ensure_ascii=False)}",
        f"- final_classification: {final_classification}",
        "",
        "## 10. 后续所有训练必须只读 canonical manifest",
        "- 后续 TRAIN360 / BASE360 训练必须只读取 external_baselines/results/dset2c_360dvo_canonical 下的 canonical manifest，不再直接 glob data/360DVO 原始目录。",
        "",
        "## 11. caveats",
        "- DSET2B 曾引用 partial 但高收益的 sequence；DSET2C 已将低于 hygiene 阈值的序列排除出 canonical manifest。",
        "- 本轮不训练、不 fine-tune、不删除 raw data。",
        "- S5 locked metrics/policy were not changed.",
    ]
    return lines


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_cfg(Path(args.config))
    audit = _read_json(Path(args.local_tree_audit))
    canonical_dir = Path(args.canonical_dir)
    summary = _read_json(canonical_dir / "canonical_manifest_summary.json")
    excluded = _read_json(canonical_dir / "excluded_sequences.json")
    train_rows = _read_jsonl(canonical_dir / "pair_manifest_train.jsonl")
    val_rows = _read_jsonl(canonical_dir / "pair_manifest_val.jsonl")
    test_rows = _read_jsonl(canonical_dir / "pair_manifest_test.jsonl")
    all_rows = _read_jsonl(canonical_dir / "pair_manifest_all.jsonl")
    dset2b = _read_json(Path(str(cfg.get("source", {}).get("dset2b_checkpoint", ""))))
    target = dset2b.get("readiness", {})
    audit_sequences = audit.get("sequences", {})

    missing_paths = []
    bad_pose_fields = 0
    quarantine_used = []
    for row in all_rows:
        for key in ["image_path_a", "image_path_b"]:
            if not Path(str(row.get(key, ""))).exists():
                missing_paths.append(str(row.get(key, "")))
        if "R_BA" not in row or "t_BA_B" not in row or "tdir_B" not in row:
            bad_pose_fields += 1
        seq = str(row.get("seq_id", ""))
        if str(audit_sequences.get(seq, {}).get("recommended_status", "")).startswith(QUARANTINE_PREFIX):
            quarantine_used.append(seq)

    train_set = set(summary.get("train_sequences", []))
    val_set = set(summary.get("val_sequences", []))
    test_set = set(summary.get("test_sequences", []))
    overlap_free = not (train_set & val_set or train_set & test_set or val_set & test_set)
    train_stats = _stats(train_rows)
    val_stats = _stats(val_rows)
    test_stats = _stats(test_rows)
    all_stats = _stats(all_rows)

    readiness = {
        "train360_ready_after_hygiene": False,
        "train_pairs_ok": train_stats.get("num_pairs", 0) >= 2000,
        "val_pairs_ok": val_stats.get("num_pairs", 0) >= 300,
        "test_pairs_ok": test_stats.get("num_pairs", 0) >= 300,
        "sequence_split_ok": bool(
            summary.get("split_by_sequence")
            and summary.get("forbid_random_pair_split")
            and overlap_free
        ),
        "no_quarantine_sequence_used": not quarantine_used,
    }
    readiness["adjacent_train_ok"] = summary.get("num_adjacent_pairs_train", 0) >= 400
    readiness["adjacent_val_ok"] = summary.get("num_adjacent_pairs_val", 0) >= 80
    readiness["adjacent_test_ok"] = summary.get("num_adjacent_pairs_test", 0) >= 80
    readiness["small_motion_risk_low"] = all_stats.get("small_motion_fraction", 1.0) <= 0.05
    readiness["train360_ready_after_hygiene"] = all(
        readiness[key]
        for key in [
            "train_pairs_ok",
            "val_pairs_ok",
            "test_pairs_ok",
            "sequence_split_ok",
            "no_quarantine_sequence_used",
            "adjacent_train_ok",
            "adjacent_val_ok",
            "adjacent_test_ok",
            "small_motion_risk_low",
        ]
    )

    dset2b_bad_refs = [
        seq for seq in dset2b.get("manifest", {}).get("sequences_all", [])
        if str(audit_sequences.get(seq, {}).get("recommended_status", "")).startswith(QUARANTINE_PREFIX)
    ]
    canonical_seq_set = set(summary.get("sequences_all", []))
    was_repaired = bool(dset2b_bad_refs and any(seq not in canonical_seq_set for seq in dset2b_bad_refs))
    has_dirty_tree = any(
        str(rec.get("recommended_status", "")).startswith(QUARANTINE_PREFIX)
        for rec in audit_sequences.values()
    )

    if missing_paths or bad_pose_fields:
        final = "DSET2C_DATASET_HYGIENE_ERROR"
    elif readiness["train360_ready_after_hygiene"] and was_repaired:
        final = "DSET2C_TRAIN360_READY_AFTER_HYGIENE"
    elif readiness["train360_ready_after_hygiene"] and has_dirty_tree:
        final = "DSET2C_CANONICAL_MANIFEST_READY"
    elif dset2b_bad_refs:
        final = "DSET2C_DSET2B_MANIFEST_REFERENCES_BAD_SEQUENCES"
    elif readiness["train360_ready_after_hygiene"]:
        final = "DSET2C_LOCAL_TREE_DIRTY_BUT_CANONICAL_READY"
    else:
        final = "DSET2C_TRAIN360_NOT_READY_AFTER_CLEANING"
    if final not in ALLOWED_CLASSIFICATIONS:
        final = "DSET2C_DATASET_HYGIENE_ERROR"

    recommendation = {
        "do_train360_baseline": readiness["train360_ready_after_hygiene"],
        "download_more_sequences": not readiness["train360_ready_after_hygiene"],
        "use_canonical_manifest_only": True,
        "do_base360_baseline": True,
    }
    payload = {
        "experiment": "DSET2C_360DVO_dataset_hygiene",
        "local_tree_audit": audit,
        "canonical_manifest": {
            "summary": summary,
            "validation_stats": {
                "all": all_stats,
                "train": train_stats,
                "val": val_stats,
                "test": test_stats,
            },
            "missing_image_paths": missing_paths,
            "bad_pose_field_count": bad_pose_fields,
            "dset2b_bad_references": dset2b_bad_refs,
            "no_raw_data_committed_policy_written": bool(cfg.get("dataset", {}).get("do_not_commit_raw_data", False)),
        },
        "excluded_sequences": excluded,
        "readiness": readiness,
        "recommendation": recommendation,
        "s5_locked_metrics_policy_unchanged": True,
        "final_classification": final,
    }
    write_json(Path(args.out_json), payload)
    Path(args.out_report).write_text(
        "\n".join(
            _report_lines(
                audit=audit,
                summary=summary,
                excluded=excluded,
                readiness=readiness,
                recommendation=recommendation,
                final_classification=final,
            )
        ) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-tree-audit", required=True)
    parser.add_argument("--canonical-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
