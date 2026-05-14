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
    "DSET2B_TRAIN360_READY",
    "DSET2B_PAIR_COUNT_STILL_INSUFFICIENT",
    "DSET2B_DOWNLOAD_BLOCKED",
    "DSET2B_POSE_LIMITED_DATASET",
    "DSET2B_ADAPTER_LIMIT_BUG_SUSPECTED",
    "DSET2B_ERROR",
]


def _load_cfg(path: Path) -> Dict[str, Any]:
    return load_yaml_like(path) or {}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _split_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"num_pairs": 0}
    tmag = np.asarray([float(row["tmag"]) for row in rows], dtype=np.float64)
    rot_deg = []
    for row in rows:
        R = np.asarray(row["R_BA"], dtype=np.float64)
        cosine = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) * 0.5))
        rot_deg.append(float(np.rad2deg(np.arccos(cosine))))
    return {
        "num_pairs": len(rows),
        "num_adjacent_pairs": sum(1 for row in rows if row.get("pair_type") == "adjacent"),
        "num_kstep_pairs": sum(1 for row in rows if row.get("pair_type") == "kstep"),
        "gt_step_mean": float(np.mean(tmag)),
        "gt_step_median": float(np.median(tmag)),
        "gt_step_p10": float(np.percentile(tmag, 10)),
        "gt_step_p25": float(np.percentile(tmag, 25)),
        "gt_step_p75": float(np.percentile(tmag, 75)),
        "gt_step_p90": float(np.percentile(tmag, 90)),
        "gt_step_p95": float(np.percentile(tmag, 95)),
        "small_motion_fraction": float(np.mean(tmag < 0.005)),
        "very_small_motion_fraction": float(np.mean(tmag < 0.002)),
        "path_length": float(np.sum(tmag)),
        "rotation_step_mean": float(np.mean(rot_deg)),
    }


def _similarity(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    if not a.get("num_pairs") or not b.get("num_pairs"):
        return {"available": False}
    return {
        "available": True,
        "median_ratio": float(a["gt_step_median"] / max(float(b["gt_step_median"]), 1.0e-12)),
        "mean_ratio": float(a["gt_step_mean"] / max(float(b["gt_step_mean"]), 1.0e-12)),
        "small_motion_fraction_delta": float(a["small_motion_fraction"] - b["small_motion_fraction"]),
    }


def _final_classification(
    download: Dict[str, Any],
    sequence_audit: Dict[str, Any],
    readiness: Dict[str, Any],
) -> str:
    if download.get("access_blocked") or download.get("dependency_blocked"):
        return "DSET2B_DOWNLOAD_BLOCKED"
    if readiness["train360_ready"]:
        return "DSET2B_TRAIN360_READY"
    audit_records = sequence_audit.get("sequences", {})
    if any(rec.get("adapter_limit_suspected") for rec in audit_records.values()):
        return "DSET2B_ADAPTER_LIMIT_BUG_SUSPECTED"
    pose_limited = sum(1 for rec in audit_records.values() if rec.get("pose_limited_suspected"))
    if pose_limited >= max(1, len(audit_records) // 3):
        return "DSET2B_POSE_LIMITED_DATASET"
    if not readiness["train_pairs_ok"] or not readiness["val_pairs_ok"] or not readiness["test_pairs_ok"]:
        return "DSET2B_PAIR_COUNT_STILL_INSUFFICIENT"
    return "DSET2B_ERROR"


def _report_lines(
    remote_inventory: Dict[str, Any],
    download: Dict[str, Any],
    sequence_audit: Dict[str, Any],
    manifest: Dict[str, Any],
    motion_distribution: Dict[str, Any],
    readiness: Dict[str, Any],
    recommendation: Dict[str, Any],
    final_classification: str,
) -> List[str]:
    top_sequences = sorted(
        sequence_audit.get("sequences", {}).items(),
        key=lambda item: (
            int(item[1].get("valid_adjacent_pairs_possible", 0)) + int(item[1].get("valid_kstep_pairs_possible", 0)),
            float(item[1].get("valid_pose_fraction", 0.0)),
        ),
        reverse=True,
    )[:12]
    lines = [
        "# DSET2B 360DVO pair count completion report",
        "",
        "## 1. 为什么做 DSET2B",
        "DSET2 已经完成 fuller split 建档，但当前 blocker 是 pair 数量不足而不是 small-motion 风险，所以 DSET2B 的任务是补齐更多 360DVO sequence、重审完整性，并判断是否可以进入 TRAIN360_spherical_pose_baseline。",
        "",
        "## 2. 远端 sequence inventory",
        f"- repo_id: {remote_inventory.get('repo_id')}",
        f"- inventory_method: {remote_inventory.get('inventory_method')}",
        f"- num_sequences: {remote_inventory.get('num_sequences')}",
        f"- dependency_missing: {remote_inventory.get('dependency_missing')}",
        f"- blockers: {remote_inventory.get('blockers', [])}",
        "",
        "## 3. 下载/补齐 summary",
        f"- selected_sequences_total: {download.get('selected_sequences_total', [])}",
        f"- new_sequences_downloaded: {download.get('new_sequences_downloaded', [])}",
        f"- repaired_existing_sequences: {download.get('repaired_existing_sequences', [])}",
        f"- oversized_sequences_skipped_for_full_sync: {download.get('oversized_sequences_skipped_for_full_sync', [])}",
        f"- synthetic_timestamps_written: {download.get('synthetic_timestamps_written', [])}",
        f"- total_bytes_downloaded: {download.get('total_bytes_downloaded')}",
        f"- download_complete: {download.get('download_complete')}",
        f"- blockers: {download.get('blockers', [])}",
        "",
        "## 4. sequence completeness audit",
        f"- num_sequences_scanned: {sequence_audit.get('num_sequences_scanned')}",
        f"- top_complete_sequences: {[name for name, _ in top_sequences]}",
        f"- top_complete_sequence_stats: {json.dumps({name: rec for name, rec in top_sequences[:6]}, ensure_ascii=False)}",
        "",
        "## 5. train/val/test split",
        f"- train_sequences: {manifest.get('train_sequences', [])}",
        f"- val_sequences: {manifest.get('val_sequences', [])}",
        f"- test_sequences: {manifest.get('test_sequences', [])}",
        "",
        "## 6. pair manifest summary",
        f"- num_sequences: {manifest.get('num_sequences')}",
        f"- num_pairs_train: {manifest.get('num_pairs_train')}",
        f"- num_pairs_val: {manifest.get('num_pairs_val')}",
        f"- num_pairs_test: {manifest.get('num_pairs_test')}",
        f"- num_adjacent_pairs_train: {manifest.get('num_adjacent_pairs_train')}",
        f"- num_adjacent_pairs_val: {manifest.get('num_adjacent_pairs_val')}",
        f"- num_adjacent_pairs_test: {manifest.get('num_adjacent_pairs_test')}",
        "",
        "## 7. motion distribution summary",
        f"- motion_distribution: {json.dumps(motion_distribution, ensure_ascii=False)}",
        "",
        "## 8. 是否达到 TRAIN360 readiness",
        f"- readiness: {json.dumps(readiness, ensure_ascii=False)}",
        f"- final_classification: {final_classification}",
        "",
        "## 9. 是否还需要更多 sequence",
        f"- download_more_sequences: {recommendation.get('download_more_sequences')}",
        "",
        "## 10. 下一步建议",
        f"- recommendation: {json.dumps(recommendation, ensure_ascii=False)}",
        "- 本轮不训练、不 fine-tune、不使用 360DVO GT 做 calibration。",
        "- S5 locked metrics/policy were not changed.",
    ]
    return lines


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_cfg(Path(args.config))
    manifest = _read_json(Path(args.manifest_summary))
    all_rows = _read_jsonl(Path(args.manifest_all))
    train_rows = _read_jsonl(Path(args.manifest_train))
    val_rows = _read_jsonl(Path(args.manifest_val))
    test_rows = _read_jsonl(Path(args.manifest_test))
    result_dir = Path(args.manifest_summary).parent
    remote_inventory = _read_json(result_dir / "remote_sequence_inventory.json")
    download = _read_json(result_dir / "download_summary.json")
    sequence_audit = _read_json(result_dir / "sequence_completeness_audit.json")
    target = cfg.get("target", {})

    motion_distribution = {
        "all": _split_stats(all_rows),
        "train": _split_stats(train_rows),
        "val": _split_stats(val_rows),
        "test": _split_stats(test_rows),
    }
    motion_distribution["train_val_similarity"] = _similarity(motion_distribution["train"], motion_distribution["val"])
    motion_distribution["train_test_similarity"] = _similarity(motion_distribution["train"], motion_distribution["test"])
    motion_distribution["val_test_similarity"] = _similarity(motion_distribution["val"], motion_distribution["test"])

    readiness = {
        "num_sequences_ok": bool(manifest.get("num_sequences", 0) >= int(cfg.get("download", {}).get("min_total_sequences", 10))),
        "train_pairs_ok": bool(manifest.get("num_pairs_train", 0) >= int(target.get("min_train_pairs", 2000))),
        "val_pairs_ok": bool(manifest.get("num_pairs_val", 0) >= int(target.get("min_val_pairs", 300))),
        "test_pairs_ok": bool(manifest.get("num_pairs_test", 0) >= int(target.get("min_test_pairs", 300))),
        "sequence_split_ok": bool(
            manifest.get("split_by_sequence")
            and manifest.get("forbid_random_pair_split")
            and manifest.get("manifest_ready")
            and len(manifest.get("train_sequences", [])) >= int(cfg.get("split", {}).get("min_train_sequences", 7))
            and len(manifest.get("val_sequences", [])) >= int(cfg.get("split", {}).get("min_val_sequences", 2))
            and len(manifest.get("test_sequences", [])) >= int(cfg.get("split", {}).get("min_test_sequences", 2))
        ),
        "small_motion_risk_low": bool(motion_distribution["all"].get("small_motion_fraction", 1.0) <= 0.05),
        "adjacent_train_ok": bool(manifest.get("num_adjacent_pairs_train", 0) >= int(target.get("min_adjacent_pairs_train", 400))),
        "adjacent_val_ok": bool(manifest.get("num_adjacent_pairs_val", 0) >= int(target.get("min_adjacent_pairs_val", 80))),
        "adjacent_test_ok": bool(manifest.get("num_adjacent_pairs_test", 0) >= int(target.get("min_adjacent_pairs_test", 80))),
        "train360_ready": False,
    }
    readiness["train360_ready"] = all(
        readiness[key]
        for key in [
            "num_sequences_ok",
            "train_pairs_ok",
            "val_pairs_ok",
            "test_pairs_ok",
            "sequence_split_ok",
            "small_motion_risk_low",
            "adjacent_train_ok",
            "adjacent_val_ok",
            "adjacent_test_ok",
        ]
    )

    final_classification = _final_classification(
        download=download,
        sequence_audit=sequence_audit,
        readiness=readiness,
    )
    recommendation = {
        "do_train360_baseline": readiness["train360_ready"],
        "do_base360_baseline": bool(manifest.get("num_sequences", 0) >= 8),
        "download_more_sequences": final_classification != "DSET2B_TRAIN360_READY",
        "keep_s5e15_as_legacy_best_candidate": True,
        "use_360dvo_as_main_dataset": True,
    }
    payload = {
        "experiment": "DSET2B_360DVO_pair_count_completion",
        "download": download,
        "sequence_completeness": sequence_audit,
        "manifest": manifest,
        "motion_distribution": motion_distribution,
        "readiness": readiness,
        "recommendation": recommendation,
        "s5_locked_metrics_policy_unchanged": True,
        "final_classification": final_classification,
    }
    if final_classification not in ALLOWED_CLASSIFICATIONS:
        payload["final_classification"] = "DSET2B_ERROR"

    write_json(Path(args.out_json), payload)
    Path(args.out_report).write_text(
        "\n".join(
            _report_lines(
                remote_inventory=remote_inventory,
                download=download,
                sequence_audit=sequence_audit,
                manifest=manifest,
                motion_distribution=motion_distribution,
                readiness=readiness,
                recommendation=recommendation,
                final_classification=payload["final_classification"],
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-summary", required=True)
    parser.add_argument("--manifest-all", required=True)
    parser.add_argument("--manifest-train", required=True)
    parser.add_argument("--manifest-val", required=True)
    parser.add_argument("--manifest-test", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
