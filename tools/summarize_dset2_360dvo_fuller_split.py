#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from s5e2_adjacent_dense_lib import write_json


ALLOWED = [
    "DSET2_360DVO_FULLER_SPLIT_READY",
    "DSET2_360DVO_TRAIN360_READY",
    "DSET2_360DVO_INSUFFICIENT_SEQUENCES",
    "DSET2_360DVO_DOWNLOAD_BLOCKED",
    "DSET2_360DVO_SPLIT_IMBALANCED",
    "DSET2_360DVO_MANIFEST_BLOCKED",
    "DSET2_ERROR",
]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    download = _read_json(Path(args.download_summary))
    manifest = _read_json(Path(args.manifest_summary))
    motion = _read_json(Path(args.motion_audit))
    dset1 = _read_json(Path(args.dset1_checkpoint))
    data3 = _read_json(Path(args.data3_checkpoint))
    gen5 = _read_json(Path(args.gen5_checkpoint))

    split = manifest
    split_stats = motion.get("split_stats", {})
    answers = motion.get("answers", {})
    readiness = {
        "num_sequences_ok": bool(split.get("num_sequences", 0) >= 7),
        "train_pairs_ok": bool(split.get("num_pairs_train", 0) >= 2000),
        "val_pairs_ok": bool(split.get("num_pairs_val", 0) >= 300),
        "test_pairs_ok": bool(split.get("num_pairs_test", 0) >= 300),
        "sequence_split_ok": bool(split.get("split_by_sequence") and split.get("forbid_random_pair_split") and split.get("manifest_ready")),
        "small_motion_risk_low": bool(answers.get("continues_to_avoid_small_motion_risk")),
        "train360_ready": bool(answers.get("meets_train360_readiness_threshold")),
    }

    if download.get("access_blocked") or "DOWNLOAD_FAILED" in " ".join(download.get("blockers", [])):
        final = "DSET2_360DVO_DOWNLOAD_BLOCKED"
    elif not manifest.get("manifest_ready"):
        final = "DSET2_360DVO_MANIFEST_BLOCKED"
    elif split.get("num_sequences", 0) < 7:
        final = "DSET2_360DVO_INSUFFICIENT_SEQUENCES"
    elif readiness["train360_ready"]:
        final = "DSET2_360DVO_TRAIN360_READY"
    elif readiness["sequence_split_ok"] and readiness["small_motion_risk_low"]:
        final = "DSET2_360DVO_FULLER_SPLIT_READY"
    else:
        final = "DSET2_360DVO_SPLIT_IMBALANCED"

    payload = {
        "experiment": "DSET2_360DVO_fuller_split_expansion",
        "download": download,
        "manifest": manifest,
        "motion_distribution": motion,
        "readiness": readiness,
        "recommendation": {
            "do_train360_baseline": readiness["train360_ready"],
            "download_more_sequences": not readiness["train360_ready"],
            "keep_s5e15_as_legacy_best_candidate": True,
            "use_360dvo_as_main_dataset": True,
        },
        "context": {
            "dset1_final_classification": dset1.get("final_classification"),
            "data3_final_classification": data3.get("final_classification"),
            "gen5_final_classification": gen5.get("final_classification"),
        },
        "s5_locked_metrics_policy_unchanged": True,
        "final_classification": final,
    }
    write_json(Path(args.out_json), payload)

    report = [
        "# DSET2 360DVO fuller split expansion report",
        "",
        "## 1. 为什么做 DSET2",
        "DSET1 已经证明 360DVO 可以替代 legacy 数据成为主数据集候选，而 GEN5 说明 true image-pair 模型在外部数据上 translation generalization 仍弱，因此需要扩展数据子集，为 TRAIN360 新基线训练准备更完整的 sequence split。",
        "",
        "## 2. 当前 DSET1 / GEN5 背景",
        json.dumps(payload["context"], ensure_ascii=False, indent=2),
        "",
        "## 3. 下载/挂载的新序列",
        json.dumps(download, ensure_ascii=False, indent=2),
        "",
        "## 4. train/val/test sequence split",
        json.dumps({k: manifest.get(k) for k in ['train_sequences', 'val_sequences', 'test_sequences', 'sequences_all']}, ensure_ascii=False, indent=2),
        "",
        "## 5. pair manifest 统计",
        json.dumps(manifest, ensure_ascii=False, indent=2),
        "",
        "## 6. motion distribution",
        json.dumps(motion, ensure_ascii=False, indent=2),
        "",
        "## 7. 与 DSET1 和 legacy DATA2 的对比",
        json.dumps(motion.get("comparison", {}), ensure_ascii=False, indent=2),
        "",
        "## 8. 是否 ready for TRAIN360",
        json.dumps(readiness, ensure_ascii=False, indent=2),
        "",
        "## 9. caveats",
        "- 本轮不训练、不 fine-tune、不使用 360DVO GT calibration。",
        "- manifest jsonl 只包含 metadata/相对路径，不包含原始图像数据。",
        "- data/360DVO 保持 local-only，不纳入 Git。",
    ]
    Path(args.out_report).write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-summary", required=True)
    parser.add_argument("--manifest-summary", required=True)
    parser.add_argument("--motion-audit", required=True)
    parser.add_argument("--dset1-checkpoint", required=True)
    parser.add_argument("--data3-checkpoint", required=True)
    parser.add_argument("--gen5-checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
