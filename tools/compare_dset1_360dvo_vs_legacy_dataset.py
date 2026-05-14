#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from s5e2_adjacent_dense_lib import write_json


ALLOWED_CLASSIFICATIONS = [
    "DSET1_360DVO_MAIN_DATASET_READY",
    "DSET1_360DVO_MULTI_SEQUENCE_READY",
    "DSET1_360DVO_SMOKE_ONLY_READY",
    "DSET1_360DVO_DOWNLOAD_BLOCKED",
    "DSET1_360DVO_INSUFFICIENT_SEQUENCES",
    "DSET1_LEGACY_DATASET_QUALITY_RISK_CONFIRMED",
    "DSET1_DATASET_MIGRATION_BLOCKED",
    "DSET1_ERROR",
]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    dset1_motion = _read_json(Path(args.dset1_motion))
    data2 = _read_json(Path(args.data2_checkpoint))
    data3 = _read_json(Path(args.data3_checkpoint))
    summary = dset1_motion.get("summary", {})
    split = summary.get("split", {})
    answers = dset1_motion.get("answers", {})
    use_360 = bool(answers.get("suitable_as_main_train_eval_dataset"))
    downgrade_legacy = bool(data3.get("recommendation", {}).get("downgrade_legacy_dataset_to_diagnostic"))
    do_train360 = bool(answers.get("enter_train360") and use_360)
    do_gen5 = True

    if summary.get("insufficient_sequences"):
        final = "DSET1_360DVO_INSUFFICIENT_SEQUENCES"
    elif use_360 and downgrade_legacy and len(summary.get("selected_sequences", [])) >= 5:
        final = "DSET1_360DVO_MAIN_DATASET_READY"
    elif use_360 and len(summary.get("selected_sequences", [])) >= 3:
        final = "DSET1_360DVO_MULTI_SEQUENCE_READY"
    elif len(summary.get("selected_sequences", [])) >= 1:
        final = "DSET1_360DVO_SMOKE_ONLY_READY"
    elif data3.get("final_classification") == "DATA3_LEGACY_IMAGE_QUALITY_RISK_CONFIRMED":
        final = "DSET1_LEGACY_DATASET_QUALITY_RISK_CONFIRMED"
    else:
        final = "DSET1_DATASET_MIGRATION_BLOCKED"

    payload = {
        "experiment": "DSET1_360DVO_main_dataset_migration",
        "dataset": "360DVO",
        "download": _read_json(Path(args.dset1_motion).parent / "download_summary.json"),
        "manifest": summary,
        "motion_distribution": dset1_motion,
        "legacy_quality_audit": data3,
        "comparison": {
            "data2_train_eval_shift": data2.get("gt_motion_distribution", {}).get("train_vs_eval_shift", {}),
            "legacy_quality_risk": data3.get("final_classification"),
            "360dvo_reduces_small_motion_risk": answers.get("reduces_small_motion_risk"),
        },
        "recommendation": {
            "use_360dvo_as_main_dataset": use_360,
            "downgrade_legacy_dataset_to_diagnostic": downgrade_legacy,
            "do_train360_baseline": do_train360,
            "do_gen5_t57b_external_eval": do_gen5,
            "keep_s5e15_as_legacy_best_candidate": True,
        },
        "s5_locked_metrics_policy_unchanged": True,
        "final_classification": final,
    }
    write_json(Path(args.out_json), payload)
    report = [
        "# DSET1 360DVO 主数据集迁移报告",
        "",
        "## 1. 为什么更换数据集",
        "旧数据集在 DATA2 中已经显示出明显 motion distribution shift，而 DATA3 进一步审计图像质量风险；因此需要评估 360DVO 是否可以升级为新主数据集。",
        "",
        "## 2. 旧数据集问题证据",
        json.dumps(
            {
                "data2_motion_shift": data2.get("gt_motion_distribution", {}).get("train_vs_eval_shift", {}),
                "data3_quality": data3.get("final_classification"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "## 3. 360DVO 下载/挂载状态",
        json.dumps(payload["download"], ensure_ascii=False, indent=2),
        "",
        "## 4. 360DVO 多序列 manifest",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "",
        "## 5. 360DVO motion distribution",
        json.dumps(dset1_motion, ensure_ascii=False, indent=2),
        "",
        "## 6. 360DVO vs legacy 对比",
        json.dumps(payload["comparison"], ensure_ascii=False, indent=2),
        "",
        "## 7. 新主数据集建议",
        json.dumps(payload["recommendation"], ensure_ascii=False, indent=2),
        "",
        "## 8. 旧数据集如何保留为 diagnostic",
        "legacy 数据更适合作为 diagnostic / low-quality stress test，而不是继续承担主训练/主评估职责。",
        "",
        "## 9. 下一步 TRAIN360 计划",
        f"- do_train360_baseline = {do_train360}",
        "",
        "## 10. caveats",
        "- 本轮不训练、不 fine-tune、不生成新 model candidate。",
        "- 360DVO 原始数据保持 local-only，不纳入 Git。",
    ]
    Path(args.out_report).write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset1-motion", required=True)
    parser.add_argument("--data2-checkpoint", required=True)
    parser.add_argument("--data3-checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
