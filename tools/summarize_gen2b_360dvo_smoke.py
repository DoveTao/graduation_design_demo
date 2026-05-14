#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from s5e2_adjacent_dense_lib import write_json


ALLOWED_CLASSIFICATIONS = [
    "GEN2B_360DVO_ONE_SEQUENCE_READY",
    "GEN2B_360DVO_MOTION_AUDIT_READY",
    "GEN2B_360DVO_EXTERNAL_EVAL_READY",
    "GEN2B_360DVO_DOWNLOAD_BLOCKED",
    "GEN2B_360DVO_POSE_CONVENTION_BLOCKED",
    "GEN2B_360DVO_ADAPTER_NOT_READY",
    "GEN2B_360DVO_NOT_SUITABLE",
    "GEN2B_ERROR",
]
CHECKPOINT_PATH = "checkpoints/GEN2B_360DVO_one_sequence_smoke_audit.json"
REPORT_PATH = "reports/gen2b_360dvo_one_sequence_smoke_audit.md"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    results_dir = Path(args.results_dir)
    download = _read_json(results_dir / "gen2b_download_summary.json")
    inspection = _read_json(results_dir / "dataset_inspection.json")
    pair_manifest = _read_json(results_dir / "pair_manifest_summary.json")
    motion_distribution = _read_json(results_dir / "motion_distribution_audit.json")
    external_smoke_eval = _read_json(results_dir / "external_smoke_eval.json")

    if download.get("access_blocked") or download.get("blockers"):
        final = "GEN2B_360DVO_DOWNLOAD_BLOCKED"
        do_gen3 = False
        main_next = "manual_download_or_mount_360dvo"
    elif "POSE_FORMAT_UNKNOWN" in inspection.get("blockers", []) or "POSE_CONVENTION_BLOCKED" in pair_manifest.get("blockers", []):
        final = "GEN2B_360DVO_POSE_CONVENTION_BLOCKED"
        do_gen3 = False
        main_next = "fix_360dvo_pose_convention"
    elif external_smoke_eval.get("model_eval_available"):
        final = "GEN2B_360DVO_EXTERNAL_EVAL_READY"
        do_gen3 = True
        main_next = "run_gen3_external_eval"
    elif inspection.get("inspection_ready") and pair_manifest.get("num_pairs", 0) > 0 and motion_distribution.get("num_pairs", 0) > 0:
        final = "GEN2B_360DVO_MOTION_AUDIT_READY"
        do_gen3 = True
        main_next = "prepare_gen3_external_eval_without_training"
    elif inspection.get("inspection_ready") and pair_manifest.get("num_pairs", 0) > 0:
        final = "GEN2B_360DVO_ONE_SEQUENCE_READY"
        do_gen3 = False
        main_next = "finish_motion_audit_then_decide_gen3"
    elif download.get("download_complete"):
        final = "GEN2B_360DVO_ADAPTER_NOT_READY"
        do_gen3 = False
        main_next = "fix_adapter_or_manifest_readiness"
    else:
        final = "GEN2B_360DVO_NOT_SUITABLE"
        do_gen3 = False
        main_next = "reassess_external_dataset_choice"

    payload = {
        "experiment": "GEN2B_360DVO_one_sequence_smoke_audit",
        "dataset": "360DVO",
        "download": download,
        "inspection": inspection,
        "pair_manifest": pair_manifest,
        "motion_distribution": motion_distribution,
        "external_smoke_eval": external_smoke_eval,
        "recommendation": {
            "do_gen3_external_eval": do_gen3,
            "keep_s5e15_as_best_candidate": True,
            "main_next_step": main_next,
        },
        "s5_locked_metrics_policy_unchanged": True,
        "final_classification": final,
    }
    write_json(Path(args.out_json), payload)

    report = [
        "# GEN2B 360DVO one-sequence smoke audit",
        "",
        "## 1. 为什么做 GEN2B",
        "GEN2 已完成 adapter 链路，但缺少本地 360DVO 数据。GEN2B 的目标是在不训练、不 fine-tune 的前提下，真实下载或挂载一个 360DVO 小序列并复跑 smoke audit。",
        "",
        "## 2. 下载/挂载状态",
        json.dumps(download, ensure_ascii=False, indent=2),
        "",
        "## 3. 选中的 360DVO sequence",
        json.dumps({"selected_sequence": download.get("selected_sequence")}, ensure_ascii=False, indent=2),
        "",
        "## 4. dataset inspection 结果",
        json.dumps(inspection, ensure_ascii=False, indent=2),
        "",
        "## 5. pair manifest summary",
        json.dumps(pair_manifest, ensure_ascii=False, indent=2),
        "",
        "## 6. motion distribution vs DATA2 seq03",
        json.dumps(motion_distribution, ensure_ascii=False, indent=2),
        "",
        "## 7. external smoke eval 是否成功",
        json.dumps(external_smoke_eval, ensure_ascii=False, indent=2),
        "",
        "## 8. blocker",
        json.dumps(
            {
                "download_blockers": download.get("blockers", []),
                "inspection_blockers": inspection.get("blockers", []),
                "manifest_blockers": pair_manifest.get("blockers", []),
                "smoke_eval_blocker": external_smoke_eval.get("eval_blocker"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "## 9. 是否进入 GEN3",
        json.dumps(payload["recommendation"], ensure_ascii=False, indent=2),
        "",
        "## 10. caveats",
        "- 本轮 no training，不 fine-tune，不生成新 candidate。",
        "- 360DVO 数据为 local-only artifact，路径为 data/360DVO，不纳入 Git。",
        "- 不提交 raw images / raw poses / downloaded archives / raw trajectory / model weights / large logs。",
        f"- checkpoint: {CHECKPOINT_PATH}",
        f"- report: {REPORT_PATH}",
    ]
    Path(args.out_report).write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
