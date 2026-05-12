#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from miniyaml import load_yaml_like
from s5e2_adjacent_dense_lib import write_json


ALLOWED_CLASSIFICATIONS = [
    "GEN2_360DVO_ADAPTER_READY",
    "GEN2_360DVO_DATA_NOT_AVAILABLE",
    "GEN2_360DVO_POSE_CONVENTION_BLOCKED",
    "GEN2_360DVO_SMOKE_EVAL_READY",
    "GEN2_360DVO_EXTERNAL_EVAL_BLOCKED",
    "GEN2_360DVO_NOT_SUITABLE_AFTER_INSPECTION",
    "GEN2_ERROR",
]
FINAL_CHECKPOINT = "checkpoints/GEN2_360DVO_external_adapter_smoke_eval.json"
FINAL_REPORT = "reports/gen2_360dvo_external_adapter_smoke_eval.md"


def _load_config(path: Path) -> Dict[str, Any]:
    return load_yaml_like(path) or {}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config = _load_config(Path(args.config))
    manifest = _read_jsonl(Path(args.manifest))
    out_smoke = Path(args.out_json)
    results_dir = out_smoke.parent
    inspection = _read_json(results_dir / "dataset_inspection.json")
    pair_summary = _read_json(results_dir / "pair_manifest_summary.json")
    motion = _read_json(results_dir / "motion_distribution_audit.json")

    dataset_only_feasibility = bool(pair_summary.get("manifest_ready") and motion.get("num_pairs", 0) > 0)
    eval_blocker = None
    model_eval_attempted = False
    model_eval_available = False

    if not inspection.get("dataset_found"):
        eval_blocker = "DATASET_NOT_FOUND_LOCALLY"
    elif "POSE_CONVENTION_BLOCKED" in pair_summary.get("blockers", []):
        eval_blocker = "POSE_CONVENTION_NOT_VERIFIED"
    elif not pair_summary.get("manifest_ready"):
        eval_blocker = "PAIR_MANIFEST_NOT_READY"
    else:
        model_eval_attempted = True
        eval_blocker = "ADAPTER_NOT_MODEL_READY"

    smoke_payload = {
        "model_eval_attempted": bool(model_eval_attempted),
        "model_eval_available": bool(model_eval_available),
        "eval_blocker": eval_blocker,
        "num_pairs": int(motion.get("num_pairs", len(manifest)) if isinstance(motion.get("num_pairs"), int) else len(manifest)),
        "component_metrics_available": False,
        "rot_mean_deg": None,
        "tdir_mean_deg": None,
        "tmag_median_ratio": None,
        "path_ratio": None,
        "ate_available": False,
        "dataset_only_feasibility": dataset_only_feasibility,
        "no_training": bool(config.get("evaluation", {}).get("no_training", True)),
    }
    write_json(out_smoke, smoke_payload)

    if not inspection.get("dataset_found"):
        final_classification = "GEN2_360DVO_DATA_NOT_AVAILABLE"
        do_gen3 = False
        main_next = "download_or_mount_360dvo_then_rerun_gen2"
    elif "POSE_CONVENTION_BLOCKED" in pair_summary.get("blockers", []):
        final_classification = "GEN2_360DVO_POSE_CONVENTION_BLOCKED"
        do_gen3 = False
        main_next = "verify_pose_convention_and_timestamp_alignment"
    elif dataset_only_feasibility and not model_eval_available:
        final_classification = "GEN2_360DVO_EXTERNAL_EVAL_BLOCKED"
        do_gen3 = True
        main_next = "wire_external_inference_bridge_for_360dvo"
    elif dataset_only_feasibility:
        final_classification = "GEN2_360DVO_SMOKE_EVAL_READY"
        do_gen3 = True
        main_next = "run_gen3_external_eval"
    else:
        final_classification = "GEN2_360DVO_NOT_SUITABLE_AFTER_INSPECTION"
        do_gen3 = False
        main_next = "fallback_to_secondary_external_dataset"

    checkpoint = {
        "experiment": "GEN2_360DVO_external_adapter_smoke_eval",
        "dataset": "360DVO",
        "inspection": inspection,
        "pair_manifest": pair_summary,
        "motion_distribution": motion,
        "external_smoke_eval": smoke_payload,
        "recommendation": {
            "do_gen3_external_eval": do_gen3,
            "do_external_training": False,
            "keep_s5e15_as_best_candidate": True,
            "main_next_step": main_next,
        },
        "s5_locked_metrics_policy_unchanged": True,
        "final_classification": final_classification,
    }
    write_json(Path(FINAL_CHECKPOINT), checkpoint)

    report = [
        "# GEN2 360DVO external adapter smoke eval",
        "",
        "## 1. 为什么做 GEN2",
        "GEN1 已将 360DVO 选为 primary external panoramic dataset，GEN2 的目标是在不训练模型的前提下验证最小 adapter、pair manifest 和外部 smoke eval feasibility。",
        "",
        "## 2. 360DVO dataset inspection",
        json.dumps(inspection, ensure_ascii=False, indent=2),
        "",
        "## 3. adapter schema",
        json.dumps(
            {
                "dataset": "360DVO",
                "target_pose_convention": "T_w_c",
                "relative_pose_terms": ["R_BA", "t_BA_B", "tdir_B"],
                "pair_types": config.get("pair_manifest", {}).get("pair_types", []),
                "k_values": config.get("pair_manifest", {}).get("k_values", []),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "## 4. pair manifest summary",
        json.dumps(pair_summary, ensure_ascii=False, indent=2),
        "",
        "## 5. motion distribution vs DATA2 当前数据",
        json.dumps(motion, ensure_ascii=False, indent=2),
        "",
        "## 6. external smoke eval 是否可行",
        json.dumps(smoke_payload, ensure_ascii=False, indent=2),
        "",
        "## 7. blockers",
        json.dumps(
            {
                "inspection_blockers": inspection.get("blockers", []),
                "manifest_blockers": pair_summary.get("blockers", []),
                "eval_blocker": eval_blocker,
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "## 8. 是否进入 GEN3",
        json.dumps(checkpoint["recommendation"], ensure_ascii=False, indent=2),
        "",
        "## 9. caveats",
        "- 本轮不训练、不 fine-tune、不生成新 candidate。",
        "- 不提交下载数据、原始图像、raw trajectory、大日志或模型权重。",
        "- 如果 360DVO 本地数据缺失，本报告只给出 adapter/blocker 审计，不假造 prediction。",
    ]
    Path(FINAL_REPORT).write_text("\n".join(report) + "\n", encoding="utf-8")
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
