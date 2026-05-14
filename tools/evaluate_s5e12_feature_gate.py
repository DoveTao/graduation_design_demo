#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from s5e2_adjacent_dense_lib import read_json, validation_from_logs, write_json


S5E11_BASELINE = {
    "observable_edge_fraction": 0.026490066225165563,
    "signed_direction_supervision_reliable_fraction": 0.026490066225165563,
    "small_motion_observable_fraction": 0.0,
}


def report_md(path: Path, ckpt: Dict[str, Any]) -> None:
    lines = [
        "# S5E12 real correspondence feature gate report",
        "",
        "## 最终分类",
        str(ckpt.get("final_classification")),
        "",
        "## dependency check 结果",
        str(ckpt.get("dependency_check")),
        "",
        "## 与 S5E11 的关系",
        "S5E12 不训练新模型，而是验证真实 correspondence / optical flow / parallax features 是否能把 signed direction supervision reliable fraction 从 S5E11 的极低水平拉高。",
        "",
        "## real correspondence feature availability",
        str(ckpt.get("correspondence_feature_availability")),
        "",
        "## optical flow availability",
        str({
            "optical_flow_available": ckpt.get("dependency_check", {}).get("optical_flow_available"),
            "flow_summary_present": ckpt.get("correspondence_quality_audit", {}).get("flow_magnitude_vs_gt_tmag_correlation"),
        }),
        "",
        "## strict essential geometry availability",
        str(ckpt.get("essential_geometry_audit")),
        "",
        "## intrinsics blocker",
        str(ckpt.get("essential_geometry_audit", {}).get("essential_geometry_blocker")),
        "",
        "## feature extraction coverage",
        str(ckpt.get("correspondence_quality_audit")),
        "",
        "## observability gate",
        str(ckpt.get("observability_gate")),
        "",
        "## small-motion observability comparison vs S5E11",
        str({
            "s5e11_small_motion_observable_fraction": S5E11_BASELINE["small_motion_observable_fraction"],
            "s5e12_small_motion_observable_fraction": ckpt.get("observability_gate", {}).get("small_motion_observable_fraction"),
        }),
        "",
        "## reliable signed direction supervision fraction",
        str({
            "s5e11": S5E11_BASELINE["signed_direction_supervision_reliable_fraction"],
            "s5e12": ckpt.get("observability_gate", {}).get("signed_direction_supervision_reliable_fraction"),
        }),
        "",
        "## correlation audit",
        str(ckpt.get("correspondence_quality_audit")),
        "",
        "## exported artifacts",
        str(ckpt.get("artifacts")),
        "",
        "## blockers",
        str(ckpt.get("blockers")),
        "",
        "## validation results",
        str(ckpt.get("validation")),
        "",
        "## caveats",
        "- S5E12 是 feature gate，不替换 S5 locked。",
        "- camera intrinsics 缺失时，strict essential geometry 必须诚实标记 unavailable。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    dep = read_json(Path(args.dependency_json))
    quality = read_json(Path(args.quality_json))
    essential = read_json(Path(args.essential_json))
    obs = read_json(Path(args.observability_json))
    corr = read_json(Path(args.features_json))

    ckpt = {
        "experiment": "S5E12_real_correspondence_feature_gate",
        "status": {
            "experimental_candidate": True,
            "official_s5_unchanged": True,
            "not_official_replacement": True,
        },
        "dependency_check": dep,
        "correspondence_feature_availability": {
            "real_correspondence_features_available": True,
            "feature_extraction_coverage": quality.get("correspondence_success_fraction"),
            "proxy_fallback_explicitly_marked": True,
        },
        "correspondence_quality_audit": quality,
        "essential_geometry_audit": essential,
        "observability_gate": {
            "observable_edge_count": obs.get("observable_edge_count"),
            "observable_edge_fraction": obs.get("observable_edge_fraction"),
            "unobservable_edge_count": obs.get("unobservable_edge_count"),
            "unobservable_edge_fraction": obs.get("unobservable_edge_fraction"),
            "small_motion_observable_count": obs.get("small_motion_observable_count"),
            "small_motion_unobservable_count": obs.get("small_motion_unobservable_count"),
            "small_motion_observable_fraction": obs.get("small_motion_observable_fraction"),
            "signed_direction_supervision_reliable_fraction": obs.get("signed_direction_supervision_reliable_fraction"),
            "observable_edge_fraction_delta_vs_s5e11": obs.get("observable_edge_fraction_delta_vs_s5e11"),
            "reliable_supervision_delta_vs_s5e11": obs.get("reliable_supervision_delta_vs_s5e11"),
        },
        "artifacts": {
            "dependency_check_json": args.dependency_json,
            "correspondence_features_json": args.features_json,
            "correspondence_quality_json": args.quality_json,
            "essential_geometry_json": args.essential_json,
            "observability_gate_json": args.observability_json,
            "correspondence_features_csv": args.features_csv,
        },
    }

    blockers = []
    if not dep.get("cv2_available"):
        blockers.append("cv2 不可用，真实 correspondence feature gate 无法启动。")
    if not dep.get("can_extract_real_correspondence_features"):
        blockers.append("关键 OpenCV API 不完整，真实 correspondence 提取仍被依赖阻塞。")
    if quality.get("correspondence_success_fraction", 0.0) < 0.35:
        blockers.append("真实匹配覆盖率偏低，大多数边的 correspondence 仍然过稀。")
    if obs.get("signed_direction_supervision_reliable_fraction", 0.0) <= S5E11_BASELINE["signed_direction_supervision_reliable_fraction"] + 0.05:
        blockers.append("reliable signed direction supervision fraction 没有显著高于 S5E11。")
    if essential.get("strict_essential_geometry_available") is False:
        blockers.append("strict essential geometry 仍受 pinhole intrinsics 缺失限制。")
    ckpt["blockers"] = blockers

    gate_passed = (
        quality.get("correspondence_success_fraction", 0.0) >= 0.35
        and obs.get("observable_edge_fraction", 0.0) > S5E11_BASELINE["observable_edge_fraction"] + 0.10
        and obs.get("signed_direction_supervision_reliable_fraction", 0.0) > S5E11_BASELINE["signed_direction_supervision_reliable_fraction"] + 0.10
    )

    if quality.get("correspondence_success_fraction", 0.0) <= 0.0:
        final = "S5E12_REGRESSION"
    elif not dep.get("cv2_available") or not dep.get("can_extract_real_correspondence_features"):
        final = "S5E12_DEPENDENCY_BLOCKED"
    elif corr.get("real_correspondence_features_used") is not True:
        final = "S5E12_PROXY_ONLY_REPEATED"
    elif gate_passed:
        final = "S5E12_FEATURE_GATE_PASSED"
    elif essential.get("strict_essential_geometry_available") is True:
        final = "S5E12_ESSENTIAL_GEOMETRY_AVAILABLE"
    elif quality.get("correspondence_success_fraction", 0.0) < 0.35:
        final = "S5E12_CORRESPONDENCE_TOO_SPARSE"
    elif obs.get("signed_direction_supervision_reliable_fraction", 0.0) <= S5E11_BASELINE["signed_direction_supervision_reliable_fraction"] + 0.05:
        final = "S5E12_OBSERVABILITY_STILL_LIMITED"
    elif dep.get("camera_intrinsics_available") is False:
        final = "S5E12_INTRINSICS_MISSING"
    else:
        final = "S5E12_DEPENDENCY_READY"
    ckpt["final_classification"] = final
    ckpt["validation"] = validation_from_logs()
    write_json(Path(args.out_json), ckpt)
    write_json(Path(args.training_status_json), {
        "attempted": True,
        "classification": final,
        "notes": [
            "S5E12 不训练新模型，重点是验证真实 correspondence / flow / parallax feature gate 是否成立。",
            "strict essential geometry 因 camera intrinsics 缺失而保持 limited。",
        ],
    })
    report_md(Path(args.out_report), ckpt)
    return ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dependency-json", required=True)
    p.add_argument("--features-json", required=True)
    p.add_argument("--features-csv", required=True)
    p.add_argument("--quality-json", required=True)
    p.add_argument("--essential-json", required=True)
    p.add_argument("--observability-json", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--training-status-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
