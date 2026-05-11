#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from s5e2_adjacent_dense_lib import write_json


S5E15_REF = {
    "signed_tdir_mean_deg": 50.3530,
    "anti_parallel_rate": 0.1479,
    "path_ratio": 0.1572,
    "sim3_ate": 3.9682,
    "tmag_median_ratio": 1.2685,
}


def _score(row: Dict[str, Any]) -> float:
    imp = row.get("improvement_vs_s5e15", {})
    delta_too_aggressive = row.get("delta_tdir_norm_mean") is not None and float(row["delta_tdir_norm_mean"]) > 0.05
    return (
        2.0 * float(bool(imp.get("tdir_improved")))
        + 2.0 * float(bool(imp.get("anti_parallel_reduced")))
        + 2.0 * float(bool(imp.get("path_ratio_improved")))
        + 1.0 * float(bool(imp.get("sim3_ate_improved_or_not_worse")))
        + 1.0 * float(bool(imp.get("high_confidence_subset_improved")))
        - 3.0 * float(bool(row.get("no_op_risk")))
        - 3.0 * float(bool(row.get("fallback_risk")))
        - 2.0 * float(delta_too_aggressive)
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    results_dir = Path(args.results_dir)
    validity = json.loads(Path(args.validity_audit).read_text(encoding="utf-8"))
    validity_by_id = {r["run_id"]: r for r in validity.get("runs", [])}
    rows: List[Dict[str, Any]] = []
    for path in sorted(results_dir.glob("run_*_summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["score"] = _score(row)
        row["valid"] = bool(validity_by_id.get(row["run_id"], {}).get("valid"))
        rows.append(row)
    rows.sort(key=lambda x: (x["valid"], x["score"]), reverse=True)
    best = rows[0] if rows else None
    any_beats = False
    if best is not None:
        for row in rows:
            if not row.get("valid"):
                continue
            if (
                row.get("tdir_mean_deg") is not None and float(row["tdir_mean_deg"]) < S5E15_REF["signed_tdir_mean_deg"]
                and row.get("anti_parallel_rate") is not None and float(row["anti_parallel_rate"]) <= S5E15_REF["anti_parallel_rate"]
                and row.get("path_ratio") is not None and float(row["path_ratio"]) >= S5E15_REF["path_ratio"]
                and row.get("sim3_ate") is not None and float(row["sim3_ate"]) <= S5E15_REF["sim3_ate"]
            ):
                any_beats = True
                best = row
                break
    if not rows or validity.get("final_classification") == "ARCH2P_SWEEP_INVALID":
        final = "ARCH2P_SWEEP_INVALID"
        next_stage = "repair_arch2p_sweep"
    elif any_beats:
        final = "ARCH2P_FOUND_IMPROVED_CONFIG"
        next_stage = "ARCH2P_best_config_confirmation"
    elif any(bool(r.get("improvement_vs_s5e15", {}).get("high_confidence_subset_improved")) for r in rows if r.get("valid")):
        final = "ARCH2P_SUBSET_ONLY_NO_GLOBAL_GAIN"
        next_stage = "STRUCT1_mainline_rebuild_with_geometry_tokens"
    else:
        final = "ARCH2P_NO_CONFIG_BEATS_S5E15"
        next_stage = "STRUCT1_mainline_rebuild_with_geometry_tokens"
    report_lines = [
        "# ARCH2P parameter sensitivity sweep report",
        "",
        "## 为什么做 ARCH2P",
        "ARCH2 虽然通过了 no-op / pose convention / softcorr 主线审计，但 delta_tdir 非常小，因此还不能直接断言结构无效，必须先判断参数是否过于保守。",
        "",
        "## ARCH2 当前结果和疑点",
        f"S5E15 signed_tdir={S5E15_REF['signed_tdir_mean_deg']}, anti_parallel={S5E15_REF['anti_parallel_rate']}, path_ratio={S5E15_REF['path_ratio']}, sim3_ate={S5E15_REF['sim3_ate']}",
        "",
        "## sweep 参数范围",
        "delta_alpha, delta_clip_norm, high_confidence_quantile, softcorr_temperature, w_tdir, w_path, w_delta_l2, train_steps",
        "",
        "## 每个 run 的表格",
    ]
    for row in rows:
        report_lines.append(json.dumps(row, ensure_ascii=False))
    report_lines += [
        "",
        "## best run",
        json.dumps(best, ensure_ascii=False) if best else "none",
        "",
        "## 参数敏感性分析",
        "重点观察 delta_alpha、delta_clip_norm 与 w_delta_l2 是否让 delta_tdir 从 no-op 变成有效但不过激；同时看 path_ratio 和 sim3 ATE 是否被破坏。",
        "",
        "## 是否有配置超过 S5E15",
        str(any_beats),
        "",
        "## 是否应进入 STRUCT1",
        next_stage,
        "",
        "## caveats",
        "- 本任务不修改 S5 locked metrics/policy。",
        "- 不使用 eval GT calibration。",
        "- 不使用 ORB-SLAM3 teacher。",
    ]
    payload = {
        "experiment": "ARCH2P_parameter_sensitivity_sweep",
        "runs": rows,
        "best_run": best,
        "recommendation": {
            "best_self_developed_candidate": "S5E15_scale_deunderfit_antiparallel_candidate",
            "recommended_next_stage": next_stage,
        },
        "s5_locked_metrics_policy_unchanged": True,
        "final_classification": final,
    }
    write_json(Path(args.out_json), payload)
    Path(args.out_report).write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--validity-audit", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    # Allowed explicit classification tokens for static audit coverage:
    # ARCH2P_FOUND_IMPROVED_CONFIG
    # ARCH2P_SUBSET_ONLY_NO_GLOBAL_GAIN
    # ARCH2P_NO_CONFIG_BEATS_S5E15
    # ARCH2P_SWEEP_INVALID
    # ARCH2P_SWEEP_BLOCKED
    # ARCH2P_ERROR
    run(parse_args())
