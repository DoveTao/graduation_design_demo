#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_JSON = REPO_ROOT / "checkpoints" / "AB1_final_ablation_baseline_comparison_results.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "final_ablation_baseline_comparison.md"

FINAL_MANIFEST_PATH = REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json"
NEGATIVE_SUMMARY_PATH = REPO_ROOT / "reports" / "final_negative_results_summary.md"
S13_SUMMARY_PATH = REPO_ROOT / "reports" / "final_s13_practical_usability_gap_summary.md"
S1D5_SUMMARY_JSON = REPO_ROOT / "checkpoints" / "S1d5_final_repro" / "s1d5_policy_eval_summary.json"
S2B_SUMMARY_JSON = REPO_ROOT / "checkpoints" / "S2b_final_repro" / "s1d5_policy_eval_summary.json"
S8_REPORT = REPO_ROOT / "checkpoints" / "S8_fine_spherical_reliability_router_report.md"
S9_REPORT = REPO_ROOT / "checkpoints" / "S9_regime_only_reliability_router_report.md"
S10_REPORT = REPO_ROOT / "checkpoints" / "S10_chain_level_path_ratio_preserving_smoother_report.md"
S11_REPORT = REPO_ROOT / "checkpoints" / "S11_tmag_scale_consistency_report.md"
S12_REPORT = REPO_ROOT / "checkpoints" / "S12_regime_balanced_sampling_report.md"
S14_REPORT = REPO_ROOT / "checkpoints" / "S14_local_window_pose_graph_optimization_report.md"
S15_REPORT = REPO_ROOT / "checkpoints" / "S15e_tiny_trajectory_retest_after_harness_fix_report.md"
S16B_REPORT = REPO_ROOT / "checkpoints" / "S16b_frozen_pretrained_backbone_probe_report.md"
S19_REPORT = REPO_ROOT / "checkpoints" / "S19_geometry_aware_pretraining_feasibility_report.md"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: Any, digits: int = 6) -> str:
    try:
        x = float(v)
    except Exception:
        return str(v)
    return f"{x:.{digits}f}"


def _resolve_path(raw: str | Path | None, default: Path) -> Path:
    if raw is None:
        return default
    path = Path(str(raw).strip())
    return path if path.is_absolute() else (REPO_ROOT / path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate final ablation and baseline comparison report from locked existing artifacts.")
    p.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="Output JSON path.")
    p.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD), help="Output Markdown path.")
    p.add_argument("--verify-s5", action="store_true", help="Run the final S5 verifier before generating the report.")
    return p.parse_args()


def _verify_s5() -> Dict[str, Any]:
    proc = subprocess.run(
        ["bash", "scripts/verify_final_candidate.sh"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.rstrip() for line in proc.stdout.splitlines() if line.strip()]
    parsed: Dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if ": " in stripped and not stripped.startswith("[S5-final-verifier]"):
            key, value = stripped.split(": ", 1)
            parsed[key] = value
    return {"status": parsed.get("status", "UNKNOWN"), "stdout": proc.stdout}


def _load_main_rows() -> List[Dict[str, Any]]:
    s1 = _read_json(S1D5_SUMMARY_JSON)
    s2 = _read_json(S2B_SUMMARY_JSON)
    s5 = _read_json(FINAL_MANIFEST_PATH)
    s5m = s5["final_metrics"]
    rows = [
        {
            "method": "S1d5_clean_dt_anchor_policy",
            "role": "scale/path_ratio collapse repaired clean baseline",
            "ATE": float(s1["ATE"]),
            "drift": float(s1["drift"]),
            "path_ratio": float(s1["metric_path_ratio"]),
            "final_candidate": False,
            "notes": "first stable clean mainline after scale repair",
            "source": str(S1D5_SUMMARY_JSON.relative_to(REPO_ROOT)),
        },
        {
            "method": "S2b_clean_fine_rot_policy",
            "role": "major clean fine-rotation improvement",
            "ATE": float(s2["ATE"]),
            "drift": float(s2["drift"]),
            "path_ratio": float(s2["metric_path_ratio"]),
            "final_candidate": False,
            "notes": "train-CV selected fine_rot=0.45 policy",
            "source": str(S2B_SUMMARY_JSON.relative_to(REPO_ROOT)),
        },
        {
            "method": "S5_clean_tmag_calibration_policy",
            "role": "final locked clean candidate",
            "ATE": float(s5m["ATE"]),
            "drift": float(s5m["drift"]),
            "path_ratio": float(s5m["path_ratio"]),
            "final_candidate": True,
            "notes": "clean but marginal gain over S2b",
            "source": str(FINAL_MANIFEST_PATH.relative_to(REPO_ROOT)),
        },
    ]
    s5_ate = rows[-1]["ATE"]
    s5_drift = rows[-1]["drift"]
    s5_path = rows[-1]["path_ratio"]
    for row in rows:
        row["delta_ATE_vs_S5"] = float(row["ATE"] - s5_ate)
        row["delta_drift_vs_S5"] = float(row["drift"] - s5_drift)
        row["delta_path_ratio_vs_S5"] = float(row["path_ratio"] - s5_path)
    return rows


def _build_component_rows(main_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_method = {row["method"]: row for row in main_rows}
    s1 = by_method["S1d5_clean_dt_anchor_policy"]
    s2 = by_method["S2b_clean_fine_rot_policy"]
    s5 = by_method["S5_clean_tmag_calibration_policy"]
    return [
        {
            "transition": "S1d5 -> S2b",
            "component_added": "fine_rot clean policy",
            "delta_ATE": float(s2["ATE"] - s1["ATE"]),
            "delta_drift": float(s2["drift"] - s1["drift"]),
            "delta_path_ratio": float(s2["path_ratio"] - s1["path_ratio"]),
            "interpretation": "adding fine_rot produced the major clean ATE/drift improvement while preserving safe path_ratio",
        },
        {
            "transition": "S2b -> S5",
            "component_added": "pred_tmag regime-aware calibration",
            "delta_ATE": float(s5["ATE"] - s2["ATE"]),
            "delta_drift": float(s5["drift"] - s2["drift"]),
            "delta_path_ratio": float(s5["path_ratio"] - s2["path_ratio"]),
            "interpretation": "tmag calibration produced only a very small clean gain, but passed train-CV, leakage audit, and lockdown and therefore became the final candidate",
        },
    ]


def _build_oracle_rows() -> List[Dict[str, Any]]:
    return [
        {
            "diagnostic": "official_current_S5",
            "ATE": 7.352288,
            "drift": 1.327343,
            "path_ratio": 0.932379,
            "interpretation": "locked clean final candidate under the historical protocol",
            "source": str(FINAL_MANIFEST_PATH.relative_to(REPO_ROOT)),
        },
        {
            "diagnostic": "oracle_R",
            "ATE": 10.590301,
            "drift": 1.516962,
            "path_ratio": 0.934986,
            "interpretation": "rotation alone does not solve the practical gap and can even worsen global metrics",
            "source": "tools/s13_practical_usability_gap_analysis.py",
        },
        {
            "diagnostic": "oracle_tdir",
            "ATE": 7.658319,
            "drift": 1.375470,
            "path_ratio": 0.934985,
            "interpretation": "translation direction alone does not solve the practical gap",
            "source": "tools/s13_practical_usability_gap_analysis.py",
        },
        {
            "diagnostic": "oracle_R_tdir",
            "ATE": 0.911024,
            "drift": 0.304305,
            "path_ratio": 0.934986,
            "interpretation": "jointly fixing R and tdir collapses most of the gap, supporting the R-TDIR-COUPLED-LIMITED diagnosis",
            "source": "tools/s13_practical_usability_gap_analysis.py",
        },
        {
            "diagnostic": "oracle_tmag",
            "ATE": 7.214477,
            "drift": 1.262237,
            "path_ratio": 0.898234,
            "interpretation": "tmag alone can improve some scalar metrics but path_ratio falls below the safe range, so tmag is not the main bottleneck",
            "source": "tools/s13_practical_usability_gap_analysis.py",
        },
    ]


def _build_negative_rows() -> List[Dict[str, Any]]:
    return [
        {
            "experiment": "S8",
            "category": "post-S5 reliability router",
            "final_classification": "TOKEN-NO-ADDED-VALUE",
            "replaced_S5": False,
            "key_reason": "fine/spherical token reliability router did not beat regime-only features or S5",
            "thesis_usage": "negative baseline showing token reliability probing is not enough",
            "source": str(S8_REPORT.relative_to(REPO_ROOT)),
        },
        {
            "experiment": "S9",
            "category": "post-S5 regime router",
            "final_classification": "NO-STABLE-REGIME-ROUTER-GAIN",
            "replaced_S5": False,
            "key_reason": "diagnostic regime signal existed but no clean final candidate passed the gate",
            "thesis_usage": "negative baseline for deployable routing",
            "source": str(S9_REPORT.relative_to(REPO_ROOT)),
        },
        {
            "experiment": "S10",
            "category": "post-S5 chain smoother",
            "final_classification": "NO-STABLE-CHAIN-SMOOTHER-GAIN",
            "replaced_S5": False,
            "key_reason": "chain-level smoothing did not produce a clean train-CV winner",
            "thesis_usage": "negative baseline for lightweight smoothing",
            "source": str(S10_REPORT.relative_to(REPO_ROOT)),
        },
        {
            "experiment": "S11",
            "category": "post-S5 tmag consistency training",
            "final_classification": "NO-STABLE-TMAG-CONSISTENCY-GAIN",
            "replaced_S5": False,
            "key_reason": "slight proxy improvements but no stable path_ratio evidence",
            "thesis_usage": "negative baseline for train-time tmag consistency",
            "source": str(S11_REPORT.relative_to(REPO_ROOT)),
        },
        {
            "experiment": "S12",
            "category": "post-S5 data balancing",
            "final_classification": "NO-STABLE-REGIME-SAMPLING-GAIN",
            "replaced_S5": False,
            "key_reason": "regime-balanced sampling still lacked clean-eligible path_ratio evidence",
            "thesis_usage": "negative baseline for data-centric rebalance",
            "source": str(S12_REPORT.relative_to(REPO_ROOT)),
        },
        {
            "experiment": "S14",
            "category": "post-S5 pose graph",
            "final_classification": "NO-STABLE-POSE-GRAPH-GAIN",
            "replaced_S5": False,
            "key_reason": "lightweight local-window pose graph worsened ATE/drift/path_ratio",
            "thesis_usage": "negative baseline for lightweight graph post-optimization",
            "source": str(S14_REPORT.relative_to(REPO_ROOT)),
        },
        {
            "experiment": "S15",
            "category": "post-S5 trajectory training",
            "final_classification": "TRAINING-STILL-UNSTABLE",
            "replaced_S5": False,
            "key_reason": "trajectory objective route remained unstable under the current harness",
            "thesis_usage": "negative baseline for tiny trajectory-level training",
            "source": str(S15_REPORT.relative_to(REPO_ROOT)),
        },
        {
            "experiment": "S16b",
            "category": "post-S5 backbone comparison",
            "final_classification": "NO-STABLE-BACKBONE-FEATURE-GAIN",
            "replaced_S5": False,
            "key_reason": "frozen ImageNet ResNet50 features were worse than current task-specific features",
            "thesis_usage": "negative baseline against generic pretrained backbone features",
            "source": str(S16B_REPORT.relative_to(REPO_ROOT)),
        },
        {
            "experiment": "S19",
            "category": "post-S5 geometry pretraining",
            "final_classification": "NO-STABLE-GEOMETRY-PRETRAINING-GAIN",
            "replaced_S5": False,
            "key_reason": "shallow geometry pretraining showed partial signal but no stable coupled R/tdir gain",
            "thesis_usage": "negative baseline against probe-head-only geometry pretraining",
            "source": str(S19_REPORT.relative_to(REPO_ROOT)),
        },
    ]


def _markdown_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            v = row.get(col, "")
            if isinstance(v, float):
                vals.append(_fmt(v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_markdown(payload: Dict[str, Any], path: Path) -> None:
    main_rows = payload["main_clean_baseline_comparison"]
    component_rows = payload["component_ablation_interpretation"]
    oracle_rows = payload["oracle_diagnostic_comparison"]
    negative_rows = payload["negative_post_s5_baselines"]
    s5 = next(row for row in main_rows if row["method"] == "S5_clean_tmag_calibration_policy")
    lines = [
        "# Final Ablation and Baseline Comparison",
        "",
        "## Scope",
        "This is an evaluation-only comparison built from existing locked or historical results. It does not run new training, does not search for a new policy, and does not change the final project conclusion.",
        "",
        "## Final candidate reminder",
        f"- final candidate: `S5_clean_tmag_calibration_policy`",
        f"- ATE = `{_fmt(s5['ATE'])}`",
        f"- drift = `{_fmt(s5['drift'])}`",
        f"- path_ratio = `{_fmt(s5['path_ratio'])}`",
        "",
        "## Main clean baseline comparison",
        _markdown_table(
            main_rows,
            [
                "method",
                "role",
                "ATE",
                "drift",
                "path_ratio",
                "delta_ATE_vs_S5",
                "delta_drift_vs_S5",
                "delta_path_ratio_vs_S5",
                "final_candidate",
                "notes",
            ],
        ),
        "",
        "## Component ablation interpretation",
        _markdown_table(
            component_rows,
            ["transition", "component_added", "delta_ATE", "delta_drift", "delta_path_ratio", "interpretation"],
        ),
        "",
        "## Oracle diagnostic comparison",
        _markdown_table(
            oracle_rows,
            ["diagnostic", "ATE", "drift", "path_ratio", "interpretation"],
        ),
        "",
        "## Negative post-S5 baselines",
        _markdown_table(
            negative_rows,
            ["experiment", "category", "final_classification", "replaced_S5", "key_reason", "thesis_usage"],
        ),
        "",
        "## Thesis-ready interpretation",
        "- S1d5 fixed the scale/path_ratio collapse and established the first clean mainline baseline.",
        "- S2b provided the major clean gain through fine rotation policy selection.",
        "- S5 provided only a very small but clean, leakage-audited, and locked tmag calibration gain over S2b.",
        "- Oracle diagnostics show that R and tdir must be improved jointly; neither one alone closes the practical gap.",
        "- Post-S5 lightweight modifications and exploratory baselines from S8 to S19 did not replace S5.",
        "- S5 remains the best clean candidate under the historical protocol, but it is not practical-ready.",
        "",
        "## Caveats",
        "- evaluation protocol remains the small-sequence historical protocol",
        "- representativeness caveat from S17/S18 still applies",
        "- S5 is not practical-ready",
        "- no deployment-readiness claim is made here",
        "- main bottleneck remains `R-TDIR-COUPLED-LIMITED`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_json = _resolve_path(args.output_json, DEFAULT_OUTPUT_JSON)
    output_md = _resolve_path(args.output_md, DEFAULT_OUTPUT_MD)

    verify_payload = None
    if args.verify_s5:
        verify_payload = _verify_s5()

    main_rows = _load_main_rows()
    payload = {
        "name": "AB1_final_ablation_baseline_comparison",
        "scope": {
            "evaluation_only": True,
            "new_candidate_selected": False,
            "s5_remains_final_clean_candidate": True,
            "s5_not_practical_ready": True,
            "main_bottleneck": "R-TDIR-COUPLED-LIMITED",
            "small_sequence_protocol_caveat": True,
            "representativeness_caveat_from_s17_s18": True,
        },
        "verify_s5": verify_payload,
        "main_clean_baseline_comparison": main_rows,
        "component_ablation_interpretation": _build_component_rows(main_rows),
        "oracle_diagnostic_comparison": _build_oracle_rows(),
        "negative_post_s5_baselines": _build_negative_rows(),
        "sources": {
            "final_manifest": str(FINAL_MANIFEST_PATH.relative_to(REPO_ROOT)),
            "negative_summary": str(NEGATIVE_SUMMARY_PATH.relative_to(REPO_ROOT)),
            "s13_summary": str(S13_SUMMARY_PATH.relative_to(REPO_ROOT)),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_markdown(payload, output_md)

    print(f"[AB1] json={output_json}")
    print(f"[AB1] md={output_md}")
    print("[AB1] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
