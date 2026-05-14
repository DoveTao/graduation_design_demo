#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_JSON = REPO_ROOT / "checkpoints" / "AB2_thesis_three_axis_ablation_results.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "thesis_three_axis_ablation.md"
AB1_JSON = REPO_ROOT / "checkpoints" / "AB1_final_ablation_baseline_comparison_results.json"
S13_REPORT = REPO_ROOT / "checkpoints" / "S13_practical_usability_gap_analysis_report.md"
S14_REPORT = REPO_ROOT / "checkpoints" / "S14_local_window_pose_graph_optimization_report.md"
FINAL_MANIFEST = REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json"

SCAN_TERMS = [
    "direct",
    "cartesian",
    "vector",
    "non_spherical",
    "nonspherical",
    "spherical",
    "coarse",
    "fine",
    "tdir",
    "tvec",
    "tmag",
    "anchor",
    "calibration",
    "pose_graph",
    "S1d5",
    "S2b",
    "S5",
    "S14",
]


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
    p = argparse.ArgumentParser(description="Generate thesis three-axis ablation report from existing locked artifacts.")
    p.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="Output JSON path.")
    p.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD), help="Output Markdown path.")
    p.add_argument("--verify-s5", action="store_true", help="Run final S5 verifier before generating the report.")
    return p.parse_args()


def _verify_s5() -> Dict[str, Any]:
    proc = subprocess.run(
        ["bash", "scripts/verify_final_candidate.sh"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    status = "UNKNOWN"
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("status:"):
            status = stripped.split(":", 1)[1].strip()
    return {"status": status, "stdout": proc.stdout}


def _repository_scan() -> Dict[str, Any]:
    pattern = "|".join(SCAN_TERMS)
    proc = subprocess.run(
        [
            "rg",
            "-n",
            pattern,
            "checkpoints",
            "reports",
            "-S",
            "--glob",
            "!checkpoints/AB2_thesis_three_axis_ablation_results.json",
            "--glob",
            "!reports/thesis_three_axis_ablation.md",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    hits = [line for line in proc.stdout.splitlines() if line.strip()]
    strict_markers = [
        "strict non-spherical baseline",
        "non_spherical baseline",
        "nonspherical baseline",
        "direct vector baseline",
        "cartesian baseline",
    ]
    strict_candidates = [line for line in hits if any(marker in line.lower() for marker in strict_markers)]
    manual_note = "A strict non-spherical retrained baseline is not available in the locked historical protocol."
    return {
        "scan_terms": SCAN_TERMS,
        "num_hits": len(hits),
        "strict_non_spherical_baseline_found": bool(strict_candidates),
        "strict_non_spherical_candidates": strict_candidates[:20],
        "manual_note": manual_note,
        "supporting_hits_sample": hits[:40],
    }


def _load_ab1() -> Dict[str, Any]:
    return _read_json(AB1_JSON)


def _extract_s13_direction_stats() -> Dict[str, float]:
    return {
        "rot_mean_deg": 20.715353,
        "rot_median_deg": 20.433789,
        "rot_p90_deg": 21.371863,
        "rot_max_deg": 21.717051,
        "tdir_mean_deg": 72.349483,
        "tdir_median_deg": 102.517940,
        "tdir_p90_deg": 110.918120,
        "tdir_max_deg": 141.610521,
        "tdir_mean_cosine_similarity": 0.226300,
        "source": str(S13_REPORT.relative_to(REPO_ROOT)),
    }


def _spherical_axis(ab1: Dict[str, Any], scan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "strict_non_spherical_baseline_available": bool(scan["strict_non_spherical_baseline_found"]),
        "strict_non_spherical_baseline_note": scan["manual_note"],
        "comparison_rows": [],
        "direction_diagnostics": _extract_s13_direction_stats(),
        "interpretation": (
            "The spherical/coarse-fine pipeline provides a structured direction representation, "
            "but current results still show that the translation-direction branch is a major limitation. "
            "This report does not claim that spherical formulation alone solved tdir prediction."
        ),
    }


def _coarse_to_fine_axis(ab1: Dict[str, Any]) -> Dict[str, Any]:
    main_rows = ab1["main_clean_baseline_comparison"]
    trans_rows = ab1["component_ablation_interpretation"]
    return {
        "rows": main_rows,
        "transitions": trans_rows,
        "interpretation": (
            "Coarse-to-fine refinement, especially the fine rotation policy from S1d5 to S2b, "
            "accounts for the largest clean improvement. The final S5 tmag calibration contributes only a marginal but reproducible gain."
        ),
    }


def _geometric_constraints_axis() -> Dict[str, Any]:
    return {
        "rows": [
            {
                "method": "S1d5_clean_dt_anchor_policy",
                "constraint": "dt-anchor / scale-path-ratio correction",
                "ATE": 7.632463,
                "drift": 1.396358,
                "path_ratio": 0.934982,
                "interpretation": "repaired the historical path_ratio collapse and stabilized the clean mainline",
            },
            {
                "method": "S5_clean_tmag_calibration_policy",
                "constraint": "tmag regime-aware calibration",
                "ATE": 7.352288,
                "drift": 1.327343,
                "path_ratio": 0.932379,
                "interpretation": "final clean candidate with marginal but locked gain",
            },
            {
                "method": "S14_local_window_pose_graph_selected",
                "constraint": "local-window pose graph optimization",
                "ATE": 2.785907,
                "drift": 1.495600,
                "path_ratio": 7.277363,
                "interpretation": "naive stronger geometry constraints worsened the diagnostic instead of helping",
            },
        ],
        "s14_baseline_diagnostic": {
            "ATE": 2.635339,
            "drift": 1.326834,
            "path_ratio": 4.944008,
            "source": str(S14_REPORT.relative_to(REPO_ROOT)),
        },
        "s14_selected_diagnostic": {
            "ATE": 2.785907,
            "drift": 1.495600,
            "path_ratio": 7.277363,
            "final_classification": "NO-STABLE-POSE-GRAPH-GAIN",
            "source": str(S14_REPORT.relative_to(REPO_ROOT)),
        },
        "interpretation": (
            "Lightweight scale/tmag constraints help stabilize path behavior, but naive local-window pose graph optimization did not improve results. "
            "Geometry-constraint claims in the thesis should therefore remain cautious."
        ),
    }


def _oracle_axis(ab1: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rows": ab1["oracle_diagnostic_comparison"],
        "interpretation": [
            "Fixing R alone does not solve the problem and can worsen global metrics.",
            "Fixing tdir alone does not solve the problem either.",
            "Fixing R and tdir jointly produces a dramatic gain, supporting the R-TDIR-COUPLED-LIMITED conclusion.",
            "Fixing tmag alone gives some scalar gain but path_ratio leaves the safe range, so tmag is not the main bottleneck.",
        ],
    }


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
    locked = payload["locked_metrics"]
    sph = payload["spherical_axis"]
    ctf = payload["coarse_to_fine_axis"]
    geo = payload["geometric_constraints_axis"]
    oracle = payload["oracle_diagnostics"]
    lines = [
        "# Thesis Three-Axis Ablation and Baseline Comparison",
        "",
        "## Scope",
        "This report reorganizes existing locked or historical results around the three thesis innovation axes. It is reporting-only, introduces no new experiment line, selects no new candidate, and leaves S5 as the final clean candidate.",
        "",
        "## Final Candidate",
        "- `S5_clean_tmag_calibration_policy`",
        f"- ATE = `{_fmt(locked['ATE'])}`",
        f"- drift = `{_fmt(locked['drift'])}`",
        f"- path_ratio = `{_fmt(locked['path_ratio'])}`",
        "",
        "## Axis 1: Spherical Formulation",
    ]
    if sph["strict_non_spherical_baseline_available"]:
        lines.extend(
            [
                _markdown_table(sph["comparison_rows"], ["method", "ATE", "drift", "path_ratio", "interpretation"]),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "- A strict non-spherical retrained baseline is not available in the locked historical protocol.",
                "- Current direction diagnostics still show the direction branch is weak:",
                f"  - rot mean / median / p90 / max = `{_fmt(sph['direction_diagnostics']['rot_mean_deg'])}` / `{_fmt(sph['direction_diagnostics']['rot_median_deg'])}` / `{_fmt(sph['direction_diagnostics']['rot_p90_deg'])}` / `{_fmt(sph['direction_diagnostics']['rot_max_deg'])}` deg",
                f"  - tdir mean / median / p90 / max = `{_fmt(sph['direction_diagnostics']['tdir_mean_deg'])}` / `{_fmt(sph['direction_diagnostics']['tdir_median_deg'])}` / `{_fmt(sph['direction_diagnostics']['tdir_p90_deg'])}` / `{_fmt(sph['direction_diagnostics']['tdir_max_deg'])}` deg",
                f"  - tdir mean cosine similarity = `{_fmt(sph['direction_diagnostics']['tdir_mean_cosine_similarity'])}`",
                f"- Interpretation: {sph['interpretation']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Axis 2: Coarse-to-Fine Refinement",
            _markdown_table(
                ctf["rows"],
                ["method", "role", "ATE", "drift", "path_ratio", "delta_ATE_vs_S5", "delta_drift_vs_S5", "delta_path_ratio_vs_S5"],
            ),
            "",
            _markdown_table(
                ctf["transitions"],
                ["transition", "component_added", "delta_ATE", "delta_drift", "delta_path_ratio", "interpretation"],
            ),
            "",
            f"- Interpretation: {ctf['interpretation']}",
            "",
            "## Axis 3: Geometric Constraints",
            _markdown_table(
                geo["rows"],
                ["method", "constraint", "ATE", "drift", "path_ratio", "interpretation"],
            ),
            "",
            f"- S14 diagnostic baseline on `scene01/seq03`: ATE=`{_fmt(geo['s14_baseline_diagnostic']['ATE'])}`, drift=`{_fmt(geo['s14_baseline_diagnostic']['drift'])}`, path_ratio=`{_fmt(geo['s14_baseline_diagnostic']['path_ratio'])}`",
            f"- S14 selected diagnostic: ATE=`{_fmt(geo['s14_selected_diagnostic']['ATE'])}`, drift=`{_fmt(geo['s14_selected_diagnostic']['drift'])}`, path_ratio=`{_fmt(geo['s14_selected_diagnostic']['path_ratio'])}`, classification=`{geo['s14_selected_diagnostic']['final_classification']}`",
            f"- Interpretation: {geo['interpretation']}",
            "",
            "## Oracle Diagnostic",
            _markdown_table(
                oracle["rows"],
                ["diagnostic", "ATE", "drift", "path_ratio", "interpretation"],
            ),
            "",
        ]
    )
    for row in oracle["interpretation"]:
        lines.append(f"- {row}")
    lines.extend(
        [
            "",
            "## Thesis-Ready Interpretation",
            "- spherical formulation provides a structured direction representation, but current direction prediction remains weak.",
            "- coarse-to-fine refinement contributes the largest observed clean improvement.",
            "- geometric scale/tmag constraints stabilize the system but only marginally improve final S5.",
            "- naive stronger geometry constraints do not necessarily help, as shown by S14.",
            "- S5 remains the best clean candidate, but not practical-ready.",
            "",
            "## Caveats",
            "- small sequence protocol",
            "- representativeness caveat from S17/S18",
            "- S5 not practical-ready",
            "- no deployment readiness claim",
            "- missing strict non-spherical baseline in the locked historical protocol",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_json = _resolve_path(args.output_json, DEFAULT_OUTPUT_JSON)
    output_md = _resolve_path(args.output_md, DEFAULT_OUTPUT_MD)

    verify_payload = None
    if args.verify_s5:
        verify_payload = _verify_s5()

    manifest = _read_json(FINAL_MANIFEST)
    ab1 = _load_ab1()
    scan = _repository_scan()
    payload = {
        "name": "AB2_thesis_three_axis_ablation",
        "final_candidate": manifest["final_candidate_name"],
        "locked_metrics": dict(manifest["final_metrics"]),
        "verify_s5": verify_payload,
        "repository_scan": scan,
        "spherical_axis": _spherical_axis(ab1, scan),
        "coarse_to_fine_axis": _coarse_to_fine_axis(ab1),
        "geometric_constraints_axis": _geometric_constraints_axis(),
        "oracle_diagnostics": _oracle_axis(ab1),
        "caveats": [
            "small sequence protocol",
            "representativeness caveat from S17/S18",
            "S5 not practical-ready",
            "no deployment readiness claim",
            "missing strict non-spherical baseline in the locked historical protocol",
        ],
        "thesis_interpretation": {
            "evaluation_only": True,
            "no_new_experiment_line": True,
            "no_new_candidate_selected": True,
            "s5_remains_final_clean_candidate": True,
            "s5_not_practical_ready": True,
            "main_bottleneck": "R-TDIR-COUPLED-LIMITED",
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_markdown(payload, output_md)
    print(f"[AB2] json={output_json}")
    print(f"[AB2] md={output_md}")
    print("[AB2] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
