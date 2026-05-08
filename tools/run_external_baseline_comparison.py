#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from export_external_baseline_dataset import export_external_baseline_dataset
from evaluate_external_baseline_trajectory import evaluate_external_baseline_trajectory


DEFAULT_PROTOCOL_MD = REPO_ROOT / "reports" / "external_algorithm_baseline_protocol.md"
DEFAULT_COMPARISON_MD = REPO_ROOT / "reports" / "external_algorithm_baseline_comparison.md"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "checkpoints" / "EXT1_external_algorithm_baseline_results.json"
DEFAULT_DATASET_DIR = REPO_ROOT / "external_baselines" / "dataset" / "scene01_seq03"
DEFAULT_RESULTS_DIR = REPO_ROOT / "external_baselines" / "results"
DEFAULT_MANIFEST = REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json"


METHOD_SPECS = [
    {
        "method": "ORB-SLAM2",
        "slug": "orbslam2",
        "category": "feature-based SLAM",
        "input": "monocular / stereo / RGB-D",
        "loop_closure": "yes",
        "learning_based": "no",
        "notes": "Protocol-compatible only if built and run on the exported sequence.",
    },
    {
        "method": "ORB-SLAM3",
        "slug": "orbslam3",
        "category": "feature-based visual / visual-inertial SLAM",
        "input": "monocular / stereo / RGB-D / visual-inertial",
        "loop_closure": "yes",
        "learning_based": "no",
        "notes": "Protocol-compatible only if built and run on the exported sequence.",
    },
    {
        "method": "DSO",
        "slug": "dso",
        "category": "direct sparse visual odometry",
        "input": "monocular",
        "loop_closure": "no",
        "learning_based": "no",
        "notes": "Requires an input mode compatible with the exported panorama sequence or a documented adapter.",
    },
    {
        "method": "DROID-SLAM",
        "slug": "droidslam",
        "category": "deep learning SLAM",
        "input": "monocular / stereo / RGB-D",
        "loop_closure": "implicit / backend-dependent",
        "learning_based": "yes",
        "notes": "Protocol-compatible only after a same-sequence trajectory is generated locally.",
    },
]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw: str | Path | None, default: Path) -> Path:
    if raw is None:
        return default
    path = Path(str(raw))
    return path if path.is_absolute() else (REPO_ROOT / path)


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return str(path.resolve())


def _md_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        values = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                values.append(f"{val:.6f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _build_method_rows(dataset_dir: Path, results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seq_slug = dataset_dir.name
    for spec in METHOD_SPECS:
        traj_path = results_dir / spec["slug"] / f"{seq_slug}_est_tum.txt"
        status = "ready_to_evaluate" if traj_path.is_file() else "pending_external_run"
        rows.append(
            {
                "method": spec["method"],
                "category": spec["category"],
                "input": spec["input"],
                "loop_closure": spec["loop_closure"],
                "learning_based": spec["learning_based"],
                "run_status": status,
                "output_trajectory_path": _repo_rel(traj_path) if traj_path.exists() else "(missing)",
                "notes": spec["notes"],
            }
        )
    rows.append(
        {
            "method": "S5",
            "category": "proposed method",
            "input": "monocular panorama / equirectangular RGB",
            "loop_closure": "no",
            "learning_based": "yes",
            "run_status": "locked_internal_result",
            "output_trajectory_path": "internal clean eval / locked metrics",
            "notes": "Final clean candidate under the historical protocol.",
        }
    )
    return rows


def _evaluate_existing_external_rows(dataset_dir: Path, results_dir: Path, locked_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
    seq_slug = dataset_dir.name
    gt_path = dataset_dir / "groundtruth_tum.txt"
    rows: List[Dict[str, Any]] = [
        {
            "method": "S5",
            "alignment": "historical_clean_eval",
            "ATE": float(locked_metrics["ATE"]),
            "drift": float(locked_metrics["drift"]),
            "path_ratio": float(locked_metrics["path_ratio"]),
            "tracking_success_rate": 1.0,
            "notes": "Locked internal clean result on the historical final test sequence.",
        }
    ]
    for spec in METHOD_SPECS:
        traj_path = results_dir / spec["slug"] / f"{seq_slug}_est_tum.txt"
        if not traj_path.is_file():
            continue
        for alignment in ("none", "se3", "sim3"):
            result = evaluate_external_baseline_trajectory(gt_path=gt_path, est_path=traj_path, alignment=alignment)
            if result.get("status") != "ok":
                rows.append(
                    {
                        "method": spec["method"],
                        "alignment": alignment,
                        "ATE": float("nan"),
                        "drift": float("nan"),
                        "path_ratio": float("nan"),
                        "tracking_success_rate": float(result.get("tracking_success_rate", 0.0)),
                        "notes": f"trajectory found but evaluation incomplete: {result.get('status', 'unknown')}",
                    }
                )
                continue
            rows.append(
                {
                    "method": spec["method"],
                    "alignment": alignment,
                    "ATE": float(result["ATE"]),
                    "drift": float(result["drift"]),
                    "path_ratio": float(result["path_ratio"]),
                    "tracking_success_rate": float(result["tracking_success_rate"]),
                    "notes": "Same-sequence evaluation only; do not compare against cross-dataset published numbers.",
                }
            )
    return rows


def _write_protocol_report(path: Path, payload: Dict[str, Any]) -> None:
    dataset = payload["dataset_export"]
    methods = payload["baseline_methods"]
    lines = [
        "# External Algorithm Baseline Protocol",
        "",
        "## Scope",
        "This protocol defines a fair external baseline comparison framework for the thesis. Only trajectories generated on the same exported sequence under the same evaluation protocol can be used for numeric comparison.",
        "Published results from different datasets are not compared directly against S5.",
        "",
        "## Dataset and Protocol",
        f"- sequence: `{dataset['sequence']}`",
        "- input modality: `monocular panorama / equirectangular RGB`",
        f"- camera intrinsics availability: `camera.yaml` with pinhole-unavailable caveat at `{dataset['camera_path']}`",
        f"- GT trajectory: `{dataset['groundtruth_path']}`",
        "- alignment modes: `none / se3 / sim3`",
        "- metrics: `ATE`, `drift` (RPE-like translation RMSE), `path_ratio`, `tracking_success_rate`",
        "- scale-sensitive path_ratio must be reported in addition to any Sim(3)-aligned ATE.",
        "",
        "## Baseline Methods",
        _md_table(
            methods,
            ["method", "category", "input", "loop_closure", "learning_based", "run_status", "output_trajectory_path", "notes"],
        ),
        "",
        "## Literature Context",
        "- ORB-SLAM2 and ORB-SLAM3 represent classical feature-based SLAM baselines.",
        "- DSO represents a direct sparse monocular odometry baseline.",
        "- DROID-SLAM represents a learning-based SLAM baseline.",
        "- Published results from other datasets are used only for method context, not for direct numeric comparison with S5.",
        "",
        "## Caveats",
        "- external baselines require matching input modality",
        "- monocular baselines have scale ambiguity",
        "- Sim(3)-aligned ATE and scale-sensitive path_ratio answer different questions",
        "- results are not comparable to published numbers from different datasets",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_comparison_report(path: Path, payload: Dict[str, Any]) -> None:
    dataset = payload["dataset_export"]
    methods = payload["baseline_methods"]
    quant = payload["quantitative_comparison"]
    lines = [
        "# External Algorithm Baseline Comparison",
        "",
        "## Scope",
        "This report provides an external literature/algorithm baseline comparison framework for the thesis.",
        "Only results generated on the same exported test sequence and under the same evaluation protocol are used for numeric comparison.",
        "Published results from different datasets are not compared directly against S5.",
        "",
        "## Dataset and Protocol",
        f"- sequence: `{dataset['sequence']}`",
        "- input modality: `monocular panorama / equirectangular RGB`",
        f"- camera intrinsics availability: `{dataset['camera_path']}` with an equirectangular caveat",
        f"- GT trajectory: `{dataset['groundtruth_path']}`",
        "- alignment modes: `none / se3 / sim3`",
        "- metrics: `ATE`, `drift`, `path_ratio`, `tracking_success_rate`",
        "",
        "## Baseline Methods",
        _md_table(
            methods,
            ["method", "category", "input", "loop_closure", "learning_based", "run_status", "output_trajectory_path", "notes"],
        ),
        "",
        "## Quantitative Comparison",
        _md_table(
            quant,
            ["method", "alignment", "ATE", "drift", "path_ratio", "tracking_success_rate", "notes"],
        ),
        "",
        "## Literature Context",
        "- ORB-SLAM2/3, DSO, and DROID-SLAM are included as representative external baselines only when they produce trajectories on the exported sequence.",
        "- Published numbers from different datasets remain method context only and are not used for direct numerical comparison against S5.",
        "",
        "## Thesis-Ready Interpretation",
        "External algorithms are included as protocol-compatible baselines only when their trajectories are generated on the same test sequence.",
        "Published results from other datasets are used only for method context.",
        "S5 remains the final clean candidate under the locked historical protocol and is not practical-ready.",
        "",
        "## Caveats",
        "- external baselines require matching input modality",
        "- monocular baselines have scale ambiguity",
        "- Sim(3)-aligned ATE and scale-sensitive path_ratio answer different questions",
        "- results are not comparable to published numbers from different datasets",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the external-baseline comparison framework and pending/available comparison tables.")
    p.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="Output json path.")
    p.add_argument("--protocol-md", default=str(DEFAULT_PROTOCOL_MD), help="Protocol markdown path.")
    p.add_argument("--comparison-md", default=str(DEFAULT_COMPARISON_MD), help="Comparison markdown path.")
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="External baseline dataset export directory.")
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="External baseline results directory.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_json = _resolve_path(args.output_json, DEFAULT_OUTPUT_JSON)
    protocol_md = _resolve_path(args.protocol_md, DEFAULT_PROTOCOL_MD)
    comparison_md = _resolve_path(args.comparison_md, DEFAULT_COMPARISON_MD)
    dataset_dir = _resolve_path(args.dataset_dir, DEFAULT_DATASET_DIR)
    results_dir = _resolve_path(args.results_dir, DEFAULT_RESULTS_DIR)

    manifest = _read_json(DEFAULT_MANIFEST)
    dataset_summary = export_external_baseline_dataset(output_dir=dataset_dir)
    method_rows = _build_method_rows(dataset_dir, results_dir)
    quant_rows = _evaluate_existing_external_rows(dataset_dir, results_dir, dict(manifest["final_metrics"]))

    payload = {
        "name": "EXT1_external_algorithm_baseline_comparison",
        "final_candidate": manifest["final_candidate_name"],
        "locked_metrics": dict(manifest["final_metrics"]),
        "dataset_export": dataset_summary,
        "baseline_methods": method_rows,
        "quantitative_comparison": quant_rows,
        "caveats": [
            "external baselines require matching input modality",
            "monocular baselines have scale ambiguity",
            "Sim(3)-aligned ATE and scale-sensitive path_ratio answer different questions",
            "results are not comparable to published numbers from different datasets",
        ],
        "thesis_interpretation": {
            "evaluation_only": True,
            "no_new_candidate_selected": True,
            "s5_remains_final_clean_candidate": True,
            "s5_not_practical_ready": True,
            "same_sequence_required_for_numeric_comparison": True,
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_protocol_report(protocol_md, payload)
    _write_comparison_report(comparison_md, payload)
    print(f"[EXT1] json={output_json}")
    print(f"[EXT1] protocol={protocol_md}")
    print(f"[EXT1] comparison={comparison_md}")
    print("[EXT1] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
