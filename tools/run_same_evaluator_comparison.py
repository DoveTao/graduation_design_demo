#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/home/dovetao/miniconda3/envs/pytorch/bin/python"

ALLOWED_CLASSIFICATIONS = [
    "S5D_SAME_EVALUATOR_COMPARISON_COMPLETE",
    "S5D_SAME_EVALUATOR_COMPARISON_PARTIAL",
    "S5D_SAME_EVALUATOR_COMPARISON_BLOCKED",
    "S5D_SAME_EVALUATOR_COMPARISON_ERROR",
]

DEFAULT_S5_EXPORT_JSON = REPO_ROOT / "checkpoints" / "S5D_dense_external_export.json"
DEFAULT_S5_TRAJ = REPO_ROOT / "external_baselines" / "results" / "s5_dense" / "scene01_seq03_s5_dense_est_tum.txt"
DEFAULT_ORB_TRAJ = REPO_ROOT / "external_baselines" / "results" / "orbslam3_fisheye_cam0" / "scene01_seq03_est_tum.txt"
DEFAULT_GT = REPO_ROOT / "external_baselines" / "dataset" / "scene01_seq03" / "groundtruth_tum.txt"
DEFAULT_OUT_JSON = REPO_ROOT / "checkpoints" / "S5D_same_evaluator_comparison_results.json"
DEFAULT_REPORT = REPO_ROOT / "reports" / "s5_orbslam3_same_evaluator_comparison.md"
DEFAULT_ORB1D_JSON = REPO_ROOT / "checkpoints" / "ORB1d_orbslam3_fisheye_evaluation_results.json"
DEFAULT_S5_RESULTS_DIR = REPO_ROOT / "external_baselines" / "results" / "s5_dense"
DEFAULT_ORB_RESULTS_DIR = REPO_ROOT / "external_baselines" / "results" / "orbslam3_fisheye_cam0"


def _resolve(path: str | Path) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else REPO_ROOT / p


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_triplet(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ate": result.get("ATE"),
        "drift": result.get("drift"),
        "path_ratio": result.get("path_ratio"),
        "status": result.get("status"),
        "num_matched_poses": result.get("num_matched_poses"),
    }


def _run_evaluator(traj: Path, gt: Path, out_dir: Path) -> Dict[str, Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}
    for alignment in ("none", "se3", "sim3"):
        out_json = out_dir / f"eval_alignment_{alignment}.json"
        cmd = [
            PYTHON,
            "tools/evaluate_external_baseline_trajectory.py",
            "--trajectory",
            _rel(traj),
            "--groundtruth",
            _rel(gt),
            "--alignment",
            alignment,
            "--output-json",
            _rel(out_json),
        ]
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        results[alignment] = _read_json(out_json)
    return results


def _load_orb_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for alignment in ("none", "se3", "sim3"):
        out[alignment] = _read_json(results_dir / f"eval_alignment_{alignment}.json")
    return out


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    s5 = payload["s5"]
    orb = payload["orbslam3"]
    rows: List[Dict[str, Any]] = []
    for method, input_name, bundle, notes in [
        ("S5 dense external export", "S5 panorama/project representation", s5, "S5 diagnostic dense export; official locked result unchanged"),
        ("ORB-SLAM3 fisheye cam0", "raw fisheye cam0", orb, "fitted KB8 compatibility YAML; partial tracking coverage"),
    ]:
        for alignment in ("none", "se3", "sim3"):
            r = bundle["alignment_results"][alignment]
            rows.append(
                {
                    "Method": method,
                    "Input": input_name,
                    "Evaluator": "external evaluator",
                    "Alignment": alignment,
                    "ATE": r.get("ate"),
                    "Drift": r.get("drift"),
                    "Path ratio": r.get("path_ratio"),
                    "Estimated poses": bundle.get("num_est_poses"),
                    "Input frames": bundle.get("num_input_frames"),
                    "Coverage": bundle.get("coverage"),
                    "Notes": notes,
                }
            )

    table = [
        "| Method | Input | Evaluator | Alignment | ATE | Drift | Path ratio | Estimated poses | Input frames | Coverage | Notes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        table.append(
            "| {Method} | {Input} | {Evaluator} | {Alignment} | {ATE} | {Drift} | {Path ratio} | {Estimated poses} | {Input frames} | {Coverage} | {Notes} |".format(
                **{k: row[k] for k in row}
            )
        )

    coverage_table = [
        "| Method | Estimated poses | Input frames | Coverage |",
        "| --- | ---: | ---: | ---: |",
        f"| S5 dense external export | {s5['num_est_poses']} | {s5['num_input_frames']} | {s5['coverage']} |",
        f"| ORB-SLAM3 fisheye cam0 | {orb['num_est_poses']} | {orb['num_input_frames']} | {orb['coverage']} |",
    ]

    lines = [
        "# S5 vs ORB-SLAM3 Same-Evaluator Comparison",
        "",
        "## Executive summary",
        "",
        f"- Experiment: `{payload['experiment']}`",
        f"- Final classification: `{payload['final_classification']}`",
        "- Checkpoint: `checkpoints/S5D_same_evaluator_comparison_results.json`",
        f"- Comparison type: `{payload['comparison_type']}`",
        f"- same_input_protocol: `{payload['same_input_protocol']}`",
        "",
        "S5 official locked metrics are unchanged and remain distinct from this external same-evaluator comparison.",
        "This external comparison does not replace the official S5 locked result.",
        "",
        "## Comparison protocol",
        "",
        "Same sequence, same GT, same external evaluator, same alignment modes: `none`, `se3`, `sim3`.",
        "",
        "## Input protocol note",
        "",
        payload["input_protocol_note"],
        "",
        "## S5 external trajectory summary",
        "",
        f"- Trajectory: `{s5['trajectory']}`",
        f"- Estimated poses: `{s5['num_est_poses']}`",
        f"- Input frames: `{s5['num_input_frames']}`",
        f"- Coverage: `{s5['coverage']}`",
        "",
        "## ORB-SLAM3 external trajectory summary",
        "",
        f"- Trajectory: `{orb['trajectory']}`",
        f"- Estimated poses: `{orb['num_est_poses']}`",
        f"- Input frames: `{orb['num_input_frames']}`",
        f"- Coverage: `{orb['coverage']}`",
        f"- Calibration caveat: `{orb['calibration_caveat']}`",
        "",
        "## Metrics table",
        "",
        *table,
        "",
        "## Coverage table",
        "",
        *coverage_table,
        "",
        "## Interpretation",
        "",
        "S5 and ORB-SLAM3 are compared here under the same trajectory evaluator and alignment policy, but they do not use identical input modalities. ORB-SLAM3 has partial tracking coverage, while the S5 dense diagnostic export covers the full timestamp stream. S5 official locked result remains unchanged.",
        "",
        "## Final classification",
        "",
        f"`{payload['final_classification']}`",
        "",
        "Allowed classifications: " + ", ".join(f"`{x}`" for x in ALLOWED_CLASSIFICATIONS) + ".",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run same-GT same-external-evaluator comparison for S5 dense export and ORB-SLAM3.")
    p.add_argument("--s5-export-json", default=str(DEFAULT_S5_EXPORT_JSON))
    p.add_argument("--s5-trajectory", default=str(DEFAULT_S5_TRAJ))
    p.add_argument("--orbslam3-trajectory", default=str(DEFAULT_ORB_TRAJ))
    p.add_argument("--groundtruth", default=str(DEFAULT_GT))
    p.add_argument("--orb1d-json", default=str(DEFAULT_ORB1D_JSON))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-report", default=str(DEFAULT_REPORT))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_json = _resolve(args.out_json)
    out_report = _resolve(args.out_report)
    s5_export_json = _resolve(args.s5_export_json)
    s5_traj = _resolve(args.s5_trajectory)
    orb_traj = _resolve(args.orbslam3_trajectory)
    orb1d_json = _resolve(args.orb1d_json)
    gt = _resolve(args.groundtruth)
    try:
        s5_export = _read_json(s5_export_json)
        if s5_export.get("final_classification") not in {"S5D_DENSE_EXPORT_READY", "S5D_SPARSE_EXPORT_ONLY"}:
            classification = "S5D_SAME_EVALUATOR_COMPARISON_BLOCKED"
            s5_results = {}
        else:
            s5_results = _run_evaluator(s5_traj, gt, DEFAULT_S5_RESULTS_DIR)
            classification = "S5D_SAME_EVALUATOR_COMPARISON_COMPLETE" if s5_export.get("final_classification") == "S5D_DENSE_EXPORT_READY" else "S5D_SAME_EVALUATOR_COMPARISON_PARTIAL"
        orb_results = _load_orb_results(DEFAULT_ORB_RESULTS_DIR)
        orb1d = _read_json(orb1d_json)
        orb_traj_meta = orb1d.get("trajectory", {})
        s5_alignment = {k: _metric_triplet(v) for k, v in s5_results.items()}
        orb_alignment = {k: _metric_triplet(v) for k, v in orb_results.items()}
        payload: Dict[str, Any] = {
            "experiment": "S5D_same_evaluator_comparison",
            "comparison_type": "same_gt_same_external_evaluator_same_alignment_policy",
            "same_input_protocol": False,
            "input_protocol_note": "S5 uses the project panorama/S5 representation, while ORB-SLAM3 uses raw fisheye cam0. The comparison is fair at the trajectory-evaluator level, not identical at the input-modality level.",
            "s5": {
                "trajectory": _rel(s5_traj),
                "num_est_poses": s5_export.get("export", {}).get("num_est_poses"),
                "num_input_frames": s5_export.get("timestamps", {}).get("num_timestamps"),
                "coverage": s5_export.get("export", {}).get("coverage"),
                "alignment_results": s5_alignment,
            },
            "orbslam3": {
                "trajectory": _rel(orb_traj),
                "num_est_poses": orb_traj_meta.get("num_est_poses"),
                "num_input_frames": orb_traj_meta.get("num_input_frames"),
                "coverage": orb_traj_meta.get("tracking_success_rate"),
                "calibration_caveat": "fitted KB8 compatibility approximation, not native factory KB8",
                "alignment_results": orb_alignment,
            },
            "s5_official_locked_metrics": {
                "ate": 7.352288,
                "drift": 1.327343,
                "path_ratio": 0.932379,
                "unchanged": True,
                "not_replaced_by_external_comparison": True,
            },
            "allowed_classifications": ALLOWED_CLASSIFICATIONS,
            "final_classification": classification,
        }
    except Exception as exc:
        payload = {
            "experiment": "S5D_same_evaluator_comparison",
            "comparison_type": "same_gt_same_external_evaluator_same_alignment_policy",
            "same_input_protocol": False,
            "input_protocol_note": "comparison failed before completion",
            "s5": {"trajectory": _rel(s5_traj), "num_est_poses": None, "num_input_frames": 454, "coverage": None, "alignment_results": {}},
            "orbslam3": {"trajectory": _rel(orb_traj), "num_est_poses": None, "num_input_frames": None, "coverage": None, "alignment_results": {}},
            "s5_official_locked_metrics": {
                "ate": 7.352288,
                "drift": 1.327343,
                "path_ratio": 0.932379,
                "unchanged": True,
                "not_replaced_by_external_comparison": True,
            },
            "allowed_classifications": ALLOWED_CLASSIFICATIONS,
            "final_classification": "S5D_SAME_EVALUATOR_COMPARISON_ERROR",
            "notes": [f"{type(exc).__name__}: {exc}"],
        }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_report(out_report, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0 if payload["final_classification"] != "S5D_SAME_EVALUATOR_COMPARISON_ERROR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
