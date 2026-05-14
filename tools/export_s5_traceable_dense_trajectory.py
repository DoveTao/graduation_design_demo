#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAIRWISE = "external_baselines/results/s5_pairwise_replay_s5d7/final_s5_pairwise_vectors_scene01_seq03.jsonl"
DEFAULT_RESTORED_DENSE = "external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt"
DEFAULT_S5D12 = "checkpoints/S5D12_dense_fill_path_audit.json"
DEFAULT_OUT_DIR = "external_baselines/results/s5_traceable_dense_s5d13"
DEFAULT_OUT_TUM = f"{DEFAULT_OUT_DIR}/scene01_seq03_traceable_dense_tum.txt"
DEFAULT_OUT_PROVENANCE = f"{DEFAULT_OUT_DIR}/edge_provenance.jsonl"
DEFAULT_OUT_JSON = "checkpoints/S5D13_traceable_dense_export.json"
DEFAULT_OUT_REPORT = "reports/s5d13_traceable_dense_export_report.md"

ALLOWED_CLASSIFICATIONS = [
    "S5D13_TRACEABLE_DENSE_EXPORT_COMPLETE",
    "S5D13_TRACEABLE_DENSE_UNAVAILABLE_SELECTED_ONLY",
    "S5D13_DENSE_FILL_BUG_CONFIRMED",
    "S5D13_PROVENANCE_UNAVAILABLE",
    "S5D13_ERROR",
]


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def _relpath(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _read_timestamps(path: Path) -> List[float]:
    vals = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            vals.append(float(s.split()[0]))
    return vals


def _read_pairwise(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        i = int(row["frame_i"])
        j = int(row["frame_j"])
        if j == i + 1:
            out[i] = row
    return out


def _source_audit(pairwise_rows: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    log_path = REPO_ROOT / "logs/s5d13_traceable_export_source_search.log"
    if not log_path.exists():
        pattern = (
            "S5_clean_tmag_calibration_policy|export_s5_dense|dense trajectory|pairwise|"
            "selected_k1|relative pose|materialize|fill|traceable|edge provenance|scene01_seq03"
        )
        cp = subprocess.run(["rg", "-n", pattern, "tools", "scripts", "reports", "checkpoints", "tests"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
        log_path.parent.mkdir(exist_ok=True)
        log_path.write_text(cp.stdout + (("\n[stderr]\n" + cp.stderr) if cp.stderr else ""), encoding="utf-8")
    can_453 = len(pairwise_rows) == 453
    return {
        "source_found": bool(pairwise_rows),
        "source_type": "adjacent_dense_direct" if can_453 else ("selected_only" if pairwise_rows else "unavailable"),
        "source_location": DEFAULT_PAIRWISE if pairwise_rows else None,
        "can_generate_453_adjacent_edges": bool(can_453),
        "source_search_log": "logs/s5d13_traceable_export_source_search.log",
        "notes": [
            "Only S5D7 selected_k1 pairwise predictions are available in the provided final-S5 artifacts."
            if pairwise_rows and not can_453
            else "Full adjacent dense source available.",
            "No GT pose or GT scale is used to generate predictions.",
        ],
    }


def _provenance_for_edge(i: int, ts: List[float], row: Dict[str, Any] | None) -> Dict[str, Any]:
    dt = float(ts[i + 1] - ts[i]) if i + 1 < len(ts) else None
    if row is None:
        return {
            "edge_index": i,
            "timestamp_i": float(ts[i]),
            "timestamp_j": float(ts[i + 1]),
            "source_type": "unavailable",
            "source_pair": None,
            "gap_id": "unavailable_nonselected_gap_72_360" if 72 <= i <= 360 else None,
            "gap_num_adjacent_steps": None,
            "gap_endpoint_delta": None,
            "emitted_step_delta": None,
            "emitted_step_norm": None,
            "dt": dt,
            "dt_units": "seconds",
            "translation_frame": "unknown",
            "rotation_convention": "unavailable",
            "uses_gt_for_prediction": False,
            "notes": ["No legal final-S5 direct adjacent prediction or documented fill source was found for this edge."],
        }
    t = [float(x) for x in row["translation"]["value"]]
    return {
        "edge_index": i,
        "timestamp_i": float(row["timestamp_i"]),
        "timestamp_j": float(row["timestamp_j"]),
        "source_type": "selected_prediction",
        "source_pair": {
            "frame_i": int(row["frame_i"]),
            "frame_j": int(row["frame_j"]),
            "pair_index": int(row.get("pair_index", -1)),
            "pair_type": row.get("pair_type", "selected_k1"),
            "source": row.get("source"),
            "candidate": row.get("candidate"),
        },
        "gap_id": None,
        "gap_num_adjacent_steps": None,
        "gap_endpoint_delta": None,
        "emitted_step_delta": t,
        "emitted_step_norm": float(np.linalg.norm(np.asarray(t, dtype=np.float64))),
        "dt": dt,
        "dt_units": "seconds",
        "translation_frame": row.get("translation", {}).get("frame", "unknown"),
        "rotation_convention": "relative_R_i_to_j_as_exported_matrix",
        "uses_gt_for_prediction": bool(row.get("gt_used_to_generate_prediction", False)),
        "notes": list(row.get("notes", [])),
    }


def _base_payload(args: argparse.Namespace, source: Dict[str, Any], num_edges: int, num_selected: int, out_tum: Path, out_prov: Path) -> Dict[str, Any]:
    classification = (
        "S5D13_TRACEABLE_DENSE_EXPORT_COMPLETE"
        if source["can_generate_453_adjacent_edges"]
        else ("S5D13_TRACEABLE_DENSE_UNAVAILABLE_SELECTED_ONLY" if source["source_type"] == "selected_only" else "S5D13_PROVENANCE_UNAVAILABLE")
    )
    return {
        "experiment": "S5D13_traceable_dense_export_with_edge_provenance",
        "inputs": {
            "s5d12_checkpoint": DEFAULT_S5D12,
            "timestamps": _relpath(_resolve(args.timestamps)),
            "groundtruth": _relpath(_resolve(args.groundtruth)),
            "selected_pairwise_jsonl": DEFAULT_PAIRWISE,
            "restored_dense_tum": DEFAULT_RESTORED_DENSE,
        },
        "traceable_export_source": source,
        "outputs": {
            "tum": _relpath(out_tum),
            "edge_provenance": _relpath(out_prov),
            "edge_metrics": f"{DEFAULT_OUT_DIR}/edge_metrics.json",
        },
        "coverage": {
            "num_poses": 0 if not source["can_generate_453_adjacent_edges"] else None,
            "num_edges": num_edges,
            "coverage_vs_454": 0.0 if not source["can_generate_453_adjacent_edges"] else None,
            "coverage_vs_453_edges": float(num_selected / 453.0),
        },
        "provenance_summary": {
            "num_direct_adjacent_prediction": 0,
            "num_selected_prediction": num_selected,
            "num_interpolated_fill": 0,
            "num_composed_fill": 0,
            "num_unavailable": max(0, num_edges - num_selected),
        },
        "metrics_by_source_type": {},
        "dominant_run_72_360": {
            "available": False,
            "trace_path": f"{DEFAULT_OUT_DIR}/dominant_run_72_360_trace.json",
            "near_constant_0p134m_steps_detected": None,
            "gap_normalization_verified": None,
        },
        "comparison_to_restored_dense": {
            "compared": False,
            "matches_restored_dense": None,
            "path_length_ratio_traceable_over_restored": None,
            "notes": ["Traceable export is selected-only/unavailable for dense gaps, so restored dense TUM is not reproduced."],
        },
        "external_eval": {"attempted": False, "none": {}, "se3": {}, "sim3": {}},
        "diagnosis": {
            "traceable_dense_available": bool(source["can_generate_453_adjacent_edges"]),
            "selected_only_blocker": bool(source["source_type"] == "selected_only" and not source["can_generate_453_adjacent_edges"]),
            "dense_fill_protocol_verified": False,
            "dense_fill_bug_confirmed": None,
            "most_likely_status": "traceable_dense_ready" if source["can_generate_453_adjacent_edges"] else ("selected_only" if source["source_type"] == "selected_only" else "provenance_unavailable"),
            "recommended_next_actions": [
                "Do not use selected-only provenance as a 454-pose dense trajectory.",
                "Recover or add diagnostic-only final-S5 adjacent dense inference if a traceable dense export is required.",
                "Expose source_type per edge in any future dense materialization path.",
                "Keep this diagnostic export separate from official S5 locked metrics.",
            ],
        },
        "s5_official_locked_metrics": {
            "ate": 7.352288,
            "drift": 1.327343,
            "path_ratio": 0.932379,
            "unchanged": True,
            "not_replaced_by_s5d13": True,
        },
        "allowed_final_classifications": ALLOWED_CLASSIFICATIONS,
        "final_classification": classification,
    }


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# S5D13 Traceable Dense Export Report",
        "",
        "## Executive summary",
        f"- final classification: `{payload['final_classification']}`",
        f"- source type: `{payload['traceable_export_source']['source_type']}`",
        f"- can generate 453 adjacent edges: `{payload['traceable_export_source']['can_generate_453_adjacent_edges']}`",
        "",
        "## S5D12 recap",
        "- S5D12 found restored dense TUM provenance unknown and dominant run 72-360 over-scaled.",
        "- S5D12 did not prove the fill-bug mechanism because pre-materialization/source code was unavailable.",
        "",
        "## Traceable export source audit",
        f"- source found: `{payload['traceable_export_source']['source_found']}`",
        f"- source location: `{payload['traceable_export_source']['source_location']}`",
        f"- search log: `{payload['traceable_export_source']['source_search_log']}`",
        "",
        "## Export coverage",
        f"- num poses: `{payload['coverage']['num_poses']}`",
        f"- num edges: `{payload['coverage']['num_edges']}`",
        f"- coverage vs 453 edges: `{payload['coverage']['coverage_vs_453_edges']}`",
        "",
        "## Edge provenance summary",
        f"- {payload['provenance_summary']}",
        "",
        "## Metrics by source type",
        f"- {payload.get('metrics_by_source_type', {})}",
        "",
        "## Dominant run 72-360 trace",
        f"- {payload['dominant_run_72_360']}",
        "",
        "## Comparison to restored dense TUM",
        f"- {payload['comparison_to_restored_dense']}",
        "",
        "## External evaluator results if available",
        f"- {payload['external_eval']}",
        "",
        "## Diagnosis",
        f"- {payload['diagnosis']}",
        "",
        "## Recommendations",
    ]
    lines.extend(f"- {x}" for x in payload["diagnosis"]["recommended_next_actions"])
    lines.extend([
        "",
        "## Caveats",
        "- diagnostic only",
        "- traceable dense export does not replace official S5 locked result",
        "- S5 locked metrics/policy unchanged",
        "- no GT used for prediction",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene01")
    ap.add_argument("--seq", default="seq03")
    ap.add_argument("--timestamps", required=True)
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--pairwise-jsonl", default=DEFAULT_PAIRWISE)
    ap.add_argument("--out-tum", default=DEFAULT_OUT_TUM)
    ap.add_argument("--out-provenance", default=DEFAULT_OUT_PROVENANCE)
    ap.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-report", default=DEFAULT_OUT_REPORT)
    args = ap.parse_args()

    ts = _read_timestamps(_resolve(args.timestamps))
    pairwise = _read_pairwise(_resolve(args.pairwise_jsonl))
    out_tum = _resolve(args.out_tum)
    out_prov = _resolve(args.out_provenance)
    out_json = _resolve(args.out_json)
    out_report = _resolve(args.out_report)
    out_tum.parent.mkdir(parents=True, exist_ok=True)
    out_prov.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    num_edges = max(0, len(ts) - 1)
    with out_prov.open("w", encoding="utf-8") as fh:
        for i in range(num_edges):
            fh.write(json.dumps(_provenance_for_edge(i, ts, pairwise.get(i)), sort_keys=True) + "\n")

    # Selected-only provenance must not masquerade as a 454-pose dense trajectory.
    out_tum.write_text("", encoding="utf-8")
    source = _source_audit(pairwise)
    payload = _base_payload(args, source, num_edges, len(pairwise), out_tum, out_prov)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(out_report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
