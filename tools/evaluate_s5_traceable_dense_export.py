#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = "external_baselines/results/s5_traceable_dense_s5d13"
DEFAULT_OUT_JSON = "checkpoints/S5D13_traceable_dense_export.json"
DEFAULT_OUT_REPORT = "reports/s5d13_traceable_dense_export_report.md"
DEFAULT_RESTORED_DENSE = "external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt"


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def _relpath(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _quat_to_r(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    x, y, z, w = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _read_tum(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists() or path.stat().st_size == 0:
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) < 8:
            continue
        rows.append({
            "timestamp": float(p[0]),
            "t": np.asarray(list(map(float, p[1:4])), dtype=np.float64),
            "R": _quat_to_r(*map(float, p[4:8])),
        })
    return rows


def _read_gt(path: Path) -> Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    rows = _read_tum(path)
    by_ts = {round(float(r["timestamp"]), 6): r for r in rows}
    out = {}
    for a, b in zip(rows[:-1], rows[1:]):
        ri, rj = a["R"], b["R"]
        ti, tj = a["t"], b["t"]
        out[(round(float(a["timestamp"]), 6), round(float(b["timestamp"]), 6))] = (ri.T @ rj, ri.T @ (tj - ti))
    return out


def _rot_deg(r: np.ndarray) -> float:
    c = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _finite(vals: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(vals), dtype=np.float64)
    return a[np.isfinite(a)]


def _pct(vals: Iterable[float], p: float) -> float | None:
    a = _finite(vals)
    return float(np.percentile(a, p)) if a.size else None


def _mean(vals: Iterable[float]) -> float | None:
    a = _finite(vals)
    return float(np.mean(a)) if a.size else None


def _metrics_for_rows(rows: List[Dict[str, Any]], gt_edges: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    vals = []
    for r in rows:
        if r.get("emitted_step_delta") is None:
            continue
        key = (round(float(r["timestamp_i"]), 6), round(float(r["timestamp_j"]), 6))
        if key not in gt_edges:
            continue
        t = np.asarray(r["emitted_step_delta"], dtype=np.float64)
        _, tgt = gt_edges[key]
        ne = float(np.linalg.norm(t))
        ng = float(np.linalg.norm(tgt))
        if ne > 1e-12 and ng > 1e-12:
            c = float(np.clip(np.dot(t / ne, tgt / ng), -1.0, 1.0))
            tdir = float(np.degrees(np.arccos(c)))
            ratio = float(ne / ng)
        else:
            tdir = math.nan
            ratio = math.nan
        vals.append({"tmag_ratio": ratio, "tdir_deg": tdir, "step_norm": ne, "gt_step_norm": ng})
    return {
        "num_edges": len(vals),
        "path_length": float(sum(v["step_norm"] for v in vals)),
        "tmag_median": _pct((v["tmag_ratio"] for v in vals), 50),
        "tmag_p90": _pct((v["tmag_ratio"] for v in vals), 90),
        "tmag_max": float(np.max(_finite(v["tmag_ratio"] for v in vals))) if _finite(v["tmag_ratio"] for v in vals).size else None,
        "tdir_mean": _mean(v["tdir_deg"] for v in vals),
        "rot_mean": None,
    }


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# S5D13 Traceable Dense Export Report",
        "",
        "## Executive summary",
        f"- final classification: `{payload['final_classification']}`",
        f"- traceable dense available: `{payload['diagnosis']['traceable_dense_available']}`",
        "",
        "## S5D12 recap",
        "- S5D12 found unknown provenance in restored dense non-selected edges and could not prove the fill mechanism.",
        "",
        "## Traceable export source audit",
        f"- {payload['traceable_export_source']}",
        "",
        "## Export coverage",
        f"- {payload['coverage']}",
        "",
        "## Edge provenance summary",
        f"- {payload['provenance_summary']}",
        "",
        "## Metrics by source type",
        f"- {payload['metrics_by_source_type']}",
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tum", required=True)
    ap.add_argument("--provenance", required=True)
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--timestamps", required=True)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-report", default=DEFAULT_OUT_REPORT)
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = _resolve(args.out_json)
    out_report = _resolve(args.out_report)
    prov_rows = [json.loads(x) for x in _resolve(args.provenance).read_text(encoding="utf-8").splitlines() if x.strip()]
    tum_rows = _read_tum(_resolve(args.tum))
    gt_edges = _read_gt(_resolve(args.groundtruth))
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for r in prov_rows:
        by_source.setdefault(r["source_type"], []).append(r)
    metrics = {k: _metrics_for_rows(v, gt_edges) for k, v in sorted(by_source.items())}

    run_rows = [r for r in prov_rows if 72 <= int(r["edge_index"]) <= 360]
    run_available = bool(run_rows) and all(r["source_type"] != "unavailable" for r in run_rows)
    step_norms = [r.get("emitted_step_norm") for r in run_rows if r.get("emitted_step_norm") is not None]
    near_constant = None
    if step_norms:
        arr = np.asarray(step_norms, dtype=np.float64)
        near_constant = bool(abs(float(np.median(arr)) - 0.134) < 0.005 and float(np.std(arr) / max(float(np.mean(arr)), 1e-12)) < 0.1)
    trace_path = out_dir / "dominant_run_72_360_trace.json"
    trace_path.write_text(json.dumps(run_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    edge_metrics_path = out_dir / "edge_metrics.json"
    summary = {
        "num_poses": len(tum_rows),
        "coverage_vs_454": float(len(tum_rows) / 454.0),
        "num_edges_by_source_type": {k: len(v) for k, v in by_source.items()},
        "metrics_by_source_type": metrics,
    }
    edge_metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    payload["coverage"].update({
        "num_poses": len(tum_rows),
        "num_edges": len(prov_rows),
        "coverage_vs_454": float(len(tum_rows) / 454.0),
        "coverage_vs_453_edges": float((len(prov_rows) - len(by_source.get("unavailable", []))) / 453.0),
    })
    payload["provenance_summary"] = {
        "num_direct_adjacent_prediction": len(by_source.get("direct_adjacent_prediction", [])),
        "num_selected_prediction": len(by_source.get("selected_prediction", [])),
        "num_interpolated_fill": len(by_source.get("interpolated_fill", [])),
        "num_composed_fill": len(by_source.get("composed_fill", [])),
        "num_unavailable": len(by_source.get("unavailable", [])),
    }
    payload["metrics_by_source_type"] = metrics
    payload["outputs"]["edge_metrics"] = _relpath(edge_metrics_path)
    payload["dominant_run_72_360"] = {
        "available": run_available,
        "trace_path": _relpath(trace_path),
        "near_constant_0p134m_steps_detected": near_constant,
        "gap_normalization_verified": None,
    }
    enough_poses = len(tum_rows) >= 2
    external_eval = {"attempted": enough_poses, "none": {}, "se3": {}, "sim3": {}}
    if enough_poses:
        for align in ("none", "se3", "sim3"):
            outp = out_dir / f"eval_alignment_{align}.json"
            subprocess.run([
                "/home/dovetao/miniconda3/envs/pytorch/bin/python",
                "tools/evaluate_external_baseline_trajectory.py",
                "--trajectory", args.tum,
                "--groundtruth", args.groundtruth,
                "--alignment", align,
                "--output-json", str(outp),
            ], cwd=REPO_ROOT, check=False)
            external_eval[align] = json.loads(outp.read_text(encoding="utf-8")) if outp.exists() else {"status": "missing"}
    payload["external_eval"] = external_eval
    payload["comparison_to_restored_dense"] = {
        "compared": enough_poses,
        "matches_restored_dense": False if not enough_poses else None,
        "path_length_ratio_traceable_over_restored": None,
        "notes": ["No full traceable dense TUM was generated; selected-only provenance cannot reproduce restored dense trajectory."],
    }
    payload["diagnosis"].update({
        "traceable_dense_available": enough_poses and payload["traceable_export_source"]["can_generate_453_adjacent_edges"],
        "selected_only_blocker": payload["traceable_export_source"]["source_type"] == "selected_only",
        "dense_fill_protocol_verified": False,
        "dense_fill_bug_confirmed": None,
        "most_likely_status": "selected_only" if payload["traceable_export_source"]["source_type"] == "selected_only" else "provenance_unavailable",
    })
    payload["final_classification"] = "S5D13_TRACEABLE_DENSE_UNAVAILABLE_SELECTED_ONLY" if payload["diagnosis"]["selected_only_blocker"] else "S5D13_PROVENANCE_UNAVAILABLE"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(out_report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
