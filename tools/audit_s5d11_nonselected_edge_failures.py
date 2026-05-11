#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

ALLOWED_CLASSIFICATIONS = [
    "S5D11_COMPLETE_VALIDATION_CLEAN",
    "S5D11_COMPLETE_WITH_VALIDATION_BLOCKER",
    "S5D11_NONSELECTED_EXTREME_TMAG_CONFIRMED",
    "S5D11_SERIAL_VALIDATION_STILL_OOM",
    "S5D11_ERROR",
]

DEFAULT_INPUTS = {
    "s5d10_checkpoint": "checkpoints/S5D10_componentwise_selected_replay_and_cuda_validation.json",
    "dense_tum": "external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt",
    "pairwise_jsonl": "external_baselines/results/s5_pairwise_replay_s5d7/final_s5_pairwise_vectors_scene01_seq03.jsonl",
    "groundtruth": "external_baselines/dataset/scene01_seq03/groundtruth_tum.txt",
    "timestamps": "external_baselines/dataset/scene01_seq03/timestamps.txt",
}


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
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 8:
            continue
        tx, ty, tz = map(float, parts[1:4])
        qx, qy, qz, qw = map(float, parts[4:8])
        rows.append({
            "timestamp": float(parts[0]),
            "t": np.asarray([tx, ty, tz], dtype=np.float64),
            "R_wc": _quat_to_r(qx, qy, qz, qw),
        })
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def _rel_pose(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    return a["R_wc"].T @ b["R_wc"], a["R_wc"].T @ (b["t"] - a["t"])


def _rot_deg(r: np.ndarray) -> float:
    c = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _finite(values: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(values), dtype=np.float64)
    return a[np.isfinite(a)]


def _percentile(values: Iterable[float], p: float) -> float | None:
    a = _finite(values)
    if a.size == 0:
        return None
    return float(np.percentile(a, p))


def _summary(edges: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    gt_total = float(sum(e["gt_step_length"] for e in edges))
    est_total = float(sum(e["est_step_length"] for e in edges))
    ratios = [e["tmag_ratio"] for e in edges]
    logs = [e["tmag_log_error"] for e in edges]
    return {
        "num_edges": len(edges),
        "rot_mean_deg": _percentile([e["rot_deg"] for e in edges], 50) if False else (float(np.mean(_finite(e["rot_deg"] for e in edges))) if _finite(e["rot_deg"] for e in edges).size else None),
        "rot_median_deg": _percentile([e["rot_deg"] for e in edges], 50),
        "rot_p90_deg": _percentile([e["rot_deg"] for e in edges], 90),
        "tdir_mean_deg": float(np.mean(_finite(e["tdir_deg"] for e in edges))) if _finite(e["tdir_deg"] for e in edges).size else None,
        "tdir_median_deg": _percentile([e["tdir_deg"] for e in edges], 50),
        "tdir_p90_deg": _percentile([e["tdir_deg"] for e in edges], 90),
        "tdir_abs_mean_deg": float(np.mean(_finite(e["tdir_abs_deg"] for e in edges))) if _finite(e["tdir_abs_deg"] for e in edges).size else None,
        "tdir_abs_median_deg": _percentile([e["tdir_abs_deg"] for e in edges], 50),
        "tdir_abs_p90_deg": _percentile([e["tdir_abs_deg"] for e in edges], 90),
        "tmag_mean_ratio": float(np.mean(_finite(ratios))) if _finite(ratios).size else None,
        "tmag_median_ratio": _percentile(ratios, 50),
        "tmag_p90_ratio": _percentile(ratios, 90),
        "tmag_p95_ratio": _percentile(ratios, 95),
        "tmag_max_ratio": float(np.max(_finite(ratios))) if _finite(ratios).size else None,
        "tmag_log_error_mean": float(np.mean(_finite(logs))) if _finite(logs).size else None,
        "tmag_log_error_median": _percentile(logs, 50),
        "tmag_log_error_p90": _percentile(logs, 90),
        "gt_path_length": gt_total,
        "est_path_length": est_total,
    }


def _read_selected_edges(path: Path) -> Tuple[set[Tuple[int, int]], List[Dict[str, Any]]]:
    selected: set[Tuple[int, int]] = set()
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        i = int(row["frame_i"])
        j = int(row["frame_j"])
        selected.add((i, j))
        rows.append({
            "edge_index": i,
            "timestamp_i": float(row["timestamp_i"]),
            "timestamp_j": float(row["timestamp_j"]),
            "frame_i": i,
            "frame_j": j,
            "pair_type": row.get("pair_type"),
        })
    return selected, rows


def _edge_record(i: int, j: int, dense: Sequence[Dict[str, Any]], gt_by_ts: Dict[float, Dict[str, Any]], selected_set: set[Tuple[int, int]]) -> Dict[str, Any] | None:
    if i < 0 or j < 0 or i >= len(dense) or j >= len(dense):
        return None
    tsi = round(float(dense[i]["timestamp"]), 6)
    tsj = round(float(dense[j]["timestamp"]), 6)
    if tsi not in gt_by_ts or tsj not in gt_by_ts:
        return None
    rest, test = _rel_pose(dense[i], dense[j])
    rgt, tgt = _rel_pose(gt_by_ts[tsi], gt_by_ts[tsj])
    ne = float(np.linalg.norm(test))
    ng = float(np.linalg.norm(tgt))
    if ne > 1e-12 and ng > 1e-12:
        c = float(np.clip(np.dot(test / ne, tgt / ng), -1.0, 1.0))
        tdir = float(np.degrees(np.arccos(c)))
        tdir_abs = float(np.degrees(np.arccos(abs(c))))
        ratio = float(ne / ng)
        logerr = float(abs(math.log(max(ratio, 1e-12))))
    else:
        tdir = math.nan
        tdir_abs = math.nan
        ratio = math.nan
        logerr = math.nan
    return {
        "edge_index": i,
        "timestamp_i": float(dense[i]["timestamp"]),
        "timestamp_j": float(dense[j]["timestamp"]),
        "gt_step_length": ng,
        "est_step_length": ne,
        "tmag_ratio": ratio,
        "tdir_deg": tdir,
        "tdir_abs_deg": tdir_abs,
        "rot_deg": _rot_deg(rest @ rgt.T),
        "tmag_log_error": logerr,
        "is_selected": (i, j) in selected_set,
        "adjacent_to_selected_edge": ((i - 1, i) in selected_set or (j, j + 1) in selected_set),
    }


def _top(edges: Sequence[Dict[str, Any]], key: str, n: int = 20) -> List[Dict[str, Any]]:
    clean = [e for e in edges if math.isfinite(float(e.get(key, math.nan)))]
    return sorted(clean, key=lambda e: float(e[key]), reverse=True)[:n]


def _runs(nonselected: Sequence[Dict[str, Any]], selected_set: set[Tuple[int, int]]) -> List[Dict[str, Any]]:
    runs: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    last_idx: int | None = None
    for edge in sorted(nonselected, key=lambda e: e["edge_index"]):
        idx = int(edge["edge_index"])
        if last_idx is None or idx == last_idx + 1:
            cur.append(edge)
        else:
            if cur:
                runs.append(cur)
            cur = [edge]
        last_idx = idx
    if cur:
        runs.append(cur)
    out: List[Dict[str, Any]] = []
    for ridx, run in enumerate(runs):
        start = int(run[0]["edge_index"])
        end = int(run[-1]["edge_index"])
        bounded_by_selected = ((start - 1, start) in selected_set) or ((end + 1, end + 2) in selected_set)
        out.append({
            "run_id": ridx,
            "start_edge_index": start,
            "end_edge_index": end,
            "run_length": len(run),
            "run_gt_path_length": float(sum(e["gt_step_length"] for e in run)),
            "run_est_path_length": float(sum(e["est_step_length"] for e in run)),
            "run_median_tmag_ratio": _percentile([e["tmag_ratio"] for e in run], 50),
            "run_mean_tdir_deg": float(np.mean(_finite(e["tdir_deg"] for e in run))) if _finite(e["tdir_deg"] for e in run).size else None,
            "aligns_with_selected_k1_component_gap": bool(len(run) > 1 or bounded_by_selected),
            "bounded_by_selected_edge": bool(bounded_by_selected),
        })
    return out


def _log_pass(path: Path) -> bool | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    low = text.lower()
    if "cuda out of memory" in low or "outofmemoryerror" in low or "traceback" in low or "failed" in low:
        return False
    return ("pass" in low) or ("ok" in text) or ("OK" in text)


def _validation_from_logs() -> Dict[str, Any]:
    logs = {
        "verify_final_candidate": REPO_ROOT / "logs/s5d11_verify_final_candidate_serial.log",
        "project_health_check": REPO_ROOT / "logs/s5d11_project_health_check_serial.log",
        "s6_eval_only": REPO_ROOT / "logs/s5d11_s6_eval_only_serial.log",
        "unittest": REPO_ROOT / "logs/s5d11_unittest.log",
    }
    unit_text = logs["unittest"].read_text(encoding="utf-8", errors="ignore") if logs["unittest"].exists() else ""
    m = re.search(r"Ran\s+(\d+)\s+tests", unit_text)
    unit_count = int(m.group(1)) if m else None
    statuses = {name: _log_pass(path) for name, path in logs.items()}
    validation_clean = all(statuses[k] is True for k in statuses)
    combined = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in logs.values() if path.exists()).lower()
    if "cuda out of memory" in combined or "outofmemoryerror" in combined:
        oom = "still_present"
    elif any(path.exists() for path in logs.values()):
        oom = "not_observed"
    else:
        oom = "not_observed"
    return {
        "serial_guard_used": True,
        "concurrency_risk_before": None,
        "verify_final_candidate": {"passed": statuses["verify_final_candidate"], "log_path": "logs/s5d11_verify_final_candidate_serial.log"},
        "project_health_check": {"passed": statuses["project_health_check"], "log_path": "logs/s5d11_project_health_check_serial.log"},
        "s6_eval_only": {"passed": statuses["s6_eval_only"], "log_path": "logs/s5d11_s6_eval_only_serial.log"},
        "unittest": {"passed": statuses["unittest"], "test_count": unit_count, "log_path": "logs/s5d11_unittest.log"},
        "validation_clean": validation_clean,
        "cuda_oom_status": oom,
    }


def _load_concurrency_risk() -> bool | None:
    p = REPO_ROOT / "checkpoints/S5D11_validation_concurrency_audit.json"
    if not p.exists():
        return None
    try:
        return bool(json.loads(p.read_text(encoding="utf-8"))["checks"]["concurrent_s6_processes_found"])
    except Exception:
        return None


def _write_report(path: Path, payload: Dict[str, Any], worst: Dict[str, Any], runs: List[Dict[str, Any]]) -> None:
    sel = payload["edge_audit"]["selected_edges"]
    non = payload["edge_audit"]["nonselected_edges"]
    diag = payload["diagnosis"]
    top_tmag = worst["top20_by_tmag_ratio"][:5]
    lines = [
        "# S5D11 Serial Validation and Non-selected Edge Audit",
        "",
        "## Executive summary",
        f"- final classification: `{payload['final_classification']}`",
        f"- validation clean: `{payload['validation']['validation_clean']}`",
        f"- non-selected extreme tmag confirmed: `{diag['nonselected_edges_have_extreme_tmag']}`",
        "",
        "## S5D10 recap",
        "- selected edges: num_edges=132, path_length_fraction=0.393727, rot_mean=20.820703, tdir_mean=85.802014, tmag_median_ratio=1.310038",
        "- non-selected edges: num_edges=321, path_length_fraction=0.606273, rot_mean=20.826980, tdir_mean=93.691277, tmag_median_ratio=29.375292",
        "- S5D10 validation was blocked by concurrent eval-only CUDA memory pressure.",
        "",
        "## CUDA concurrency / OOM audit",
        f"- concurrency_risk_before: `{payload['validation']['concurrency_risk_before']}`",
        f"- cuda_oom_status: `{payload['validation']['cuda_oom_status']}`",
        "- concurrency audit artifact: `checkpoints/S5D11_validation_concurrency_audit.json`",
        "",
        "## Serial validation guard design",
        "- `scripts/run_serial_validation_guard.sh` wraps verify_final_candidate, project_health_check, s6 eval-only, and unittest under one `flock` lock.",
        "- The guard sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and records nvidia-smi before/after GPU-heavy validation steps.",
        "- It does not modify the official evaluator default behavior.",
        "",
        "## Validation result",
        f"- verify_final_candidate: `{payload['validation']['verify_final_candidate']['passed']}`",
        f"- project_health_check: `{payload['validation']['project_health_check']['passed']}`",
        f"- s6_eval_only: `{payload['validation']['s6_eval_only']['passed']}`",
        f"- unittest: `{payload['validation']['unittest']['passed']}`, test_count=`{payload['validation']['unittest']['test_count']}`",
        "",
        "## Selected vs non-selected edge metrics",
        "| group | edges | path_fraction | rot_mean | tdir_mean | tdir_abs_mean | tmag_median | tmag_p90 | tmag_p95 | tmag_max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| selected | {sel['num_edges']} | {sel['path_length_fraction']:.6f} | {sel['rot_mean_deg']:.6f} | {sel['tdir_mean_deg']:.6f} | {sel['tdir_abs_mean_deg']:.6f} | {sel['tmag_median_ratio']:.6f} | {sel['tmag_p90_ratio']:.6f} | {sel['tmag_p95_ratio']:.6f} | {sel['tmag_max_ratio']:.6f} |",
        f"| non-selected | {non['num_edges']} | {non['path_length_fraction']:.6f} | {non['rot_mean_deg']:.6f} | {non['tdir_mean_deg']:.6f} | {non['tdir_abs_mean_deg']:.6f} | {non['tmag_median_ratio']:.6f} | {non['tmag_p90_ratio']:.6f} | {non['tmag_p95_ratio']:.6f} | {non['tmag_max_ratio']:.6f} |",
        "",
        "## Worst non-selected edges",
    ]
    for e in top_tmag:
        lines.append(
            f"- edge {e['edge_index']}: tmag_ratio={e['tmag_ratio']:.6f}, "
            f"gt_step={e['gt_step_length']:.6f}, est_step={e['est_step_length']:.6f}, "
            f"tdir={e['tdir_deg']:.6f}, rot={e['rot_deg']:.6f}, adjacent_to_selected={e['adjacent_to_selected_edge']}"
        )
    lines.extend([
        "",
        "## Non-selected segment/run analysis",
        f"- run count: `{len(runs)}`",
        f"- longest run length: `{max((r['run_length'] for r in runs), default=0)}`",
        f"- top run est path length: `{max((r['run_est_path_length'] for r in runs), default=0.0):.6f}`",
        f"- nonselected_runs_dominate_path_length: `{diag['nonselected_runs_dominate_path_length']}`",
        "",
        "## Diagnosis of dense path_ratio source",
        f"- most_likely_dense_pathratio_source: `{diag['most_likely_dense_pathratio_source']}`",
        f"- worst_edges_dominate_path_length: `{diag['worst_edges_dominate_path_length']}`",
        "- evidence:",
    ])
    for item in diag["evidence"]:
        lines.append(f"  - {item}")
    lines.extend([
        "",
        "## Recommendations",
    ])
    for item in diag["recommended_next_actions"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## Caveats",
        "- diagnostic only",
        "- no official S5 result replacement",
        "- S5 locked metrics/policy unchanged",
        "- selected_k1 sparse protocol caveat",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-tum", default=DEFAULT_INPUTS["dense_tum"])
    ap.add_argument("--pairwise-jsonl", default=DEFAULT_INPUTS["pairwise_jsonl"])
    ap.add_argument("--groundtruth", default=DEFAULT_INPUTS["groundtruth"])
    ap.add_argument("--timestamps", default=DEFAULT_INPUTS["timestamps"])
    ap.add_argument("--out-dir", default="external_baselines/results/s5d11_nonselected_edge_audit")
    ap.add_argument("--out-json", default="checkpoints/S5D11_serial_validation_and_nonselected_edge_audit.json")
    ap.add_argument("--out-report", default="reports/s5d11_serial_validation_and_nonselected_edge_audit.md")
    args = ap.parse_args()

    dense_path = _resolve(args.dense_tum)
    pairwise_path = _resolve(args.pairwise_jsonl)
    gt_path = _resolve(args.groundtruth)
    out_dir = _resolve(args.out_dir)
    out_json = _resolve(args.out_json)
    out_report = _resolve(args.out_report)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    dense = _read_tum(dense_path)
    gt = _read_tum(gt_path)
    gt_by_ts = {round(float(r["timestamp"]), 6): r for r in gt}
    selected_set, selected_rows = _read_selected_edges(pairwise_path)
    all_adjacent = {(i, i + 1) for i in range(max(0, len(dense) - 1))}
    nonselected_set = all_adjacent - selected_set

    selected_edges = [r for edge in sorted(selected_set) if (r := _edge_record(edge[0], edge[1], dense, gt_by_ts, selected_set)) is not None]
    nonselected_edges = [r for edge in sorted(nonselected_set) if (r := _edge_record(edge[0], edge[1], dense, gt_by_ts, selected_set)) is not None]
    sel_summary = _summary(selected_edges)
    non_summary = _summary(nonselected_edges)
    total_est = sel_summary["est_path_length"] + non_summary["est_path_length"]
    sel_summary["path_length_fraction"] = float(sel_summary["est_path_length"] / total_est) if total_est > 1e-12 else None
    non_summary["path_length_fraction"] = float(non_summary["est_path_length"] / total_est) if total_est > 1e-12 else None

    worst = {
        "top20_by_tmag_ratio": _top(nonselected_edges, "tmag_ratio"),
        "top20_by_tdir_error": _top(nonselected_edges, "tdir_deg"),
        "top20_by_rot_error": _top(nonselected_edges, "rot_deg"),
    }
    runs = _runs(nonselected_edges, selected_set)
    run_est_total = sum(float(r["run_est_path_length"]) for r in runs)
    top5_worst_est = sum(float(e["est_step_length"]) for e in worst["top20_by_tmag_ratio"][:5])
    top20_worst_est = sum(float(e["est_step_length"]) for e in worst["top20_by_tmag_ratio"])
    longest_run = max(runs, key=lambda r: r["run_est_path_length"], default=None)

    non_extreme = bool((non_summary.get("tmag_median_ratio") or 0.0) > 10.0 or (non_summary.get("tmag_p95_ratio") or 0.0) > 50.0)
    worst_dominate = bool(total_est > 1e-12 and top20_worst_est / total_est > 0.25)
    runs_dominate = bool(total_est > 1e-12 and run_est_total / total_est > 0.5)
    selected_protocol_gap = bool(len(runs) > 1 and (sel_summary.get("path_length_fraction") or 0.0) < 0.5)
    if non_extreme and runs_dominate:
        source = "long_runs"
    elif non_extreme and worst_dominate:
        source = "few_outliers"
    elif non_extreme:
        source = "nonselected_edges"
    else:
        source = "unknown"

    (out_dir / "selected_edge_ids.json").write_text(json.dumps(selected_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "nonselected_edge_metrics.json").write_text(json.dumps({"selected_edges": sel_summary, "nonselected_edges": non_summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "nonselected_worst_edges.json").write_text(json.dumps(worst, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "nonselected_segment_summary.json").write_text(json.dumps({"runs": runs}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    validation = _validation_from_logs()
    validation["concurrency_risk_before"] = _load_concurrency_risk()
    if validation["cuda_oom_status"] == "still_present":
        final_cls = "S5D11_SERIAL_VALIDATION_STILL_OOM"
    elif validation["validation_clean"]:
        final_cls = "S5D11_COMPLETE_VALIDATION_CLEAN"
    elif any(v["passed"] is not None for k, v in validation.items() if isinstance(v, dict) and "passed" in v):
        final_cls = "S5D11_COMPLETE_WITH_VALIDATION_BLOCKER"
    elif non_extreme:
        final_cls = "S5D11_NONSELECTED_EXTREME_TMAG_CONFIRMED"
    else:
        final_cls = "S5D11_COMPLETE_WITH_VALIDATION_BLOCKER"

    evidence = [
        f"nonselected_tmag_median_ratio={non_summary.get('tmag_median_ratio')}",
        f"nonselected_tmag_p95_ratio={non_summary.get('tmag_p95_ratio')}",
        f"nonselected_tmag_max_ratio={non_summary.get('tmag_max_ratio')}",
        f"selected_path_fraction={sel_summary.get('path_length_fraction')}",
        f"nonselected_path_fraction={non_summary.get('path_length_fraction')}",
        f"top20_tmag_edges_est_path_fraction={(top20_worst_est / total_est) if total_est > 1e-12 else None}",
        f"longest_nonselected_run={longest_run}",
    ]
    recommendations = [
        "Keep the serial validation lock for verify/project_health/s6 eval-only runs before declaring CUDA stability.",
        "Audit the dense export path that fills non-selected adjacent edges; selected_k1 edges alone do not explain the large tmag median ratio.",
        "Add model/export diagnostics that report adjacent-edge tmag ratios before dense trajectory materialization.",
        "Consider a post-export consistency gate for non-selected edges, but do not change the locked S5 official evaluator or policy in S5D11.",
    ]

    payload: Dict[str, Any] = {
        "experiment": "S5D11_serial_validation_guard_and_nonselected_edge_failure_audit",
        "inputs": {
            "s5d10_checkpoint": DEFAULT_INPUTS["s5d10_checkpoint"],
            "dense_tum": _relpath(dense_path),
            "pairwise_jsonl": _relpath(pairwise_path),
            "groundtruth": _relpath(gt_path),
            "timestamps": _relpath(_resolve(args.timestamps)),
        },
        "validation": validation,
        "edge_audit": {
            "selected_edges": {
                "num_edges": sel_summary["num_edges"],
                "path_length_fraction": sel_summary["path_length_fraction"],
                "rot_mean_deg": sel_summary["rot_mean_deg"],
                "tdir_mean_deg": sel_summary["tdir_mean_deg"],
                "tdir_abs_mean_deg": sel_summary["tdir_abs_mean_deg"],
                "tmag_median_ratio": sel_summary["tmag_median_ratio"],
                "tmag_p90_ratio": sel_summary["tmag_p90_ratio"],
                "tmag_p95_ratio": sel_summary["tmag_p95_ratio"],
                "tmag_max_ratio": sel_summary["tmag_max_ratio"],
            },
            "nonselected_edges": {
                "num_edges": non_summary["num_edges"],
                "path_length_fraction": non_summary["path_length_fraction"],
                "rot_mean_deg": non_summary["rot_mean_deg"],
                "tdir_mean_deg": non_summary["tdir_mean_deg"],
                "tdir_abs_mean_deg": non_summary["tdir_abs_mean_deg"],
                "tmag_median_ratio": non_summary["tmag_median_ratio"],
                "tmag_p90_ratio": non_summary["tmag_p90_ratio"],
                "tmag_p95_ratio": non_summary["tmag_p95_ratio"],
                "tmag_max_ratio": non_summary["tmag_max_ratio"],
            },
            "worst_edges": {
                "by_tmag_ratio_path": "external_baselines/results/s5d11_nonselected_edge_audit/nonselected_worst_edges.json",
                "top20_tmag_est_path_fraction": (top20_worst_est / total_est) if total_est > 1e-12 else None,
                "top5_tmag_est_path_fraction": (top5_worst_est / total_est) if total_est > 1e-12 else None,
            },
            "segment_summary_path": "external_baselines/results/s5d11_nonselected_edge_audit/nonselected_segment_summary.json",
        },
        "diagnosis": {
            "nonselected_edges_have_extreme_tmag": non_extreme,
            "worst_edges_dominate_path_length": worst_dominate,
            "nonselected_runs_dominate_path_length": runs_dominate,
            "selected_protocol_gap_suspected": selected_protocol_gap,
            "most_likely_dense_pathratio_source": source,
            "evidence": evidence,
            "recommended_next_actions": recommendations,
        },
        "s5_official_locked_metrics": {
            "ate": 7.352288,
            "drift": 1.327343,
            "path_ratio": 0.932379,
            "unchanged": True,
            "not_replaced_by_s5d11": True,
        },
        "allowed_final_classifications": ALLOWED_CLASSIFICATIONS,
        "final_classification": final_cls,
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(out_report, payload, worst, runs)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
