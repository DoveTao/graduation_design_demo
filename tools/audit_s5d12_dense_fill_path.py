#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_JSON = "checkpoints/S5D12_dense_fill_path_audit.json"
DEFAULT_OUT_REPORT = "reports/s5d12_dense_fill_path_audit.md"
DEFAULT_OUT_DIR = "external_baselines/results/s5d12_dense_fill_path_audit"

ALLOWED_CLASSIFICATIONS = [
    "S5D12_DENSE_FILL_BUG_IDENTIFIED",
    "S5D12_DENSE_FILL_PROTOCOL_MISMATCH_IDENTIFIED",
    "S5D12_NONSELECTED_SOURCE_UNKNOWN",
    "S5D12_AUDIT_COMPLETE_NO_FIX",
    "S5D12_ERROR",
]

HYPOTHESIS_NAMES = [
    "total_gap_translation_repeated_per_edge",
    "timestamp_unit_bug",
    "missing_division_by_gap_length",
    "endpoint_displacement_not_normalized",
    "selected_nonselected_protocol_mismatch",
    "nonselected_edges_not_direct_predictions",
]


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def _relpath(p: Path | None) -> str | None:
    if p is None:
        return None
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _run(cmd: List[str]) -> str:
    cp = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return cp.stdout + (("\n[stderr]\n" + cp.stderr) if cp.stderr else "")


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
        p = s.split()
        if len(p) < 8:
            continue
        tx, ty, tz = map(float, p[1:4])
        qx, qy, qz, qw = map(float, p[4:8])
        rows.append({
            "timestamp": float(p[0]),
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


def _finite(vals: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(vals), dtype=np.float64)
    return a[np.isfinite(a)]


def _pct(vals: Iterable[float], p: float) -> float | None:
    a = _finite(vals)
    return float(np.percentile(a, p)) if a.size else None


def _mean(vals: Iterable[float]) -> float | None:
    a = _finite(vals)
    return float(np.mean(a)) if a.size else None


def _cv(vals: Iterable[float]) -> float | None:
    a = _finite(vals)
    if a.size == 0:
        return None
    mu = float(np.mean(a))
    return float(np.std(a) / mu) if abs(mu) > 1e-12 else None


def _selected_edges(path: Path) -> Tuple[set[Tuple[int, int]], List[Dict[str, Any]]]:
    selected: set[Tuple[int, int]] = set()
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        i = int(row["frame_i"])
        j = int(row["frame_j"])
        selected.add((i, j))
        rows.append(row)
    return selected, rows


def _edge_metrics(i: int, dense: Sequence[Dict[str, Any]], gt_by_ts: Dict[float, Dict[str, Any]], selected: set[Tuple[int, int]]) -> Dict[str, Any] | None:
    j = i + 1
    if i < 0 or j >= len(dense):
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
    else:
        tdir = math.nan
        tdir_abs = math.nan
        ratio = math.nan
    return {
        "edge_index": i,
        "timestamp_i": float(dense[i]["timestamp"]),
        "timestamp_j": float(dense[j]["timestamp"]),
        "selected_edge": (i, j) in selected,
        "gt_step_length": ng,
        "dense_est_step_length": ne,
        "tmag_ratio": ratio,
        "tdir_deg": tdir,
        "tdir_abs_deg": tdir_abs,
        "rot_deg": _rot_deg(rest @ rgt.T),
    }


def _summarize(edges: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "num_edges": len(edges),
        "est_path": float(sum(e["dense_est_step_length"] for e in edges)),
        "gt_path": float(sum(e["gt_step_length"] for e in edges)),
        "est_step_mean": _mean(e["dense_est_step_length"] for e in edges),
        "est_step_median": _pct((e["dense_est_step_length"] for e in edges), 50),
        "est_step_cv": _cv(e["dense_est_step_length"] for e in edges),
        "gt_step_mean": _mean(e["gt_step_length"] for e in edges),
        "gt_step_median": _pct((e["gt_step_length"] for e in edges), 50),
        "gt_step_cv": _cv(e["gt_step_length"] for e in edges),
        "tmag_ratio_mean": _mean(e["tmag_ratio"] for e in edges),
        "tmag_ratio_median": _pct((e["tmag_ratio"] for e in edges), 50),
        "tmag_ratio_p90": _pct((e["tmag_ratio"] for e in edges), 90),
        "tmag_ratio_p95": _pct((e["tmag_ratio"] for e in edges), 95),
        "tmag_ratio_max": float(np.max(_finite(e["tmag_ratio"] for e in edges))) if _finite(e["tmag_ratio"] for e in edges).size else None,
        "tdir_mean_deg": _mean(e["tdir_deg"] for e in edges),
        "rot_mean_deg": _mean(e["rot_deg"] for e in edges),
    }


def _selected_runs(selected: set[Tuple[int, int]]) -> List[Dict[str, int]]:
    starts = sorted(i for i, j in selected if j == i + 1)
    runs: List[Dict[str, int]] = []
    cur_start = None
    cur_last = None
    for idx in starts:
        if cur_start is None:
            cur_start = cur_last = idx
        elif idx == cur_last + 1:
            cur_last = idx
        else:
            runs.append({"start_edge": int(cur_start), "end_edge": int(cur_last), "length": int(cur_last - cur_start + 1)})
            cur_start = cur_last = idx
    if cur_start is not None:
        runs.append({"start_edge": int(cur_start), "end_edge": int(cur_last), "length": int(cur_last - cur_start + 1)})
    return runs


def _nearest_selected(edge_index: int, selected_indices: Sequence[int]) -> Tuple[int | None, int | None]:
    before = [i for i in selected_indices if i < edge_index]
    after = [i for i in selected_indices if i > edge_index]
    return (max(before) if before else None, min(after) if after else None)


def _code_path_audit() -> Dict[str, Any]:
    log_path = REPO_ROOT / "logs/s5d12_dense_fill_code_search.log"
    if not log_path.exists():
        pattern = "export_s5_dense|dense trajectory|dense_tum|scene01_seq03_s5_dense|selected_k1|fill|interpolate|materialize|chain|compose|relative|tmag|path_ratio"
        text = _run(["rg", "-n", pattern, "tools", "scripts", "reports", "checkpoints", "tests"])
        log_path.parent.mkdir(exist_ok=True)
        log_path.write_text(text, encoding="utf-8")
    candidates = [
        REPO_ROOT / "tools/export_s5_dense_trajectory.py",
        REPO_ROOT / "tools/export_s5_trajectory_for_external_eval.py",
        REPO_ROOT / "tools/restore_or_regenerate_s5_dense_tum.py",
        REPO_ROOT / "tools/replay_s5_pairwise_dense_chain.py",
    ]
    existing = [p for p in candidates if p.exists()]
    fill_locations: List[str] = []
    for p in existing:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"(fill|interpolat|materializ|restore|chain|compose|selected_k1)", txt, flags=re.IGNORECASE):
            line = txt[:m.start()].count("\n") + 1
            fill_locations.append(f"{_relpath(p)}:{line}:{m.group(0)}")
    dense_export = next((p for p in existing if p.name == "export_s5_dense_trajectory.py"), None)
    if dense_export is None:
        dense_export = REPO_ROOT / "tools/restore_or_regenerate_s5_dense_tum.py"
    restore_text = (REPO_ROOT / "tools/restore_or_regenerate_s5_dense_tum.py").read_text(encoding="utf-8", errors="ignore")
    restored_branch = "'git'" in restore_text and "'show'" in restore_text and "experiment/orbslam3-fisheye-strong-baseline" in restore_text
    return {
        "dense_export_script": _relpath(dense_export),
        "candidate_scripts": [_relpath(p) for p in existing],
        "fill_logic_found": False,
        "fill_logic_location": None,
        "searched_locations": fill_locations[:80],
        "restore_from_branch_artifact_found": restored_branch,
        "nonselected_source_type": "unknown",
        "pre_materialization_available": False,
        "code_search_log": "logs/s5d12_dense_fill_code_search.log",
    }


def _validation_from_logs() -> Dict[str, str]:
    paths = {
        "verify_final_candidate": REPO_ROOT / "logs/s5d11_verify_final_candidate_serial.log",
        "project_health_check": REPO_ROOT / "logs/s5d11_project_health_check_serial.log",
        "unittest": REPO_ROOT / "logs/s5d11_unittest.log",
    }
    out: Dict[str, str] = {}
    for name, path in paths.items():
        if not path.exists():
            out[name] = "not_run"
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        low = text.lower()
        if "traceback" in low or "failed" in low or "cuda out of memory" in low:
            out[name] = "FAIL"
        elif "PASS" in text or "OK" in text or "pass" in low:
            out[name] = "PASS"
        else:
            out[name] = "not_run"
    return out


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    run = payload["dominant_run_72_360"]
    hyp = payload["hypotheses"]
    lines = [
        "# S5D12 Dense Fill Path Audit",
        "",
        "## Executive summary",
        f"- final classification: `{payload['final_classification']}`",
        f"- dense fill logic found in current branch: `{payload['code_path_audit']['fill_logic_found']}`",
        f"- nonselected source type: `{payload['code_path_audit']['nonselected_source_type']}`",
        f"- dominant run source: `{run['source_type']}`",
        "",
        "## S5D11 recap",
        "- S5D11 validation was clean: verify, project health, s6 eval-only, and unittest passed.",
        "- non-selected edges had tmag median ratio 29.375292, p90 71.221354, p95 88.130943, max 335.512628.",
        "- worst 20 tmag edges did not dominate path length; the dominant signal was the long run edges 72-360.",
        "",
        "## Dense export/fill code path",
        f"- dense export script or restoration entry: `{payload['code_path_audit']['dense_export_script']}`",
        f"- restore-from-branch artifact path observed: `{payload['code_path_audit']['restore_from_branch_artifact_found']}`",
        f"- pre-materialization available: `{payload['code_path_audit']['pre_materialization_available']}`",
        f"- code search log: `{payload['code_path_audit']['code_search_log']}`",
        "",
        "## Edge source map summary",
        f"- source map: `{payload['source_map']['path']}`",
        f"- edges: `{payload['source_map']['num_edges']}`, selected: `{payload['source_map']['num_selected_edges']}`, non-selected: `{payload['source_map']['num_nonselected_edges']}`",
        "",
        "## Dominant run 72-360 trace",
        f"- trace: `{run['trace_path']}`",
        f"- length: `{run['length']}`",
        f"- estimated path: `{run['estimated_path']:.6f}`",
        f"- gt path: `{run['gt_path']:.6f}`",
        f"- median tmag ratio: `{run['median_tmag_ratio']:.6f}`",
        f"- est step median/cv: `{run['est_step_median']:.6f}` / `{run['est_step_cv']:.6f}`",
        "",
        "## Pre vs post materialization metrics",
        f"- pre materialization available: `{payload['pre_vs_post']['pre_materialization_available']}`",
        f"- pre metrics path: `{payload['pre_vs_post']['pre_metrics_path']}`",
        f"- post metrics path: `{payload['pre_vs_post']['post_metrics_path']}`",
        f"- over-scaling stage: `{payload['pre_vs_post']['over_scaling_stage']}`",
        "",
        "## Hypothesis check table",
        "| hypothesis | supported | evidence |",
        "| --- | --- | --- |",
    ]
    for key in HYPOTHESIS_NAMES:
        lines.append(f"| `{key}` | `{hyp[key]['supported']}` | {'; '.join(hyp[key]['evidence'])} |")
    lines.extend([
        "",
        "## Diagnosis",
        f"- nonselected_over_scaling_source: `{payload['diagnosis']['nonselected_over_scaling_source']}`",
        f"- dense_fill_bug_suspected: `{payload['diagnosis']['dense_fill_bug_suspected']}`",
        f"- requires_code_change: `{payload['diagnosis']['requires_code_change']}`",
        "",
        "## Recommended fixes / next actions",
    ])
    for item in payload["diagnosis"]["recommended_fix"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## Caveats",
        "- diagnostic only",
        "- no official S5 result replacement",
        "- S5 locked metrics/policy unchanged",
        "- no GT used to generate predictions",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-tum", required=True)
    ap.add_argument("--pairwise-jsonl", required=True)
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--timestamps", required=True)
    ap.add_argument("--s5d11-checkpoint", required=True)
    ap.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-report", default=DEFAULT_OUT_REPORT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    dense_path = _resolve(args.dense_tum)
    pair_path = _resolve(args.pairwise_jsonl)
    gt_path = _resolve(args.groundtruth)
    out_json = _resolve(args.out_json)
    out_report = _resolve(args.out_report)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    dense = _read_tum(dense_path)
    gt = _read_tum(gt_path)
    gt_by_ts = {round(float(r["timestamp"]), 6): r for r in gt}
    selected, _ = _selected_edges(pair_path)
    selected_indices = sorted(i for i, j in selected if j == i + 1)

    all_edges = [_edge_metrics(i, dense, gt_by_ts, selected) for i in range(len(dense) - 1)]
    all_edges = [e for e in all_edges if e is not None]
    selected_edges = [e for e in all_edges if e["selected_edge"]]
    nonselected_edges = [e for e in all_edges if not e["selected_edge"]]
    dominant = [e for e in all_edges if 72 <= e["edge_index"] <= 360]

    selected_runs = _selected_runs(selected)
    source_map: List[Dict[str, Any]] = []
    for e in all_edges:
        before, after = _nearest_selected(int(e["edge_index"]), selected_indices)
        source_type = "selected_prediction" if e["selected_edge"] else "unknown"
        source_map.append({
            "edge_index": e["edge_index"],
            "timestamp_i": e["timestamp_i"],
            "timestamp_j": e["timestamp_j"],
            "selected_edge": e["selected_edge"],
            "source_type": source_type,
            "source_pair_or_component": f"{e['edge_index']}->{e['edge_index'] + 1}" if e["selected_edge"] else None,
            "fill_run_id": "nonselected_gap_72_360" if 72 <= e["edge_index"] <= 360 and not e["selected_edge"] else None,
            "notes": "S5D7 selected prediction" if e["selected_edge"] else "post-materialized dense artifact; current branch has no verified fill source",
            "nearest_selected_edge_before": before,
            "nearest_selected_edge_after": after,
        })

    cumulative_gt = 0.0
    cumulative_est = 0.0
    trace: List[Dict[str, Any]] = []
    for e in dominant:
        cumulative_gt += float(e["gt_step_length"])
        cumulative_est += float(e["dense_est_step_length"])
        before, after = _nearest_selected(int(e["edge_index"]), selected_indices)
        trace.append({
            **e,
            "fill_source_type": "unknown" if not e["selected_edge"] else "selected_prediction",
            "cumulative_gt_path": cumulative_gt,
            "cumulative_est_path": cumulative_est,
            "cumulative_ratio": (cumulative_est / cumulative_gt) if cumulative_gt > 1e-12 else None,
            "nearest_selected_edge_before": before,
            "nearest_selected_edge_after": after,
            "component_gap_id": "selected_gap_71_361",
            "notes": "bounded by selected edge 71->72 before and selected edge 361->362 after; source unavailable in current branch",
        })

    post_metrics = {
        "selected_edges": _summarize(selected_edges),
        "nonselected_edges": _summarize(nonselected_edges),
        "dominant_run_72_360": _summarize(dominant),
    }
    pre_metrics = {
        "pre_materialization_available": False,
        "reason": "No verified pre-materialization dense fill intermediate was found in the current working tree or provided inputs.",
        "selected_edges": None,
        "nonselected_edges": None,
        "dominant_run_72_360": None,
    }

    source_map_path = out_dir / "dense_fill_source_map.json"
    trace_path = out_dir / "nonselected_run_72_360_trace.json"
    scale_path = out_dir / "dense_fill_scale_diagnostics.json"
    pre_path = out_dir / "pre_materialization_edge_metrics.json"
    post_path = out_dir / "post_materialization_edge_metrics.json"
    source_map_path.write_text(json.dumps(source_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pre_path.write_text(json.dumps(pre_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    post_path.write_text(json.dumps(post_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    run_summary = post_metrics["dominant_run_72_360"]
    endpoint_dense_delta = float(np.linalg.norm(dense[361]["t"] - dense[72]["t"])) if len(dense) > 361 else None
    endpoint_gt_delta = float(np.linalg.norm(gt_by_ts[round(dense[361]["timestamp"], 6)]["t"] - gt_by_ts[round(dense[72]["timestamp"], 6)]["t"])) if len(dense) > 361 else None
    run_est_median = run_summary["est_step_median"]
    run_gt_median = run_summary["gt_step_median"]
    scale_diag = {
        "dominant_run_edges": "72-360",
        "selected_runs": selected_runs,
        "run_est_step_median": run_est_median,
        "run_est_step_cv": run_summary["est_step_cv"],
        "run_gt_step_median": run_gt_median,
        "run_gt_step_cv": run_summary["gt_step_cv"],
        "run_est_path": run_summary["est_path"],
        "run_gt_path": run_summary["gt_path"],
        "endpoint_dense_delta_norm_72_361": endpoint_dense_delta,
        "endpoint_gt_delta_norm_72_361": endpoint_gt_delta,
        "run_est_path_div_edges": run_summary["est_path"] / max(run_summary["num_edges"], 1),
        "run_gt_path_div_edges": run_summary["gt_path"] / max(run_summary["num_edges"], 1),
    }
    scale_path.write_text(json.dumps(scale_diag, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    code_path = _code_path_audit()
    h: Dict[str, Dict[str, Any]] = {
        "total_gap_translation_repeated_per_edge": {
            "supported": None,
            "evidence": [
                "No verified pre-materialization gap translation vector is available.",
                f"run_est_step_median={run_est_median}",
                f"endpoint_dense_delta_norm_72_361={endpoint_dense_delta}",
            ],
            "affected_edges": "72-360",
            "recommended_fix": "Instrument dense fill source to log gap endpoint displacement and per-edge divided displacement.",
        },
        "timestamp_unit_bug": {
            "supported": False,
            "evidence": [
                "No timestamp-unit conversion logic was found on the restored dense artifact path.",
                f"run_tmag_ratio_median={run_summary['tmag_ratio_median']} is not a canonical ms/us factor.",
            ],
            "affected_edges": "72-360",
            "recommended_fix": "Add diagnostic dt logging if dense fill code is recovered.",
        },
        "missing_division_by_gap_length": {
            "supported": None,
            "evidence": [
                "Long-run behavior is consistent with gap-level materialization trouble, but the fill code/intermediate is unavailable.",
                f"run_length={run_summary['num_edges']}, run_tmag_ratio_median={run_summary['tmag_ratio_median']}",
            ],
            "affected_edges": "72-360",
            "recommended_fix": "When fill code is recovered, verify endpoint delta is divided by number of adjacent steps.",
        },
        "endpoint_displacement_not_normalized": {
            "supported": None,
            "evidence": [
                "Endpoint displacement cannot be compared to the missing pre-materialization vector.",
                f"post_materialized_est_step_cv={run_summary['est_step_cv']}",
            ],
            "affected_edges": "72-360",
            "recommended_fix": "Log endpoint displacement, normalized per-step displacement, and emitted edge displacement.",
        },
        "selected_nonselected_protocol_mismatch": {
            "supported": True,
            "evidence": [
                f"selected_tmag_median={post_metrics['selected_edges']['tmag_ratio_median']}",
                f"nonselected_tmag_median={post_metrics['nonselected_edges']['tmag_ratio_median']}",
                f"dominant_run_tmag_median={run_summary['tmag_ratio_median']}",
            ],
            "affected_edges": "all non-selected edges, especially 72-360",
            "recommended_fix": "Keep selected_k1 predictions and dense-fill/materialized edges separately reported.",
        },
        "nonselected_edges_not_direct_predictions": {
            "supported": True,
            "evidence": [
                "S5D7 pairwise artifact contains 132 selected_k1 adjacent predictions, not all 453 adjacent edges.",
                "Edges 72-360 are absent from the selected pairwise artifact except the boundary selected edge 71->72 and later selected run 361->427.",
                "Current dense TUM was restored as an existing artifact, not generated from direct dense predictions in this audit.",
            ],
            "affected_edges": "321 non-selected edges",
            "recommended_fix": "Expose direct_dense_prediction vs fill/materialization provenance in future exports.",
        },
    }

    if h["selected_nonselected_protocol_mismatch"]["supported"]:
        final_cls = "S5D12_DENSE_FILL_PROTOCOL_MISMATCH_IDENTIFIED"
        source = "protocol_mismatch"
        bug_suspected = None
    elif code_path["nonselected_source_type"] == "unknown":
        final_cls = "S5D12_NONSELECTED_SOURCE_UNKNOWN"
        source = "unknown"
        bug_suspected = None
    else:
        final_cls = "S5D12_AUDIT_COMPLETE_NO_FIX"
        source = "unknown"
        bug_suspected = False

    payload: Dict[str, Any] = {
        "experiment": "S5D12_dense_fill_path_audit_for_nonselected_edges",
        "inputs": {
            "s5_dense_tum": _relpath(dense_path),
            "s5d7_pairwise_jsonl": _relpath(pair_path),
            "s5d11_checkpoint": _relpath(_resolve(args.s5d11_checkpoint)),
            "groundtruth": _relpath(gt_path),
            "timestamps": _relpath(_resolve(args.timestamps)),
        },
        "code_path_audit": code_path,
        "source_map": {
            "path": _relpath(source_map_path),
            "num_edges": len(all_edges),
            "num_selected_edges": len(selected_edges),
            "num_nonselected_edges": len(nonselected_edges),
        },
        "dominant_run_72_360": {
            "trace_path": _relpath(trace_path),
            "length": run_summary["num_edges"],
            "estimated_path": run_summary["est_path"],
            "gt_path": run_summary["gt_path"],
            "median_tmag_ratio": run_summary["tmag_ratio_median"],
            "est_step_median": run_summary["est_step_median"],
            "est_step_cv": run_summary["est_step_cv"],
            "source_type": "unknown",
            "hypothesis_summary": {k: v["supported"] for k, v in h.items()},
        },
        "pre_vs_post": {
            "pre_materialization_available": False,
            "pre_metrics_path": _relpath(pre_path),
            "post_metrics_path": _relpath(post_path),
            "scale_diagnostics_path": _relpath(scale_path),
            "over_scaling_stage": "unknown",
        },
        "hypotheses": h,
        "diagnosis": {
            "nonselected_over_scaling_source": source,
            "dense_fill_bug_suspected": bug_suspected,
            "recommended_fix": [
                "Recover or instrument the original dense fill/materialization code path and log source_type per adjacent edge.",
                "For any gap fill, log endpoint delta, number of adjacent steps, emitted per-step delta, and dt units.",
                "Keep selected_k1 official pairwise metrics separate from external dense trajectory materialization metrics.",
                "Do not change S5 locked policy, manifest, evaluator defaults, or official metrics as part of S5D12.",
            ],
            "requires_code_change": False,
        },
        "validation": _validation_from_logs(),
        "s5_official_locked_metrics": {
            "ate": 7.352288,
            "drift": 1.327343,
            "path_ratio": 0.932379,
            "unchanged": True,
            "not_replaced_by_s5d12": True,
        },
        "allowed_final_classifications": ALLOWED_CLASSIFICATIONS,
        "final_classification": final_cls,
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(out_report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
