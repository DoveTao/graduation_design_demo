#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

OFFICIAL_LOCKED = {"ATE": 7.352288, "drift": 1.327343, "path_ratio": 0.932379}
DENSE_EXTERNAL_REFERENCE = {
    "ATE_none": 21.681522,
    "ATE_se3": 8.231469,
    "ATE_sim3": 4.079123,
    "path_ratio": 2.777268,
}


def _resolve(path: str | Path) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else REPO_ROOT / p


def _run_guard() -> None:
    subprocess.run(["bash", "scripts/verify_final_candidate.sh"], cwd=REPO_ROOT, check=True)


def _quat_xyzw_to_rot(q: Iterable[float]) -> np.ndarray:
    q = np.asarray(list(q), dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1.0e-12)
    x, y, z, w = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_wxyz_to_rot(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = np.asarray(list(q), dtype=np.float64)
    return _quat_xyzw_to_rot([x, y, z, w])


def _read_tum(path: Path, quat_order: str = "xyzw") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        vals = [float(x) for x in parts[:8]]
        q = vals[4:8]
        R = _quat_xyzw_to_rot(q) if quat_order == "xyzw" else _quat_wxyz_to_rot(q)
        rows.append(
            {
                "timestamp": vals[0],
                "t": np.asarray(vals[1:4], dtype=np.float64),
                "q_raw": np.asarray(q, dtype=np.float64),
                "R": R,
            }
        )
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def _discover_dense(search_root: Path) -> Path | None:
    candidates = sorted(search_root.glob("**/*scene01_seq03*est_tum*.txt"))
    s5 = [p for p in candidates if "s5" in str(p).lower()]
    if s5:
        dense = [p for p in s5 if "dense" in str(p).lower()]
        return dense[0] if dense else s5[0]
    exact = sorted(search_root.glob("**/scene01_seq03_est_tum.txt"))
    return exact[0] if exact else (candidates[0] if candidates else None)


def _match(gt: List[Dict[str, Any]], est: List[Dict[str, Any]], tol: float = 1.0e-6) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    gt_by_ts = {round(float(r["timestamp"]), 6): r for r in gt}
    pairs = []
    for row in est:
        key = round(float(row["timestamp"]), 6)
        g = gt_by_ts.get(key)
        if g is not None and abs(float(g["timestamp"]) - float(row["timestamp"])) <= tol:
            pairs.append((g, row))
    return pairs


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())


def _steps(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(points[1:] - points[:-1], axis=1)


def _q(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": math.nan, "median": math.nan, "p90": math.nan, "max": math.nan}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def _umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    dim, n = src.shape
    mean_src = src.mean(axis=1, keepdims=True)
    mean_dst = dst.mean(axis=1, keepdims=True)
    src_c = src - mean_src
    dst_c = dst - mean_dst
    cov = (dst_c @ src_c.T) / float(n)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    scale = 1.0
    if with_scale:
        var_src = np.sum(src_c * src_c) / float(n)
        scale = float(np.trace(np.diag(D) @ S) / max(var_src, 1.0e-12))
    t = (mean_dst - scale * R @ mean_src).reshape(dim)
    return scale, R, t


def _ate_metrics(gt_pts: np.ndarray, est_pts: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, with_scale in (("none", None), ("se3", False), ("sim3", True)):
        if name == "none":
            aligned = est_pts
        else:
            s, R, t = _umeyama(est_pts.T, gt_pts.T, with_scale=bool(with_scale))
            aligned = (s * (R @ est_pts.T)).T + t.reshape(1, 3)
        out[f"ATE_{name}"] = float(np.sqrt(np.mean(np.sum((aligned - gt_pts) ** 2, axis=1))))
    return out


def _rot_angle(R: np.ndarray) -> float:
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(c)))


def _rotation_continuity(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    vals = []
    for a, b in zip(rows[:-1], rows[1:]):
        vals.append(_rot_angle(np.asarray(a["R"]).T @ np.asarray(b["R"])))
    return _q(vals)


def _variant_rows(rows: List[Dict[str, Any]], variant: str) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        t = np.asarray(r["t"], dtype=np.float64)
        R = np.asarray(r["R"], dtype=np.float64)
        if variant == "original_xyzw":
            t2, R2 = t, R
        elif variant == "invert_pose":
            R2 = R.T
            t2 = -R.T @ t
        elif variant == "reverse_positions":
            # Position-order diagnostic only; timestamps stay sorted for matching.
            continue
        elif variant == "axis_flip_x":
            F = np.diag([-1.0, 1.0, 1.0])
            t2, R2 = F @ t, F @ R @ F
        elif variant == "axis_flip_y":
            F = np.diag([1.0, -1.0, 1.0])
            t2, R2 = F @ t, F @ R @ F
        elif variant == "axis_flip_z":
            F = np.diag([1.0, 1.0, -1.0])
            t2, R2 = F @ t, F @ R @ F
        else:
            raise ValueError(variant)
        rec = dict(r)
        rec["t"] = t2
        rec["R"] = R2
        out.append(rec)
    return out


def _variant_reverse(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ts = [r["timestamp"] for r in rows]
    rev = list(reversed(rows))
    out = []
    for ts_val, src in zip(ts, rev):
        rec = dict(src)
        rec["timestamp"] = ts_val
        out.append(rec)
    return out


def _evaluate_variant(name: str, gt_rows: List[Dict[str, Any]], est_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    matches = _match(gt_rows, est_rows)
    gt_pts = np.asarray([g["t"] for g, _ in matches], dtype=np.float64)
    est_pts = np.asarray([e["t"] for _, e in matches], dtype=np.float64)
    if len(matches) < 2:
        return {"variant": name, "status": "insufficient_matches", "num_matches": len(matches)}
    gt_len = _path_length(gt_pts)
    est_len = _path_length(est_pts)
    metrics = _ate_metrics(gt_pts, est_pts)
    metrics.update(
        {
            "variant": name,
            "status": "ok",
            "num_matches": int(len(matches)),
            "path_ratio": float(est_len / max(gt_len, 1.0e-12)),
            "path_length": float(est_len),
            "rotation_continuity": _rotation_continuity(est_rows),
            "translation_step_distribution": _q(_steps(est_pts)),
        }
    )
    return metrics


def _dense_summary(gt_rows: List[Dict[str, Any]], est_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    matches = _match(gt_rows, est_rows)
    ts = [float(r["timestamp"]) for r in est_rows]
    q_norms = [float(np.linalg.norm(r["q_raw"])) for r in est_rows]
    pts = np.asarray([r["t"] for r in est_rows], dtype=np.float64)
    step = _steps(pts)
    return {
        "num_pred_poses": int(len(est_rows)),
        "num_gt_poses": int(len(gt_rows)),
        "matched_poses": int(len(matches)),
        "timestamp_match_rate": float(len(matches) / max(len(gt_rows), 1)),
        "timestamps_strictly_increasing": bool(all(ts[i] < ts[i + 1] for i in range(len(ts) - 1))),
        "duplicate_timestamps": int(len(ts) - len(set(ts))),
        "has_nan_or_inf": bool(any(not np.isfinite(x) for r in est_rows for x in [r["timestamp"], *r["t"], *r["q_raw"]])),
        "quaternion_norm": _q(q_norms),
        "quaternion_norm_close_to_1": bool(np.allclose(q_norms, 1.0, atol=1.0e-3, rtol=0.0)),
        "position_step_distribution": _q(step),
        "position_jump_outlier_count": int(np.sum(step > (np.median(step) + 10.0 * max(np.std(step), 1.0e-12)))) if len(step) else 0,
    }


def _path_audit(gt_rows: List[Dict[str, Any]], est_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    matches = _match(gt_rows, est_rows)
    gt_pts = np.asarray([g["t"] for g, _ in matches], dtype=np.float64)
    est_pts = np.asarray([e["t"] for _, e in matches], dtype=np.float64)
    pred_steps = _steps(est_pts)
    gt_steps = _steps(gt_pts)
    log_err = np.log(np.maximum(pred_steps, 1.0e-12)) - np.log(np.maximum(gt_steps, 1.0e-12))
    pred_len = _path_length(est_pts)
    gt_len = _path_length(gt_pts)
    ratio = float(pred_len / max(gt_len, 1.0e-12))
    mismatch = abs(ratio - OFFICIAL_LOCKED["path_ratio"]) > 0.25
    return {
        "pred_path_length_dense": float(pred_len),
        "gt_path_length_matched": float(gt_len),
        "gt_path_length_full": float(_path_length(np.asarray([g["t"] for g in gt_rows], dtype=np.float64))),
        "path_ratio_dense": ratio,
        "official_locked_path_ratio": OFFICIAL_LOCKED["path_ratio"],
        "dense_external_reference_path_ratio": DENSE_EXTERNAL_REFERENCE["path_ratio"],
        "path_ratio_mismatch": bool(mismatch),
        "scale_factor_to_match_gt": float(gt_len / max(pred_len, 1.0e-12)),
        "suspected_scale_over_application": bool(ratio > 1.5 and np.median(pred_steps) > 1.5 * max(float(np.median(gt_steps)), 1.0e-12)),
        "pred_tmag_distribution": _q(pred_steps),
        "gt_tmag_distribution": _q(gt_steps),
        "log_tmag_error_distribution": _q(log_err),
    }


def _official_scope() -> Dict[str, Any]:
    eval_text = (_resolve("tools/eval_clean_policy.py")).read_text(encoding="utf-8")
    export_text = (_resolve("tools/export_s5_dense_trajectory.py")).read_text(encoding="utf-8")
    return {
        "official_eval_scope": "test split odometry evaluation via eval_odometry_sequence over selected odometry chains and configured eval k-list",
        "dense_export_scope": "scene01/seq03 split=None dense adjacent k=1 all-frame stream exported to TUM for external evaluator",
        "same_pair_source": "unknown",
        "path_ratio_definition_match": False,
        "evidence": {
            "official_uses_split_test": 'split="test"' in eval_text,
            "official_uses_eval_odometry_sequence": "eval_odometry_sequence" in eval_text,
            "dense_uses_split_none": "split=None" in export_text,
            "dense_uses_k1": "k_list=(1,)" in export_text,
            "dense_writes_camera_center": "_camera_center_from_T_c0_np" in export_text,
        },
    }


def _classify(path_audit: Dict[str, Any], variants: List[Dict[str, Any]], scope: Dict[str, Any], found: bool) -> Tuple[str, str]:
    if not found:
        return "S5D2-DENSE-EXPORT-UNAVAILABLE", "Regenerate or locate the dense S5 TUM export before comparing with external baselines."
    original = next(v for v in variants if v["variant"] == "original_xyzw")
    better = [
        v
        for v in variants
        if v.get("status") == "ok"
        and v["variant"] != "original_xyzw"
        and v.get("ATE_se3", 1e9) < 0.75 * original.get("ATE_se3", 1e9)
        and abs(v.get("path_ratio", 0.0) - OFFICIAL_LOCKED["path_ratio"]) < abs(original.get("path_ratio", 0.0) - OFFICIAL_LOCKED["path_ratio"])
    ]
    if better:
        return "S5D2-CONVENTION-MISMATCH-SUSPECTED", "Fix the dense export pose convention before model experiments."
    if bool(scope["path_ratio_definition_match"]) is False:
        return "S5D2-EVAL-SCOPE-MISMATCH", "Do not compare official S5 locked path_ratio directly with dense external path_ratio; audit scope-specific scale behavior separately."
    if path_audit["suspected_scale_over_application"]:
        return "S5D2-SCALE-COMPOSITION-BUG-SUSPECTED", "Inspect translation magnitude application and pose composition in the dense exporter."
    return "S5D2-INCONCLUSIVE", "Inspect dense export generation and official odometry debug chains manually."


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    dense = payload["dense_export_summary"]
    pa = payload["path_length_audit"]
    tmag = payload["tmag_distribution"]
    rows = [
        "| variant | ATE_none | ATE_se3 | ATE_sim3 | path_ratio | notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for v in payload["convention_variants"]:
        notes = "ok" if v.get("status") == "ok" else v.get("status", "")
        rows.append(
            "| {variant} | {none} | {se3} | {sim3} | {ratio} | {notes} |".format(
                variant=v["variant"],
                none=_fmt(v.get("ATE_none")),
                se3=_fmt(v.get("ATE_se3")),
                sim3=_fmt(v.get("ATE_sim3")),
                ratio=_fmt(v.get("path_ratio")),
                notes=notes,
            )
        )
    lines = [
        "# S5D2 Dense Export Convention Audit",
        "",
        "## Scope",
        "",
        "This report audits the S5 dense external trajectory export. It does not change S5 policy, locked metrics, split, or official eval convention, and it does not start a new algorithm experiment.",
        "",
        "## Reference Metrics",
        "",
        f"S5 official locked: ATE = {OFFICIAL_LOCKED['ATE']:.6f}, drift = {OFFICIAL_LOCKED['drift']:.6f}, official path_ratio = {OFFICIAL_LOCKED['path_ratio']:.6f}.",
        "",
        f"S5 dense external: none ATE = {DENSE_EXTERNAL_REFERENCE['ATE_none']:.6f}, SE(3) ATE = {DENSE_EXTERNAL_REFERENCE['ATE_se3']:.6f}, Sim(3) ATE = {DENSE_EXTERNAL_REFERENCE['ATE_sim3']:.6f}, path_ratio = {DENSE_EXTERNAL_REFERENCE['path_ratio']:.6f}.",
        "",
        "## Dense Export Summary",
        "",
        f"- trajectory path: `{payload['dense_export_path']}`",
        f"- dense_export_found: `{payload['dense_export_found']}`",
        f"- num poses: `{dense.get('num_pred_poses')}`",
        f"- matched poses: `{dense.get('matched_poses')}`",
        f"- timestamp status: strictly increasing `{dense.get('timestamps_strictly_increasing')}`, match rate `{_fmt(dense.get('timestamp_match_rate'))}`",
        f"- quaternion status: close to unit norm `{dense.get('quaternion_norm_close_to_1')}`",
        "",
        "## Path Ratio Audit",
        "",
        f"- pred path length: `{_fmt(pa.get('pred_path_length_dense'))}`",
        f"- GT path length: `{_fmt(pa.get('gt_path_length_matched'))}`",
        f"- dense path_ratio mismatch: `{pa.get('path_ratio_mismatch')}`",
        f"- dense path_ratio: `{_fmt(pa.get('path_ratio_dense'))}`",
        f"- scale_factor_to_match_gt: `{_fmt(pa.get('scale_factor_to_match_gt'))}`",
        f"- suspected_scale_over_application: `{pa.get('suspected_scale_over_application')}`",
        f"- pred tmag mean/median/p90/max: `{_fmt(tmag['pred_tmag']['mean'])}` / `{_fmt(tmag['pred_tmag']['median'])}` / `{_fmt(tmag['pred_tmag']['p90'])}` / `{_fmt(tmag['pred_tmag']['max'])}`",
        f"- GT tmag mean/median/p90/max: `{_fmt(tmag['gt_tmag']['mean'])}` / `{_fmt(tmag['gt_tmag']['median'])}` / `{_fmt(tmag['gt_tmag']['p90'])}` / `{_fmt(tmag['gt_tmag']['max'])}`",
        f"- log tmag error mean/median/p90/max: `{_fmt(tmag['log_tmag_error']['mean'])}` / `{_fmt(tmag['log_tmag_error']['median'])}` / `{_fmt(tmag['log_tmag_error']['p90'])}` / `{_fmt(tmag['log_tmag_error']['max'])}`",
        "",
        "## Convention Variant Audit",
        "",
        *rows,
        "",
        "## Official vs Dense Scope",
        "",
        f"- official evaluator scope: {payload['official_vs_dense_scope']['official_eval_scope']}",
        f"- dense export scope: {payload['official_vs_dense_scope']['dense_export_scope']}",
        f"- same_pair_source: `{payload['official_vs_dense_scope']['same_pair_source']}`",
        f"- path_ratio_definition_match: `{payload['official_vs_dense_scope']['path_ratio_definition_match']}`",
        "",
        "## Final Classification",
        "",
        f"`{payload['final_classification']}`",
        "",
        "## Recommendation",
        "",
        payload["recommendation"],
        "",
        "## Caveats",
        "",
        "- no S5 policy change",
        "- no final candidate change",
        "- no new clean-result claim",
        "- ORB-SLAM3 remains external baseline only",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "nan"
    return f"{x:.6f}" if math.isfinite(x) else "nan"


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit S5 dense external export convention against official evaluator scope.")
    ap.add_argument("--s5-policy", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--search-root", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()

    _run_guard()
    policy_path = _resolve(args.s5_policy)
    manifest_path = _resolve(args.manifest)
    gt_path = _resolve(args.gt)
    search_root = _resolve(args.search_root)
    out_json = _resolve(args.output_json)
    out_md = _resolve(args.output_md)

    dense_path = _discover_dense(search_root)
    dense_found = dense_path is not None and dense_path.exists()
    gt_rows = _read_tum(gt_path, quat_order="xyzw") if gt_path.exists() else []
    est_rows = _read_tum(dense_path, quat_order="xyzw") if dense_found else []

    dense_summary = _dense_summary(gt_rows, est_rows) if dense_found else {
        "num_pred_poses": 0,
        "num_gt_poses": len(gt_rows),
        "matched_poses": 0,
        "timestamp_match_rate": 0.0,
    }
    path_audit = _path_audit(gt_rows, est_rows) if dense_found else {}
    est_wxyz = _read_tum(dense_path, quat_order="wxyz") if dense_found else []
    variants = []
    if dense_found:
        for name in ["original_xyzw", "invert_pose", "axis_flip_x", "axis_flip_y", "axis_flip_z"]:
            variants.append(_evaluate_variant(name, gt_rows, _variant_rows(est_rows, name)))
        variants.append(_evaluate_variant("quaternion_order_wxyz", gt_rows, est_wxyz))
        variants.append(_evaluate_variant("reverse_position_order", gt_rows, _variant_reverse(est_rows)))
    scope = _official_scope()
    classification, recommendation = _classify(path_audit, variants, scope, dense_found)
    tmag = {
        "pred_tmag": path_audit.get("pred_tmag_distribution", {}),
        "gt_tmag": path_audit.get("gt_tmag_distribution", {}),
        "log_tmag_error": path_audit.get("log_tmag_error_distribution", {}),
    }
    payload = {
        "experiment_name": "S5D2_dense_export_convention_audit",
        "dense_export_path": str(dense_path.relative_to(REPO_ROOT) if dense_found and dense_path.is_relative_to(REPO_ROOT) else dense_path),
        "dense_export_found": bool(dense_found),
        "num_pred_poses": int(dense_summary.get("num_pred_poses", 0)),
        "num_gt_poses": int(dense_summary.get("num_gt_poses", 0)),
        "matched_poses": int(dense_summary.get("matched_poses", 0)),
        "official_locked_metrics": OFFICIAL_LOCKED,
        "dense_external_metrics": DENSE_EXTERNAL_REFERENCE,
        "dense_export_summary": dense_summary,
        "path_length_audit": path_audit,
        "tmag_distribution": tmag,
        "convention_variants": variants,
        "official_vs_dense_scope": scope,
        "final_classification": classification,
        "recommendation": recommendation,
        "policy_path": str(policy_path.relative_to(REPO_ROOT) if policy_path.is_relative_to(REPO_ROOT) else policy_path),
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT) if manifest_path.is_relative_to(REPO_ROOT) else manifest_path),
        "s5_policy_unchanged": True,
        "s5_locked_metrics_unchanged": True,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(out_md, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
