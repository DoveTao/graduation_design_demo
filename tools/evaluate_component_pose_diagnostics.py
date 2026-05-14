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
S5_LOCKED = {
    "ate": 7.352288,
    "drift": 1.327343,
    "path_ratio": 0.932379,
    "unchanged": True,
    "not_replaced_by_component_diagnostics": True,
}
SAME_EVAL_FALLBACK_BRANCH = "experiment/orbslam3-fisheye-strong-baseline"
SAME_EVAL_FALLBACK_JSON = "checkpoints/S5D_same_evaluator_comparison_results.json"
S5_TRAJ_FALLBACK = "external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt"
ORB_TRAJ_FALLBACK = "external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt"
ALLOWED_CLASSIFICATIONS = [
    "S5D2_COMPONENT_DIAGNOSTICS_COMPLETE",
    "S5D2_COMPONENT_DIAGNOSTICS_PARTIAL",
    "S5D2_COMPONENT_DIAGNOSTICS_BLOCKED",
    "S5D2_COMPONENT_DIAGNOSTICS_ERROR",
]
DEFAULT_GROUNDTRUTH = "external_baselines/dataset/scene01_seq03/groundtruth_tum.txt"
DEFAULT_TIMESTAMPS = "external_baselines/dataset/scene01_seq03/timestamps.txt"
DEFAULT_S5_TRAJECTORY = "external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt"
DEFAULT_ORB_TRAJECTORY = "external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt"
DEFAULT_OUT_JSON = "checkpoints/S5D2_component_diagnostics.json"
DEFAULT_OUT_REPORT = "reports/s5_orbslam3_component_diagnostics.md"
DEFAULT_OUT_DIR = "external_baselines/results/component_diagnostics"


def _resolve(path: str | Path) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else REPO_ROOT / p


def _git_show_bytes(branch: str, repo_path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{branch}:{repo_path}"], cwd=REPO_ROOT)


def _load_bytes_with_fallback(path: Path, fallback_branch: str, fallback_repo_path: str) -> Tuple[bytes, str]:
    if path.exists():
        return path.read_bytes(), "working_tree"
    return _git_show_bytes(fallback_branch, fallback_repo_path), f"git_show:{fallback_branch}"


def _run_guard() -> None:
    subprocess.run(["bash", "scripts/verify_final_candidate.sh"], cwd=REPO_ROOT, check=True)


def _quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
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


def _read_tum_bytes(content: bytes, source_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in content.decode("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        ts = float(parts[0])
        tx, ty, tz = (float(parts[1]), float(parts[2]), float(parts[3]))
        qx, qy, qz, qw = (float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]))
        rows.append(
            {
                "timestamp": ts,
                "t": np.asarray([tx, ty, tz], dtype=np.float64),
                "q": np.asarray([qx, qy, qz, qw], dtype=np.float64),
                "R_wc": _quat_xyzw_to_rot(qx, qy, qz, qw),
            }
        )
    if not rows:
        raise RuntimeError(f"No valid TUM rows found in {source_name}")
    rows.sort(key=lambda x: x["timestamp"])
    return rows


def _read_timestamps(path: Path) -> List[float]:
    vals = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            vals.append(float(line.split()[0]))
    return vals


def _summary_stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": math.nan,
            "median": math.nan,
            "p90": math.nan,
            "p95": math.nan,
            "max": math.nan,
        }
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _check_rows(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ts = [float(r["timestamp"]) for r in rows]
    qn = [float(np.linalg.norm(r["q"])) for r in rows]
    monotonic = all(ts[i] < ts[i + 1] for i in range(len(ts) - 1))
    duplicates = len(ts) - len(set(ts))
    has_bad = any(not np.isfinite(x) for r in rows for x in [r["timestamp"], *r["t"], *r["q"]])
    return {
        "name": name,
        "num_poses": len(rows),
        "timestamps_strictly_increasing": monotonic,
        "duplicate_timestamps": duplicates,
        "quaternion_norm_summary": _summary_stats(qn),
        "quaternion_norm_close_to_1": bool(np.allclose(qn, 1.0, atol=1.0e-3, rtol=0.0)),
        "has_nan_or_inf": has_bad,
    }


def _relative_motion(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    # T_rel = inverse(T_wc_i) @ T_wc_j
    R_i = a["R_wc"]
    R_j = b["R_wc"]
    t_i = a["t"]
    t_j = b["t"]
    R_rel = R_i.T @ R_j
    t_rel = R_i.T @ (t_j - t_i)
    return R_rel, t_rel


def _rot_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    R_err = R_est @ R_gt.T
    cos_theta = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(cos_theta)))


def _pair_metrics(
    gt_a: Dict[str, Any],
    gt_b: Dict[str, Any],
    est_a: Dict[str, Any],
    est_b: Dict[str, Any],
    min_translation_norm: float,
) -> Dict[str, Any]:
    R_gt_rel, t_gt_rel = _relative_motion(gt_a, gt_b)
    R_est_rel, t_est_rel = _relative_motion(est_a, est_b)
    rot_deg = _rot_error_deg(R_est_rel, R_gt_rel)
    out = {
        "rot_error_deg": rot_deg,
        "tdir_error_deg": None,
        "tdir_cosine": None,
        "tmag_log_error": None,
        "tmag_ratio": None,
        "skip_reason": None,
    }
    gt_norm = float(np.linalg.norm(t_gt_rel))
    est_norm = float(np.linalg.norm(t_est_rel))
    if gt_norm < min_translation_norm:
        out["skip_reason"] = "small_gt_translation"
        return out
    if est_norm < min_translation_norm:
        out["skip_reason"] = "small_est_translation"
        return out
    cosine = float(np.clip(np.dot(t_est_rel / est_norm, t_gt_rel / gt_norm), -1.0, 1.0))
    out["tdir_cosine"] = cosine
    out["tdir_error_deg"] = float(math.degrees(math.acos(cosine)))  # arccos on normalized direction cosine
    out["tmag_ratio"] = est_norm / gt_norm
    out["tmag_log_error"] = abs(math.log(est_norm / gt_norm))  # log(norm(t_est) / norm(t_gt))
    return out


def _rows_by_timestamp(rows: List[Dict[str, Any]]) -> Dict[float, Dict[str, Any]]:
    return {round(float(r["timestamp"]), 6): r for r in rows}


def _adjacent_pairs_from_rows(rows: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    return list(zip(rows[:-1], rows[1:]))


def _build_orb_matched_pair_keys(
    gt_rows: List[Dict[str, Any]],
    orb_rows: List[Dict[str, Any]],
    tolerance: float,
) -> Tuple[List[Tuple[float, float]], int]:
    gt_map = _rows_by_timestamp(gt_rows)
    pair_keys: List[Tuple[float, float]] = []
    missing = 0
    for a, b in _adjacent_pairs_from_rows(orb_rows):
        ka = round(float(a["timestamp"]), 6)
        kb = round(float(b["timestamp"]), 6)
        if ka in gt_map and kb in gt_map and abs(a["timestamp"] - ka) <= tolerance and abs(b["timestamp"] - kb) <= tolerance:
            pair_keys.append((ka, kb))
        else:
            missing += 1
    return pair_keys, missing


def _evaluate_dataset(
    *,
    name: str,
    method: str,
    pair_source: str,
    gt_rows: List[Dict[str, Any]],
    est_rows: List[Dict[str, Any]],
    min_translation_norm: float,
    tolerance: float,
    forced_pair_keys: List[Tuple[float, float]] | None = None,
) -> Dict[str, Any]:
    gt_map = _rows_by_timestamp(gt_rows)
    est_map = _rows_by_timestamp(est_rows)
    notes: List[str] = []
    skipped_counts = {
        "small_gt_translation": 0,
        "small_est_translation": 0,
        "timestamp_missing": 0,
    }
    rot_vals: List[float] = []
    tdir_vals: List[float] = []
    tdir_cos_vals: List[float] = []
    tmag_log_vals: List[float] = []
    tmag_ratio_vals: List[float] = []

    if forced_pair_keys is None:
        candidate_pairs = [(round(float(a["timestamp"]), 6), round(float(b["timestamp"]), 6)) for a, b in _adjacent_pairs_from_rows(est_rows)]
    else:
        candidate_pairs = list(forced_pair_keys)

    valid_pose_keys = sorted({k for pair in candidate_pairs for k in pair if k in est_map and k in gt_map})
    for ka, kb in candidate_pairs:
        gt_a = gt_map.get(ka)
        gt_b = gt_map.get(kb)
        est_a = est_map.get(ka)
        est_b = est_map.get(kb)
        if gt_a is None or gt_b is None or est_a is None or est_b is None:
            skipped_counts["timestamp_missing"] += 1
            continue
        metrics = _pair_metrics(gt_a, gt_b, est_a, est_b, min_translation_norm)
        rot_vals.append(metrics["rot_error_deg"])
        if metrics["skip_reason"] is None:
            tdir_vals.append(float(metrics["tdir_error_deg"]))
            tdir_cos_vals.append(float(metrics["tdir_cosine"]))
            tmag_log_vals.append(float(metrics["tmag_log_error"]))
            tmag_ratio_vals.append(float(metrics["tmag_ratio"]))
        else:
            skipped_counts[metrics["skip_reason"]] += 1

    total_pairs = len(candidate_pairs)
    if total_pairs > 0 and (skipped_counts["timestamp_missing"] / total_pairs) > 0.2:
        notes.append("High timestamp-missing drop ratio; inspect pair-source coverage.")
    if total_pairs > 0 and ((skipped_counts["small_gt_translation"] + skipped_counts["small_est_translation"]) / total_pairs) > 0.2:
        notes.append("High small-translation drop ratio; local tdir/tmag statistics may be sparse.")

    metrics = {
        "rotation": {
            "mean_deg": _summary_stats(rot_vals)["mean"],
            "median_deg": _summary_stats(rot_vals)["median"],
            "p90_deg": _summary_stats(rot_vals)["p90"],
            "p95_deg": _summary_stats(rot_vals)["p95"],
            "max_deg": _summary_stats(rot_vals)["max"],
        },
        "translation_direction": {
            "mean_deg": _summary_stats(tdir_vals)["mean"],
            "median_deg": _summary_stats(tdir_vals)["median"],
            "p90_deg": _summary_stats(tdir_vals)["p90"],
            "p95_deg": _summary_stats(tdir_vals)["p95"],
            "max_deg": _summary_stats(tdir_vals)["max"],
            "mean_cosine": _summary_stats(tdir_cos_vals)["mean"],
            "median_cosine": _summary_stats(tdir_cos_vals)["median"],
        },
        "translation_magnitude": {
            "mean_log_error": _summary_stats(tmag_log_vals)["mean"],
            "median_log_error": _summary_stats(tmag_log_vals)["median"],
            "p90_log_error": _summary_stats(tmag_log_vals)["p90"],
            "p95_log_error": _summary_stats(tmag_log_vals)["p95"],
            "max_log_error": _summary_stats(tmag_log_vals)["max"],
            "mean_ratio": _summary_stats(tmag_ratio_vals)["mean"],
            "median_ratio": _summary_stats(tmag_ratio_vals)["median"],
            "p90_ratio": _summary_stats(tmag_ratio_vals)["p90"],
        },
    }
    coverage = float(len(est_rows) / max(len(gt_rows), 1))
    return {
        "name": name,
        "method": method,
        "pair_source": pair_source,
        "trajectory": None,
        "groundtruth": None,
        "num_poses": len(est_rows) if forced_pair_keys is None else len(valid_pose_keys),
        "num_gt_poses": len(gt_rows),
        "num_pose_pairs": total_pairs,
        "num_valid_rot_pairs": len(rot_vals),
        "num_valid_tdir_pairs": len(tdir_vals),
        "num_valid_tmag_pairs": len(tmag_log_vals),
        "coverage": coverage,
        "timestamp_match_status": "ok" if skipped_counts["timestamp_missing"] == 0 else "partial",
        "metrics": metrics,
        "skipped_pairs": skipped_counts,
        "notes": notes,
    }


def _main_summary_row(payload: Dict[str, Any], output_json: str) -> Dict[str, Any]:
    m = payload["metrics"]
    return {
        "output_json": output_json,
        "num_poses": payload["num_poses"],
        "num_pose_pairs": payload["num_pose_pairs"],
        "coverage": payload["coverage"],
        "rot_mean_deg": m["rotation"]["mean_deg"],
        "rot_median_deg": m["rotation"]["median_deg"],
        "rot_p90_deg": m["rotation"]["p90_deg"],
        "tdir_mean_deg": m["translation_direction"]["mean_deg"],
        "tdir_median_deg": m["translation_direction"]["median_deg"],
        "tdir_p90_deg": m["translation_direction"]["p90_deg"],
        "tdir_mean_cosine": m["translation_direction"]["mean_cosine"],
        "tmag_mean_log_error": m["translation_magnitude"]["mean_log_error"],
        "tmag_median_log_error": m["translation_magnitude"]["median_log_error"],
        "tmag_p90_log_error": m["translation_magnitude"]["p90_log_error"],
        "tmag_mean_ratio": m["translation_magnitude"]["mean_ratio"],
        "tmag_median_ratio": m["translation_magnitude"]["median_ratio"],
    }


def _load_same_eval_metrics() -> Dict[str, Any]:
    local_path = _resolve(SAME_EVAL_FALLBACK_JSON)
    if local_path.exists():
        return json.loads(local_path.read_text(encoding="utf-8"))
    raw = _git_show_bytes(SAME_EVAL_FALLBACK_BRANCH, SAME_EVAL_FALLBACK_JSON)
    return json.loads(raw.decode("utf-8"))


def _fmt(x: Any) -> str:
    try:
        val = float(x)
    except Exception:
        return "nan"
    return f"{val:.6f}" if math.isfinite(val) else "nan"


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    s5 = payload["component_results"]["s5_full"]
    orb = payload["component_results"]["orbslam3_tracked"]
    s5m = payload["component_results"]["s5_on_orb_matched_pairs"]
    orbm = payload["component_results"]["orbslam3_on_matched_pairs"]
    lines = [
        "# S5 vs ORB-SLAM3 Component Diagnostics",
        "",
        "## Executive Summary",
        "",
        f"Final classification: `{payload['final_classification']}`",
        f"Dominant S5 error source: `{payload['interpretation']['dominant_s5_error_source']}`",
        "",
        "## Why component diagnostics are needed",
        "",
        "Trajectory-level ATE, drift, and path_ratio describe overall trajectory behavior, but they do not isolate whether the gap is mainly due to rotation, translation direction, translation magnitude / scale, or cumulative composition drift.",
        "",
        "## Inputs",
        "",
        f"- S5 dense trajectory: `{payload['inputs']['s5_trajectory']}`",
        f"- ORB-SLAM3 trajectory: `{payload['inputs']['orbslam3_trajectory']}`",
        f"- GT: `{payload['inputs']['groundtruth']}`",
        f"- timestamps: `{payload['inputs']['timestamps']}`",
        "",
        "## Protocol",
        "",
        "- TUM format: `timestamp tx ty tz qx qy qz qw`",
        "- T_wc convention: translation is world-frame camera position and quaternion is interpreted as `qx qy qz qw`",
        "- relative transform definition: `inverse(T_i) @ T_j`",
        "- rotation error: geodesic angle of `R_est_rel @ inverse(R_gt_rel)`",
        "- translation direction error: `arccos(dot(normalize(t_est), normalize(t_gt)))`",
        "- translation magnitude log error: `abs(log(norm(t_est) / norm(t_gt)))`",
        f"- min translation threshold: `{payload['protocol']['min_translation_norm']}`",
        f"- timestamp matching tolerance: `{payload['protocol']['timestamp_tolerance']}`",
        "",
        "## Results Table 1: S5 Full Sequence",
        "",
        "| num poses | num pairs | rot mean | rot median | rot p90 | tdir mean | tdir median | tdir p90 | tmag mean log err | tmag median log err | tmag p90 log err | tmag mean ratio | tmag median ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {_fmt(s5['num_poses'])} | {_fmt(s5['num_pose_pairs'])} | {_fmt(s5['rot_mean_deg'])} | {_fmt(s5['rot_median_deg'])} | {_fmt(s5['rot_p90_deg'])} | {_fmt(s5['tdir_mean_deg'])} | {_fmt(s5['tdir_median_deg'])} | {_fmt(s5['tdir_p90_deg'])} | {_fmt(s5['tmag_mean_log_error'])} | {_fmt(s5['tmag_median_log_error'])} | {_fmt(s5['tmag_p90_log_error'])} | {_fmt(s5['tmag_mean_ratio'])} | {_fmt(s5['tmag_median_ratio'])} |",
        "",
        "## Results Table 2: ORB-SLAM3 Tracked Sequence",
        "",
        "| num poses | num pairs | rot mean | rot median | rot p90 | tdir mean | tdir median | tdir p90 | tmag mean log err | tmag median log err | tmag p90 log err | tmag mean ratio | tmag median ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {_fmt(orb['num_poses'])} | {_fmt(orb['num_pose_pairs'])} | {_fmt(orb['rot_mean_deg'])} | {_fmt(orb['rot_median_deg'])} | {_fmt(orb['rot_p90_deg'])} | {_fmt(orb['tdir_mean_deg'])} | {_fmt(orb['tdir_median_deg'])} | {_fmt(orb['tdir_p90_deg'])} | {_fmt(orb['tmag_mean_log_error'])} | {_fmt(orb['tmag_median_log_error'])} | {_fmt(orb['tmag_p90_log_error'])} | {_fmt(orb['tmag_mean_ratio'])} | {_fmt(orb['tmag_median_ratio'])} |",
        "",
        "## Results Table 3: Matched Pair Comparison",
        "",
        "| method | num pairs | rot mean | rot median | rot p90 | tdir mean | tdir median | tdir p90 | tdir mean cosine | tmag mean log err | tmag median log err | tmag p90 log err | tmag mean ratio | tmag median ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| S5-on-ORB-matched-pairs | {_fmt(s5m['num_pose_pairs'])} | {_fmt(s5m['rot_mean_deg'])} | {_fmt(s5m['rot_median_deg'])} | {_fmt(s5m['rot_p90_deg'])} | {_fmt(s5m['tdir_mean_deg'])} | {_fmt(s5m['tdir_median_deg'])} | {_fmt(s5m['tdir_p90_deg'])} | {_fmt(s5m['tdir_mean_cosine'])} | {_fmt(s5m['tmag_mean_log_error'])} | {_fmt(s5m['tmag_median_log_error'])} | {_fmt(s5m['tmag_p90_log_error'])} | {_fmt(s5m['tmag_mean_ratio'])} | {_fmt(s5m['tmag_median_ratio'])} |",
        f"| ORB-SLAM3-on-same-matched-pairs | {_fmt(orbm['num_pose_pairs'])} | {_fmt(orbm['rot_mean_deg'])} | {_fmt(orbm['rot_median_deg'])} | {_fmt(orbm['rot_p90_deg'])} | {_fmt(orbm['tdir_mean_deg'])} | {_fmt(orbm['tdir_median_deg'])} | {_fmt(orbm['tdir_p90_deg'])} | {_fmt(orbm['tdir_mean_cosine'])} | {_fmt(orbm['tmag_mean_log_error'])} | {_fmt(orbm['tmag_median_log_error'])} | {_fmt(orbm['tmag_p90_log_error'])} | {_fmt(orbm['tmag_mean_ratio'])} | {_fmt(orbm['tmag_median_ratio'])} |",
        "",
        "## Interpretation",
        "",
    ]
    for line in payload["interpretation"]["evidence"]:
        lines.append(f"- {line}")
    lines += [
        "",
        "## Recommendations",
        "",
    ]
    for rec in payload["interpretation"]["recommendations"]:
        lines.append(f"- {rec}")
    lines += [
        "",
        "## Caveats",
        "",
        "- This diagnostic does not replace the official S5 locked result.",
        "- S5 official locked metrics/policy were not changed.",
        "- ORB-SLAM3 uses raw fisheye cam0 with fitted KB8 compatibility calibration.",
        "- S5 and ORB-SLAM3 are not same-input-protocol.",
        "- ORB-SLAM3 has partial coverage.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate rotation / tdir / tmag component diagnostics on TUM trajectories.")
    ap.add_argument("--groundtruth", default=DEFAULT_GROUNDTRUTH)
    ap.add_argument("--s5-trajectory", default=DEFAULT_S5_TRAJECTORY)
    ap.add_argument("--orb-trajectory", default=DEFAULT_ORB_TRAJECTORY)
    ap.add_argument("--timestamps", default=DEFAULT_TIMESTAMPS)
    ap.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-report", default=DEFAULT_OUT_REPORT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--timestamp-tolerance", type=float, default=1e-6)
    ap.add_argument("--min-translation-norm", type=float, default=1e-8)
    args = ap.parse_args()

    _run_guard()

    gt_path = _resolve(args.groundtruth)
    s5_path = _resolve(args.s5_trajectory)
    orb_path = _resolve(args.orb_trajectory)
    timestamps_path = _resolve(args.timestamps)
    out_json = _resolve(args.out_json)
    out_report = _resolve(args.out_report)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_rows = _read_tum_bytes(gt_path.read_bytes(), str(gt_path))
    s5_bytes, s5_source = _load_bytes_with_fallback(s5_path, SAME_EVAL_FALLBACK_BRANCH, S5_TRAJ_FALLBACK)
    orb_bytes, orb_source = _load_bytes_with_fallback(orb_path, SAME_EVAL_FALLBACK_BRANCH, ORB_TRAJ_FALLBACK)
    s5_rows = _read_tum_bytes(s5_bytes, f"{s5_path} ({s5_source})")
    orb_rows = _read_tum_bytes(orb_bytes, f"{orb_path} ({orb_source})")
    timestamp_vals = _read_timestamps(timestamps_path)
    same_eval = _load_same_eval_metrics()

    gt_check = _check_rows("groundtruth", gt_rows)
    s5_check = _check_rows("s5", s5_rows)
    orb_check = _check_rows("orbslam3", orb_rows)

    orb_pair_keys, orb_missing = _build_orb_matched_pair_keys(gt_rows, orb_rows, float(args.timestamp_tolerance))

    s5_full_payload = _evaluate_dataset(
        name="s5_full",
        method="S5",
        pair_source="all_adjacent_pairs",
        gt_rows=gt_rows,
        est_rows=s5_rows,
        min_translation_norm=float(args.min_translation_norm),
        tolerance=float(args.timestamp_tolerance),
    )
    orb_tracked_payload = _evaluate_dataset(
        name="orbslam3_tracked",
        method="ORB-SLAM3",
        pair_source="orbslam3_adjacent_tracked_pairs",
        gt_rows=gt_rows,
        est_rows=orb_rows,
        min_translation_norm=float(args.min_translation_norm),
        tolerance=float(args.timestamp_tolerance),
    )
    s5_matched_payload = _evaluate_dataset(
        name="s5_on_orb_matched_pairs",
        method="S5",
        pair_source="orbslam3_tracked_adjacent_pairs",
        gt_rows=gt_rows,
        est_rows=s5_rows,
        min_translation_norm=float(args.min_translation_norm),
        tolerance=float(args.timestamp_tolerance),
        forced_pair_keys=orb_pair_keys,
    )
    orb_matched_payload = _evaluate_dataset(
        name="orbslam3_on_matched_pairs",
        method="ORB-SLAM3",
        pair_source="orbslam3_tracked_adjacent_pairs",
        gt_rows=gt_rows,
        est_rows=orb_rows,
        min_translation_norm=float(args.min_translation_norm),
        tolerance=float(args.timestamp_tolerance),
        forced_pair_keys=orb_pair_keys,
    )

    file_map = {
        "s5_full": out_dir / "s5_full_component_diagnostics.json",
        "orbslam3_tracked": out_dir / "orbslam3_tracked_component_diagnostics.json",
        "s5_on_orb_matched_pairs": out_dir / "s5_on_orb_matched_component_diagnostics.json",
        "orbslam3_on_matched_pairs": out_dir / "orbslam3_on_matched_component_diagnostics.json",
    }
    for payload, key in (
        (s5_full_payload, "s5_full"),
        (orb_tracked_payload, "orbslam3_tracked"),
        (s5_matched_payload, "s5_on_orb_matched_pairs"),
        (orb_matched_payload, "orbslam3_on_matched_pairs"),
    ):
        payload["trajectory"] = str(s5_path if "s5" in key else orb_path)
        payload["groundtruth"] = str(gt_path)
        file_map[key].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    dominant = "unknown"
    evidence: List[str] = []
    recommendations: List[str] = []
    if (
        math.isfinite(s5_matched_payload["metrics"]["translation_magnitude"]["mean_ratio"])
        and s5_matched_payload["metrics"]["translation_magnitude"]["mean_ratio"] > 1.5
    ):
        dominant = "tmag"
        evidence.append("On ORB-matched pairs, S5 mean tmag ratio is substantially above 1, which points to local translation magnitude over-scaling.")
        recommendations.extend(
            [
                "prioritize translation magnitude calibration under the dense-export / same-evaluator scope",
                "revisit short-window pose consistency with explicit scale-aware objectives",
            ]
        )
    if (
        math.isfinite(s5_matched_payload["metrics"]["translation_direction"]["mean_deg"])
        and s5_matched_payload["metrics"]["translation_direction"]["mean_deg"] > 35.0
    ):
        dominant = "mixed" if dominant != "unknown" else "tdir"
        evidence.append("S5 translation-direction error on matched pairs is materially elevated, so local tdir quality also contributes.")
        recommendations.append("add stronger tdir supervision or a direction-focused loss on matched local motions")
    if (
        math.isfinite(s5_matched_payload["metrics"]["rotation"]["mean_deg"])
        and s5_matched_payload["metrics"]["rotation"]["mean_deg"] > 15.0
    ):
        dominant = "mixed" if dominant != "unknown" else "rotation"
        evidence.append("S5 rotation error on matched pairs is high enough that local orientation quality cannot be ignored.")
        recommendations.append("consider stronger SO(3) geodesic rotation supervision and short-window rotational consistency")
    if dominant == "unknown":
        dominant = "mixed"
        evidence.append("No single component dominates cleanly; rotation, tdir, and tmag need to be read together.")
    if orb_tracked_payload["coverage"] < 1.0:
        evidence.append("ORB-SLAM3 remains a high-precision partial-tracking baseline: local component quality is measured only on tracked segments.")
        recommendations.append("treat ORB-SLAM3 as a successful-segment teacher or distillation target rather than as a full-coverage replacement")
    recommendations.extend(
        [
            "keep the dense export convention audit in scope before interpreting full-sequence scale behavior too aggressively",
            "investigate geometry-aware S5 input encoding or short-window consistency modules for local pose quality",
        ]
    )

    existing_s5 = same_eval["s5"]
    existing_orb = same_eval["orbslam3"]
    main_payload = {
        "experiment": "S5D2_component_diagnostics_rot_tdir_tmag",
        "inputs": {
            "groundtruth": str(gt_path.relative_to(REPO_ROOT) if gt_path.is_relative_to(REPO_ROOT) else gt_path),
            "timestamps": str(timestamps_path.relative_to(REPO_ROOT) if timestamps_path.is_relative_to(REPO_ROOT) else timestamps_path),
            "s5_trajectory": str(s5_path),
            "orbslam3_trajectory": str(orb_path),
        },
        "protocol": {
            "tum_pose_convention": "T_wc with translation as world-frame camera position and quaternion qx qy qz qw",
            "relative_transform_definition": "inverse(T_i) @ T_j",
            "timestamp_tolerance": float(args.timestamp_tolerance),
            "min_translation_norm": float(args.min_translation_norm),
        },
        "pose_input_checks": {
            "groundtruth": gt_check,
            "s5": s5_check,
            "orbslam3": orb_check,
            "num_input_timestamps": len(timestamp_vals),
            "orb_pair_keys_missing_gt": orb_missing,
            "s5_source_mode": s5_source,
            "orb_source_mode": orb_source,
        },
        "s5_official_locked_metrics": S5_LOCKED,
        "existing_external_trajectory_metrics": {
            "s5": {
                "coverage": existing_s5["coverage"],
                "num_est_poses": existing_s5["num_est_poses"],
                "num_input_frames": existing_s5["num_input_frames"],
                "none": existing_s5["alignment_results"]["none"],
                "se3": existing_s5["alignment_results"]["se3"],
                "sim3": existing_s5["alignment_results"]["sim3"],
            },
            "orbslam3": {
                "coverage": existing_orb["coverage"],
                "num_est_poses": existing_orb["num_est_poses"],
                "num_input_frames": existing_orb["num_input_frames"],
                "none": existing_orb["alignment_results"]["none"],
                "se3": existing_orb["alignment_results"]["se3"],
                "sim3": existing_orb["alignment_results"]["sim3"],
            },
        },
        "component_results": {
            "s5_full": _main_summary_row(s5_full_payload, str(file_map["s5_full"])),
            "orbslam3_tracked": _main_summary_row(orb_tracked_payload, str(file_map["orbslam3_tracked"])),
            "s5_on_orb_matched_pairs": _main_summary_row(s5_matched_payload, str(file_map["s5_on_orb_matched_pairs"])),
            "orbslam3_on_matched_pairs": _main_summary_row(orb_matched_payload, str(file_map["orbslam3_on_matched_pairs"])),
        },
        "interpretation": {
            "dominant_s5_error_source": dominant,
            "evidence": evidence,
            "recommendations": recommendations[:5],
        },
        "fairness_notes": {
            "same_gt": True,
            "same_component_metric_definitions": True,
            "same_input_protocol": False,
            "input_protocol_note": "S5 uses project panorama/S5 representation; ORB-SLAM3 uses raw fisheye cam0.",
            "orbslam3_calibration_caveat": "fitted KB8 compatibility approximation, not native factory KB8",
        },
        "final_classification": "S5D2_COMPONENT_DIAGNOSTICS_COMPLETE",
        "notes": [
            "Component diagnostics use raw local relative motions from exported trajectories.",
            "These diagnostics do not replace the official S5 locked result.",
        ],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(main_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(out_report, main_payload)
    print(json.dumps(main_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
