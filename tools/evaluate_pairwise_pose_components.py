#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRED = REPO_ROOT / "external_baselines" / "results" / "pano_orb_vo" / "scene01_seq03_est_tum.txt"
DEFAULT_GT = REPO_ROOT / "external_baselines" / "dataset" / "scene01_seq03" / "groundtruth_tum.txt"
DEFAULT_PAIR_DIAG = REPO_ROOT / "external_baselines" / "results" / "pano_orb_vo" / "pair_diagnostics.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "checkpoints" / "SB1b_pano_orb_vo_component_diagnostics.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "pano_orb_vo_component_diagnostics.md"
DEFAULT_SB1_JSON = REPO_ROOT / "checkpoints" / "SB1_pano_orb_vo_baseline_results.json"
DEFAULT_SB2B_JSON = REPO_ROOT / "checkpoints" / "SB2b_s5_export_compatibility_audit.json"
DEFAULT_MANIFEST = REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json"

S5_S13_REFERENCE = {
    "method": "S5_S13_reference",
    "source": "S13 diagnostics reference only; not same external full-coverage evaluator",
    "num_valid_pairs": None,
    "rot_mean_deg": 20.715353,
    "rot_median_deg": 20.433789,
    "rot_p90_deg": 21.371863,
    "rot_max_deg": 21.717051,
    "tdir_mean_deg": 72.349483,
    "tdir_median_deg": 102.517940,
    "tdir_p90_deg": 110.918120,
    "tdir_max_deg": 141.610521,
    "tdir_mean_cosine": 0.226300,
    "tmag_mean_log_error": 0.899289,
    "tmag_median_log_error": 0.793141,
    "tmag_p90_log_error": 1.742507,
    "tmag_max_log_error": 1.985574,
}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw: str | Path | None, default: Path) -> Path:
    if raw is None:
        return default
    path = Path(str(raw))
    return path if path.is_absolute() else (REPO_ROOT / path)


def _quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1.0e-12)
    x, y, z, w = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _read_tum(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        ts = float(parts[0])
        t = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
        qx, qy, qz, qw = map(float, parts[4:8])
        R = _quat_xyzw_to_rot(qx, qy, qz, qw)
        rows.append({"timestamp": ts, "t": t, "R": R})
    rows.sort(key=lambda x: x["timestamp"])
    return rows


def _match_rows(gt_rows: List[Dict[str, Any]], pred_rows: List[Dict[str, Any]], tol: float = 1.0e-3) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    matches: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    j = 0
    for gt in gt_rows:
        while j < len(pred_rows) and pred_rows[j]["timestamp"] < gt["timestamp"] - tol:
            j += 1
        best = None
        for cand_idx in (j - 1, j, j + 1):
            if cand_idx < 0 or cand_idx >= len(pred_rows):
                continue
            pred = pred_rows[cand_idx]
            dt = abs(pred["timestamp"] - gt["timestamp"])
            if dt <= tol and (best is None or dt < best[0]):
                best = (dt, pred)
        if best is not None:
            matches.append((gt, best[1]))
    return matches


def _pose_inv(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Rt = R.T
    return Rt, -(Rt @ t)


def _pose_compose(R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return R1 @ R2, R1 @ t2 + t1


def _pose_relative(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    R_inv, t_inv = _pose_inv(a["R"], a["t"])
    return _pose_compose(R_inv, t_inv, b["R"], b["t"])


def _angle_deg_from_rot(R: np.ndarray) -> float:
    trace = float(np.trace(R))
    val = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    return float(math.degrees(math.acos(val)))


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def _fmt(v: Any, digits: int = 6) -> str:
    try:
        x = float(v)
    except Exception:
        return str(v)
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def _evaluate_components(pred_path: Path, gt_path: Path) -> Dict[str, Any]:
    pred_rows = _read_tum(pred_path)
    gt_rows = _read_tum(gt_path)
    matches = _match_rows(gt_rows, pred_rows)
    if len(matches) < 2:
        raise RuntimeError("Need at least 2 matched poses for pairwise component diagnostics.")

    rot_errors: List[float] = []
    tdir_errors: List[float] = []
    tdir_cosines: List[float] = []
    tmag_log_errors: List[float] = []
    invalid_tdir = 0
    invalid_tmag = 0
    num_pairs = 0

    for idx in range(len(matches) - 1):
        gt_a, pred_a = matches[idx]
        gt_b, pred_b = matches[idx + 1]
        R_gt, t_gt = _pose_relative(gt_a, gt_b)
        R_pred, t_pred = _pose_relative(pred_a, pred_b)
        num_pairs += 1

        R_err = R_pred.T @ R_gt
        rot_errors.append(_angle_deg_from_rot(R_err))

        t_gt_norm = float(np.linalg.norm(t_gt))
        t_pred_norm = float(np.linalg.norm(t_pred))
        if t_gt_norm <= 1.0e-12 or t_pred_norm <= 1.0e-12:
            invalid_tdir += 1
        else:
            tdir_gt = t_gt / t_gt_norm
            tdir_pred = t_pred / t_pred_norm
            cos = float(np.clip(np.dot(tdir_pred, tdir_gt), -1.0, 1.0))
            tdir_cosines.append(cos)
            tdir_errors.append(float(math.degrees(math.acos(cos))))

        if t_gt_norm <= 1.0e-12 or t_pred_norm <= 1.0e-12:
            invalid_tmag += 1
        else:
            tmag_log_errors.append(float(abs(math.log((t_pred_norm + 1.0e-12) / (t_gt_norm + 1.0e-12)))))

    rot_q = _quantiles(rot_errors)
    tdir_q = _quantiles(tdir_errors)
    tmag_q = _quantiles(tmag_log_errors)
    tdir_mean_cosine = float(np.mean(np.asarray(tdir_cosines, dtype=np.float64))) if tdir_cosines else float("nan")

    return {
        "method": "Pano-ORB-VO",
        "source": "same-evaluator full-coverage TUM pairwise adjacent-pose diagnostic",
        "num_pairs": int(num_pairs),
        "num_valid_pairs": int(num_pairs),
        "invalid_tdir_pairs": int(invalid_tdir),
        "invalid_tmag_pairs": int(invalid_tmag),
        "rot_mean_deg": rot_q["mean"],
        "rot_median_deg": rot_q["median"],
        "rot_p90_deg": rot_q["p90"],
        "rot_max_deg": rot_q["max"],
        "tdir_mean_deg": tdir_q["mean"],
        "tdir_median_deg": tdir_q["median"],
        "tdir_p90_deg": tdir_q["p90"],
        "tdir_max_deg": tdir_q["max"],
        "tdir_mean_cosine": tdir_mean_cosine,
        "tmag_mean_log_error": tmag_q["mean"],
        "tmag_median_log_error": tmag_q["median"],
        "tmag_p90_log_error": tmag_q["p90"],
        "tmag_max_log_error": tmag_q["max"],
    }


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    pano = payload["pano_orb_vo_component_metrics"]
    traj = payload["pano_orb_vo_trajectory_metrics"]
    s5 = payload["s5_locked_metrics"]
    s5_ref = payload["s5_reference_component_metrics"]
    track = payload["pano_orb_vo_tracking_diagnostics"]

    cols = [
        "method",
        "source",
        "num_valid_pairs",
        "rot_mean_deg",
        "rot_median_deg",
        "rot_p90_deg",
        "rot_max_deg",
        "tdir_mean_deg",
        "tdir_median_deg",
        "tdir_p90_deg",
        "tdir_max_deg",
        "tdir_mean_cosine",
        "tmag_mean_log_error",
        "tmag_median_log_error",
        "tmag_p90_log_error",
        "tmag_max_log_error",
    ]
    rows = [pano, s5_ref]
    table = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            v = row.get(col, "")
            vals.append(_fmt(v) if isinstance(v, float) else str(v))
        table.append("| " + " | ".join(vals) + " |")

    lines = [
        "# Pano-ORB-VO Component Diagnostics",
        "",
        "## Scope",
        "This is a component-level diagnostic and does not change the baseline trajectory or reselect any candidate.",
        "If an S5 reference row is included, it comes from S13 diagnostics and is not a same-evaluator full-coverage row.",
        "",
        "## Trajectory-level reminder",
        f"- Pano-ORB-VO none/se3/sim3 ATE = `{_fmt(traj['none']['ATE'])}` / `{_fmt(traj['se3']['ATE'])}` / `{_fmt(traj['sim3']['ATE'])}`",
        f"- Pano-ORB-VO none/se3/sim3 drift = `{_fmt(traj['none']['drift'])}` / `{_fmt(traj['se3']['drift'])}` / `{_fmt(traj['sim3']['drift'])}`",
        f"- Pano-ORB-VO path_ratio = `{_fmt(traj['none']['path_ratio'])}`",
        f"- S5 locked metrics: ATE=`{_fmt(s5['ATE'])}`, drift=`{_fmt(s5['drift'])}`, path_ratio=`{_fmt(s5['path_ratio'])}`",
        f"- SB2b caveat: {payload['sb2b_s5_export_caveat']}",
        "",
        "## Pairwise component metrics",
        *table,
        "",
        "## Feature tracking diagnostics",
        f"- tracking_success_rate = `{_fmt(track['tracking_success_rate'])}`",
        f"- mean_inliers = `{_fmt(track['mean_inliers'])}`",
        f"- median_inliers = `{_fmt(track['median_inliers'])}`",
        f"- selected_view_yaw distribution = `{track['selected_view_yaw_distribution']}`",
        "",
        "## Interpretation",
        payload["interpretation"],
        "",
        "## Caveats",
        "- panorama-derived virtual pinhole baseline",
        "- not original fisheye baseline",
        "- component metrics are pairwise diagnostics",
        "- tmag diagnostics are scale-policy-dependent",
        "- trajectory-level and pairwise metrics answer different questions",
        "- S5 authoritative metrics remain locked clean evaluator metrics",
        "- S5 external TUM export is sparse diagnostic only per SB2b",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pairwise rotation / translation-direction / tmag diagnostics from trajectory TUM files.")
    p.add_argument("--pred", default=str(DEFAULT_PRED), help="Predicted TUM trajectory path.")
    p.add_argument("--gt", default=str(DEFAULT_GT), help="Groundtruth TUM trajectory path.")
    p.add_argument("--pair-diagnostics", default=str(DEFAULT_PAIR_DIAG), help="Optional pair diagnostics json path.")
    p.add_argument("--sb1-json", default=str(DEFAULT_SB1_JSON), help="SB1 baseline result json path.")
    p.add_argument("--sb2b-json", default=str(DEFAULT_SB2B_JSON), help="SB2b compatibility audit json path.")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Final manifest path.")
    p.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="Output json path.")
    p.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD), help="Output markdown path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pred_path = _resolve_path(args.pred, DEFAULT_PRED)
    gt_path = _resolve_path(args.gt, DEFAULT_GT)
    pair_diag_path = _resolve_path(args.pair_diagnostics, DEFAULT_PAIR_DIAG)
    sb1_path = _resolve_path(args.sb1_json, DEFAULT_SB1_JSON)
    sb2b_path = _resolve_path(args.sb2b_json, DEFAULT_SB2B_JSON)
    manifest_path = _resolve_path(args.manifest, DEFAULT_MANIFEST)
    output_json = _resolve_path(args.output_json, DEFAULT_OUTPUT_JSON)
    output_md = _resolve_path(args.output_md, DEFAULT_OUTPUT_MD)

    pano_metrics = _evaluate_components(pred_path, gt_path)
    sb1 = _read_json(sb1_path)
    sb2b = _read_json(sb2b_path)
    manifest = _read_json(manifest_path)
    pair_diag = json.loads(pair_diag_path.read_text(encoding="utf-8")) if pair_diag_path.is_file() else []

    yaw_counter = Counter(float(row.get("selected_view_yaw", float("nan"))) for row in pair_diag)
    yaw_dist = {str(int(k) if float(k).is_integer() else k): int(v) for k, v in sorted(yaw_counter.items(), key=lambda kv: kv[0])}
    tracking = {
        "tracking_success_rate": float(sb1["tracking_success_rate"]),
        "mean_inliers": float(sb1["mean_inliers"]),
        "median_inliers": float(sb1["median_inliers"]),
        "selected_view_yaw_distribution": yaw_dist,
        "num_pairs": int(len(pair_diag)),
    }

    interpretation = (
        "Pano-ORB-VO provides a strong protocol-compatible classical geometry baseline at the pairwise component level. "
        "Its rotation and translation-direction diagnostics can be compared against the S5 S13 reference only qualitatively, "
        "because the S5 reference row is not a same-evaluator full-coverage trajectory export. "
        "The Pano-ORB-VO tmag diagnostics remain scale-policy-dependent, so trajectory-level path_ratio should still be used to judge scale stability. "
        "This report does not support a blanket superiority claim for S5, and it does not change the final-candidate status."
    )

    payload = {
        "name": "SB1b_pano_orb_vo_component_diagnostics",
        "pano_orb_vo_component_metrics": pano_metrics,
        "pano_orb_vo_trajectory_metrics": sb1["main_evaluation"],
        "pano_orb_vo_tracking_diagnostics": tracking,
        "s5_reference_component_metrics": S5_S13_REFERENCE,
        "s5_locked_metrics": dict(manifest["final_metrics"]),
        "sb2b_s5_export_caveat": "S5 external TUM export is sparse diagnostic only and not used for full-coverage comparison.",
        "interpretation": interpretation,
        "caveats": [
            "panorama-derived virtual pinhole baseline",
            "not original fisheye baseline",
            "component metrics are pairwise diagnostics",
            "tmag diagnostics are scale-policy-dependent",
            "trajectory-level and pairwise metrics answer different questions",
            "S5 authoritative metrics remain locked clean evaluator metrics",
            "S5 external TUM export is sparse diagnostic only per SB2b",
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_report(output_md, payload)
    print(f"[SB1b] json={output_json}")
    print(f"[SB1b] md={output_md}")
    print("[SB1b] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
