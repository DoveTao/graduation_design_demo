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
    "not_replaced_by_s5d3": True,
}
ALLOWED_CLASSIFICATIONS = [
    "S5D3_COMPLETE_VALIDATION_CLEAN",
    "S5D3_COMPLETE_WITH_VALIDATION_BLOCKER",
    "S5D3_DIAGNOSTICS_PARTIAL",
    "S5D3_BLOCKED",
    "S5D3_ERROR",
]
DEFAULT_GROUNDTRUTH = "external_baselines/dataset/scene01_seq03/groundtruth_tum.txt"
DEFAULT_S5_TRAJECTORY = "external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt"
DEFAULT_ORB_TRAJECTORY = "external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt"
DEFAULT_TIMESTAMPS = "external_baselines/dataset/scene01_seq03/timestamps.txt"
DEFAULT_OUT_JSON = "checkpoints/S5D3_validation_and_alignment_aware_component_audit.json"
DEFAULT_OUT_REPORT = "reports/s5d3_validation_and_alignment_aware_component_audit.md"
DEFAULT_OUT_DIR = "external_baselines/results/component_diagnostics_s5d3"
SAME_EVAL_FALLBACK_BRANCH = "experiment/orbslam3-fisheye-strong-baseline"
S5_TRAJ_FALLBACK = "external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt"
ORB_TRAJ_FALLBACK = "external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt"


def _umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    if src.shape != dst.shape:
        raise ValueError("src and dst must have the same shape")
    dim, n = src.shape
    mean_src = src.mean(axis=1, keepdims=True)
    mean_dst = dst.mean(axis=1, keepdims=True)
    src_centered = src - mean_src
    dst_centered = dst - mean_dst
    cov = (dst_centered @ src_centered.T) / float(n)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_src = np.sum(src_centered * src_centered) / float(n)
        scale = float(np.trace(np.diag(D) @ S) / max(var_src, 1.0e-12))
    else:
        scale = 1.0
    t = (mean_dst - scale * R @ mean_src).reshape(dim)
    return scale, R, t


def _resolve(p: str | Path) -> Path:
    q = Path(str(p))
    return q if q.is_absolute() else REPO_ROOT / q


def _run(cmd: str, log_path: str) -> Tuple[bool, str]:
    lp = _resolve(log_path)
    lp.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(["bash", "-lc", f"{cmd} 2>&1 | tee {lp}"], cwd=REPO_ROOT)
    return proc.returncode == 0, str(lp.relative_to(REPO_ROOT))


def _git_show_bytes(branch: str, repo_path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{branch}:{repo_path}"], cwd=REPO_ROOT)


def _load_bytes_with_fallback(path: Path, fallback_repo_path: str) -> Tuple[bytes, str]:
    if path.exists():
        return path.read_bytes(), "working_tree"
    try:
        return _git_show_bytes(SAME_EVAL_FALLBACK_BRANCH, fallback_repo_path), f"git_show:{SAME_EVAL_FALLBACK_BRANCH}"
    except Exception:
        alt = REPO_ROOT / "external_baselines" / "results" / "s5" / "scene01_seq03_est_tum.txt"
        if "orbslam3" not in fallback_repo_path and alt.exists():
            return alt.read_bytes(), "working_tree_alt"
        raise


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


def _read_tum(path: Path) -> List[Dict[str, Any]]:
    return _read_tum_bytes(path.read_bytes(), str(path))


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
        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
        rows.append(
            {
                "timestamp": ts,
                "t": np.asarray([tx, ty, tz], dtype=np.float64),
                "R_wc": _quat_xyzw_to_rot(qx, qy, qz, qw),
            }
        )
    if not rows:
        raise RuntimeError(f"No valid TUM rows in {source_name}")
    rows.sort(key=lambda x: x["timestamp"])
    return rows


def _read_timestamps(path: Path) -> List[float]:
    vals: List[float] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            vals.append(float(line.split()[0]))
    return vals


def _summary(v: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(v), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": math.nan, "median": math.nan, "p90": math.nan}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def _rows_by_ts(rows: List[Dict[str, Any]]) -> Dict[float, Dict[str, Any]]:
    return {round(float(r["timestamp"]), 6): r for r in rows}


def _relative(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    R_i, R_j = a["R_wc"], b["R_wc"]
    t_i, t_j = a["t"], b["t"]
    R_rel = R_i.T @ R_j
    t_rel = R_i.T @ (t_j - t_i)
    return R_rel, t_rel


def _rot_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    R_err = R_est @ R_gt.T
    c = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(c)))


def _align_rows(est_rows: List[Dict[str, Any]], gt_rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if mode == "none":
        return [{"timestamp": r["timestamp"], "t": r["t"].copy(), "R_wc": r["R_wc"].copy()} for r in est_rows]
    gt_map = _rows_by_ts(gt_rows)
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for er in est_rows:
        k = round(float(er["timestamp"]), 6)
        if k in gt_map:
            pairs.append((er, gt_map[k]))
    if len(pairs) < 3:
        raise RuntimeError(f"Need >=3 matched poses for {mode} alignment")
    est_pts = np.asarray([e["t"] for e, _ in pairs], dtype=np.float64)
    gt_pts = np.asarray([g["t"] for _, g in pairs], dtype=np.float64)
    with_scale = mode == "sim3"
    scale, R_a, t_a = _umeyama(est_pts.T, gt_pts.T, with_scale=with_scale)
    out: List[Dict[str, Any]] = []
    for r in est_rows:
        t_new = scale * (R_a @ r["t"]) + t_a
        R_new = R_a @ r["R_wc"]
        out.append({"timestamp": r["timestamp"], "t": t_new, "R_wc": R_new})
    return out


def _orb_adjacent_pair_keys(orb_rows: List[Dict[str, Any]], gt_rows: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    gt_map = _rows_by_ts(gt_rows)
    out: List[Tuple[float, float]] = []
    for a, b in zip(orb_rows[:-1], orb_rows[1:]):
        ka, kb = round(float(a["timestamp"]), 6), round(float(b["timestamp"]), 6)
        if ka in gt_map and kb in gt_map:
            out.append((ka, kb))
    return out


def _evaluate_set(
    name: str,
    gt_rows: List[Dict[str, Any]],
    est_rows: List[Dict[str, Any]],
    pair_keys: List[Tuple[float, float]],
    thresholds: List[Dict[str, Any]],
) -> Dict[str, Any]:
    gt_map = _rows_by_ts(gt_rows)
    est_map = _rows_by_ts(est_rows)
    sweeps: List[Dict[str, Any]] = []
    for th in thresholds:
        rot_vals: List[float] = []
        tdir_vals: List[float] = []
        tdir_cos_vals: List[float] = []
        tmag_log_vals: List[float] = []
        tmag_ratio_vals: List[float] = []
        valid_tdir = 0
        for ka, kb in pair_keys:
            ga, gb = gt_map.get(ka), gt_map.get(kb)
            ea, eb = est_map.get(ka), est_map.get(kb)
            if ga is None or gb is None or ea is None or eb is None:
                continue
            Rg, tg = _relative(ga, gb)
            Re, te = _relative(ea, eb)
            rot_vals.append(_rot_deg(Re, Rg))
            ng, ne = float(np.linalg.norm(tg)), float(np.linalg.norm(te))
            if ng < th["value"] or ne < th["value"]:
                continue
            valid_tdir += 1
            cosv = float(np.clip(np.dot(te / ne, tg / ng), -1.0, 1.0))
            tdir_cos_vals.append(cosv)
            tdir_vals.append(float(math.degrees(math.acos(cosv))))
            ratio = ne / ng
            tmag_ratio_vals.append(ratio)
            tmag_log_vals.append(abs(math.log(ratio)))
        rot_s, tdir_s, tmag_s = _summary(rot_vals), _summary(tdir_vals), _summary(tmag_log_vals)
        ratio_s = _summary(tmag_ratio_vals)
        sweeps.append(
            {
                "threshold_name": th["name"],
                "threshold_type": th["type"],
                "threshold_value": th["value"],
                "num_pose_pairs": len(pair_keys),
                "num_valid_tdir_pairs": valid_tdir,
                "valid_tdir_pair_ratio": float(valid_tdir / max(len(pair_keys), 1)),
                "rot_mean_deg": rot_s["mean"],
                "rot_median_deg": rot_s["median"],
                "rot_p90_deg": rot_s["p90"],
                "tdir_mean_deg": tdir_s["mean"],
                "tdir_median_deg": tdir_s["median"],
                "tdir_p90_deg": tdir_s["p90"],
                "tdir_mean_cosine": _summary(tdir_cos_vals)["mean"],
                "tmag_mean_log_error": tmag_s["mean"],
                "tmag_median_log_error": tmag_s["median"],
                "tmag_p90_log_error": tmag_s["p90"],
                "tmag_mean_ratio": ratio_s["mean"],
                "tmag_median_ratio": ratio_s["median"],
            }
        )
    return {"set_name": name, "threshold_sweep": sweeps}


def _threshold_plan(gt_rows: List[Dict[str, Any]], abs_th: List[float], rel_th: List[float]) -> Tuple[float, List[Dict[str, Any]]]:
    gt_steps = [float(np.linalg.norm(_relative(a, b)[1])) for a, b in zip(gt_rows[:-1], gt_rows[1:])]
    med = float(np.median(np.asarray(gt_steps, dtype=np.float64))) if gt_steps else 0.0
    out: List[Dict[str, Any]] = []
    for v in abs_th:
        out.append({"name": f"abs_{v:g}", "type": "absolute", "value": float(v)})
    for r in rel_th:
        out.append({"name": f"median_gt_step_x_{r:.2f}", "type": "relative", "value": float(med * r)})
    return med, out


def _extract_unittest_count(log_path: Path) -> int | None:
    txt = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    for line in txt.splitlines():
        if "Ran " in line and " tests" in line:
            try:
                return int(line.split("Ran ", 1)[1].split(" tests", 1)[0].strip())
            except Exception:
                return None
    return None


def _git_clean() -> bool:
    p = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return p.stdout.strip() == ""


def _pick_recommended(sweep: List[Dict[str, Any]]) -> Dict[str, Any]:
    key = "median_gt_step_x_0.05"
    for row in sweep:
        if row["threshold_name"] == key:
            return row
    return sweep[-1]


def _fmt(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "nan"
    return f"{v:.6f}" if math.isfinite(v) else "nan"


def _write_md(path: Path, payload: Dict[str, Any]) -> None:
    rs = payload["alignment_aware_diagnostics"]["recommended_threshold_summary"]
    lines = [
        "# S5D3 Validation and Alignment-aware Component Audit",
        "",
        "## Executive summary",
        "",
        f"Final classification: `{payload['final_classification']}`",
        "",
        "## Validation status",
        "",
        f"- verify_final_candidate: `{'PASS' if payload['validation']['verify_final_candidate']['passed'] else 'FAIL'}`",
        f"- project_health_check: `{'PASS' if payload['validation']['project_health_check']['passed'] else 'FAIL'}`",
        f"- unittest: `{'PASS' if payload['validation']['unittest']['passed'] else 'FAIL'}`",
        f"- s6_final_clean_candidate_lockdown_audit.py --eval-only: `{'PASS' if payload['validation']['s6_lockdown_eval_only']['passed'] else 'FAIL'}`",
        f"- root cause summary: {payload['validation']['s6_lockdown_eval_only']['root_cause_summary']}",
        f"- branch clean: `{payload['validation']['branch_clean']}`",
        "",
        "## Why S5D3 is needed",
        "",
        "S5D2 raw tdir is near 90-100 deg for both S5 and ORB-SLAM3. ORB-SLAM3 can still have low aligned ATE, so raw local tdir can be confounded by gauge/alignment and tiny-step pairs.",
        "",
        "## Protocol",
        "",
        "- TUM pose convention: `timestamp tx ty tz qx qy qz qw`",
        "- T_wc convention: translation is world-frame camera position, quaternion order is `qx qy qz qw`",
        "- relative transform: `inverse(T_i) @ T_j`",
        "- alignment modes: `none / se3 / sim3`",
        "- threshold sweep: absolute and `median_gt_step * {0.01,0.05,0.10}`",
        "- matched-pair fairness: same ORB tracked adjacent timestamp pairs; no ORB interpolation",
        "- no official S5 metric replacement",
        "",
        "## Results",
        "",
        f"Recommended threshold: `median_gt_step * 0.05 = {_fmt(payload['alignment_aware_diagnostics']['recommended_threshold_value'])}`",
        "",
    ]
    for align in ("none", "se3", "sim3"):
        lines += [f"### alignment={align}", "", "| set | num valid pairs | valid ratio | rot mean/median/p90 | tdir mean/median/p90 | tdir mean cosine | tmag mean/median/p90 log | tmag mean/median ratio |", "|---|---:|---:|---|---|---:|---|---|"]
        for s in ("s5_full", "orbslam3_tracked", "s5_on_orb_matched_pairs", "orbslam3_on_matched_pairs"):
            r = rs[align][s]
            lines.append(
                f"| {s} | {_fmt(r['num_valid_tdir_pairs'])} | {_fmt(r['valid_tdir_pair_ratio'])} | {_fmt(r['rot_mean_deg'])}/{_fmt(r['rot_median_deg'])}/{_fmt(r['rot_p90_deg'])} | {_fmt(r['tdir_mean_deg'])}/{_fmt(r['tdir_median_deg'])}/{_fmt(r['tdir_p90_deg'])} | {_fmt(r['tdir_mean_cosine'])} | {_fmt(r['tmag_mean_log_error'])}/{_fmt(r['tmag_median_log_error'])}/{_fmt(r['tmag_p90_log_error'])} | {_fmt(r['tmag_mean_ratio'])}/{_fmt(r['tmag_median_ratio'])} |"
            )
        lines += [""]
    lines += [
        "## Interpretation",
        "",
        f"- S5 rotation error confirmed: `{payload['summary_findings']['s5_rotation_error_confirmed']}`",
        f"- S5 tmag over-scaling confirmed: `{payload['summary_findings']['s5_tmag_overscale_confirmed']}`",
        f"- tdir reliable after threshold/alignment: `{payload['summary_findings']['tdir_reliable_after_threshold_alignment']}`",
        f"- dominant S5 error source: `{payload['summary_findings']['dominant_s5_error_source']}`",
        "",
        "## Recommendations",
        "",
    ]
    for r in payload["summary_findings"]["recommendations"]:
        lines.append(f"- {r}")
    lines += [
        "",
        "## Caveats",
        "",
        "- S5D3 does not replace the official S5 locked result.",
        "- S5 official locked metrics/policy were not changed.",
        "- ORB-SLAM3 remains an external strong baseline.",
        "- ORB-SLAM3 has partial coverage.",
        "- ORB-SLAM3 uses fitted KB8 compatibility calibration.",
        "- same_input_protocol=false.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="S5D3 validation and alignment-aware component diagnostics")
    ap.add_argument("--groundtruth", default=DEFAULT_GROUNDTRUTH)
    ap.add_argument("--s5-trajectory", default=DEFAULT_S5_TRAJECTORY)
    ap.add_argument("--orb-trajectory", default=DEFAULT_ORB_TRAJECTORY)
    ap.add_argument("--timestamps", default=DEFAULT_TIMESTAMPS)
    ap.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-report", default=DEFAULT_OUT_REPORT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--timestamp-tolerance", type=float, default=1e-6)
    ap.add_argument("--thresholds", default="1e-8,1e-6,1e-5,1e-4,1e-3,1e-2")
    ap.add_argument("--relative-thresholds", default="0.01,0.05,0.10")
    args = ap.parse_args()

    gt_path = _resolve(args.groundtruth)
    s5_path = _resolve(args.s5_trajectory)
    orb_path = _resolve(args.orb_trajectory)
    ts_path = _resolve(args.timestamps)
    out_json = _resolve(args.out_json)
    out_report = _resolve(args.out_report)
    out_dir = _resolve(args.out_dir)

    if not (gt_path.exists() and ts_path.exists()):
        payload = {
            "experiment": "S5D3_validation_and_alignment_aware_component_audit",
            "final_classification": "S5D3_BLOCKED",
            "validation": {"validation_clean": False},
            "error": "Missing required input file(s).",
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0

    v_pass, v_log = _run("bash scripts/verify_final_candidate.sh", "logs/s5d3_verify_final_candidate.log")
    h_pass, h_log = _run("bash scripts/project_health_check.sh", "logs/s5d3_project_health_check.log")
    u_pass, u_log = _run("/home/dovetao/miniconda3/envs/pytorch/bin/python -m unittest discover -s tests -q", "logs/s5d3_unittest.log")
    s6_pass, s6_log = _run("/home/dovetao/miniconda3/envs/pytorch/bin/python tools/s6_final_clean_candidate_lockdown_audit.py --eval-only", "logs/s5d3_s6_lockdown_eval_only.log")

    gt_rows = _read_tum(gt_path)
    s5_bytes, s5_source = _load_bytes_with_fallback(s5_path, S5_TRAJ_FALLBACK)
    orb_bytes, orb_source = _load_bytes_with_fallback(orb_path, ORB_TRAJ_FALLBACK)
    s5_rows = _read_tum_bytes(s5_bytes, f"{s5_path} ({s5_source})")
    orb_rows = _read_tum_bytes(orb_bytes, f"{orb_path} ({orb_source})")
    _ = _read_timestamps(ts_path)

    abs_th = [float(x) for x in args.thresholds.split(",") if x.strip()]
    rel_th = [float(x) for x in args.relative_thresholds.split(",") if x.strip()]
    median_gt_step, threshold_rows = _threshold_plan(gt_rows, abs_th, rel_th)

    gt_keys = [round(float(r["timestamp"]), 6) for r in gt_rows]
    s5_full_pairs = list(zip(gt_keys[:-1], gt_keys[1:]))
    orb_keys = [round(float(r["timestamp"]), 6) for r in orb_rows]
    orb_tracked_pairs = list(zip(orb_keys[:-1], orb_keys[1:]))
    matched_pairs = _orb_adjacent_pair_keys(orb_rows, gt_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Dict[str, Any]]] = {a: {} for a in ("none", "se3", "sim3")}
    result_files: Dict[str, Dict[str, str]] = {
        "s5_full": {},
        "orbslam3_tracked": {},
        "s5_on_orb_matched_pairs": {},
        "orbslam3_on_matched_pairs": {},
    }

    for align in ("none", "se3", "sim3"):
        s5_aligned = _align_rows(s5_rows, gt_rows, align)
        orb_aligned = _align_rows(orb_rows, gt_rows, align)
        packs = {
            "s5_full": _evaluate_set("s5_full", gt_rows, s5_aligned, s5_full_pairs, threshold_rows),
            "orbslam3_tracked": _evaluate_set("orbslam3_tracked", gt_rows, orb_aligned, orb_tracked_pairs, threshold_rows),
            "s5_on_orb_matched_pairs": _evaluate_set("s5_on_orb_matched_pairs", gt_rows, s5_aligned, matched_pairs, threshold_rows),
            "orbslam3_on_matched_pairs": _evaluate_set("orbslam3_on_matched_pairs", gt_rows, orb_aligned, matched_pairs, threshold_rows),
        }
        for name, pack in packs.items():
            fname = {
                "s5_full": f"s5_full_{align}_threshold_sweep.json",
                "orbslam3_tracked": f"orbslam3_tracked_{align}_threshold_sweep.json",
                "s5_on_orb_matched_pairs": f"s5_on_orb_matched_{align}_threshold_sweep.json",
                "orbslam3_on_matched_pairs": f"orbslam3_on_matched_{align}_threshold_sweep.json",
            }[name]
            p = out_dir / fname
            p.write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result_files[name][align] = str(p.relative_to(REPO_ROOT))
            results[align][name] = _pick_recommended(pack["threshold_sweep"])

    s5_none = results["none"]["s5_on_orb_matched_pairs"]
    s5_se3 = results["se3"]["s5_on_orb_matched_pairs"]
    s5_sim3 = results["sim3"]["s5_on_orb_matched_pairs"]
    rot_confirmed = bool(np.isfinite(s5_none["rot_mean_deg"]) and s5_none["rot_mean_deg"] > 10.0)
    tmag_confirmed = bool(np.isfinite(s5_sim3["tmag_median_ratio"]) and s5_sim3["tmag_median_ratio"] > 1.5)
    tdir_reliable = bool(np.isfinite(s5_se3["tdir_mean_deg"]) and s5_se3["tdir_mean_deg"] < 45.0)

    if rot_confirmed and tmag_confirmed:
        dominant = "mixed"
    elif rot_confirmed:
        dominant = "rotation"
    elif tmag_confirmed:
        dominant = "tmag"
    else:
        dominant = "unknown"

    validation_clean = bool(v_pass and h_pass and u_pass)
    if all([v_pass, h_pass, u_pass, s6_pass]):
        final_classification = "S5D3_COMPLETE_VALIDATION_CLEAN"
    else:
        final_classification = "S5D3_COMPLETE_WITH_VALIDATION_BLOCKER"

    payload = {
        "experiment": "S5D3_validation_and_alignment_aware_component_audit",
        "inputs": {
            "groundtruth": str(gt_path.relative_to(REPO_ROOT)),
            "timestamps": str(ts_path.relative_to(REPO_ROOT)),
            "s5_trajectory": str(s5_path.relative_to(REPO_ROOT)),
            "orbslam3_trajectory": str(orb_path.relative_to(REPO_ROOT)),
            "s5d2_checkpoint": "checkpoints/S5D2_component_diagnostics.json",
            "s5_source_mode": s5_source,
            "orb_source_mode": orb_source,
        },
        "s5_official_locked_metrics": S5_LOCKED,
        "validation": {
            "verify_final_candidate": {"passed": v_pass, "log_path": v_log},
            "project_health_check": {"passed": h_pass, "log_path": h_log},
            "unittest": {"passed": u_pass, "log_path": u_log, "test_count": _extract_unittest_count(_resolve(u_log))},
            "s6_lockdown_eval_only": {
                "passed": s6_pass,
                "log_path": s6_log,
                "root_cause_summary": "PASS; no lockdown mismatch found." if s6_pass else "FAIL in eval-only chain; inspect log.",
            },
            "branch_clean": _git_clean(),
            "validation_clean": validation_clean,
        },
        "alignment_aware_diagnostics": {
            "alignments": ["none", "se3", "sim3"],
            "thresholds": {
                "absolute": abs_th,
                "relative_to_median_gt_step": rel_th,
                "median_gt_step": median_gt_step,
            },
            "result_files": result_files,
            "recommended_threshold_name": "median_gt_step_x_0.05",
            "recommended_threshold_value": median_gt_step * 0.05,
            "recommended_threshold_summary": results,
        },
        "summary_findings": {
            "s5_rotation_error_confirmed": rot_confirmed,
            "s5_tmag_overscale_confirmed": tmag_confirmed,
            "tdir_reliable_after_threshold_alignment": tdir_reliable,
            "tdir_interpretation": "tdir improves under se3/sim3 and larger thresholds if tiny-step/gauge effects dominate.",
            "dominant_s5_error_source": dominant,
            "evidence": [
                f"S5 matched rot mean (none): {_fmt(s5_none['rot_mean_deg'])} deg",
                f"S5 matched tdir mean (none/se3/sim3): {_fmt(s5_none['tdir_mean_deg'])} / {_fmt(s5_se3['tdir_mean_deg'])} / {_fmt(s5_sim3['tdir_mean_deg'])} deg",
                f"S5 matched tmag median ratio (none/se3/sim3): {_fmt(s5_none['tmag_median_ratio'])} / {_fmt(s5_se3['tmag_median_ratio'])} / {_fmt(s5_sim3['tmag_median_ratio'])}",
            ],
            "recommendations": [
                "Run dense export convention audit on axis/sign/frame consistency.",
                "Add explicit tmag scale calibration regularization across short windows.",
                "Strengthen SO(3) geodesic rotation loss and short-window pose consistency.",
                "If tdir remains high after alignment-aware thresholds, add tdir-focused loss.",
                "Use ORB-SLAM3 successful segments as distillation targets.",
            ],
        },
        "fairness_notes": {
            "same_gt": True,
            "same_component_metric_definitions": True,
            "same_input_protocol": False,
            "input_protocol_note": "S5 uses project panorama/S5 representation; ORB-SLAM3 uses raw fisheye cam0.",
            "orbslam3_calibration_caveat": "fitted KB8 compatibility approximation, not native factory KB8",
            "orbslam3_partial_coverage": True,
            "matched_pair_rules": {
                "no_orb_interpolation": True,
                "same_matched_timestamp_pairs": True,
            },
        },
        "final_classification": final_classification,
        "classification_options": ALLOWED_CLASSIFICATIONS,
        "guardrails": {
            "does_not_modify_s5_policy": True,
            "does_not_modify_final_manifest": True,
            "does_not_modify_train_test_split": True,
            "does_not_modify_official_evaluator": True,
            "does_not_modify_s5_locked_metrics": True,
            "does_not_replace_official_s5_locked_result": True,
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(out_report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
