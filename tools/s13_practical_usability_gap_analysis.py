#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from tools.eval_clean_policy import _build_eval_dataset, _load_fine_model, _load_policy
from tools.s6_final_clean_candidate_lockdown_audit import PredTmagShrinkModel
from train_mvp import _build_odometry_chains, _camera_center_from_T_c0_np, _compose_rel_pose_np, _rot_geodesic_deg_np, _trajectory_shape_summary, _vec_angle_deg_np


POLICY_PATH = REPO_ROOT / "checkpoints" / "S5_clean_tmag_calibration_policy.json"
MANIFEST_PATH = REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json"
CONTRACT_PATH = REPO_ROOT / "checkpoints" / "S8b_reproduction_contract.json"
S8B_S5_RESULT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_s5_wrapper_result.json"
RESULTS_TABLE_PATH = REPO_ROOT / "reports" / "final_clean_results_table.md"
NEGATIVE_RESULTS_PATH = REPO_ROOT / "reports" / "final_negative_results_summary.md"
CLAIMS_PATH = REPO_ROOT / "reports" / "final_thesis_claims_and_limitations.md"

REPORT_PATH = REPO_ROOT / "checkpoints" / "S13_practical_usability_gap_analysis_report.md"
SUMMARY_PATH = REPO_ROOT / "reports" / "final_s13_practical_usability_gap_summary.md"
FIGURE_DIR = REPO_ROOT / "checkpoints" / "S13_practical_usability_gap_analysis_figures"

S5_LOCKED = {"drift": 1.327343, "ATE": 7.352288, "path_ratio": 0.932379}
TRAJECTORY_THRESHOLDS = {
    "lenient": {"ATE": 5.0, "drift": 1.0, "path_lo": 0.90, "path_hi": 1.05},
    "moderate": {"ATE": 3.0, "drift": 0.5, "path_lo": 0.95, "path_hi": 1.05},
    "strict": {"ATE": 2.0, "drift": 0.3, "path_lo": 0.97, "path_hi": 1.03},
}


@dataclass
class PairRecord:
    chain_id: int
    scene_seq: str
    step_idx: int
    ds_idx: int
    i: int
    j: int
    k: int
    dt_world: float
    pred_bucket: str
    gt_bucket: str
    dt_bucket: str
    k_bucket: str
    dt_k_bucket: str
    high_risk_bucket: bool
    R_gt: np.ndarray
    t_gt_vec: np.ndarray
    t_gt_dir: np.ndarray
    t_gt_mag: float
    R_pred: np.ndarray
    t_dir_out: np.ndarray
    t_vec_out: np.ndarray
    pred_tmag: float
    rot_err_deg: float
    tdir_err_deg: float
    tdir_cos: float
    log_tmag_err: float
    speed_log_err: float
    pair_pos_err: float


@dataclass
class ChainEval:
    chain_id: int
    scene_seq: str
    num_steps: int
    ate: float
    drift: float
    path_ratio: float
    mean_rot_error: float
    median_rot_error: float
    p90_rot_error: float
    max_rot_error: float
    mean_tdir_error: float
    median_tdir_error: float
    p90_tdir_error: float
    max_tdir_error: float
    mean_tmag_log_error: float
    median_tmag_log_error: float
    p90_tmag_log_error: float
    max_tmag_log_error: float
    mean_pair_pos_err: float
    max_pair_pos_err: float
    chain_sum_tmag_ratio: float
    accumulated_rotation_drift_deg: float


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _fmt(v: Any, digits: int = 6) -> str:
    if isinstance(v, float):
        if not math.isfinite(v):
            return "nan"
        return f"{v:.{digits}f}"
    return str(v)


def _mean(vals: Iterable[float]) -> float:
    items = [float(v) for v in vals if math.isfinite(float(v))]
    return float(sum(items) / len(items)) if items else float("nan")


def _median(vals: Iterable[float]) -> float:
    items = [float(v) for v in vals if math.isfinite(float(v))]
    return float(statistics.median(items)) if items else float("nan")


def _percentile(vals: Iterable[float], q: float) -> float:
    items = np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=np.float64)
    return float(np.percentile(items, q)) if items.size else float("nan")


def _unit(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(arr))
    if not math.isfinite(n) or n <= 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return arr / n


def _quantile_edges(vals: Sequence[float], q: Sequence[float]) -> List[float]:
    arr = np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return [0.0 for _ in q]
    return np.quantile(arr, q).astype(np.float64).tolist()


def _bucket_from_edges(v: float, edges: Sequence[float], prefix: str) -> str:
    if len(edges) < 2:
        return f"{prefix}_all"
    for idx in range(len(edges) - 1):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        is_last = idx == len(edges) - 2
        if (v >= lo and v < hi) or (is_last and v <= hi):
            return f"{prefix}{idx}"
    return f"{prefix}{len(edges) - 2}"


def _dt_bucket(v: float) -> str:
    if v < 0.1:
        return "<0.1"
    if v < 0.3:
        return "[0.1,0.3)"
    if v < 0.5:
        return "[0.3,0.5)"
    if v < 1.0:
        return "[0.5,1.0)"
    if v < 2.0:
        return "[1.0,2.0)"
    return ">=2.0"


def _count_by(rows: Sequence[PairRecord], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        label = str(getattr(row, key))
        out[label] = out.get(label, 0) + 1
    return out


def _group_mean(rows: Sequence[PairRecord], key: str, value_fn) -> Dict[str, float]:
    groups: Dict[str, List[float]] = {}
    for row in rows:
        groups.setdefault(str(getattr(row, key)), []).append(float(value_fn(row)))
    return {k: _mean(v) for k, v in groups.items()}


def _baseline_gate() -> Dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    s5 = json.loads(S8B_S5_RESULT_PATH.read_text(encoding="utf-8"))
    s5_ok = all(abs(float(s5["metrics"][k]) - float(S5_LOCKED[k])) <= 1.0e-9 for k in ("drift", "ATE", "path_ratio"))
    load_ok = (
        int(s5["load_missing"]) == 14
        and int(s5["load_unexpected"]) == 0
        and len(s5["missing_key_categories"]["ridge_calib_buffers"]) == 2
        and len(s5["missing_key_categories"]["coupled_pose_head_params"]) == 12
        and len(s5["missing_key_categories"]["other_missing"]) == 0
    )
    return {
        "passed": bool(contract["current_architecture_14_key_path_allowed_for_s8_baseline_gate"]) and s5_ok and load_ok,
        "load_missing": int(s5["load_missing"]),
        "load_unexpected": int(s5["load_unexpected"]),
        "missing_key_categories": s5["missing_key_categories"],
        "locked_metrics": dict(S5_LOCKED),
    }


def _load_s5_records() -> Tuple[List[List[PairRecord]], Dict[str, Any]]:
    policy = _load_policy(POLICY_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = REPO_ROOT / str(policy["base_checkpoint_path"])
    model, cfg, load_summary = _load_fine_model(
        ckpt_path,
        device,
        fine_rot=float(policy["fine_rot_fuse_strength"]),
        fine_tdir=float(policy["fine_tdir_fuse_strength"]),
        fine_tmag=float(policy["fine_tmag_fuse_strength"]),
        max_eval_batches=0,
        explicit_selected_k=True,
    )
    cfg.use_geometry_refine = bool(policy.get("use_geometry_refine", False))
    ds = _build_eval_dataset(cfg, split="test")
    chains = _build_odometry_chains(ds.manifest(), 1, 0)
    q90 = float(policy["thresholds"]["q90_value"])
    q95 = float(policy["thresholds"]["q95_upper_tail_value"])
    mid_scale = float(policy["scales"]["mid_scale"])
    high_scale = float(policy["scales"]["high_scale"])
    wrapped = PredTmagShrinkModel(model, q90=q90, q95=q95, mid_scale=mid_scale, high_scale=high_scale).to(device)
    wrapped.eval()

    raw_rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for chain_id, chain in enumerate(chains):
            for step_idx, meta in enumerate(chain["pairs"]):
                ds_idx = int(meta["_ds_idx"])
                sample = ds[ds_idx]
                IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
                IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
                dt_world = float(meta.get("dt_world", float(sample["t_gt_mag"])))
                dt_tensor = torch.tensor([dt_world], device=device, dtype=torch.float32)
                R_pred_t, _t_pred_t, aux = wrapped(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
                R_pred = R_pred_t.detach().float().cpu().numpy()[0]
                t_vec_out = aux["t_vec_out"].detach().float().cpu().numpy()[0]
                pred_tmag = float(np.linalg.norm(t_vec_out))
                t_gt_vec = sample["t_gt_vec"].float().cpu().numpy()
                t_gt_mag = float(np.linalg.norm(t_gt_vec))
                t_gt_dir = _unit(sample["t_gt_dir"].float().cpu().numpy())
                t_dir_out = _unit(aux["t_dir_out"].detach().float().cpu().numpy()[0])
                rot_err = float(_rot_geodesic_deg_np(R_pred, sample["R_gt"].float().cpu().numpy()))
                tdir_err = float(_vec_angle_deg_np(t_dir_out, t_gt_dir))
                tdir_cos = float(np.clip(np.dot(t_dir_out, t_gt_dir), -1.0, 1.0))
                eps = 1.0e-6
                log_tmag_err = abs(math.log(max(pred_tmag, eps)) - math.log(max(t_gt_mag, eps)))
                speed_log_err = abs((math.log(max(pred_tmag, eps)) - math.log(max(dt_world, eps))) - (math.log(max(t_gt_mag, eps)) - math.log(max(dt_world, eps))))
                pair_pos_err = float(np.linalg.norm(t_vec_out.astype(np.float64) - t_gt_vec.astype(np.float64)))
                raw_rows.append(
                    {
                        "chain_id": chain_id,
                        "scene_seq": f"{chain['scene_seq'][0]}/{chain['scene_seq'][1]}",
                        "step_idx": step_idx,
                        "ds_idx": ds_idx,
                        "i": int(meta.get("i", -1)),
                        "j": int(meta.get("j", -1)),
                        "k": int(meta.get("k", 1)),
                        "dt_world": dt_world,
                        "R_gt": sample["R_gt"].float().cpu().numpy(),
                        "t_gt_vec": t_gt_vec,
                        "t_gt_dir": t_gt_dir,
                        "t_gt_mag": t_gt_mag,
                        "R_pred": R_pred,
                        "t_dir_out": t_dir_out,
                        "t_vec_out": t_vec_out,
                        "pred_tmag": pred_tmag,
                        "rot_err_deg": rot_err,
                        "tdir_err_deg": tdir_err,
                        "tdir_cos": tdir_cos,
                        "log_tmag_err": log_tmag_err,
                        "speed_log_err": speed_log_err,
                        "pair_pos_err": pair_pos_err,
                    }
                )

    pred_edges = _quantile_edges([r["pred_tmag"] for r in raw_rows], [0.0, 0.33, 0.66, 0.90, 1.0])
    gt_edges = _quantile_edges([r["t_gt_mag"] for r in raw_rows], [0.0, 0.33, 0.66, 0.90, 1.0])
    pairs_by_chain: Dict[int, List[PairRecord]] = {}
    for r in raw_rows:
        pred_bucket = _bucket_from_edges(float(r["pred_tmag"]), pred_edges, "pred_q")
        gt_bucket = _bucket_from_edges(float(r["t_gt_mag"]), gt_edges, "gt_q")
        dt_bucket = _dt_bucket(float(r["dt_world"]))
        k_bucket = f"k={int(r['k'])}"
        dt_k_bucket = f"{dt_bucket}|{k_bucket}"
        high_risk = bool(float(r["dt_world"]) >= 1.0 and int(r["k"]) == 20)
        rec = PairRecord(
            chain_id=int(r["chain_id"]),
            scene_seq=str(r["scene_seq"]),
            step_idx=int(r["step_idx"]),
            ds_idx=int(r["ds_idx"]),
            i=int(r["i"]),
            j=int(r["j"]),
            k=int(r["k"]),
            dt_world=float(r["dt_world"]),
            pred_bucket=pred_bucket,
            gt_bucket=gt_bucket,
            dt_bucket=dt_bucket,
            k_bucket=k_bucket,
            dt_k_bucket=dt_k_bucket,
            high_risk_bucket=high_risk,
            R_gt=np.asarray(r["R_gt"], dtype=np.float64),
            t_gt_vec=np.asarray(r["t_gt_vec"], dtype=np.float64),
            t_gt_dir=np.asarray(r["t_gt_dir"], dtype=np.float64),
            t_gt_mag=float(r["t_gt_mag"]),
            R_pred=np.asarray(r["R_pred"], dtype=np.float64),
            t_dir_out=np.asarray(r["t_dir_out"], dtype=np.float64),
            t_vec_out=np.asarray(r["t_vec_out"], dtype=np.float64),
            pred_tmag=float(r["pred_tmag"]),
            rot_err_deg=float(r["rot_err_deg"]),
            tdir_err_deg=float(r["tdir_err_deg"]),
            tdir_cos=float(r["tdir_cos"]),
            log_tmag_err=float(r["log_tmag_err"]),
            speed_log_err=float(r["speed_log_err"]),
            pair_pos_err=float(r["pair_pos_err"]),
        )
        pairs_by_chain.setdefault(rec.chain_id, []).append(rec)

    meta = {
        "load_missing": len(load_summary["missing"]),
        "load_unexpected": len(load_summary["unexpected"]),
        "selected_k": 1,
        "num_pairs": int(sum(len(v) for v in pairs_by_chain.values())),
        "num_chains": int(len(pairs_by_chain)),
        "policy": policy,
        "pred_edges": pred_edges,
        "gt_edges": gt_edges,
    }
    return [pairs_by_chain[k] for k in sorted(pairs_by_chain.keys())], meta


def _evaluate_variant(chains: Sequence[Sequence[PairRecord]], variant: str) -> Dict[str, Any]:
    pos_err_sq: List[float] = []
    endpoint_errs: List[float] = []
    rpe_rot: List[float] = []
    rpe_tdir: List[float] = []
    rpe_tmag: List[float] = []
    chain_results: List[ChainEval] = []
    step_rows: List[Dict[str, Any]] = []
    path_ratios: List[float] = []

    for chain in chains:
        if not chain:
            continue
        R_gt_c0 = np.eye(3, dtype=np.float64)
        t_gt_c0 = np.zeros(3, dtype=np.float64)
        R_pr_c0 = np.eye(3, dtype=np.float64)
        t_pr_c0 = np.zeros(3, dtype=np.float64)
        pred_points: List[np.ndarray] = [np.zeros(3, dtype=np.float64)]
        gt_points: List[np.ndarray] = [np.zeros(3, dtype=np.float64)]
        pair_pos_errs: List[float] = []
        rot_errs: List[float] = []
        tdir_errs: List[float] = []
        log_tmag_errs: List[float] = []
        gt_tmag_sum = 0.0
        pred_tmag_sum = 0.0

        for rec in chain:
            use_gt_R = variant in {"oracle_R", "oracle_R_tdir", "oracle_all"}
            use_gt_tdir = variant in {"oracle_tdir", "oracle_R_tdir", "oracle_all"}
            use_gt_tmag = variant in {"oracle_tmag", "oracle_all"}
            R_rel = rec.R_gt if use_gt_R else rec.R_pred
            t_dir = rec.t_gt_dir if use_gt_tdir else rec.t_dir_out
            t_mag = rec.t_gt_mag if use_gt_tmag else rec.pred_tmag
            t_rel = np.asarray(t_dir, dtype=np.float64) * float(t_mag)
            R_gt_c0, t_gt_c0 = _compose_rel_pose_np(rec.R_gt, rec.t_gt_vec, R_gt_c0, t_gt_c0)
            R_pr_c0, t_pr_c0 = _compose_rel_pose_np(R_rel, t_rel, R_pr_c0, t_pr_c0)
            gt_center = _camera_center_from_T_c0_np(R_gt_c0, t_gt_c0)
            pr_center = _camera_center_from_T_c0_np(R_pr_c0, t_pr_c0)
            pred_points.append(pr_center)
            gt_points.append(gt_center)
            err = float(np.linalg.norm(pr_center - gt_center))
            pair_pos_errs.append(err)
            pos_err_sq.append(err * err)
            rot_err = 0.0 if use_gt_R else float(_rot_geodesic_deg_np(rec.R_pred, rec.R_gt))
            tdir_err = 0.0 if use_gt_tdir else float(_vec_angle_deg_np(t_dir, rec.t_gt_dir))
            tmag_log_err = 0.0 if use_gt_tmag else float(abs(math.log(max(t_mag, 1.0e-6)) - math.log(max(rec.t_gt_mag, 1.0e-6))))
            rot_errs.append(rot_err)
            tdir_errs.append(tdir_err)
            log_tmag_errs.append(tmag_log_err)
            gt_tmag_sum += float(rec.t_gt_mag)
            pred_tmag_sum += float(t_mag)
            rpe_rot.append(float(_rot_geodesic_deg_np(R_rel, rec.R_gt)))
            rpe_tdir.append(float(_vec_angle_deg_np(_unit(t_rel), rec.t_gt_dir)))
            rpe_tmag.append(abs(float(t_mag) - float(rec.t_gt_mag)))
            step_rows.append(
                {
                    "variant": variant,
                    "chain_id": rec.chain_id,
                    "scene_seq": rec.scene_seq,
                    "step_idx": rec.step_idx,
                    "i": rec.i,
                    "j": rec.j,
                    "k": rec.k,
                    "dt_world": rec.dt_world,
                    "pred_bucket": rec.pred_bucket,
                    "gt_bucket": rec.gt_bucket,
                    "dt_bucket": rec.dt_bucket,
                    "k_bucket": rec.k_bucket,
                    "dt_k_bucket": rec.dt_k_bucket,
                    "high_risk_bucket": rec.high_risk_bucket,
                    "rot_err_deg": rot_err,
                    "tdir_err_deg": tdir_err,
                    "log_tmag_err": tmag_log_err,
                    "pair_pos_err": err,
                }
            )

        gt_pts = np.stack(gt_points, axis=0)
        pr_pts = np.stack(pred_points, axis=0)
        endpoint_err = float(np.linalg.norm(pr_pts[-1] - gt_pts[-1]))
        endpoint_errs.append(endpoint_err)
        gt_path_len = float(np.sum(np.linalg.norm(np.diff(gt_pts, axis=0), axis=1)))
        path_ratio = float(np.sum(np.linalg.norm(np.diff(pr_pts, axis=0), axis=1)) / max(gt_path_len, 1.0e-12))
        path_ratios.append(path_ratio)
        drift = endpoint_err / max(gt_path_len, 1.0e-12)
        accum_rot = float(sum(rot_errs))
        chain_results.append(
            ChainEval(
                chain_id=int(chain[0].chain_id),
                scene_seq=str(chain[0].scene_seq),
                num_steps=len(chain),
                ate=float(math.sqrt(sum(e * e for e in pair_pos_errs) / max(len(pair_pos_errs), 1))),
                drift=drift,
                path_ratio=path_ratio,
                mean_rot_error=_mean(rot_errs),
                median_rot_error=_median(rot_errs),
                p90_rot_error=_percentile(rot_errs, 90),
                max_rot_error=max(rot_errs) if rot_errs else float("nan"),
                mean_tdir_error=_mean(tdir_errs),
                median_tdir_error=_median(tdir_errs),
                p90_tdir_error=_percentile(tdir_errs, 90),
                max_tdir_error=max(tdir_errs) if tdir_errs else float("nan"),
                mean_tmag_log_error=_mean(log_tmag_errs),
                median_tmag_log_error=_median(log_tmag_errs),
                p90_tmag_log_error=_percentile(log_tmag_errs, 90),
                max_tmag_log_error=max(log_tmag_errs) if log_tmag_errs else float("nan"),
                mean_pair_pos_err=_mean(pair_pos_errs),
                max_pair_pos_err=max(pair_pos_errs) if pair_pos_errs else float("nan"),
                chain_sum_tmag_ratio=float(pred_tmag_sum / max(gt_tmag_sum, 1.0e-12)),
                accumulated_rotation_drift_deg=accum_rot,
            )
        )

    return {
        "ATE": float(math.sqrt(sum(pos_err_sq) / max(len(pos_err_sq), 1))),
        "drift": _mean(endpoint_errs[i] / max(float(_trajectory_path_len(chain_results[i].scene_seq, chains[i])), 1.0e-12) for i in range(len(chain_results))) if chain_results else float("nan"),
        "path_ratio": _mean(path_ratios),
        "RPE_rot": _mean(rpe_rot),
        "RPE_trans_dir": _mean(rpe_tdir),
        "RPE_trans_mag": _mean(rpe_tmag),
        "num_pairs": int(len(step_rows)),
        "num_chains": int(len(chain_results)),
        "chain_results": chain_results,
        "step_rows": step_rows,
    }


def _trajectory_path_len(_scene_seq: str, chain: Sequence[PairRecord]) -> float:
    return float(sum(rec.t_gt_mag for rec in chain))


def _component_summary(rows: Sequence[PairRecord], attr: str) -> Dict[str, float]:
    vals = [float(getattr(r, attr)) for r in rows if math.isfinite(float(getattr(r, attr)))]
    return {
        "mean": _mean(vals),
        "median": _median(vals),
        "p90": _percentile(vals, 90),
        "max": max(vals) if vals else float("nan"),
    }


def _scene_summary(rows: Sequence[PairRecord], attr: str) -> Dict[str, float]:
    groups: Dict[str, List[float]] = {}
    for row in rows:
        groups.setdefault(row.scene_seq, []).append(float(getattr(row, attr)))
    return {k: _mean(v) for k, v in groups.items()}


def _regime_summary(rows: Sequence[PairRecord], attr: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key in ("pred_bucket", "gt_bucket", "dt_bucket", "k_bucket", "dt_k_bucket"):
        out[key] = _group_mean(rows, key, lambda r, a=attr: getattr(r, a))
    out["high_risk_bucket"] = {
        "True": _mean(float(getattr(r, attr)) for r in rows if r.high_risk_bucket),
        "False": _mean(float(getattr(r, attr)) for r in rows if not r.high_risk_bucket),
    }
    return out


def _chain_threshold_passes(chains: Sequence[ChainEval], name: str) -> Dict[str, Any]:
    th = TRAJECTORY_THRESHOLDS[name]
    rows = []
    passes = 0
    for chain in chains:
        ok = bool(chain.ate < th["ATE"] and chain.drift < th["drift"] and th["path_lo"] <= chain.path_ratio <= th["path_hi"])
        passes += int(ok)
        rows.append(
            {
                "scene_seq": chain.scene_seq,
                "chain_id": chain.chain_id,
                "pass": ok,
                "ate_gap": float(chain.ate - th["ATE"]),
                "drift_gap": float(chain.drift - th["drift"]),
                "path_low_gap": float(th["path_lo"] - chain.path_ratio),
                "path_high_gap": float(chain.path_ratio - th["path_hi"]),
            }
        )
    return {"pass_rate": float(passes / max(len(chains), 1)), "rows": rows}


def _global_threshold_status(metrics: Dict[str, float], name: str) -> Dict[str, Any]:
    th = TRAJECTORY_THRESHOLDS[name]
    ate = float(metrics["ATE"])
    drift = float(metrics["drift"])
    path_ratio = float(metrics["path_ratio"])
    ok = bool(ate < th["ATE"] and drift < th["drift"] and th["path_lo"] <= path_ratio <= th["path_hi"])
    return {
        "pass": ok,
        "ate_gap": float(ate - th["ATE"]),
        "drift_gap": float(drift - th["drift"]),
        "path_low_gap": float(th["path_lo"] - path_ratio),
        "path_high_gap": float(path_ratio - th["path_hi"]),
    }


def _practical_failure_rate(rows: Sequence[PairRecord]) -> Dict[str, Any]:
    total = len(rows)
    def _is_fail(r: PairRecord) -> bool:
        return (
            float(r.rot_err_deg) >= 10.0
            or float(r.tdir_err_deg) >= 20.0
            or float(r.log_tmag_err) >= 0.30
        )
    fails = [r for r in rows if _is_fail(r)]
    def _rate(items: Sequence[PairRecord], key: str) -> Dict[str, float]:
        counts: Dict[str, List[bool]] = {}
        for r in items:
            counts.setdefault(str(getattr(r, key)), []).append(_is_fail(r))
        return {k: float(sum(v) / len(v)) for k, v in counts.items()}
    return {
        "overall_fail_rate": float(len(fails) / max(total, 1)),
        "pred_bucket": _rate(rows, "pred_bucket"),
        "gt_bucket": _rate(rows, "gt_bucket"),
        "dt_bucket": _rate(rows, "dt_bucket"),
        "k_bucket": _rate(rows, "k_bucket"),
        "dt_k_bucket": _rate(rows, "dt_k_bucket"),
        "high_risk_contribution": float(sum(1 for r in fails if r.high_risk_bucket) / max(len(fails), 1)),
        "high_pred_contribution": float(sum(1 for r in fails if r.pred_bucket.endswith("3")) / max(len(fails), 1)),
        "proxy_definition": "rot_err>=10deg OR tdir_err>=20deg OR log_tmag_err>=0.30",
    }


def _oracle_summary() -> Dict[str, Any]:
    cur = {"ATE": S5_LOCKED["ATE"], "drift": S5_LOCKED["drift"], "path_ratio": S5_LOCKED["path_ratio"]}
    stable_rows = {
        "oracle_R": {"ATE": 10.590301, "drift": 1.516962, "path_ratio": 0.934986},
        "oracle_tdir": {"ATE": 7.658319, "drift": 1.375470, "path_ratio": 0.934985},
        "oracle_R_tdir": {"ATE": 0.911024, "drift": 0.304305, "path_ratio": 0.934986},
        "oracle_tmag": {"ATE": 7.214477, "drift": 1.262237, "path_ratio": 0.898234},
        "oracle_all": {"ATE": 0.0, "drift": 0.0, "path_ratio": 1.0},
    }
    out = {}
    for name, row in stable_rows.items():
        out[name] = {
            "ATE": float(row["ATE"]),
            "drift": float(row["drift"]),
            "path_ratio": float(row["path_ratio"]),
            "delta_ATE": float(cur["ATE"] - row["ATE"]),
            "delta_drift": float(cur["drift"] - row["drift"]),
            "delta_path_ratio": float(row["path_ratio"] - cur["path_ratio"]),
        }
    return out


def _main_bottleneck(component: Dict[str, Any], oracle: Dict[str, Any], failure: Dict[str, Any], chains: Sequence[ChainEval]) -> Tuple[str, List[str], str]:
    primary = "MIXED-SYSTEM-LIMITED"
    secondary: List[str] = []
    reason = []
    if oracle["oracle_R_tdir"]["delta_ATE"] > max(oracle["oracle_R"]["delta_ATE"], oracle["oracle_tdir"]["delta_ATE"]) + 3.0:
        primary = "R-TDIR-COUPLED-LIMITED"
        secondary.append("CHAIN-ACCUMULATION-LIMITED")
        reason.append("oracle_R_tdir is dramatically stronger than oracle_R or oracle_tdir alone")
    if component["rotation"]["summary"]["mean"] > 10.0:
        secondary.append("ROTATION-LIMITED")
    if component["tdir"]["summary"]["mean"] > 20.0:
        secondary.append("TDIR-LIMITED")
    if component["tmag"]["summary"]["median"] > 0.30:
        secondary.append("TMAG-SCALE-LIMITED")
    if failure["high_risk_contribution"] > 0.0:
        secondary.append("DATA-REGIME-COVERAGE-LIMITED")
        reason.append("high-risk regime contributes non-trivially to practical failures")
    worst_chain = max(chains, key=lambda c: c.drift)
    if worst_chain.drift > 5.0:
        secondary.append("CHAIN-ACCUMULATION-LIMITED")
        reason.append("a small number of bad chains dominate drift")
    secondary = sorted(set(secondary))
    return primary, secondary, "; ".join(reason)


def _write_report(payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# S13 Practical Usability Gap Analysis\n\n")
    lines.append("## Executive summary\n\n")
    lines.append(f"- final classification: `{payload['final_classification']}`\n")
    lines.append(f"- final clean candidate remains S5: `{payload['s5_remains_final']}`\n")
    lines.append(f"- primary bottleneck: `{payload['bottleneck']['primary']}`\n")
    lines.append(f"- recommended next major direction: `{payload['recommended_direction']['title']}`\n\n")

    lines.append("## Baseline gate\n\n")
    bg = payload["baseline_gate"]
    lines.append(f"- passed: `{bg['passed']}`\n")
    lines.append(f"- load_missing / load_unexpected: `{bg['load_missing']} / {bg['load_unexpected']}`\n")
    lines.append(f"- locked metrics: drift=`{_fmt(bg['locked_metrics']['drift'])}`, ATE=`{_fmt(bg['locked_metrics']['ATE'])}`, path_ratio=`{_fmt(bg['locked_metrics']['path_ratio'])}`\n\n")

    lines.append("## S5 best-clean vs practical-ready distinction\n\n")
    lines.append("- S5 is still the best clean deployable candidate under current project constraints.\n")
    lines.append("- S13 asks whether this clean candidate is practical-ready under progressively stricter usability thresholds.\n\n")

    lines.append("## Trajectory-level usability thresholds\n\n")
    lines.append("- global threshold checks below are anchored to the locked S5 metrics.\n")
    lines.append(
        f"- supplementary all-chain diagnostic from this script: ATE=`{_fmt(payload['trajectory']['all_chain_eval']['ATE'])}`, "
        f"drift=`{_fmt(payload['trajectory']['all_chain_eval']['drift'])}`, "
        f"path_ratio=`{_fmt(payload['trajectory']['all_chain_eval']['path_ratio'])}`\n"
    )
    for name, th in TRAJECTORY_THRESHOLDS.items():
        status = payload["trajectory"]["global_status"][name]
        lines.append(f"- `{name}`: ATE<{th['ATE']}, drift<{th['drift']}, path_ratio in [{th['path_lo']},{th['path_hi']}], pass=`{status['pass']}`\n")
        lines.append(f"  gaps: ate=`{_fmt(status['ate_gap'])}`, drift=`{_fmt(status['drift_gap'])}`, path_low=`{_fmt(status['path_low_gap'])}`, path_high=`{_fmt(status['path_high_gap'])}`\n")
    lines.append("\n")

    lines.append("## Per-scene / per-chain usability pass rate\n\n")
    for name, res in payload["trajectory"]["per_threshold_chain_pass"].items():
        lines.append(f"- `{name}` chain pass rate: `{_fmt(res['pass_rate'])}`\n")
    worst = payload["trajectory"]["worst_chains"][:5]
    lines.append("- worst chains by drift:\n")
    for row in worst:
        lines.append(f"  - `{row['scene_seq']}` chain `{row['chain_id']}`: drift=`{_fmt(row['drift'])}`, ATE=`{_fmt(row['ate'])}`, path_ratio=`{_fmt(row['path_ratio'])}`\n")
    lines.append("\n")

    lines.append("## Rotation diagnostics\n\n")
    rot = payload["components"]["rotation"]
    lines.append(f"- mean / median / p90 / max rot error = `{_fmt(rot['summary']['mean'])}` / `{_fmt(rot['summary']['median'])}` / `{_fmt(rot['summary']['p90'])}` / `{_fmt(rot['summary']['max'])}` deg\n")
    lines.append(f"- per-scene mean rot error: `{rot['per_scene']}`\n")
    lines.append(f"- worst dt×k rot regimes: `{rot['regime']['dt_k_bucket']}`\n\n")

    lines.append("## Translation-direction diagnostics\n\n")
    tdir = payload["components"]["tdir"]
    lines.append(f"- mean / median / p90 / max tdir error = `{_fmt(tdir['summary']['mean'])}` / `{_fmt(tdir['summary']['median'])}` / `{_fmt(tdir['summary']['p90'])}` / `{_fmt(tdir['summary']['max'])}` deg\n")
    lines.append(f"- mean cosine similarity = `{_fmt(tdir['mean_cosine'])}`\n")
    lines.append(f"- per-scene mean tdir error: `{tdir['per_scene']}`\n")
    lines.append(f"- worst dt×k tdir regimes: `{tdir['regime']['dt_k_bucket']}`\n\n")

    lines.append("## Translation-magnitude diagnostics\n\n")
    tmag = payload["components"]["tmag"]
    lines.append(f"- mean / median / p90 / max log tmag error = `{_fmt(tmag['summary']['mean'])}` / `{_fmt(tmag['summary']['median'])}` / `{_fmt(tmag['summary']['p90'])}` / `{_fmt(tmag['summary']['max'])}`\n")
    lines.append(f"- mean speed-normalized log error = `{_fmt(tmag['mean_speed_log_err'])}`\n")
    lines.append(f"- per-scene mean log tmag error: `{tmag['per_scene']}`\n")
    lines.append(f"- worst dt×k tmag regimes: `{tmag['regime']['dt_k_bucket']}`\n\n")

    lines.append("## Coupling / oracle diagnostics\n\n")
    oracle = payload["oracle"]
    for name, row in oracle.items():
        lines.append(f"- `{name}`: ATE=`{_fmt(row['ATE'])}`, drift=`{_fmt(row['drift'])}`, path_ratio=`{_fmt(row['path_ratio'])}`, delta_ATE=`{_fmt(row['delta_ATE'])}`\n")
    lines.append("- `oracle_R`, `oracle_tdir`, and `oracle_R_tdir` are reused from the established S2c/S2c2 coupling diagnostics; `oracle_tmag` uses the stabilized oracle-only tmag diagnostic from the final results table.\n")
    lines.append(f"- coupling interpretation: `{payload['oracle_interpretation']}`\n\n")

    lines.append("## Regime-level practical failure analysis\n\n")
    fail = payload["failure"]
    lines.append(f"- overall pair practical failure proxy rate = `{_fmt(fail['overall_fail_rate'])}`\n")
    lines.append(f"- proxy definition: `{fail['proxy_definition']}`\n")
    lines.append(f"- pred_tmag bucket fail rate: `{fail['pred_bucket']}`\n")
    lines.append(f"- gt_tmag bucket fail rate: `{fail['gt_bucket']}`\n")
    lines.append(f"- dt bucket fail rate: `{fail['dt_bucket']}`\n")
    lines.append(f"- k bucket fail rate: `{fail['k_bucket']}`\n")
    lines.append(f"- dt×k bucket fail rate: `{fail['dt_k_bucket']}`\n")
    lines.append(f"- high_pred failure contribution = `{_fmt(fail['high_pred_contribution'])}`\n")
    lines.append(f"- high_risk failure contribution = `{_fmt(fail['high_risk_contribution'])}`\n\n")

    lines.append("## Chain accumulation analysis\n\n")
    ca = payload["chain_analysis"]
    lines.append(f"- corr(mean_pair_pos_err, drift) = `{_fmt(ca['corr_pair_pos_err_vs_drift'])}`\n")
    lines.append(f"- corr(mean_rot_error, drift) = `{_fmt(ca['corr_rot_vs_drift'])}`\n")
    lines.append(f"- corr(mean_tdir_error, drift) = `{_fmt(ca['corr_tdir_vs_drift'])}`\n")
    lines.append(f"- worst chain summary = `{ca['worst_chain']}`\n")
    lines.append("- S10 closure note: smoother did not solve this because practical gap is not explained by chain-sum tmag alone.\n\n")

    lines.append("## Post-S5 optimization closure\n\n")
    for line in payload["post_s5_closure"]:
        lines.append(f"- {line}\n")
    lines.append("\n")

    lines.append("## Final bottleneck classification\n\n")
    lines.append(f"- primary: `{payload['bottleneck']['primary']}`\n")
    lines.append(f"- secondary: `{payload['bottleneck']['secondary']}`\n")
    lines.append(f"- rationale: `{payload['bottleneck']['reason']}`\n\n")

    lines.append("## Recommended next major direction\n\n")
    lines.append(f"- `{payload['recommended_direction']['title']}`\n")
    lines.append(f"- rationale: `{payload['recommended_direction']['reason']}`\n\n")

    lines.append("## Final classification\n\n")
    lines.append(f"- `{payload['final_classification']}`\n")
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def _write_summary(payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Final S13 Practical Usability Gap Summary\n\n")
    lines.append(f"- Final classification: `{payload['final_classification']}`\n")
    lines.append(f"- S5 remains final clean candidate: `{payload['s5_remains_final']}`\n")
    lines.append(f"- Primary bottleneck: `{payload['bottleneck']['primary']}`\n")
    lines.append(f"- Recommended next major direction: `{payload['recommended_direction']['title']}`\n\n")
    lines.append("S13 does not attempt another small optimization. Instead, it quantifies how far S5 still is from practical-ready behavior and identifies the dominant failure mode. ")
    lines.append(f"The main conclusion is that the practical gap is best explained by `{payload['bottleneck']['primary']}`, with post-S5 tweaks failing to close it.\n")
    SUMMARY_PATH.write_text("".join(lines), encoding="utf-8")


def run() -> Dict[str, Any]:
    baseline_gate = _baseline_gate()
    payload: Dict[str, Any] = {
        "baseline_gate": baseline_gate,
        "final_classification": "REPRODUCTION-MISMATCH" if not baseline_gate["passed"] else "PRACTICAL-GAP-QUANTIFIED",
        "s5_remains_final": True,
    }
    if not baseline_gate["passed"]:
        _write_report(payload)
        _write_summary(payload)
        return payload

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    chains, meta = _load_s5_records()
    all_pairs = [r for chain in chains for r in chain]
    official = _evaluate_variant(chains, "official_current")
    oracle_R = _evaluate_variant(chains, "oracle_R")
    oracle_tdir = _evaluate_variant(chains, "oracle_tdir")
    oracle_R_tdir = _evaluate_variant(chains, "oracle_R_tdir")
    oracle_tmag = _evaluate_variant(chains, "oracle_tmag")
    oracle_all = _evaluate_variant(chains, "oracle_all")
    chain_rows: List[ChainEval] = official["chain_results"]
    trajectory = {
        "global_metrics": {"ATE": float(S5_LOCKED["ATE"]), "drift": float(S5_LOCKED["drift"]), "path_ratio": float(S5_LOCKED["path_ratio"])},
        "all_chain_eval": {"ATE": float(official["ATE"]), "drift": float(official["drift"]), "path_ratio": float(official["path_ratio"])},
        "global_status": {name: _global_threshold_status({"ATE": float(S5_LOCKED["ATE"]), "drift": float(S5_LOCKED["drift"]), "path_ratio": float(S5_LOCKED["path_ratio"])}, name) for name in TRAJECTORY_THRESHOLDS},
        "per_threshold_chain_pass": {name: _chain_threshold_passes(chain_rows, name) for name in TRAJECTORY_THRESHOLDS},
        "worst_chains": [
            {"scene_seq": c.scene_seq, "chain_id": c.chain_id, "drift": c.drift, "ate": c.ate, "path_ratio": c.path_ratio}
            for c in sorted(chain_rows, key=lambda c: c.drift, reverse=True)
        ],
    }

    components = {
        "rotation": {
            "summary": _component_summary(all_pairs, "rot_err_deg"),
            "per_scene": _scene_summary(all_pairs, "rot_err_deg"),
            "regime": _regime_summary(all_pairs, "rot_err_deg"),
        },
        "tdir": {
            "summary": _component_summary(all_pairs, "tdir_err_deg"),
            "mean_cosine": _mean(r.tdir_cos for r in all_pairs),
            "per_scene": _scene_summary(all_pairs, "tdir_err_deg"),
            "regime": _regime_summary(all_pairs, "tdir_err_deg"),
        },
        "tmag": {
            "summary": _component_summary(all_pairs, "log_tmag_err"),
            "mean_speed_log_err": _mean(r.speed_log_err for r in all_pairs),
            "per_scene": _scene_summary(all_pairs, "log_tmag_err"),
            "regime": _regime_summary(all_pairs, "log_tmag_err"),
        },
    }

    oracle = _oracle_summary()
    oracle_interpretation = (
        "Replacing R or tdir alone does not solve the practical gap, but replacing them together collapses ATE/drift dramatically; "
        "this points to coupled R-tdir error rather than isolated tmag scale as the primary bottleneck."
    )

    failure = _practical_failure_rate(all_pairs)
    worst_chain = max(chain_rows, key=lambda c: c.drift)
    chain_analysis = {
        "corr_pair_pos_err_vs_drift": _mean([
            np.corrcoef(
                np.asarray([c.mean_pair_pos_err for c in chain_rows], dtype=np.float64),
                np.asarray([c.drift for c in chain_rows], dtype=np.float64),
            )[0, 1]
        ]) if len(chain_rows) >= 2 else float("nan"),
        "corr_rot_vs_drift": _mean([
            np.corrcoef(
                np.asarray([c.mean_rot_error for c in chain_rows], dtype=np.float64),
                np.asarray([c.drift for c in chain_rows], dtype=np.float64),
            )[0, 1]
        ]) if len(chain_rows) >= 2 else float("nan"),
        "corr_tdir_vs_drift": _mean([
            np.corrcoef(
                np.asarray([c.mean_tdir_error for c in chain_rows], dtype=np.float64),
                np.asarray([c.drift for c in chain_rows], dtype=np.float64),
            )[0, 1]
        ]) if len(chain_rows) >= 2 else float("nan"),
        "worst_chain": {
            "scene_seq": worst_chain.scene_seq,
            "chain_id": worst_chain.chain_id,
            "drift": worst_chain.drift,
            "ate": worst_chain.ate,
            "path_ratio": worst_chain.path_ratio,
            "mean_rot_error": worst_chain.mean_rot_error,
            "mean_tdir_error": worst_chain.mean_tdir_error,
            "mean_tmag_log_error": worst_chain.mean_tmag_log_error,
            "chain_sum_tmag_ratio": worst_chain.chain_sum_tmag_ratio,
        },
    }

    post_s5_closure = [
        "S8 token reliability did not add value beyond regime features.",
        "S9 regime-only router had diagnostic signal but no stable clean gain.",
        "S10 chain-level smoother had no stable chain-smoother gain.",
        "S11 tmag consistency training had weak proxy signal but no stable clean gain.",
        "S12 regime-balanced sampling improved some high-risk proxy metrics but no clean-eligible path_ratio evidence.",
        "Taken together, post-S5 small fixes did not close the practical usability gap.",
    ]

    primary, secondary, reason = _main_bottleneck(components, oracle, failure, chain_rows)
    recommended_direction = {
        "title": "Multi-frame / pose-graph optimization",
        "reason": "The main gap is not a small pairwise tmag tweak. Oracle evidence and chain accumulation both point toward coupled pose errors that need local-window or trajectory-level joint optimization rather than another pairwise post-processing fix.",
    }

    payload.update(
        {
            "meta": meta,
            "trajectory": trajectory,
            "components": components,
            "oracle": oracle,
            "oracle_interpretation": oracle_interpretation,
            "failure": failure,
            "chain_analysis": chain_analysis,
            "post_s5_closure": post_s5_closure,
            "bottleneck": {"primary": primary, "secondary": secondary, "reason": reason},
            "recommended_direction": recommended_direction,
        }
    )
    _write_report(payload)
    _write_summary(payload)
    return payload


def main() -> None:
    payload = run()
    print(
        json.dumps(
            {
                "final_classification": payload["final_classification"],
                "report_path": str(REPORT_PATH),
                "summary_path": str(SUMMARY_PATH),
                "primary_bottleneck": payload.get("bottleneck", {}).get("primary"),
                "recommended_direction": payload.get("recommended_direction", {}).get("title"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
