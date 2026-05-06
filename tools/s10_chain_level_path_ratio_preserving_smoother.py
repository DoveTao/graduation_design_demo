#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from config import Config
from dataset_pano_only import RflyPanoPanoramaPairsEvalFixedKList
from eval_clean_policy import DtBucketScaledMagnitudeModel, _cfg_from_dict, _load_ckpt_cfg
from model import PanoramaRelPoseModel
from train_mvp import (
    _annotate_trajectory_shape_rows,
    _build_odometry_chains,
    _camera_center_from_T_c0_np,
    _compose_rel_pose_np,
    _odom_debug_chain_summary,
    _rot_geodesic_deg_np,
    _trajectory_shape_summary,
    _vec_angle_deg_np,
)


S2B_POLICY_PATH = REPO_ROOT / "checkpoints" / "S2b_clean_fine_rot_policy.json"
S5_POLICY_PATH = REPO_ROOT / "checkpoints" / "S5_clean_tmag_calibration_policy.json"
S8B_CONTRACT_PATH = REPO_ROOT / "checkpoints" / "S8b_reproduction_contract.json"
S8B_S2B_RESULT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_s2b_wrapper_result.json"
S8B_S5_RESULT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_s5_wrapper_result.json"

REPORT_PATH = REPO_ROOT / "checkpoints" / "S10_chain_level_path_ratio_preserving_smoother_report.md"
CANDIDATES_PATH = REPO_ROOT / "checkpoints" / "S10_chain_level_path_ratio_preserving_smoother_candidates.json"
POLICY_PATH = REPO_ROOT / "checkpoints" / "S10_chain_level_path_ratio_preserving_smoother_policy.json"
FIG_DIR = REPO_ROOT / "checkpoints" / "S10_chain_level_path_ratio_preserving_smoother_figures"
SUMMARY_PATH = REPO_ROOT / "reports" / "final_s10_chain_smoother_summary.md"

S5_LOCKED = {"drift": 1.327343, "ATE": 7.352288, "path_ratio": 0.932379}
CURRENT_ARCH_BENIGN_MISSING = {
    "ridge_calib_buffers": 2,
    "coupled_pose_head_params": 12,
    "other_missing": 0,
}
SAFE_PATH_RANGE = (0.90, 0.97)
RENORM_SAFE_RANGE = (0.85, 1.15)
CHANGED_PAIR_PCT_SOFT_MAX = 0.30


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    params: Dict[str, Any]
    simplicity_rank: Tuple[float, float, float]


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _mean(vals: Iterable[float]) -> float:
    arr = np.asarray(list(vals), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _std(vals: Iterable[float]) -> float:
    arr = np.asarray(list(vals), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return 0.0
    return float(arr.std(ddof=0))


def _nanmean(vals: Sequence[float]) -> float:
    arr = np.asarray(list(vals), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size > 0 else float("nan")


def _rmse(vals: Sequence[float]) -> float:
    arr = np.asarray(list(vals), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(math.sqrt(float(arr.mean()))) if arr.size > 0 else float("nan")


def _build_cfg_from_policy(policy: Dict[str, Any]) -> Tuple[Config, Path]:
    ckpt_path = REPO_ROOT / str(policy["base_checkpoint_path"])
    cfg = _cfg_from_dict(_load_ckpt_cfg(ckpt_path))
    cfg.use_fine_stage = True
    cfg.fine_rot_fuse_strength = float(policy["fine_rot_fuse_strength"])
    cfg.fine_tdir_fuse_strength = float(policy["fine_tdir_fuse_strength"])
    cfg.fine_tmag_fuse_strength = float(policy["fine_tmag_fuse_strength"])
    cfg.use_geometry_refine = False
    cfg.tmag_condition_on_dt = False
    cfg.batch_size = 1
    cfg.num_workers = 0
    cfg.pin_memory = False
    cfg.odom_eval_prefer_k = 1
    cfg.odom_eval_fallback_to_min_k = False
    return cfg, ckpt_path


def _build_dataset(cfg: Config, split: str) -> RflyPanoPanoramaPairsEvalFixedKList:
    return RflyPanoPanoramaPairsEvalFixedKList(
        data_root=str(cfg.data_root),
        split=split,
        split_by=str(cfg.split_by),
        train_ratio=float(cfg.train_ratio),
        split_seed=int(cfg.split_seed),
        H=int(cfg.H),
        W=int(cfg.W),
        k_list=tuple(int(x) for x in cfg.eval_k_list),
        pair_step=int(getattr(cfg, "eval_pair_step", 1)),
        min_dt=float(cfg.eval_min_dt),
        max_dt=None if getattr(cfg, "eval_max_dt", None) is None else float(cfg.eval_max_dt),
    )


def _load_current_arch_model(cfg: Config, ckpt_path: Path, device: torch.device) -> Tuple[PanoramaRelPoseModel, Dict[str, Any]]:
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(ckpt_path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    missing = list(msg.missing_keys)
    unexpected = list(msg.unexpected_keys)
    categories = {
        "ridge_calib_buffers": [k for k in missing if k.endswith("ridge_calib_raw_center")],
        "coupled_pose_head_params": [k for k in missing if k.startswith("coupled_pose_head.")],
    }
    categories["other_missing"] = [k for k in missing if k not in set(categories["ridge_calib_buffers"]) | set(categories["coupled_pose_head_params"])]
    return model, {"missing": missing, "unexpected": unexpected, "categories": categories}


def _build_s2b_model(device: torch.device) -> Tuple[DtBucketScaledMagnitudeModel, Config, Dict[str, Any], Dict[str, Any]]:
    policy = _read_json(S2B_POLICY_PATH)
    cfg, ckpt_path = _build_cfg_from_policy(policy)
    base_model, load_summary = _load_current_arch_model(cfg, ckpt_path, device)
    factors = {str(k): float(v) for k, v in policy["effective_bucket_factors"].items()}
    wrapped = DtBucketScaledMagnitudeModel(base_model, factors).to(device)
    return wrapped, cfg, policy, load_summary


def _load_s5_policy() -> Dict[str, Any]:
    return _read_json(S5_POLICY_PATH)


def _s5_scale_from_pred(pred_tmag: float, s5_policy: Dict[str, Any]) -> float:
    q90 = float(s5_policy["thresholds"]["q90_value"])
    q95 = float(s5_policy["thresholds"]["q95_upper_tail_value"])
    mid = float(s5_policy["scales"]["mid_scale"])
    high = float(s5_policy["scales"]["high_scale"])
    if pred_tmag >= q95:
        return high
    if pred_tmag >= q90:
        return mid
    return 1.0


def _baseline_gate() -> Dict[str, Any]:
    contract = _read_json(S8B_CONTRACT_PATH)
    s2b = _read_json(S8B_S2B_RESULT_PATH)
    s5 = _read_json(S8B_S5_RESULT_PATH)
    s5_ok = all(abs(float(s5["metrics"][k]) - float(S5_LOCKED[k])) <= 1.0e-9 for k in ("drift", "ATE", "path_ratio"))
    s2b_load_ok = (
        int(s2b["load_missing"]) == 14
        and int(s2b["load_unexpected"]) == 0
        and len(s2b["missing_key_categories"]["ridge_calib_buffers"]) == CURRENT_ARCH_BENIGN_MISSING["ridge_calib_buffers"]
        and len(s2b["missing_key_categories"]["coupled_pose_head_params"]) == CURRENT_ARCH_BENIGN_MISSING["coupled_pose_head_params"]
        and len(s2b["missing_key_categories"]["other_missing"]) == CURRENT_ARCH_BENIGN_MISSING["other_missing"]
    )
    s5_load_ok = (
        int(s5["load_missing"]) == 14
        and int(s5["load_unexpected"]) == 0
        and len(s5["missing_key_categories"]["ridge_calib_buffers"]) == CURRENT_ARCH_BENIGN_MISSING["ridge_calib_buffers"]
        and len(s5["missing_key_categories"]["coupled_pose_head_params"]) == CURRENT_ARCH_BENIGN_MISSING["coupled_pose_head_params"]
        and len(s5["missing_key_categories"]["other_missing"]) == CURRENT_ARCH_BENIGN_MISSING["other_missing"]
    )
    passed = bool(contract["current_architecture_14_key_path_allowed_for_s8_baseline_gate"]) and s2b_load_ok and s5_load_ok and s5_ok
    return {
        "passed": passed,
        "contract_path": str(S8B_CONTRACT_PATH),
        "s2b": s2b,
        "s5": s5,
        "missing_unexpected_ok": s2b_load_ok and s5_load_ok,
        "locked_s5_metrics_ok": s5_ok,
    }


def _feature_audit() -> List[Dict[str, Any]]:
    return [
        {"feature": "R_i from S5", "inference_visible": True, "allowed": True},
        {"feature": "tdir_i from S5", "inference_visible": True, "allowed": True},
        {"feature": "tmag_i from S5", "inference_visible": True, "allowed": True},
        {"feature": "pred_tmag_i", "inference_visible": True, "allowed": True},
        {"feature": "dt_i", "inference_visible": True, "allowed": True},
        {"feature": "k_i", "inference_visible": True, "allowed": True},
        {"feature": "S5 bucket id", "inference_visible": True, "allowed": True},
        {"feature": "fine token / spherical token", "inference_visible": True, "allowed": False},
        {"feature": "gt_tmag / gt pose / pair_pos_err", "inference_visible": False, "allowed": False},
    ]


def _candidate_specs() -> List[CandidateSpec]:
    specs: List[CandidateSpec] = []
    for window in (3, 5, 7):
        for alpha in (0.25, 0.50, 0.75):
            specs.append(
                CandidateSpec(
                    name=f"median_w{window}_a{str(alpha).replace('.', 'p')}_preserve",
                    family="median_blend",
                    params={"window": window, "alpha": alpha, "renorm_mode": "preserve_chain_sum"},
                    simplicity_rank=(1.0, float(window), float(alpha)),
                )
            )
    for window in (5, 7):
        for mad_threshold in (2.5, 3.0, 3.5):
            for clip_strength in (0.5, 0.75, 1.0):
                specs.append(
                    CandidateSpec(
                        name=f"spike_w{window}_m{str(mad_threshold).replace('.', 'p')}_c{str(clip_strength).replace('.', 'p')}_preserve",
                        family="spike_clip",
                        params={
                            "window": window,
                            "mad_threshold": mad_threshold,
                            "clip_strength": clip_strength,
                            "renorm_mode": "preserve_chain_sum",
                        },
                        simplicity_rank=(2.0, float(window), float(mad_threshold) + float(clip_strength)),
                    )
                )
    for mode in ("pred_q90", "dt_ge_1", "pred_q90_or_dt_ge_1", "pred_q90_and_dt_ge_1"):
        specs.append(
            CandidateSpec(
                name=f"regime_spike_{mode}_preserve",
                family="regime_spike_clip",
                params={
                    "window": 5,
                    "mad_threshold": 3.0,
                    "clip_strength": 0.75,
                    "regime_mode": mode,
                    "renorm_mode": "preserve_chain_sum",
                },
                simplicity_rank=(3.0, 5.0, float(len(mode))),
            )
        )
    for window in (5, 7):
        specs.append(
            CandidateSpec(
                name=f"hybrid_w{window}_predq90ordt1_a0p50_preserve",
                family="hybrid_risk_blend",
                params={
                    "window": window,
                    "alpha": 0.50,
                    "mad_threshold": 2.5,
                    "clip_strength": 0.75,
                    "regime_mode": "pred_q90_or_dt_ge_1",
                    "renorm_mode": "preserve_chain_sum",
                },
                simplicity_rank=(4.0, float(window), 0.5),
            )
        )
    specs.extend(
        [
            CandidateSpec(
                name="median_w5_a0p50_none",
                family="median_blend",
                params={"window": 5, "alpha": 0.50, "renorm_mode": "none"},
                simplicity_rank=(5.0, 5.0, 0.5),
            ),
            CandidateSpec(
                name="spike_w5_m3p0_c0p75_none",
                family="spike_clip",
                params={"window": 5, "mad_threshold": 3.0, "clip_strength": 0.75, "renorm_mode": "none"},
                simplicity_rank=(5.0, 5.0, 3.75),
            ),
            CandidateSpec(
                name="regime_spike_predq90_none",
                family="regime_spike_clip",
                params={"window": 5, "mad_threshold": 3.0, "clip_strength": 0.75, "regime_mode": "pred_q90", "renorm_mode": "none"},
                simplicity_rank=(5.0, 5.0, 9.0),
            ),
        ]
    )
    return specs


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n <= 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return v / n


def _pair_bucket(dt_world: float, k: int) -> str:
    if dt_world < 0.1:
        dt_bucket = "<0.1"
    elif dt_world < 0.3:
        dt_bucket = "[0.1,0.3)"
    elif dt_world < 0.5:
        dt_bucket = "[0.3,0.5)"
    elif dt_world < 1.0:
        dt_bucket = "[0.5,1.0)"
    else:
        dt_bucket = ">=1.0"
    return f"{dt_bucket}|k={int(k)}"


def _collect_train_groups(cfg: Config):
    train_ds = _build_dataset(cfg, split="train")
    manifest = train_ds.manifest()
    groups = sorted({(str(m.get("scene")), str(m.get("seq"))) for m in manifest})
    seq_to_indices: Dict[Tuple[str, str], List[int]] = {g: [] for g in groups}
    for idx, meta in enumerate(manifest):
        seq_to_indices[(str(meta.get("scene")), str(meta.get("seq")))].append(idx)
    target_groups = [("scene01", "seq01"), ("scene01", "seq02")]
    missing = [g for g in target_groups if g not in seq_to_indices]
    if missing:
        raise RuntimeError(f"missing expected S10 train-CV groups: {missing}")
    return train_ds, target_groups, {g: seq_to_indices[g] for g in target_groups}


def _build_chain_records(model: DtBucketScaledMagnitudeModel, ds, s5_policy: Dict[str, Any], device: torch.device) -> List[Dict[str, Any]]:
    manifest = ds.manifest()
    available_k = sorted({int(m["k"]) for m in manifest if m.get("k", None) is not None})
    if 1 not in available_k:
        raise RuntimeError(f"S10 official odom scope expects selected_k=1, got available_k={available_k}")
    chains = _build_odometry_chains(manifest, selected_k=1, max_pairs=0)
    out: List[Dict[str, Any]] = []
    with torch.no_grad():
        for chain_idx, chain in enumerate(chains):
            rows: List[Dict[str, Any]] = []
            for item in chain.get("pairs", []):
                sample = ds[int(item["_ds_idx"])]
                IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
                IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
                R_gt = sample["R_gt"].float().numpy()
                t_gt_vec = sample.get("t_gt_vec", None)
                if t_gt_vec is not None:
                    t_gt = t_gt_vec.float().numpy()
                else:
                    t_gt_dir = sample["t_gt_dir"].float().numpy()
                    t_gt_mag = float(sample["t_gt_mag"])
                    t_gt = t_gt_dir * t_gt_mag
                t_gt = np.asarray(t_gt, dtype=np.float64).reshape(3)
                t_gt_mag = float(np.linalg.norm(t_gt))
                dt_val = float(sample.get("dt_world", sample.get("t_gt_mag", 0.01)))
                dt_tensor = torch.tensor([dt_val], device=device, dtype=torch.float32)
                R_pred, t_pred, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
                R_pred_np = R_pred.detach().float().cpu().numpy()[0].astype(np.float64)
                if aux.get("t_vec_out", None) is not None:
                    t_vec_s2b = aux["t_vec_out"].detach().float().cpu().numpy()[0].astype(np.float64)
                    pred_tmag_s2b = float(np.linalg.norm(t_vec_s2b))
                    t_dir_pred = _normalize_vec(t_vec_s2b)
                else:
                    pred_tmag_s2b = float(aux["t_mag"].detach().float().view(-1)[0].cpu()) if aux.get("t_mag", None) is not None else float("nan")
                    t_dir_pred_t = aux.get("t_dir_out", t_pred)
                    t_dir_pred = _normalize_vec(t_dir_pred_t.detach().float().cpu().numpy()[0])
                    t_vec_s2b = t_dir_pred * pred_tmag_s2b
                s5_scale = _s5_scale_from_pred(pred_tmag_s2b, s5_policy)
                t_vec_s5 = t_vec_s2b * float(s5_scale)
                tmag_s5 = float(np.linalg.norm(t_vec_s5))
                rows.append(
                    {
                        "scene_seq": str(chain.get("scene_seq", "unknown")),
                        "chain_idx": int(chain_idx),
                        "ds_idx": int(item["_ds_idx"]),
                        "i": int(item.get("i", -1)),
                        "j": int(item.get("j", -1)),
                        "k": int(item.get("k", -1)),
                        "dt_world": float(dt_val),
                        "pred_tmag_s2b": float(pred_tmag_s2b),
                        "s5_scale": float(s5_scale),
                        "s5_bucket_id": "high" if abs(s5_scale - 0.80) < 1.0e-9 else ("mid" if abs(s5_scale - 0.95) < 1.0e-9 else "id"),
                        "R_gt": R_gt.astype(np.float64).tolist(),
                        "t_gt": t_gt.astype(np.float64).tolist(),
                        "t_gt_mag": float(t_gt_mag),
                        "R_pred": R_pred_np.astype(np.float64).tolist(),
                        "t_dir_pred": t_dir_pred.astype(np.float64).tolist(),
                        "tmag_s5": float(tmag_s5),
                        "t_vec_s5": t_vec_s5.astype(np.float64).tolist(),
                    }
                )
            if rows:
                out.append({"scene_seq": str(chain.get("scene_seq", "unknown")), "chain_idx": int(chain_idx), "rows": rows})
    return out


def _rolling_stats(vals: Sequence[float], window: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(vals, dtype=np.float64)
    n = arr.size
    med = np.zeros((n,), dtype=np.float64)
    mad = np.zeros((n,), dtype=np.float64)
    half = max(int(window) // 2, 0)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = arr[lo:hi]
        m = float(np.median(chunk))
        med[i] = m
        mad_chunk = np.abs(chunk - m)
        mad[i] = float(np.median(mad_chunk))
    return med, mad


def _fit_candidate(spec: CandidateSpec, train_chains: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    fit = {"family": spec.family, **spec.params}
    if spec.family in {"regime_spike_clip", "hybrid_risk_blend"}:
        vals = np.asarray([float(r["pred_tmag_s2b"]) for chain in train_chains for r in chain["rows"]], dtype=np.float64)
        fit["pred_q90"] = float(np.quantile(vals, 0.90))
    return fit


def _compact_chain_rows(chain_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "scene_seq": str(row["scene_seq"]),
            "num_pairs": int(row["num_pairs"]),
            "renorm_factor": float(row["renorm_factor"]),
            "changed_pair_pct": float(row["changed_pair_pct"]),
            "mean_abs_tmag_change": float(row["mean_abs_tmag_change"]),
            "mean_metric_pos_err": _safe_float(row["mean_metric_pos_err"]),
            "final_metric_pos_err": _safe_float(row["final_metric_pos_err"]),
            "path_ratio": _safe_float(row["path_ratio"]),
            "gt_path_length": _safe_float(row["gt_path_length"]),
        }
        for row in chain_rows
    ]


def _compact_eval_pack(eval_pack: Dict[str, Any], *, include_chain_rows: bool = False) -> Dict[str, Any]:
    payload = {
        "metrics": dict(eval_pack["metrics"]),
        "pair_metrics": dict(eval_pack["pair_metrics"]),
        "change_metrics": dict(eval_pack["change_metrics"]),
        "selected_k": int(eval_pack["selected_k"]),
    }
    if include_chain_rows:
        payload["chain_rows"] = _compact_chain_rows(eval_pack["chain_rows"])
    return payload


def _compact_payload_for_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "baseline_gate": payload["baseline_gate"],
        "feature_audit": payload["feature_audit"],
        "baseline_cv_summary": payload.get("baseline_cv_summary"),
        "baseline_test_summary": payload.get("baseline_test_summary"),
        "candidates": payload.get("candidates", []),
        "selected": payload.get("selected"),
        "final_test": payload.get("final_test"),
        "final_classification": payload["final_classification"],
        "s10_replaces_s5": payload["s10_replaces_s5"],
        "leakage_audit": payload["leakage_audit"],
    }
    return out


def _cli_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    selected = payload.get("selected")
    final_test = payload.get("final_test")
    return {
        "final_classification": payload["final_classification"],
        "s10_replaces_s5": payload["s10_replaces_s5"],
        "baseline_gate_passed": bool(payload["baseline_gate"]["passed"]),
        "selected_candidate": None if selected is None else selected["name"],
        "final_test_run": final_test is not None,
        "candidate_count": len(payload.get("candidates", [])),
        "report_path": str(REPORT_PATH),
        "candidates_path": str(CANDIDATES_PATH),
        "policy_path": str(POLICY_PATH) if POLICY_PATH.exists() else None,
    }


def _risk_mask(rows: Sequence[Dict[str, Any]], fit: Dict[str, Any]) -> np.ndarray:
    mode = str(fit.get("regime_mode", "none"))
    out = []
    for row in rows:
        pred = float(row["pred_tmag_s2b"])
        dt_val = float(row["dt_world"])
        k_val = int(row["k"])
        pred_q90 = float(fit.get("pred_q90", float("inf")))
        dtk20 = dt_val >= 1.0 and k_val == 20
        dt_ge_1 = dt_val >= 1.0
        if mode == "pred_q90":
            out.append(pred >= pred_q90)
        elif mode == "dt_ge_1":
            out.append(dt_ge_1 or dtk20)
        elif mode == "pred_q90_or_dt_ge_1":
            out.append(pred >= pred_q90 or dt_ge_1 or dtk20)
        elif mode == "pred_q90_and_dt_ge_1":
            out.append((pred >= pred_q90 and dt_ge_1) or dtk20)
        else:
            out.append(True)
    return np.asarray(out, dtype=bool)


def _apply_candidate_to_chain(chain_rows: Sequence[Dict[str, Any]], fit: Dict[str, Any]) -> Dict[str, Any]:
    base_tmag = np.asarray([float(r["tmag_s5"]) for r in chain_rows], dtype=np.float64)
    log_tmag = np.log(np.clip(base_tmag, 1.0e-8, None))
    raw = log_tmag.copy()
    family = str(fit["family"])
    risk_mask = np.ones((len(chain_rows),), dtype=bool)
    if family in {"regime_spike_clip", "hybrid_risk_blend"}:
        risk_mask = _risk_mask(chain_rows, fit)
    if family == "median_blend":
        window = int(fit["window"])
        alpha = float(fit["alpha"])
        med, _mad = _rolling_stats(log_tmag, window)
        raw = (1.0 - alpha) * log_tmag + alpha * med
    elif family in {"spike_clip", "regime_spike_clip"}:
        window = int(fit["window"])
        mad_threshold = float(fit["mad_threshold"])
        clip_strength = float(fit["clip_strength"])
        med, mad = _rolling_stats(log_tmag, window)
        scale = np.maximum(mad * 1.4826, 1.0e-6)
        raw = log_tmag.copy()
        for i in range(len(raw)):
            if not risk_mask[i]:
                continue
            if raw[i] > med[i] + mad_threshold * scale[i]:
                raw[i] = med[i] + (raw[i] - med[i]) * (1.0 - clip_strength)
    elif family == "hybrid_risk_blend":
        window = int(fit["window"])
        alpha = float(fit["alpha"])
        mad_threshold = float(fit["mad_threshold"])
        clip_strength = float(fit["clip_strength"])
        med, mad = _rolling_stats(log_tmag, window)
        scale = np.maximum(mad * 1.4826, 1.0e-6)
        raw = log_tmag.copy()
        for i in range(len(raw)):
            if not risk_mask[i]:
                continue
            blended = (1.0 - alpha) * raw[i] + alpha * med[i]
            if raw[i] > med[i] + mad_threshold * scale[i]:
                raw[i] = med[i] + (blended - med[i]) * (1.0 - clip_strength)
            else:
                raw[i] = blended
    else:
        raise ValueError(family)

    tmag_after_raw = np.exp(raw)
    renorm_mode = str(fit.get("renorm_mode", "none"))
    renorm_factor = 1.0
    if renorm_mode == "preserve_chain_sum":
        sum_before = float(base_tmag.sum())
        sum_after = float(tmag_after_raw.sum())
        if sum_after > 1.0e-12 and math.isfinite(sum_after):
            renorm_factor = float(sum_before / sum_after)
        tmag_final = tmag_after_raw * float(renorm_factor)
    else:
        tmag_final = tmag_after_raw
    changed = np.abs(tmag_final - base_tmag) > 1.0e-6
    return {
        "tmag_before": base_tmag,
        "tmag_after_raw": tmag_after_raw,
        "tmag_final": tmag_final,
        "changed_mask": changed,
        "renorm_factor": float(renorm_factor),
        "risk_mask": risk_mask,
    }


def _evaluate_chains(chains: Sequence[Dict[str, Any]], fit: Dict[str, Any], *, label_thresholds: Dict[str, float]) -> Dict[str, Any]:
    metric_pos_err_sq: List[float] = []
    endpoint_errs: List[float] = []
    total_traj_len = 0.0
    pair_rows: List[Dict[str, Any]] = []
    chain_rows: List[Dict[str, Any]] = []
    renorm_factors: List[float] = []
    changed_pairs = 0
    total_pairs = 0
    for chain in chains:
        rows = chain["rows"]
        smooth = _apply_candidate_to_chain(rows, fit)
        renorm_factors.append(float(smooth["renorm_factor"]))
        R_gt_c0 = np.eye(3, dtype=np.float64)
        t_gt_c0 = np.zeros(3, dtype=np.float64)
        R_pred_c0 = np.eye(3, dtype=np.float64)
        t_pred_c0 = np.zeros(3, dtype=np.float64)
        dbg_rows: List[Dict[str, Any]] = []
        chain_len = 0.0
        for idx, row in enumerate(rows):
            R_gt = np.asarray(row["R_gt"], dtype=np.float64)
            t_gt = np.asarray(row["t_gt"], dtype=np.float64)
            R_pred = np.asarray(row["R_pred"], dtype=np.float64)
            t_dir_pred = np.asarray(row["t_dir_pred"], dtype=np.float64)
            tmag_before = float(smooth["tmag_before"][idx])
            tmag_after = float(smooth["tmag_final"][idx])
            t_vec_after = t_dir_pred * tmag_after
            R_gt_c0, t_gt_c0 = _compose_rel_pose_np(R_gt, t_gt, R_gt_c0, t_gt_c0)
            R_pred_c0, t_pred_c0 = _compose_rel_pose_np(R_pred, t_vec_after, R_pred_c0, t_pred_c0)
            p_gt = _camera_center_from_T_c0_np(R_gt_c0, t_gt_c0)
            p_pred = _camera_center_from_T_c0_np(R_pred_c0, t_pred_c0)
            pos_err = float(np.linalg.norm(p_pred - p_gt))
            metric_pos_err_sq.append(float(pos_err ** 2))
            pair_err = abs(tmag_after - float(row["t_gt_mag"]))
            chain_len += float(row["t_gt_mag"])
            total_pairs += 1
            if bool(smooth["changed_mask"][idx]):
                changed_pairs += 1
            rec = {
                "scene_seq": str(chain["scene_seq"]),
                "step_idx": int(idx),
                "i": int(row["i"]),
                "j": int(row["j"]),
                "k": int(row["k"]),
                "dt_world": float(row["dt_world"]),
                "pred_tmag_s2b": float(row["pred_tmag_s2b"]),
                "tmag_gt": float(row["t_gt_mag"]),
                "tmag_before": float(tmag_before),
                "tmag_after": float(tmag_after),
                "tmag_abs_change": abs(float(tmag_after - tmag_before)),
                "pair_pos_err": float(pair_err),
                "changed": bool(smooth["changed_mask"][idx]),
                "risk_triggered": bool(smooth["risk_mask"][idx]),
                "renorm_factor": float(smooth["renorm_factor"]),
                "rot_err_deg": _rot_geodesic_deg_np(R_pred, R_gt),
                "tdir_err_deg": _vec_angle_deg_np(t_dir_pred, t_gt),
                "gt_x": float(p_gt[0]),
                "gt_y": float(p_gt[1]),
                "gt_z": float(p_gt[2]),
                "metric_x": float(p_pred[0]),
                "metric_y": float(p_pred[1]),
                "metric_z": float(p_pred[2]),
                "metric_pos_err": float(pos_err),
            }
            pair_rows.append(rec)
            dbg_rows.append(dict(rec))
        p_gt_end = _camera_center_from_T_c0_np(R_gt_c0, t_gt_c0)
        p_pred_end = _camera_center_from_T_c0_np(R_pred_c0, t_pred_c0)
        endpoint_errs.append(float(np.linalg.norm(p_pred_end - p_gt_end)))
        total_traj_len += float(chain_len)
        _annotate_trajectory_shape_rows(dbg_rows, ("metric",))
        chain_summary = _odom_debug_chain_summary(dbg_rows, str(chain["scene_seq"]), segment_count=4, topk_steps=10)
        shape_metric = _trajectory_shape_summary(dbg_rows, "metric")
        chain_rows.append(
            {
                "scene_seq": str(chain["scene_seq"]),
                "num_pairs": int(len(rows)),
                "renorm_factor": float(smooth["renorm_factor"]),
                "changed_pair_pct": float(np.mean(smooth["changed_mask"])) if len(rows) else 0.0,
                "mean_abs_tmag_change": float(np.mean(np.abs(smooth["tmag_final"] - smooth["tmag_before"]))) if len(rows) else 0.0,
                "mean_metric_pos_err": _safe_float(chain_summary.get("mean_metric_pos_err")),
                "final_metric_pos_err": _safe_float(chain_summary.get("final_metric_pos_err")),
                "path_ratio": _safe_float(shape_metric.get("path_length_ratio")),
                "gt_path_length": _safe_float(shape_metric.get("gt_path_length")),
                "shape_metric": shape_metric,
            }
        )

    pair_errs = np.asarray([float(r["pair_pos_err"]) for r in pair_rows], dtype=np.float64)
    q80 = float(label_thresholds["q80"])
    regime_rows_map: Dict[str, List[float]] = {}
    baseline_regime_map: Dict[str, List[float]] = {}
    for row in pair_rows:
        bucket = _pair_bucket(float(row["dt_world"]), int(row["k"]))
        regime_rows_map.setdefault(bucket, []).append(float(row["pair_pos_err"]))
        baseline_regime_map.setdefault(bucket, []).append(float(abs(row["tmag_before"] - row["tmag_gt"])))
    regime_rows = []
    for bucket in sorted(set(regime_rows_map.keys()) | set(baseline_regime_map.keys())):
        before_arr = np.asarray(baseline_regime_map.get(bucket, []), dtype=np.float64)
        after_arr = np.asarray(regime_rows_map.get(bucket, []), dtype=np.float64)
        delta_mean = float("nan")
        if before_arr.size > 0 and after_arr.size > 0 and before_arr.size == after_arr.size:
            delta_mean = float(np.mean(after_arr - before_arr))
        regime_rows.append(
            {
                "bucket": bucket,
                "baseline_pair_pos_err": _mean(baseline_regime_map.get(bucket, [])),
                "smoothed_pair_pos_err": _mean(regime_rows_map.get(bucket, [])),
                "delta_pair_pos_err": delta_mean,
            }
        )
    worst_bucket = next((r for r in regime_rows if r["bucket"] == ">=1.0|k=20"), None)
    official_path_ratio = float(chain_rows[0]["path_ratio"]) if chain_rows else float("nan")
    return {
        "metrics": {
            "ATE": _rmse(metric_pos_err_sq),
            "drift": _mean(endpoint_errs),
            "path_ratio": official_path_ratio,
            "trajectory_length": float(total_traj_len),
            "num_pairs": int(total_pairs),
            "num_chains": int(len(chain_rows)),
        },
        "pair_metrics": {
            "mean_pair_pos_err": _mean(pair_errs),
            "worst_bucket_pair_pos_err": float("nan") if worst_bucket is None else float(worst_bucket["smoothed_pair_pos_err"]),
            "high_error_mass": _mean(float(r["pair_pos_err"]) for r in pair_rows if float(abs(r["tmag_before"] - r["tmag_gt"])) >= q80),
            "regime_rows": regime_rows,
        },
        "change_metrics": {
            "changed_pair_percentage": float(changed_pairs / max(total_pairs, 1)),
            "mean_abs_tmag_change": _mean(float(r["tmag_abs_change"]) for r in pair_rows),
            "renorm_factor_mean": _mean(renorm_factors),
            "renorm_factor_std": _std(renorm_factors),
            "renorm_factor_min": min(renorm_factors) if renorm_factors else float("nan"),
            "renorm_factor_max": max(renorm_factors) if renorm_factors else float("nan"),
            "max_chain_renorm_factor": max((abs(float(v)) for v in renorm_factors), default=float("nan")),
        },
        "chain_rows": chain_rows,
        "pair_rows": pair_rows,
        "selected_k": 1,
    }


def _label_thresholds_from_chains(chains: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    errs = np.asarray([abs(float(r["tmag_s5"]) - float(r["t_gt_mag"])) for c in chains for r in c["rows"]], dtype=np.float64)
    return {"q80": float(np.quantile(errs, 0.80)), "q90": float(np.quantile(errs, 0.90))}


def _hard_gate(candidate_eval: Dict[str, Any], baseline_eval: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    cand_m = candidate_eval["metrics"]
    base_m = baseline_eval["metrics"]
    change_m = candidate_eval["change_metrics"]
    mean_path = float(cand_m["path_ratio"])
    mean_ate = float(cand_m["ATE"])
    mean_drift = float(cand_m["drift"])
    base_mean_ate = float(base_m["ATE"])
    base_mean_drift = float(base_m["drift"])
    max_chain_renorm = float(change_m["max_chain_renorm_factor"])
    changed_pct = float(change_m["changed_pair_percentage"])
    long_chain_rows = [r for r in candidate_eval["chain_rows"] if float(r.get("gt_path_length", 0.0)) >= 1.0]
    chain_path_ok = all(SAFE_PATH_RANGE[0] <= float(r["path_ratio"]) <= SAFE_PATH_RANGE[1] for r in long_chain_rows if math.isfinite(float(r["path_ratio"])))
    ok = (
        SAFE_PATH_RANGE[0] <= mean_path <= SAFE_PATH_RANGE[1]
        and mean_ate <= base_mean_ate + 1.0e-9
        and mean_drift <= base_mean_drift + 1.0e-9
        and RENORM_SAFE_RANGE[0] <= max_chain_renorm <= RENORM_SAFE_RANGE[1]
        and chain_path_ok
        and (changed_pct <= CHANGED_PAIR_PCT_SOFT_MAX or mean_ate < base_mean_ate - 0.02)
    )
    return ok, {
        "mean_path_ratio": mean_path,
        "path_ratio_distance_to_locked_s5": abs(mean_path - S5_LOCKED["path_ratio"]),
        "baseline_mean_ATE": base_mean_ate,
        "mean_ATE": mean_ate,
        "baseline_mean_drift": base_mean_drift,
        "mean_drift": mean_drift,
        "changed_pair_percentage": changed_pct,
        "max_chain_renorm_factor": max_chain_renorm,
        "chain_path_ok": chain_path_ok,
    }


def _select_candidate(records: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    eligible = [r for r in records if bool(r["satisfies_clean_gate"])]
    if eligible:
        eligible.sort(
            key=lambda r: (
                float(r["cv_mean_ATE"]),
                float(r["cv_mean_drift"]),
                abs(float(r["cv_mean_path_ratio"]) - S5_LOCKED["path_ratio"]),
                float(r["cv_changed_pair_percentage"]),
                r["simplicity_rank"],
            )
        )
        return eligible[0]
    diagnostic = [
        r
        for r in records
        if float(r["cv_mean_pair_pos_err_delta_vs_s5"]) < -1.0e-4
        and SAFE_PATH_RANGE[0] <= float(r["cv_mean_path_ratio"]) <= 1.02
    ]
    if diagnostic:
        diagnostic.sort(
            key=lambda r: (
                float(r["cv_mean_pair_pos_err_delta_vs_s5"]),
                abs(float(r["cv_mean_path_ratio"]) - S5_LOCKED["path_ratio"]),
                r["simplicity_rank"],
            )
        )
        return diagnostic[0]
    return None


def _final_classification(selected: Dict[str, Any] | None, final_test: Dict[str, Any] | None, *, baseline_gate_ok: bool) -> Tuple[str, bool]:
    if not baseline_gate_ok:
        return "REPRODUCTION-MISMATCH", False
    if selected is None:
        return "NO-STABLE-CHAIN-SMOOTHER-GAIN", False
    if final_test is None:
        if float(selected.get("cv_mean_pair_pos_err_delta_vs_s5", 0.0)) < -1.0e-4:
            return "CHAIN-SMOOTHER-DIAGNOSTIC-GAIN", False
        return "NO-STABLE-CHAIN-SMOOTHER-GAIN", False
    replacement = (
        float(final_test["metrics"]["ATE"]) < S5_LOCKED["ATE"]
        and float(final_test["metrics"]["drift"]) <= S5_LOCKED["drift"] + 1.0e-6
        and SAFE_PATH_RANGE[0] <= float(final_test["metrics"]["path_ratio"]) <= SAFE_PATH_RANGE[1]
        and abs(float(final_test["metrics"]["path_ratio"]) - S5_LOCKED["path_ratio"]) <= 0.02
        and bool(final_test["leakage_audit"]["passed"])
        and bool(final_test["scene_chain_audit"]["passed"])
    )
    if replacement:
        return "CHAIN-SMOOTHER-CLEAN-GAIN", True
    return "CHAIN-SMOOTHER-CLEAN-BUT-MARGINAL", False


def _plot_cv_metric(rows: Sequence[Dict[str, Any]], *, metric_key: str, baseline_value: float, title: str, path: Path) -> None:
    labels = [str(r["name"]) for r in rows]
    vals = [float(r[metric_key]) for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 4), dpi=140)
    ax.bar(x, vals)
    ax.axhline(baseline_value, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_report(payload: Dict[str, Any]) -> None:
    baseline_gate = payload["baseline_gate"]
    candidates = payload["candidates"]
    selected = payload.get("selected")
    final_test = payload.get("final_test")
    feature_audit = payload["feature_audit"]
    lines: List[str] = []
    lines.append("# S10 Chain-Level Path-Ratio Preserving Smoother Report\n\n")
    lines.append("## Executive summary\n\n")
    lines.append(f"- final classification: `{payload['final_classification']}`\n")
    lines.append(f"- S10 replaces S5: `{payload['s10_replaces_s5']}`\n")
    lines.append(f"- selected candidate: `{selected['name'] if selected is not None else 'None'}`\n")
    lines.append(f"- official odom scope selected_k: `1`\n\n")

    lines.append("## Motivation\n\n")
    lines.append("- S8 ruled out added reliability value from token features.\n")
    lines.append("- S9 suggested that local fallback-like behaviors can show diagnostic signal but tend to disturb global path ratio.\n")
    lines.append("- S10 therefore moves from pair-level routing to chain-level tmag smoothing with explicit chain-sum preservation.\n\n")

    lines.append("## Baseline gate using S8b contract\n\n")
    lines.append(f"- gate passed: `{baseline_gate['passed']}`\n")
    lines.append(f"- contract: `{baseline_gate['contract_path']}`\n")
    lines.append(f"- locked S5 metrics preserved: drift=`{_fmt(float(baseline_gate['s5']['metrics']['drift']))}`, ATE=`{_fmt(float(baseline_gate['s5']['metrics']['ATE']))}`, path_ratio=`{_fmt(float(baseline_gate['s5']['metrics']['path_ratio']))}`\n\n")

    lines.append("## Inference-visible inputs\n\n")
    lines.append("| feature | inference-visible | allowed |\n")
    lines.append("| --- | --- | --- |\n")
    for row in feature_audit:
        lines.append(f"| {row['feature']} | {row['inference_visible']} | {row['allowed']} |\n")
    lines.append("\n")

    lines.append("## Candidate families\n\n")
    lines.append("- local log-tmag rolling median blend\n")
    lines.append("- rolling MAD spike clipping\n")
    lines.append("- regime-aware spike clipping\n")
    lines.append("- conservative hybrid blend+clip\n")
    lines.append("- renorm mode includes `preserve_chain_sum` and a few `none` controls\n")
    lines.append("- note: official odometry clean scope uses `selected_k=1`, so `k==20` trigger paths remain diagnostic-only and are effectively inactive under final odom evaluation.\n\n")

    lines.append("## Train-CV selection protocol\n\n")
    lines.append("- folds: `scene01/seq01` and `scene01/seq02`\n")
    lines.append("- no model training; only chain-level inference-time post-processing of S5 pair outputs\n")
    lines.append("- train-only quantiles are used for regime-aware triggers\n")
    lines.append("- final test is allowed only for a train-CV clean-gate candidate\n\n")

    lines.append("## CV results table\n\n")
    lines.append("| candidate | family | cv_mean_ATE | cv_mean_drift | cv_mean_path | pair_pos_err | changed_pct | max_renorm | clean_gate |\n")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
    for row in candidates:
        lines.append(
            f"| {row['name']} | {row['family']} | {_fmt(row['cv_mean_ATE'])} | {_fmt(row['cv_mean_drift'])} | {_fmt(row['cv_mean_path_ratio'])} | "
            f"{_fmt(row['cv_mean_pair_pos_err'])} | {_fmt(row['cv_changed_pair_percentage'])} | {_fmt(row['cv_max_chain_renorm_factor'])} | {row['satisfies_clean_gate']} |\n"
        )
    lines.append("\n")

    lines.append("## Hard-gate audit\n\n")
    for row in candidates:
        gp = row["gate_payload"]
        lines.append(
            f"- `{row['name']}`: mean_ATE `{_fmt(gp['mean_ATE'])}` vs baseline `{_fmt(gp['baseline_mean_ATE'])}`, "
            f"mean_drift `{_fmt(gp['mean_drift'])}` vs baseline `{_fmt(gp['baseline_mean_drift'])}`, "
            f"mean_path `{_fmt(gp['mean_path_ratio'])}`, dist_to_locked_s5 `{_fmt(gp['path_ratio_distance_to_locked_s5'])}`, "
            f"changed_pct `{_fmt(gp['changed_pair_percentage'])}`, max_renorm `{_fmt(gp['max_chain_renorm_factor'])}`, gate=`{row['satisfies_clean_gate']}`\n"
        )
    lines.append("\n")

    if final_test is not None:
        lines.append("## Final test result, if selected\n\n")
        lines.append(f"- candidate: `{final_test['candidate_name']}`\n")
        lines.append(f"- drift = `{_fmt(final_test['metrics']['drift'])}`\n")
        lines.append(f"- ATE = `{_fmt(final_test['metrics']['ATE'])}`\n")
        lines.append(f"- path_ratio = `{_fmt(final_test['metrics']['path_ratio'])}`\n")
        lines.append(f"- mean_pair_pos_err = `{_fmt(final_test['pair_metrics']['mean_pair_pos_err'])}`\n")
        lines.append(f"- worst_bucket_pair_pos_err = `{_fmt(final_test['pair_metrics']['worst_bucket_pair_pos_err'])}`\n")
        lines.append(f"- high_error_mass = `{_fmt(final_test['pair_metrics']['high_error_mass'])}`\n")
        lines.append(f"- changed_pair_percentage = `{_fmt(final_test['change_metrics']['changed_pair_percentage'])}`\n")
        lines.append(f"- renorm_factor_mean/std = `{_fmt(final_test['change_metrics']['renorm_factor_mean'])}` / `{_fmt(final_test['change_metrics']['renorm_factor_std'])}`\n\n")
    else:
        lines.append("## Final test result, if selected\n\n")
        lines.append("- no final test was run because no candidate passed the train-CV clean gate.\n\n")

    lines.append("## Regime-level before/after\n\n")
    if final_test is not None:
        for row in final_test["pair_metrics"]["regime_rows"]:
            lines.append(
                f"- `{row['bucket']}`: baseline=`{_fmt(row['baseline_pair_pos_err'])}` -> smoothed=`{_fmt(row['smoothed_pair_pos_err'])}`, delta=`{_fmt(row['delta_pair_pos_err'])}`\n"
            )
    else:
        lines.append("- not available because no final test candidate was run.\n")
    lines.append("\n")

    lines.append("## Scene/chain-level before/after\n\n")
    if final_test is not None:
        for row in final_test["chain_rows"]:
            lines.append(
                f"- `{row['scene_seq']}`: mean_metric_pos_err=`{_fmt(row['mean_metric_pos_err'])}`, final_metric_pos_err=`{_fmt(row['final_metric_pos_err'])}`, path_ratio=`{_fmt(row['path_ratio'])}`, renorm=`{_fmt(row['renorm_factor'])}`\n"
            )
    else:
        lines.append("- not available because no final test candidate was run.\n")
    lines.append("\n")

    lines.append("## Leakage audit\n\n")
    for k, v in payload["leakage_audit"].items():
        lines.append(f"- {k}: `{v}`\n")
    lines.append("\n")

    lines.append("## Final classification\n\n")
    lines.append(f"- `{payload['final_classification']}`\n")
    lines.append(f"- S5 remains final clean candidate: `{not payload['s10_replaces_s5']}`\n")
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def _write_summary(payload: Dict[str, Any]) -> None:
    selected = payload.get("selected")
    final_test = payload.get("final_test")
    lines: List[str] = []
    lines.append("# Final S10 Chain Smoother Summary\n\n")
    lines.append(f"- Final classification: `{payload['final_classification']}`\n")
    lines.append(f"- S10 replaces S5: `{payload['s10_replaces_s5']}`\n")
    lines.append(f"- S5 remains final clean candidate: `{not payload['s10_replaces_s5']}`\n")
    lines.append(f"- Selected candidate: `{selected['name'] if selected is not None else 'None'}`\n\n")
    lines.append("S10 tested chain-level tmag smoothing on top of S5 while keeping rotation and translation direction unchanged. ")
    lines.append("Its main purpose was to preserve path ratio explicitly through per-chain renormalization, addressing the failure mode observed in S9.\n\n")
    if final_test is not None:
        lines.append(
            f"The selected smoother produced final-test drift/ATE/path_ratio = `{_fmt(final_test['metrics']['drift'])}` / "
            f"`{_fmt(final_test['metrics']['ATE'])}` / `{_fmt(final_test['metrics']['path_ratio'])}` against the locked S5 baseline "
            f"`{S5_LOCKED['drift']}` / `{S5_LOCKED['ATE']}` / `{S5_LOCKED['path_ratio']}`.\n\n"
        )
    else:
        lines.append("No candidate passed the clean train-CV gate, so S10 remains a diagnostic/future-work result rather than a new final candidate.\n\n")
    lines.append("Interpretation: if S10 still fails, then the remaining clean improvement space after S5 is not easily unlocked by lightweight chain-level tmag smoothing alone.\n")
    SUMMARY_PATH.write_text("".join(lines), encoding="utf-8")


def run_main() -> Dict[str, Any]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    baseline_gate = _baseline_gate()
    payload: Dict[str, Any] = {
        "baseline_gate": baseline_gate,
        "feature_audit": _feature_audit(),
        "final_classification": "REPRODUCTION-MISMATCH" if not baseline_gate["passed"] else "INCONCLUSIVE",
        "s10_replaces_s5": False,
        "leakage_audit": {
            "passed": False if not baseline_gate["passed"] else True,
            "token_features_used": False,
            "gt_tmag_as_inference_feature": False,
            "gt_pose_as_inference_feature": False,
            "test_used_for_selection": False,
            "train_cv_selection_only": True,
            "forbidden_feature_used": False,
            "current_arch_contract_match": bool(baseline_gate["passed"]),
        },
    }
    if not baseline_gate["passed"]:
        payload["candidates"] = []
        payload["selected"] = None
        payload["final_test"] = None
        payload["baseline_cv_summary"] = None
        payload["baseline_test_summary"] = None
        _write_json(CANDIDATES_PATH, _compact_payload_for_json(payload))
        _write_report(payload)
        _write_summary(payload)
        return payload

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s2b_model, cfg, _s2b_policy, load_summary = _build_s2b_model(device)
    s5_policy = _load_s5_policy()
    train_ds_full, groups, seq_to_indices = _collect_train_groups(cfg)
    test_ds = _build_dataset(cfg, split="test")

    train_chains_full = _build_chain_records(s2b_model, train_ds_full, s5_policy, device)
    test_chains = _build_chain_records(s2b_model, test_ds, s5_policy, device)
    train_thresholds_full = _label_thresholds_from_chains(train_chains_full)

    fold_plan = []
    for fold_idx, seq in enumerate(groups):
        val_idx = seq_to_indices[seq]
        train_idx = [i for g, idxs in seq_to_indices.items() if g != seq for i in idxs]
        fold_plan.append(
            {
                "fold_name": f"fold_{fold_idx}_{seq[0]}_{seq[1]}",
                "train_ds": torch.utils.data.Subset(train_ds_full, train_idx),
                "val_ds": torch.utils.data.Subset(train_ds_full, val_idx),
            }
        )

    fold_train_chains: Dict[str, List[Dict[str, Any]]] = {}
    fold_val_chains: Dict[str, List[Dict[str, Any]]] = {}
    fold_thresholds: Dict[str, Dict[str, float]] = {}
    for fold in fold_plan:
        train_subset = _SubsetWithManifest(train_ds_full, fold["train_ds"].indices)
        val_subset = _SubsetWithManifest(train_ds_full, fold["val_ds"].indices)
        tr_chains = _build_chain_records(s2b_model, train_subset, s5_policy, device)
        va_chains = _build_chain_records(s2b_model, val_subset, s5_policy, device)
        fold_train_chains[fold["fold_name"]] = tr_chains
        fold_val_chains[fold["fold_name"]] = va_chains
        fold_thresholds[fold["fold_name"]] = _label_thresholds_from_chains(tr_chains)

    baseline_cv_rows = []
    for fold in fold_plan:
        fold_name = fold["fold_name"]
        eval_pack = _evaluate_chains(fold_val_chains[fold_name], {"family": "median_blend", "window": 3, "alpha": 0.0, "renorm_mode": "none"}, label_thresholds=fold_thresholds[fold_name])
        baseline_cv_rows.append({"fold": fold_name, **_compact_eval_pack(eval_pack, include_chain_rows=True)})
    baseline_test = _evaluate_chains(test_chains, {"family": "median_blend", "window": 3, "alpha": 0.0, "renorm_mode": "none"}, label_thresholds=train_thresholds_full)

    candidates: List[Dict[str, Any]] = []
    for spec in _candidate_specs():
        print(f"[S10] evaluating {spec.name}", flush=True)
        fold_rows = []
        fit_payloads = []
        for fold in fold_plan:
            fold_name = fold["fold_name"]
            fit = _fit_candidate(spec, fold_train_chains[fold_name])
            fit_payloads.append({"fold": fold_name, "fit_payload": fit})
            eval_pack = _evaluate_chains(fold_val_chains[fold_name], fit, label_thresholds=fold_thresholds[fold_name])
            fold_rows.append({"fold": fold_name, **_compact_eval_pack(eval_pack, include_chain_rows=True)})
        cv_mean_ate = _mean(float(r["metrics"]["ATE"]) for r in fold_rows)
        cv_mean_drift = _mean(float(r["metrics"]["drift"]) for r in fold_rows)
        cv_mean_path = _mean(float(r["metrics"]["path_ratio"]) for r in fold_rows)
        cv_mean_pair = _mean(float(r["pair_metrics"]["mean_pair_pos_err"]) for r in fold_rows)
        cv_mean_worst = _mean(float(r["pair_metrics"]["worst_bucket_pair_pos_err"]) for r in fold_rows)
        cv_changed_pct = _mean(float(r["change_metrics"]["changed_pair_percentage"]) for r in fold_rows)
        cv_mean_abs_change = _mean(float(r["change_metrics"]["mean_abs_tmag_change"]) for r in fold_rows)
        cv_max_renorm = max(float(r["change_metrics"]["max_chain_renorm_factor"]) for r in fold_rows)
        cv_pair_delta = _mean(
            float(r["pair_metrics"]["mean_pair_pos_err"]) - float(b["pair_metrics"]["mean_pair_pos_err"])
            for r, b in zip(fold_rows, baseline_cv_rows)
        )
        gate_ok, gate_payload = _hard_gate(
            {
                "metrics": {"ATE": cv_mean_ate, "drift": cv_mean_drift, "path_ratio": cv_mean_path},
                "change_metrics": {"changed_pair_percentage": cv_changed_pct, "max_chain_renorm_factor": cv_max_renorm},
                "chain_rows": [cr for r in fold_rows for cr in r["chain_rows"]],
            },
            {
                "metrics": {
                    "ATE": _mean(float(r["metrics"]["ATE"]) for r in baseline_cv_rows),
                    "drift": _mean(float(r["metrics"]["drift"]) for r in baseline_cv_rows),
                    "path_ratio": _mean(float(r["metrics"]["path_ratio"]) for r in baseline_cv_rows),
                }
            },
        )
        candidates.append(
            {
                "name": spec.name,
                "family": spec.family,
                "params": spec.params,
                "simplicity_rank": spec.simplicity_rank,
                "fit_payloads": fit_payloads,
                "fit_summary": fit_payloads[-1]["fit_payload"] if fit_payloads else {},
                "fold_rows": fold_rows,
                "cv_mean_ATE": cv_mean_ate,
                "cv_mean_drift": cv_mean_drift,
                "cv_mean_path_ratio": cv_mean_path,
                "cv_mean_pair_pos_err": cv_mean_pair,
                "cv_mean_worst_bucket_pair_pos_err": cv_mean_worst,
                "cv_changed_pair_percentage": cv_changed_pct,
                "cv_mean_abs_tmag_change": cv_mean_abs_change,
                "cv_max_chain_renorm_factor": cv_max_renorm,
                "cv_mean_pair_pos_err_delta_vs_s5": cv_pair_delta,
                "satisfies_clean_gate": gate_ok,
                "gate_payload": gate_payload,
            }
        )

    candidates.sort(key=lambda r: (float(r["cv_mean_ATE"]), float(r["cv_mean_drift"]), r["simplicity_rank"]))
    baseline_cv_ate = _mean(float(r["metrics"]["ATE"]) for r in baseline_cv_rows)
    baseline_cv_path = _mean(float(r["metrics"]["path_ratio"]) for r in baseline_cv_rows)
    baseline_cv_pair = _mean(float(r["pair_metrics"]["mean_pair_pos_err"]) for r in baseline_cv_rows)
    _plot_cv_metric(candidates, metric_key="cv_mean_ATE", baseline_value=baseline_cv_ate, title="S10 Mean CV ATE by Candidate", path=FIG_DIR / "cv_mean_ate.png")
    _plot_cv_metric(candidates, metric_key="cv_mean_path_ratio", baseline_value=baseline_cv_path, title="S10 Mean CV Path Ratio by Candidate", path=FIG_DIR / "cv_mean_path_ratio.png")
    _plot_cv_metric(candidates, metric_key="cv_mean_pair_pos_err", baseline_value=baseline_cv_pair, title="S10 Mean CV Pair Error by Candidate", path=FIG_DIR / "cv_mean_pair_pos_err.png")

    selected = _select_candidate(candidates)
    final_test = None
    if selected is not None and bool(selected["satisfies_clean_gate"]):
        spec = next(s for s in _candidate_specs() if s.name == selected["name"])
        fit = _fit_candidate(spec, train_chains_full)
        final_eval = _evaluate_chains(test_chains, fit, label_thresholds=train_thresholds_full)
        scene_chain_audit = {
            "passed": all(
                SAFE_PATH_RANGE[0] <= float(r["path_ratio"]) <= SAFE_PATH_RANGE[1]
                and float(r["renorm_factor"]) >= RENORM_SAFE_RANGE[0]
                and float(r["renorm_factor"]) <= RENORM_SAFE_RANGE[1]
                for r in final_eval["chain_rows"]
                if math.isfinite(float(r["path_ratio"]))
            )
        }
        final_test = {
            "candidate_name": spec.name,
            "family": spec.family,
            "fit_payload": fit,
            "metrics": final_eval["metrics"],
            "pair_metrics": final_eval["pair_metrics"],
            "change_metrics": final_eval["change_metrics"],
            "chain_rows": _compact_chain_rows(final_eval["chain_rows"]),
            "scene_chain_audit": scene_chain_audit,
            "leakage_audit": payload["leakage_audit"],
        }
        policy_payload = {
            "name": "S10_chain_level_path_ratio_preserving_smoother_policy",
            "base_s5_policy_path": str(S5_POLICY_PATH.relative_to(REPO_ROOT)),
            "baseline_contract_path": str(S8B_CONTRACT_PATH.relative_to(REPO_ROOT)),
            "candidate_name": spec.name,
            "family": spec.family,
            "fit_payload": fit,
            "load_missing": 14,
            "load_unexpected": 0,
            "eval_scope": {
                "selected_k": 1,
                "num_pairs": int(final_eval["metrics"]["num_pairs"]),
                "num_chains": int(final_eval["metrics"]["num_chains"]),
                "path_ratio_safe_range": list(SAFE_PATH_RANGE),
            },
            "changed_pair_percentage": final_eval["change_metrics"]["changed_pair_percentage"],
            "renorm_factor_stats": {
                "mean": final_eval["change_metrics"]["renorm_factor_mean"],
                "std": final_eval["change_metrics"]["renorm_factor_std"],
                "min": final_eval["change_metrics"]["renorm_factor_min"],
                "max": final_eval["change_metrics"]["renorm_factor_max"],
            },
            "final_test_metrics": final_eval["metrics"],
        }
        _write_json(POLICY_PATH, policy_payload)
    else:
        if POLICY_PATH.exists():
            POLICY_PATH.unlink()

    final_classification, replaces = _final_classification(selected, final_test, baseline_gate_ok=bool(baseline_gate["passed"]))
    payload.update(
        {
            "baseline_cv_summary": {
                "fold_rows": baseline_cv_rows,
                "mean_metrics": {
                    "ATE": baseline_cv_ate,
                    "drift": _mean(float(r["metrics"]["drift"]) for r in baseline_cv_rows),
                    "path_ratio": baseline_cv_path,
                    "mean_pair_pos_err": baseline_cv_pair,
                },
            },
            "baseline_test_summary": _compact_eval_pack(baseline_test, include_chain_rows=True),
            "candidates": candidates,
            "selected": selected,
            "final_test": final_test,
            "final_classification": final_classification,
            "s10_replaces_s5": replaces,
        }
    )
    _write_json(CANDIDATES_PATH, _compact_payload_for_json(payload))
    _write_report(payload)
    _write_summary(payload)
    return payload


class _SubsetWithManifest(torch.utils.data.Dataset):
    def __init__(self, base_ds, indices: Sequence[int]) -> None:
        self.base_ds = base_ds
        self.indices = [int(i) for i in indices]
        self._manifest = [base_ds.manifest()[i] for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base_ds[self.indices[idx]]

    def manifest(self):
        return list(self._manifest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "eval_policy"], default="run")
    ap.add_argument("--policy", default=str(POLICY_PATH))
    args = ap.parse_args()
    if args.mode == "eval_policy":
        policy = _read_json(Path(args.policy))
        payload = {
            "policy_path": str(Path(args.policy)),
            "base_s5_policy_path": policy.get("base_s5_policy_path"),
            "candidate_name": policy.get("candidate_name"),
            "family": policy.get("family"),
            "window": policy.get("fit_payload", {}).get("window"),
            "mad_threshold": policy.get("fit_payload", {}).get("mad_threshold"),
            "alpha": policy.get("fit_payload", {}).get("alpha"),
            "clip_strength": policy.get("fit_payload", {}).get("clip_strength"),
            "renorm_mode": policy.get("fit_payload", {}).get("renorm_mode"),
            "load_missing": policy.get("load_missing"),
            "load_unexpected": policy.get("load_unexpected"),
            "eval_scope": policy.get("eval_scope"),
            "changed_pair_percentage": policy.get("changed_pair_percentage"),
            "renorm_factor_stats": policy.get("renorm_factor_stats"),
            "final_test_metrics": policy.get("final_test_metrics"),
        }
        print(json.dumps(_json_safe(payload), indent=2))
        return
    payload = run_main()
    print(json.dumps(_json_safe(_cli_summary(payload)), indent=2))


if __name__ == "__main__":
    main()
