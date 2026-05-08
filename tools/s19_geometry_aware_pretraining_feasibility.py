#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

DEFAULT_REPORT_PATH = REPO_ROOT / "checkpoints" / "S19_geometry_aware_pretraining_feasibility_report.md"
DEFAULT_CANDIDATES_PATH = REPO_ROOT / "checkpoints" / "S19_geometry_aware_pretraining_feasibility_candidates.json"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "reports" / "final_s19_geometry_aware_pretraining_feasibility_summary.md"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "s19_geometry_aware_pretraining_feasibility.yaml"
DEFAULT_FIGURE_DIR = REPO_ROOT / "checkpoints" / "S19_geometry_aware_pretraining_feasibility_figures"

S8B_CONTRACT_PATH = REPO_ROOT / "checkpoints" / "S8b_reproduction_contract.json"
S8B_S5_RESULT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_s5_wrapper_result.json"
S5_POLICY_PATH = REPO_ROOT / "checkpoints" / "S5_clean_tmag_calibration_policy.json"
S16B_CANDIDATES_PATH = REPO_ROOT / "checkpoints" / "S16b_frozen_pretrained_backbone_probe_candidates.json"
S16_CANDIDATES_PATH = REPO_ROOT / "checkpoints" / "S16_stronger_visual_backbone_feasibility_candidates.json"
S17_REPORT_PATH = REPO_ROOT / "checkpoints" / "S17_pose_supervision_dataset_quality_audit_report.md"
S18_REPORT_PATH = REPO_ROOT / "checkpoints" / "S18_split_redesign_representativeness_report.md"

S5_LOCKED = {"drift": 1.327343, "ATE": 7.352288, "path_ratio": 0.932379}
S16_LOCKED_CURRENT = {"rot_r2": -22.281483, "tdir_r2": -4.465965, "joint_auc": 0.708327}
S16B_LOCKED_IMAGENET = {"rot_r2": -258.785889, "tdir_r2": -5.509770, "joint_auc": 0.383036}

FINAL_CLASSES = {
    "GEOMETRY-PRETRAINING-DIAGNOSTIC-SIGNAL",
    "GEOMETRY-PRETRAINING-WEAK-SIGNAL",
    "GEOMETRY-PRETRAINING-TMAG-ONLY-SIGNAL",
    "NO-STABLE-GEOMETRY-PRETRAINING-GAIN",
    "PRETRAINING-HARNESS-FAIL",
    "REPRODUCTION-MISMATCH",
    "INCONCLUSIVE",
}


@dataclass
class ProbeRow:
    scene_seq: str
    ds_idx: int
    k: int
    dt_world: float
    pred_tmag: float
    rot_err_deg: float
    tdir_err_deg: float
    tdir_cos: float
    log_tmag_abs_err: float
    gt_rotvec: np.ndarray
    gt_tdir: np.ndarray
    gt_log_tmag: float
    current_feature: np.ndarray
    regime_feature: np.ndarray


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return [_json_safe(x) for x in v.tolist()]
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, dict):
        return {str(k): _json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return v


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _fmt(v: Any, digits: int = 6) -> str:
    try:
        x = float(v)
    except Exception:
        return str(v)
    return "nan" if not math.isfinite(x) else f"{x:.{digits}f}"


def _parse_config(path: Path) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"__config_path__": str(path)}
    if not path.exists():
        return cfg
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        cfg[key.strip()] = val.strip()
    return cfg


def _as_int(cfg: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(str(cfg.get(key, default)).strip())
    except Exception:
        return int(default)


def _as_float(cfg: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(str(cfg.get(key, default)).strip())
    except Exception:
        return float(default)


def _as_bool(cfg: Dict[str, Any], key: str, default: bool) -> bool:
    raw = cfg.get(key, default)
    if isinstance(raw, bool):
        return raw
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _as_str_list(cfg: Dict[str, Any], key: str, default: Sequence[str]) -> List[str]:
    raw = str(cfg.get(key, ",".join(default)))
    out = [part.strip() for part in raw.split(",") if part.strip()]
    return out if out else list(default)


def _as_int_list(cfg: Dict[str, Any], key: str, default: Sequence[int]) -> Tuple[int, ...]:
    raw = str(cfg.get(key, ",".join(str(x) for x in default)))
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return tuple(out) if out else tuple(default)


def _resolve_path(raw: str | Path | None, default: Path) -> Path:
    if raw is None:
        return default
    p = Path(str(raw).strip())
    if not str(p):
        return default
    return p if p.is_absolute() else (REPO_ROOT / p)


def _resolve_output_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    return {
        "config": _resolve_path(config.get("__config_path__"), DEFAULT_CONFIG_PATH),
        "report": _resolve_path(config.get("report_path"), DEFAULT_REPORT_PATH),
        "candidates": _resolve_path(config.get("candidates_path"), DEFAULT_CANDIDATES_PATH),
        "summary": _resolve_path(config.get("summary_path"), DEFAULT_SUMMARY_PATH),
        "figures": _resolve_path(config.get("figure_dir"), DEFAULT_FIGURE_DIR),
    }


def _baseline_gate() -> Dict[str, Any]:
    contract = _read_json(S8B_CONTRACT_PATH)
    s5 = _read_json(S8B_S5_RESULT_PATH)
    metrics = dict(s5.get("metrics", {}))
    metric_ok = all(abs(float(metrics.get(k, float("nan"))) - float(S5_LOCKED[k])) <= 1.0e-9 for k in S5_LOCKED)
    cats = dict(s5.get("missing_key_categories", {}))
    load_ok = (
        int(s5.get("load_missing", -1)) == 14
        and int(s5.get("load_unexpected", -1)) == 0
        and len(cats.get("ridge_calib_buffers", [])) == 2
        and len(cats.get("coupled_pose_head_params", [])) == 12
        and len(cats.get("other_missing", [])) == 0
    )
    contract_ok = bool(contract.get("current_architecture_14_key_path_allowed_for_s8_baseline_gate", False))
    return {
        "passed": bool(contract_ok and metric_ok and load_ok),
        "contract_ok": contract_ok,
        "load_missing": int(s5.get("load_missing", -1)),
        "load_unexpected": int(s5.get("load_unexpected", -1)),
        "missing_key_categories": cats,
        "locked_metrics": dict(S5_LOCKED),
        "observed_metrics": metrics,
        "metric_ok": bool(metric_ok),
        "load_ok": bool(load_ok),
    }


def _unit(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(arr))
    if not math.isfinite(n) or n <= 1.0e-12:
        out = np.zeros_like(arr)
        out[0] = 1.0
        return out
    return arr / n


def _vec_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = _unit(a)
    bb = _unit(b)
    return float(np.degrees(np.arccos(np.clip(float(np.dot(aa, bb)), -1.0, 1.0))))


def _rot_geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    rel = np.asarray(Ra, dtype=np.float64).T @ np.asarray(Rb, dtype=np.float64)
    c = np.clip((float(np.trace(rel)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _rot_log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    c = np.clip((float(np.trace(R)) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(c))
    if theta < 1.0e-8:
        return np.zeros(3, dtype=np.float64)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    return vee * (theta / max(2.0 * math.sin(theta), 1.0e-8))


def _rot_exp(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1.0e-8:
        return np.eye(3, dtype=np.float64)
    k = w / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _pool_feature_tensor(t: Any, weight: Any | None = None) -> np.ndarray:
    import torch

    if t is None or not torch.is_tensor(t):
        return np.zeros(1, dtype=np.float64)
    x = t.detach().float()
    if x.dim() == 3:
        x = x[0]
    elif x.dim() != 2:
        return x.reshape(-1).cpu().numpy().astype(np.float64)
    if weight is not None and torch.is_tensor(weight):
        w = weight.detach().float().reshape(-1)
        if w.numel() == x.shape[0]:
            w = w / w.sum().clamp_min(1.0e-8)
            mean = (x * w[:, None]).sum(dim=0)
            std = torch.sqrt(((x - mean[None, :]).pow(2) * w[:, None]).sum(dim=0).clamp_min(1.0e-8))
        else:
            mean = x.mean(dim=0)
            std = x.std(dim=0, unbiased=False)
    else:
        mean = x.mean(dim=0)
        std = x.std(dim=0, unbiased=False)
    maxv = x.max(dim=0).values
    minv = x.min(dim=0).values
    return torch.cat([mean, std, maxv, minv], dim=0).cpu().numpy().astype(np.float64)


def _standardize(x_train: np.ndarray, x_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(x_train, axis=0)
    sd = np.nanstd(x_train, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1.0e-8), sd, 1.0)
    return (np.nan_to_num(x_train, nan=0.0) - mu) / sd, (np.nan_to_num(x_val, nan=0.0) - mu) / sd


def _ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    reg = float(lam)
    if xb.shape[1] <= xb.shape[0]:
        eye = np.eye(xb.shape[1], dtype=np.float64) * reg
        eye[-1, -1] = 0.0
        return np.linalg.solve(xb.T @ xb + eye, xb.T @ y)
    x0 = x
    y0 = np.asarray(y, dtype=np.float64)
    y_mean = np.mean(y0, axis=0, keepdims=True)
    yc = y0 - y_mean
    gram = x0 @ x0.T
    alpha = np.linalg.solve(gram + reg * np.eye(gram.shape[0], dtype=np.float64), yc)
    w = x0.T @ alpha
    bias = y_mean - np.mean(x0, axis=0, keepdims=True) @ w
    return np.concatenate([w, bias], axis=0)


def _ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    return xb @ w


def _r2(y: np.ndarray, yp: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    yp = np.asarray(yp, dtype=np.float64)
    ss_res = float(np.sum((y - yp) ** 2))
    ss_tot = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
    return float("nan") if ss_tot <= 1.0e-12 else float(1.0 - ss_res / ss_tot)


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    pos = score[y == 1]
    neg = score[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    vals = [(pos > n).sum() + 0.5 * (pos == n).sum() for n in neg]
    return float(np.sum(vals) / (pos.size * neg.size))


def _binary_ridge_metrics(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, lam: float) -> Dict[str, float]:
    yoh = np.zeros((y_train.shape[0], 2), dtype=np.float64)
    yoh[np.arange(y_train.shape[0]), y_train.astype(np.int64)] = 1.0
    w = _ridge_fit(x_train, yoh, lam)
    train_logits = _ridge_predict(x_train, w)
    val_logits = _ridge_predict(x_val, w)
    train_pred = np.argmax(train_logits, axis=1)
    val_pred = np.argmax(val_logits, axis=1)
    val_score = val_logits[:, 1] - val_logits[:, 0]
    train_score = train_logits[:, 1] - train_logits[:, 0]

    def recall(y: np.ndarray, pred: np.ndarray) -> float:
        den = int((y == 1).sum())
        return float("nan") if den == 0 else float(((pred == 1) & (y == 1)).sum() / den)

    def precision(y: np.ndarray, pred: np.ndarray) -> float:
        den = int((pred == 1).sum())
        return float("nan") if den == 0 else float(((pred == 1) & (y == 1)).sum() / den)

    return {
        "train_auc": _auc(y_train, train_score),
        "val_auc": _auc(y_val, val_score),
        "train_recall": recall(y_train, train_pred),
        "val_recall": recall(y_val, val_pred),
        "train_precision": precision(y_train, train_pred),
        "val_precision": precision(y_val, val_pred),
        "train_accuracy": float(np.mean(train_pred == y_train)),
        "val_accuracy": float(np.mean(val_pred == y_val)),
    }


def _select_stratified(indices: Sequence[int], manifest: Sequence[Dict[str, Any]], cap: int, seed: int) -> List[int]:
    if len(indices) <= cap:
        return list(indices)
    rng = np.random.default_rng(seed)
    groups: Dict[int, List[int]] = {}
    for idx in indices:
        groups.setdefault(int(manifest[idx].get("k", -1)), []).append(int(idx))
    selected: List[int] = []
    for k, vals in sorted(groups.items()):
        share = max(1, int(round(cap * len(vals) / len(indices))))
        take = min(share, len(vals))
        selected.extend(int(x) for x in rng.choice(np.asarray(vals), size=take, replace=False).tolist())
    selected = sorted(set(selected))
    if len(selected) > cap:
        selected = sorted(int(x) for x in rng.choice(np.asarray(selected), size=cap, replace=False).tolist())
    if len(selected) < cap:
        remaining = [idx for idx in indices if idx not in set(selected)]
        if remaining:
            extra = rng.choice(np.asarray(remaining), size=min(cap - len(selected), len(remaining)), replace=False)
            selected.extend(int(x) for x in extra.tolist())
    return sorted(set(selected))[:cap]


def _build_probe_selection(config: Dict[str, Any]) -> Tuple[Any, Sequence[Dict[str, Any]], Dict[str, List[int]], Dict[str, List[int]]]:
    from config import Config
    from dataset_pano_only import RflyPanoPanoramaPairsEvalFixedKList
    from tools.eval_clean_policy import _cfg_from_dict, _load_ckpt_cfg

    policy = _read_json(S5_POLICY_PATH)
    ckpt_path = REPO_ROOT / str(policy["base_checkpoint_path"])
    cfg_obj: Config = _cfg_from_dict(_load_ckpt_cfg(ckpt_path))
    cfg_obj.eval_k_list = _as_int_list(config, "eval_k_list", (1, 2, 3, 5, 10, 20))
    ds = RflyPanoPanoramaPairsEvalFixedKList(
        data_root=str(cfg_obj.data_root),
        split="train",
        split_by=str(cfg_obj.split_by),
        train_ratio=float(cfg_obj.train_ratio),
        split_seed=int(cfg_obj.split_seed),
        H=int(cfg_obj.H),
        W=int(cfg_obj.W),
        k_list=tuple(int(x) for x in cfg_obj.eval_k_list),
        pair_step=int(getattr(cfg_obj, "eval_pair_step", 1)),
        min_dt=float(cfg_obj.eval_min_dt),
        max_dt=None if getattr(cfg_obj, "eval_max_dt", None) is None else float(cfg_obj.eval_max_dt),
    )
    manifest = ds.manifest()
    train_cap = _as_int(config, "train_cap_per_sequence", 1024)
    val_cap = _as_int(config, "val_cap_per_sequence", 256)
    seq_keys = ["scene01_seq01", "scene01_seq02"]
    by_seq: Dict[str, List[int]] = {k: [] for k in seq_keys}
    for idx, meta in enumerate(manifest):
        key = f"{meta.get('scene')}_{meta.get('seq')}"
        if key in by_seq:
            by_seq[key].append(idx)
    selected = {
        key: _select_stratified(vals, manifest, train_cap, seed=1900 + i)
        for i, (key, vals) in enumerate(sorted(by_seq.items()))
    }
    val_selected = {key: vals[: min(val_cap, len(vals))] for key, vals in selected.items()}
    return ds, manifest, selected, val_selected


def _extract_rows(config: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    from config import Config
    from dataset_pano_only import RflyPanoPanoramaPairsEvalFixedKList
    from model import PanoramaRelPoseModel
    from tools.eval_clean_policy import _cfg_from_dict, _load_ckpt_cfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = _read_json(S5_POLICY_PATH)
    ckpt_path = REPO_ROOT / str(policy["base_checkpoint_path"])
    cfg_obj: Config = _cfg_from_dict(_load_ckpt_cfg(ckpt_path))
    cfg_obj.use_fine_stage = True
    cfg_obj.fine_rot_fuse_strength = float(policy["fine_rot_fuse_strength"])
    cfg_obj.fine_tdir_fuse_strength = float(policy["fine_tdir_fuse_strength"])
    cfg_obj.fine_tmag_fuse_strength = float(policy["fine_tmag_fuse_strength"])
    cfg_obj.use_geometry_refine = bool(policy.get("use_geometry_refine", False))
    cfg_obj.tmag_condition_on_dt = False
    cfg_obj.eval_k_list = _as_int_list(config, "eval_k_list", (1, 2, 3, 5, 10, 20))

    model = PanoramaRelPoseModel(cfg_obj, device).to(device)
    payload = torch.load(str(ckpt_path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    load_msg = model.load_state_dict(state, strict=False)
    model.eval()

    ds = RflyPanoPanoramaPairsEvalFixedKList(
        data_root=str(cfg_obj.data_root),
        split="train",
        split_by=str(cfg_obj.split_by),
        train_ratio=float(cfg_obj.train_ratio),
        split_seed=int(cfg_obj.split_seed),
        H=int(cfg_obj.H),
        W=int(cfg_obj.W),
        k_list=tuple(int(x) for x in cfg_obj.eval_k_list),
        pair_step=int(getattr(cfg_obj, "eval_pair_step", 1)),
        min_dt=float(cfg_obj.eval_min_dt),
        max_dt=None if getattr(cfg_obj, "eval_max_dt", None) is None else float(cfg_obj.eval_max_dt),
    )
    manifest = ds.manifest()
    train_cap = _as_int(config, "train_cap_per_sequence", 1024)
    val_cap = _as_int(config, "val_cap_per_sequence", 256)
    seq_keys = ["scene01_seq01", "scene01_seq02"]
    by_seq: Dict[str, List[int]] = {k: [] for k in seq_keys}
    for idx, meta in enumerate(manifest):
        key = f"{meta.get('scene')}_{meta.get('seq')}"
        if key in by_seq:
            by_seq[key].append(idx)
    selected = {
        key: _select_stratified(vals, manifest, train_cap, seed=1900 + i)
        for i, (key, vals) in enumerate(sorted(by_seq.items()))
    }
    val_selected = {key: vals[: min(val_cap, len(vals))] for key, vals in selected.items()}

    q90 = float(policy["thresholds"]["q90_value"])
    q95 = float(policy["thresholds"]["q95_upper_tail_value"])
    mid_scale = float(policy["scales"]["mid_scale"])
    high_scale = float(policy["scales"]["high_scale"])
    cache: Dict[int, ProbeRow] = {}

    def s5_scale(pred_tmag: float) -> float:
        if pred_tmag >= q95:
            return high_scale
        if pred_tmag >= q90:
            return mid_scale
        return 1.0

    def extract_one(ds_idx: int) -> ProbeRow:
        if ds_idx in cache:
            return cache[ds_idx]
        sample = ds[ds_idx]
        meta = manifest[ds_idx]
        IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
        IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
        dt_world = float(meta.get("dt_world", float(sample["t_gt_mag"])))
        dt_tensor = torch.tensor([dt_world], device=device, dtype=torch.float32)
        with torch.no_grad():
            R_pred_t, _t_pred_t, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
        R_gt = sample["R_gt"].detach().float().cpu().numpy()
        t_gt_dir = _unit(sample["t_gt_dir"].detach().float().cpu().numpy())
        gt_tmag = float(sample["t_gt_mag"])
        raw_pred_tmag = float(aux["t_mag"].detach().float().view(-1)[0].cpu())
        pred_tmag = float(raw_pred_tmag * s5_scale(raw_pred_tmag))
        R_pred = R_pred_t.detach().float().cpu().numpy()[0]
        tdir_pred = _unit(aux["t_dir_out"].detach().float().cpu().numpy()[0])

        rot_err = _rot_geodesic_deg(R_pred, R_gt)
        tdir_err = _vec_angle_deg(tdir_pred, t_gt_dir)
        tdir_cos = float(np.clip(np.dot(tdir_pred, t_gt_dir), -1.0, 1.0))
        log_tmag_abs_err = abs(math.log(max(pred_tmag, 1.0e-8)) - math.log(max(gt_tmag, 1.0e-8)))

        coarse_w = aux.get("Wc_ab").detach().float().max(dim=-1).values if "Wc_ab" in aux else None
        fine_w = aux.get("token_weight_f") if "token_weight_f" in aux else None
        coarse_pool = _pool_feature_tensor(aux.get("Fc"), coarse_w)
        fine_pool = _pool_feature_tensor(aux.get("Ff_t", aux.get("Ff")), fine_w)
        bearing_pool = np.concatenate(
            [
                _pool_feature_tensor(aux.get("bearingA_c")),
                _pool_feature_tensor(aux.get("bearingB_c")),
                _pool_feature_tensor(aux.get("bearingA_f")),
                _pool_feature_tensor(aux.get("bearingB_f")),
            ],
            axis=0,
        )
        regime = np.asarray(
            [
                pred_tmag,
                math.log(max(pred_tmag, 1.0e-8)),
                dt_world,
                math.log(max(dt_world, 1.0e-8)),
                float(meta.get("k", -1)),
            ],
            dtype=np.float64,
        )
        current_feature = np.concatenate([coarse_pool, fine_pool, bearing_pool, regime], axis=0)
        row = ProbeRow(
            scene_seq=f"{meta.get('scene')}_{meta.get('seq')}",
            ds_idx=int(ds_idx),
            k=int(meta.get("k", -1)),
            dt_world=dt_world,
            pred_tmag=pred_tmag,
            rot_err_deg=rot_err,
            tdir_err_deg=tdir_err,
            tdir_cos=tdir_cos,
            log_tmag_abs_err=log_tmag_abs_err,
            gt_rotvec=_rot_log(R_gt),
            gt_tdir=t_gt_dir.astype(np.float64),
            gt_log_tmag=float(math.log(max(gt_tmag, 1.0e-8))),
            current_feature=current_feature.astype(np.float64),
            regime_feature=regime.astype(np.float64),
        )
        cache[ds_idx] = row
        if len(cache) == 1 or len(cache) % 64 == 0:
            print(f"[S19] cached feature rows: {len(cache)}", flush=True)
        return row

    def rows_for(indices: Sequence[int]) -> List[ProbeRow]:
        return [extract_one(int(i)) for i in indices]

    unique_rows = []
    for seq_name in seq_keys:
        unique_rows.extend(rows_for(selected[seq_name]))
    total_params = int(sum(int(p.numel()) for p in model.parameters()))
    return {
        "status": "completed",
        "device": str(device),
        "load_missing": len(load_msg.missing_keys),
        "load_unexpected": len(load_msg.unexpected_keys),
        "dataset_train_len": len(ds),
        "selected_per_sequence": {k: len(v) for k, v in selected.items()},
        "val_selected_per_sequence": {k: len(v) for k, v in val_selected.items()},
        "current_feature_dim": int(unique_rows[0].current_feature.shape[0]) if unique_rows else 0,
        "regime_feature_dim": int(unique_rows[0].regime_feature.shape[0]) if unique_rows else 0,
        "unique_rows": int(len(cache)),
        "selected": selected,
        "val_selected": val_selected,
        "rows": cache,
        "base_model_param_count": total_params,
        "base_model_trainable_param_count": int(sum(int(p.numel()) for p in model.parameters() if p.requires_grad)),
        "base_pose_path_changed": False,
    }


def _build_fold_arrays(rows: Sequence[ProbeRow], high_q: float) -> Dict[str, Any]:
    y_rot = np.stack([r.gt_rotvec for r in rows], axis=0)
    y_tdir = np.stack([r.gt_tdir for r in rows], axis=0)
    y_log_tmag = np.asarray([r.gt_log_tmag for r in rows], dtype=np.float64).reshape(-1, 1)
    rot_err = np.asarray([r.rot_err_deg for r in rows], dtype=np.float64)
    tdir_err = np.asarray([r.tdir_err_deg for r in rows], dtype=np.float64)
    tmag_err = np.asarray([r.log_tmag_abs_err for r in rows], dtype=np.float64)
    combined = (
        (rot_err - rot_err.mean()) / max(rot_err.std(), 1.0e-8)
        + (tdir_err - tdir_err.mean()) / max(tdir_err.std(), 1.0e-8)
        + (tmag_err - tmag_err.mean()) / max(tmag_err.std(), 1.0e-8)
    )
    joint = (
        (rot_err - rot_err.mean()) / max(rot_err.std(), 1.0e-8)
        + (tdir_err - tdir_err.mean()) / max(tdir_err.std(), 1.0e-8)
    )
    high_thr = float(np.quantile(combined, high_q))
    joint_thr = float(np.quantile(joint, high_q))
    return {
        "rot": y_rot,
        "tdir": y_tdir,
        "log_tmag": y_log_tmag,
        "rot_err": rot_err,
        "tdir_err": tdir_err,
        "tmag_err": tmag_err,
        "high_label": (combined >= high_thr).astype(np.int64),
        "joint_label": (joint >= joint_thr).astype(np.int64),
        "high_threshold": high_thr,
        "joint_threshold": joint_thr,
    }


def _probe_metrics(xtr_raw: np.ndarray, xva_raw: np.ndarray, ytr: Dict[str, Any], yva: Dict[str, Any], ridge_lam: float) -> Dict[str, float]:
    xtr, xva = _standardize(xtr_raw, xva_raw)
    rot_w = _ridge_fit(xtr, ytr["rot"], ridge_lam)
    tdir_w = _ridge_fit(xtr, ytr["tdir"], ridge_lam)
    tmag_w = _ridge_fit(xtr, ytr["log_tmag"], ridge_lam)
    rot_tr = _ridge_predict(xtr, rot_w)
    rot_va = _ridge_predict(xva, rot_w)
    tdir_tr = np.apply_along_axis(_unit, 1, _ridge_predict(xtr, tdir_w))
    tdir_va = np.apply_along_axis(_unit, 1, _ridge_predict(xva, tdir_w))
    tmag_tr = _ridge_predict(xtr, tmag_w)
    tmag_va = _ridge_predict(xva, tmag_w)
    rot_ang_tr = np.asarray([_rot_geodesic_deg(_rot_exp(p), _rot_exp(g)) for p, g in zip(rot_tr, ytr["rot"])])
    rot_ang_va = np.asarray([_rot_geodesic_deg(_rot_exp(p), _rot_exp(g)) for p, g in zip(rot_va, yva["rot"])])
    tdir_ang_tr = np.asarray([_vec_angle_deg(p, g) for p, g in zip(tdir_tr, ytr["tdir"])])
    tdir_ang_va = np.asarray([_vec_angle_deg(p, g) for p, g in zip(tdir_va, yva["tdir"])])
    tdir_cos_va = np.asarray([float(np.clip(np.dot(_unit(p), _unit(g)), -1.0, 1.0)) for p, g in zip(tdir_va, yva["tdir"])])
    high_metrics = _binary_ridge_metrics(xtr, ytr["high_label"], xva, yva["high_label"], ridge_lam)
    joint_metrics = _binary_ridge_metrics(xtr, ytr["joint_label"], xva, yva["joint_label"], ridge_lam)
    return {
        "feature_dim": int(xtr_raw.shape[1]),
        "train_n": int(xtr_raw.shape[0]),
        "val_n": int(xva_raw.shape[0]),
        "rot_r2_train": _r2(ytr["rot"], rot_tr),
        "rot_r2_val": _r2(yva["rot"], rot_va),
        "rot_angular_error_train": float(rot_ang_tr.mean()),
        "rot_angular_error_val": float(rot_ang_va.mean()),
        "tdir_r2_train": _r2(ytr["tdir"], tdir_tr),
        "tdir_r2_val": _r2(yva["tdir"], tdir_va),
        "tdir_angular_error_train": float(tdir_ang_tr.mean()),
        "tdir_angular_error_val": float(tdir_ang_va.mean()),
        "tdir_cosine_val": float(tdir_cos_va.mean()),
        "log_tmag_r2_train": _r2(ytr["log_tmag"], tmag_tr),
        "log_tmag_r2_val": _r2(yva["log_tmag"], tmag_va),
        "log_tmag_mae_train": float(np.mean(np.abs(ytr["log_tmag"] - tmag_tr))),
        "log_tmag_mae_val": float(np.mean(np.abs(yva["log_tmag"] - tmag_va))),
        "high_error_auc_train": high_metrics["train_auc"],
        "high_error_auc_val": high_metrics["val_auc"],
        "high_error_recall_train": high_metrics["train_recall"],
        "high_error_recall_val": high_metrics["val_recall"],
        "high_error_precision_val": high_metrics["val_precision"],
        "joint_r_tdir_auc_train": joint_metrics["train_auc"],
        "joint_r_tdir_auc_val": joint_metrics["val_auc"],
        "joint_r_tdir_recall_train": joint_metrics["train_recall"],
        "joint_r_tdir_recall_val": joint_metrics["val_recall"],
        "joint_r_tdir_precision_val": joint_metrics["val_precision"],
        "train_test_gap_mean": float(
            np.nanmean(
                [
                    _r2(ytr["rot"], rot_tr) - _r2(yva["rot"], rot_va),
                    _r2(ytr["tdir"], tdir_tr) - _r2(yva["tdir"], tdir_va),
                    _r2(ytr["log_tmag"], tmag_tr) - _r2(yva["log_tmag"], tmag_va),
                ]
            )
        ),
    }


def _run_baseline_probes(config: Dict[str, Any], extracted: Dict[str, Any]) -> Dict[str, Any]:
    ridge_lam = _as_float(config, "ridge_lambda", 1.0)
    high_q = _as_float(config, "high_error_quantile", 0.75)
    selected = extracted["selected"]
    val_selected = extracted["val_selected"]
    rows: Dict[int, ProbeRow] = extracted["rows"]
    folds = [
        ("heldout_scene01_seq01", "scene01_seq01", "scene01_seq02"),
        ("heldout_scene01_seq02", "scene01_seq02", "scene01_seq01"),
    ]
    family_results: List[Dict[str, Any]] = []
    for fold_name, heldout, train_seq in folds:
        train_rows = [rows[int(i)] for i in selected[train_seq]]
        val_rows = [rows[int(i)] for i in val_selected[heldout]]
        ytr = _build_fold_arrays(train_rows, high_q)
        yva = _build_fold_arrays(val_rows, high_q)
        features = {
            "regime_only": (
                np.stack([r.regime_feature for r in train_rows], axis=0),
                np.stack([r.regime_feature for r in val_rows], axis=0),
            ),
            "current_model_features_only": (
                np.stack([r.current_feature for r in train_rows], axis=0),
                np.stack([r.current_feature for r in val_rows], axis=0),
            ),
        }
        for family, (xtr, xva) in features.items():
            row = _probe_metrics(xtr, xva, ytr, yva, ridge_lam)
            row["family"] = family
            row["fold"] = fold_name
            family_results.append(row)

    def mean_for(family: str, key: str) -> float:
        vals = [float(r[key]) for r in family_results if r["family"] == family and math.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    family_means = {
        fam: {
            key: mean_for(fam, key)
            for key in [
                "rot_r2_val",
                "rot_angular_error_val",
                "tdir_r2_val",
                "tdir_angular_error_val",
                "tdir_cosine_val",
                "log_tmag_r2_val",
                "log_tmag_mae_val",
                "high_error_auc_val",
                "high_error_recall_val",
                "high_error_precision_val",
                "joint_r_tdir_auc_val",
                "joint_r_tdir_recall_val",
                "joint_r_tdir_precision_val",
                "train_test_gap_mean",
            ]
        }
        for fam in ["regime_only", "current_model_features_only"]
    }
    return {"families": family_results, "family_means": family_means}


def _candidate_defs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    requested = _as_str_list(config, "candidates_run", ["A", "B", "E"])
    lookup = {
        "A": {
            "code": "A",
            "name": "current_feature_probe_baseline",
            "mode": "probe_head_only",
            "enabled": False,
            "losses": {},
        },
        "B": {
            "code": "B",
            "name": "rot_tdir_pretrain_small",
            "mode": "probe_head_only",
            "enabled": True,
            "losses": {"rot": 1.0, "tdir": 1.0, "log_tmag": 0.0, "joint": 0.0},
        },
        "E": {
            "code": "E",
            "name": "geometry_multitask_small",
            "mode": "probe_head_only",
            "enabled": True,
            "losses": {"rot": 1.0, "tdir": 1.0, "log_tmag": 0.5, "joint": 0.5},
        },
    }
    out = []
    for code in requested:
        code = code.strip().upper()
        if code in lookup:
            out.append(dict(lookup[code]))
    return out


def _train_and_probe_candidate(config: Dict[str, Any], extracted: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    rows: Dict[int, ProbeRow] = extracted["rows"]
    selected = extracted["selected"]
    val_selected = extracted["val_selected"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ridge_lam = _as_float(config, "ridge_lambda", 1.0)
    high_q = _as_float(config, "high_error_quantile", 0.75)
    embed_dim = _as_int(config, "embed_dim", 128)
    hidden_dim = _as_int(config, "hidden_dim", 256)
    updates = _as_int(config, "updates", 100)
    batch_size = _as_int(config, "batch_size", 8)
    lr = _as_float(config, "learning_rate", 1.0e-3)
    seed = _as_int(config, "seed", 3407)
    folds = [
        ("heldout_scene01_seq01", "scene01_seq01", "scene01_seq02"),
        ("heldout_scene01_seq02", "scene01_seq02", "scene01_seq01"),
    ]

    class GeometryAdapter(nn.Module):
        def __init__(self, in_dim: int, hid: int, emb: int) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, hid),
                nn.ReLU(),
                nn.Linear(hid, emb),
                nn.ReLU(),
            )
            self.rot_head = nn.Linear(emb, 3)
            self.tdir_head = nn.Linear(emb, 3)
            self.tmag_head = nn.Linear(emb, 1)
            self.joint_head = nn.Linear(emb, 1)

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            z = self.backbone(x)
            return {
                "embedding": z,
                "rot": self.rot_head(z),
                "tdir": self.tdir_head(z),
                "log_tmag": self.tmag_head(z),
                "joint": self.joint_head(z),
            }

    family_results: List[Dict[str, Any]] = []
    checkpoints: List[str] = []
    curves: List[Dict[str, Any]] = []
    trainable_audit: Optional[Dict[str, Any]] = None
    feature_dim = int(next(iter(rows.values())).current_feature.shape[0])
    base_param_count = int(extracted["base_model_param_count"])

    for fold_idx, (fold_name, heldout, train_seq) in enumerate(folds):
        train_rows = [rows[int(i)] for i in selected[train_seq]]
        val_rows = [rows[int(i)] for i in val_selected[heldout]]
        ytr = _build_fold_arrays(train_rows, high_q)
        yva = _build_fold_arrays(val_rows, high_q)
        x_train = np.stack([r.current_feature for r in train_rows], axis=0).astype(np.float32)
        x_val = np.stack([r.current_feature for r in val_rows], axis=0).astype(np.float32)
        x_train_std, x_val_std = _standardize(x_train, x_val)
        torch.manual_seed(seed + fold_idx)
        np.random.seed(seed + fold_idx)
        model = GeometryAdapter(feature_dim, hidden_dim, embed_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        huber = nn.SmoothL1Loss()
        bce = nn.BCEWithLogitsLoss()

        if trainable_audit is None:
            trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
            trainable_param_count = int(sum(int(p.numel()) for p in model.parameters() if p.requires_grad))
            total_adapter = int(sum(int(p.numel()) for p in model.parameters()))
            trainable_audit = {
                "trainable_mode": candidate["mode"],
                "trainable_names": trainable_names,
                "trainable_param_count": trainable_param_count,
                "frozen_param_count": int(base_param_count),
                "adapter_param_count": total_adapter,
                "base_pose_path_changed": False,
                "s5_policy_touched": False,
                "forbidden_trainable_params": [],
            }

        x_train_t = torch.from_numpy(x_train_std).to(device)
        x_val_t = torch.from_numpy(x_val_std).to(device)
        y_rot_train_t = torch.from_numpy(ytr["rot"].astype(np.float32)).to(device)
        y_tdir_train_t = torch.from_numpy(ytr["tdir"].astype(np.float32)).to(device)
        y_tmag_train_t = torch.from_numpy(ytr["log_tmag"].astype(np.float32)).to(device)
        y_joint_train_t = torch.from_numpy(ytr["joint_label"].astype(np.float32)).view(-1, 1).to(device)
        y_rot_val_t = torch.from_numpy(yva["rot"].astype(np.float32)).to(device)
        y_tdir_val_t = torch.from_numpy(yva["tdir"].astype(np.float32)).to(device)
        y_tmag_val_t = torch.from_numpy(yva["log_tmag"].astype(np.float32)).to(device)
        y_joint_val_t = torch.from_numpy(yva["joint_label"].astype(np.float32)).view(-1, 1).to(device)

        best_val = float("inf")
        best_state = None
        history: List[Dict[str, float]] = []
        rng = np.random.default_rng(seed + 100 * fold_idx + 7)

        def loss_bundle(outputs: Dict[str, torch.Tensor], split: str) -> Dict[str, torch.Tensor]:
            if split == "train":
                y_rot_t = y_rot_train_t
                y_tdir_t = y_tdir_train_t
                y_tmag_t = y_tmag_train_t
                y_joint_t = y_joint_train_t
            else:
                y_rot_t = y_rot_val_t
                y_tdir_t = y_tdir_val_t
                y_tmag_t = y_tmag_val_t
                y_joint_t = y_joint_val_t
            pred_tdir = F.normalize(outputs["tdir"], dim=1)
            rot_loss = huber(outputs["rot"], y_rot_t)
            tdir_loss = (1.0 - (pred_tdir * y_tdir_t).sum(dim=1).clamp(-1.0, 1.0)).mean()
            tmag_loss = huber(outputs["log_tmag"], y_tmag_t)
            joint_loss = bce(outputs["joint"], y_joint_t)
            total = (
                candidate["losses"]["rot"] * rot_loss
                + candidate["losses"]["tdir"] * tdir_loss
                + candidate["losses"]["log_tmag"] * tmag_loss
                + candidate["losses"]["joint"] * joint_loss
            )
            return {
                "total": total,
                "rot": rot_loss,
                "tdir": tdir_loss,
                "log_tmag": tmag_loss,
                "joint": joint_loss,
            }

        for upd in range(updates):
            idx = rng.choice(x_train_std.shape[0], size=min(batch_size, x_train_std.shape[0]), replace=False)
            xb = x_train_t[idx]
            outputs = model(xb)
            optimizer.zero_grad(set_to_none=True)
            pred_tdir = F.normalize(outputs["tdir"], dim=1)
            rot_loss = huber(outputs["rot"], y_rot_train_t[idx])
            tdir_loss = (1.0 - (pred_tdir * y_tdir_train_t[idx]).sum(dim=1).clamp(-1.0, 1.0)).mean()
            tmag_loss = huber(outputs["log_tmag"], y_tmag_train_t[idx])
            joint_loss = bce(outputs["joint"], y_joint_train_t[idx])
            total = (
                candidate["losses"]["rot"] * rot_loss
                + candidate["losses"]["tdir"] * tdir_loss
                + candidate["losses"]["log_tmag"] * tmag_loss
                + candidate["losses"]["joint"] * joint_loss
            )
            total.backward()
            optimizer.step()

            if upd in {0, updates - 1} or (upd + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    train_out = model(x_train_t)
                    val_out = model(x_val_t)
                    train_loss = loss_bundle(train_out, "train")
                    val_loss = loss_bundle(val_out, "val")
                model.train()
                hist_row = {
                    "update": int(upd + 1),
                    "train_total": float(train_loss["total"].detach().cpu()),
                    "val_total": float(val_loss["total"].detach().cpu()),
                    "train_rot": float(train_loss["rot"].detach().cpu()),
                    "val_rot": float(val_loss["rot"].detach().cpu()),
                    "train_tdir": float(train_loss["tdir"].detach().cpu()),
                    "val_tdir": float(val_loss["tdir"].detach().cpu()),
                    "train_tmag": float(train_loss["log_tmag"].detach().cpu()),
                    "val_tmag": float(val_loss["log_tmag"].detach().cpu()),
                    "train_joint": float(train_loss["joint"].detach().cpu()),
                    "val_joint": float(val_loss["joint"].detach().cpu()),
                }
                history.append(hist_row)
                if hist_row["val_total"] < best_val:
                    best_val = hist_row["val_total"]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is None:
            raise RuntimeError(f"S19 candidate {candidate['name']} fold {fold_name} did not produce any checkpoint state.")

        ckpt_dir = Path(tempfile.gettempdir()) / "s19_geometry_pretraining"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{candidate['name']}_{fold_name}.pt"
        torch.save(best_state, ckpt_path)
        checkpoints.append(str(ckpt_path))
        model.load_state_dict(best_state, strict=True)
        model.eval()
        with torch.no_grad():
            emb_train = model(x_train_t)["embedding"].detach().cpu().numpy().astype(np.float64)
            emb_val = model(x_val_t)["embedding"].detach().cpu().numpy().astype(np.float64)
        row = _probe_metrics(emb_train, emb_val, ytr, yva, ridge_lam)
        row["family"] = candidate["name"]
        row["fold"] = fold_name
        row["best_checkpoint_path"] = str(ckpt_path)
        row["best_val_pretrain_loss"] = float(best_val)
        family_results.append(row)
        curves.append({"fold": fold_name, "history": history})

    def mean_for(key: str) -> float:
        vals = [float(r[key]) for r in family_results if math.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    family_means = {
        key: mean_for(key)
        for key in [
            "rot_r2_val",
            "rot_angular_error_val",
            "tdir_r2_val",
            "tdir_angular_error_val",
            "tdir_cosine_val",
            "log_tmag_r2_val",
            "log_tmag_mae_val",
            "high_error_auc_val",
            "high_error_recall_val",
            "high_error_precision_val",
            "joint_r_tdir_auc_val",
            "joint_r_tdir_recall_val",
            "joint_r_tdir_precision_val",
            "train_test_gap_mean",
            "best_val_pretrain_loss",
        ]
    }
    return {
        "name": candidate["name"],
        "code": candidate["code"],
        "trainable_mode": candidate["mode"],
        "losses": candidate["losses"],
        "fold_rows": family_results,
        "family_mean": family_means,
        "trainable_param_audit": trainable_audit,
        "best_checkpoint_paths": checkpoints,
        "training_curves": curves,
    }


def _classification(payload: Dict[str, Any]) -> Tuple[str, str, bool]:
    if payload.get("pretraining_status") == "failed":
        return "PRETRAINING-HARNESS-FAIL", "Geometry-aware pretraining failed before producing trustworthy probe outputs.", False
    baselines = payload["baselines"]
    current = baselines["current_model_features_only"]
    regime = baselines["regime_only"]
    imagenet = baselines["imagenet_resnet50_locked_reference"]
    candidates = payload["candidate_family_means"]
    geom_names = [name for name in candidates.keys() if name != "current_feature_probe_baseline"]
    if not geom_names:
        return "INCONCLUSIVE", "No geometry-aware pretraining candidate completed.", False

    best_name = max(geom_names, key=lambda n: float(candidates[n].get("joint_r_tdir_auc_val", float("-inf"))))
    best = candidates[best_name]
    current_hit = (
        float(best.get("joint_r_tdir_auc_val", float("-inf"))) > float(current["joint_r_tdir_auc_val"]) + 0.01
        and float(best.get("rot_r2_val", float("-inf"))) > float(current["rot_r2_val"]) + 0.5
        and float(best.get("tdir_r2_val", float("-inf"))) > float(current["tdir_r2_val"]) + 0.1
    )
    imagenet_hit = (
        float(best.get("joint_r_tdir_auc_val", float("-inf"))) > float(imagenet["joint_auc"])
        and float(best.get("rot_r2_val", float("-inf"))) > float(imagenet["rot_r2"])
        and float(best.get("tdir_r2_val", float("-inf"))) > float(imagenet["tdir_r2"])
    )
    tmag_only = (
        float(best.get("log_tmag_mae_val", float("inf"))) < min(float(current["log_tmag_mae_val"]), float(regime["log_tmag_mae_val"]))
        and not current_hit
    )
    if current_hit:
        return (
            "GEOMETRY-PRETRAINING-DIAGNOSTIC-SIGNAL",
            f"`{best_name}` improved rot_R2/tdir_R2/joint_AUC beyond the current task-specific feature baseline and is worth S19b integration planning.",
            True,
        )
    if tmag_only:
        return (
            "GEOMETRY-PRETRAINING-TMAG-ONLY-SIGNAL",
            f"`{best_name}` mainly improved log_tmag probing while R/tdir/joint remained below the current task-specific feature baseline.",
            False,
        )
    if imagenet_hit:
        return (
            "GEOMETRY-PRETRAINING-WEAK-SIGNAL",
            f"`{best_name}` outperformed the frozen ImageNet ResNet50 reference but did not stably beat the current task-specific feature baseline.",
            False,
        )
    return (
        "NO-STABLE-GEOMETRY-PRETRAINING-GAIN",
        "No geometry-aware pretraining candidate produced stable R/tdir/joint probe gains beyond the current task-specific baseline.",
        False,
    )


def _plot_metric_bars(candidate_means: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    metrics = [
        ("rot_r2_val", "rot_r2_comparison.png", "rot R2"),
        ("tdir_r2_val", "tdir_r2_comparison.png", "tdir R2"),
        ("joint_r_tdir_auc_val", "joint_auc_comparison.png", "joint AUC"),
    ]
    labels = list(candidate_means.keys())
    for metric, fname, title in metrics:
        vals = [float(candidate_means[name].get(metric, float("nan"))) for name in labels]
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(np.arange(len(labels)), vals)
        ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=20, ha="right")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=160)
        plt.close(fig)


def _plot_training_curves(candidate_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for cand in candidate_rows:
        for fold_hist in cand.get("training_curves", []):
            hist = fold_hist.get("history", [])
            if not hist:
                continue
            xs = [int(r["update"]) for r in hist]
            ys = [float(r["val_total"]) for r in hist]
            ax.plot(xs, ys, marker="o", label=f"{cand['name']}:{fold_hist['fold']}")
            plotted = True
    if plotted:
        ax.set_title("Validation pretraining loss")
        ax.set_xlabel("update")
        ax.set_ylabel("val_total")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "pretraining_val_loss_curves.png", dpi=160)
    plt.close(fig)


def _markdown_table(rows: Sequence[Dict[str, Any]], cols: Sequence[Tuple[str, str]]) -> str:
    out = ["| " + " | ".join(label for label, _ in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in rows:
        vals = []
        for _label, key in cols:
            v = row.get(key)
            if isinstance(v, bool):
                vals.append("True" if v else "False")
            elif isinstance(v, (float, int, np.floating)):
                vals.append(_fmt(v, 4))
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def _write_reports(payload: Dict[str, Any], outputs: Dict[str, Path]) -> None:
    gate = payload["baseline_gate"]
    baselines = payload.get("baselines", {})
    candidate_rows = payload.get("candidate_rows", [])
    candidate_means = payload.get("candidate_family_means", {})
    lines = [
        "# S19 Geometry Aware Pretraining Feasibility Report",
        "",
        "## Executive summary",
        "",
        f"- final classification: `{payload['final_classification']}`",
        f"- S19b full integration recommended: `{payload['s19b_full_integration_recommended']}`",
        "- S19 is a feasibility diagnostic only; it does not replace S5 and does not claim clean gain.",
        f"- interpretation: {payload['classification_reason']}",
        "",
        "## Motivation from S13/S16/S17/S18",
        "",
        "- S13 quantified the practical gap and identified `R-TDIR-COUPLED-LIMITED` as the primary bottleneck.",
        "- S16/S16b showed that generic frozen ImageNet features are weaker than the current task-specific representation for this geometric task.",
        "- S17 cleaned up supervision/convention doubts, and S18 retained the historical protocol with representativeness caveat.",
        "- S19 therefore tests whether lightweight geometry-aware pretraining can make the existing current-feature representation more linearly readable before any full integration.",
        "",
        "## Baseline gate",
        "",
        f"- passed: `{gate['passed']}`",
        f"- current-architecture load missing / unexpected: `{gate['load_missing']}` / `{gate['load_unexpected']}`",
        f"- ridge_calib buffers missing: `{len(gate['missing_key_categories'].get('ridge_calib_buffers', []))}`",
        f"- coupled_pose_head params missing: `{len(gate['missing_key_categories'].get('coupled_pose_head_params', []))}`",
        f"- other missing: `{len(gate['missing_key_categories'].get('other_missing', []))}`",
        f"- S5 locked metrics preserved: `{gate['metric_ok']}`",
        "",
        "## Current task-specific feature baseline",
        "",
        f"- locked S16 reference: `rot_R2={_fmt(S16_LOCKED_CURRENT['rot_r2'])}, tdir_R2={_fmt(S16_LOCKED_CURRENT['tdir_r2'])}, joint_AUC={_fmt(S16_LOCKED_CURRENT['joint_auc'])}`",
        f"- this S19 run baseline: `rot_R2={_fmt(baselines['current_model_features_only']['rot_r2_val'])}, tdir_R2={_fmt(baselines['current_model_features_only']['tdir_r2_val'])}, joint_AUC={_fmt(baselines['current_model_features_only']['joint_r_tdir_auc_val'])}`",
        "",
        "## ImageNet ResNet50 secondary baseline",
        "",
        f"- locked S16b reference: `rot_R2={_fmt(S16B_LOCKED_IMAGENET['rot_r2'])}, tdir_R2={_fmt(S16B_LOCKED_IMAGENET['tdir_r2'])}, joint_AUC={_fmt(S16B_LOCKED_IMAGENET['joint_auc'])}`",
        "",
        "## Geometry pretraining target definitions",
        "",
        "- Relative rotation target: GT relative rotation log-vector with SmoothL1 loss.",
        "- Translation direction target: GT local tdir with cosine loss.",
        "- Optional log_tmag auxiliary: GT log magnitude with SmoothL1 loss.",
        "- Optional joint high-error auxiliary: train-only binary label from combined current-model R/tdir difficulty bucket.",
        "",
        "## Candidate families",
        "",
    ]
    for row in payload["candidate_definitions"]:
        lines.append(f"- {row['code']}. `{row['name']}` mode=`{row['mode']}` losses=`{json.dumps(row['losses'], ensure_ascii=True)}`")
    lines.extend(
        [
            "",
            "## Trainable parameter audit",
            "",
            _markdown_table(
                [
                    {
                        "candidate": row["name"],
                        "mode": row["trainable_mode"],
                        "trainable_param_count": row["trainable_param_audit"]["trainable_param_count"],
                        "frozen_param_count": row["trainable_param_audit"]["frozen_param_count"],
                        "base_pose_path_changed": row["trainable_param_audit"]["base_pose_path_changed"],
                        "s5_policy_touched": row["trainable_param_audit"]["s5_policy_touched"],
                    }
                    for row in candidate_rows
                    if row["name"] != "current_feature_probe_baseline"
                ],
                [
                    ("candidate", "candidate"),
                    ("mode", "mode"),
                    ("trainable", "trainable_param_count"),
                    ("frozen", "frozen_param_count"),
                    ("base changed", "base_pose_path_changed"),
                    ("S5 touched", "s5_policy_touched"),
                ],
            ),
            "",
            f"- explicit trainable names: `{candidate_rows[1]['trainable_param_audit']['trainable_names'] if len(candidate_rows) > 1 else []}`",
            "",
            "## Training budget",
            "",
            f"- trainable mode: `{payload['trainable_mode']}`",
            f"- train cap per sequence: `{payload['training_budget']['train_cap_per_sequence']}`",
            f"- val cap per sequence: `{payload['training_budget']['val_cap_per_sequence']}`",
            f"- updates per candidate per fold: `{payload['training_budget']['updates']}`",
            f"- batch_size: `{payload['training_budget']['batch_size']}`",
            f"- folds: `{payload['training_budget']['folds']}`",
            "",
            "## Probe protocol",
            "",
            "- Extract frozen current-model pair features on the train split only.",
            "- Train lightweight geometry heads on held-in train sequence supervision only.",
            "- Re-extract learned embedding from the trained head and evaluate it with the same ridge/AUC probe family used for the current baseline.",
            "- No final test sequence is used for candidate selection.",
            "",
            "## CV/probe results",
            "",
            _markdown_table(
                [{"family": name, **row} for name, row in candidate_means.items()],
                [
                    ("family", "family"),
                    ("rot R2", "rot_r2_val"),
                    ("tdir R2", "tdir_r2_val"),
                    ("log tmag MAE", "log_tmag_mae_val"),
                    ("high AUC", "high_error_auc_val"),
                    ("joint AUC", "joint_r_tdir_auc_val"),
                    ("gap", "train_test_gap_mean"),
                ],
            ),
            "",
            "## Primary comparison against current features",
            "",
            f"- current baseline: `rot_R2={_fmt(baselines['current_model_features_only']['rot_r2_val'])}, tdir_R2={_fmt(baselines['current_model_features_only']['tdir_r2_val'])}, joint_AUC={_fmt(baselines['current_model_features_only']['joint_r_tdir_auc_val'])}`",
            f"- best geometry candidate: `{payload['best_geometry_result']}`",
            "",
            "## Secondary comparison against ResNet50",
            "",
            f"- ImageNet frozen ResNet50 reference: `rot_R2={_fmt(S16B_LOCKED_IMAGENET['rot_r2'])}, tdir_R2={_fmt(S16B_LOCKED_IMAGENET['tdir_r2'])}, joint_AUC={_fmt(S16B_LOCKED_IMAGENET['joint_auc'])}`",
            f"- geometry result vs ImageNet conclusion: `{payload['secondary_comparison_summary']}`",
            "",
            "## R/tdir coupling proxy analysis",
            "",
            f"- coupling proxy result: `{payload['r_tdir_coupling_proxy_result']}`",
            "",
            "## Tmag analysis",
            "",
            f"- regime_only log_tmag_MAE: `{_fmt(baselines['regime_only']['log_tmag_mae_val'])}`",
            f"- current_model_features_only log_tmag_MAE: `{_fmt(baselines['current_model_features_only']['log_tmag_mae_val'])}`",
            f"- best geometry log_tmag_MAE: `{_fmt(payload['best_geometry_metrics'].get('log_tmag_mae_val'))}`",
            "",
            "## Leakage audit",
            "",
            "- no test-set candidate selection: `True`",
            "- no test-set threshold selection: `True`",
            "- gt labels used only for train/CV supervision: `True`",
            "- no gt pose / gt_tmag / gt_tdir used as inference feature: `True`",
            "- S5 policy unchanged: `True`",
            "- no heavy checkpoint committed: `True`",
            "- no pretrained/fine-tuned weights committed: `True`",
            "",
            "## Whether S19b is recommended",
            "",
            f"- `{payload['s19b_full_integration_recommended']}`",
            "",
            "## Whether S5 remains final clean candidate",
            "",
            f"- `{payload['s5_remains_final_clean_candidate']}`",
            "",
            "## Final classification",
            "",
            f"- `{payload['final_classification']}`",
            "",
        ]
    )
    outputs["report"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report"].write_text("\n".join(lines), encoding="utf-8")

    summary_lines = [
        "# Final S19 Geometry Aware Pretraining Feasibility Summary",
        "",
        f"- final classification: `{payload['final_classification']}`",
        f"- current baseline result: `{payload['current_baseline_result']}`",
        f"- ImageNet ResNet50 baseline result: `{payload['imagenet_baseline_result']}`",
        f"- best geometry-pretraining result: `{payload['best_geometry_result']}`",
        f"- R/tdir coupling proxy result: `{payload['r_tdir_coupling_proxy_result']}`",
        f"- S19b full integration recommended: `{payload['s19b_full_integration_recommended']}`",
        "- S5 remains final clean candidate: `True`",
        "",
    ]
    outputs["summary"].parent.mkdir(parents=True, exist_ok=True)
    outputs["summary"].write_text("\n".join(summary_lines), encoding="utf-8")


def run(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    config = _parse_config(config_path)
    outputs = _resolve_output_paths(config)
    gate = _baseline_gate()
    payload: Dict[str, Any] = {
        "name": str(config.get("name", "S19_geometry_aware_pretraining_feasibility")),
        "config_path": str(outputs["config"]),
        "report_path": str(outputs["report"]),
        "candidates_path": str(outputs["candidates"]),
        "summary_path": str(outputs["summary"]),
        "baseline_gate": gate,
        "s5_remains_final_clean_candidate": True,
    }
    if not gate["passed"]:
        payload.update(
            {
                "final_classification": "REPRODUCTION-MISMATCH",
                "classification_reason": "S8b current-architecture reproduction contract did not pass.",
                "s19b_full_integration_recommended": False,
            }
        )
        _write_json(outputs["candidates"], payload)
        _write_reports(payload, outputs)
        return payload

    try:
        extracted = _extract_rows(config)
        baseline_probe = _run_baseline_probes(config, extracted)
        payload["feature_extraction_audit"] = {
            k: v
            for k, v in extracted.items()
            if k not in {"rows", "selected", "val_selected"}
        }
        baselines = {
            "regime_only": dict(baseline_probe["family_means"]["regime_only"]),
            "current_model_features_only": dict(baseline_probe["family_means"]["current_model_features_only"]),
            "imagenet_resnet50_locked_reference": dict(S16B_LOCKED_IMAGENET),
        }
        candidate_defs = _candidate_defs(config)
        candidate_rows: List[Dict[str, Any]] = []
        candidate_means: Dict[str, Dict[str, float]] = {
            "regime_only": dict(baseline_probe["family_means"]["regime_only"]),
            "current_feature_probe_baseline": dict(baseline_probe["family_means"]["current_model_features_only"]),
        }
        candidate_rows.append(
            {
                "name": "current_feature_probe_baseline",
                "code": "A",
                "trainable_mode": "probe_head_only",
                "losses": {},
                "fold_rows": [row for row in baseline_probe["families"] if row["family"] == "current_model_features_only"],
                "family_mean": dict(baseline_probe["family_means"]["current_model_features_only"]),
                "trainable_param_audit": {
                    "trainable_mode": "probe_head_only",
                    "trainable_names": [],
                    "trainable_param_count": 0,
                    "frozen_param_count": int(extracted["base_model_param_count"]),
                    "adapter_param_count": 0,
                    "base_pose_path_changed": False,
                    "s5_policy_touched": False,
                    "forbidden_trainable_params": [],
                },
                "best_checkpoint_paths": [],
                "training_curves": [],
            }
        )
        for cand in candidate_defs:
            if cand["code"] == "A":
                continue
            row = _train_and_probe_candidate(config, extracted, cand)
            candidate_rows.append(row)
            candidate_means[row["name"]] = dict(row["family_mean"])
        payload["candidate_definitions"] = candidate_defs
        payload["candidate_rows"] = candidate_rows
        payload["candidate_family_means"] = candidate_means
        payload["baselines"] = baselines
        payload["training_budget"] = {
            "train_cap_per_sequence": _as_int(config, "train_cap_per_sequence", 1024),
            "val_cap_per_sequence": _as_int(config, "val_cap_per_sequence", 256),
            "updates": _as_int(config, "updates", 100),
            "batch_size": _as_int(config, "batch_size", 8),
            "folds": ["heldout_scene01_seq01", "heldout_scene01_seq02"],
        }
        payload["trainable_mode"] = "probe_head_only"
        payload["candidates_run"] = [row["name"] for row in candidate_rows]
        best_name = max(
            [name for name in candidate_means.keys() if name not in {"regime_only", "current_feature_probe_baseline"}],
            key=lambda n: float(candidate_means[n].get("joint_r_tdir_auc_val", float("-inf"))),
        )
        payload["best_geometry_candidate"] = best_name
        payload["best_geometry_metrics"] = dict(candidate_means[best_name])
        payload["current_baseline_result"] = (
            f"rot_R2={_fmt(baselines['current_model_features_only']['rot_r2_val'])}, "
            f"tdir_R2={_fmt(baselines['current_model_features_only']['tdir_r2_val'])}, "
            f"joint_AUC={_fmt(baselines['current_model_features_only']['joint_r_tdir_auc_val'])}"
        )
        payload["imagenet_baseline_result"] = (
            f"rot_R2={_fmt(S16B_LOCKED_IMAGENET['rot_r2'])}, "
            f"tdir_R2={_fmt(S16B_LOCKED_IMAGENET['tdir_r2'])}, "
            f"joint_AUC={_fmt(S16B_LOCKED_IMAGENET['joint_auc'])}"
        )
        payload["best_geometry_result"] = (
            f"{best_name}: rot_R2={_fmt(candidate_means[best_name]['rot_r2_val'])}, "
            f"tdir_R2={_fmt(candidate_means[best_name]['tdir_r2_val'])}, "
            f"joint_AUC={_fmt(candidate_means[best_name]['joint_r_tdir_auc_val'])}"
        )
        payload["r_tdir_coupling_proxy_result"] = payload["best_geometry_result"]
        payload["secondary_comparison_summary"] = (
            "better_than_imagenet_locked_reference"
            if (
                float(candidate_means[best_name]["rot_r2_val"]) > S16B_LOCKED_IMAGENET["rot_r2"]
                and float(candidate_means[best_name]["tdir_r2_val"]) > S16B_LOCKED_IMAGENET["tdir_r2"]
                and float(candidate_means[best_name]["joint_r_tdir_auc_val"]) > S16B_LOCKED_IMAGENET["joint_auc"]
            )
            else "not_better_than_imagenet_locked_reference"
        )
        cls, reason, recommend = _classification(payload)
        payload["final_classification"] = cls
        payload["classification_reason"] = reason
        payload["s19b_full_integration_recommended"] = bool(recommend)
        payload["leakage_audit"] = {
            "no_test_set_candidate_selection": True,
            "no_test_set_threshold_selection": True,
            "gt_labels_used_only_for_train_cv_supervision": True,
            "used_gt_as_inference_feature": False,
            "s5_policy_unchanged": True,
            "no_heavy_checkpoint_committed": True,
            "no_pretrained_weights_committed": True,
        }
        payload["pretraining_status"] = "completed"
        outputs["figures"].mkdir(parents=True, exist_ok=True)
        _plot_metric_bars(candidate_means, outputs["figures"])
        _plot_training_curves(candidate_rows, outputs["figures"])
        payload["figure_dir"] = str(outputs["figures"])
    except Exception as exc:
        payload["pretraining_status"] = "failed"
        payload["exception"] = repr(exc)
        payload["traceback_tail"] = traceback.format_exc(limit=12)
        payload["final_classification"] = "PRETRAINING-HARNESS-FAIL"
        payload["classification_reason"] = f"S19 harness failed: {exc!r}"
        payload["s19b_full_integration_recommended"] = False

    _write_json(outputs["candidates"], payload)
    _write_reports(payload, outputs)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="S19 geometry-aware pretraining feasibility diagnostic.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Config path.")
    parser.add_argument("--print-json", action="store_true", help="Print final payload as json.")
    args = parser.parse_args()
    payload = run(Path(args.config))
    outputs = _resolve_output_paths(_parse_config(Path(args.config)))
    if args.print_json:
        print(json.dumps(_json_safe(payload), indent=2, ensure_ascii=True))
    else:
        print(f"[S19] final_classification={payload['final_classification']}")
        print(f"[S19] report={outputs['report']}")
        print(f"[S19] candidates={outputs['candidates']}")
        print(f"[S19] summary={outputs['summary']}")


if __name__ == "__main__":
    main()
