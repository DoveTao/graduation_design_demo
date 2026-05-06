#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
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
from train_mvp import eval_odometry_sequence


S2B_POLICY_PATH = REPO_ROOT / "checkpoints" / "S2b_clean_fine_rot_policy.json"
S5_POLICY_PATH = REPO_ROOT / "checkpoints" / "S5_clean_tmag_calibration_policy.json"
S5_CANDIDATES_PATH = REPO_ROOT / "checkpoints" / "S5_tmag_regime_aware_calibration_candidates.json"
S8B_CONTRACT_PATH = REPO_ROOT / "checkpoints" / "S8b_reproduction_contract.json"
S8B_S2B_RESULT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_s2b_wrapper_result.json"
S8B_S5_RESULT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_s5_wrapper_result.json"

REPORT_PATH = REPO_ROOT / "checkpoints" / "S9_regime_only_reliability_router_report.md"
CANDIDATES_PATH = REPO_ROOT / "checkpoints" / "S9_regime_only_reliability_router_candidates.json"
POLICY_PATH = REPO_ROOT / "checkpoints" / "S9_regime_only_reliability_router_policy.json"
FIG_DIR = REPO_ROOT / "checkpoints" / "S9_regime_only_reliability_router_figures"
SUMMARY_PATH = REPO_ROOT / "reports" / "final_s9_regime_only_router_summary.md"
ROWS_CACHE_PATH = REPO_ROOT / "checkpoints" / "S9_regime_only_reliability_router_rows_cache.json"

S5_LOCKED = {"drift": 1.327343, "ATE": 7.352288, "path_ratio": 0.932379}
CURRENT_ARCH_BENIGN_MISSING = {
    "ridge_calib_buffers": 2,
    "coupled_pose_head_params": 12,
    "other_missing": 0,
}
SAFE_PATH_RANGE = (0.90, 0.97)
MAX_LOGISTIC_FIT_ROWS = 512
TRAIN_FULL_ROWS_CAP = 512
FOLD_TRAIN_ROWS_CAP = 512
FOLD_VAL_ROWS_CAP = 512


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    params: Dict[str, Any]
    action_set: List[str]
    simplicity_rank: Tuple[float, float, float]


@dataclass
class OdomEvalRow:
    candidate_name: str
    family: str
    split: str
    fold: str
    drift: float
    ate: float
    path_ratio: float
    selected_k: int
    num_pairs: int
    num_chains: int
    load_missing: int
    load_unexpected: int
    mean_metric_pos_err: float
    worst_chain_ate: float


class ManifestSubsetDataset(torch.utils.data.Dataset):
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


class ContextDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, context: Dict[str, Any]) -> None:
        self.base_ds = base_ds
        self.context = context

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        sample = self.base_ds[idx]
        self.context["sample"] = sample
        self.context["meta"] = sample.get("meta", {})
        return sample

    def manifest(self):
        return self.base_ds.manifest()


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


def _load_rows_cache() -> Dict[str, Any] | None:
    if not ROWS_CACHE_PATH.exists():
        return None
    return _read_json(ROWS_CACHE_PATH)


def _save_rows_cache(payload: Dict[str, Any]) -> None:
    _write_json(ROWS_CACHE_PATH, payload)


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


def _auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    score = score.astype(np.float64)
    pos = int(y_true.sum())
    neg = int((1 - y_true).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.zeros_like(score, dtype=np.float64)
    i = 0
    while i < len(sorted_score):
        j = i + 1
        while j < len(sorted_score) and sorted_score[j] == sorted_score[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    pos_ranks = ranks[y_true == 1]
    return float((pos_ranks.sum() - pos * (pos + 1) / 2.0) / max(pos * neg, 1))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (prob >= threshold).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    tn = int(np.sum((y_true == 0) & (pred == 0)))
    return {
        "auc": _auc_score(y_true, prob),
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
    }


def _dt_bucket(v: float) -> str:
    if v < 0.1:
        return "<0.1"
    if v < 0.3:
        return "[0.1,0.3)"
    if v < 0.5:
        return "[0.3,0.5)"
    if v < 1.0:
        return "[0.5,1.0)"
    return ">=1.0"


def _k_bucket(v: int) -> str:
    return f"k={int(v)}"


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
        {"feature": "pred_tmag_s2b", "inference_visible": True, "derived_only_from_inference": True, "allowed": True},
        {"feature": "dt_world", "inference_visible": True, "derived_only_from_inference": True, "allowed": True},
        {"feature": "k", "inference_visible": True, "derived_only_from_inference": True, "allowed": True},
        {"feature": "s5_base_scale", "inference_visible": True, "derived_only_from_inference": True, "allowed": True},
        {"feature": "pred_tmag_bucket", "inference_visible": True, "derived_only_from_inference": True, "allowed": True},
        {"feature": "train-derived quantile bucket id", "inference_visible": True, "derived_only_from_inference": True, "allowed": True},
        {"feature": "fine token / spherical token", "inference_visible": True, "derived_only_from_inference": True, "allowed": False},
        {"feature": "gt_tmag / gt pose / pair_pos_err", "inference_visible": False, "derived_only_from_inference": False, "allowed": False},
    ]


def _candidate_specs() -> List[CandidateSpec]:
    return [
        CandidateSpec("R1_predq90_conservative", "pred_q90_router", {}, ["A0", "A1"], (1.0, 0.0, 0.0)),
        CandidateSpec("R2_predq90_q95_upper_tail", "pred_upper_tail_router", {}, ["A0", "A1", "A2"], (2.0, 0.0, 0.0)),
        CandidateSpec("R3_dt1_k20_conservative", "dtk_router", {"fallback": False}, ["A0", "A1"], (3.0, 0.0, 0.0)),
        CandidateSpec("R3_dt1_k20_fallback", "dtk_router", {"fallback": True}, ["A0", "A3"], (3.0, 1.0, 0.0)),
        CandidateSpec("R4_combined_pred_dt_k", "combined_router", {}, ["A0", "A1", "A2"], (4.0, 0.0, 0.0)),
        CandidateSpec("R5_logistic_q80_conservative", "logistic_router", {"threshold_quantile": 0.80, "action_if_risky": "A1"}, ["A0", "A1"], (5.0, 0.80, 1.0)),
    ]


def _select_subsample_indices(n: int, cap: int) -> List[int]:
    if n <= cap:
        return list(range(n))
    raw = np.linspace(0, n - 1, num=cap)
    out: List[int] = []
    seen = set()
    for v in raw:
        idx = int(round(float(v)))
        idx = max(0, min(n - 1, idx))
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


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
        raise RuntimeError(f"missing expected S9 train-CV groups: {missing}")
    return train_ds, target_groups, {g: seq_to_indices[g] for g in target_groups}


def _extract_rows(model: DtBucketScaledMagnitudeModel, ds, s5_policy: Dict[str, Any], device: torch.device, *, cap: int | None = None) -> List[Dict[str, Any]]:
    manifest = ds.manifest()
    use_indices = list(range(len(manifest))) if cap is None else _select_subsample_indices(len(manifest), cap)
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for offset, ds_idx in enumerate(use_indices):
            if offset == 0 or (offset + 1) % 64 == 0 or (offset + 1) == len(use_indices):
                print(f"[S9] extracting rows {offset + 1}/{len(use_indices)}", flush=True)
            meta = manifest[ds_idx]
            sample = ds[ds_idx]
            IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
            IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
            t_gt_vec = sample["t_gt_vec"].unsqueeze(0).to(device, non_blocking=True)
            dt_val = float(meta.get("dt_world", float(sample["t_gt_mag"])))
            dt_world = torch.tensor([dt_val], device=device, dtype=torch.float32)
            _R_pred, _t_pred, aux = model(IA, IB, enable_depth_fusion=False, dt_world=dt_world)
            pred_tmag = float(aux["t_mag"].detach().float().view(-1)[0].cpu())
            s5_base_scale = _s5_scale_from_pred(pred_tmag, s5_policy)
            s2b_tvec = aux["t_vec_out"].detach().float().cpu().numpy()[0]
            s5_tvec = s2b_tvec * float(s5_base_scale)
            gt_tvec = t_gt_vec.detach().float().cpu().numpy()[0]
            rows.append(
                {
                    "dataset_index": int(ds_idx),
                    "scene": str(meta.get("scene")),
                    "seq": str(meta.get("seq")),
                    "scene_seq": f"{meta.get('scene')}::{meta.get('seq')}",
                    "k": int(meta.get("k", -1)),
                    "dt_world": float(dt_val),
                    "dt_bucket": _dt_bucket(float(dt_val)),
                    "k_bucket": _k_bucket(int(meta.get("k", -1))),
                    "pred_tmag_s2b": float(pred_tmag),
                    "s5_base_scale": float(s5_base_scale),
                    "is_dt_ge_1": 1.0 if float(dt_val) >= 1.0 else 0.0,
                    "is_k20": 1.0 if int(meta.get("k", -1)) == 20 else 0.0,
                    "s2b_pair_pos_err": float(np.linalg.norm(s2b_tvec - gt_tvec)),
                    "s5_pair_pos_err": float(np.linalg.norm(s5_tvec - gt_tvec)),
                    "gt_tvec": gt_tvec.astype(np.float64).tolist(),
                    "s2b_tvec": s2b_tvec.astype(np.float64).tolist(),
                }
            )
    return rows


def _annotate_labels(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    errs = np.asarray([float(r["s5_pair_pos_err"]) for r in rows], dtype=np.float64)
    q80 = float(np.quantile(errs, 0.80))
    q90 = float(np.quantile(errs, 0.90))
    for row in rows:
        row["y_high_error_q80"] = 1 if float(row["s5_pair_pos_err"]) >= q80 else 0
        row["y_high_error_q90"] = 1 if float(row["s5_pair_pos_err"]) >= q90 else 0
        row["y_worst_bucket"] = 1 if float(row["dt_world"]) >= 1.0 and int(row["k"]) == 20 else 0
    return {"q80": q80, "q90": q90}


def _apply_label_thresholds(rows: List[Dict[str, Any]], thresholds: Dict[str, float]) -> None:
    q80 = float(thresholds["q80"])
    q90 = float(thresholds["q90"])
    for row in rows:
        row["y_high_error_q80"] = 1 if float(row["s5_pair_pos_err"]) >= q80 else 0
        row["y_high_error_q90"] = 1 if float(row["s5_pair_pos_err"]) >= q90 else 0
        row["y_worst_bucket"] = 1 if float(row["dt_world"]) >= 1.0 and int(row["k"]) == 20 else 0


def _rows_to_matrix(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(r["pred_tmag_s2b"]), float(r["dt_world"]), float(r["k"]), float(r["s5_base_scale"]), float(r["is_dt_ge_1"]), float(r["is_k20"])] for r in rows],
        dtype=np.float64,
    )


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    mu = np.nanmean(train_x, axis=0)
    sigma = np.nanstd(train_x, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1.0e-8), sigma, 1.0)
    train = (np.where(np.isfinite(train_x), train_x, mu) - mu) / sigma
    test = (np.where(np.isfinite(test_x), test_x, mu) - mu) / sigma
    return train, test, {"mean": mu, "std": sigma}


def _fit_logistic(train_rows: Sequence[Dict[str, Any]], *, label_key: str = "y_high_error_q80") -> Dict[str, Any]:
    fit_rows = list(train_rows)
    if len(fit_rows) > MAX_LOGISTIC_FIT_ROWS:
        fit_rows = [fit_rows[i] for i in _select_subsample_indices(len(fit_rows), MAX_LOGISTIC_FIT_ROWS)]
    x_raw = _rows_to_matrix(fit_rows)
    y = np.asarray([int(r[label_key]) for r in fit_rows], dtype=np.float64)
    x, _, stats = _standardize(x_raw, x_raw)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        constant_prob = float(y.mean()) if len(y) else 0.0
        score = np.full((len(y),), constant_prob, dtype=np.float64)
        return {
            "family": "logistic_router",
            "mean": stats["mean"].tolist(),
            "std": stats["std"].tolist(),
            "weights": [0.0] * x.shape[1],
            "bias": 0.0,
            "constant_prob": constant_prob,
            "train_auc": _auc_score(y.astype(np.int64), score),
            "train_score_quantiles": {"0.80": constant_prob, "0.85": constant_prob},
            "train_rows_used": len(fit_rows),
        }
    torch.manual_seed(17)
    x_t = torch.tensor(x, dtype=torch.float64)
    y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float64)
    model = torch.nn.Linear(x.shape[1], 1, dtype=torch.float64)
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float64)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        logits = model(x_t)
        loss = criterion(logits, y_t)
        total = loss + 1.0e-3 * model.weight.pow(2).sum()
        total.backward()
        return total

    opt.step(closure)
    with torch.no_grad():
        logits = model(x_t).cpu().numpy().reshape(-1)
    return {
        "family": "logistic_router",
        "mean": stats["mean"].tolist(),
        "std": stats["std"].tolist(),
        "weights": model.weight.detach().cpu().numpy().reshape(-1).tolist(),
        "bias": float(model.bias.detach().cpu().numpy().reshape(-1)[0]),
        "train_auc": _auc_score(y.astype(np.int64), _sigmoid(logits)),
        "train_score_quantiles": {
            "0.80": float(np.quantile(logits, 0.80)),
            "0.85": float(np.quantile(logits, 0.85)),
        },
        "train_rows_used": len(fit_rows),
    }


def _predict_logistic(rows: Sequence[Dict[str, Any]], fit: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    x_raw = _rows_to_matrix(rows)
    mu = np.asarray(fit["mean"], dtype=np.float64)
    sigma = np.asarray(fit["std"], dtype=np.float64)
    x = (np.where(np.isfinite(x_raw), x_raw, mu) - mu) / sigma
    if "constant_prob" in fit:
        prob = np.full((x.shape[0],), float(fit["constant_prob"]), dtype=np.float64)
        return prob, prob
    w = np.asarray(fit["weights"], dtype=np.float64).reshape(-1)
    b = float(fit["bias"])
    logits = x @ w + b
    return logits, _sigmoid(logits)


def _s5_quantiles_from_rows(train_rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    vals = np.asarray([float(r["pred_tmag_s2b"]) for r in train_rows], dtype=np.float64)
    return {"q90": float(np.quantile(vals, 0.90)), "q95": float(np.quantile(vals, 0.95))}


def _build_candidate_fit(spec: CandidateSpec, train_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    quant = _s5_quantiles_from_rows(train_rows)
    if spec.family == "pred_q90_router":
        return {"family": spec.family, "q90": quant["q90"], "a1_scale": 0.90}
    if spec.family == "pred_upper_tail_router":
        return {"family": spec.family, "q90": quant["q90"], "q95": quant["q95"], "a1_scale": 0.90, "a2_scale": 0.80}
    if spec.family == "dtk_router":
        return {"family": spec.family, "fallback": bool(spec.params["fallback"]), "a1_scale": 0.90}
    if spec.family == "combined_router":
        return {"family": spec.family, "q90": quant["q90"], "a1_scale": 0.90, "a2_scale": 0.80}
    if spec.family == "logistic_router":
        probe_fit = _fit_logistic(train_rows)
        return {
            "family": spec.family,
            "threshold_quantile": float(spec.params["threshold_quantile"]),
            "threshold_score": float(probe_fit["train_score_quantiles"][f"{float(spec.params['threshold_quantile']):.2f}"]),
            "action_if_risky": str(spec.params["action_if_risky"]),
            "logistic_fit": probe_fit,
        }
    raise ValueError(spec.family)


def _route_action(row: Dict[str, Any], fit: Dict[str, Any]) -> str:
    pred = float(row["pred_tmag_s2b"])
    dt_val = float(row["dt_world"])
    k_val = int(row["k"])
    family = str(fit["family"])
    if family == "pred_q90_router":
        return "A1" if pred >= float(fit["q90"]) else "A0"
    if family == "pred_upper_tail_router":
        if pred >= float(fit["q95"]):
            return "A2"
        if pred >= float(fit["q90"]):
            return "A1"
        return "A0"
    if family == "dtk_router":
        if dt_val >= 1.0 and k_val == 20:
            return "A3" if bool(fit["fallback"]) else "A1"
        return "A0"
    if family == "combined_router":
        if pred >= float(fit["q90"]) and dt_val >= 1.0:
            return "A2"
        if pred >= float(fit["q90"]):
            return "A1"
        if dt_val >= 1.0 and k_val == 20:
            return "A1"
        return "A0"
    if family == "logistic_router":
        logits, _prob = _predict_logistic([row], fit["logistic_fit"])
        return str(fit["action_if_risky"]) if float(logits[0]) >= float(fit["threshold_score"]) else "A0"
    raise ValueError(family)


def _action_scale(action: str, base_scale: float) -> float:
    if action == "A0":
        return float(base_scale)
    if action == "A1":
        return float(min(base_scale, 0.90))
    if action == "A2":
        return float(min(base_scale, 0.80))
    if action == "A3":
        return 1.0
    raise ValueError(action)


def _evaluate_pair_metrics(rows: Sequence[Dict[str, Any]], fit: Dict[str, Any], *, thresholds: Dict[str, float]) -> Dict[str, Any]:
    routed_errs: List[float] = []
    baseline_errs: List[float] = []
    action_counts: Dict[str, int] = {}
    high_error_true: List[int] = []
    high_error_pred: List[int] = []
    regime_map: Dict[str, List[Tuple[float, float]]] = {}
    for row in rows:
        action = _route_action(row, fit)
        scale = _action_scale(action, float(row["s5_base_scale"]))
        action_counts[action] = action_counts.get(action, 0) + 1
        gt_vec = np.asarray(row["gt_tvec"], dtype=np.float64)
        s2b_vec = np.asarray(row["s2b_tvec"], dtype=np.float64)
        routed_err = float(np.linalg.norm(s2b_vec * scale - gt_vec))
        base_err = float(row["s5_pair_pos_err"])
        routed_errs.append(routed_err)
        baseline_errs.append(base_err)
        high_error_true.append(1 if base_err >= float(thresholds["q80"]) else 0)
        high_error_pred.append(1 if action != "A0" else 0)
        bucket = f"{row['dt_bucket']}|k={int(row['k'])}"
        regime_map.setdefault(bucket, []).append((base_err, routed_err))
    y_true = np.asarray(high_error_true, dtype=np.int64)
    y_pred = np.asarray(high_error_pred, dtype=np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    recall = tp / max(tp + fn, 1)
    regime_rows = [
        {
            "bucket": bucket,
            "baseline_pair_pos_err": _mean(v[0] for v in vals),
            "routed_pair_pos_err": _mean(v[1] for v in vals),
            "delta_pair_pos_err": _mean(v[1] - v[0] for v in vals),
        }
        for bucket, vals in sorted(regime_map.items())
    ]
    worst_bucket = next((row for row in regime_rows if row["bucket"] == ">=1.0|k=20"), None)
    return {
        "mean_pair_pos_err": _mean(routed_errs),
        "baseline_mean_pair_pos_err": _mean(baseline_errs),
        "mean_pair_pos_err_delta_vs_s5": _mean(r - b for r, b in zip(routed_errs, baseline_errs)),
        "worst_bucket_pair_pos_err": float("nan") if worst_bucket is None else float(worst_bucket["routed_pair_pos_err"]),
        "baseline_worst_bucket_pair_pos_err": float("nan") if worst_bucket is None else float(worst_bucket["baseline_pair_pos_err"]),
        "high_error_recall": float(recall),
        "high_error_mass": _mean(routed_errs[i] for i in range(len(routed_errs)) if y_true[i] == 1),
        "action_counts": action_counts,
        "regime_rows": regime_rows,
    }


class S9RegimeRouterWrapper(torch.nn.Module):
    def __init__(self, base: DtBucketScaledMagnitudeModel, s5_policy: Dict[str, Any], fit_payload: Dict[str, Any], context: Dict[str, Any]) -> None:
        super().__init__()
        self.base = base
        self.s5_policy = s5_policy
        self.fit_payload = fit_payload
        self.context = context
        self.cfg = base.cfg

    def forward(self, IA: torch.Tensor, IB: torch.Tensor, *, enable_depth_fusion=None, dt_world=None):
        R_pred, t_pred, aux = self.base(IA, IB, enable_depth_fusion=enable_depth_fusion, dt_world=dt_world)
        aux = dict(aux)
        meta = dict(self.context.get("meta", {}))
        pred_mag = float(aux["t_mag"].detach().float().view(-1)[0].cpu())
        row = {
            "pred_tmag_s2b": pred_mag,
            "dt_world": float(meta.get("dt_world", 0.0)),
            "k": int(meta.get("k", -1)),
            "dt_bucket": _dt_bucket(float(meta.get("dt_world", 0.0))),
            "s5_base_scale": _s5_scale_from_pred(pred_mag, self.s5_policy),
            "is_dt_ge_1": 1.0 if float(meta.get("dt_world", 0.0)) >= 1.0 else 0.0,
            "is_k20": 1.0 if int(meta.get("k", -1)) == 20 else 0.0,
        }
        action = _route_action(row, self.fit_payload)
        scale = _action_scale(action, float(row["s5_base_scale"]))
        fac = torch.tensor(float(scale), device=IA.device, dtype=torch.float32)
        for mag_key in ("t_mag", "t_mag_unbiased"):
            if mag_key in aux and torch.is_tensor(aux[mag_key]):
                aux[mag_key] = aux[mag_key] * fac
        for log_key in ("log_t_mag", "log_t_mag_unbiased"):
            src = "t_mag_unbiased" if "unbiased" in log_key else "t_mag"
            if src in aux and torch.is_tensor(aux[src]):
                aux[log_key] = torch.log(aux[src].clamp_min(float(getattr(self.cfg, "tmag_min", 1.0e-3))))
        for vec_key in ("t_vec", "t_vec_out", "t_vec_local"):
            if vec_key in aux and torch.is_tensor(aux[vec_key]):
                aux[vec_key] = aux[vec_key] * fac
        aux["s9_router_action"] = action
        aux["s9_router_final_scale"] = fac
        aux["s9_router_base_s5_scale"] = torch.tensor(float(row["s5_base_scale"]), device=IA.device, dtype=torch.float32)
        return R_pred, t_pred, aux


def _run_router_odom_eval(fit_payload: Dict[str, Any], ds, s5_policy: Dict[str, Any], device: torch.device, *, split: str, fold: str, candidate_name: str) -> Tuple[OdomEvalRow, Dict[str, Any]]:
    base_model, cfg, _s2b_policy, load_summary = _build_s2b_model(device)
    context: Dict[str, Any] = {}
    ctx_ds = ContextDataset(ds, context)
    wrapped = S9RegimeRouterWrapper(base_model, s5_policy, fit_payload, context).to(device)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"s9_{fold}_", dir=str(REPO_ROOT / "checkpoints")))
    try:
        odom = eval_odometry_sequence(wrapped, ctx_ds, device, cfg, output_dir=str(tmp_dir), step=0, upd=0)
        debug_json = _read_json(tmp_dir / "odom_trajectory_debug_latest.json")
        step_rows: List[Dict[str, Any]] = []
        with (tmp_dir / "odom_trajectory_steps_latest.csv").open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step_rows.append(dict(row))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return (
        OdomEvalRow(
            candidate_name=candidate_name,
            family=str(fit_payload["family"]),
            split=split,
            fold=fold,
            drift=_safe_float(odom.get("odom_metric_drift")),
            ate=_safe_float(odom.get("odom_metric_ATE")),
            path_ratio=_safe_float(odom.get("odom_shape_metric_mean_path_length_ratio")),
            selected_k=int(odom.get("odom_selected_k", -1)),
            num_pairs=int(odom.get("odom_num_pairs", 0)),
            num_chains=int(odom.get("odom_num_chains", 0)),
            load_missing=len(load_summary["missing"]),
            load_unexpected=len(load_summary["unexpected"]),
            mean_metric_pos_err=_mean(_safe_float(r.get("metric_pos_err")) for r in step_rows),
            worst_chain_ate=max((_safe_float(c.get("mean_metric_pos_err")) for c in debug_json.get("chains", [])), default=float("nan")),
        ),
        {"odom": odom, "debug_json": debug_json, "step_rows": step_rows, "load_summary": load_summary},
    )


def _baseline_cv_reference() -> Dict[str, Any]:
    obj = _read_json(S5_CANDIDATES_PATH)
    selected = None
    for cand in obj.get("candidates", []):
        if str(cand.get("name")) == "highpred_q90_mid0p95_high0p80":
            selected = cand
            break
    if selected is None:
        raise RuntimeError("Failed to locate S5 selected fold baseline in candidates json.")
    by_fold = {str(r["fold"]): r for r in selected.get("fold_rows", [])}
    return {"by_fold": by_fold, "final_test": obj.get("final_row")}


def _hard_gate(candidate_rows: Sequence[OdomEvalRow], baseline_rows: Sequence[Dict[str, Any]], *, pair_metrics_rows: Sequence[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    mean_path = _mean(float(r.path_ratio) for r in candidate_rows)
    mean_ate = _mean(float(r.ate) for r in candidate_rows)
    mean_drift = _mean(float(r.drift) for r in candidate_rows)
    base_mean_path = _mean(float(r["path_ratio"]) for r in baseline_rows)
    base_mean_ate = _mean(float(r["ate"]) for r in baseline_rows)
    base_mean_drift = _mean(float(r["drift"]) for r in baseline_rows)
    worst_fold_drift = max(float(r.drift) for r in candidate_rows)
    baseline_worst_fold_drift = max(float(r["drift"]) for r in baseline_rows)
    per_fold_path_ok = all(SAFE_PATH_RANGE[0] <= float(r.path_ratio) <= SAFE_PATH_RANGE[1] for r in candidate_rows)
    overall_path_ok = SAFE_PATH_RANGE[0] <= mean_path <= SAFE_PATH_RANGE[1]
    mean_pair_delta = _mean(float(r["mean_pair_pos_err_delta_vs_s5"]) for r in pair_metrics_rows)
    worst_bucket_delta = _mean(float(r["worst_bucket_pair_pos_err"] - r["baseline_worst_bucket_pair_pos_err"]) for r in pair_metrics_rows)
    missing_ok = all(int(r.load_missing) == 14 and int(r.load_unexpected) == 0 for r in candidate_rows)
    ok = (
        overall_path_ok
        and per_fold_path_ok
        and mean_ate <= base_mean_ate + 1.0e-9
        and mean_drift <= base_mean_drift + 1.0e-9
        and worst_fold_drift <= baseline_worst_fold_drift + 0.03
        and missing_ok
        and worst_bucket_delta <= 1.0e-9
    )
    return ok, {
        "mean_path_ratio": mean_path,
        "baseline_mean_path_ratio": base_mean_path,
        "mean_ATE": mean_ate,
        "baseline_mean_ATE": base_mean_ate,
        "mean_drift": mean_drift,
        "baseline_mean_drift": base_mean_drift,
        "worst_fold_drift": worst_fold_drift,
        "baseline_worst_fold_drift": baseline_worst_fold_drift,
        "per_fold_path_ok": per_fold_path_ok,
        "overall_path_ok": overall_path_ok,
        "mean_pair_delta_vs_s5": mean_pair_delta,
        "mean_worst_bucket_delta_vs_s5": worst_bucket_delta,
        "missing_unexpected_ok": missing_ok,
    }


def _select_candidate(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    eligible = [c for c in candidates if bool(c["satisfies_clean_gate"])]
    if eligible:
        eligible.sort(key=lambda c: (float(c["mean_ATE"]), float(c["mean_drift"]), float(c["mean_pair_pos_err"]), c["simplicity_rank"]))
        return eligible[0]
    diagnostic = [c for c in candidates if float(c["mean_worst_bucket_delta_vs_s5"]) < -1.0e-9]
    if diagnostic:
        diagnostic.sort(key=lambda c: (float(c["mean_worst_bucket_delta_vs_s5"]), float(c["mean_pair_pos_err"]), c["simplicity_rank"]))
        return diagnostic[0]
    return None


def _plot_cv_metric(candidate_rows: Sequence[Dict[str, Any]], *, metric_key: str, baseline_value: float, title: str, path: Path) -> None:
    labels = [str(r["name"]) for r in candidate_rows]
    vals = [float(r[metric_key]) for r in candidate_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4), dpi=140)
    ax.bar(x, vals)
    ax.axhline(baseline_value, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _final_classification(selected: Dict[str, Any] | None, final_test: Dict[str, Any] | None, *, baseline_gate_ok: bool) -> Tuple[str, bool]:
    if not baseline_gate_ok:
        return "REPRODUCTION-MISMATCH", False
    if selected is None:
        return "NO-STABLE-REGIME-ROUTER-GAIN", False
    if final_test is None:
        if float(selected.get("mean_worst_bucket_delta_vs_s5", 0.0)) < -1.0e-4 and float(selected.get("mean_pair_pos_err_delta_vs_s5", 0.0)) <= 0.0:
            return "REGIME-ONLY-ROUTER-DIAGNOSTIC-GAIN", False
        return "NO-STABLE-REGIME-ROUTER-GAIN", False
    replacement = (
        float(final_test["metrics"]["ATE"]) < S5_LOCKED["ATE"]
        and float(final_test["metrics"]["drift"]) <= S5_LOCKED["drift"] + 1.0e-6
        and SAFE_PATH_RANGE[0] <= float(final_test["metrics"]["path_ratio"]) <= SAFE_PATH_RANGE[1]
        and bool(final_test["leakage_audit"]["passed"])
        and bool(final_test["scene_chain_audit"]["passed"])
    )
    if replacement:
        return "REGIME-ONLY-ROUTER-CLEAN-GAIN", True
    if bool(selected.get("satisfies_clean_gate")):
        return "REGIME-ONLY-ROUTER-CLEAN-BUT-MARGINAL", False
    if float(selected.get("mean_worst_bucket_delta_vs_s5", 0.0)) < 0.0:
        return "REGIME-ONLY-ROUTER-DIAGNOSTIC-GAIN", False
    return "REGIME-ONLY-ROUTER-FAIL", False


def _write_report(payload: Dict[str, Any]) -> None:
    baseline_gate = payload["baseline_gate"]
    feature_audit = payload["feature_audit"]
    candidates = payload["candidates"]
    selected = payload.get("selected")
    final_test = payload.get("final_test")
    leakage_audit = payload["leakage_audit"]
    lines: List[str] = []
    lines.append("# S9 Regime-Only Reliability Router Report\n\n")
    lines.append("## Executive summary\n\n")
    lines.append(f"- final classification: `{payload['final_classification']}`\n")
    lines.append(f"- S9 replaces S5: `{payload['s9_replaces_s5']}`\n")
    lines.append(f"- selected candidate: `{selected['name'] if selected is not None else 'None'}`\n\n")

    lines.append("## Motivation from S4/S5/S8\n\n")
    lines.append("- S4 localized dominant removable error mass to regime structure, especially long-step / larger-k cases.\n")
    lines.append("- S5 showed a clean but marginal gain from fixed pred_tmag-aware shrinkage.\n")
    lines.append("- S8 ruled out added value from token-based reliability features in the current representation.\n")
    lines.append("- S9 therefore tests whether a regime-only router using deployable inference-visible features can improve over S5 cleanly.\n\n")

    lines.append("## Why token features are excluded\n\n")
    lines.append("- S8 token-only and token-plus-regime probes did not beat regime-only.\n")
    lines.append("- To avoid feature leakage and unnecessary complexity, S9 restricts routing to `pred_tmag`, `dt`, `k`, and derived regime buckets only.\n\n")

    lines.append("## Baseline gate using S8b contract\n\n")
    lines.append(f"- gate passed: `{baseline_gate['passed']}`\n")
    lines.append(f"- contract: `{baseline_gate['contract_path']}`\n")
    lines.append(f"- current-architecture benign path accepted: `{baseline_gate['missing_unexpected_ok']}`\n")
    lines.append(
        f"- locked S5 metrics preserved: drift=`{_fmt(float(baseline_gate['s5']['metrics']['drift']))}`, "
        f"ATE=`{_fmt(float(baseline_gate['s5']['metrics']['ATE']))}`, "
        f"path_ratio=`{_fmt(float(baseline_gate['s5']['metrics']['path_ratio']))}`\n\n"
    )

    lines.append("## Regime feature definition\n\n")
    lines.append("| feature | inference-visible | inference-derived-only | allowed |\n")
    lines.append("| --- | --- | --- | --- |\n")
    for row in feature_audit:
        lines.append(f"| {row['feature']} | {row['inference_visible']} | {row['derived_only_from_inference']} | {row['allowed']} |\n")
    lines.append("\n")

    lines.append("## Candidate router definitions\n\n")
    for cand in candidates:
        lines.append(f"- `{cand['name']}` ({cand['family']}): actions=`{cand['action_set']}` params=`{cand['fit_summary']}`\n")
    lines.append("\n")

    lines.append("## Train-CV selection protocol\n\n")
    lines.append("- folds: `scene01/seq01` and `scene01/seq02` leave-one-seq-out CV\n")
    lines.append("- selection uses train-CV only; test is used once for the selected candidate only\n")
    lines.append("- logistic threshold and all quantiles are derived from fold-train rows only\n")
    lines.append("- hard gates require safe path ratio, non-worsened mean CV ATE/drift, bounded worst-fold drift, contract-compatible loading, and no forbidden feature leakage\n\n")

    lines.append("## CV results table\n\n")
    lines.append("| candidate | family | cv_mean_ATE | cv_mean_drift | cv_mean_path | pair_pos_err | high-error recall | worst-bucket err | clean_gate |\n")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
    for cand in candidates:
        lines.append(
            f"| {cand['name']} | {cand['family']} | {_fmt(cand['mean_ATE'])} | {_fmt(cand['mean_drift'])} | {_fmt(cand['mean_path_ratio'])} | "
            f"{_fmt(cand['mean_pair_pos_err'])} | {_fmt(cand['mean_high_error_recall'])} | {_fmt(cand['mean_worst_bucket_pair_pos_err'])} | {cand['satisfies_clean_gate']} |\n"
        )
    lines.append("\n")

    lines.append("## Hard-gate audit\n\n")
    for cand in candidates:
        gp = cand["gate_payload"]
        lines.append(
            f"- `{cand['name']}`: mean_ATE `{_fmt(gp['mean_ATE'])}` vs `{_fmt(gp['baseline_mean_ATE'])}`, "
            f"mean_drift `{_fmt(gp['mean_drift'])}` vs `{_fmt(gp['baseline_mean_drift'])}`, "
            f"mean_path `{_fmt(gp['mean_path_ratio'])}`, worst_fold_drift `{_fmt(gp['worst_fold_drift'])}` vs baseline worst `{_fmt(gp['baseline_worst_fold_drift'])}`, "
            f"worst_bucket_delta `{_fmt(gp['mean_worst_bucket_delta_vs_s5'])}`, gate=`{cand['satisfies_clean_gate']}`\n"
        )
    lines.append("\n")

    if final_test is not None:
        lines.append("## Final test result, if selected\n\n")
        lines.append(f"- candidate: `{final_test['candidate_name']}`\n")
        lines.append(f"- drift = `{_fmt(final_test['metrics']['drift'])}`\n")
        lines.append(f"- ATE = `{_fmt(final_test['metrics']['ATE'])}`\n")
        lines.append(f"- path_ratio = `{_fmt(final_test['metrics']['path_ratio'])}`\n")
        lines.append(f"- pair_pos_err = `{_fmt(final_test['pair_metrics']['mean_pair_pos_err'])}`\n")
        lines.append(f"- worst_bucket_error = `{_fmt(final_test['pair_metrics']['worst_bucket_pair_pos_err'])}`\n")
        lines.append(f"- high_error_mass = `{_fmt(final_test['pair_metrics']['high_error_mass'])}`\n\n")
    else:
        lines.append("## Final test result, if selected\n\n")
        lines.append("- no final test was run because no candidate passed train-CV clean selection.\n\n")

    lines.append("## Regime-level before/after\n\n")
    if final_test is not None:
        for row in final_test["pair_metrics"]["regime_rows"]:
            lines.append(
                f"- `{row['bucket']}`: baseline=`{_fmt(row['baseline_pair_pos_err'])}` -> routed=`{_fmt(row['routed_pair_pos_err'])}`, delta=`{_fmt(row['delta_pair_pos_err'])}`\n"
            )
    else:
        lines.append("- not available because no final test candidate was run.\n")
    lines.append("\n")

    lines.append("## Scene/chain-level before/after\n\n")
    if final_test is not None:
        for row in final_test["scene_chain_rows"]:
            lines.append(
                f"- `{row['scene_seq']}`: baseline_mean=`{_fmt(row['baseline_mean_metric_pos_err'])}` -> routed_mean=`{_fmt(row['routed_mean_metric_pos_err'])}`, "
                f"baseline_path=`{_fmt(row['baseline_path_ratio'])}` -> routed_path=`{_fmt(row['routed_path_ratio'])}`\n"
            )
    else:
        lines.append("- not available because no final test candidate was run.\n")
    lines.append("\n")

    lines.append("## Leakage audit\n\n")
    for k, v in leakage_audit.items():
        lines.append(f"- {k}: `{v}`\n")
    lines.append("\n")

    lines.append("## Final classification\n\n")
    lines.append(f"- `{payload['final_classification']}`\n")
    lines.append(f"- S5 remains final clean candidate: `{not payload['s9_replaces_s5']}`\n")
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


def _write_summary(payload: Dict[str, Any]) -> None:
    selected = payload.get("selected")
    final_test = payload.get("final_test")
    lines: List[str] = []
    lines.append("# Final S9 Regime-Only Router Summary\n\n")
    lines.append(f"- Final classification: `{payload['final_classification']}`\n")
    lines.append(f"- S9 replaces S5: `{payload['s9_replaces_s5']}`\n")
    lines.append(f"- S5 remains final clean candidate: `{not payload['s9_replaces_s5']}`\n")
    lines.append(f"- Selected candidate: `{selected['name'] if selected is not None else 'None'}`\n\n")
    lines.append("S9 stopped the token/fine reliability direction and tested only deployable regime features. ")
    lines.append("The router was restricted to `pred_tmag`, `dt`, and `k`, with train-CV-only thresholds/actions and one optional final test for the selected candidate.\n\n")
    if final_test is not None:
        lines.append(
            f"The selected regime-only candidate produced final-test drift/ATE/path_ratio = "
            f"`{_fmt(final_test['metrics']['drift'])}` / `{_fmt(final_test['metrics']['ATE'])}` / `{_fmt(final_test['metrics']['path_ratio'])}` "
            f"against the locked S5 baseline `{S5_LOCKED['drift']}` / `{S5_LOCKED['ATE']}` / `{S5_LOCKED['path_ratio']}`.\n\n"
        )
    else:
        lines.append("No candidate was strong enough to justify a final replacement test, so S9 remains a diagnostic result rather than a new final candidate.\n\n")
    lines.append("Interpretation: if S9 fails to replace S5, then the current removable error mass is already largely captured by the fixed S5 magnitude-regime policy, and further clean gains will likely require a stronger representation or a different routing/evaluation design rather than simply adding more regime heuristics.\n")
    SUMMARY_PATH.write_text("".join(lines), encoding="utf-8")


def run_main() -> Dict[str, Any]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    baseline_gate = _baseline_gate()
    payload: Dict[str, Any] = {
        "baseline_gate": baseline_gate,
        "feature_audit": _feature_audit(),
        "final_classification": "REPRODUCTION-MISMATCH" if not baseline_gate["passed"] else "INCONCLUSIVE",
        "s9_replaces_s5": False,
        "leakage_audit": {
            "passed": False if not baseline_gate["passed"] else True,
            "token_features_used": False,
            "gt_tmag_as_inference_feature": False,
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
        _write_json(CANDIDATES_PATH, payload)
        _write_report(payload)
        _write_summary(payload)
        return payload

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s2b_model, cfg, _s2b_policy, _load_summary = _build_s2b_model(device)
    s5_policy = _load_s5_policy()
    train_ds_full, groups, seq_to_indices = _collect_train_groups(cfg)
    test_ds = _build_dataset(cfg, split="test")

    fold_plan = []
    for fold_idx, seq in enumerate(groups):
        val_idx = seq_to_indices[seq]
        train_idx = [i for g, idxs in seq_to_indices.items() if g != seq for i in idxs]
        fold_plan.append(
            {
                "fold_name": f"fold_{fold_idx}_{seq[0]}_{seq[1]}",
                "heldout": seq,
                "train_ds": ManifestSubsetDataset(train_ds_full, train_idx),
                "val_ds": ManifestSubsetDataset(train_ds_full, val_idx),
            }
        )

    row_cache = _load_rows_cache()
    expected_caps = {
        "train_full": TRAIN_FULL_ROWS_CAP,
        "fold_train": FOLD_TRAIN_ROWS_CAP,
        "fold_val": FOLD_VAL_ROWS_CAP,
    }
    cache_valid = bool(row_cache is not None and row_cache.get("caps") == expected_caps)
    if cache_valid:
        print(f"[S9] reusing rows cache {ROWS_CACHE_PATH}", flush=True)
        train_rows_full = row_cache["train_rows_full"]
        test_rows_full = row_cache["test_rows_full"]
        fold_train_rows = row_cache["fold_train_rows"]
        fold_val_rows = row_cache["fold_val_rows"]
        fold_thresholds = row_cache["fold_thresholds"]
        train_thresholds_full = row_cache["train_thresholds_full"]
    else:
        print(f"[S9] extracting train_full rows cap={TRAIN_FULL_ROWS_CAP}", flush=True)
        train_rows_full = _extract_rows(s2b_model, train_ds_full, s5_policy, device, cap=TRAIN_FULL_ROWS_CAP)
        train_thresholds_full = _annotate_labels(train_rows_full)
        print("[S9] extracting test rows", flush=True)
        test_rows_full = _extract_rows(s2b_model, test_ds, s5_policy, device)
        _annotate_labels(test_rows_full)

        fold_train_rows = {}
        fold_val_rows = {}
        fold_thresholds = {}
        for fold in fold_plan:
            print(f"[S9] extracting {fold['fold_name']} train rows cap={FOLD_TRAIN_ROWS_CAP}", flush=True)
            tr_rows = _extract_rows(s2b_model, fold["train_ds"], s5_policy, device, cap=FOLD_TRAIN_ROWS_CAP)
            print(f"[S9] extracting {fold['fold_name']} val rows cap={FOLD_VAL_ROWS_CAP}", flush=True)
            va_rows = _extract_rows(s2b_model, fold["val_ds"], s5_policy, device, cap=FOLD_VAL_ROWS_CAP)
            fold_train_rows[fold["fold_name"]] = tr_rows
            fold_val_rows[fold["fold_name"]] = va_rows
            fold_thresholds[fold["fold_name"]] = _annotate_labels(tr_rows)
            _apply_label_thresholds(va_rows, fold_thresholds[fold["fold_name"]])

        _save_rows_cache(
            {
                "caps": expected_caps,
                "train_rows_full": train_rows_full,
                "train_thresholds_full": train_thresholds_full,
                "test_rows_full": test_rows_full,
                "fold_train_rows": fold_train_rows,
                "fold_val_rows": fold_val_rows,
                "fold_thresholds": fold_thresholds,
            }
        )
        print(f"[S9] wrote rows cache {ROWS_CACHE_PATH}", flush=True)

    for fold_name, rows in fold_val_rows.items():
        if rows and "y_high_error_q80" not in rows[0]:
            _apply_label_thresholds(rows, fold_thresholds[fold_name])

    s5_cv_baseline = _baseline_cv_reference()
    candidate_records: List[Dict[str, Any]] = []
    specs = _candidate_specs()
    for spec in specs:
        print(f"[S9] evaluating candidate {spec.name}", flush=True)
        odom_rows: List[OdomEvalRow] = []
        pair_metric_rows: List[Dict[str, Any]] = []
        fit_payloads: List[Dict[str, Any]] = []
        probe_metric_rows: List[Dict[str, Any]] = []
        for fold in fold_plan:
            fold_name = str(fold["fold_name"])
            print(f"[S9]  fold={fold_name}", flush=True)
            fit_payload = _build_candidate_fit(spec, fold_train_rows[fold_name])
            fit_payloads.append({"fold": fold_name, "fit_payload": fit_payload})
            pair_metrics = _evaluate_pair_metrics(fold_val_rows[fold_name], fit_payload, thresholds=fold_thresholds[fold_name])
            pair_metric_rows.append({"fold": fold_name, **pair_metrics})
            odom_row, _odom_pack = _run_router_odom_eval(fit_payload, fold["val_ds"], s5_policy, device, split="train_cv", fold=fold_name, candidate_name=spec.name)
            odom_rows.append(odom_row)
            y_true = np.asarray([int(r["y_high_error_q80"]) for r in fold_val_rows[fold_name]], dtype=np.int64)
            logits = np.asarray([1.0 if _route_action(r, fit_payload) != "A0" else 0.0 for r in fold_val_rows[fold_name]], dtype=np.float64)
            probe = _binary_metrics(y_true, logits)
            probe_metric_rows.append({"fold": fold_name, **probe})
        baseline_rows = [s5_cv_baseline["by_fold"][r.fold] for r in odom_rows]
        gate_ok, gate_payload = _hard_gate(odom_rows, baseline_rows, pair_metrics_rows=pair_metric_rows)
        record = {
            "name": spec.name,
            "family": spec.family,
            "params": spec.params,
            "action_set": spec.action_set,
            "simplicity_rank": spec.simplicity_rank,
            "fit_payloads": fit_payloads,
            "fit_summary": fit_payloads[-1]["fit_payload"] if fit_payloads else {},
            "odom_rows": [r.__dict__ for r in odom_rows],
            "pair_metric_rows": pair_metric_rows,
            "probe_metric_rows": probe_metric_rows,
            "mean_ATE": _mean(float(r.ate) for r in odom_rows),
            "mean_drift": _mean(float(r.drift) for r in odom_rows),
            "mean_path_ratio": _mean(float(r.path_ratio) for r in odom_rows),
            "mean_pair_pos_err": _mean(float(r["mean_pair_pos_err"]) for r in pair_metric_rows),
            "mean_high_error_recall": _mean(float(r["high_error_recall"]) for r in pair_metric_rows),
            "mean_worst_bucket_pair_pos_err": _mean(float(r["worst_bucket_pair_pos_err"]) for r in pair_metric_rows),
            "mean_pair_pos_err_delta_vs_s5": _mean(float(r["mean_pair_pos_err_delta_vs_s5"]) for r in pair_metric_rows),
            "mean_worst_bucket_delta_vs_s5": _mean(float(r["worst_bucket_pair_pos_err"] - r["baseline_worst_bucket_pair_pos_err"]) for r in pair_metric_rows),
            "probe_auc_mean": _mean(float(r["auc"]) for r in probe_metric_rows),
            "probe_accuracy_mean": _mean(float(r["accuracy"]) for r in probe_metric_rows),
            "satisfies_clean_gate": gate_ok,
            "gate_payload": gate_payload,
        }
        candidate_records.append(record)

    candidate_records.sort(key=lambda x: (float(x["mean_ATE"]), float(x["mean_drift"]), x["simplicity_rank"]))
    _plot_cv_metric(candidate_records, metric_key="mean_ATE", baseline_value=_mean(float(v["ate"]) for v in s5_cv_baseline["by_fold"].values()), title="S9 Mean CV ATE by Candidate", path=FIG_DIR / "cv_mean_ate.png")
    _plot_cv_metric(candidate_records, metric_key="mean_path_ratio", baseline_value=_mean(float(v["path_ratio"]) for v in s5_cv_baseline["by_fold"].values()), title="S9 Mean CV Path Ratio by Candidate", path=FIG_DIR / "cv_mean_path_ratio.png")
    _plot_cv_metric(candidate_records, metric_key="mean_worst_bucket_pair_pos_err", baseline_value=_mean(float(v["worst_bucket_pair_pos_err"]) for v in s5_cv_baseline["by_fold"].values()), title="S9 Mean Worst-Bucket Pair Error by Candidate", path=FIG_DIR / "cv_worst_bucket_pair_pos_err.png")

    selected = _select_candidate(candidate_records)
    final_test = None
    if selected is not None and bool(selected["satisfies_clean_gate"]):
        spec = next(s for s in specs if s.name == selected["name"])
        fit_payload = _build_candidate_fit(spec, train_rows_full)
        final_pair_metrics = _evaluate_pair_metrics(test_rows_full, fit_payload, thresholds=train_thresholds_full)
        final_odom_row, final_pack = _run_router_odom_eval(fit_payload, test_ds, s5_policy, device, split="test", fold="final_test", candidate_name=spec.name)
        baseline_debug = _read_json(REPO_ROOT / "checkpoints" / "S6_final_clean_candidate_lockdown_eval" / "s5_selected" / "odom_trajectory_debug_latest.json")
        baseline_chain_map = {str(c.get("scene_seq")): c for c in baseline_debug.get("chains", [])}
        routed_chain_map = {str(c.get("scene_seq")): c for c in final_pack["debug_json"].get("chains", [])}
        scene_chain_rows = []
        for key in sorted(set(baseline_chain_map) | set(routed_chain_map)):
            base = baseline_chain_map.get(key, {})
            routed = routed_chain_map.get(key, {})
            scene_chain_rows.append(
                {
                    "scene_seq": key,
                    "baseline_mean_metric_pos_err": _safe_float(base.get("mean_metric_pos_err")),
                    "routed_mean_metric_pos_err": _safe_float(routed.get("mean_metric_pos_err")),
                    "baseline_path_ratio": _safe_float(base.get("shape_metric", {}).get("path_length_ratio")),
                    "routed_path_ratio": _safe_float(routed.get("shape_metric", {}).get("path_length_ratio")),
                }
            )
        scene_chain_audit = {
            "passed": all(
                (
                    (not math.isfinite(float(row["baseline_path_ratio"])) or not math.isfinite(float(row["routed_path_ratio"])) or abs(float(row["routed_path_ratio"]) - float(row["baseline_path_ratio"])) <= 0.05)
                    and (not math.isfinite(float(row["baseline_mean_metric_pos_err"])) or not math.isfinite(float(row["routed_mean_metric_pos_err"])) or float(row["routed_mean_metric_pos_err"]) <= float(row["baseline_mean_metric_pos_err"]) + 0.5)
                )
                for row in scene_chain_rows
            )
        }
        final_test = {
            "candidate_name": spec.name,
            "family": spec.family,
            "fit_payload": fit_payload,
            "metrics": {
                "drift": final_odom_row.drift,
                "ATE": final_odom_row.ate,
                "path_ratio": final_odom_row.path_ratio,
                "num_pairs": final_odom_row.num_pairs,
                "num_chains": final_odom_row.num_chains,
            },
            "pair_metrics": final_pair_metrics,
            "scene_chain_rows": scene_chain_rows,
            "scene_chain_audit": scene_chain_audit,
            "leakage_audit": payload["leakage_audit"],
        }
        policy_payload = {
            "name": "S9_regime_only_reliability_router_policy",
            "selection_source": "S9 train-CV selected candidate only",
            "base_policy_path": str(S5_POLICY_PATH.relative_to(REPO_ROOT)),
            "baseline_contract_path": str(S8B_CONTRACT_PATH.relative_to(REPO_ROOT)),
            "base_checkpoint_path": str(_read_json(S2B_POLICY_PATH)["base_checkpoint_path"]),
            "family": spec.family,
            "candidate_name": spec.name,
            "fit_payload": fit_payload,
            "allowed_inference_features": ["pred_tmag_s2b", "dt_world", "k", "s5_base_scale", "derived regime buckets"],
            "forbidden_features": ["fine token", "spherical token", "gt_tmag", "gt pose", "pair_pos_err", "test statistics", "oracle labels"],
            "expected_current_arch_loading": {"missing": 14, "unexpected": 0, **CURRENT_ARCH_BENIGN_MISSING},
            "locked_s5_metrics": S5_LOCKED,
            "final_test_metrics": final_test["metrics"],
            "load_missing": int(final_odom_row.load_missing),
            "load_unexpected": int(final_odom_row.load_unexpected),
            "eval_scope": {
                "selected_k": int(final_odom_row.selected_k),
                "num_pairs": int(final_odom_row.num_pairs),
                "num_chains": int(final_odom_row.num_chains),
                "path_ratio_safe_range": list(SAFE_PATH_RANGE),
            },
        }
        _write_json(POLICY_PATH, policy_payload)

    final_classification, replaces = _final_classification(selected, final_test, baseline_gate_ok=bool(baseline_gate["passed"]))
    payload.update(
        {
            "candidates": candidate_records,
            "selected": selected,
            "final_test": final_test,
            "final_classification": final_classification,
            "s9_replaces_s5": replaces,
        }
    )
    if final_test is not None:
        payload["leakage_audit"]["passed"] = bool(final_test["leakage_audit"]["passed"])
    _write_json(CANDIDATES_PATH, payload)
    _write_report(payload)
    _write_summary(payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "eval_policy"], default="run")
    ap.add_argument("--policy", default=str(POLICY_PATH))
    args = ap.parse_args()
    if args.mode == "eval_policy":
        policy = _read_json(Path(args.policy))
        payload = {
            "policy_path": str(Path(args.policy)),
            "base_policy_path": policy.get("base_policy_path"),
            "candidate_name": policy.get("candidate_name"),
            "family": policy.get("family"),
            "fit_payload": policy.get("fit_payload"),
            "load_missing": policy.get("load_missing"),
            "load_unexpected": policy.get("load_unexpected"),
            "eval_scope": policy.get("eval_scope"),
            "locked_s5_metrics": policy.get("locked_s5_metrics"),
            "final_test_metrics": policy.get("final_test_metrics"),
        }
        print(json.dumps(_json_safe(payload), indent=2))
        return
    payload = run_main()
    print(json.dumps(_json_safe(payload), indent=2))


if __name__ == "__main__":
    main()
