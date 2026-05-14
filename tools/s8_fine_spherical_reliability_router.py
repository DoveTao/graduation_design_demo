#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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


S2B_POLICY_PATH = REPO_ROOT / "checkpoints" / "S2b_clean_fine_rot_policy.json"
S5_POLICY_PATH = REPO_ROOT / "checkpoints" / "S5_clean_tmag_calibration_policy.json"
S8B_CONTRACT_PATH = REPO_ROOT / "checkpoints" / "S8b_reproduction_contract.json"
S8B_S2B_RESULT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_s2b_wrapper_result.json"
S8B_S5_RESULT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_s5_wrapper_result.json"

DEFAULT_CACHE_PATH = REPO_ROOT / "checkpoints" / "S8_fine_spherical_reliability_router_rows_cache.jsonl"
DEFAULT_CANDIDATES_PATH = REPO_ROOT / "checkpoints" / "S8_fine_spherical_reliability_router_candidates.json"
DEFAULT_REPORT_PATH = REPO_ROOT / "checkpoints" / "S8_fine_spherical_reliability_router_report.md"

S5_LOCKED = {"drift": 1.327343, "ATE": 7.352288, "path_ratio": 0.932379}
CURRENT_ARCH_BENIGN_MISSING = {
    "ridge_calib_buffers": 2,
    "coupled_pose_head_params": 12,
    "other_missing": 0,
}

DEFAULT_TRAIN_FULL_CAP = 512
DEFAULT_FOLD_CAP = 256
DEFAULT_FLUSH_EVERY = 32
MIN_TRAIN_ROWS = 32
MIN_VAL_ROWS = 16


@dataclass(frozen=True)
class SplitSpec:
    name: str
    dataset_kind: str
    dataset_indices: Sequence[int]
    cap: int
    seed: int


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


def _append_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=True) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


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


def _binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(np.int64)
    y_pred = (prob >= threshold).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    acc = (tp + tn) / max(len(y_true), 1)
    return {
        "auc": _auc_score(y_true, prob),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
    }


def _mean(vals: Iterable[float]) -> float:
    arr = np.asarray(list(vals), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _std(vals: Iterable[float]) -> float:
    arr = np.asarray(list(vals), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else float("nan")
    return float(arr.std(ddof=0))


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


def _select_stratified_indices(ds: RflyPanoPanoramaPairsEvalFixedKList, limit: int, *, seed: int, allowed_indices: Sequence[int] | None = None) -> List[int]:
    if allowed_indices is None:
        allowed = list(range(len(ds.manifest())))
    else:
        allowed = [int(i) for i in allowed_indices]
    if limit <= 0 or len(allowed) <= limit:
        return list(allowed)
    manifest = ds.manifest()
    groups: Dict[Tuple[str, str, int], List[int]] = {}
    for idx in allowed:
        meta = manifest[idx]
        key = (str(meta.get("scene")), str(meta.get("seq")), int(meta.get("k", -1)))
        groups.setdefault(key, []).append(idx)
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    total = len(allowed)
    for key, idxs in sorted(groups.items(), key=lambda kv: kv[0]):
        _ = key
        share = max(1, int(round(limit * len(idxs) / total)))
        take = min(len(idxs), share)
        chosen = rng.choice(np.asarray(idxs, dtype=np.int64), size=take, replace=False)
        selected.extend(int(x) for x in chosen.tolist())
    selected = sorted(set(selected))
    if len(selected) > limit:
        selected = sorted(rng.choice(np.asarray(selected, dtype=np.int64), size=limit, replace=False).tolist())
    return selected[:limit]


def _transport_entropy(W: torch.Tensor) -> torch.Tensor:
    W = W.detach().float().clamp_min(1.0e-9)
    return -(W * W.log()).sum(dim=-1).mean(dim=-1)


def _transport_confidence(W: torch.Tensor) -> torch.Tensor:
    return W.detach().float().max(dim=-1).values.mean(dim=-1)


def _weight_stats(weight: torch.Tensor) -> Dict[str, float]:
    w = weight.detach().float().view(-1)
    return {
        "mean": float(w.mean().cpu()),
        "max": float(w.max().cpu()),
        "std": float(w.std(unbiased=False).cpu()),
        "entropy": float((-(w.clamp_min(1.0e-9) * w.clamp_min(1.0e-9).log()).sum()).cpu()),
    }


def _allowed_mask_density(mask: torch.Tensor | None) -> float:
    if mask is None:
        return float("nan")
    return float(mask.detach().float().mean().cpu())


def _no_candidate_row_ratio(mask: torch.Tensor | None) -> float:
    if mask is None:
        return float("nan")
    row_any = mask.detach().bool().any(dim=-1).float()
    return float((1.0 - row_any).mean().cpu())


def _summarize_vector(prefix: str, vec: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    abs_arr = np.abs(arr)
    q10, q50, q90 = np.quantile(arr, [0.10, 0.50, 0.90])
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_l2": float(np.linalg.norm(arr)),
        f"{prefix}_abs_mean": float(abs_arr.mean()),
        f"{prefix}_abs_p90": float(np.quantile(abs_arr, 0.90)),
        f"{prefix}_median": float(q50),
        f"{prefix}_q10": float(q10),
        f"{prefix}_q90": float(q90),
    }


def _pool_token_features(feat: torch.Tensor, weight: torch.Tensor | None = None) -> np.ndarray:
    feat = feat.detach().float()
    if feat.dim() != 3 or feat.shape[0] != 1:
        raise ValueError(f"Unexpected token feature shape: {tuple(feat.shape)}")
    feat = feat[0]
    if weight is None:
        mean = feat.mean(dim=0)
        diff = feat - mean.unsqueeze(0)
        std = torch.sqrt(diff.pow(2).mean(dim=0).clamp_min(1.0e-8))
        maxv = feat.max(dim=0).values
    else:
        w = weight.detach().float().view(-1)
        w = w / w.sum().clamp_min(1.0e-8)
        mean = (feat * w.unsqueeze(-1)).sum(dim=0)
        diff = feat - mean.unsqueeze(0)
        std = torch.sqrt((diff.pow(2) * w.unsqueeze(-1)).sum(dim=0).clamp_min(1.0e-8))
        maxv = feat.max(dim=0).values
    return torch.cat([mean, maxv, std], dim=0).cpu().numpy().astype(np.float64)


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


def _feature_audit() -> List[Dict[str, Any]]:
    return [
        {"feature": "pred_tmag_s2b", "shape": "[1]", "source_module": "model.forward aux['t_mag'] after S2b", "inference_visible": True, "deployable_safe": True},
        {"feature": "dt_world", "shape": "[1]", "source_module": "dataset meta['dt_world']", "inference_visible": True, "deployable_safe": True},
        {"feature": "k", "shape": "[1]", "source_module": "dataset meta['k']", "inference_visible": True, "deployable_safe": True},
        {"feature": "fine token summary", "shape": "[16]", "source_module": "weighted pooled Ff_t compressed to scalar stats", "inference_visible": True, "deployable_safe": True},
        {"feature": "spherical token summary", "shape": "[12]", "source_module": "weighted pooled Fc compressed to scalar stats", "inference_visible": True, "deployable_safe": True},
        {"feature": "gt pose / gt_tmag / pair_pos_err", "shape": "n/a", "source_module": "diagnostics labels only", "inference_visible": False, "deployable_safe": False},
    ]


FINE_TOKEN_KEYS = [
    "fine_pool_mean",
    "fine_pool_std",
    "fine_pool_min",
    "fine_pool_max",
    "fine_pool_l2",
    "fine_pool_abs_mean",
    "fine_pool_abs_p90",
    "fine_pool_median",
    "fine_pool_q10",
    "fine_pool_q90",
    "fine_confidence",
    "fine_entropy",
    "fine_weight_mean",
    "fine_weight_max",
    "fine_weight_std",
    "fine_weight_entropy",
]

SPHERICAL_TOKEN_KEYS = [
    "spherical_pool_mean",
    "spherical_pool_std",
    "spherical_pool_min",
    "spherical_pool_max",
    "spherical_pool_l2",
    "spherical_pool_abs_mean",
    "spherical_pool_abs_p90",
    "spherical_pool_median",
    "spherical_pool_q10",
    "spherical_pool_q90",
    "coarse_confidence",
    "coarse_entropy",
]

REGIME_KEYS = ["pred_tmag_s2b", "dt_world", "k", "s5_base_scale", "is_dt_ge_1", "is_k20"]

FEATURE_FAMILIES: Dict[str, List[str]] = {
    "regime_only": REGIME_KEYS,
    "fine_token_only": FINE_TOKEN_KEYS,
    "spherical_token_only": SPHERICAL_TOKEN_KEYS,
    "token_plus_regime": FINE_TOKEN_KEYS + SPHERICAL_TOKEN_KEYS + REGIME_KEYS,
}


def _row_key(split_name: str, ds_idx: int) -> str:
    return f"{split_name}::{int(ds_idx)}"


def _rows_to_matrix(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(r.get(key, 0.0)) for key in keys] for r in rows], dtype=np.float64)


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    mu = np.nanmean(train_x, axis=0)
    sigma = np.nanstd(train_x, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1.0e-8), sigma, 1.0)
    train = (np.where(np.isfinite(train_x), train_x, mu) - mu) / sigma
    test = (np.where(np.isfinite(test_x), test_x, mu) - mu) / sigma
    return train, test, {"mean": mu, "std": sigma}


def _fit_logistic(train_rows: Sequence[Dict[str, Any]], family: str, *, label_key: str, seed: int) -> Dict[str, Any]:
    x_raw = _rows_to_matrix(train_rows, FEATURE_FAMILIES[family])
    y = np.asarray([int(r[label_key]) for r in train_rows], dtype=np.float64)
    x, _, stats = _standardize(x_raw, x_raw)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        constant_prob = float(y.mean()) if len(y) else 0.0
        score = np.full((len(y),), constant_prob, dtype=np.float64)
        return {
            "family": family,
            "feature_keys": FEATURE_FAMILIES[family],
            "mean": stats["mean"].tolist(),
            "std": stats["std"].tolist(),
            "weights": [0.0] * x.shape[1],
            "bias": math.log(constant_prob / max(1.0 - constant_prob, 1.0e-6)) if 0.0 < constant_prob < 1.0 else 0.0,
            "constant_prob": constant_prob,
            "label_key": label_key,
            "train_auc": _auc_score(y.astype(np.int64), score),
            "train_accuracy": float(np.mean((score >= 0.5).astype(np.float64) == y)),
        }
    torch.manual_seed(seed)
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
        l2 = 1.0e-3 * model.weight.pow(2).sum()
        total = loss + l2
        total.backward()
        return total

    opt.step(closure)
    with torch.no_grad():
        logits = model(x_t).cpu().numpy().reshape(-1)
    prob = _sigmoid(logits)
    return {
        "family": family,
        "feature_keys": FEATURE_FAMILIES[family],
        "mean": stats["mean"].tolist(),
        "std": stats["std"].tolist(),
        "weights": model.weight.detach().cpu().numpy().reshape(-1).tolist(),
        "bias": float(model.bias.detach().cpu().numpy().reshape(-1)[0]),
        "label_key": label_key,
        "train_auc": _auc_score(y.astype(np.int64), prob),
        "train_accuracy": float(np.mean((prob >= 0.5).astype(np.float64) == y)),
    }


def _predict_family(rows: Sequence[Dict[str, Any]], fit: Dict[str, Any]) -> np.ndarray:
    x_raw = _rows_to_matrix(rows, fit["feature_keys"])
    mu = np.asarray(fit["mean"], dtype=np.float64)
    sigma = np.asarray(fit["std"], dtype=np.float64)
    x = (np.where(np.isfinite(x_raw), x_raw, mu) - mu) / sigma
    if "constant_prob" in fit:
        return np.full((x.shape[0],), float(fit["constant_prob"]), dtype=np.float64)
    w = np.asarray(fit["weights"], dtype=np.float64).reshape(-1)
    b = float(fit["bias"])
    logits = x @ w + b
    return _sigmoid(logits)


def _worst_bucket_recall(rows: Sequence[Dict[str, Any]], y_true: np.ndarray, prob: np.ndarray, *, threshold: float = 0.5) -> float:
    pred = (prob >= threshold).astype(np.int64)
    mask = np.asarray([float(r["dt_world"]) >= 1.0 and int(r["k"]) == 20 for r in rows], dtype=bool)
    pos = mask & (y_true == 1)
    if int(pos.sum()) == 0:
        return float("nan")
    return float(((pred == 1) & pos).sum() / max(pos.sum(), 1))


def _annotate_labels(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    errs = np.asarray([float(r["s5_pair_pos_err"]) for r in rows], dtype=np.float64)
    if len(errs) == 0:
        return {"q80": float("nan"), "q90": float("nan")}
    q80 = float(np.quantile(errs, 0.80))
    q90 = float(np.quantile(errs, 0.90))
    for row in rows:
        row["y_high_error_q80"] = 1 if float(row["s5_pair_pos_err"]) >= q80 else 0
        row["y_high_error_q90"] = 1 if float(row["s5_pair_pos_err"]) >= q90 else 0
        row["y_worst_regime"] = 1 if float(row["dt_world"]) >= 1.0 and int(row["k"]) == 20 else 0
    return {"q80": q80, "q90": q90}


def _apply_fold_labels(rows: List[Dict[str, Any]], thresholds: Dict[str, float]) -> None:
    q80 = float(thresholds["q80"])
    q90 = float(thresholds["q90"])
    for row in rows:
        row["y_high_error_q80"] = 1 if float(row["s5_pair_pos_err"]) >= q80 else 0
        row["y_high_error_q90"] = 1 if float(row["s5_pair_pos_err"]) >= q90 else 0
        row["y_worst_regime"] = 1 if float(row["dt_world"]) >= 1.0 and int(row["k"]) == 20 else 0


def _family_eval(rows: Sequence[Dict[str, Any]], fit: Dict[str, Any], *, label_key: str = "y_high_error_q80") -> Dict[str, float]:
    y_true = np.asarray([int(r[label_key]) for r in rows], dtype=np.int64)
    prob = _predict_family(rows, fit)
    metrics = _binary_metrics(y_true, prob)
    metrics["high_error_recall"] = metrics["recall"]
    metrics["worst_bucket_recall_ge_1p0_k20"] = _worst_bucket_recall(rows, y_true, prob)
    return metrics


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
    }


def _extract_row(
    model: DtBucketScaledMagnitudeModel,
    s5_policy: Dict[str, Any],
    ds,
    ds_idx: int,
    *,
    split_name: str,
    device: torch.device,
) -> Dict[str, Any]:
    manifest = ds.manifest()
    meta = manifest[ds_idx]
    sample = ds[ds_idx]
    IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
    IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
    t_gt_vec = sample["t_gt_vec"].unsqueeze(0).to(device, non_blocking=True)
    dt_val = float(meta.get("dt_world", float(sample["t_gt_mag"])))
    dt_world = torch.tensor([dt_val], device=device, dtype=torch.float32)
    with torch.no_grad():
        _R_pred, _t_pred, aux = model(IA, IB, enable_depth_fusion=False, dt_world=dt_world)
        coarse_weight = aux["Wc_ab"].detach().float().max(dim=-1).values
        fine_weight = aux["token_weight_f"].detach().float()
        fine_pool = _pool_token_features(aux["Ff_t"], fine_weight)
        spherical_pool = _pool_token_features(aux["Fc"], coarse_weight)
        fine_weight_stats = _weight_stats(aux["token_weight_f"])
        pred_tmag_s2b = float(aux["t_mag"].detach().float().view(-1)[0].cpu())
        s5_base_scale = _s5_scale_from_pred(pred_tmag_s2b, s5_policy)
        s2b_tvec = aux["t_vec_out"].detach().float().cpu().numpy()[0]
        s5_tvec = s2b_tvec * float(s5_base_scale)
        gt_tvec = t_gt_vec.detach().float().cpu().numpy()[0]
    row: Dict[str, Any] = {
        "split": split_name,
        "cache_key": _row_key(split_name, ds_idx),
        "ds_idx": int(ds_idx),
        "scene": str(meta.get("scene")),
        "seq": str(meta.get("seq")),
        "scene_seq": f"{meta.get('scene')}::{meta.get('seq')}",
        "i": int(meta.get("i", -1)),
        "j": int(meta.get("j", -1)),
        "k": int(meta.get("k", -1)),
        "dt_world": float(dt_val),
        "dt_bucket": _dt_bucket(float(dt_val)),
        "k_bucket": _k_bucket(int(meta.get("k", -1))),
        "pred_tmag_s2b": float(pred_tmag_s2b),
        "s5_base_scale": float(s5_base_scale),
        "is_dt_ge_1": 1.0 if float(dt_val) >= 1.0 else 0.0,
        "is_k20": 1.0 if int(meta.get("k", -1)) == 20 else 0.0,
        "fine_confidence": float(_transport_confidence(aux["Wf_ab"]).cpu().view(-1)[0]),
        "fine_entropy": float(_transport_entropy(aux["Wf_ab"]).cpu().view(-1)[0]),
        "coarse_confidence": float(_transport_confidence(aux["Wc_ab"]).cpu().view(-1)[0]),
        "coarse_entropy": float(_transport_entropy(aux["Wc_ab"]).cpu().view(-1)[0]),
        "fine_weight_mean": float(fine_weight_stats["mean"]),
        "fine_weight_max": float(fine_weight_stats["max"]),
        "fine_weight_std": float(fine_weight_stats["std"]),
        "fine_weight_entropy": float(fine_weight_stats["entropy"]),
        "allowed_mask_density": _allowed_mask_density(aux.get("allowed_mask")),
        "no_candidate_row_ratio": _no_candidate_row_ratio(aux.get("allowed_mask")),
        "s2b_pair_pos_err": float(np.linalg.norm(s2b_tvec - gt_tvec)),
        "s5_pair_pos_err": float(np.linalg.norm(s5_tvec - gt_tvec)),
    }
    row.update(_summarize_vector("fine_pool", fine_pool))
    row.update(_summarize_vector("spherical_pool", spherical_pool))
    return row


def _collect_train_groups(cfg: Config) -> Tuple[RflyPanoPanoramaPairsEvalFixedKList, List[Tuple[str, str]], Dict[Tuple[str, str], List[int]]]:
    train_ds = _build_dataset(cfg, split="train")
    manifest = train_ds.manifest()
    groups = sorted({(str(m.get("scene")), str(m.get("seq"))) for m in manifest})
    seq_to_indices: Dict[Tuple[str, str], List[int]] = {g: [] for g in groups}
    for idx, meta in enumerate(manifest):
        seq_to_indices[(str(meta.get("scene")), str(meta.get("seq")))].append(idx)
    target_groups = [("scene01", "seq01"), ("scene01", "seq02")]
    missing = [g for g in target_groups if g not in seq_to_indices]
    if missing:
        raise RuntimeError(f"missing expected S8 train-CV groups: {missing}")
    target_map = {g: seq_to_indices[g] for g in target_groups}
    return train_ds, target_groups, target_map


def _planned_splits(cfg: Config, *, train_full_cap: int, fold_cap: int) -> Tuple[RflyPanoPanoramaPairsEvalFixedKList, List[SplitSpec]]:
    train_ds, groups, seq_to_indices = _collect_train_groups(cfg)
    specs: List[SplitSpec] = []
    specs.append(
        SplitSpec(
            name="train_full",
            dataset_kind="train",
            dataset_indices=_select_stratified_indices(train_ds, train_full_cap, seed=0),
            cap=train_full_cap,
            seed=0,
        )
    )
    for fold_idx, seq in enumerate(groups):
        val_idx = seq_to_indices[seq]
        train_idx = [i for g, idxs in seq_to_indices.items() if g != seq for i in idxs]
        fold_name = f"fold_{fold_idx}_{seq[0]}_{seq[1]}"
        specs.append(
            SplitSpec(
                name=f"{fold_name}_train",
                dataset_kind="train",
                dataset_indices=_select_stratified_indices(train_ds, fold_cap, seed=10 + fold_idx, allowed_indices=train_idx),
                cap=fold_cap,
                seed=10 + fold_idx,
            )
        )
        specs.append(
            SplitSpec(
                name=f"{fold_name}_val",
                dataset_kind="train",
                dataset_indices=_select_stratified_indices(train_ds, fold_cap, seed=20 + fold_idx, allowed_indices=val_idx),
                cap=fold_cap,
                seed=20 + fold_idx,
            )
        )
    return train_ds, specs


def _ensure_cache(
    model: DtBucketScaledMagnitudeModel,
    cfg: Config,
    s5_policy: Dict[str, Any],
    *,
    cache_path: Path,
    device: torch.device,
    train_full_cap: int,
    fold_cap: int,
    resume: bool,
    flush_every: int,
) -> Dict[str, Any]:
    train_ds, split_specs = _planned_splits(cfg, train_full_cap=train_full_cap, fold_cap=fold_cap)
    cache_exists_before_run = cache_path.exists()
    existing_rows = _read_jsonl(cache_path) if resume else []
    row_by_key = {str(r.get("cache_key")): r for r in existing_rows if isinstance(r, dict) and "cache_key" in r}
    cache_reused: List[str] = []
    cache_written: List[str] = []
    pending_flush: List[Dict[str, Any]] = []
    all_rows = list(existing_rows)
    for spec in split_specs:
        expected_keys = [_row_key(spec.name, idx) for idx in spec.dataset_indices]
        missing_indices = [idx for idx, key in zip(spec.dataset_indices, expected_keys) if key not in row_by_key]
        if len(missing_indices) == 0 and expected_keys:
            cache_reused.append(spec.name)
            continue
        for offset, ds_idx in enumerate(missing_indices):
            if offset == 0 or (offset + 1) % flush_every == 0 or (offset + 1) == len(missing_indices):
                print(f"[S8] extracting {spec.name} {offset + 1}/{len(missing_indices)}", flush=True)
            row = _extract_row(model, s5_policy, train_ds, ds_idx, split_name=spec.name, device=device)
            row_by_key[row["cache_key"]] = row
            all_rows.append(row)
            pending_flush.append(row)
            if len(pending_flush) >= flush_every:
                _append_jsonl(cache_path, pending_flush)
                pending_flush = []
        if missing_indices:
            cache_written.append(spec.name)
    if pending_flush:
        _append_jsonl(cache_path, pending_flush)
    split_rows: Dict[str, List[Dict[str, Any]]] = {}
    for spec in split_specs:
        rows = [row_by_key[_row_key(spec.name, idx)] for idx in spec.dataset_indices if _row_key(spec.name, idx) in row_by_key]
        split_rows[spec.name] = rows
    return {
        "cache_path": str(cache_path),
        "cache_exists_before_run": cache_exists_before_run,
        "reused_splits": sorted(set(cache_reused)),
        "written_splits": sorted(set(cache_written)),
        "split_rows": split_rows,
        "sample_caps": {"train_full_cap": train_full_cap, "fold_train_cap": fold_cap, "fold_val_cap": fold_cap},
    }


def _compare_to_regime(family_row: Dict[str, Any], regime_row: Dict[str, Any]) -> Dict[str, Any]:
    delta_auc = float(family_row["cv_mean_auc"] - regime_row["cv_mean_auc"])
    delta_acc = float(family_row["cv_mean_accuracy"] - regime_row["cv_mean_accuracy"])
    delta_recall = float(family_row["cv_mean_high_error_recall"] - regime_row["cv_mean_high_error_recall"])
    exceeds = bool(delta_auc >= 0.02 and delta_recall >= -0.02 and delta_acc >= -0.02)
    clear_gain = bool(delta_auc >= 0.02 and delta_recall >= 0.05 and delta_acc >= -0.01)
    return {
        "delta_auc": delta_auc,
        "delta_accuracy": delta_acc,
        "delta_high_error_recall": delta_recall,
        "exceeds_regime_only": exceeds,
        "clear_gain_vs_regime_only": clear_gain,
    }


def _run_probe(cache_pack: Dict[str, Any], *, no_odometry_route: bool, skip_final_test: bool) -> Dict[str, Any]:
    split_rows = cache_pack["split_rows"]
    train_full_rows = [dict(r) for r in split_rows["train_full"]]
    label_thresholds = _annotate_labels(train_full_rows)
    fold_names = sorted({name.rsplit("_", 1)[0] for name in split_rows if name.startswith("fold_")})
    fold_rows: List[Dict[str, Any]] = []
    evidence_sufficient = True

    for fold_name in fold_names:
        train_rows = [dict(r) for r in split_rows[f"{fold_name}_train"]]
        val_rows = [dict(r) for r in split_rows[f"{fold_name}_val"]]
        if len(train_rows) < MIN_TRAIN_ROWS or len(val_rows) < MIN_VAL_ROWS:
            evidence_sufficient = False
            continue
        fold_thresholds = _annotate_labels(train_rows)
        _apply_fold_labels(val_rows, fold_thresholds)
        for family in FEATURE_FAMILIES:
            fit = _fit_logistic(train_rows, family, label_key="y_high_error_q80", seed=17 + len(fold_rows))
            metrics = _family_eval(val_rows, fit, label_key="y_high_error_q80")
            fold_rows.append(
                {
                    "fold": fold_name,
                    "family": family,
                    "num_train_rows": len(train_rows),
                    "num_val_rows": len(val_rows),
                    "threshold_q80": float(fold_thresholds["q80"]),
                    "threshold_q90": float(fold_thresholds["q90"]),
                    **metrics,
                }
            )

    family_summary_rows: List[Dict[str, Any]] = []
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for row in fold_rows:
        by_family.setdefault(str(row["family"]), []).append(row)
    for family in FEATURE_FAMILIES:
        items = by_family.get(family, [])
        if not items:
            evidence_sufficient = False
            family_summary_rows.append(
                {
                    "family": family,
                    "num_folds": 0,
                    "cv_mean_auc": float("nan"),
                    "cv_std_auc": float("nan"),
                    "cv_mean_accuracy": float("nan"),
                    "cv_std_accuracy": float("nan"),
                    "cv_mean_high_error_recall": float("nan"),
                    "cv_std_high_error_recall": float("nan"),
                    "cv_mean_precision": float("nan"),
                    "cv_std_precision": float("nan"),
                    "cv_mean_worst_bucket_recall_ge_1p0_k20": float("nan"),
                    "cv_std_worst_bucket_recall_ge_1p0_k20": float("nan"),
                    "fold_rows": [],
                }
            )
            continue
        family_summary_rows.append(
            {
                "family": family,
                "num_folds": len(items),
                "cv_mean_auc": _mean(float(r["auc"]) for r in items),
                "cv_std_auc": _std(float(r["auc"]) for r in items),
                "cv_mean_accuracy": _mean(float(r["accuracy"]) for r in items),
                "cv_std_accuracy": _std(float(r["accuracy"]) for r in items),
                "cv_mean_high_error_recall": _mean(float(r["high_error_recall"]) for r in items),
                "cv_std_high_error_recall": _std(float(r["high_error_recall"]) for r in items),
                "cv_mean_precision": _mean(float(r["precision"]) for r in items),
                "cv_std_precision": _std(float(r["precision"]) for r in items),
                "cv_mean_worst_bucket_recall_ge_1p0_k20": _mean(float(r["worst_bucket_recall_ge_1p0_k20"]) for r in items),
                "cv_std_worst_bucket_recall_ge_1p0_k20": _std(float(r["worst_bucket_recall_ge_1p0_k20"]) for r in items),
                "fold_rows": items,
            }
        )

    regime_row = next((row for row in family_summary_rows if row["family"] == "regime_only"), None)
    if regime_row is not None:
        for row in family_summary_rows:
            if row["family"] == "regime_only":
                row["vs_regime_only"] = {
                    "delta_auc": 0.0,
                    "delta_accuracy": 0.0,
                    "delta_high_error_recall": 0.0,
                    "exceeds_regime_only": False,
                    "clear_gain_vs_regime_only": False,
                }
            else:
                row["vs_regime_only"] = _compare_to_regime(row, regime_row)

    fine_row = next((row for row in family_summary_rows if row["family"] == "fine_token_only"), None)
    spherical_row = next((row for row in family_summary_rows if row["family"] == "spherical_token_only"), None)
    token_plus_regime_row = next((row for row in family_summary_rows if row["family"] == "token_plus_regime"), None)

    best_token_only_row = None
    token_only_label = "none"
    if fine_row is not None and spherical_row is not None:
        best_token_only_row = max([fine_row, spherical_row], key=lambda r: (_safe_float(r["cv_mean_auc"], -1.0), _safe_float(r["cv_mean_high_error_recall"], -1.0)))
        token_only_label = str(best_token_only_row["family"])

    if regime_row is None or token_plus_regime_row is None or best_token_only_row is None:
        evidence_sufficient = False

    if not evidence_sufficient:
        final_classification = "INCONCLUSIVE"
        recommend_s8c = False
    elif bool(token_plus_regime_row["vs_regime_only"]["clear_gain_vs_regime_only"]):
        final_classification = "TOKEN-RELIABILITY-DIAGNOSTIC-GAIN"
        recommend_s8c = True
    else:
        final_classification = "TOKEN-NO-ADDED-VALUE"
        recommend_s8c = False

    return {
        "label_thresholds": label_thresholds,
        "family_summary_rows": family_summary_rows,
        "fold_rows": fold_rows,
        "best_token_only_family": token_only_label,
        "best_token_only_vs_regime_only": None if best_token_only_row is None else best_token_only_row["vs_regime_only"],
        "token_plus_regime_vs_regime_only": None if token_plus_regime_row is None else token_plus_regime_row["vs_regime_only"],
        "evidence_sufficient": evidence_sufficient,
        "final_classification": final_classification,
        "recommend_s8c": recommend_s8c,
        "skip_final_test": bool(skip_final_test),
        "no_odometry_route": bool(no_odometry_route),
        "s5_remains_final_clean_candidate": True,
    }


def _write_report(payload: Dict[str, Any], report_path: Path) -> None:
    baseline_gate = payload["baseline_gate"]
    cache_info = payload["cache_info"]
    probe = payload.get("probe", {})
    feature_audit = payload["feature_audit"]
    lines: List[str] = []
    lines.append("# S8 Fine/Spherical Reliability Router Diagnostic Report\n\n")
    lines.append("## Status\n\n")
    lines.append("- S8b baseline contract gate has already passed.\n")
    lines.append("- This run switched to staged diagnostic mode because of runtime / budget limits.\n")
    lines.append(f"- Final classification: `{payload['final_classification']}`.\n")
    lines.append(f"- S5 remains final clean candidate: `{payload['s5_remains_final_clean_candidate']}`.\n")
    lines.append(f"- Final route test skipped: `{payload['skip_final_test']}`.\n")
    lines.append(f"- Odometry route search disabled: `{payload['no_odometry_route']}`.\n\n")

    lines.append("## Baseline Gate\n\n")
    lines.append(f"- Gate passed: `{baseline_gate['passed']}`.\n")
    lines.append(f"- Contract: `{baseline_gate['contract_path']}`.\n")
    lines.append(
        f"- Locked S5 wrapper metrics: drift=`{_fmt(float(baseline_gate['s5']['metrics']['drift']))}`, "
        f"ATE=`{_fmt(float(baseline_gate['s5']['metrics']['ATE']))}`, "
        f"path_ratio=`{_fmt(float(baseline_gate['s5']['metrics']['path_ratio']))}`.\n\n"
    )

    lines.append("## Cache And Caps\n\n")
    lines.append(f"- Cache path: `{cache_info['cache_path']}`.\n")
    lines.append(f"- Reused cache splits: `{', '.join(cache_info['reused_splits']) if cache_info['reused_splits'] else 'none'}`.\n")
    lines.append(f"- Newly written cache splits: `{', '.join(cache_info['written_splits']) if cache_info['written_splits'] else 'none'}`.\n")
    lines.append(
        f"- Sample caps used: train_full=`{cache_info['sample_caps']['train_full_cap']}`, "
        f"fold_train=`{cache_info['sample_caps']['fold_train_cap']}`, "
        f"fold_val=`{cache_info['sample_caps']['fold_val_cap']}`.\n\n"
    )

    lines.append("## Feature Audit\n\n")
    lines.append("| feature | shape | source | inference-visible | deployable-safe |\n")
    lines.append("| --- | --- | --- | --- | --- |\n")
    for row in feature_audit:
        lines.append(f"| {row['feature']} | {row['shape']} | {row['source_module']} | {row['inference_visible']} | {row['deployable_safe']} |\n")
    lines.append("\n")

    lines.append("## Probe Families\n\n")
    lines.append("- `regime_only`: `pred_tmag + dt + k + S5 base scale`.\n")
    lines.append("- `fine_token_only`: compact fine-token summary only.\n")
    lines.append("- `spherical_token_only`: compact spherical-token summary only.\n")
    lines.append("- `token_plus_regime`: both token summaries plus `pred_tmag + dt + k`.\n")
    lines.append("- Models used: logistic regression only.\n\n")

    lines.append("## CV Results\n\n")
    lines.append("| family | folds | AUC mean/std | acc mean/std | high-error recall mean/std | precision mean/std | worst-bucket recall mean/std | exceeds regime_only |\n")
    lines.append("| --- | ---: | --- | --- | --- | --- | --- | --- |\n")
    for row in probe.get("family_summary_rows", []):
        compare = row.get("vs_regime_only", {})
        exceeds = compare.get("exceeds_regime_only", False)
        lines.append(
            f"| {row['family']} | {row['num_folds']} | {_fmt(row['cv_mean_auc'])}/{_fmt(row['cv_std_auc'])} | "
            f"{_fmt(row['cv_mean_accuracy'])}/{_fmt(row['cv_std_accuracy'])} | "
            f"{_fmt(row['cv_mean_high_error_recall'])}/{_fmt(row['cv_std_high_error_recall'])} | "
            f"{_fmt(row['cv_mean_precision'])}/{_fmt(row['cv_std_precision'])} | "
            f"{_fmt(row['cv_mean_worst_bucket_recall_ge_1p0_k20'])}/{_fmt(row['cv_std_worst_bucket_recall_ge_1p0_k20'])} | "
            f"{exceeds} |\n"
        )
    lines.append("\n")

    best_token_only = probe.get("best_token_only_family")
    token_only_vs = probe.get("best_token_only_vs_regime_only") or {}
    token_plus_vs = probe.get("token_plus_regime_vs_regime_only") or {}
    lines.append("## Core Comparison\n\n")
    lines.append(f"- Best token-only family: `{best_token_only}`.\n")
    lines.append(
        f"- token-only vs regime_only: delta_auc=`{_fmt(_safe_float(token_only_vs.get('delta_auc')) )}`, "
        f"delta_accuracy=`{_fmt(_safe_float(token_only_vs.get('delta_accuracy')) )}`, "
        f"delta_high_error_recall=`{_fmt(_safe_float(token_only_vs.get('delta_high_error_recall')) )}`, "
        f"exceeds=`{token_only_vs.get('exceeds_regime_only', False)}`.\n"
    )
    lines.append(
        f"- token_plus_regime vs regime_only: delta_auc=`{_fmt(_safe_float(token_plus_vs.get('delta_auc')) )}`, "
        f"delta_accuracy=`{_fmt(_safe_float(token_plus_vs.get('delta_accuracy')) )}`, "
        f"delta_high_error_recall=`{_fmt(_safe_float(token_plus_vs.get('delta_high_error_recall')) )}`, "
        f"exceeds=`{token_plus_vs.get('exceeds_regime_only', False)}`.\n\n"
    )

    lines.append("## Decision\n\n")
    lines.append(f"- Evidence sufficient for S8c decision: `{probe.get('evidence_sufficient', False)}`.\n")
    lines.append(f"- Recommend S8c follow-up final router test later: `{probe.get('recommend_s8c', False)}`.\n")
    lines.append(f"- S5 remains final clean candidate: `{payload['s5_remains_final_clean_candidate']}`.\n")
    lines.append(f"- Final classification: `{payload['final_classification']}`.\n")
    report_path.write_text("".join(lines), encoding="utf-8")


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    baseline_gate = _baseline_gate()
    payload: Dict[str, Any] = {
        "baseline_gate": baseline_gate,
        "feature_audit": _feature_audit(),
        "final_classification": "REPRODUCTION-MISMATCH" if not baseline_gate["passed"] else "INCONCLUSIVE",
        "s5_remains_final_clean_candidate": True,
        "skip_final_test": bool(args.skip_final_test),
        "no_odometry_route": bool(args.no_odometry_route),
    }
    if not baseline_gate["passed"]:
        _write_json(args.candidates_path, payload)
        _write_report(payload | {"cache_info": {"cache_path": str(args.cache_path), "reused_splits": [], "written_splits": [], "sample_caps": {"train_full_cap": args.train_full_cap, "fold_train_cap": args.fold_cap, "fold_val_cap": args.fold_cap}}}, args.report_path)
        return payload

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s2b_model, cfg, _s2b_policy, _load_summary = _build_s2b_model(device)
    s5_policy = _load_s5_policy()
    cache_runtime = _ensure_cache(
        s2b_model,
        cfg,
        s5_policy,
        cache_path=args.cache_path,
        device=device,
        train_full_cap=int(args.train_full_cap),
        fold_cap=int(args.fold_cap),
        resume=bool(args.resume),
        flush_every=DEFAULT_FLUSH_EVERY,
    )
    payload["cache_info"] = {
        "cache_path": cache_runtime["cache_path"],
        "cache_exists_before_run": cache_runtime["cache_exists_before_run"],
        "reused_splits": cache_runtime["reused_splits"],
        "written_splits": cache_runtime["written_splits"],
        "sample_caps": cache_runtime["sample_caps"],
    }

    if args.stage == "extract":
        _write_json(args.candidates_path, payload)
        _write_report(payload | {"probe": {}}, args.report_path)
        return payload

    if args.stage in {"probe", "route"}:
        if args.stage == "route" and not args.no_odometry_route:
            raise RuntimeError("Full route stage is intentionally disabled in staged diagnostic mode. Re-run with --no-odometry-route.")
        probe = _run_probe(cache_runtime, no_odometry_route=bool(args.no_odometry_route), skip_final_test=bool(args.skip_final_test))
        payload["probe"] = probe
        payload["final_classification"] = str(probe["final_classification"])
        payload["recommend_s8c"] = bool(probe["recommend_s8c"])
        payload["s5_remains_final_clean_candidate"] = bool(probe["s5_remains_final_clean_candidate"])
        _write_json(args.candidates_path, payload)
        _write_report(payload, args.report_path)
        return payload

    if args.stage == "report":
        if not args.candidates_path.exists():
            raise FileNotFoundError(f"missing candidates json for report stage: {args.candidates_path}")
        payload = _read_json(args.candidates_path)
        if "cache_info" not in payload:
            payload["cache_info"] = {"cache_path": str(args.cache_path), "reused_splits": [], "written_splits": [], "sample_caps": {"train_full_cap": args.train_full_cap, "fold_train_cap": args.fold_cap, "fold_val_cap": args.fold_cap}}
        _write_report(payload, args.report_path)
        return payload

    raise ValueError(args.stage)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "eval_policy"], default="run")
    ap.add_argument("--policy", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stage", choices=["extract", "probe", "route", "report"], default="probe")
    ap.add_argument("--train-full-cap", type=int, default=DEFAULT_TRAIN_FULL_CAP)
    ap.add_argument("--fold-cap", type=int, default=DEFAULT_FOLD_CAP)
    ap.add_argument("--skip-final-test", action="store_true")
    ap.add_argument("--no-odometry-route", action="store_true")
    ap.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    ap.add_argument("--candidates-path", type=Path, default=DEFAULT_CANDIDATES_PATH)
    ap.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = ap.parse_args()

    if args.mode == "eval_policy":
        out = {
            "mode": "eval_policy",
            "supported": False,
            "message": "eval_policy is disabled in staged diagnostic mode; use --stage probe/report instead.",
            "policy_path": args.policy,
        }
        print(json.dumps(out, indent=2))
        return

    payload = _run(args)
    print(json.dumps(_json_safe(payload), indent=2))


if __name__ == "__main__":
    main()
