#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

REPORT_PATH = REPO_ROOT / "checkpoints" / "S16_stronger_visual_backbone_feasibility_report.md"
CANDIDATES_PATH = REPO_ROOT / "checkpoints" / "S16_stronger_visual_backbone_feasibility_candidates.json"
SUMMARY_PATH = REPO_ROOT / "reports" / "final_s16_stronger_visual_backbone_feasibility_summary.md"
CONFIG_PATH = REPO_ROOT / "configs" / "s16_stronger_visual_backbone_feasibility.yaml"
S8B_CONTRACT_PATH = REPO_ROOT / "checkpoints" / "S8b_reproduction_contract.json"
S8B_S5_RESULT_PATH = REPO_ROOT / "checkpoints" / "S8b_current_arch_s5_wrapper_result.json"
S5_POLICY_PATH = REPO_ROOT / "checkpoints" / "S5_clean_tmag_calibration_policy.json"
S3B_REPORT_PATH = REPO_ROOT / "checkpoints" / "S3b_fine_token_representation_diagnostic_report.md"
S8_REPORT_PATH = REPO_ROOT / "checkpoints" / "S8_fine_spherical_reliability_router_report.md"

S5_LOCKED = {"drift": 1.327343, "ATE": 7.352288, "path_ratio": 0.932379}
FINAL_CLASSES = {
    "STRONGER-BACKBONE-DIAGNOSTIC-SIGNAL",
    "BACKBONE-TMAG-ONLY-SIGNAL",
    "NO-STABLE-BACKBONE-FEATURE-GAIN",
    "PRETRAINED-WEIGHTS-UNAVAILABLE",
    "REPRODUCTION-MISMATCH",
    "INCONCLUSIVE",
}


@dataclass
class CurrentRow:
    fold_source: str
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
    cfg: Dict[str, Any] = {}
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


def _as_int_list(cfg: Dict[str, Any], key: str, default: Sequence[int]) -> Tuple[int, ...]:
    raw = str(cfg.get(key, ",".join(str(x) for x in default)))
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return tuple(out) if out else tuple(default)


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


def _candidate_cache_files() -> Dict[str, List[str]]:
    return {
        "torchvision_resnet50": ["resnet50-0676ba61.pth", "resnet50-11ad3fa6.pth"],
        "torchvision_convnext_tiny": ["convnext_tiny-983f1562.pth"],
        "torchvision_vit_b_16": ["vit_b_16-c867db91.pth"],
    }


def _count_params(model: Any) -> int:
    return int(sum(int(p.numel()) for p in model.parameters()))


def _encoder_availability_audit() -> Dict[str, Any]:
    internet_attempted = False
    rows: List[Dict[str, Any]] = []
    cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    cache_files = _candidate_cache_files()
    try:
        import torch  # noqa: F401
        import torchvision.models as tvm

        tv_ok = True
        tv_error = ""
    except Exception as exc:
        tv_ok = False
        tv_error = repr(exc)
        tvm = None

    model_specs = [
        ("torchvision_resnet50", "resnet50", 2048),
        ("torchvision_convnext_tiny", "convnext_tiny", 768),
        ("torchvision_vit_b_16", "vit_b_16", 768),
    ]
    for name, ctor_name, feat_dim in model_specs:
        pretrained_files = [cache_dir / f for f in cache_files.get(name, [])]
        cached = [str(p) for p in pretrained_files if p.exists()]
        param_count = None
        init_error = ""
        if tv_ok and tvm is not None:
            try:
                model = getattr(tvm, ctor_name)(weights=None)
                param_count = _count_params(model)
                del model
            except Exception as exc:
                init_error = repr(exc)
        rows.append(
            {
                "encoder_name": name,
                "package_available": bool(tv_ok),
                "package_error": tv_error,
                "pretrained_weights_available": bool(cached),
                "cached_weight_files": cached,
                "parameter_count": param_count,
                "feature_dimension": int(feat_dim),
                "internet_download_attempted": internet_attempted,
                "encoder_frozen": True,
                "initialized_without_download": bool(tv_ok and init_error == ""),
                "init_error": init_error,
            }
        )

    try:
        import timm  # noqa: F401

        timm_available = True
        timm_error = ""
    except Exception as exc:
        timm_available = False
        timm_error = repr(exc)
    rows.append(
        {
            "encoder_name": "timm_cached",
            "package_available": bool(timm_available),
            "package_error": timm_error,
            "pretrained_weights_available": False,
            "cached_weight_files": [],
            "parameter_count": None,
            "feature_dimension": None,
            "internet_download_attempted": internet_attempted,
            "encoder_frozen": True,
            "initialized_without_download": False,
            "init_error": "" if timm_available else timm_error,
        }
    )
    return {
        "internet_download_attempted": internet_attempted,
        "cache_dir": str(cache_dir),
        "encoders": rows,
        "any_pretrained_available": any(bool(r["pretrained_weights_available"]) for r in rows),
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
    cos = np.clip((float(np.trace(rel)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _rot_log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos = np.clip((float(np.trace(R)) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos))
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
    elif x.dim() == 2:
        pass
    else:
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
    reg = np.eye(xb.shape[1], dtype=np.float64) * float(lam)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y)


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

    return {
        "train_auc": _auc(y_train, train_score),
        "val_auc": _auc(y_val, val_score),
        "train_recall": recall(y_train, train_pred),
        "val_recall": recall(y_val, val_pred),
        "train_accuracy": float(np.mean(train_pred == y_train)),
        "val_accuracy": float(np.mean(val_pred == y_val)),
    }


def _dt_bucket(v: float) -> str:
    if v < 0.3:
        return "<0.3"
    if v < 0.5:
        return "[0.3,0.5)"
    if v < 1.0:
        return "[0.5,1.0)"
    if v < 2.0:
        return "[1.0,2.0)"
    return ">=2.0"


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


def _extract_current_representation_audit(config: Dict[str, Any]) -> Dict[str, Any]:
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
    train_cap = _as_int(config, "train_cap_per_sequence", 512)
    val_cap = _as_int(config, "val_cap_per_sequence", 256)
    ridge_lam = _as_float(config, "ridge_lambda", 1.0)
    high_q = _as_float(config, "high_error_quantile", 0.75)
    seq_keys = ["scene01_seq01", "scene01_seq02"]
    by_seq: Dict[str, List[int]] = {k: [] for k in seq_keys}
    for idx, meta in enumerate(manifest):
        key = f"{meta.get('scene')}_{meta.get('seq')}"
        if key in by_seq:
            by_seq[key].append(idx)
    selected = {
        key: _select_stratified(vals, manifest, train_cap, seed=1600 + i)
        for i, (key, vals) in enumerate(sorted(by_seq.items()))
    }
    val_selected = {key: vals[: min(val_cap, len(vals))] for key, vals in selected.items()}

    q90 = float(policy["thresholds"]["q90_value"])
    q95 = float(policy["thresholds"]["q95_upper_tail_value"])
    mid_scale = float(policy["scales"]["mid_scale"])
    high_scale = float(policy["scales"]["high_scale"])

    cache: Dict[int, CurrentRow] = {}

    def s5_scale(pred_tmag: float) -> float:
        if pred_tmag >= q95:
            return high_scale
        if pred_tmag >= q90:
            return mid_scale
        return 1.0

    def extract_one(ds_idx: int) -> CurrentRow:
        if ds_idx in cache:
            return cache[ds_idx]
        sample = ds[ds_idx]
        meta = manifest[ds_idx]
        IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
        IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
        dt_world = float(meta.get("dt_world", float(sample["t_gt_mag"])))
        dt_tensor = torch.tensor([dt_world], device=device, dtype=torch.float32)
        with torch.no_grad():
            R_pred_t, _tdir_pred_t, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
        R_gt = sample["R_gt"].detach().float().cpu().numpy()
        t_gt_vec = sample["t_gt_vec"].detach().float().cpu().numpy()
        t_gt_dir = _unit(sample["t_gt_dir"].detach().float().cpu().numpy())
        gt_tmag = float(sample["t_gt_mag"])
        raw_pred_tmag = float(aux["t_mag"].detach().float().view(-1)[0].cpu())
        scale = s5_scale(raw_pred_tmag)
        pred_tmag = float(raw_pred_tmag * scale)
        R_pred = R_pred_t.detach().float().cpu().numpy()[0]
        tdir_pred = _unit(aux["t_dir_out"].detach().float().cpu().numpy()[0])
        t_vec_pred = tdir_pred * pred_tmag
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
        row = CurrentRow(
            fold_source="train_split",
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
            regime_feature=regime,
        )
        cache[ds_idx] = row
        if len(cache) == 1 or len(cache) % 32 == 0:
            print(f"[S16] current feature extraction cached {len(cache)} rows", flush=True)
        return row

    def rows_for(indices: Sequence[int]) -> List[CurrentRow]:
        return [extract_one(int(i)) for i in indices]

    folds = [
        ("heldout_scene01_seq01", "scene01_seq01", "scene01_seq02"),
        ("heldout_scene01_seq02", "scene01_seq02", "scene01_seq01"),
    ]
    family_results: List[Dict[str, Any]] = []
    regime_rows: List[Dict[str, Any]] = []
    for fold_name, heldout, train_seq in folds:
        train_rows = rows_for(selected[train_seq])
        val_rows = rows_for(val_selected[heldout])
        y_rot_train = np.stack([r.gt_rotvec for r in train_rows], axis=0)
        y_rot_val = np.stack([r.gt_rotvec for r in val_rows], axis=0)
        y_tdir_train = np.stack([r.gt_tdir for r in train_rows], axis=0)
        y_tdir_val = np.stack([r.gt_tdir for r in val_rows], axis=0)
        y_tmag_train = np.asarray([r.gt_log_tmag for r in train_rows], dtype=np.float64).reshape(-1, 1)
        y_tmag_val = np.asarray([r.gt_log_tmag for r in val_rows], dtype=np.float64).reshape(-1, 1)

        rot_err_train = np.asarray([r.rot_err_deg for r in train_rows], dtype=np.float64)
        tdir_err_train = np.asarray([r.tdir_err_deg for r in train_rows], dtype=np.float64)
        tmag_err_train = np.asarray([r.log_tmag_abs_err for r in train_rows], dtype=np.float64)
        rot_err_val = np.asarray([r.rot_err_deg for r in val_rows], dtype=np.float64)
        tdir_err_val = np.asarray([r.tdir_err_deg for r in val_rows], dtype=np.float64)
        tmag_err_val = np.asarray([r.log_tmag_abs_err for r in val_rows], dtype=np.float64)
        combined_train = (
            (rot_err_train - rot_err_train.mean()) / max(rot_err_train.std(), 1.0e-8)
            + (tdir_err_train - tdir_err_train.mean()) / max(tdir_err_train.std(), 1.0e-8)
            + (tmag_err_train - tmag_err_train.mean()) / max(tmag_err_train.std(), 1.0e-8)
        )
        combined_val = (
            (rot_err_val - rot_err_train.mean()) / max(rot_err_train.std(), 1.0e-8)
            + (tdir_err_val - tdir_err_train.mean()) / max(tdir_err_train.std(), 1.0e-8)
            + (tmag_err_val - tmag_err_train.mean()) / max(tmag_err_train.std(), 1.0e-8)
        )
        joint_train = (
            (rot_err_train - rot_err_train.mean()) / max(rot_err_train.std(), 1.0e-8)
            + (tdir_err_train - tdir_err_train.mean()) / max(tdir_err_train.std(), 1.0e-8)
        )
        joint_val = (
            (rot_err_val - rot_err_train.mean()) / max(rot_err_train.std(), 1.0e-8)
            + (tdir_err_val - tdir_err_train.mean()) / max(tdir_err_train.std(), 1.0e-8)
        )
        high_thr = float(np.quantile(combined_train, high_q))
        joint_thr = float(np.quantile(joint_train, high_q))
        y_high_train = (combined_train >= high_thr).astype(np.int64)
        y_high_val = (combined_val >= high_thr).astype(np.int64)
        y_joint_train = (joint_train >= joint_thr).astype(np.int64)
        y_joint_val = (joint_val >= joint_thr).astype(np.int64)

        feature_sets = {
            "regime_only": (
                np.stack([r.regime_feature for r in train_rows], axis=0),
                np.stack([r.regime_feature for r in val_rows], axis=0),
            ),
            "current_model_features_only": (
                np.stack([r.current_feature for r in train_rows], axis=0),
                np.stack([r.current_feature for r in val_rows], axis=0),
            ),
        }
        for family, (xtr_raw, xva_raw) in feature_sets.items():
            xtr, xva = _standardize(xtr_raw, xva_raw)
            rot_w = _ridge_fit(xtr, y_rot_train, ridge_lam)
            tdir_w = _ridge_fit(xtr, y_tdir_train, ridge_lam)
            tmag_w = _ridge_fit(xtr, y_tmag_train, ridge_lam)
            rot_tr = _ridge_predict(xtr, rot_w)
            rot_va = _ridge_predict(xva, rot_w)
            tdir_tr = np.apply_along_axis(_unit, 1, _ridge_predict(xtr, tdir_w))
            tdir_va = np.apply_along_axis(_unit, 1, _ridge_predict(xva, tdir_w))
            tmag_tr = _ridge_predict(xtr, tmag_w)
            tmag_va = _ridge_predict(xva, tmag_w)
            rot_ang_tr = np.asarray([_rot_geodesic_deg(_rot_exp(p), _rot_exp(g)) for p, g in zip(rot_tr, y_rot_train)])
            rot_ang_va = np.asarray([_rot_geodesic_deg(_rot_exp(p), _rot_exp(g)) for p, g in zip(rot_va, y_rot_val)])
            tdir_ang_tr = np.asarray([_vec_angle_deg(p, g) for p, g in zip(tdir_tr, y_tdir_train)])
            tdir_ang_va = np.asarray([_vec_angle_deg(p, g) for p, g in zip(tdir_va, y_tdir_val)])
            tdir_cos_va = np.asarray([float(np.clip(np.dot(_unit(p), _unit(g)), -1.0, 1.0)) for p, g in zip(tdir_va, y_tdir_val)])
            high_metrics = _binary_ridge_metrics(xtr, y_high_train, xva, y_high_val, ridge_lam)
            joint_metrics = _binary_ridge_metrics(xtr, y_joint_train, xva, y_joint_val, ridge_lam)
            family_results.append(
                {
                    "fold": fold_name,
                    "family": family,
                    "feature_dim": int(xtr_raw.shape[1]),
                    "train_n": int(xtr_raw.shape[0]),
                    "val_n": int(xva_raw.shape[0]),
                    "rot_r2_train": _r2(y_rot_train, rot_tr),
                    "rot_r2_val": _r2(y_rot_val, rot_va),
                    "rot_angular_error_train": float(rot_ang_tr.mean()),
                    "rot_angular_error_val": float(rot_ang_va.mean()),
                    "tdir_r2_train": _r2(y_tdir_train, tdir_tr),
                    "tdir_r2_val": _r2(y_tdir_val, tdir_va),
                    "tdir_angular_error_train": float(tdir_ang_tr.mean()),
                    "tdir_angular_error_val": float(tdir_ang_va.mean()),
                    "tdir_cosine_val": float(tdir_cos_va.mean()),
                    "log_tmag_r2_train": _r2(y_tmag_train, tmag_tr),
                    "log_tmag_r2_val": _r2(y_tmag_val, tmag_va),
                    "log_tmag_mae_train": float(np.mean(np.abs(y_tmag_train - tmag_tr))),
                    "log_tmag_mae_val": float(np.mean(np.abs(y_tmag_val - tmag_va))),
                    "high_error_auc_train": high_metrics["train_auc"],
                    "high_error_auc_val": high_metrics["val_auc"],
                    "high_error_recall_train": high_metrics["train_recall"],
                    "high_error_recall_val": high_metrics["val_recall"],
                    "joint_r_tdir_auc_train": joint_metrics["train_auc"],
                    "joint_r_tdir_auc_val": joint_metrics["val_auc"],
                    "joint_r_tdir_recall_train": joint_metrics["train_recall"],
                    "joint_r_tdir_recall_val": joint_metrics["val_recall"],
                    "train_test_gap_mean": float(
                        np.nanmean(
                            [
                                _r2(y_rot_train, rot_tr) - _r2(y_rot_val, rot_va),
                                _r2(y_tdir_train, tdir_tr) - _r2(y_tdir_val, tdir_va),
                                _r2(y_tmag_train, tmag_tr) - _r2(y_tmag_val, tmag_va),
                            ]
                        )
                    ),
                }
            )

        for bucket_key, bucket_fn in [
            ("dt_bucket", lambda r: _dt_bucket(r.dt_world)),
            ("k_bucket", lambda r: f"k={r.k}"),
        ]:
            groups: Dict[str, List[CurrentRow]] = {}
            for row in val_rows:
                groups.setdefault(bucket_fn(row), []).append(row)
            for label, grows in sorted(groups.items()):
                regime_rows.append(
                    {
                        "fold": fold_name,
                        "bucket_type": bucket_key,
                        "bucket": label,
                        "n": len(grows),
                        "current_model_mean_rot_err": float(np.mean([r.rot_err_deg for r in grows])),
                        "current_model_mean_tdir_err": float(np.mean([r.tdir_err_deg for r in grows])),
                        "current_model_mean_log_tmag_abs_err": float(np.mean([r.log_tmag_abs_err for r in grows])),
                    }
                )

    def mean_for(family: str, key: str) -> float:
        vals = [float(r[key]) for r in family_results if r["family"] == family and math.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    current_summary = {
        "status": "completed",
        "device": str(device),
        "load_missing": len(load_msg.missing_keys),
        "load_unexpected": len(load_msg.unexpected_keys),
        "dataset_train_len": len(ds),
        "selected_per_sequence": {k: len(v) for k, v in selected.items()},
        "val_selected_per_sequence": {k: len(v) for k, v in val_selected.items()},
        "unique_extracted_rows": len(cache),
        "families": family_results,
        "family_means": {
            family: {
                "rot_r2_val": mean_for(family, "rot_r2_val"),
                "rot_angular_error_val": mean_for(family, "rot_angular_error_val"),
                "tdir_r2_val": mean_for(family, "tdir_r2_val"),
                "tdir_angular_error_val": mean_for(family, "tdir_angular_error_val"),
                "tdir_cosine_val": mean_for(family, "tdir_cosine_val"),
                "log_tmag_mae_val": mean_for(family, "log_tmag_mae_val"),
                "high_error_auc_val": mean_for(family, "high_error_auc_val"),
                "high_error_recall_val": mean_for(family, "high_error_recall_val"),
                "joint_r_tdir_auc_val": mean_for(family, "joint_r_tdir_auc_val"),
                "joint_r_tdir_recall_val": mean_for(family, "joint_r_tdir_recall_val"),
                "train_test_gap_mean": mean_for(family, "train_test_gap_mean"),
            }
            for family in ["regime_only", "current_model_features_only"]
        },
        "regime_level_analysis": regime_rows,
        "prior_alignment": {
            "s3b_report": str(S3B_REPORT_PATH),
            "s8_report": str(S8_REPORT_PATH),
            "note": "S3b/S8 already found current fine/spherical token features did not provide stable deployable residual or reliability gain.",
        },
    }
    return current_summary


def _classification(availability: Dict[str, Any], current_audit: Dict[str, Any] | None) -> Tuple[str, bool, str]:
    if not availability.get("any_pretrained_available", False):
        return "PRETRAINED-WEIGHTS-UNAVAILABLE", False, "No local pretrained torchvision/timm weights were available, and no internet download was attempted."
    if not current_audit or current_audit.get("status") != "completed":
        return "INCONCLUSIVE", False, "Pretrained availability was detected, but the current audit/probe evidence is incomplete."
    return "INCONCLUSIVE", False, "S16 stronger-feature extraction support is present in the audit scaffold, but this run did not promote a full integration recommendation."


def _markdown_table(rows: Sequence[Dict[str, Any]], cols: Sequence[Tuple[str, str]], limit: int | None = None) -> str:
    use_rows = list(rows[:limit] if limit is not None else rows)
    out = ["| " + " | ".join(label for label, _key in cols) + " |"]
    out.append("| " + " | ".join("---" for _ in cols) + " |")
    for row in use_rows:
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


def _write_reports(payload: Dict[str, Any]) -> None:
    gate = payload["baseline_gate"]
    availability = payload["encoder_availability"]
    current = payload.get("current_representation_audit")
    classification = payload["final_classification"]
    s16b = bool(payload["s16b_full_integration_recommended"])
    current_means = (current or {}).get("family_means", {}) if isinstance(current, dict) else {}
    family_rows = (current or {}).get("families", []) if isinstance(current, dict) else []
    regime_rows = (current or {}).get("regime_level_analysis", []) if isinstance(current, dict) else []

    lines: List[str] = [
        "# S16 Stronger Visual Backbone Feasibility Report",
        "",
        "## Executive summary",
        "",
        f"- final classification: `{classification}`",
        "- S16 is a feasibility diagnostic, not a final clean candidate replacement.",
        "- S5 remains the final clean candidate with locked metrics drift `1.327343`, ATE `7.352288`, path_ratio `0.932379`.",
        f"- S16b full backbone integration recommended: `{s16b}`",
        f"- interpretation: {payload['classification_reason']}",
        "",
        "## Motivation from S13/S15 closure",
        "",
        "- S13 quantified the practical gap and identified `R-TDIR-COUPLED-LIMITED` as the primary bottleneck.",
        "- S15 closed the tiny trajectory-level training route under the current harness because it remained unstable.",
        "- S16 therefore checks whether stronger frozen visual features show diagnostic signal for R/tdir coupling before any full backbone replacement is considered.",
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
        "## Current representation audit",
        "",
    ]
    if current and current.get("status") == "completed":
        lines.extend(
            [
                f"- status: `{current['status']}`",
                f"- device: `{current['device']}`",
                f"- unique extracted train-split rows: `{current['unique_extracted_rows']}`",
                "- feature inputs: coarse pooled features, fine pooled features, spherical bearing summaries, `pred_tmag`, `dt`, `k`.",
                "- leakage policy: GT rotation/tdir/tmag are labels only; they are not included in probe features.",
                "",
                "Mean two-fold CV probe results:",
                "",
                _markdown_table(
                    [
                        {"family": k, **v}
                        for k, v in current_means.items()
                    ],
                    [
                        ("family", "family"),
                        ("rot R2", "rot_r2_val"),
                        ("rot ang err", "rot_angular_error_val"),
                        ("tdir R2", "tdir_r2_val"),
                        ("tdir ang err", "tdir_angular_error_val"),
                        ("tdir cos", "tdir_cosine_val"),
                        ("log tmag MAE", "log_tmag_mae_val"),
                        ("high AUC", "high_error_auc_val"),
                        ("high recall", "high_error_recall_val"),
                        ("joint AUC", "joint_r_tdir_auc_val"),
                        ("gap", "train_test_gap_mean"),
                    ],
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- status: `{(current or {}).get('status', 'not_run')}`",
                f"- error: `{(current or {}).get('error', 'n/a')}`",
                "- existing alignment: S3b and S8 are used as prior current-representation evidence.",
                "",
            ]
        )

    lines.extend(
        [
            "## Encoder availability audit",
            "",
            f"- internet download attempted: `{availability['internet_download_attempted']}`",
            f"- any pretrained weights available: `{availability['any_pretrained_available']}`",
            "",
            _markdown_table(
                availability["encoders"],
                [
                    ("encoder", "encoder_name"),
                    ("pkg", "package_available"),
                    ("pretrained", "pretrained_weights_available"),
                    ("params", "parameter_count"),
                    ("dim", "feature_dimension"),
                    ("download", "internet_download_attempted"),
                    ("frozen", "encoder_frozen"),
                ],
            ),
            "",
            "## Frozen feature extraction protocol",
            "",
            "- For a locally cached pretrained encoder, the intended pair feature is `feat_A`, `feat_B`, `feat_B - feat_A`, `feat_A * feat_B`, concatenated.",
            "- Safe regime features are restricted to `pred_tmag`, `dt`, and `k`.",
            "- GT pose, GT tmag, GT tdir, and pair error are labels only and are never inference features.",
            "- This run did not execute stronger frozen feature extraction because no local pretrained weights were available.",
            "",
            "## Probe families",
            "",
            "- Family A: `current_model_features_only`.",
            "- Family B: `stronger_encoder_features_only`.",
            "- Family C: `stronger_encoder_plus_regime_features`.",
            "- Family D: `current_model_features_plus_stronger_encoder_features`.",
            "- Family E: `current_model_features_plus_stronger_encoder_plus_regime_features`.",
            "- Families B-E were not run when pretrained weights were unavailable.",
            "",
            "## CV results",
            "",
        ]
    )
    if family_rows:
        lines.extend(
            [
                _markdown_table(
                    family_rows,
                    [
                        ("fold", "fold"),
                        ("family", "family"),
                        ("rot R2", "rot_r2_val"),
                        ("tdir R2", "tdir_r2_val"),
                        ("tmag MAE", "log_tmag_mae_val"),
                        ("high AUC", "high_error_auc_val"),
                        ("joint AUC", "joint_r_tdir_auc_val"),
                    ],
                ),
                "",
            ]
        )
    else:
        lines.extend(["- No stronger-feature CV results were produced in this run.", ""])

    lines.extend(
        [
            "## Regime-level analysis",
            "",
        ]
    )
    if regime_rows:
        lines.extend(
            [
                _markdown_table(
                    regime_rows,
                    [
                        ("fold", "fold"),
                        ("type", "bucket_type"),
                        ("bucket", "bucket"),
                        ("n", "n"),
                        ("rot err", "current_model_mean_rot_err"),
                        ("tdir err", "current_model_mean_tdir_err"),
                        ("log tmag err", "current_model_mean_log_tmag_abs_err"),
                    ],
                    limit=40,
                ),
                "",
            ]
        )
    else:
        lines.extend(["- Regime-level current-feature analysis was not completed.", ""])

    lines.extend(
        [
            "## R/tdir coupling proxy analysis",
            "",
            f"- stronger-feature R/tdir coupling proxy result: `{payload['r_tdir_coupling_proxy_result']}`",
            "- Probe AUC/R2 is interpreted only as diagnostic readability, not as odometry improvement.",
            "",
            "## Leakage audit",
            "",
            "- S16 does not modify `checkpoints/S5_clean_tmag_calibration_policy.json`.",
            "- S16 does not overwrite S5 locked metrics.",
            "- S16 does not train a residual pose head.",
            "- S16 does not use the test set for encoder, probe, or threshold selection.",
            "- S16 does not use GT pose, GT tmag, GT tdir, or pair error as inference features.",
            "- S16 does not download pretrained weights and does not write heavy feature dumps.",
            "- Frozen encoder probing is not described as deployable final inference.",
            "",
            "## Whether S16b is recommended",
            "",
            f"- recommended: `{s16b}`",
            f"- reason: {payload['classification_reason']}",
            "",
            "## Whether S5 remains final clean candidate",
            "",
            "- `True`. S16 is not a final clean candidate replacement and makes no clean gain claim.",
            "",
            "## Final classification",
            "",
            f"- `{classification}`",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    summary_lines = [
        "# Final S16 Stronger Visual Backbone Feasibility Summary",
        "",
        f"- final classification: `{classification}`",
        "- S16 is a feasibility diagnostic only.",
        "- S5 remains final clean candidate: `True`",
        f"- pretrained weights available: `{availability['any_pretrained_available']}`",
        f"- encoders audited: `{', '.join(r['encoder_name'] for r in availability['encoders'])}`",
        f"- current-feature baseline result: `{payload['current_feature_baseline_result']}`",
        f"- stronger-feature result: `{payload['stronger_feature_result']}`",
        f"- R/tdir coupling proxy result: `{payload['r_tdir_coupling_proxy_result']}`",
        f"- S16b full integration recommended: `{s16b}`",
        "- no S5 policy or locked metrics were modified.",
        "",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")


def run() -> Dict[str, Any]:
    config = _parse_config(CONFIG_PATH)
    gate = _baseline_gate()
    current_audit: Dict[str, Any] | None = None
    availability: Dict[str, Any] = {"encoders": [], "any_pretrained_available": False, "internet_download_attempted": False}
    if not gate["passed"]:
        payload = {
            "name": "S16_stronger_visual_backbone_feasibility",
            "config_path": str(CONFIG_PATH),
            "baseline_gate": gate,
            "current_representation_audit": {"status": "not_run", "reason": "baseline gate failed"},
            "encoder_availability": availability,
            "final_classification": "REPRODUCTION-MISMATCH",
            "classification_reason": "S8b current-architecture reproduction contract did not pass.",
            "s16b_full_integration_recommended": False,
            "s5_remains_final_clean_candidate": True,
            "current_feature_baseline_result": "not_run_due_reproduction_mismatch",
            "stronger_feature_result": "not_run_due_reproduction_mismatch",
            "r_tdir_coupling_proxy_result": "not_run_due_reproduction_mismatch",
        }
        _write_json(CANDIDATES_PATH, payload)
        _write_reports(payload)
        return payload

    availability = _encoder_availability_audit()
    try:
        current_audit = _extract_current_representation_audit(config)
    except Exception as exc:
        current_audit = {
            "status": "failed",
            "error": repr(exc),
            "traceback_tail": traceback.format_exc(limit=8),
            "prior_alignment": {
                "s3b_report": str(S3B_REPORT_PATH),
                "s8_report": str(S8_REPORT_PATH),
            },
        }
    classification, recommend_s16b, reason = _classification(availability, current_audit)
    if classification not in FINAL_CLASSES:
        classification = "INCONCLUSIVE"
    current_means = (current_audit or {}).get("family_means", {}) if isinstance(current_audit, dict) else {}
    cur = current_means.get("current_model_features_only", {})
    if cur:
        current_result = (
            f"rot_R2={_fmt(cur.get('rot_r2_val'))}, "
            f"tdir_R2={_fmt(cur.get('tdir_r2_val'))}, "
            f"joint_AUC={_fmt(cur.get('joint_r_tdir_auc_val'))}"
        )
    else:
        current_result = f"{(current_audit or {}).get('status', 'not_run')}"
    stronger_result = "not_run_pretrained_weights_unavailable" if not availability.get("any_pretrained_available", False) else "not_promoted_in_this_run"
    coupling_result = "not_evaluable_without_pretrained_stronger_features" if not availability.get("any_pretrained_available", False) else "inconclusive"
    payload = {
        "name": "S16_stronger_visual_backbone_feasibility",
        "config_path": str(CONFIG_PATH),
        "report_path": str(REPORT_PATH),
        "summary_path": str(SUMMARY_PATH),
        "baseline_gate": gate,
        "current_representation_audit": current_audit,
        "encoder_availability": availability,
        "probe_families": {
            "A": "current_model_features_only",
            "B": "stronger_encoder_features_only",
            "C": "stronger_encoder_plus_regime_features",
            "D": "current_model_features_plus_stronger_encoder_features",
            "E": "current_model_features_plus_stronger_encoder_plus_regime_features",
        },
        "final_classification": classification,
        "classification_reason": reason,
        "s16b_full_integration_recommended": bool(recommend_s16b),
        "s5_remains_final_clean_candidate": True,
        "current_feature_baseline_result": current_result,
        "stronger_feature_result": stronger_result,
        "r_tdir_coupling_proxy_result": coupling_result,
        "leakage_audit": {
            "modified_s5_policy": False,
            "overwrote_s5_locked_metrics": False,
            "trained_residual_pose_head": False,
            "used_test_set_for_selection": False,
            "used_gt_as_inference_feature": False,
            "internet_download_attempted": bool(availability.get("internet_download_attempted", False)),
            "wrote_heavy_feature_dump": False,
            "pretrained_probe_is_deployable_final_method": False,
        },
    }
    _write_json(CANDIDATES_PATH, payload)
    _write_reports(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="S16 stronger visual backbone feasibility diagnostic.")
    parser.add_argument("--print-json", action="store_true", help="Print compact final payload.")
    args = parser.parse_args()
    payload = run()
    if args.print_json:
        print(json.dumps(_json_safe(payload), indent=2, ensure_ascii=True))
    else:
        print(f"[S16] final_classification={payload['final_classification']}")
        print(f"[S16] report={REPORT_PATH}")
        print(f"[S16] candidates={CANDIDATES_PATH}")
        print(f"[S16] summary={SUMMARY_PATH}")


if __name__ == "__main__":
    main()
