#!/usr/bin/env python3
from __future__ import annotations

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
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from eval_clean_policy import (  # type: ignore
    DtBucketScaledMagnitudeModel,
    _build_eval_dataset,
    _cfg_from_dict,
    _load_ckpt_cfg,
    _q,
    _safe_float,
)
from model import PanoramaRelPoseModel
from train_mvp import eval_odometry_sequence


POLICY_PATH = REPO_ROOT / "checkpoints" / "S2b_clean_fine_rot_policy.json"
CKPT_PATH = REPO_ROOT / "checkpoints" / "T57b_no_dt_multiscale_tmag_head_400" / "final.pt"
REPORT_PATH = REPO_ROOT / "checkpoints" / "S5_tmag_regime_aware_calibration_report.md"
CANDIDATES_PATH = REPO_ROOT / "checkpoints" / "S5_tmag_regime_aware_calibration_candidates.json"
FIG_DIR = REPO_ROOT / "checkpoints" / "S5_tmag_regime_aware_calibration_figures"
S2B_DRIFT = 1.327402
S2B_ATE = 7.352371
S2B_PATH_RATIO = 0.934984
MAX_FIT_PAIRS = 512


@dataclass
class CandidateSpec:
    name: str
    family: str
    params: Dict[str, Any]
    clean_eligible: bool
    simplicity_rank: Tuple[float, float, float]


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


def _fmt(v: Any, digits: int = 6) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.{digits}f}"
    return str(v)


def _load_base_cfg():
    cfg_dict = _load_ckpt_cfg(CKPT_PATH)
    cfg = _cfg_from_dict(cfg_dict)
    cfg.use_fine_stage = True
    cfg.fine_rot_fuse_strength = 0.45
    cfg.fine_tdir_fuse_strength = 0.0
    cfg.fine_tmag_fuse_strength = 0.0
    cfg.use_geometry_refine = False
    cfg.tmag_condition_on_dt = False
    return cfg


def _load_base_model(device: torch.device) -> Tuple[PanoramaRelPoseModel, Dict[str, Any]]:
    cfg = _load_base_cfg()
    model = PanoramaRelPoseModel(cfg, device).to(device)
    payload = torch.load(str(CKPT_PATH), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    return model, {"missing": list(msg.missing_keys), "unexpected": list(msg.unexpected_keys)}


def _build_s2b_model(device: torch.device):
    base_model, load_summary = _load_base_model(device)
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    factors = {str(k): float(v) for k, v in policy["effective_bucket_factors"].items()}
    wrapped = DtBucketScaledMagnitudeModel(base_model, factors).to(device)
    return wrapped, base_model.cfg, load_summary


def _build_loader(cfg, ds) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(getattr(cfg, "pin_memory", False)),
        drop_last=False,
    )


def _candidate_specs() -> List[CandidateSpec]:
    specs = [
        CandidateSpec("baseline_noop", "noop", {}, True, (0.0, 0.0, 0.0)),
        CandidateSpec("fit_piecewise_clip_0p70_1p10", "pred_piecewise_fit", {"clip_lo": 0.70, "clip_hi": 1.10, "quantiles": [0.0, 0.67, 0.90, 1.0]}, True, (1.0, 0.70, 1.10)),
        CandidateSpec("fit_piecewise_clip_0p80_1p05", "pred_piecewise_fit", {"clip_lo": 0.80, "clip_hi": 1.05, "quantiles": [0.0, 0.67, 0.90, 1.0]}, True, (1.0, 0.80, 1.05)),
        CandidateSpec("highpred_q80_mid1p00_high0p90", "high_pred_shrink", {"q_mid": 0.80, "mid_scale": 1.00, "high_scale": 0.90}, True, (2.0, 0.80, 0.90)),
        CandidateSpec("highpred_q90_mid0p95_high0p80", "high_pred_shrink", {"q_mid": 0.90, "mid_scale": 0.95, "high_scale": 0.80}, True, (2.0, 0.90, 0.80)),
        CandidateSpec("interaction_dt1_k20_scale0p85", "dtk_interaction", {"pred_q": 0.90, "interaction_scale": 0.85}, True, (3.0, 0.90, 0.85)),
        CandidateSpec("winsor_q90", "winsorize", {"pred_q": 0.90}, True, (4.0, 0.90, 0.0)),
        CandidateSpec("oracle_gt_piecewise_clip_0p70_1p10", "oracle_gt_piecewise", {"clip_lo": 0.70, "clip_hi": 1.10, "quantiles": [0.0, 0.67, 0.90, 1.0]}, False, (9.0, 0.70, 1.10)),
    ]
    return specs


def _subsample_indices(n: int, cap: int) -> List[int]:
    if n <= cap:
        return list(range(n))
    raw = np.linspace(0, n - 1, num=cap)
    out = []
    seen = set()
    for v in raw:
        idx = int(round(float(v)))
        idx = max(0, min(n - 1, idx))
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def _train_records(model, ds, device: torch.device, *, max_pairs: int = MAX_FIT_PAIRS) -> List[Dict[str, Any]]:
    rows = []
    manifest = ds.manifest()
    sample_indices = _subsample_indices(len(manifest), max_pairs)
    with torch.no_grad():
        for ds_idx in sample_indices:
            meta = manifest[ds_idx]
            sample = ds[ds_idx]
            IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
            IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
            dt_world = float(meta.get("dt_world", sample.get("t_gt_mag", 0.0)))
            dt_tensor = torch.tensor([dt_world], device=device, dtype=torch.float32)
            _, _t_pred, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
            pred_mag = float(aux["t_mag"].detach().float().view(-1).cpu().numpy()[0])
            rows.append(
                {
                    "dataset_index": int(ds_idx),
                    "scene": str(meta.get("scene")),
                    "seq": str(meta.get("seq")),
                    "k": int(meta.get("k", -1)),
                    "dt_world": float(dt_world),
                    "tmag_gt": float(sample["t_gt_mag"]),
                    "tmag_pred": float(pred_mag),
                    "ratio_gt_over_pred": float(float(sample["t_gt_mag"]) / max(pred_mag, 1.0e-8)),
                }
            )
    return rows


def _eval_pair_records(model, ds, device: torch.device) -> List[Dict[str, Any]]:
    rows = []
    manifest = ds.manifest()
    with torch.no_grad():
        for ds_idx, meta in enumerate(manifest):
            sample = ds[ds_idx]
            IA = sample["IA"].unsqueeze(0).to(device, non_blocking=True)
            IB = sample["IB"].unsqueeze(0).to(device, non_blocking=True)
            dt_world = float(meta.get("dt_world", sample.get("t_gt_mag", 0.0)))
            dt_tensor = torch.tensor([dt_world], device=device, dtype=torch.float32)
            _, _t_pred, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt_tensor)
            pred_mag = float(aux["t_mag"].detach().float().view(-1).cpu().numpy()[0])
            gt_mag = float(sample["t_gt_mag"])
            rows.append(
                {
                    "dataset_index": int(ds_idx),
                    "scene": str(meta.get("scene")),
                    "seq": str(meta.get("seq")),
                    "k": int(meta.get("k", -1)),
                    "dt_world": float(meta.get("dt_world", gt_mag)),
                    "tmag_gt": gt_mag,
                    "tmag_pred": pred_mag,
                }
            )
    return rows


def _quantile_edges(vals: Sequence[float], q: Sequence[float]) -> List[float]:
    arr = np.asarray(list(vals), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [0.0, 1.0]
    edges = np.quantile(arr, q).astype(np.float64).tolist()
    edges[0] -= 1.0e-6
    edges[-1] += 1.0e-6
    for idx in range(1, len(edges)):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = edges[idx - 1] + 1.0e-6
    return [float(x) for x in edges]


def _bucket_idx(v: float, edges: Sequence[float]) -> int:
    for idx in range(len(edges) - 1):
        if float(edges[idx]) <= float(v) < float(edges[idx + 1]):
            return idx
    return len(edges) - 2


def _fit_candidate(spec: CandidateSpec, train_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    params = dict(spec.params)
    pred_vals = [float(r["tmag_pred"]) for r in train_rows]
    gt_vals = [float(r["tmag_gt"]) for r in train_rows]
    if spec.family == "noop":
        return {"family": spec.family}
    if spec.family == "pred_piecewise_fit":
        edges = _quantile_edges(pred_vals, params["quantiles"])
        factors = []
        for bi in range(len(edges) - 1):
            ratios = [float(r["ratio_gt_over_pred"]) for r in train_rows if _bucket_idx(float(r["tmag_pred"]), edges) == bi]
            med = float(np.median(ratios)) if ratios else 1.0
            med = max(float(params["clip_lo"]), min(float(params["clip_hi"]), med))
            factors.append(float(med))
        return {"family": spec.family, "edges": edges, "factors": factors, **params}
    if spec.family == "high_pred_shrink":
        q_mid = float(np.quantile(np.asarray(pred_vals, dtype=np.float64), float(params["q_mid"])))
        q_hi = float(np.quantile(np.asarray(pred_vals, dtype=np.float64), 0.95))
        return {"family": spec.family, "q_mid_value": q_mid, "q_hi_value": q_hi, **params}
    if spec.family == "dtk_interaction":
        q_val = float(np.quantile(np.asarray(pred_vals, dtype=np.float64), float(params["pred_q"])))
        return {"family": spec.family, "pred_q_value": q_val, **params}
    if spec.family == "winsorize":
        q_val = float(np.quantile(np.asarray(pred_vals, dtype=np.float64), float(params["pred_q"])))
        return {"family": spec.family, "cap_value": q_val, **params}
    if spec.family == "oracle_gt_piecewise":
        edges = _quantile_edges(gt_vals, params["quantiles"])
        factors = []
        for bi in range(len(edges) - 1):
            ratios = [float(r["ratio_gt_over_pred"]) for r in train_rows if _bucket_idx(float(r["tmag_gt"]), edges) == bi]
            med = float(np.median(ratios)) if ratios else 1.0
            med = max(float(params["clip_lo"]), min(float(params["clip_hi"]), med))
            factors.append(float(med))
        return {"family": spec.family, "edges": edges, "factors": factors, **params}
    raise ValueError(f"unsupported family: {spec.family}")


def _candidate_scale(fit_payload: Dict[str, Any], pred_tmag: float, meta: Dict[str, Any], sample: Dict[str, Any]) -> float:
    family = fit_payload["family"]
    if family == "noop":
        return 1.0
    if family == "pred_piecewise_fit":
        bi = _bucket_idx(float(pred_tmag), fit_payload["edges"])
        return float(fit_payload["factors"][bi])
    if family == "high_pred_shrink":
        if float(pred_tmag) >= float(fit_payload["q_hi_value"]):
            return float(fit_payload["high_scale"])
        if float(pred_tmag) >= float(fit_payload["q_mid_value"]):
            return float(fit_payload["mid_scale"])
        return 1.0
    if family == "dtk_interaction":
        dt_val = float(meta.get("dt_world", sample.get("t_gt_mag", 0.0)))
        k_val = int(meta.get("k", -1))
        if dt_val >= 1.0 and k_val == 20 and float(pred_tmag) >= float(fit_payload["pred_q_value"]):
            return float(fit_payload["interaction_scale"])
        return 1.0
    if family == "winsorize":
        cap = float(fit_payload["cap_value"])
        return min(1.0, cap / max(float(pred_tmag), 1.0e-8))
    if family == "oracle_gt_piecewise":
        gt = float(sample.get("t_gt_mag", 0.0))
        bi = _bucket_idx(gt, fit_payload["edges"])
        return float(fit_payload["factors"][bi])
    raise ValueError(f"unsupported family: {family}")


class RegimeAwareCalibrationWrapper(torch.nn.Module):
    def __init__(self, base: DtBucketScaledMagnitudeModel, fit_payload: Dict[str, Any], context: Dict[str, Any]) -> None:
        super().__init__()
        self.base = base
        self.fit_payload = dict(fit_payload)
        self.context = context
        self.cfg = base.cfg

    def forward(self, IA: torch.Tensor, IB: torch.Tensor, *, enable_depth_fusion=None, dt_world=None):
        R_pred, t_pred, aux = self.base(IA, IB, enable_depth_fusion=enable_depth_fusion, dt_world=dt_world)
        meta = dict(self.context.get("meta", {}))
        sample = dict(self.context.get("sample", {}))
        pred_mag = float(aux["t_mag"].detach().float().view(-1)[0].cpu())
        scale = _candidate_scale(self.fit_payload, pred_mag, meta, sample)
        if abs(scale - 1.0) < 1.0e-12:
            aux = dict(aux)
            aux["s5_calibration_scale"] = torch.tensor(1.0, device=IA.device, dtype=torch.float32)
            return R_pred, t_pred, aux
        aux = dict(aux)
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
        aux["s5_calibration_scale"] = fac
        return R_pred, t_pred, aux


@dataclass
class EvalRow:
    candidate_name: str
    split: str
    fold: str
    drift: float
    ate: float
    path_ratio: float
    rpe_rot: float
    rpe_trans_dir: float
    rpe_trans_mag: float
    rot: float
    tdir_abs: float
    tdir_local_A_abs: float
    pair_pos_err_mean: float
    worst_bucket_pair_pos_err: float
    high_error_rate: float
    tmag_ratio_mean: float
    odom_selected_k: int
    num_pairs: int
    num_chains: int
    load_missing: int
    load_unexpected: int
    output_dir: str


def _run_eval(
    device: torch.device,
    ds,
    candidate_name: str,
    fit_payload: Dict[str, Any],
    *,
    split: str,
    fold: str,
    out_dir: Path,
) -> EvalRow:
    base, cfg, load_summary = _build_s2b_model(device)
    if load_summary["unexpected"]:
        raise RuntimeError(f"{candidate_name}: unexpected checkpoint keys: {load_summary['unexpected'][:8]}")
    context: Dict[str, Any] = {}
    ctx_ds = ContextDataset(ds, context)
    wrapped = RegimeAwareCalibrationWrapper(base, fit_payload, context).to(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    odom = eval_odometry_sequence(wrapped, ctx_ds, device, cfg, output_dir=str(out_dir), step=0, upd=0)
    pair_rows = _eval_pair_records(wrapped, ctx_ds, device)
    pos_errs = []
    bucket_errors: Dict[str, List[float]] = {}
    ratio_vals = []
    for rec in pair_rows:
        gt = float(rec["tmag_gt"])
        pred = float(rec["tmag_pred"])
        pos = abs(pred - gt)
        pos_errs.append(pos)
        ratio = pred / max(gt, 1.0e-8)
        ratio_vals.append(ratio)
        dt = float(rec["dt_world"])
        if 0.1 <= dt < 0.3:
            dt_bucket = "[0.1,0.3)"
        elif 0.3 <= dt < 0.5:
            dt_bucket = "[0.3,0.5)"
        elif 0.5 <= dt < 1.0:
            dt_bucket = "[0.5,1)"
        elif dt < 0.1:
            dt_bucket = "<0.1"
        else:
            dt_bucket = ">=1.0"
        k_val = int(rec.get("k", -1))
        label = f"{dt_bucket}|k={k_val}"
        bucket_errors.setdefault(label, []).append(pos)
    high_err = []
    if pos_errs:
        thresh = float(np.percentile(np.asarray(pos_errs, dtype=np.float64), 75))
        high_err = [1.0 if pos >= thresh else 0.0 for pos in pos_errs]
    worst_bucket = max((np.mean(v) for v in bucket_errors.values()), default=float("nan"))
    return EvalRow(
        candidate_name=candidate_name,
        split=split,
        fold=fold,
        drift=_safe_float(odom.get("odom_metric_drift")),
        ate=_safe_float(odom.get("odom_metric_ATE")),
        path_ratio=_safe_float(odom.get("odom_shape_metric_mean_path_length_ratio")),
        rpe_rot=_safe_float(odom.get("odom_metric_RPE_rot")),
        rpe_trans_dir=_safe_float(odom.get("odom_metric_RPE_trans_dir")),
        rpe_trans_mag=_safe_float(odom.get("odom_metric_RPE_trans_mag")),
        rot=float("nan"),
        tdir_abs=float("nan"),
        tdir_local_A_abs=float("nan"),
        pair_pos_err_mean=float(np.mean(pos_errs)) if pos_errs else float("nan"),
        worst_bucket_pair_pos_err=float(worst_bucket),
        high_error_rate=float(np.mean(high_err)) if high_err else float("nan"),
        tmag_ratio_mean=float(np.mean(ratio_vals)) if ratio_vals else float("nan"),
        odom_selected_k=int(odom.get("odom_selected_k", -1)),
        num_pairs=int(odom.get("odom_num_pairs", 0)),
        num_chains=int(odom.get("odom_num_chains", 0)),
        load_missing=len(load_summary["missing"]),
        load_unexpected=len(load_summary["unexpected"]),
        output_dir=str(out_dir.relative_to(REPO_ROOT)),
    )


def _mean(rows: Sequence[EvalRow], attr: str) -> float:
    vals = [float(getattr(r, attr)) for r in rows]
    return float(np.mean(vals)) if vals else float("nan")


def _collect_train_groups():
    cfg = _load_base_cfg()
    train_ds = _build_eval_dataset(cfg, split="train")
    manifest = train_ds.manifest()
    groups = sorted({(str(m.get("scene")), str(m.get("seq"))) for m in manifest})
    seq_to_indices: Dict[Tuple[str, str], List[int]] = {g: [] for g in groups}
    for idx, meta in enumerate(manifest):
        key = (str(meta.get("scene")), str(meta.get("seq")))
        seq_to_indices[key].append(idx)
    target_groups = [("scene01", "seq01"), ("scene01", "seq02")]
    missing = [g for g in target_groups if g not in seq_to_indices]
    if missing:
        raise RuntimeError(f"missing expected S2/S3 CV groups: {missing}")
    target_map = {g: seq_to_indices[g] for g in target_groups}
    return cfg, train_ds, target_groups, target_map


def _hard_gate(
    cand_rows: Sequence[EvalRow],
    baseline_rows: Sequence[EvalRow],
    *,
    clean_eligible: bool,
) -> Tuple[bool, Dict[str, Any]]:
    mean_path = _mean(cand_rows, "path_ratio")
    mean_ate = _mean(cand_rows, "ate")
    mean_drift = _mean(cand_rows, "drift")
    base_mean_path = _mean(baseline_rows, "path_ratio")
    base_mean_ate = _mean(baseline_rows, "ate")
    base_mean_drift = _mean(baseline_rows, "drift")
    worst_fold_drift = max(float(r.drift) for r in cand_rows)
    baseline_worst_fold_drift = max(float(r.drift) for r in baseline_rows)
    per_fold_path_ok = all(abs(float(r.path_ratio) - float(b.path_ratio)) <= 0.03 for r, b in zip(cand_rows, baseline_rows))
    ok = (
        clean_eligible
        and abs(mean_path - base_mean_path) <= 0.03
        and per_fold_path_ok
        and mean_drift <= base_mean_drift + 1.0e-9
        and mean_ate < base_mean_ate - 1.0e-9
        and worst_fold_drift <= baseline_worst_fold_drift + 0.03
        and all(int(r.load_unexpected) == int(b.load_unexpected) for r, b in zip(cand_rows, baseline_rows))
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
    }


def _select_candidate(aggregates: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    eligible = [a for a in aggregates if a["satisfies_clean_gate"]]
    if not eligible:
        return None
    eligible.sort(key=lambda a: (a["mean_ATE"], a["mean_drift"], a["simplicity_rank"]))
    return eligible[0]


def _plot_cv_ate(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    labels = [r["name"] for r in rows]
    vals = [float(r["mean_ATE"]) for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    ax.bar(x, vals)
    ax.axhline(S2B_ATE, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("Mean CV ATE by Candidate")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_cv_path(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    labels = [r["name"] for r in rows]
    vals = [float(r["mean_path_ratio"]) for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    ax.bar(x, vals)
    ax.axhline(S2B_PATH_RATIO, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("Mean CV Path Ratio by Candidate")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, train_ds_full, groups, seq_to_indices = _collect_train_groups()
    test_ds = _build_eval_dataset(cfg, split="test")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

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

    baseline_fit = {"family": "noop"}
    baseline_rows = []
    for fold in fold_plan:
        row = _run_eval(
            device,
            fold["val_ds"],
            "baseline_noop",
            baseline_fit,
            split="train_cv",
            fold=fold["fold_name"],
            out_dir=REPO_ROOT / "checkpoints" / "S5_tmag_regime_aware_calibration" / fold["fold_name"] / "baseline_noop",
        )
        baseline_rows.append(row)
    baseline_test_row = _run_eval(
        device,
        test_ds,
        "baseline_noop",
        baseline_fit,
        split="test_baseline",
        fold="baseline_test",
        out_dir=REPO_ROOT / "checkpoints" / "S5_tmag_regime_aware_calibration" / "baseline_test" / "baseline_noop",
    )

    fold_train_rows_map: Dict[str, List[Dict[str, Any]]] = {}
    for fold in fold_plan:
        train_model, _, load_summary = _build_s2b_model(device)
        if load_summary["unexpected"]:
            raise RuntimeError(f"unexpected checkpoint keys while preparing {fold['fold_name']}: {load_summary['unexpected'][:8]}")
        fold_train_rows_map[str(fold["fold_name"])] = _train_records(train_model, fold["train_ds"], device)

    full_train_model, _, load_summary = _build_s2b_model(device)
    if load_summary["unexpected"]:
        raise RuntimeError(f"unexpected checkpoint keys while preparing full-train model: {load_summary['unexpected'][:8]}")
    full_train_rows = None

    candidate_records: List[Dict[str, Any]] = []
    specs = _candidate_specs()
    for spec in specs:
        fold_rows: List[EvalRow] = []
        fit_payloads = []
        for fold in fold_plan:
            train_rows = fold_train_rows_map[str(fold["fold_name"])]
            fit_payload = _fit_candidate(spec, train_rows)
            fit_payloads.append({"fold": fold["fold_name"], "fit_payload": fit_payload})
            row = _run_eval(
                device,
                fold["val_ds"],
                spec.name,
                fit_payload,
                split="train_cv",
                fold=fold["fold_name"],
                out_dir=REPO_ROOT / "checkpoints" / "S5_tmag_regime_aware_calibration" / fold["fold_name"] / spec.name,
            )
            fold_rows.append(row)
        gate_ok, gate_payload = _hard_gate(fold_rows, baseline_rows, clean_eligible=spec.clean_eligible)
        candidate_records.append(
            {
                "name": spec.name,
                "family": spec.family,
                "params": spec.params,
                "clean_eligible": spec.clean_eligible,
                "simplicity_rank": spec.simplicity_rank,
                "fit_payloads": fit_payloads,
                "fold_rows": [r.__dict__ for r in fold_rows],
                "mean_ATE": _mean(fold_rows, "ate"),
                "mean_drift": _mean(fold_rows, "drift"),
                "mean_path_ratio": _mean(fold_rows, "path_ratio"),
                "mean_pair_pos_err": _mean(fold_rows, "pair_pos_err_mean"),
                "worst_fold_drift": max(float(r.drift) for r in fold_rows),
                "satisfies_clean_gate": gate_ok,
                "gate_payload": gate_payload,
            }
        )

    selected = _select_candidate(candidate_records)
    final_row = None
    oracle_row = None
    final_classification = "NO-STABLE-CALIBRATION-GAIN"
    selected_name = None

    if selected is not None:
        selected_name = str(selected["name"])
        spec = next(s for s in specs if s.name == selected_name)
        if full_train_rows is None:
            full_train_rows = _train_records(full_train_model, train_ds_full, device)
        fit_payload = _fit_candidate(spec, full_train_rows)
        final_row = _run_eval(
            device,
            test_ds,
            selected_name,
            fit_payload,
            split="test",
            fold="final_test",
            out_dir=REPO_ROOT / "checkpoints" / "S5_tmag_regime_aware_calibration" / "final_test" / selected_name,
        )
        if (
            final_row.ate < S2B_ATE
            and final_row.drift <= S2B_DRIFT + 1.0e-9
            and abs(final_row.path_ratio - S2B_PATH_RATIO) <= 0.03
        ):
            if spec.family == "dtk_interaction":
                final_classification = "INTERACTION-DT-K-TMAG-CLEAN-GAIN"
            else:
                final_classification = "TMAG-CALIBRATION-CLEAN-GAIN"
        else:
            final_classification = "TMAG-CALIBRATION-DIAGNOSTIC-ONLY"
    else:
        final_classification = "NO-STABLE-CALIBRATION-GAIN"

    oracle_spec = next(s for s in specs if s.family == "oracle_gt_piecewise")
    if full_train_rows is None:
        full_train_rows = _train_records(full_train_model, train_ds_full, device)
    oracle_fit = _fit_candidate(oracle_spec, full_train_rows)
    oracle_row = _run_eval(
        device,
        test_ds,
        oracle_spec.name,
        oracle_fit,
        split="test",
        fold="oracle_test",
        out_dir=REPO_ROOT / "checkpoints" / "S5_tmag_regime_aware_calibration" / "oracle_test" / oracle_spec.name,
    )

    CANDIDATES_PATH.write_text(
        json.dumps(
            {
                "baseline_fold_rows": [r.__dict__ for r in baseline_rows],
                "baseline_test_row": baseline_test_row.__dict__,
                "candidates": candidate_records,
                "selected": selected,
                "final_row": None if final_row is None else final_row.__dict__,
                "oracle_row": None if oracle_row is None else oracle_row.__dict__,
                "final_classification": final_classification,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    _plot_cv_ate(candidate_records, FIG_DIR / "cv_mean_ate.png")
    _plot_cv_path(candidate_records, FIG_DIR / "cv_mean_path_ratio.png")

    lines: List[str] = []
    lines.append("# S5 tmag regime-aware calibration report\n\n")
    lines.append("## Executive summary\n\n")
    lines.append(f"- S2b baseline target: drift=`{S2B_DRIFT}`, ATE=`{S2B_ATE}`, path_ratio=`{S2B_PATH_RATIO}`\n")
    lines.append(f"- final classification: `{final_classification}`\n")
    lines.append(f"- selected candidate: `{selected_name if selected_name is not None else 'None'}`\n\n")

    lines.append("## Baseline reproduction\n\n")
    lines.append(f"- policy: `{POLICY_PATH}`\n")
    lines.append(f"- base checkpoint: `{CKPT_PATH}`\n")
    lines.append(f"- official S2b reference: drift=`{S2B_DRIFT}`, ATE=`{S2B_ATE}`, path_ratio=`{S2B_PATH_RATIO}`\n")
    lines.append(
        f"- reproduced baseline test: drift=`{_fmt(baseline_test_row.drift)}`, "
        f"ATE=`{_fmt(baseline_test_row.ate)}`, path_ratio=`{_fmt(baseline_test_row.path_ratio)}`\n"
    )
    lines.append(f"- train-CV baseline mean drift=`{_fmt(_mean(baseline_rows, 'drift'))}`, mean ATE=`{_fmt(_mean(baseline_rows, 'ate'))}`, mean path_ratio=`{_fmt(_mean(baseline_rows, 'path_ratio'))}`\n\n")

    lines.append("## Candidate definitions\n\n")
    for spec in specs:
        lines.append(f"- `{spec.name}` ({spec.family}): clean_eligible=`{spec.clean_eligible}` params=`{spec.params}`\n")
    lines.append("\n")

    lines.append("## Train-CV selection protocol\n\n")
    lines.append(f"- folds: `{[f['fold_name'] for f in fold_plan]}`\n")
    lines.append("- no model parameter training\n")
    lines.append("- fit uses train-only labels on each fold train subset\n")
    lines.append("- final selection excludes oracle-only candidates\n")
    lines.append("- hard gates: mean ATE improves over S2b fold baseline, mean drift does not worsen, path_ratio stays within baseline +/- 0.03, worst-fold drift within baseline worst + 0.03, unexpected unchanged\n\n")

    lines.append("## CV results table\n\n")
    lines.append("| candidate | family | mean_drift | mean_ATE | mean_path_ratio | mean_pair_pos_err | worst_fold_drift | clean_gate |\n")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
    for rec in candidate_records:
        lines.append(
            f"| {rec['name']} | {rec['family']} | {_fmt(rec['mean_drift'])} | {_fmt(rec['mean_ATE'])} | {_fmt(rec['mean_path_ratio'])} | {_fmt(rec['mean_pair_pos_err'])} | {_fmt(rec['worst_fold_drift'])} | {rec['satisfies_clean_gate']} |\n"
        )
    lines.append("\n")

    lines.append("## Hard-gate audit\n\n")
    for rec in candidate_records:
        gp = rec["gate_payload"]
        lines.append(
            f"- `{rec['name']}`: mean_ATE `{_fmt(gp['mean_ATE'])}` vs baseline `{_fmt(gp['baseline_mean_ATE'])}`, "
            f"mean_drift `{_fmt(gp['mean_drift'])}` vs baseline `{_fmt(gp['baseline_mean_drift'])}`, "
            f"mean_path `{_fmt(gp['mean_path_ratio'])}` vs baseline `{_fmt(gp['baseline_mean_path_ratio'])}`, "
            f"worst_fold_drift `{_fmt(gp['worst_fold_drift'])}` vs baseline worst `{_fmt(gp['baseline_worst_fold_drift'])}`, "
            f"per_fold_path_ok=`{gp['per_fold_path_ok']}` -> clean_gate=`{rec['satisfies_clean_gate']}`\n"
        )
    lines.append("\n")

    if final_row is not None:
        lines.append("## Final test result\n\n")
        lines.append(f"- candidate: `{selected_name}`\n")
        lines.append(f"- drift = `{_fmt(final_row.drift)}`\n")
        lines.append(f"- ATE = `{_fmt(final_row.ate)}`\n")
        lines.append(f"- path_ratio = `{_fmt(final_row.path_ratio)}`\n")
        lines.append(f"- RPE_rot = `{_fmt(final_row.rpe_rot)}`\n")
        lines.append(f"- RPE_trans_dir = `{_fmt(final_row.rpe_trans_dir)}`\n")
        lines.append(f"- RPE_trans_mag = `{_fmt(final_row.rpe_trans_mag)}`\n")
        lines.append(f"- pair_pos_err_mean = `{_fmt(final_row.pair_pos_err_mean)}`\n")
        lines.append(f"- worst_bucket_pair_pos_err = `{_fmt(final_row.worst_bucket_pair_pos_err)}`\n")
        lines.append(f"- tmag_ratio_mean = `{_fmt(final_row.tmag_ratio_mean)}`\n\n")
    else:
        lines.append("## Final test result\n\n")
        lines.append("- no clean candidate was selected, so no final test was run\n\n")

    lines.append("## Regime-level before/after comparison\n\n")
    lines.append(f"- baseline worst-fold drift = `{_fmt(max(float(r.drift) for r in baseline_rows))}`\n")
    if final_row is not None:
        lines.append(f"- selected candidate final pair_pos_err_mean = `{_fmt(final_row.pair_pos_err_mean)}`\n")
        lines.append(f"- selected candidate final worst_bucket_pair_pos_err = `{_fmt(final_row.worst_bucket_pair_pos_err)}`\n")
    lines.append("- S4 indicated the dominant removable mass was in high tmag regime; S5 tests whether train-CV-selected inference-time shrinkage/capping can exploit that cleanly.\n\n")

    lines.append("## Oracle-only diagnostic\n\n")
    if oracle_row is not None:
        lines.append(f"- oracle candidate: `{oracle_spec.name}`\n")
        lines.append(f"- oracle drift = `{_fmt(oracle_row.drift)}`\n")
        lines.append(f"- oracle ATE = `{_fmt(oracle_row.ate)}`\n")
        lines.append(f"- oracle path_ratio = `{_fmt(oracle_row.path_ratio)}`\n")
        lines.append("- oracle uses gt_tmag bucket at fit/application time for upper-bound diagnosis only and is not deployment-eligible\n\n")

    lines.append("## Final classification\n\n")
    lines.append(f"- `{final_classification}`\n")
    if final_classification in {"TMAG-CALIBRATION-CLEAN-GAIN", "INTERACTION-DT-K-TMAG-CLEAN-GAIN"}:
        lines.append("- candidate is eligible to replace S2b as the current clean candidate\n")
    else:
        lines.append("- S2b remains the current clean candidate\n")
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "report": str(REPORT_PATH.relative_to(REPO_ROOT)),
                "final_classification": final_classification,
                "selected_candidate": selected_name,
                "final_test": None if final_row is None else {
                    "drift": final_row.drift,
                    "ATE": final_row.ate,
                    "path_ratio": final_row.path_ratio,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
