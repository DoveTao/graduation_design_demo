"""
File: train_mvp.py
Description:
    Training entry point for the MVP version of the graduation design project.
    This file handles experiment setup, dataset loading, model construction,
    loss computation, optimization, validation, logging, and checkpoint saving.

Main Components:
    - Environment, random seed, and experiment directory initialization
    - Dataset and DataLoader construction for train, train-eval, and test splits
    - Model, optimizer, learning-rate scheduler, and AMP setup
    - Training loop with pose, epipolar, reliability, and optional depth losses
    - Evaluation diagnostics, bucketed metrics, visualization export, and checkpoints

Usage / Role:
    Serves as the main training script for relative pose learning on panoramic
    image pairs.

Notes:
    This script is tailored for the graduation design MVP, coarse matching,
    optional fine-stage extensions, epipolar matching loss, limited GPU memory,
    and checkpoint-based experiment comparison.
"""

import argparse
import ast
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config import Config
from dataset_pano_only import RflyPanoPanoramaPairsEvalFixedKList, RflyPanoPanoramaPairsMixedK
from losses import (
    depth_smoothness_loss,
    epipolar_gt_band_nll_loss,
    epipolar_gt_matching_loss,
    epipolar_simplified_loss,
    erp_photometric_loss,
    pose_loss,
    translation_direction_loss,
    translation_magnitude_loss,
)
from model import PanoramaRelPoseModel
from erp_sampling import warp_erp_with_depth_pose
from geometry_refine import refine_pose_from_matches
from pose_head import matrix_geodesic_distance


def _cfg_to_dict(cfg):
    try:
        return asdict(cfg)
    except Exception:
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}


def _parse_args():
    parser = argparse.ArgumentParser(description="Train the panoramic relative pose MVP model.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a Config field. Can be passed multiple times.",
    )
    return parser.parse_args()


def _coerce_cfg_value(raw: str, current: Any) -> Any:
    text = str(raw).strip()
    if isinstance(current, bool):
        lowered = text.lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean value: {raw}")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(text)
    if isinstance(current, float):
        return float(text)
    if isinstance(current, tuple):
        try:
            value = ast.literal_eval(text)
        except Exception:
            value = tuple(part.strip() for part in text.split(",") if part.strip())
        if not isinstance(value, tuple):
            value = tuple(value) if isinstance(value, list) else (value,)
        if len(current) > 0:
            elem_type = type(current[0])
            value = tuple(elem_type(v) for v in value)
        return value
    if current is None:
        try:
            return ast.literal_eval(text)
        except Exception:
            return text
    if isinstance(current, str):
        return text
    try:
        return type(current)(text)
    except Exception:
        return text


def _apply_cfg_overrides(cfg: Config, overrides: List[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected KEY=VALUE.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key or not hasattr(cfg, key):
            raise ValueError(f"Unknown Config field: {key}")
        old_value = getattr(cfg, key)
        new_value = _coerce_cfg_value(raw_value, old_value)
        setattr(cfg, key, new_value)
        print(f"[CfgOverride] {key}: {old_value!r} -> {new_value!r}")


def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def _bucket_metric(rec: Dict[str, Any], *keys: str, default: float = float("nan")) -> float:
    for key in keys:
        if key in rec:
            try:
                return float(rec[key])
            except Exception:
                return default
    return default


def _write_eval_buckets_csv(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = [
        "split", "bucket_type", "bucket_label", "count", "rot", "tdir", "tdir_abs",
        "local_A_abs", "tmag_rel_err", "epi_mass", "top1", "top5", "entropy", "cycle_error",
    ]
    bucket_specs = [
        ("test", "k", payload.get("bucket_k", {})),
        ("test", "dt", payload.get("bucket_dt", {})),
        ("test", "kdt", payload.get("bucket_k_dt", {})),
        ("train", "k", payload.get("train_bucket_k", {})),
        ("train", "dt", payload.get("train_bucket_dt", {})),
        ("train", "kdt", payload.get("train_bucket_k_dt", {})),
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for split, bucket_type, bucket in bucket_specs:
            for label in sorted(bucket.keys(), key=_bucket_sort_key):
                rec = bucket[label]
                writer.writerow({
                    "split": split,
                    "bucket_type": bucket_type,
                    "bucket_label": label,
                    "count": int(rec.get("count", 0)),
                    "rot": _bucket_metric(rec, "rot"),
                    "tdir": _bucket_metric(rec, "tdir"),
                    "tdir_abs": _bucket_metric(rec, "tdir_abs"),
                    "local_A_abs": _bucket_metric(rec, "tdir_local_A_abs", "local_A_abs"),
                    "tmag_rel_err": _bucket_metric(rec, "tmag_rel_err"),
                    "epi_mass": _bucket_metric(rec, "epi_mass_in_gt_band", "epi_mass"),
                    "top1": _bucket_metric(rec, "top1_in_gt_band", "top1"),
                    "top5": _bucket_metric(rec, "top5_in_gt_band", "top5"),
                    "entropy": _bucket_metric(rec, "matching_entropy", "entropy"),
                    "cycle_error": _bucket_metric(rec, "cycle_error"),
                })


def _matching_diag_payload(step: int, upd: int, test_diag: Dict[str, Any], train_diag: Dict[str, Any]) -> Dict[str, Any]:
    metric_keys = [
        "epi_mass_in_gt_band", "top1_in_gt_band", "top5_in_gt_band",
        "matching_entropy", "max_matching_prob", "cycle_error",
        "coarse_epi_mass", "coarse_top1", "coarse_top5", "coarse_entropy",
        "coarse_max_prob", "coarse_cycle_error",
        "fine_epi_mass", "fine_top1", "fine_top5", "fine_entropy",
        "fine_max_prob", "fine_cycle_error",
        "routing_recall_in_gt_band", "allowed_mask_density", "no_candidate_row_ratio",
        "fine_routing_recall_in_gt_band", "fine_allowed_mask_density", "fine_no_candidate_row_ratio",
    ]

    def _select(diag: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key in metric_keys:
            if key not in diag:
                continue
            try:
                out[key] = float(diag[key])
            except Exception:
                pass
        return out

    return {
        "step": int(step),
        "upd": int(upd),
        "test": _select(test_diag),
        "train": _select(train_diag),
    }


def _latest_eval_metrics(eval_rec: Dict[str, Any]) -> Dict[str, Any]:
    latest_eval_keys = [
        "rot", "tdir", "tdir_abs",
        "raw", "flip", "R@t", "R@(-t)", "Rt@t", "Rt@(-t)",
        "tdir_local_A", "tdir_local_A_abs",
        "tmag_abs_err", "tmag_rel_err", "trans_vec_l2",
        "geom_refine_success_rate", "geom_rot", "geom_tdir", "geom_tdir_abs",
        "fused_rot", "fused_tdir_abs",
        "epi_mass_in_gt_band", "top1_in_gt_band", "top5_in_gt_band",
        "matching_entropy", "max_matching_prob", "cycle_error",
        "coarse_epi_mass", "coarse_top1", "coarse_top5", "coarse_entropy",
        "fine_epi_mass", "fine_top1", "fine_top5", "fine_entropy",
        "routing_recall_in_gt_band", "allowed_mask_density", "no_candidate_row_ratio",
    ]
    out: Dict[str, Any] = {}
    for key in latest_eval_keys:
        if key in eval_rec and isinstance(eval_rec[key], (int, float)):
            out[key] = float(eval_rec[key])
    for key, value in eval_rec.items():
        if not key.startswith("odom_"):
            continue
        if isinstance(value, (int, float)):
            out[key] = float(value)
        elif isinstance(value, (str, bool, list)):
            out[key] = value
    return out


def _scheduler_lambda(warmup_updates: int, hold_updates: int, drop1_updates: int, drop1_scale: float, drop2_scale: float):
    warmup_updates = max(int(warmup_updates), 0)
    hold_updates = max(int(hold_updates), warmup_updates)
    drop1_updates = max(int(drop1_updates), hold_updates)

    drop1_scale = float(drop1_scale)
    drop2_scale = float(drop2_scale)

    def fn(step_idx: int):
        s = int(step_idx)
        if warmup_updates > 0 and s < warmup_updates:
            return max(s / max(warmup_updates, 1), 1e-6)
        if s < hold_updates:
            return 1.0
        if s < drop1_updates:
            return drop1_scale
        return drop2_scale

    return fn


def _seed_everything(seed: int, deterministic: bool = False):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def _make_worker_init_fn(base_seed: int):
    base_seed = int(base_seed)

    def _worker_init_fn(worker_id: int):
        s = base_seed + int(worker_id)
        random.seed(s)
        np.random.seed(s % (2**32))
        torch.manual_seed(s)

    return _worker_init_fn


def _save_eval_manifest(path: str, ds, cfg):
    if not hasattr(ds, "manifest"):
        return
    payload = {
        "split_summary": getattr(ds, "split_summary", {}),
        "eval_pairs": ds.manifest(),
        "cfg_eval": {
            "eval_use_fixed_pairs": getattr(cfg, "eval_use_fixed_pairs", False),
            "eval_k_list": list(getattr(cfg, "eval_k_list", [])),
            "eval_pair_step": int(getattr(cfg, "eval_pair_step", 1)),
            "eval_min_dt": float(getattr(cfg, "eval_min_dt", 0.0)),
            "eval_max_dt": None if getattr(cfg, "eval_max_dt", None) is None else float(getattr(cfg, "eval_max_dt")),
        },
    }
    _save_json(path, payload)


def _print_dataset_sanity(name: str, ds) -> None:
    seqs_data = getattr(ds, "seqs_data", None)
    if not seqs_data:
        print(f"[Sanity-{name}] unavailable")
        return

    rot_k1 = []
    dt_k1 = []
    for sd in seqs_data:
        R_w = sd.get("R_w", None)
        t_w = sd.get("t_w", None)
        if R_w is None or t_w is None:
            continue
        n = int(sd.get("n", len(R_w)))
        for i in range(max(n - 1, 0)):
            R_rel = R_w[i + 1].T @ R_w[i]
            tr = float(np.trace(R_rel))
            cos = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
            rot_k1.append(float(np.degrees(np.arccos(cos))))
            dt_k1.append(float(np.linalg.norm(t_w[i + 1] - t_w[i])))

    if not rot_k1:
        print(f"[Sanity-{name}] no adjacent k=1 pairs")
        return

    rot_k1 = np.asarray(rot_k1, dtype=np.float32)
    dt_k1 = np.asarray(dt_k1, dtype=np.float32)
    print(
        f"[Sanity-{name}] k=1 rot_deg mean={rot_k1.mean():.3f} p50={np.percentile(rot_k1, 50):.3f} "
        f"p90={np.percentile(rot_k1, 90):.3f} max={rot_k1.max():.3f} | "
        f"dt_world mean={dt_k1.mean():.3f} p50={np.percentile(dt_k1, 50):.3f} "
        f"p90={np.percentile(dt_k1, 90):.3f} max={dt_k1.max():.3f} | n={rot_k1.size}"
    )


def _print_manifest_dt_sanity(name: str, ds) -> None:
    if not hasattr(ds, "manifest"):
        return
    manifest = ds.manifest()
    if not manifest:
        print(f"[Sanity-{name}] manifest empty")
        return
    dt = np.asarray([float(m["dt_world"]) for m in manifest if m.get("dt_world", None) is not None], dtype=np.float32)
    ks = sorted({int(m["k"]) for m in manifest if m.get("k", None) is not None})
    if dt.size == 0:
        print(f"[Sanity-{name}] no dt_world in manifest")
        return
    print(
        f"[Sanity-{name}] manifest k_list={ks} | dt_world min={dt.min():.3f} p10={np.percentile(dt, 10):.3f} "
        f"p50={np.percentile(dt, 50):.3f} p90={np.percentile(dt, 90):.3f} max={dt.max():.3f} | n={dt.size}"
    )


def _meta_batch_field(meta: Any, key: str, bsz: int, default=None):
    if not isinstance(meta, dict) or key not in meta:
        return [default] * int(bsz)
    v = meta[key]
    if torch.is_tensor(v):
        if v.ndim == 0:
            return [v.detach().cpu().item()] * int(bsz)
        arr = v.detach().cpu().view(-1).tolist()
        if len(arr) == int(bsz):
            return arr
        if len(arr) == 1:
            return arr * int(bsz)
        return (arr + [default] * int(bsz))[: int(bsz)]
    if isinstance(v, np.ndarray):
        arr = v.reshape(-1).tolist()
        if len(arr) == int(bsz):
            return arr
        if len(arr) == 1:
            return arr * int(bsz)
        return (arr + [default] * int(bsz))[: int(bsz)]
    if isinstance(v, (list, tuple)):
        arr = list(v)
        if len(arr) == int(bsz):
            return arr
        if len(arr) == 1:
            return arr * int(bsz)
        return (arr + [default] * int(bsz))[: int(bsz)]
    return [v] * int(bsz)


def _dt_bucket_label(dt_world: Optional[float], edges) -> str:
    if dt_world is None:
        return 'dt=unknown'
    try:
        x = float(dt_world)
    except Exception:
        return 'dt=unknown'
    prev = None
    for edge in edges:
        edge = float(edge)
        if x < edge:
            if prev is None:
                return f'dt<{edge:g}'
            return f'{prev:g}<=dt<{edge:g}'
        prev = edge
    return f'dt>={float(edges[-1]):g}' if len(edges) > 0 else 'dt=all'




def _translation_weight_from_dt(
    meta: Any,
    bsz: int,
    *,
    small_dt_thresh: float,
    small_dt_t_weight: float,
    ignore_dt_below: float = 0.0,
    ignore_weight: float = 0.0,
):
    dt_list = _meta_batch_field(meta, 'dt_world', bsz, default=None)
    w = []
    for x in dt_list:
        try:
            dt = float(x) if x is not None else None
            if dt is not None and float(ignore_dt_below) > 0.0 and dt < float(ignore_dt_below):
                w.append(float(ignore_weight))
            elif dt is not None and dt < float(small_dt_thresh):
                w.append(float(small_dt_t_weight))
            else:
                w.append(1.0)
        except Exception:
            w.append(1.0)
    return w


def _rotation_weight_from_k(meta: Any, bsz: int, *, large_k_thresh: int, large_k_rot_weight: float):
    k_list = _meta_batch_field(meta, 'k', bsz, default=None)
    weight_hi = float(large_k_rot_weight)
    if weight_hi <= 0.0:
        weight_hi = 1.0
    w = []
    for x in k_list:
        try:
            if x is not None and int(x) >= int(large_k_thresh):
                w.append(weight_hi)
            else:
                w.append(1.0)
        except Exception:
            w.append(1.0)
    return w


def _translation_magnitude_gt(batch: Dict[str, Any], meta: Any, bsz: int, device: torch.device) -> Optional[torch.Tensor]:
    t_gt_mag = batch.get("t_gt_mag", None)
    if t_gt_mag is not None:
        return t_gt_mag.to(device, non_blocking=True).float().view(-1)
    dt_list = _meta_batch_field(meta, "dt_world", bsz, default=None)
    if any(v is None for v in dt_list):
        return None
    try:
        vals = [float(v) for v in dt_list]
    except Exception:
        return None
    return torch.tensor(vals, device=device, dtype=torch.float32).view(-1)


def _cfg_tmag_weight(cfg: Config) -> float:
    if hasattr(cfg, "w_tmag"):
        return float(getattr(cfg, "w_tmag", 0.0))
    return float(getattr(cfg, "w_t_mag", 0.0))


def _cfg_tmag_effective_weight(cfg: Config, upd: int) -> float:
    base_w = _cfg_tmag_weight(cfg)
    if base_w <= 0.0:
        return 0.0
    start = int(getattr(cfg, "tmag_start_updates", 0))
    if upd < start:
        return 0.0
    ramp_updates = int(getattr(cfg, "tmag_ramp_updates", 0))
    if ramp_updates <= 0:
        return base_w
    ramp = min(1.0, max(0.0, float(upd - start) / float(ramp_updates)))
    return base_w * ramp


def _cfg_tdir_anchor_effective_weight(cfg: Config, upd: int) -> float:
    base_w = float(getattr(cfg, "w_tdir_anchor", 0.0))
    if (not bool(getattr(cfg, "use_tdir_anchor_loss", False))) or base_w <= 0.0:
        return 0.0
    start = int(getattr(cfg, "tdir_anchor_start_updates", 0))
    if upd < start:
        return 0.0
    ramp_updates = int(getattr(cfg, "tdir_anchor_ramp_updates", 0))
    if ramp_updates <= 0:
        return base_w
    ramp = min(1.0, max(0.0, float(upd - start) / float(ramp_updates)))
    return base_w * ramp


def _tdir_anchor_weight_from_meta(
    meta: Any,
    bsz: int,
    *,
    min_dt: float = 0.2,
    min_k: int = 0,
    device: torch.device,
) -> torch.Tensor:
    dt_list = _meta_batch_field(meta, "dt_world", bsz, default=None)
    k_list = _meta_batch_field(meta, "k", bsz, default=None)
    weights = []
    for dt_val, k_val in zip(dt_list, k_list):
        keep = True
        if float(min_dt) > 0.0:
            try:
                keep = keep and (dt_val is not None) and (float(dt_val) >= float(min_dt))
            except Exception:
                keep = False
        if int(min_k) > 0:
            try:
                keep = keep and (k_val is not None) and (int(k_val) >= int(min_k))
            except Exception:
                keep = False
        weights.append(1.0 if keep else 0.0)
    return torch.tensor(weights, device=device, dtype=torch.float32).view(-1)


def _translation_direction_anchor_loss(
    t_student: torch.Tensor,
    t_teacher: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    student = F.normalize(t_student.float(), dim=-1, eps=1e-6)
    teacher = F.normalize(t_teacher.detach().float(), dim=-1, eps=1e-6)
    loss = 1.0 - torch.sum(student * teacher, dim=-1).clamp(-1.0, 1.0)
    w = sample_weight.float().view(-1).to(loss.device)
    if float(w.sum().detach().cpu()) <= 0.0:
        return torch.zeros((), device=loss.device)
    return (loss.view(-1) * w).sum() / w.sum().clamp_min(1e-6)


def _bucket_init() -> Dict[str, Dict[str, float]]:
    return {}


def _bucket_update(
    store: Dict[str, Dict[str, float]],
    label: str,
    rot: float,
    tdir: float,
    tdir_abs: float,
    tdir_local_A: float,
    tdir_local_A_abs: float,
    epi_mass_in_gt_band: float,
    top1_in_gt_band: float,
    top5_in_gt_band: float,
    matching_entropy: float,
    max_matching_prob: float,
    cycle_error: float,
    tmag_rel_err: float = float("nan"),
):
    rec = store.setdefault(label, {
        'count': 0,
        'rot_sum': 0.0,
        'tdir_sum': 0.0,
        'tdir_abs_sum': 0.0,
        'tdir_local_A_sum': 0.0,
        'tdir_local_A_abs_sum': 0.0,
        'epi_mass_in_gt_band_sum': 0.0,
        'top1_in_gt_band_sum': 0.0,
        'top5_in_gt_band_sum': 0.0,
        'matching_entropy_sum': 0.0,
        'max_matching_prob_sum': 0.0,
        'cycle_error_sum': 0.0,
        'tmag_rel_err_sum': 0.0,
        'tmag_rel_err_count': 0,
    })
    rec['count'] += 1
    rec['rot_sum'] += float(rot)
    rec['tdir_sum'] += float(tdir)
    rec['tdir_abs_sum'] += float(tdir_abs)
    if np.isfinite(tdir_local_A):
        rec['tdir_local_A_sum'] += float(tdir_local_A)
    if np.isfinite(tdir_local_A_abs):
        rec['tdir_local_A_abs_sum'] += float(tdir_local_A_abs)
    if np.isfinite(epi_mass_in_gt_band):
        rec['epi_mass_in_gt_band_sum'] += float(epi_mass_in_gt_band)
    if np.isfinite(top1_in_gt_band):
        rec['top1_in_gt_band_sum'] += float(top1_in_gt_band)
    if np.isfinite(top5_in_gt_band):
        rec['top5_in_gt_band_sum'] += float(top5_in_gt_band)
    if np.isfinite(matching_entropy):
        rec['matching_entropy_sum'] += float(matching_entropy)
    if np.isfinite(max_matching_prob):
        rec['max_matching_prob_sum'] += float(max_matching_prob)
    if np.isfinite(cycle_error):
        rec['cycle_error_sum'] += float(cycle_error)
    if np.isfinite(tmag_rel_err):
        rec['tmag_rel_err_sum'] += float(tmag_rel_err)
        rec['tmag_rel_err_count'] += 1


def _bucket_finalize(store: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for k, v in store.items():
        n = max(int(v['count']), 1)
        out[k] = {
            'count': int(v['count']),
            'rot': float(v['rot_sum'] / n),
            'tdir': float(v['tdir_sum'] / n),
            'tdir_abs': float(v['tdir_abs_sum'] / n),
            'tdir_local_A': float(v['tdir_local_A_sum'] / n),
            'tdir_local_A_abs': float(v['tdir_local_A_abs_sum'] / n),
            'epi_mass_in_gt_band': float(v['epi_mass_in_gt_band_sum'] / n),
            'top1_in_gt_band': float(v['top1_in_gt_band_sum'] / n),
            'top5_in_gt_band': float(v['top5_in_gt_band_sum'] / n),
            'matching_entropy': float(v['matching_entropy_sum'] / n),
            'max_matching_prob': float(v['max_matching_prob_sum'] / n),
            'cycle_error': float(v['cycle_error_sum'] / n),
            'tmag_rel_err': (
                float(v['tmag_rel_err_sum'] / max(int(v.get('tmag_rel_err_count', 0)), 1))
                if int(v.get('tmag_rel_err_count', 0)) > 0 else float('nan')
            ),
        }
    return out


def _bucket_weighted_mean(bucket: Dict[str, Dict[str, float]], labels, metric: str) -> Tuple[float, int]:
    total = 0.0
    count = 0
    for label in labels:
        rec = bucket.get(str(label), {})
        n = int(rec.get("count", 0))
        val = float(rec.get(metric, float("nan")))
        if n > 0 and math.isfinite(val):
            total += val * n
            count += n
    if count <= 0:
        return float("nan"), 0
    return float(total / count), int(count)


def _bucket_sort_key(label: str):
    if label.startswith('k='):
        try:
            return (0, float(label.split('=')[1]))
        except Exception:
            return (0, 1e9)
    if label.startswith('dt<'):
        try:
            return (1, float(label.split('<')[1]))
        except Exception:
            return (1, 1e9)
    if '<=dt<' in label:
        try:
            lo = float(label.split('<=dt<')[0])
            return (1, lo)
        except Exception:
            return (1, 1e9)
    if label.startswith('dt>='):
        try:
            return (1, float(label.split('>=')[1]))
        except Exception:
            return (1, 1e9)
    return (9, label)


def _format_bucket_summary(prefix: str, bucket: Dict[str, Dict[str, float]]) -> str:
    if not bucket:
        return f'[{prefix}] none'
    parts = []
    for label in sorted(bucket.keys(), key=_bucket_sort_key):
        rec = bucket[label]
        parts.append(
            f"{label}:n={rec['count']} rot={rec['rot']:.2f} tdir={rec['tdir']:.2f} "
            f"tdir_abs={rec['tdir_abs']:.2f} local_abs={rec['tdir_local_A_abs']:.2f} "
            f"epi_mass={rec['epi_mass_in_gt_band']:.3f} top1={rec['top1_in_gt_band']:.3f} "
            f"tmag_rel={rec.get('tmag_rel_err', float('nan')):.3f}"
        )
    return f'[{prefix}] ' + ' | '.join(parts)


def _epipolar_matching_diagnostics(
    W_ab: Optional[torch.Tensor],
    W_ba: Optional[torch.Tensor],
    bearing_a: Optional[torch.Tensor],
    bearing_b: Optional[torch.Tensor],
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    *,
    angle_thresh_deg: float,
    allowed_mask: Optional[torch.Tensor] = None,
    routing_mask: Optional[torch.Tensor] = None,
    topk: int = 5,
    min_plane_norm: float = 1e-4,
) -> Dict[str, torch.Tensor]:
    if W_ab is None or bearing_a is None or bearing_b is None:
        return {}

    W_ab = W_ab.float()
    bearing_a = F.normalize(bearing_a.float(), dim=-1, eps=1e-6)
    bearing_b = F.normalize(bearing_b.float(), dim=-1, eps=1e-6)
    R_gt = R_gt.float()
    t_gt = F.normalize(t_gt.float(), dim=-1, eps=1e-6)

    bA_in_B = torch.matmul(bearing_a, R_gt.transpose(-1, -2))
    plane_n_raw = torch.cross(t_gt[:, None, :].expand_as(bA_in_B), bA_in_B, dim=-1)
    plane_norm = torch.linalg.norm(plane_n_raw, dim=-1)
    valid_plane = plane_norm > float(min_plane_norm)
    plane_n = F.normalize(plane_n_raw, dim=-1, eps=1e-6)
    residual = torch.abs(torch.einsum("bnc,bmc->bnm", plane_n, bearing_b)).clamp(0.0, 1.0)

    band = max(math.sin(math.radians(float(angle_thresh_deg))), 1e-6)
    gt_band = residual <= band
    gt_band_unmasked = gt_band & valid_plane.unsqueeze(-1)
    if allowed_mask is not None:
        gt_band = gt_band & allowed_mask.to(dtype=torch.bool)
    gt_band = gt_band & valid_plane.unsqueeze(-1)

    row_mass = (W_ab * gt_band.to(W_ab.dtype)).sum(dim=-1)
    row_valid = gt_band.any(dim=-1).to(W_ab.dtype)
    epi_mass = (row_mass * row_valid).sum(dim=-1) / row_valid.sum(dim=-1).clamp_min(1.0)

    top1_idx = W_ab.argmax(dim=-1, keepdim=True)
    top1_hit = torch.gather(gt_band, -1, top1_idx).squeeze(-1).to(W_ab.dtype)
    top1 = (top1_hit * row_valid).sum(dim=-1) / row_valid.sum(dim=-1).clamp_min(1.0)

    k = min(int(topk), int(W_ab.shape[-1]))
    topk_idx = torch.topk(W_ab, k=k, dim=-1).indices
    topk_hit = torch.gather(gt_band, -1, topk_idx).any(dim=-1).to(W_ab.dtype)
    top5 = (topk_hit * row_valid).sum(dim=-1) / row_valid.sum(dim=-1).clamp_min(1.0)

    entropy_rows = -(W_ab.clamp_min(1e-9) * W_ab.clamp_min(1e-9).log()).sum(dim=-1)
    entropy = entropy_rows.mean(dim=-1)
    max_prob = W_ab.max(dim=-1).values.mean(dim=-1)

    if W_ba is not None:
        cyc = torch.matmul(W_ab, W_ba.float())
        I = torch.eye(cyc.shape[-1], device=cyc.device, dtype=cyc.dtype).unsqueeze(0)
        cycle_error = ((cyc - I) ** 2).mean(dim=(-2, -1))
    else:
        cycle_error = torch.full((W_ab.shape[0],), float("nan"), device=W_ab.device, dtype=W_ab.dtype)

    out = {
        "epi_mass_in_gt_band": epi_mass.detach(),
        "top1_in_gt_band": top1.detach(),
        "top5_in_gt_band": top5.detach(),
        "matching_entropy": entropy.detach(),
        "max_matching_prob": max_prob.detach(),
        "cycle_error": cycle_error.detach(),
    }
    if allowed_mask is not None:
        allowed = allowed_mask.to(dtype=torch.bool)
        out["allowed_mask_density"] = allowed.to(W_ab.dtype).mean(dim=(-2, -1)).detach()
        out["no_candidate_row_ratio"] = (~allowed.any(dim=-1)).to(W_ab.dtype).mean(dim=-1).detach()
    if routing_mask is not None:
        routing = routing_mask.to(dtype=torch.bool)
        row_has_gt = gt_band_unmasked.any(dim=-1).to(W_ab.dtype)
        row_recalled = (routing & gt_band_unmasked).any(dim=-1).to(W_ab.dtype)
        out["routing_recall_in_gt_band"] = (
            (row_recalled * row_has_gt).sum(dim=-1) / row_has_gt.sum(dim=-1).clamp_min(1.0)
        ).detach()
    return out


def _save_ckpt(path, model, optimizer, scaler, scheduler, cfg, step, upd, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "cfg": _cfg_to_dict(cfg),
            "step": int(step),
            "upd": int(upd),
            "metrics": dict(metrics),
        },
        path,
    )


def _save_model_ckpt(path, model, cfg, step, upd, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": _cfg_to_dict(cfg),
            "step": int(step),
            "upd": int(upd),
            "metrics": dict(metrics),
        },
        path,
    )


def _load_model_init_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device,
    *,
    strict: bool = False,
    label: str = "InitCkpt",
) -> None:
    ckpt_path = os.path.expanduser(str(path))
    if not ckpt_path:
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"{label} not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"{label} has no model state_dict: {ckpt_path}")
    missing, unexpected = model.load_state_dict(state, strict=bool(strict))
    print(
        f"[{label}] loaded {ckpt_path} | strict={bool(strict)} | "
        f"missing={len(missing)} | unexpected={len(unexpected)}"
    )
    if missing:
        print(f"[{label}] missing preview={list(missing)[:8]}")
    if unexpected:
        print(f"[{label}] unexpected preview={list(unexpected)[:8]}")


def _first_item(v: Any) -> Any:
    if torch.is_tensor(v):
        return v[0]
    if isinstance(v, np.ndarray):
        return v[0]
    if isinstance(v, (list, tuple)):
        return v[0]
    return v


def _meta_first(meta: Any) -> Any:
    if isinstance(meta, dict):
        return {k: _meta_first(v) for k, v in meta.items()}
    return _first_item(meta)


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _crop_mat(mat: Optional[np.ndarray], max_side: int) -> Optional[np.ndarray]:
    if mat is None:
        return None
    if mat.ndim < 2:
        return mat
    h = min(max_side, mat.shape[-2])
    w = min(max_side, mat.shape[-1])
    return mat[:h, :w]



def _downsample_to_like(img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if img.shape[-2:] == ref.shape[-2:]:
        return img
    return F.interpolate(img, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def _depth_weight_ramp(upd: int, cfg: Config) -> float:
    start = int(getattr(cfg, "depth_warmup_updates", 0))
    ramp = max(int(getattr(cfg, "depth_photo_ramp_updates", 1)), 1)
    if upd < start:
        return 0.0
    return float(min(1.0, (upd - start + 1) / float(ramp)))




def _to_hwc_img01(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if x is None:
        return None
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim != 3:
        return None
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, 0.0, 1.0)


def _robust_vis_map(x: Optional[np.ndarray], mask: Optional[np.ndarray] = None, q_low: float = 5.0, q_high: float = 95.0) -> Optional[np.ndarray]:
    if x is None:
        return None
    x = np.asarray(x).astype(np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 2:
        return None
    valid = np.isfinite(x)
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        if m.ndim == 3 and m.shape[-1] == 1:
            m = m[..., 0]
        if m.shape == x.shape:
            valid = valid & (m > 0.5)
    vals = x[valid]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.percentile(vals, q_low))
    hi = float(np.percentile(vals, q_high))
    if not np.isfinite(lo):
        lo = float(vals.min())
    if not np.isfinite(hi):
        hi = float(vals.max())
    if hi <= lo:
        hi = lo + 1e-6
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    y[~np.isfinite(y)] = 0.0
    return y.astype(np.float32)


def _payload_diag(payload: Dict[str, Any]) -> Dict[str, Any]:
    diag: Dict[str, Any] = {}
    inv_depth = payload.get("inv_depth_s16")
    valid = payload.get("warp_valid_s")
    photo_err = payload.get("photo_err_s")
    Wf = payload.get("Wf_ab")
    Wf_raw = payload.get("Wf_ab_raw")
    routing = payload.get("routing_mask")
    allowed = payload.get("allowed_mask")
    epi_residual = payload.get("epi_residual")

    if inv_depth is not None:
        d = np.asarray(inv_depth, dtype=np.float32)
        vals = d[np.isfinite(d)]
        if vals.size > 0:
            diag["inv_depth_min"] = float(vals.min())
            diag["inv_depth_p05"] = float(np.percentile(vals, 5))
            diag["inv_depth_p50"] = float(np.percentile(vals, 50))
            diag["inv_depth_p95"] = float(np.percentile(vals, 95))
            diag["inv_depth_max"] = float(vals.max())

    if valid is not None:
        v = np.asarray(valid, dtype=np.float32)
        diag["warp_valid_ratio"] = float(v.mean())

    if photo_err is not None:
        e = np.asarray(photo_err, dtype=np.float32)
        if valid is not None:
            vm = np.asarray(valid, dtype=np.float32) > 0.5
            vals = e[vm]
        else:
            vals = e[np.isfinite(e)]
        if vals.size > 0:
            diag["photo_err_mean_valid"] = float(vals.mean())
            diag["photo_err_p90_valid"] = float(np.percentile(vals, 90))

    if routing is not None:
        diag["routing_keep_ratio"] = float(np.asarray(routing, dtype=np.float32).mean())
    if allowed is not None:
        diag["allowed_ratio"] = float(np.asarray(allowed, dtype=np.float32).mean())

    if Wf is not None:
        W = np.asarray(Wf, dtype=np.float32)
        W = np.clip(W, 1e-9, 1.0)
        ent = -(W * np.log(W)).sum(axis=-1)
        diag["softcorr_row_entropy_mean"] = float(ent.mean())
        diag["softcorr_row_entropy_std"] = float(ent.std())
    if Wf is not None and epi_residual is not None:
        W = np.asarray(Wf, dtype=np.float32)
        R = np.asarray(epi_residual, dtype=np.float32)
        denom = float(np.clip(W.sum(), 1e-9, None))
        diag["epi_residual_weighted_mean"] = float((W * R).sum() / denom)
    if Wf is not None and Wf_raw is not None:
        W = np.asarray(Wf, dtype=np.float32)
        Wr = np.asarray(Wf_raw, dtype=np.float32)
        diag["guided_minus_raw_l1"] = float(np.mean(np.abs(W - Wr)))
    return diag


def _plot_depth_overview(payload: Dict[str, Any], path: str) -> None:
    IA_s = _to_hwc_img01(payload.get("IA_s"))
    IB_s = _to_hwc_img01(payload.get("IB_s"))
    Iwarp_s = _to_hwc_img01(payload.get("Iwarp_s"))
    valid = payload.get("warp_valid_s")
    inv_depth = payload.get("inv_depth_s16")
    photo_err = payload.get("photo_err_s")
    if IA_s is None or inv_depth is None:
        return

    valid2 = None
    if valid is not None:
        v = np.asarray(valid)
        if v.ndim == 3 and v.shape[0] == 1:
            v = v[0]
        if v.ndim == 3 and v.shape[-1] == 1:
            v = v[..., 0]
        valid2 = v.astype(np.float32)

    depth_vis = _robust_vis_map(inv_depth, None)
    err_vis = _robust_vis_map(photo_err, valid2 if valid2 is not None else None, q_low=0.0, q_high=95.0)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    axs = axes.ravel()

    axs[0].imshow(IA_s)
    axs[0].set_title("Reference IA_s")
    axs[0].axis("off")

    if Iwarp_s is not None:
        axs[1].imshow(Iwarp_s)
        axs[1].set_title("Warped from IB")
    else:
        axs[1].text(0.5, 0.5, "Iwarp unavailable", ha="center", va="center")
    axs[1].axis("off")

    if err_vis is not None:
        im2 = axs[2].imshow(err_vis, cmap="magma", vmin=0.0, vmax=1.0)
        axs[2].set_title("Photometric error")
        plt.colorbar(im2, ax=axs[2], fraction=0.046)
    else:
        axs[2].text(0.5, 0.5, "photo_err unavailable", ha="center", va="center")
    axs[2].axis("off")

    if depth_vis is not None:
        im3 = axs[3].imshow(depth_vis, cmap="inferno", vmin=0.0, vmax=1.0)
        axs[3].set_title("Inverse depth s16")
        plt.colorbar(im3, ax=axs[3], fraction=0.046)
    else:
        axs[3].text(0.5, 0.5, "inv_depth unavailable", ha="center", va="center")
    axs[3].axis("off")

    if valid2 is not None:
        im4 = axs[4].imshow(valid2, cmap="gray", vmin=0.0, vmax=1.0)
        axs[4].set_title("Warp valid mask")
        plt.colorbar(im4, ax=axs[4], fraction=0.046)
    elif IB_s is not None:
        axs[4].imshow(IB_s)
        axs[4].set_title("Target IB_s")
    else:
        axs[4].text(0.5, 0.5, "valid/IB unavailable", ha="center", va="center")
    axs[4].axis("off")

    diag = payload.get("diag", {})
    diag_lines = [f"{k}: {v:.4f}" if isinstance(v, (int, float, np.floating)) else f"{k}: {v}" for k, v in diag.items()]
    axs[5].axis("off")
    axs[5].text(0.01, 0.99, "\n".join(diag_lines[:14]) if diag_lines else "No diagnostics", va="top", ha="left", fontsize=9, family="monospace")

    meta = payload.get("meta", {})
    fig.suptitle(f"Depth visualization | {meta}", fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)

def _extract_vis_payload(batch: Dict[str, Any], aux: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    meta = _meta_first(batch.get("meta", {}))
    payload: Dict[str, Any] = {
        "meta": meta,
        "IA": _to_numpy(_first_item(batch.get("IA"))),
        "IB": _to_numpy(_first_item(batch.get("IB"))),
        "IA_s": _to_numpy(_first_item(aux.get("IA_s"))),
        "IB_s": _to_numpy(_first_item(aux.get("IB_s"))),
        "Iwarp_s": _to_numpy(_first_item(aux.get("Iwarp_s"))),
        "warp_valid_s": _to_numpy(_first_item(aux.get("warp_valid_s"))),
        "photo_err_s": _to_numpy(_first_item(aux.get("photo_err_s"))),
        "R_gt": _to_numpy(_first_item(batch.get("R_gt"))),
        "t_gt_dir": _to_numpy(_first_item(batch.get("t_gt_dir"))),
        "Wf_ab_raw": _to_numpy(_first_item(aux.get("Wf_ab_raw"))),
        "Wf_ab": _to_numpy(_first_item(aux.get("Wf_ab"))),
        "routing_mask": _to_numpy(_first_item(aux.get("routing_mask"))),
        "allowed_mask": _to_numpy(_first_item(aux.get("allowed_mask"))),
        "epi_bias": _to_numpy(_first_item(aux.get("epi_bias"))),
        "epi_residual": _to_numpy(_first_item(aux.get("epi_residual"))),
        "epi_cost": _to_numpy(_first_item(aux.get("epi_cost"))),
        "logits_f": _to_numpy(_first_item(aux.get("logits_f"))),
        "logits_f_biased": _to_numpy(_first_item(aux.get("logits_f_biased"))),
        "stage": aux.get("stage", "-"),
        "inv_depth_s16": _to_numpy(_first_item(aux.get("inv_depth_s16"))),
    }
    payload["Wf_ab_raw_crop"] = _crop_mat(payload["Wf_ab_raw"], cfg.vis_plot_max_side)
    payload["Wf_ab_crop"] = _crop_mat(payload["Wf_ab"], cfg.vis_plot_max_side)
    payload["routing_mask_crop"] = _crop_mat(payload["routing_mask"], cfg.vis_plot_max_side)
    payload["allowed_mask_crop"] = _crop_mat(payload["allowed_mask"], cfg.vis_plot_max_side)
    payload["epi_bias_crop"] = _crop_mat(payload["epi_bias"], cfg.vis_plot_max_side)
    payload["epi_residual_crop"] = _crop_mat(payload["epi_residual"], cfg.vis_plot_max_side)
    payload["diag"] = _payload_diag(payload)
    return payload

def _save_vis_payload_npz(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat = {}
    for k, v in payload.items():
        if isinstance(v, dict):
            flat[k] = np.array(json.dumps(v, ensure_ascii=False, default=_json_default), dtype=object)
        elif v is None:
            flat[k] = np.array([], dtype=np.float32)
        elif isinstance(v, str):
            flat[k] = np.array(v, dtype=object)
        else:
            flat[k] = np.asarray(v)
    np.savez_compressed(path, **flat)


def _plot_softcorr_overview(payload: Dict[str, Any], path: str) -> None:
    raw = payload.get("Wf_ab_raw_crop")
    guided = payload.get("Wf_ab_crop")
    allowed = payload.get("allowed_mask_crop")
    residual = payload.get("epi_residual_crop")
    bias = payload.get("epi_bias_crop")
    if raw is None or guided is None:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    axs = axes.ravel()
    im0 = axs[0].imshow(raw, aspect="auto", cmap="viridis")
    axs[0].set_title("Before geometry\nWf_ab_raw")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(guided, aspect="auto", cmap="viridis")
    axs[1].set_title("After geometry\nWf_ab")
    plt.colorbar(im1, ax=axs[1], fraction=0.046)

    if allowed is not None:
        im2 = axs[2].imshow(allowed.astype(float), aspect="auto", cmap="gray_r", vmin=0.0, vmax=1.0)
        axs[2].set_title("Allowed mask")
        plt.colorbar(im2, ax=axs[2], fraction=0.046)
    else:
        axs[2].axis("off")

    if residual is not None:
        im3 = axs[3].imshow(residual, aspect="auto", cmap="magma")
        axs[3].set_title("Epipolar residual")
        plt.colorbar(im3, ax=axs[3], fraction=0.046)
    else:
        axs[3].axis("off")

    if bias is not None:
        im4 = axs[4].imshow(bias, aspect="auto", cmap="coolwarm")
        axs[4].set_title("Epipolar bias")
        plt.colorbar(im4, ax=axs[4], fraction=0.046)
    else:
        axs[4].axis("off")

    if raw is not None and guided is not None:
        diff = guided - raw
        im5 = axs[5].imshow(diff, aspect="auto", cmap="coolwarm")
        axs[5].set_title("Guided - raw")
        plt.colorbar(im5, ax=axs[5], fraction=0.046)
    else:
        axs[5].axis("off")

    title_meta = payload.get("meta", {})
    fig.suptitle(f"Soft correspondence visualization | {title_meta}", fontsize=12)
    for ax in axs:
        ax.set_xlabel("B tokens")
        ax.set_ylabel("A tokens")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_eval_curves(history: List[Dict[str, Any]], path: str) -> None:
    if not history:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    upd = [h["upd"] for h in history]
    tdir_abs = [h["tdir_abs"] for h in history]
    joint = [h["joint_score"] for h in history]
    rot = [h["rot"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax1.plot(upd, tdir_abs, marker="o", linewidth=1.6, label="tdir_abs")
    ax1.plot(upd, rot, marker="s", linewidth=1.3, label="rot")
    ax1.set_xlabel("Update")
    ax1.set_ylabel("Error (deg)")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(upd, joint, marker="^", linewidth=1.3, label="joint_score")
    ax2.set_ylabel("Joint score")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.suptitle("Validation curves")
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def eval_model(model, loader, device, cfg: Config, *, collect_vis: bool = False, vis_index: int = 0):
    model.eval()

    rot_sum = 0.0
    tdir_sum = 0.0
    tdir_abs_sum = 0.0
    tdir_local_A_sum = 0.0
    tdir_local_A_abs_sum = 0.0
    t_raw_local_A_abs_sum = 0.0
    t_geo_local_A_abs_sum = 0.0
    tdir_cos_sum = 0.0
    tdir_flip_count = 0.0
    tdir_local_A_cos_sum = 0.0
    tdir_local_A_flip_count = 0.0
    tmag_abs_sum = 0.0
    tmag_rel_sum = 0.0
    n_tmag = 0
    trans_vec_l2_sum = 0.0
    n_trans_vec = 0
    geom_refine_success_sum = 0.0
    geom_refine_attempts = 0
    geom_rot_sum = 0.0
    geom_tdir_sum = 0.0
    geom_tdir_abs_sum = 0.0
    fused_rot_sum = 0.0
    fused_tdir_abs_sum = 0.0
    n_geom = 0
    stable_rot_sum = 0.0
    stable_tdir_sum = 0.0
    stable_tdir_abs_sum = 0.0
    stable_tdir_local_A_abs_sum = 0.0
    stable_n = 0
    diag_sums = {
        "epi_mass_in_gt_band": 0.0,
        "top1_in_gt_band": 0.0,
        "top5_in_gt_band": 0.0,
        "matching_entropy": 0.0,
        "max_matching_prob": 0.0,
        "cycle_error": 0.0,
    }
    extra_diag_sums: Dict[str, float] = {}
    extra_diag_counts: Dict[str, int] = {}

    def _accum_extra_diag(name: str, value: torch.Tensor) -> None:
        arr = value.detach().float().cpu().view(-1).numpy()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        extra_diag_sums[name] = extra_diag_sums.get(name, 0.0) + float(arr.sum())
        extra_diag_counts[name] = extra_diag_counts.get(name, 0) + int(arr.size)

    def _accum_matching_diag(prefix: str, diag: Dict[str, torch.Tensor]) -> None:
        mapping = {
            "epi_mass_in_gt_band": f"{prefix}_epi_mass",
            "top1_in_gt_band": f"{prefix}_top1",
            "top5_in_gt_band": f"{prefix}_top5",
            "matching_entropy": f"{prefix}_entropy",
            "max_matching_prob": f"{prefix}_max_prob",
            "cycle_error": f"{prefix}_cycle_error",
        }
        for src, dst in mapping.items():
            if src in diag:
                _accum_extra_diag(dst, diag[src])
        if prefix == "fine":
            for key in ["routing_recall_in_gt_band", "allowed_mask_density", "no_candidate_row_ratio"]:
                if key in diag:
                    _accum_extra_diag(key, diag[key])
                    _accum_extra_diag(f"fine_{key}", diag[key])
    n = 0
    n_local = 0
    n_t_raw = 0
    n_t_geo = 0
    vis_payload = None

    bucket_k = _bucket_init()
    bucket_dt = _bucket_init()
    bucket_k_dt = _bucket_init()
    dt_edges = tuple(float(x) for x in getattr(cfg, 'eval_dt_bucket_edges', (0.2, 0.5, 1.0, 2.0)))
    stable_min_dt = float(getattr(cfg, "stable_eval_min_dt", 0.5))

    acc = {
        'raw': 0.0,
        'flip': 0.0,
        'R@t': 0.0,
        'R@(-t)': 0.0,
        'Rt@t': 0.0,
        'Rt@(-t)': 0.0,
        'cnt': 0,
    }
    variant_desc = {
        'raw': 'output frame B, baseline B->A',
        'flip': 'output frame B, baseline A->B',
        'R@t': 'compare to R*t_gt (algebraic diagnostic only)',
        'R@(-t)': 'compare to -R*t_gt',
        'Rt@t': 'compare to R^T*t_gt (B->A mapped to A-local)',
        'Rt@(-t)': 'compare to -R^T*t_gt',
    }

    for bi, batch in enumerate(loader):
        if cfg.max_eval_batches and bi >= cfg.max_eval_batches:
            break

        IA = batch['IA'].to(device, non_blocking=True)
        IB = batch['IB'].to(device, non_blocking=True)
        R_gt = batch['R_gt'].to(device, non_blocking=True)
        t_gt = batch['t_gt_dir'].to(device, non_blocking=True)
        meta = batch.get('meta', {})
        t_gt_mag = _translation_magnitude_gt(batch, meta, IA.shape[0], device)

        R_pred, t_pred, aux = model(IA, IB, enable_depth_fusion=True)
        t_eval_pred = aux.get('t_dir_out', t_pred) if isinstance(aux, dict) else t_pred

        use_fine_diag = bool(cfg.use_fine_stage) and (aux.get("Wf_ab", None) is not None)
        if use_fine_diag:
            W_eval = aux.get("Wf_ab", None)
            W_eval_ba = aux.get("Wf_ba", None)
            bearingA_eval = aux.get("bearingA_f", None)
            bearingB_eval = aux.get("bearingB_f", None)
            allowed_eval = aux.get("allowed_mask", None)
        else:
            W_eval = aux.get("Wc_ab", None)
            W_eval_ba = aux.get("Wc_ba", None)
            bearingA_eval = aux.get("bearingA_c", None)
            bearingB_eval = aux.get("bearingB_c", None)
            allowed_eval = None

        diag_batch = _epipolar_matching_diagnostics(
            W_eval,
            W_eval_ba,
            bearingA_eval,
            bearingB_eval,
            R_gt,
            t_gt,
            angle_thresh_deg=float(cfg.epi_angle_thresh_deg),
            allowed_mask=allowed_eval,
            topk=5,
        )
        coarse_diag_batch = _epipolar_matching_diagnostics(
            aux.get("Wc_ab", None),
            aux.get("Wc_ba", None),
            aux.get("bearingA_c", None),
            aux.get("bearingB_c", None),
            R_gt,
            t_gt,
            angle_thresh_deg=float(cfg.epi_angle_thresh_deg),
            allowed_mask=None,
            topk=5,
        )
        _accum_matching_diag("coarse", coarse_diag_batch)
        if aux.get("Wf_ab", None) is not None:
            fine_diag_batch = _epipolar_matching_diagnostics(
                aux.get("Wf_ab", None),
                aux.get("Wf_ba", None),
                aux.get("bearingA_f", None),
                aux.get("bearingB_f", None),
                R_gt,
                t_gt,
                angle_thresh_deg=float(cfg.epi_angle_thresh_deg),
                allowed_mask=aux.get("allowed_mask", None),
                routing_mask=aux.get("routing_mask", None),
                topk=5,
            )
            _accum_matching_diag("fine", fine_diag_batch)

        if collect_vis and vis_payload is None and bi == int(vis_index):
            depth_key = f"inv_depth_s{int(cfg.depth_loss_scale)}"
            if depth_key in aux and aux.get(depth_key) is not None:
                inv_depth = aux[depth_key]
                IA_small = _downsample_to_like(IA, inv_depth)
                IB_small = _downsample_to_like(IB, inv_depth)
                t_warp = aux.get('t_dir_out', aux.get('t_dir', t_pred))
                Iwarp, valid = warp_erp_with_depth_pose(
                    inv_depth,
                    R_pred,
                    t_warp,
                    IB_small,
                    translation_scale=1.0,
                    min_depth=cfg.depth_min,
                    max_depth=cfg.depth_max,
                )
                photo_err = (IA_small - Iwarp).abs().mean(dim=1, keepdim=True)
                aux['IA_s'] = IA_small.detach()
                aux['IB_s'] = IB_small.detach()
                aux['Iwarp_s'] = Iwarp.detach()
                aux['warp_valid_s'] = valid.detach()
                aux['photo_err_s'] = (photo_err * valid.float()).detach()
            vis_payload = _extract_vis_payload(batch, aux, cfg)

        rot_rad = matrix_geodesic_distance(R_pred.float(), R_gt.float())
        rot_deg = rot_rad * (180.0 / math.pi)

        tp = F.normalize(t_eval_pred.float(), dim=-1, eps=1e-6)
        tg = F.normalize(t_gt.float(), dim=-1, eps=1e-6)
        cos = torch.sum(tp * tg, dim=-1).clamp(-1.0, 1.0)
        ang = torch.acos(cos) * (180.0 / math.pi)
        ang_abs = torch.minimum(ang, 180.0 - ang)

        bsz = IA.shape[0]
        rot_sum += float(rot_deg.sum().cpu())
        tdir_sum += float(ang.sum().cpu())
        tdir_abs_sum += float(ang_abs.sum().cpu())
        tdir_cos_sum += float(cos.sum().cpu())
        tdir_flip_count += float((cos < 0.0).to(torch.float32).sum().cpu())
        n += bsz

        if bool(getattr(cfg, "use_geometry_refine", False)):
            use_refine_fine = (
                bool(getattr(cfg, "geom_refine_use_fine_if_available", True))
                and aux.get("Wf_ab", None) is not None
            )
            if use_refine_fine:
                W_ref = aux.get("Wf_ab", None)
                W_ref_ba = aux.get("Wf_ba", None)
                bearingA_ref = aux.get("bearingA_f", None)
                bearingB_ref = aux.get("bearingB_f", None)
                allowed_ref = aux.get("allowed_mask", None)
            else:
                W_ref = aux.get("Wc_ab", None)
                W_ref_ba = aux.get("Wc_ba", None)
                bearingA_ref = aux.get("bearingA_c", None)
                bearingB_ref = aux.get("bearingB_c", None)
                allowed_ref = None
            if W_ref is not None and bearingA_ref is not None and bearingB_ref is not None:
                R_geom, t_geom, geom_diag = refine_pose_from_matches(
                    W_ref,
                    bearingA_ref,
                    bearingB_ref,
                    R_pred.float(),
                    tp.float(),
                    W_ba=W_ref_ba,
                    allowed_mask=allowed_ref,
                    min_prob=float(getattr(cfg, "geom_refine_min_prob", 0.01)),
                    max_matches=int(getattr(cfg, "geom_refine_max_matches", 512)),
                    mutual_check=bool(getattr(cfg, "geom_refine_mutual_check", False)),
                )
                success_mask = torch.tensor(geom_diag.get("success", []), device=device, dtype=torch.bool)
                if success_mask.numel() == bsz:
                    geom_refine_success_sum += float(success_mask.to(torch.float32).sum().cpu())
                    geom_refine_attempts += bsz
                    if bool(getattr(cfg, "geom_refine_fallback_to_network", True)):
                        R_fused = torch.where(success_mask.view(-1, 1, 1), R_geom.float(), R_pred.float())
                        t_fused = torch.where(success_mask.view(-1, 1), t_geom.float(), tp.float())
                    else:
                        R_fused = R_geom.float()
                        t_fused = t_geom.float()
                    geom_rot = matrix_geodesic_distance(R_geom.float(), R_gt.float()) * (180.0 / math.pi)
                    geom_cos = torch.sum(F.normalize(t_geom.float(), dim=-1, eps=1e-6) * tg, dim=-1).clamp(-1.0, 1.0)
                    geom_ang = torch.acos(geom_cos) * (180.0 / math.pi)
                    geom_ang_abs = torch.minimum(geom_ang, 180.0 - geom_ang)
                    fused_rot = matrix_geodesic_distance(R_fused.float(), R_gt.float()) * (180.0 / math.pi)
                    fused_cos = torch.sum(F.normalize(t_fused.float(), dim=-1, eps=1e-6) * tg, dim=-1).clamp(-1.0, 1.0)
                    fused_ang = torch.acos(fused_cos) * (180.0 / math.pi)
                    fused_ang_abs = torch.minimum(fused_ang, 180.0 - fused_ang)
                    geom_rot_sum += float(geom_rot.sum().cpu())
                    geom_tdir_sum += float(geom_ang.sum().cpu())
                    geom_tdir_abs_sum += float(geom_ang_abs.sum().cpu())
                    fused_rot_sum += float(fused_rot.sum().cpu())
                    fused_tdir_abs_sum += float(fused_ang_abs.sum().cpu())
                    n_geom += bsz

        t_mag_pred = aux.get("t_mag", None) if isinstance(aux, dict) else None
        tmag_rel_np = np.full((bsz,), np.nan, dtype=np.float32)
        if t_mag_pred is not None and t_gt_mag is not None:
            mag_pred = t_mag_pred.float().view(-1).clamp_min(1e-6)
            mag_gt = t_gt_mag.float().view(-1).clamp_min(1e-6)
            mag_abs = torch.abs(mag_pred - mag_gt)
            mag_rel = mag_abs / mag_gt
            tmag_abs_sum += float(mag_abs.sum().cpu())
            tmag_rel_sum += float(mag_rel.sum().cpu())
            n_tmag += int(mag_gt.numel())
            tmag_rel_np = mag_rel.detach().cpu().view(-1).numpy()

            t_vec_pred = aux.get("t_vec_out", aux.get("t_vec", None)) if isinstance(aux, dict) else None
            if t_vec_pred is not None:
                gt_vec = tg * mag_gt.view(-1, 1)
                vec_l2 = torch.linalg.norm(t_vec_pred.float() - gt_vec.float(), dim=-1)
                trans_vec_l2_sum += float(vec_l2.sum().cpu())
                n_trans_vec += int(vec_l2.numel())

        rot_np = rot_deg.detach().cpu().view(-1).numpy()
        tdir_np = ang.detach().cpu().view(-1).numpy()
        tdir_abs_np = ang_abs.detach().cpu().view(-1).numpy()
        diag_np = {}
        for key in diag_sums.keys():
            if key in diag_batch:
                arr = diag_batch[key].detach().cpu().view(-1).numpy()
                diag_np[key] = arr
                diag_sums[key] += float(arr.sum())
            else:
                diag_np[key] = np.full((bsz,), np.nan, dtype=np.float32)

        tg_R = F.normalize(torch.matmul(R_gt.float(), tg.unsqueeze(-1)).squeeze(-1), dim=-1, eps=1e-6)
        tg_Rt = F.normalize(torch.matmul(R_gt.float().transpose(-1, -2), tg.unsqueeze(-1)).squeeze(-1), dim=-1, eps=1e-6)

        t_local_pred = aux.get('t_dir_local', None) if isinstance(aux, dict) else None
        t_local_frame = aux.get('t_local_frame', None) if isinstance(aux, dict) else None
        if t_local_pred is not None and t_local_frame == 'A':
            tp_local = F.normalize(t_local_pred.float(), dim=-1, eps=1e-6)
            cos_local = torch.sum(tp_local * tg_Rt, dim=-1).clamp(-1.0, 1.0)
            ang_local = torch.acos(cos_local) * (180.0 / math.pi)
            ang_local_abs = torch.minimum(ang_local, 180.0 - ang_local)
            tdir_local_A_sum += float(ang_local.sum().cpu())
            tdir_local_A_abs_sum += float(ang_local_abs.sum().cpu())
            tdir_local_A_cos_sum += float(cos_local.sum().cpu())
            tdir_local_A_flip_count += float((cos_local < 0.0).to(torch.float32).sum().cpu())
            n_local += bsz
            tdir_local_np = ang_local.detach().cpu().view(-1).numpy()
            tdir_local_abs_np = ang_local_abs.detach().cpu().view(-1).numpy()
        else:
            tdir_local_np = np.full((bsz,), np.nan, dtype=np.float32)
            tdir_local_abs_np = np.full((bsz,), np.nan, dtype=np.float32)

        t_raw_local = aux.get("t_dir_raw", None) if isinstance(aux, dict) else None
        if t_raw_local is not None:
            traw = F.normalize(t_raw_local.float(), dim=-1, eps=1e-6)
            ang_raw_local = torch.acos(torch.sum(traw * tg_Rt, dim=-1).clamp(-1.0, 1.0)) * (180.0 / math.pi)
            ang_raw_local_abs = torch.minimum(ang_raw_local, 180.0 - ang_raw_local)
            t_raw_local_A_abs_sum += float(ang_raw_local_abs.sum().cpu())
            n_t_raw += bsz

        t_geo_local = aux.get("t_geo_local", None) if isinstance(aux, dict) else None
        if t_geo_local is not None:
            tgeo = F.normalize(t_geo_local.float(), dim=-1, eps=1e-6)
            ang_geo_local = torch.acos(torch.sum(tgeo * tg_Rt, dim=-1).clamp(-1.0, 1.0)) * (180.0 / math.pi)
            ang_geo_local_abs = torch.minimum(ang_geo_local, 180.0 - ang_geo_local)
            t_geo_local_A_abs_sum += float(ang_geo_local_abs.sum().cpu())
            n_t_geo += bsz

        meta_k = _meta_batch_field(meta, 'k', bsz, default=None)
        meta_dt = _meta_batch_field(meta, 'dt_world', bsz, default=None)
        for i in range(bsz):
            try:
                is_stable_dt = (meta_dt[i] is not None) and (float(meta_dt[i]) >= stable_min_dt)
            except Exception:
                is_stable_dt = False
            if is_stable_dt:
                stable_rot_sum += float(rot_np[i])
                stable_tdir_sum += float(tdir_np[i])
                stable_tdir_abs_sum += float(tdir_abs_np[i])
                if np.isfinite(tdir_local_abs_np[i]):
                    stable_tdir_local_A_abs_sum += float(tdir_local_abs_np[i])
                stable_n += 1
            k_label = f"k={int(meta_k[i])}" if meta_k[i] is not None else 'k=unknown'
            dt_label = _dt_bucket_label(meta_dt[i], dt_edges)
            _bucket_update(
                bucket_k, k_label, rot_np[i], tdir_np[i], tdir_abs_np[i], tdir_local_np[i], tdir_local_abs_np[i],
                diag_np["epi_mass_in_gt_band"][i], diag_np["top1_in_gt_band"][i], diag_np["top5_in_gt_band"][i],
                diag_np["matching_entropy"][i], diag_np["max_matching_prob"][i], diag_np["cycle_error"][i],
                tmag_rel_np[i],
            )
            _bucket_update(
                bucket_dt, dt_label, rot_np[i], tdir_np[i], tdir_abs_np[i], tdir_local_np[i], tdir_local_abs_np[i],
                diag_np["epi_mass_in_gt_band"][i], diag_np["top1_in_gt_band"][i], diag_np["top5_in_gt_band"][i],
                diag_np["matching_entropy"][i], diag_np["max_matching_prob"][i], diag_np["cycle_error"][i],
                tmag_rel_np[i],
            )
            _bucket_update(
                bucket_k_dt, f"{k_label}|{dt_label}", rot_np[i], tdir_np[i], tdir_abs_np[i], tdir_local_np[i], tdir_local_abs_np[i],
                diag_np["epi_mass_in_gt_band"][i], diag_np["top1_in_gt_band"][i], diag_np["top5_in_gt_band"][i],
                diag_np["matching_entropy"][i], diag_np["max_matching_prob"][i], diag_np["cycle_error"][i],
                tmag_rel_np[i],
            )

        def ang_deg(a, b):
            c = torch.sum(a * b, dim=-1).clamp(-1.0, 1.0)
            return torch.acos(c) * (180.0 / math.pi)

        acc['raw'] += float(ang_deg(tp, tg).sum().cpu())
        acc['flip'] += float(ang_deg(tp, -tg).sum().cpu())
        acc['R@t'] += float(ang_deg(tp, tg_R).sum().cpu())
        acc['R@(-t)'] += float(ang_deg(tp, -tg_R).sum().cpu())
        acc['Rt@t'] += float(ang_deg(tp, tg_Rt).sum().cpu())
        acc['Rt@(-t)'] += float(ang_deg(tp, -tg_Rt).sum().cpu())
        acc['cnt'] += bsz

    rot = rot_sum / max(n, 1)
    tdir = tdir_sum / max(n, 1)
    tdir_abs = tdir_abs_sum / max(n, 1)
    tdir_local_A = tdir_local_A_sum / max(n_local, 1) if n_local > 0 else float('nan')
    tdir_local_A_abs = tdir_local_A_abs_sum / max(n_local, 1) if n_local > 0 else float('nan')
    diag_mean = {k: (v / max(n, 1)) for k, v in diag_sums.items()}
    diag_mean["t_raw_local_A_abs"] = t_raw_local_A_abs_sum / max(n_t_raw, 1) if n_t_raw > 0 else float("nan")
    diag_mean["t_geo_local_A_abs"] = t_geo_local_A_abs_sum / max(n_t_geo, 1) if n_t_geo > 0 else float("nan")
    diag_mean["tdir_cos"] = tdir_cos_sum / max(n, 1)
    diag_mean["tdir_flip_rate"] = tdir_flip_count / max(n, 1)
    diag_mean["tdir_local_A_cos"] = tdir_local_A_cos_sum / max(n_local, 1) if n_local > 0 else float("nan")
    diag_mean["tdir_local_A_flip_rate"] = tdir_local_A_flip_count / max(n_local, 1) if n_local > 0 else float("nan")
    diag_mean["tmag_abs_err"] = tmag_abs_sum / max(n_tmag, 1) if n_tmag > 0 else float("nan")
    diag_mean["tmag_rel_err"] = tmag_rel_sum / max(n_tmag, 1) if n_tmag > 0 else float("nan")
    diag_mean["trans_vec_l2"] = trans_vec_l2_sum / max(n_trans_vec, 1) if n_trans_vec > 0 else float("nan")
    diag_mean["stable_n"] = float(stable_n)
    diag_mean["stable_min_dt"] = float(stable_min_dt)
    diag_mean["stable_rot"] = stable_rot_sum / max(stable_n, 1)
    diag_mean["stable_tdir"] = stable_tdir_sum / max(stable_n, 1)
    diag_mean["stable_tdir_abs"] = stable_tdir_abs_sum / max(stable_n, 1)
    diag_mean["stable_tdir_local_A_abs"] = stable_tdir_local_A_abs_sum / max(stable_n, 1)
    for key, total in extra_diag_sums.items():
        diag_mean[key] = total / max(int(extra_diag_counts.get(key, 0)), 1)
    diag_mean["geom_refine_success_rate"] = (
        geom_refine_success_sum / max(geom_refine_attempts, 1) if geom_refine_attempts > 0 else float("nan")
    )
    diag_mean["geom_rot"] = geom_rot_sum / max(n_geom, 1) if n_geom > 0 else float("nan")
    diag_mean["geom_tdir"] = geom_tdir_sum / max(n_geom, 1) if n_geom > 0 else float("nan")
    diag_mean["geom_tdir_abs"] = geom_tdir_abs_sum / max(n_geom, 1) if n_geom > 0 else float("nan")
    diag_mean["fused_rot"] = fused_rot_sum / max(n_geom, 1) if n_geom > 0 else float("nan")
    diag_mean["fused_tdir_abs"] = fused_tdir_abs_sum / max(n_geom, 1) if n_geom > 0 else float("nan")

    if acc['cnt'] > 0:
        denom = float(acc['cnt'])
        mean_map = {k: acc[k] / denom for k in ['raw', 'flip', 'R@t', 'R@(-t)', 'Rt@t', 'Rt@(-t)']}
        ranked = sorted(mean_map.items(), key=lambda kv: kv[1])
        best_key, best_val = ranked[0]
        gap2 = ranked[1][1] - ranked[0][1] if len(ranked) > 1 else float('nan')
        msg = (
            f"[TDIR-CHK] best={best_key} ({variant_desc[best_key]})={best_val:.2f}° | gap2={gap2:.2f}° | "
            f"raw={mean_map['raw']:.2f} flip={mean_map['flip']:.2f} "
            f"R@t={mean_map['R@t']:.2f} R@(-t)={mean_map['R@(-t)']:.2f} "
            f"Rt@t={mean_map['Rt@t']:.2f} Rt@(-t)={mean_map['Rt@(-t)']:.2f} (n={int(denom)})"
        )
    else:
        mean_map = {}
        msg = '[TDIR-CHK] n=0'

    if n_local > 0:
        msg_local = (
            f"[TDIR-LOCAL] local_A={tdir_local_A:.2f}° | local_A_abs={tdir_local_A_abs:.2f}° | "
            f"gt_local_A=R^T*t_gt | n={n_local}"
        )
    else:
        msg_local = '[TDIR-LOCAL] unavailable'

    bucket_k_fin = _bucket_finalize(bucket_k)
    bucket_dt_fin = _bucket_finalize(bucket_dt)
    bucket_k_dt_fin = _bucket_finalize(bucket_k_dt)
    bucket_msgs = [
        _format_bucket_summary('Eval-BKT-k', bucket_k_fin),
        _format_bucket_summary('Eval-BKT-dt', bucket_dt_fin),
        _format_bucket_summary('Eval-BKT-kdt', bucket_k_dt_fin),
    ]

    model.train()
    return (
        rot, tdir, tdir_abs, tdir_local_A, tdir_local_A_abs, {**mean_map, **diag_mean}, msg, msg_local, vis_payload,
        bucket_k_fin, bucket_dt_fin, bucket_k_dt_fin, bucket_msgs,
    )


def _rot_geodesic_deg_np(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    rel = R_pred.astype(np.float64) @ R_gt.astype(np.float64).T
    cos = (float(np.trace(rel)) - 1.0) * 0.5
    cos = max(-1.0, min(1.0, cos))
    return float(math.degrees(math.acos(cos)))


def _vec_angle_deg_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    cos = float(np.dot(a, b) / (na * nb))
    cos = max(-1.0, min(1.0, cos))
    return float(math.degrees(math.acos(cos)))


def _compose_rel_pose_np(R_rel: np.ndarray, t_rel: np.ndarray, R_cur0: np.ndarray, t_cur0: np.ndarray):
    R_next0 = R_rel.astype(np.float64) @ R_cur0.astype(np.float64)
    t_next0 = R_rel.astype(np.float64) @ t_cur0.astype(np.float64) + t_rel.astype(np.float64)
    return R_next0, t_next0


def _camera_center_from_T_c0_np(R_c0: np.ndarray, t_c0: np.ndarray) -> np.ndarray:
    return -(R_c0.astype(np.float64).T @ t_c0.astype(np.float64))


def _build_odometry_chains(manifest: List[Dict[str, Any]], selected_k: int, max_pairs: int = 0):
    groups: Dict[Any, Dict[int, Dict[str, Any]]] = {}
    for ds_idx, meta in enumerate(manifest):
        try:
            k = int(meta.get("k"))
            i = int(meta.get("i"))
        except Exception:
            continue
        if k != int(selected_k):
            continue
        key = (meta.get("scene", "unknown"), meta.get("seq", "unknown"))
        item = dict(meta)
        item["_ds_idx"] = int(ds_idx)
        groups.setdefault(key, {})[i] = item

    chains = []
    used_pairs = 0
    max_pairs = int(max_pairs)
    for key in sorted(groups.keys()):
        remaining = dict(groups[key])
        while remaining:
            cur_i = min(remaining.keys())
            chain = []
            while cur_i in remaining:
                item = remaining.pop(cur_i)
                chain.append(item)
                used_pairs += 1
                if max_pairs > 0 and used_pairs >= max_pairs:
                    break
                try:
                    cur_i = int(item["j"])
                except Exception:
                    break
            if chain:
                chains.append({"scene_seq": key, "pairs": chain})
            if max_pairs > 0 and used_pairs >= max_pairs:
                return chains
    return chains


@torch.no_grad()
def eval_odometry_sequence(model, ds, device, cfg: Config) -> Dict[str, Any]:
    if not hasattr(ds, "manifest"):
        return {"odom_status": "skipped", "odom_reason": "dataset_has_no_manifest"}

    manifest = ds.manifest()
    if not manifest:
        return {"odom_status": "skipped", "odom_reason": "empty_manifest"}

    available_k = sorted({int(m["k"]) for m in manifest if m.get("k", None) is not None})
    if not available_k:
        return {"odom_status": "skipped", "odom_reason": "manifest_has_no_k"}

    prefer_k = int(getattr(cfg, "odom_eval_prefer_k", 1))
    if prefer_k in available_k:
        selected_k = prefer_k
        fallback = False
        reason = "preferred_k"
    elif bool(getattr(cfg, "odom_eval_fallback_to_min_k", True)):
        selected_k = int(available_k[0])
        fallback = True
        reason = f"preferred_k_{prefer_k}_missing_fallback_to_k_{selected_k}"
    else:
        return {
            "odom_status": "skipped",
            "odom_reason": f"preferred_k_{prefer_k}_missing",
            "odom_available_k": available_k,
        }

    chains = _build_odometry_chains(
        manifest,
        selected_k,
        max_pairs=int(getattr(cfg, "odom_eval_max_pairs", 0)),
    )
    chains = [c for c in chains if len(c.get("pairs", [])) > 0]
    if not chains:
        return {
            "odom_status": "skipped",
            "odom_reason": "no_valid_chains",
            "odom_selected_k": int(selected_k),
            "odom_available_k": available_k,
        }

    model.eval()

    metric_pos_err_sq = []
    dir_pos_err_sq = []
    metric_rpe_rot = []
    metric_rpe_trans_dir = []
    metric_rpe_trans_mag = []
    dir_rpe_rot = []
    dir_rpe_trans_dir = []
    metric_endpoint_err_sum = 0.0
    dir_endpoint_err_sum = 0.0
    total_traj_len = 0.0
    metric_segments = 0
    dir_segments = 0
    num_pairs = 0
    metric_pairs = 0

    for chain in chains:
        R_gt_c0 = np.eye(3, dtype=np.float64)
        t_gt_c0 = np.zeros(3, dtype=np.float64)
        R_metric_c0 = np.eye(3, dtype=np.float64)
        t_metric_c0 = np.zeros(3, dtype=np.float64)
        R_dir_c0 = np.eye(3, dtype=np.float64)
        t_dir_c0 = np.zeros(3, dtype=np.float64)
        chain_len = 0.0
        chain_metric_ok = True

        for item in chain["pairs"]:
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
            t_gt_mag = float(np.linalg.norm(t_gt))
            if t_gt_mag <= 1e-12:
                continue

            R_pred, t_pred, aux = model(IA, IB, enable_depth_fusion=True)
            R_pred_np = R_pred.detach().float().cpu().numpy()[0]
            t_dir_pred_t = aux.get("t_dir_out", t_pred) if isinstance(aux, dict) else t_pred
            t_dir_pred = F.normalize(t_dir_pred_t.detach().float(), dim=-1, eps=1e-6).cpu().numpy()[0]

            t_vec_metric = None
            if isinstance(aux, dict):
                if aux.get("t_vec_out", None) is not None:
                    t_vec_metric = aux["t_vec_out"].detach().float().cpu().numpy()[0]
                elif aux.get("t_mag", None) is not None:
                    t_mag_pred = float(aux["t_mag"].detach().float().view(-1).cpu().numpy()[0])
                    t_vec_metric = t_dir_pred * t_mag_pred

            metric_rpe_rot.append(_rot_geodesic_deg_np(R_pred_np, R_gt))
            dir_rpe_rot.append(metric_rpe_rot[-1])
            metric_rpe_trans_dir.append(_vec_angle_deg_np(t_vec_metric, t_gt) if t_vec_metric is not None else float("nan"))
            dir_rpe_trans_dir.append(_vec_angle_deg_np(t_dir_pred, t_gt))
            if t_vec_metric is not None:
                metric_rpe_trans_mag.append(abs(float(np.linalg.norm(t_vec_metric)) - t_gt_mag))
                metric_pairs += 1
            else:
                chain_metric_ok = False

            t_dir_only = t_dir_pred * t_gt_mag
            R_gt_c0, t_gt_c0 = _compose_rel_pose_np(R_gt, t_gt, R_gt_c0, t_gt_c0)
            if t_vec_metric is not None:
                R_metric_c0, t_metric_c0 = _compose_rel_pose_np(R_pred_np, t_vec_metric, R_metric_c0, t_metric_c0)
            R_dir_c0, t_dir_c0 = _compose_rel_pose_np(R_pred_np, t_dir_only, R_dir_c0, t_dir_c0)

            p_gt = _camera_center_from_T_c0_np(R_gt_c0, t_gt_c0)
            if t_vec_metric is not None:
                p_metric = _camera_center_from_T_c0_np(R_metric_c0, t_metric_c0)
                metric_pos_err_sq.append(float(np.sum((p_metric - p_gt) ** 2)))
            p_dir = _camera_center_from_T_c0_np(R_dir_c0, t_dir_c0)
            dir_pos_err_sq.append(float(np.sum((p_dir - p_gt) ** 2)))

            chain_len += t_gt_mag
            num_pairs += 1

        if chain_len > 1e-12:
            p_gt_end = _camera_center_from_T_c0_np(R_gt_c0, t_gt_c0)
            if chain_metric_ok and metric_pairs > 0:
                p_metric_end = _camera_center_from_T_c0_np(R_metric_c0, t_metric_c0)
                metric_endpoint_err_sum += float(np.linalg.norm(p_metric_end - p_gt_end))
                metric_segments += 1
            p_dir_end = _camera_center_from_T_c0_np(R_dir_c0, t_dir_c0)
            dir_endpoint_err_sum += float(np.linalg.norm(p_dir_end - p_gt_end))
            dir_segments += 1
            total_traj_len += float(chain_len)

    def _nanmean(vals):
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size > 0 else float("nan")

    def _rmse(vals):
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(math.sqrt(float(arr.mean()))) if arr.size > 0 else float("nan")

    status = "ok" if num_pairs > 0 else "skipped"
    payload: Dict[str, Any] = {
        "odom_status": status,
        "odom_reason": reason if status == "ok" else "no_valid_pairs",
        "odom_available_k": available_k,
        "odom_selected_k": int(selected_k),
        "odom_used_fallback": bool(fallback),
        "odom_num_chains": int(len(chains)),
        "odom_num_pairs": int(num_pairs),
        "odom_metric_num_pairs": int(metric_pairs),
        "odom_direction_only_num_pairs": int(num_pairs),
        "odom_metric_RPE_rot": _nanmean(metric_rpe_rot),
        "odom_metric_RPE_trans_dir": _nanmean(metric_rpe_trans_dir),
        "odom_metric_RPE_trans_mag": _nanmean(metric_rpe_trans_mag),
        "odom_metric_ATE": _rmse(metric_pos_err_sq),
        "odom_metric_drift": float(metric_endpoint_err_sum / max(metric_segments, 1)) if metric_segments > 0 else float("nan"),
        "odom_metric_trajectory_length": float(total_traj_len),
        "odom_metric_length_normalized_drift": float(metric_endpoint_err_sum / max(total_traj_len, 1e-12)) if metric_segments > 0 else float("nan"),
        "odom_direction_only_RPE_rot": _nanmean(dir_rpe_rot),
        "odom_direction_only_RPE_trans_dir": _nanmean(dir_rpe_trans_dir),
        "odom_direction_only_ATE": _rmse(dir_pos_err_sq),
        "odom_direction_only_drift": float(dir_endpoint_err_sum / max(dir_segments, 1)) if dir_segments > 0 else float("nan"),
        "odom_direction_only_trajectory_length": float(total_traj_len),
        "odom_direction_only_length_normalized_drift": float(dir_endpoint_err_sum / max(total_traj_len, 1e-12)) if dir_segments > 0 else float("nan"),
    }
    model.train()
    return payload


def main():
    args = _parse_args()
    cfg = Config()
    _apply_cfg_overrides(cfg, args.overrides)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    _seed_everything(cfg.data_seed, deterministic=bool(cfg.deterministic))

    print("=" * 80)
    print(f"[Env ] torch {torch.__version__}")
    print(f"[Env ] cuda available: {torch.cuda.is_available()} | device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        print(f"[Env ] GPU  : {props.name}")
        print(f"[Env ] VRAM : total={props.total_memory/1024**3:.2f} GB")
        print(f"[Env ] CC   : {props.major}.{props.minor}")
        try:
            print(f"[Env ] BF16 : {torch.cuda.is_bf16_supported()}")
        except Exception:
            pass
    else:
        dev = torch.device("cpu")

    torch.set_float32_matmul_precision(cfg.matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
    torch.backends.cudnn.deterministic = bool(cfg.deterministic)
    torch.backends.cudnn.benchmark = bool(cfg.benchmark)

    print(f"[Cfg ] model_variant: coarse_interaction={cfg.use_coarse_interaction} | fine_stage={cfg.use_fine_stage} | epi_bias={cfg.use_epipolar_bias} | epi_loss={cfg.use_epipolar_loss} | depth={cfg.use_depth_branch}")
    print(f"[Cfg ] HxW={cfg.H}x{cfg.W} | D={cfg.D} | Nc={cfg.Nc} | Nf={cfg.Nf} | p={cfg.p}")
    print(f"[Cfg ] temp(coarse/fine)={cfg.coarse_temperature}/{cfg.fine_temperature} | logits_clip={cfg.logits_clip} | topk_coarse={cfg.topk_coarse}")
    print(
        f"[Cfg ] feature: patch_coords={getattr(cfg, 'patch_embed_use_coords', False)} | "
        f"patch_pool={getattr(cfg, 'patch_embed_pool_mode', 'avg')} | "
        f"patch_avgmax={getattr(cfg, 'patch_embed_avgmax_pool', False)} | "
        f"bearing_fuse={getattr(cfg, 'use_bearing_fuse', False)} | "
        f"cross_context={getattr(cfg, 'use_cross_context', False)}"
        f"x{getattr(cfg, 'cross_context_layers', 1)}"
        f"@{getattr(cfg, 'cross_context_strength', 0.25)} | "
        f"t_branch={getattr(cfg, 'use_translation_feature_branch', False)}"
        f":{getattr(cfg, 'translation_patch_pool_mode', 'gated_avgmax')}"
        f":gate={getattr(cfg, 'translation_patch_pool_gate_init', -2.0)}"
        f":pos={getattr(cfg, 'translation_pos_enc_scale', 1.0)}"
        f":L{getattr(cfg, 'translation_branch_encoder_layers', 0)}"
        f":detach={getattr(cfg, 'translation_branch_detach_match', True)} | "
        f"pose_stats_pool={getattr(cfg, 'pose_use_stats_pool', False)}"
    )
    print(
        f"[Cfg ] loss: w_pose={cfg.w_pose} | w_pose_out_t={getattr(cfg, 'w_pose_output_t', 0.0)} | t_alpha={cfg.pose_t_alpha} | "
        f"t_oriented={getattr(cfg, 'pose_t_oriented_weight', 1.0)} | "
        f"t_axis={getattr(cfg, 'pose_t_axis_weight', 0.0)} | "
        f"rot_k>={getattr(cfg, 'large_k_rot_thresh', 40)}x{getattr(cfg, 'large_k_rot_weight', 1.0)} | "
        f"tdir_dt<{getattr(cfg, 'tdir_loss_ignore_dt_below', 0.0)}x{getattr(cfg, 'tdir_loss_ignore_weight', 0.0)} | "
        f"w_epi={cfg.w_epi} | w_coarse_pose_aux={getattr(cfg, 'w_coarse_pose_aux', 0.0)} | "
        f"w_tmag={_cfg_tmag_weight(cfg)} | tmag_loss={getattr(cfg, 'tmag_loss_type', 'log_smooth_l1')} | "
        f"tmag_start={getattr(cfg, 'tmag_start_updates', 0)} | tmag_ramp={getattr(cfg, 'tmag_ramp_updates', 0)} | "
        f"tmag_detach={bool(getattr(cfg, 'tmag_detach_features', False))} | "
        f"tdir_anchor={bool(getattr(cfg, 'use_tdir_anchor_loss', False))}:w={getattr(cfg, 'w_tdir_anchor', 0.0)}"
        f":dt>={getattr(cfg, 'tdir_anchor_min_dt', 0.2)}:k>={getattr(cfg, 'tdir_anchor_min_k', 0)}"
    )
    print(
        f"[Cfg ] lr={cfg.lr} | wd={cfg.wd} | warmup_updates={cfg.warmup_updates} | "
        f"hold<{cfg.lr_hold_updates} | drop1<{cfg.lr_drop1_updates}@{cfg.lr_drop1_scale} | drop2@{cfg.lr_drop2_scale}"
    )
    print(f"[Cfg ] split_by={cfg.split_by} | train_ratio={cfg.train_ratio} | split_seed={cfg.split_seed} | data_seed={cfg.data_seed}")
    print(
        f"[Cfg ] train_aug: color={getattr(cfg, 'train_color_aug', False)} | "
        f"strength={getattr(cfg, 'train_color_aug_strength', 0.0)}"
    )
    print("=" * 80)

    train_ds = RflyPanoPanoramaPairsMixedK(
        data_root=cfg.data_root,
        split="train",
        split_by=cfg.split_by,
        train_ratio=cfg.train_ratio,
        split_seed=cfg.split_seed,
        seed=cfg.data_seed,
        H=cfg.H,
        W=cfg.W,
        min_dt=cfg.min_dt,
        k_choices=cfg.k_choices,
        k_probs=cfg.k_probs,
        strict_dt=True,
        color_aug=bool(getattr(cfg, "train_color_aug", False)),
        color_aug_strength=float(getattr(cfg, "train_color_aug_strength", 1.0)),
    )
    train_ds.max_dt = cfg.max_dt

    if bool(cfg.eval_use_fixed_pairs):
        test_ds = RflyPanoPanoramaPairsEvalFixedKList(
            data_root=cfg.data_root,
            split="test",
            split_by=cfg.split_by,
            train_ratio=cfg.train_ratio,
            split_seed=cfg.split_seed,
            H=cfg.H,
            W=cfg.W,
            k_list=cfg.eval_k_list,
            pair_step=cfg.eval_pair_step,
            min_dt=cfg.eval_min_dt,
            max_dt=cfg.eval_max_dt,
        )
    else:
        test_ds = RflyPanoPanoramaPairsMixedK(
            data_root=cfg.data_root,
            split="test",
            split_by=cfg.split_by,
            train_ratio=cfg.train_ratio,
            split_seed=cfg.split_seed,
            seed=cfg.data_seed,
            H=cfg.H,
            W=cfg.W,
            min_dt=cfg.min_dt,
            k_choices=cfg.k_choices,
            k_probs=cfg.k_probs,
            strict_dt=True,
        )
        test_ds.max_dt = cfg.max_dt

    # Fixed-pair train evaluation set for diagnosing train/test gap.
    # It uses the same deterministic evaluation protocol as test_ds, but on the train split.
    train_eval_ds = RflyPanoPanoramaPairsEvalFixedKList(
        data_root=cfg.data_root,
        split="train",
        split_by=cfg.split_by,
        train_ratio=cfg.train_ratio,
        split_seed=cfg.split_seed,
        H=cfg.H,
        W=cfg.W,
        k_list=cfg.eval_k_list,
        pair_step=cfg.eval_pair_step,
        min_dt=cfg.eval_min_dt,
        max_dt=cfg.eval_max_dt,
    )

    train_keys = set(getattr(train_ds, "sequence_keys", []))
    test_keys = set(getattr(test_ds, "sequence_keys", []))
    overlap = train_keys & test_keys
    if overlap:
        raise RuntimeError(f"train/test split leakage detected: {len(overlap)} overlapping groups, e.g. {sorted(list(overlap))[:8]}")

    for tag, ds in [("train", train_ds), ("train_eval", train_eval_ds), ("test", test_ds)]:
        groups = getattr(ds, "sequence_keys", [])
        print(
            f"[Split] {tag} by {cfg.split_by} | groups={len(groups)} | "
            f"seqs={len(groups)} | pairs={len(ds)} | preview={groups[:4]}"
        )
    print(
        f"[Eval ] protocol={'fixed_pairs' if bool(cfg.eval_use_fixed_pairs) else 'mixed_k_random'} | "
        f"test_max_eval_batches={cfg.max_eval_batches} | "
        f"train_max_eval_batches={getattr(cfg, 'max_train_eval_batches', 128)}"
    )
    _print_dataset_sanity("train", train_ds)
    _print_dataset_sanity("train_eval", train_eval_ds)
    _print_dataset_sanity("test", test_ds)
    _print_manifest_dt_sanity("train_eval", train_eval_ds)
    _print_manifest_dt_sanity("test", test_ds)

    ckpt_root = os.path.join(cfg.ckpt_dir, cfg.exp_name)
    vis_root = os.path.join(ckpt_root, "vis")
    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(vis_root, exist_ok=True)
    if bool(cfg.eval_use_fixed_pairs):
        _save_eval_manifest(os.path.join(ckpt_root, "eval_pairs_manifest.json"), test_ds, cfg)

    worker_init = _make_worker_init_fn(cfg.data_seed)
    train_gen = torch.Generator()
    train_gen.manual_seed(int(cfg.data_seed))
    test_gen = torch.Generator()
    test_gen.manual_seed(int(cfg.data_seed) + 1)

    def _loader_worker_kwargs(num_workers: int) -> Dict[str, Any]:
        if int(num_workers) <= 0:
            return {}
        return {
            "persistent_workers": bool(getattr(cfg, "persistent_workers", True)),
            "prefetch_factor": int(getattr(cfg, "prefetch_factor", 2)),
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        worker_init_fn=worker_init,
        generator=train_gen,
        **_loader_worker_kwargs(cfg.num_workers),
    )
    test_num_workers = 0 if bool(cfg.eval_use_fixed_pairs) else cfg.num_workers
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=test_num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        worker_init_fn=worker_init if not bool(cfg.eval_use_fixed_pairs) else None,
        generator=test_gen,
        **_loader_worker_kwargs(test_num_workers),
    )

    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        worker_init_fn=None,
        generator=test_gen,
        **_loader_worker_kwargs(0),
    )

    model = PanoramaRelPoseModel(cfg, dev).to(dev)
    _load_model_init_checkpoint(
        model,
        str(getattr(cfg, "init_checkpoint", "")),
        dev,
        strict=bool(getattr(cfg, "strict_load_checkpoint", False)),
    )
    if bool(getattr(cfg, "eval_only", False)):
        t_eval0 = time.perf_counter()
        (
            rot,
            tdir,
            tdir_abs,
            tdir_local_A,
            tdir_local_A_abs,
            tdiag,
            tdir_msg,
            tdir_local_msg,
            vis_payload,
            bucket_k,
            bucket_dt,
            bucket_k_dt,
            bucket_msgs,
        ) = eval_model(
            model,
            test_loader,
            dev,
            cfg,
            collect_vis=bool(getattr(cfg, "save_vis_examples", False)),
            vis_index=int(getattr(cfg, "vis_eval_index", 0)),
        )
        t_eval = time.perf_counter() - t_eval0
        joint_score = tdir_abs + float(cfg.joint_rot_weight) * rot
        joint_local_A_abs_score = tdir_local_A_abs + float(cfg.joint_rot_weight) * rot
        joint_eligible = (rot < float(cfg.joint_rot_thresh_deg)) and (tdir_abs < float(cfg.joint_tdir_thresh_deg))
        joint_local_A_abs_eligible = (
            (rot < float(cfg.joint_rot_thresh_deg))
            and (tdir_local_A_abs < float(cfg.joint_tdir_thresh_deg))
        )

        metrics: Dict[str, Any] = {
            "step": 0,
            "upd": 0,
            "rot": float(rot),
            "tdir": float(tdir),
            "tdir_abs": float(tdir_abs),
            "tdir_local_A": float(tdir_local_A),
            "tdir_local_A_abs": float(tdir_local_A_abs),
            "joint_score": float(joint_score),
            "joint_local_A_abs_score": float(joint_local_A_abs_score),
            "joint_eligible": int(joint_eligible),
            "joint_local_A_abs_eligible": int(joint_local_A_abs_eligible),
            "bucket_k": bucket_k,
            "bucket_dt": bucket_dt,
            "bucket_k_dt": bucket_k_dt,
            **{k: float(v) for k, v in tdiag.items()},
        }

        print(
            f"[EvalOnly] rot={rot:.4f}° | tdir={tdir:.4f}° | tdir_abs={tdir_abs:.4f}° | "
            f"tdir_local_A={tdir_local_A:.4f}° | tdir_local_A_abs={tdir_local_A_abs:.4f}° | "
            f"epi_mass={tdiag.get('epi_mass_in_gt_band', float('nan')):.4f} | "
            f"top1={tdiag.get('top1_in_gt_band', float('nan')):.4f} | "
            f"top5={tdiag.get('top5_in_gt_band', float('nan')):.4f} | "
            f"ent={tdiag.get('matching_entropy', float('nan')):.4f} | "
            f"cyc={tdiag.get('cycle_error', float('nan')):.4f} | "
            f"tmag_rel={tdiag.get('tmag_rel_err', float('nan')):.3f} | "
            f"tvec_l2={tdiag.get('trans_vec_l2', float('nan')):.3f} | "
            f"joint_abs={joint_score:.4f} | joint_local={joint_local_A_abs_score:.4f} | time={t_eval:.2f}s"
        )
        print(tdir_msg)
        print(tdir_local_msg)
        for _bucket_msg in bucket_msgs:
            print(_bucket_msg)

        odom_metrics: Dict[str, Any] = {}
        if bool(getattr(cfg, "use_odometry_eval", True)):
            t_odom0 = time.perf_counter()
            odom_metrics = eval_odometry_sequence(model, test_ds, dev, cfg)
            t_odom = time.perf_counter() - t_odom0
            if odom_metrics.get("odom_status") == "ok":
                print(
                    f"[OdomEval] k={odom_metrics.get('odom_selected_k')} "
                    f"fallback={int(bool(odom_metrics.get('odom_used_fallback', False)))} | "
                    f"pairs={odom_metrics.get('odom_num_pairs', 0)} | "
                    f"RPE_rot={odom_metrics.get('odom_metric_RPE_rot', float('nan')):.4f}° | "
                    f"ATE={odom_metrics.get('odom_metric_ATE', float('nan')):.4f} | "
                    f"drift={odom_metrics.get('odom_metric_drift', float('nan')):.4f} | "
                    f"dir_ATE={odom_metrics.get('odom_direction_only_ATE', float('nan')):.4f} | "
                    f"time={t_odom:.2f}s"
                )
            else:
                print(f"[OdomEval] skipped: {odom_metrics.get('odom_reason', 'unknown')} | time={t_odom:.2f}s")
            if bool(getattr(cfg, "save_odom_metrics_latest", True)):
                _save_json(os.path.join(ckpt_root, "odom_metrics_latest.json"), {"step": 0, "upd": 0, **odom_metrics})
            metrics.update(odom_metrics)

        eval_history = [metrics]
        if bool(cfg.save_eval_history):
            _save_json(os.path.join(ckpt_root, "eval_history.json"), {"history": eval_history})
        if bool(getattr(cfg, 'save_eval_buckets_latest', True)):
            bucket_payload = {
                'step': 0,
                'upd': 0,
                'bucket_k': bucket_k,
                'bucket_dt': bucket_dt,
                'bucket_k_dt': bucket_k_dt,
                'train_bucket_k': {},
                'train_bucket_dt': {},
                'train_bucket_k_dt': {},
            }
            _save_json(os.path.join(ckpt_root, 'eval_buckets_latest.json'), bucket_payload)
            _write_eval_buckets_csv(os.path.join(ckpt_root, 'eval_buckets_latest.csv'), bucket_payload)
            _save_json(
                os.path.join(ckpt_root, 'matching_diag_latest.json'),
                _matching_diag_payload(0, 0, tdiag, {}),
            )
        if bool(getattr(cfg, "save_vis_examples", False)) and vis_payload is not None:
            if bool(getattr(cfg, "save_vis_payload_npz", False)):
                _save_vis_payload_npz(os.path.join(vis_root, "vis_example_latest.npz"), vis_payload)
            if bool(getattr(cfg, "save_vis_diag_json", True)):
                _save_json(os.path.join(vis_root, "vis_diag_latest.json"), vis_payload.get("diag", {}))
            _plot_softcorr_overview(vis_payload, os.path.join(vis_root, "vis_softcorr_latest.png"))
            _plot_depth_overview(vis_payload, os.path.join(vis_root, "vis_depth_latest.png"))

        final_summary = {
            "best_rot": float(rot),
            "best_tdir_raw": float(tdir),
            "best_tdir_abs": float(tdir_abs),
            "best_tdir_local_A": float(tdir_local_A),
            "best_tdir_local_A_abs": float(tdir_local_A_abs),
            "best_joint": float(joint_score),
            "best_joint_local_A_abs": float(joint_local_A_abs_score),
            "bad_forward": 0,
            "skip_updates": 0,
            "total_steps": 0,
            "total_updates": 0,
            "nan_like_event_rate_per_step": 0.0,
            "nan_like_event_rate_per_update": 0.0,
            "selection_method": "eval_only",
            "eval_points": 1,
            "tmag_base_weight": float(_cfg_tmag_weight(cfg)),
            "tmag_start_updates": int(getattr(cfg, "tmag_start_updates", 0)),
            "tmag_ramp_updates": int(getattr(cfg, "tmag_ramp_updates", 0)),
            "tmag_detach_features": bool(getattr(cfg, "tmag_detach_features", False)),
            "init_checkpoint": str(getattr(cfg, "init_checkpoint", "")),
            "strict_load_checkpoint": bool(getattr(cfg, "strict_load_checkpoint", False)),
            "use_tdir_anchor_loss": bool(getattr(cfg, "use_tdir_anchor_loss", False)),
            "tdir_anchor_checkpoint": str(getattr(cfg, "tdir_anchor_checkpoint", "")),
            "w_tdir_anchor": float(getattr(cfg, "w_tdir_anchor", 0.0)),
            "tdir_anchor_min_dt": float(getattr(cfg, "tdir_anchor_min_dt", 0.2)),
            "tdir_anchor_min_k": int(getattr(cfg, "tdir_anchor_min_k", 0)),
            "tdir_anchor_start_updates": int(getattr(cfg, "tdir_anchor_start_updates", 0)),
            "tdir_anchor_ramp_updates": int(getattr(cfg, "tdir_anchor_ramp_updates", 0)),
            "tdir_loss_ignore_dt_below": float(getattr(cfg, "tdir_loss_ignore_dt_below", 0.0)),
            "tdir_loss_ignore_weight": float(getattr(cfg, "tdir_loss_ignore_weight", 0.0)),
            "last_eval": _latest_eval_metrics(metrics),
        }
        if bool(cfg.save_final_summary):
            _save_json(os.path.join(ckpt_root, "final_summary.json"), final_summary)
        print(
            f"[Done ] eval_only finished | rot={rot:.4f}° | tdir_abs={tdir_abs:.4f}° | "
            f"tdir_local_A_abs={tdir_local_A_abs:.4f}° | joint={joint_score:.4f} | "
            f"joint_local={joint_local_A_abs_score:.4f} | ckpt_dir={ckpt_root}"
        )
        return

    tdir_anchor_model = None
    if bool(getattr(cfg, "use_tdir_anchor_loss", False)) and float(getattr(cfg, "w_tdir_anchor", 0.0)) > 0.0:
        anchor_ckpt = str(getattr(cfg, "tdir_anchor_checkpoint", ""))
        if not anchor_ckpt:
            raise ValueError("use_tdir_anchor_loss=True requires tdir_anchor_checkpoint.")
        tdir_anchor_model = PanoramaRelPoseModel(cfg, dev).to(dev)
        _load_model_init_checkpoint(tdir_anchor_model, anchor_ckpt, dev, strict=False, label="TdirAnchor")
        tdir_anchor_model.eval()
        for p in tdir_anchor_model.parameters():
            p.requires_grad_(False)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=_scheduler_lambda(
            warmup_updates=cfg.warmup_updates,
            hold_updates=cfg.lr_hold_updates,
            drop1_updates=cfg.lr_drop1_updates,
            drop1_scale=cfg.lr_drop1_scale,
            drop2_scale=cfg.lr_drop2_scale,
        ),
    )

    use_amp = bool(cfg.amp) and dev.type == "cuda"
    if cfg.amp_dtype.lower() == "auto":
        amp_dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    elif cfg.amp_dtype.lower() in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    step = 0
    upd = 0
    skip_updates = 0
    bad_forward = 0
    last_bad_grad_name = "-"
    last_grad_norm = float("nan")
    best_rot = float("inf")
    best_tdir_abs = float("inf")
    best_tdir_raw = float("inf")
    best_tdir_local_A = float("inf")
    best_tdir_local_A_abs = float("inf")
    best_joint = float("inf")
    best_joint_local_A_abs = float("inf")
    best_odom_metric = float("inf")
    best_odom_drift = float("inf")
    best_odom_upd = -1
    best_odom_tdir_abs = float("inf")
    best_odom_tmag_rel_err = float("inf")
    best_odom_checkpoint = ""
    best_smallk_odom_metric = float("inf")
    best_smallk_odom_drift = float("inf")
    best_smallk_odom_upd = -1
    best_smallk_odom_tdir_abs = float("inf")
    best_smallk_odom_tmag_rel_err = float("inf")
    best_smallk_odom_count = 0
    best_smallk_odom_checkpoint = ""
    eval_history: List[Dict[str, Any]] = []
    vis_dumped = False

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)

    while step < cfg.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        t0 = time.perf_counter()
        IA = batch["IA"].to(dev, non_blocking=True)
        IB = batch["IB"].to(dev, non_blocking=True)
        R_gt = batch["R_gt"].to(dev, non_blocking=True)
        t_gt = batch["t_gt_dir"].to(dev, non_blocking=True)
        meta = batch.get("meta", None)
        t_gt_mag = _translation_magnitude_gt(batch, meta, IA.shape[0], dev)
        dt_t_weight = torch.tensor(
            _translation_weight_from_dt(
                meta,
                IA.shape[0],
                small_dt_thresh=float(getattr(cfg, "small_dt_thresh", 0.2)),
                small_dt_t_weight=float(getattr(cfg, "small_dt_t_weight", 0.35)),
                ignore_dt_below=float(getattr(cfg, "tdir_loss_ignore_dt_below", 0.0)),
                ignore_weight=float(getattr(cfg, "tdir_loss_ignore_weight", 0.0)),
            ),
            device=dev,
            dtype=torch.float32,
        )
        rot_k_weight = torch.tensor(
            _rotation_weight_from_k(
                meta,
                IA.shape[0],
                large_k_thresh=int(getattr(cfg, "large_k_rot_thresh", 40)),
                large_k_rot_weight=float(getattr(cfg, "large_k_rot_weight", 1.0)),
            ),
            device=dev,
            dtype=torch.float32,
        )
        t_data = time.perf_counter()

        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
            depth_fusion_active = bool(cfg.use_depth_branch) and (upd >= int(getattr(cfg, "depth_fuse_start_updates", 0)))
            R_pred, t_pred, aux = model(IA, IB, enable_depth_fusion=depth_fusion_active)

            t_pose_pred = aux.get("t_dir_local", t_pred)
            t_pose_frame = aux.get("t_local_frame", "B") if aux.get("t_dir_local", None) is not None else "B"
            L_pose = pose_loss(
                R_pred,
                t_pose_pred,
                R_gt,
                t_gt,
                pose_t_alpha=cfg.pose_t_alpha,
                pose_t_oriented_weight=float(getattr(cfg, "pose_t_oriented_weight", 1.0)),
                pose_t_axis_weight=float(getattr(cfg, "pose_t_axis_weight", 0.0)),
                pred_t_frame=t_pose_frame,
                rot_sample_weight=rot_k_weight,
                t_sample_weight=dt_t_weight,
            )
            if (
                float(getattr(cfg, "w_pose_output_t", 0.0)) > 0.0
                and isinstance(aux, dict)
                and aux.get("t_dir_out", None) is not None
            ):
                L_pose = L_pose + float(cfg.w_pose_output_t) * translation_direction_loss(
                    aux["t_dir_out"],
                    t_gt,
                    R_gt,
                    pred_t_frame="B",
                    oriented_weight=float(getattr(cfg, "pose_t_oriented_weight", 1.0)),
                    axis_weight=float(getattr(cfg, "pose_t_axis_weight", 0.0)),
                    sample_weight=dt_t_weight,
                )
            fine_pose_fuse_strength = float(getattr(cfg, "fine_pose_fuse_strength", 1.0))
            coarse_pose_aux_w = float(getattr(cfg, "w_coarse_pose_aux", 0.0))
            if bool(cfg.use_fine_stage) and fine_pose_fuse_strength <= 1e-8:
                coarse_pose_aux_w = 0.0

            if coarse_pose_aux_w > 0.0 and bool(cfg.use_fine_stage) and aux.get("Rc", None) is not None and aux.get("tc_dir", None) is not None:
                L_pose_coarse = pose_loss(
                    aux["Rc"],
                    aux["tc_dir"],
                    R_gt,
                    t_gt,
                    pose_t_alpha=cfg.pose_t_alpha,
                    pose_t_oriented_weight=float(getattr(cfg, "pose_t_oriented_weight", 1.0)),
                    pose_t_axis_weight=float(getattr(cfg, "pose_t_axis_weight", 0.0)),
                    pred_t_frame=aux.get("t_local_frame", "A"),
                    rot_sample_weight=rot_k_weight,
                    t_sample_weight=dt_t_weight,
                )
            else:
                L_pose_coarse = torch.zeros((), device=dev)

            tmag_w_eff = _cfg_tmag_effective_weight(cfg, upd)
            if (
                bool(getattr(cfg, "use_translation_magnitude_head", True))
                and tmag_w_eff > 0.0
                and t_gt_mag is not None
                and isinstance(aux, dict)
                and aux.get("t_mag", None) is not None
            ):
                L_t_mag = translation_magnitude_loss(
                    aux["t_mag"],
                    t_gt_mag,
                    loss_type=str(getattr(cfg, "tmag_loss_type", "log_smooth_l1")),
                    eps=float(getattr(cfg, "tmag_min", 1.0e-3)),
                    sample_weight=None,
                )
            else:
                L_t_mag = torch.zeros((), device=dev)

            tdir_anchor_w_eff = _cfg_tdir_anchor_effective_weight(cfg, upd)
            L_tdir_anchor = torch.zeros((), device=dev)
            tdir_anchor_n = 0.0
            if tdir_anchor_model is not None and tdir_anchor_w_eff > 0.0:
                anchor_weight = _tdir_anchor_weight_from_meta(
                    meta,
                    IA.shape[0],
                    min_dt=float(getattr(cfg, "tdir_anchor_min_dt", 0.2)),
                    min_k=int(getattr(cfg, "tdir_anchor_min_k", 0)),
                    device=dev,
                )
                tdir_anchor_n = float((anchor_weight > 0.0).sum().detach().cpu())
                if tdir_anchor_n > 0.0:
                    with torch.no_grad():
                        _, t_teacher_pred, aux_teacher = tdir_anchor_model(
                            IA,
                            IB,
                            enable_depth_fusion=depth_fusion_active,
                        )
                    t_student_dir = aux.get("t_dir_out", t_pred) if isinstance(aux, dict) else t_pred
                    t_teacher_dir = (
                        aux_teacher.get("t_dir_out", t_teacher_pred)
                        if isinstance(aux_teacher, dict)
                        else t_teacher_pred
                    )
                    assert t_student_dir.shape == t_teacher_dir.shape, (
                        f"tdir anchor shape mismatch: student={tuple(t_student_dir.shape)}, "
                        f"teacher={tuple(t_teacher_dir.shape)}"
                    )
                    L_tdir_anchor = _translation_direction_anchor_loss(
                        t_student_dir,
                        t_teacher_dir,
                        anchor_weight,
                    )

            use_fine_epi = bool(cfg.use_fine_stage) and (aux.get("Wf_ab", None) is not None)
            if use_fine_epi:
                W_ab = aux.get("Wf_ab", None)
                W_ba = aux.get("Wf_ba", None)
                cA = _first_not_none(aux.get("cA_f", None), aux.get("cA_c", None))
                cB = _first_not_none(aux.get("cB_f", None), aux.get("cB_c", None))
                bearingA = aux.get("bearingA_f", None)
                bearingB = aux.get("bearingB_f", None)
                epi_allowed_mask = aux.get("allowed_mask", None)
            else:
                W_ab = aux.get("Wc_ab", None)
                W_ba = aux.get("Wc_ba", None)
                cA = aux.get("cA_c", None)
                cB = aux.get("cB_c", None)
                bearingA = aux.get("bearingA_c", None)
                bearingB = aux.get("bearingB_c", None)
                epi_allowed_mask = None

            if W_ab is not None and bearingA is not None and bearingB is not None:
                assert W_ab.shape[1] == bearingA.shape[1], (
                    f"Epipolar shape mismatch: W_ab={tuple(W_ab.shape)}, bearingA={tuple(bearingA.shape)}"
                )
                assert W_ab.shape[2] == bearingB.shape[1], (
                    f"Epipolar shape mismatch: W_ab={tuple(W_ab.shape)}, bearingB={tuple(bearingB.shape)}"
                )

            if W_ab is not None and cfg.w_x > 0:
                P = W_ab.float().clamp_min(1e-9)
                L_x = (-P * P.log()).sum(dim=-1).mean()
            else:
                L_x = torch.zeros((), device=dev)

            if W_ab is not None and W_ba is not None and cfg.w_cyc > 0:
                cyc = torch.matmul(W_ab.float(), W_ba.float())
                I = torch.eye(cyc.shape[-1], device=dev).unsqueeze(0)
                L_cyc = F.mse_loss(cyc, I)
            else:
                L_cyc = torch.zeros((), device=dev)

            if cA is not None and cB is not None and cfg.w_rel > 0:
                L_rel = 0.5 * (F.softplus(-cA.float()).mean() + F.softplus(-cB.float()).mean())
            else:
                L_rel = torch.zeros((), device=dev)


            if cfg.use_epipolar_loss and W_ab is not None and bearingA is not None and bearingB is not None and cfg.w_epi > 0:
                epi_loss_type = str(getattr(cfg, "epi_loss_type", "gt_match_ce")).lower()
                if epi_loss_type in ("gt_band_nll", "gt_band_mass", "band_nll", "mass"):
                    L_epi = epipolar_gt_band_nll_loss(
                        W_ab=W_ab,
                        W_ba=W_ba,
                        bearing_a=bearingA,
                        bearing_b=bearingB,
                        R_gt=R_gt,
                        t_gt=t_gt,
                        allowed_mask=epi_allowed_mask,
                        angle_thresh_deg=cfg.epi_angle_thresh_deg,
                        use_bidir=cfg.epi_loss_use_bidir,
                    )
                elif epi_loss_type in ("gt_match_ce", "gt_epipolar_match", "gt_soft_ce", "ce"):
                    L_epi = epipolar_gt_matching_loss(
                        W_ab=W_ab,
                        W_ba=W_ba,
                        bearing_a=bearingA,
                        bearing_b=bearingB,
                        R_gt=R_gt,
                        t_gt=t_gt,
                        allowed_mask=epi_allowed_mask,
                        angle_thresh_deg=cfg.epi_angle_thresh_deg,
                        use_bidir=cfg.epi_loss_use_bidir,
                        temperature=float(getattr(cfg, "epi_gt_target_temp", 1.0)),
                    )
                else:
                    L_epi = epipolar_simplified_loss(
                        W_ab=W_ab,
                        W_ba=W_ba,
                        bearing_a=bearingA,
                        bearing_b=bearingB,
                        R_gt=R_gt,
                        t_gt=t_gt,
                        allowed_mask=epi_allowed_mask,
                        angle_thresh_deg=cfg.epi_angle_thresh_deg,
                        use_bidir=cfg.epi_loss_use_bidir,
                    )
            else:
                L_epi = torch.zeros((), device=dev)

            Wc_aux = aux.get("Wc_ab", None)
            Wc_ba_aux = aux.get("Wc_ba", None)
            bearingA_c_aux = aux.get("bearingA_c", None)
            bearingB_c_aux = aux.get("bearingB_c", None)
            if (
                bool(cfg.use_fine_stage)
                and cfg.use_epipolar_loss
                and Wc_aux is not None
                and bearingA_c_aux is not None
                and bearingB_c_aux is not None
                and float(getattr(cfg, "w_coarse_epi_aux", 0.0)) > 0.0
            ):
                L_epi_coarse = epipolar_gt_band_nll_loss(
                    W_ab=Wc_aux,
                    W_ba=Wc_ba_aux,
                    bearing_a=bearingA_c_aux,
                    bearing_b=bearingB_c_aux,
                    R_gt=R_gt,
                    t_gt=t_gt,
                    allowed_mask=None,
                    angle_thresh_deg=cfg.epi_angle_thresh_deg,
                    use_bidir=cfg.epi_loss_use_bidir,
                )
            else:
                L_epi_coarse = torch.zeros((), device=dev)

            depth_key = f"inv_depth_s{int(cfg.depth_loss_scale)}"
            depth_ramp = _depth_weight_ramp(upd, cfg)
            if bool(cfg.use_depth_branch) and (depth_ramp > 0.0) and depth_key in aux and cfg.w_photo > 0:
                inv_depth = aux[depth_key]
                IA_small = _downsample_to_like(IA, inv_depth)
                IB_small = _downsample_to_like(IB, inv_depth)
                R_warp = R_pred.detach() if bool(cfg.depth_use_detached_pose) else R_pred
                t_warp = aux.get("t_dir_out", aux.get("t_dir", t_pred))
                t_warp = t_warp.detach() if bool(cfg.depth_use_detached_pose) else t_warp
                Iwarp, valid = warp_erp_with_depth_pose(
                    inv_depth,
                    R_warp,
                    t_warp,
                    IB_small,
                    translation_scale=1.0,
                    min_depth=cfg.depth_min,
                    max_depth=cfg.depth_max,
                )
                L_photo = erp_photometric_loss(
                    IA_small,
                    Iwarp,
                    valid,
                    I_tgt_identity=IB_small if bool(getattr(cfg, "depth_use_automask", True)) else None,
                    use_automask=bool(getattr(cfg, "depth_use_automask", True)),
                )
                L_smooth = depth_smoothness_loss(inv_depth, IA_small) if cfg.w_smooth > 0 else torch.zeros((), device=dev)
                photo_err = (IA_small - Iwarp).abs().mean(dim=1, keepdim=True)
                aux["IA_s"] = IA_small.detach()
                aux["IB_s"] = IB_small.detach()
                aux["Iwarp_s"] = Iwarp.detach()
                aux["warp_valid_s"] = valid.detach()
                aux["photo_err_s"] = (photo_err * valid.float()).detach()
            else:
                L_photo = torch.zeros((), device=dev)
                L_smooth = torch.zeros((), device=dev)

            photo_w = float(cfg.w_photo) * depth_ramp
            smooth_w = float(cfg.w_smooth) * depth_ramp
            epi_ramp_updates = int(getattr(cfg, "epi_ramp_updates", 0))
            epi_ramp = 1.0 if epi_ramp_updates <= 0 else min(1.0, max(0.0, float(upd) / float(epi_ramp_updates)))
            epi_w = float(cfg.w_epi) * epi_ramp
            L = (
                cfg.w_pose * L_pose
                + coarse_pose_aux_w * L_pose_coarse
                + cfg.w_x * L_x
                + cfg.w_cyc * L_cyc
                + cfg.w_rel * L_rel
                + epi_w * L_epi
                + float(getattr(cfg, "w_coarse_epi_aux", 0.0)) * epi_ramp * L_epi_coarse
                + tmag_w_eff * L_t_mag
                + tdir_anchor_w_eff * L_tdir_anchor
                + photo_w * L_photo
                + smooth_w * L_smooth
            )
            L_scaled = L / float(cfg.grad_accum)

        t_fwd = time.perf_counter()

        forward_ok = all(
            bool(torch.isfinite(x).all())
            for x in [
                R_pred, t_pred, L_pose, L_pose_coarse, L_t_mag, L_tdir_anchor,
                L_x, L_cyc, L_rel, L_epi, L_epi_coarse, L_photo, L_smooth, L,
            ]
        )
        if not forward_ok:
            bad_forward += 1
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0 or step == 1:
                print(f"[Warn ] non-finite forward detected, batch skipped | bad_forward={bad_forward}")
            continue

        if use_scaler:
            scaler.scale(L_scaled).backward()
        else:
            L_scaled.backward()
        t_bwd = time.perf_counter()

        if (step + 1) % cfg.grad_accum == 0:
            if use_scaler:
                scaler.unscale_(optimizer)

            bad_name = None
            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_name = name
                    break

            if bad_name is not None:
                skip_updates += 1
                last_bad_grad_name = bad_name
                last_grad_norm = float("nan")
                optimizer.zero_grad(set_to_none=True)
                if use_scaler:
                    scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    skip_updates += 1
                    last_bad_grad_name = "clip_grad_norm"
                    last_grad_norm = float("nan")
                    optimizer.zero_grad(set_to_none=True)
                    if use_scaler:
                        scaler.update()
                else:
                    last_grad_norm = float(grad_norm.detach().cpu())
                    last_bad_grad_name = "-"
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    upd += 1

                    if cfg.eval_every and upd % cfg.eval_every == 0:
                        want_vis = bool(cfg.save_vis_examples) and (bool(cfg.vis_dump_every_eval) or not vis_dumped)

                        # ------------------------------------------------------------
                        # 1) Train split evaluation: diagnose train/test generalization gap.
                        # ------------------------------------------------------------
                        old_max_eval_batches = cfg.max_eval_batches
                        cfg.max_eval_batches = int(getattr(cfg, "max_train_eval_batches", 128))

                        t_train_eval0 = time.perf_counter()
                        (
                            train_rot,
                            train_tdir,
                            train_tdir_abs,
                            train_tdir_local_A,
                            train_tdir_local_A_abs,
                            train_tdiag,
                            train_tdir_msg,
                            train_tdir_local_msg,
                            _train_vis_payload,
                            train_bucket_k,
                            train_bucket_dt,
                            train_bucket_k_dt,
                            train_bucket_msgs,
                        ) = eval_model(
                            model,
                            train_eval_loader,
                            dev,
                            cfg,
                            collect_vis=False,
                            vis_index=0,
                        )
                        t_train_eval = time.perf_counter() - t_train_eval0
                        cfg.max_eval_batches = old_max_eval_batches

                        print(
                            f"[Eval-Train] upd {upd:05d} (step {step:05d}) | "
                            f"rot={train_rot:.4f}° | tdir={train_tdir:.4f}° | "
                            f"tdir_abs={train_tdir_abs:.4f}° | "
                            f"tdir_local_A={train_tdir_local_A:.4f}° | "
                            f"tdir_local_A_abs={train_tdir_local_A_abs:.4f}° | "
                            f"epi_mass={train_tdiag.get('epi_mass_in_gt_band', float('nan')):.4f} | "
                            f"top1={train_tdiag.get('top1_in_gt_band', float('nan')):.4f} | "
                            f"top5={train_tdiag.get('top5_in_gt_band', float('nan')):.4f} | "
                            f"ent={train_tdiag.get('matching_entropy', float('nan')):.4f} | "
                            f"pmax={train_tdiag.get('max_matching_prob', float('nan')):.4f} | "
                            f"cyc={train_tdiag.get('cycle_error', float('nan')):.4f} | "
                            f"tmag_abs={train_tdiag.get('tmag_abs_err', float('nan')):.3f} | "
                            f"tmag_rel={train_tdiag.get('tmag_rel_err', float('nan')):.3f} | "
                            f"tvec_l2={train_tdiag.get('trans_vec_l2', float('nan')):.3f} | "
                            f"raw_t_abs={train_tdiag.get('t_raw_local_A_abs', float('nan')):.2f}° | "
                            f"geo_t_abs={train_tdiag.get('t_geo_local_A_abs', float('nan')):.2f}° | "
                            f"gref={train_tdiag.get('geom_refine_success_rate', float('nan')):.2f} | "
                            f"time={t_train_eval:.2f}s"
                        )
                        print("[Eval-Train] " + train_tdir_msg.replace("[TDIR-CHK] ", ""))
                        print("[Eval-Train] " + train_tdir_local_msg.replace("[TDIR-LOCAL] ", ""))
                        for _bucket_msg in train_bucket_msgs:
                            print("[Eval-Train] " + _bucket_msg)

                        # ------------------------------------------------------------
                        # 2) Test split evaluation: still used for checkpoint selection.
                        # ------------------------------------------------------------
                        t_eval0 = time.perf_counter()
                        (
                            rot,
                            tdir,
                            tdir_abs,
                            tdir_local_A,
                            tdir_local_A_abs,
                            tdiag,
                            tdir_msg,
                            tdir_local_msg,
                            vis_payload,
                            bucket_k,
                            bucket_dt,
                            bucket_k_dt,
                            bucket_msgs,
                        ) = eval_model(
                            model,
                            test_loader,
                            dev,
                            cfg,
                            collect_vis=want_vis,
                            vis_index=int(cfg.vis_eval_index),
                        )
                        t_eval = time.perf_counter() - t_eval0
                        joint_score = tdir_abs + float(cfg.joint_rot_weight) * rot
                        joint_local_A_abs_score = tdir_local_A_abs + float(cfg.joint_rot_weight) * rot
                        joint_eligible = (rot < float(cfg.joint_rot_thresh_deg)) and (tdir_abs < float(cfg.joint_tdir_thresh_deg))
                        joint_local_A_abs_eligible = (
                            (rot < float(cfg.joint_rot_thresh_deg))
                            and (tdir_local_A_abs < float(cfg.joint_tdir_thresh_deg))
                        )

                        metrics = {
                            # test metrics: checkpoint selection still uses these
                            "rot": rot,
                            "tdir": tdir,
                            "tdir_abs": tdir_abs,
                            "tdir_local_A": tdir_local_A,
                            "tdir_local_A_abs": tdir_local_A_abs,
                            "joint_score": joint_score,
                            "joint_local_A_abs_score": joint_local_A_abs_score,
                            "joint_eligible": int(joint_eligible),
                            "joint_local_A_abs_eligible": int(joint_local_A_abs_eligible),
                            "bucket_k": bucket_k,
                            "bucket_dt": bucket_dt,
                            "bucket_k_dt": bucket_k_dt,

                            # train-eval metrics: diagnostic only
                            "train_rot": train_rot,
                            "train_tdir": train_tdir,
                            "train_tdir_abs": train_tdir_abs,
                            "train_tdir_local_A": train_tdir_local_A,
                            "train_tdir_local_A_abs": train_tdir_local_A_abs,
                            "train_bucket_k": train_bucket_k,
                            "train_bucket_dt": train_bucket_dt,
                            "train_bucket_k_dt": train_bucket_k_dt,

                            **{f"tdir_diag_{k}": v for k, v in tdiag.items()},
                            **{f"train_tdir_diag_{k}": v for k, v in train_tdiag.items()},
                        }

                        print(
                            f"[Eval ] upd {upd:05d} (step {step:05d}) | rot={rot:.4f}° | "
                            f"tdir={tdir:.4f}° | tdir_abs={tdir_abs:.4f}° | "
                            f"tdir_local_A={tdir_local_A:.4f}° | tdir_local_A_abs={tdir_local_A_abs:.4f}° | "
                            f"epi_mass={tdiag.get('epi_mass_in_gt_band', float('nan')):.4f} | "
                            f"top1={tdiag.get('top1_in_gt_band', float('nan')):.4f} | "
                            f"top5={tdiag.get('top5_in_gt_band', float('nan')):.4f} | "
                            f"ent={tdiag.get('matching_entropy', float('nan')):.4f} | "
                            f"pmax={tdiag.get('max_matching_prob', float('nan')):.4f} | "
                            f"cyc={tdiag.get('cycle_error', float('nan')):.4f} | "
                            f"tcos={tdiag.get('tdir_cos', float('nan')):.3f} | "
                            f"flip_rate={tdiag.get('tdir_flip_rate', float('nan')):.3f} | "
                            f"tmag_abs={tdiag.get('tmag_abs_err', float('nan')):.3f} | "
                            f"tmag_rel={tdiag.get('tmag_rel_err', float('nan')):.3f} | "
                            f"tvec_l2={tdiag.get('trans_vec_l2', float('nan')):.3f} | "
                            f"stable_abs={tdiag.get('stable_tdir_abs', float('nan')):.2f}° | "
                            f"stable_local={tdiag.get('stable_tdir_local_A_abs', float('nan')):.2f}° | "
                            f"raw_t_abs={tdiag.get('t_raw_local_A_abs', float('nan')):.2f}° | "
                            f"geo_t_abs={tdiag.get('t_geo_local_A_abs', float('nan')):.2f}° | "
                            f"gref={tdiag.get('geom_refine_success_rate', float('nan')):.2f} | "
                            f"geom_rot={tdiag.get('geom_rot', float('nan')):.2f}° | "
                            f"geom_abs={tdiag.get('geom_tdir_abs', float('nan')):.2f}° | "
                            f"joint_abs={joint_score:.4f} | joint_local={joint_local_A_abs_score:.4f} | "
                            f"joint_ok={int(joint_eligible)} | joint_local_ok={int(joint_local_A_abs_eligible)} | time={t_eval:.2f}s"
                        )
                        print(tdir_msg)
                        print(tdir_local_msg)
                        for _bucket_msg in bucket_msgs:
                            print(_bucket_msg)

                        odom_metrics: Dict[str, Any] = {}
                        if bool(getattr(cfg, "use_odometry_eval", True)):
                            t_odom0 = time.perf_counter()
                            odom_metrics = eval_odometry_sequence(model, test_ds, dev, cfg)
                            t_odom = time.perf_counter() - t_odom0
                            if odom_metrics.get("odom_status") == "ok":
                                print(
                                    f"[OdomEval] k={odom_metrics.get('odom_selected_k')} "
                                    f"fallback={int(bool(odom_metrics.get('odom_used_fallback', False)))} | "
                                    f"pairs={odom_metrics.get('odom_num_pairs', 0)} | "
                                    f"RPE_rot={odom_metrics.get('odom_metric_RPE_rot', float('nan')):.4f}° | "
                                    f"ATE={odom_metrics.get('odom_metric_ATE', float('nan')):.4f} | "
                                    f"drift={odom_metrics.get('odom_metric_drift', float('nan')):.4f} | "
                                    f"dir_ATE={odom_metrics.get('odom_direction_only_ATE', float('nan')):.4f} | "
                                    f"time={t_odom:.2f}s"
                                )
                            else:
                                print(f"[OdomEval] skipped: {odom_metrics.get('odom_reason', 'unknown')} | time={t_odom:.2f}s")
                            if bool(getattr(cfg, "save_odom_metrics_latest", True)):
                                _save_json(
                                    os.path.join(ckpt_root, "odom_metrics_latest.json"),
                                    {"step": int(step), "upd": int(upd), **odom_metrics},
                                )
                            metrics.update(odom_metrics)

                        odom_metric_key = str(getattr(cfg, "odom_select_metric", "odom_metric_drift"))
                        odom_status_ok = (not bool(getattr(cfg, "odom_select_require_status_ok", True))) or (
                            odom_metrics.get("odom_status") == "ok"
                        )
                        odom_metric_val = float(odom_metrics.get(odom_metric_key, float("inf")))
                        odom_tmag_rel = float(tdiag.get("tmag_rel_err", float("inf")))
                        odom_tdir_ok = float(tdir_abs) <= float(getattr(cfg, "odom_select_max_tdir_abs", 25.0))
                        odom_tmag_ok = odom_tmag_rel <= float(getattr(cfg, "odom_select_max_tmag_rel", 0.9))
                        odom_metric_ok = math.isfinite(odom_metric_val)
                        odom_select_ok = odom_status_ok and odom_tdir_ok and odom_tmag_ok and odom_metric_ok
                        metrics.update({
                            "odom_select_metric": odom_metric_key,
                            "odom_select_value": odom_metric_val,
                            "odom_select_ok": int(odom_select_ok),
                            "odom_select_tdir_ok": int(odom_tdir_ok),
                            "odom_select_tmag_ok": int(odom_tmag_ok),
                            "odom_select_status_ok": int(odom_status_ok),
                            "odom_select_tmag_rel_err": odom_tmag_rel,
                        })
                        if (
                            bool(getattr(cfg, "save_best_odom_checkpoint", True))
                            and odom_select_ok
                            and odom_metric_val < best_odom_metric
                        ):
                            best_odom_metric = odom_metric_val
                            best_odom_drift = float(odom_metrics.get("odom_metric_drift", float("nan")))
                            best_odom_upd = int(upd)
                            best_odom_tdir_abs = float(tdir_abs)
                            best_odom_tmag_rel_err = float(odom_tmag_rel)
                            best_odom_checkpoint = "best_odom_drift.pt"
                            _save_model_ckpt(os.path.join(ckpt_root, best_odom_checkpoint), model, cfg, step, upd, metrics)
                            print(
                                f"[OdomSelect] saved {best_odom_checkpoint} | "
                                f"{odom_metric_key}={best_odom_metric:.4f} | "
                                f"tdir_abs={best_odom_tdir_abs:.4f}° | "
                                f"tmag_rel={best_odom_tmag_rel_err:.4f} | upd={best_odom_upd:05d}"
                            )

                        smallk_labels = [
                            f"k={int(k)}" for k in tuple(getattr(cfg, "smallk_select_k_list", (1, 2, 3)))
                        ]
                        smallk_tdir_abs, smallk_count = _bucket_weighted_mean(bucket_k, smallk_labels, "tdir_abs")
                        smallk_tmag_rel, _ = _bucket_weighted_mean(bucket_k, smallk_labels, "tmag_rel_err")
                        smallk_metric_key = str(getattr(cfg, "smallk_select_metric", "odom_metric_drift"))
                        smallk_metric_map = {
                            **odom_metrics,
                            "smallk_tdir_abs": smallk_tdir_abs,
                            "smallk_tmag_rel_err": smallk_tmag_rel,
                        }
                        smallk_metric_val = float(smallk_metric_map.get(smallk_metric_key, float("inf")))
                        smallk_status_ok = (not bool(getattr(cfg, "smallk_select_require_status_ok", True))) or (
                            odom_metrics.get("odom_status") == "ok"
                        )
                        smallk_metric_ok = math.isfinite(smallk_metric_val)
                        smallk_count_ok = int(smallk_count) > 0
                        smallk_tdir_ok = (
                            math.isfinite(float(smallk_tdir_abs))
                            and float(smallk_tdir_abs) <= float(getattr(cfg, "smallk_select_max_tdir_abs", 28.0))
                        )
                        smallk_tmag_ok = (
                            math.isfinite(float(smallk_tmag_rel))
                            and float(smallk_tmag_rel) <= float(getattr(cfg, "smallk_select_max_tmag_rel", 0.95))
                        )
                        smallk_select_ok = (
                            smallk_status_ok
                            and smallk_metric_ok
                            and smallk_count_ok
                            and smallk_tdir_ok
                            and smallk_tmag_ok
                        )
                        metrics.update({
                            "smallk_select_metric": smallk_metric_key,
                            "smallk_select_value": smallk_metric_val,
                            "smallk_select_ok": int(smallk_select_ok),
                            "smallk_select_status_ok": int(smallk_status_ok),
                            "smallk_select_count_ok": int(smallk_count_ok),
                            "smallk_select_tdir_ok": int(smallk_tdir_ok),
                            "smallk_select_tmag_ok": int(smallk_tmag_ok),
                            "smallk_select_tdir_abs": float(smallk_tdir_abs),
                            "smallk_select_tmag_rel_err": float(smallk_tmag_rel),
                            "smallk_select_count": int(smallk_count),
                        })
                        if (
                            bool(getattr(cfg, "save_best_smallk_odom_checkpoint", True))
                            and smallk_select_ok
                            and smallk_metric_val < best_smallk_odom_metric
                        ):
                            best_smallk_odom_metric = smallk_metric_val
                            best_smallk_odom_drift = float(odom_metrics.get("odom_metric_drift", float("nan")))
                            best_smallk_odom_upd = int(upd)
                            best_smallk_odom_tdir_abs = float(smallk_tdir_abs)
                            best_smallk_odom_tmag_rel_err = float(smallk_tmag_rel)
                            best_smallk_odom_count = int(smallk_count)
                            best_smallk_odom_checkpoint = "best_smallk_odom.pt"
                            _save_model_ckpt(os.path.join(ckpt_root, best_smallk_odom_checkpoint), model, cfg, step, upd, metrics)
                            print(
                                f"[SmallKOdomSelect] saved {best_smallk_odom_checkpoint} | "
                                f"{smallk_metric_key}={best_smallk_odom_metric:.4f} | "
                                f"k={tuple(getattr(cfg, 'smallk_select_k_list', (1, 2, 3)))} "
                                f"tdir_abs={best_smallk_odom_tdir_abs:.4f}° | "
                                f"tmag_rel={best_smallk_odom_tmag_rel_err:.4f} | "
                                f"count={best_smallk_odom_count} | upd={best_smallk_odom_upd:05d}"
                            )

                        if bool(getattr(cfg, "save_last_eval_checkpoint", False)):
                            _save_ckpt(os.path.join(ckpt_root, "last_eval.pt"), model, optimizer, scaler, scheduler, cfg, step, upd, metrics)
                        if rot < best_rot:
                            best_rot = rot
                            if bool(getattr(cfg, "save_metric_checkpoints", False)):
                                _save_model_ckpt(os.path.join(ckpt_root, "best_rot.pt"), model, cfg, step, upd, metrics)
                        if tdir < best_tdir_raw:
                            best_tdir_raw = tdir
                            if bool(getattr(cfg, "save_metric_checkpoints", False)):
                                _save_model_ckpt(os.path.join(ckpt_root, "best_tdir_raw.pt"), model, cfg, step, upd, metrics)
                        if tdir_abs < best_tdir_abs:
                            best_tdir_abs = tdir_abs
                            if bool(getattr(cfg, "save_metric_checkpoints", False)):
                                _save_model_ckpt(os.path.join(ckpt_root, "best_tdir_abs.pt"), model, cfg, step, upd, metrics)
                        if tdir_local_A < best_tdir_local_A:
                            best_tdir_local_A = tdir_local_A
                            if bool(getattr(cfg, "save_metric_checkpoints", False)):
                                _save_model_ckpt(os.path.join(ckpt_root, "best_tdir_local_A.pt"), model, cfg, step, upd, metrics)
                        if tdir_local_A_abs < best_tdir_local_A_abs:
                            best_tdir_local_A_abs = tdir_local_A_abs
                            if bool(getattr(cfg, "save_metric_checkpoints", False)):
                                _save_model_ckpt(os.path.join(ckpt_root, "best_tdir_local_A_abs.pt"), model, cfg, step, upd, metrics)
                        if joint_eligible and joint_score < best_joint:
                            best_joint = joint_score
                            if bool(getattr(cfg, "save_best_joint_checkpoint", True)):
                                _save_model_ckpt(os.path.join(ckpt_root, "best_joint.pt"), model, cfg, step, upd, metrics)
                        if joint_local_A_abs_eligible and joint_local_A_abs_score < best_joint_local_A_abs:
                            best_joint_local_A_abs = joint_local_A_abs_score
                            if bool(getattr(cfg, "save_best_local_joint_checkpoint", True)):
                                _save_model_ckpt(os.path.join(ckpt_root, "best_joint_local_A_abs.pt"), model, cfg, step, upd, metrics)

                        eval_rec = {
                            "step": int(step),
                            "upd": int(upd),
                            "lr": float(optimizer.param_groups[0]["lr"]),

                            # test
                            "rot": float(rot),
                            "tdir": float(tdir),
                            "tdir_abs": float(tdir_abs),
                            "tdir_local_A": float(tdir_local_A),
                            "tdir_local_A_abs": float(tdir_local_A_abs),
                            "joint_score": float(joint_score),
                            "joint_local_A_abs_score": float(joint_local_A_abs_score),
                            "joint_eligible": int(joint_eligible),
                            "joint_local_A_abs_eligible": int(joint_local_A_abs_eligible),
                            "bucket_k": bucket_k,
                            "bucket_dt": bucket_dt,
                            "bucket_k_dt": bucket_k_dt,

                            # train-eval
                            "train_rot": float(train_rot),
                            "train_tdir": float(train_tdir),
                            "train_tdir_abs": float(train_tdir_abs),
                            "train_tdir_local_A": float(train_tdir_local_A),
                            "train_tdir_local_A_abs": float(train_tdir_local_A_abs),
                            "train_bucket_k": train_bucket_k,
                            "train_bucket_dt": train_bucket_dt,
                            "train_bucket_k_dt": train_bucket_k_dt,

                            "bad_forward": int(bad_forward),
                            "skip_updates": int(skip_updates),
                            **odom_metrics,
                            **{k: float(v) for k, v in tdiag.items()},
                            **{f"train_{k}": float(v) for k, v in train_tdiag.items()},
                        }
                        eval_history.append(eval_rec)
                        if bool(cfg.save_eval_history):
                            _save_json(os.path.join(ckpt_root, "eval_history.json"), {"history": eval_history})
                        if bool(getattr(cfg, 'save_eval_bucket_history', False)) or bool(getattr(cfg, 'save_eval_buckets_latest', True)):
                            bucket_payload = {
                                'step': int(step),
                                'upd': int(upd),
                                'bucket_k': bucket_k,
                                'bucket_dt': bucket_dt,
                                'bucket_k_dt': bucket_k_dt,
                                'train_bucket_k': train_bucket_k,
                                'train_bucket_dt': train_bucket_dt,
                                'train_bucket_k_dt': train_bucket_k_dt,
                            }
                            if bool(getattr(cfg, 'save_eval_bucket_history', False)):
                                _save_json(os.path.join(ckpt_root, f"eval_buckets_upd{upd:05d}.json"), bucket_payload)
                            if bool(getattr(cfg, 'save_eval_buckets_latest', True)):
                                _save_json(os.path.join(ckpt_root, 'eval_buckets_latest.json'), bucket_payload)
                                _write_eval_buckets_csv(os.path.join(ckpt_root, 'eval_buckets_latest.csv'), bucket_payload)
                                _save_json(
                                    os.path.join(ckpt_root, 'matching_diag_latest.json'),
                                    _matching_diag_payload(step, upd, tdiag, train_tdiag),
                                )

                        if want_vis and vis_payload is not None:
                            tag = f"upd{upd:05d}"
                            if bool(getattr(cfg, "save_vis_payload_npz", False)):
                                _save_vis_payload_npz(os.path.join(vis_root, f"vis_example_{tag}.npz"), vis_payload)
                                _save_vis_payload_npz(os.path.join(vis_root, "vis_example_latest.npz"), vis_payload)
                            if bool(getattr(cfg, "save_vis_diag_json", True)):
                                _save_json(os.path.join(vis_root, f"vis_diag_{tag}.json"), vis_payload.get("diag", {}))
                                _save_json(os.path.join(vis_root, "vis_diag_latest.json"), vis_payload.get("diag", {}))
                            _plot_softcorr_overview(vis_payload, os.path.join(vis_root, f"vis_softcorr_{tag}.png"))
                            _plot_softcorr_overview(vis_payload, os.path.join(vis_root, "vis_softcorr_latest.png"))
                            _plot_depth_overview(vis_payload, os.path.join(vis_root, f"vis_depth_{tag}.png"))
                            _plot_depth_overview(vis_payload, os.path.join(vis_root, "vis_depth_latest.png"))
                            vis_dumped = True

        t_opt = time.perf_counter()

        if step % cfg.log_every == 0:
            with torch.no_grad():
                p0_mean = float(next(model.parameters()).mean().detach().cpu())
            scaler_scale = float(scaler.get_scale()) if use_scaler else 1.0
            lr_now = optimizer.param_groups[0]["lr"]
            dt = time.perf_counter() - t0
            ep = step // max(len(train_loader), 1)
            print(
                f"[Train] step {step:05d} upd {upd:05d} ep{ep:03d} | "
                f"L={float(L.detach().cpu()):.3f} | pose={float(L_pose.detach().cpu()):.3f} | "
                f"pose_c={float(L_pose_coarse.detach().cpu()):.3f} | "
                f"tmag={float(L_t_mag.detach().cpu()):.4f} | tmag_w={tmag_w_eff:.4g} | "
                f"tdir_anchor={float(L_tdir_anchor.detach().cpu()):.4f} | anchor_w={tdir_anchor_w_eff:.4g} | anchor_n={tdir_anchor_n:.0f} | "
                f"x={float(L_x.detach().cpu()):.4f} | cyc={float(L_cyc.detach().cpu()):.4f} | "
                f"rel={float(L_rel.detach().cpu()):.4f} | epi={float(L_epi.detach().cpu()):.4f} | epi_c={float(L_epi_coarse.detach().cpu()):.4f} | "
                f"photo={float(L_photo.detach().cpu()):.4f} | smooth={float(L_smooth.detach().cpu()):.4f} | depth_ramp={depth_ramp:.2f} | "
                f"lr={lr_now:.6e}"
            )
            print(
                f"[Time ] avg/iter={dt*1000.0:.1f}ms | data={(t_data-t0)*1000.0:.1f}ms "
                f"fwd={(t_fwd-t_data)*1000.0:.1f}ms bwd={(t_bwd-t_fwd)*1000.0:.1f}ms opt={(t_opt-t_bwd)*1000.0:.1f}ms"
            )
            print(
                f"[Stat ] grad_norm={last_grad_norm:.6f} | scaler_scale={scaler_scale} | "
                f"skip_updates={skip_updates} | bad_forward={bad_forward} | bad_grad={last_bad_grad_name} | p0_mean={p0_mean:.6e}"
            )

            if step == 0:
                for k in ["Wc_ab", "Wf_ab", "bearingA_c", "bearingA_f", "routing_mask", "allowed_mask", "Wf_ab_raw", "epi_residual", "inv_depth_s16", "Iwarp_s", "warp_valid_s", "photo_err_s"]:
                    if isinstance(aux, dict) and k in aux and aux[k] is not None:
                        print(f"[DBG] {k}: {tuple(aux[k].shape)}")
                print(f"[Shapes] IA {tuple(IA.shape)} | R {tuple(R_gt.shape)} | t {tuple(t_gt.shape)} | stage={aux.get('stage', '-')} | t_local={aux.get('t_local_frame', '-')} -> t_out={aux.get('t_output_frame', '-')}")
                print("-" * 80)

        step += 1

    final_metrics = {
        "best_rot": best_rot,
        "best_tdir_raw": best_tdir_raw,
        "best_tdir_abs": best_tdir_abs,
        "best_tdir_local_A": best_tdir_local_A,
        "best_tdir_local_A_abs": best_tdir_local_A_abs,
        "best_joint": best_joint,
        "best_joint_local_A_abs": best_joint_local_A_abs,
        "best_odom_metric": best_odom_metric,
        "best_odom_drift": best_odom_drift,
        "best_odom_upd": best_odom_upd,
        "best_odom_tdir_abs": best_odom_tdir_abs,
        "best_odom_tmag_rel_err": best_odom_tmag_rel_err,
        "best_odom_checkpoint": best_odom_checkpoint,
        "best_smallk_odom_metric": best_smallk_odom_metric,
        "best_smallk_odom_drift": best_smallk_odom_drift,
        "best_smallk_odom_upd": best_smallk_odom_upd,
        "best_smallk_odom_tdir_abs": best_smallk_odom_tdir_abs,
        "best_smallk_odom_tmag_rel_err": best_smallk_odom_tmag_rel_err,
        "best_smallk_odom_count": best_smallk_odom_count,
        "best_smallk_odom_checkpoint": best_smallk_odom_checkpoint,
    }
    if bool(getattr(cfg, "save_last_train_state", True)):
        _save_ckpt(
            os.path.join(ckpt_root, "last_train_state.pt"),
            model,
            optimizer,
            scaler,
            scheduler,
            cfg,
            step,
            upd,
            final_metrics,
        )

    nan_like_events = int(bad_forward + skip_updates)
    latest_eval = eval_history[-1] if len(eval_history) > 0 else {}
    latest_eval_metrics = _latest_eval_metrics(latest_eval)
    final_summary = {
        **final_metrics,
        "bad_forward": int(bad_forward),
        "skip_updates": int(skip_updates),
        "total_steps": int(step),
        "total_updates": int(upd),
        "nan_like_event_rate_per_step": float(nan_like_events / max(step, 1)),
        "nan_like_event_rate_per_update": float(nan_like_events / max(upd, 1)),
        "selection_method": "best_joint",
        "eval_points": len(eval_history),
        "tmag_base_weight": float(_cfg_tmag_weight(cfg)),
        "tmag_start_updates": int(getattr(cfg, "tmag_start_updates", 0)),
        "tmag_ramp_updates": int(getattr(cfg, "tmag_ramp_updates", 0)),
        "tmag_detach_features": bool(getattr(cfg, "tmag_detach_features", False)),
        "init_checkpoint": str(getattr(cfg, "init_checkpoint", "")),
        "strict_load_checkpoint": bool(getattr(cfg, "strict_load_checkpoint", False)),
        "use_tdir_anchor_loss": bool(getattr(cfg, "use_tdir_anchor_loss", False)),
        "tdir_anchor_checkpoint": str(getattr(cfg, "tdir_anchor_checkpoint", "")),
        "w_tdir_anchor": float(getattr(cfg, "w_tdir_anchor", 0.0)),
        "tdir_anchor_min_dt": float(getattr(cfg, "tdir_anchor_min_dt", 0.2)),
        "tdir_anchor_min_k": int(getattr(cfg, "tdir_anchor_min_k", 0)),
        "tdir_anchor_start_updates": int(getattr(cfg, "tdir_anchor_start_updates", 0)),
        "tdir_anchor_ramp_updates": int(getattr(cfg, "tdir_anchor_ramp_updates", 0)),
        "tdir_loss_ignore_dt_below": float(getattr(cfg, "tdir_loss_ignore_dt_below", 0.0)),
        "tdir_loss_ignore_weight": float(getattr(cfg, "tdir_loss_ignore_weight", 0.0)),
        "save_best_odom_checkpoint": bool(getattr(cfg, "save_best_odom_checkpoint", True)),
        "odom_select_metric": str(getattr(cfg, "odom_select_metric", "odom_metric_drift")),
        "odom_select_max_tdir_abs": float(getattr(cfg, "odom_select_max_tdir_abs", 25.0)),
        "odom_select_max_tmag_rel": float(getattr(cfg, "odom_select_max_tmag_rel", 0.9)),
        "odom_select_require_status_ok": bool(getattr(cfg, "odom_select_require_status_ok", True)),
        "save_best_smallk_odom_checkpoint": bool(getattr(cfg, "save_best_smallk_odom_checkpoint", True)),
        "smallk_select_metric": str(getattr(cfg, "smallk_select_metric", "odom_metric_drift")),
        "smallk_select_k_list": list(tuple(getattr(cfg, "smallk_select_k_list", (1, 2, 3)))),
        "smallk_select_max_tdir_abs": float(getattr(cfg, "smallk_select_max_tdir_abs", 28.0)),
        "smallk_select_max_tmag_rel": float(getattr(cfg, "smallk_select_max_tmag_rel", 0.95)),
        "smallk_select_require_status_ok": bool(getattr(cfg, "smallk_select_require_status_ok", True)),
        "last_eval": latest_eval_metrics,
    }
    if bool(cfg.save_final_summary):
        _save_json(os.path.join(ckpt_root, "final_summary.json"), final_summary)
    if bool(cfg.save_eval_history):
        _save_json(os.path.join(ckpt_root, "eval_history.json"), {"history": eval_history})
    if bool(cfg.plot_curves_after_train) and len(eval_history) > 0:
        _plot_eval_curves(eval_history, os.path.join(ckpt_root, "val_curves.png"))

    print(
        f"[Done ] training finished | best_rot={best_rot:.4f}° | "
        f"best_tdir_raw={best_tdir_raw:.4f}° | best_tdir_abs={best_tdir_abs:.4f}° | "
        f"best_tdir_local_A={best_tdir_local_A:.4f}° | best_tdir_local_A_abs={best_tdir_local_A_abs:.4f}° | "
        f"best_joint={best_joint:.4f} | best_joint_local_A_abs={best_joint_local_A_abs:.4f} | ckpt_dir={ckpt_root}"
    )


if __name__ == "__main__":
    main()
