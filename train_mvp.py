import json
import math
import os
import random
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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
)
from model import PanoramaRelPoseModel
from erp_sampling import warp_erp_with_depth_pose
from pose_head import matrix_geodesic_distance


def _cfg_to_dict(cfg):
    try:
        return asdict(cfg)
    except Exception:
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}


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




def _translation_weight_from_dt(meta: Any, bsz: int, *, small_dt_thresh: float, small_dt_t_weight: float):
    dt_list = _meta_batch_field(meta, 'dt_world', bsz, default=None)
    w = []
    for x in dt_list:
        try:
            if x is not None and float(x) < float(small_dt_thresh):
                w.append(float(small_dt_t_weight))
            else:
                w.append(1.0)
        except Exception:
            w.append(1.0)
    return w

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
        }
    return out


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
            f"epi_mass={rec['epi_mass_in_gt_band']:.3f} top1={rec['top1_in_gt_band']:.3f}"
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

    return {
        "epi_mass_in_gt_band": epi_mass.detach(),
        "top1_in_gt_band": top1.detach(),
        "top5_in_gt_band": top5.detach(),
        "matching_entropy": entropy.detach(),
        "max_matching_prob": max_prob.detach(),
        "cycle_error": cycle_error.detach(),
    }


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
    diag_sums = {
        "epi_mass_in_gt_band": 0.0,
        "top1_in_gt_band": 0.0,
        "top5_in_gt_band": 0.0,
        "matching_entropy": 0.0,
        "max_matching_prob": 0.0,
        "cycle_error": 0.0,
    }
    n = 0
    n_local = 0
    vis_payload = None

    bucket_k = _bucket_init()
    bucket_dt = _bucket_init()
    bucket_k_dt = _bucket_init()
    dt_edges = tuple(float(x) for x in getattr(cfg, 'eval_dt_bucket_edges', (0.2, 0.5, 1.0, 2.0)))

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
        n += bsz

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
            ang_local = torch.acos(torch.sum(tp_local * tg_Rt, dim=-1).clamp(-1.0, 1.0)) * (180.0 / math.pi)
            ang_local_abs = torch.minimum(ang_local, 180.0 - ang_local)
            tdir_local_A_sum += float(ang_local.sum().cpu())
            tdir_local_A_abs_sum += float(ang_local_abs.sum().cpu())
            n_local += bsz
            tdir_local_np = ang_local.detach().cpu().view(-1).numpy()
            tdir_local_abs_np = ang_local_abs.detach().cpu().view(-1).numpy()
        else:
            tdir_local_np = np.full((bsz,), np.nan, dtype=np.float32)
            tdir_local_abs_np = np.full((bsz,), np.nan, dtype=np.float32)

        meta_k = _meta_batch_field(meta, 'k', bsz, default=None)
        meta_dt = _meta_batch_field(meta, 'dt_world', bsz, default=None)
        for i in range(bsz):
            k_label = f"k={int(meta_k[i])}" if meta_k[i] is not None else 'k=unknown'
            dt_label = _dt_bucket_label(meta_dt[i], dt_edges)
            _bucket_update(
                bucket_k, k_label, rot_np[i], tdir_np[i], tdir_abs_np[i], tdir_local_np[i], tdir_local_abs_np[i],
                diag_np["epi_mass_in_gt_band"][i], diag_np["top1_in_gt_band"][i], diag_np["top5_in_gt_band"][i],
                diag_np["matching_entropy"][i], diag_np["max_matching_prob"][i], diag_np["cycle_error"][i]
            )
            _bucket_update(
                bucket_dt, dt_label, rot_np[i], tdir_np[i], tdir_abs_np[i], tdir_local_np[i], tdir_local_abs_np[i],
                diag_np["epi_mass_in_gt_band"][i], diag_np["top1_in_gt_band"][i], diag_np["top5_in_gt_band"][i],
                diag_np["matching_entropy"][i], diag_np["max_matching_prob"][i], diag_np["cycle_error"][i]
            )
            _bucket_update(
                bucket_k_dt, f"{k_label}|{dt_label}", rot_np[i], tdir_np[i], tdir_abs_np[i], tdir_local_np[i], tdir_local_abs_np[i],
                diag_np["epi_mass_in_gt_band"][i], diag_np["top1_in_gt_band"][i], diag_np["top5_in_gt_band"][i],
                diag_np["matching_entropy"][i], diag_np["max_matching_prob"][i], diag_np["cycle_error"][i]
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


def main():
    cfg = Config()
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
        f"[Cfg ] lr={cfg.lr} | wd={cfg.wd} | warmup_updates={cfg.warmup_updates} | "
        f"hold<{cfg.lr_hold_updates} | drop1<{cfg.lr_drop1_updates}@{cfg.lr_drop1_scale} | drop2@{cfg.lr_drop2_scale}"
    )
    print(f"[Cfg ] split_by={cfg.split_by} | train_ratio={cfg.train_ratio} | split_seed={cfg.split_seed} | data_seed={cfg.data_seed}")
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

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        worker_init_fn=worker_init,
        generator=train_gen,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0 if bool(cfg.eval_use_fixed_pairs) else cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        worker_init_fn=worker_init if not bool(cfg.eval_use_fixed_pairs) else None,
        generator=test_gen,
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
    )

    model = PanoramaRelPoseModel(cfg, dev).to(dev)
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
        dt_t_weight = torch.tensor(
            _translation_weight_from_dt(
                meta,
                IA.shape[0],
                small_dt_thresh=float(getattr(cfg, "small_dt_thresh", 0.2)),
                small_dt_t_weight=float(getattr(cfg, "small_dt_t_weight", 0.35)),
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
                pred_t_frame=t_pose_frame,
                t_sample_weight=dt_t_weight,
            )
            if bool(cfg.use_fine_stage) and aux.get("Rc", None) is not None and aux.get("tc_dir", None) is not None:
                L_pose_coarse = pose_loss(
                    aux["Rc"],
                    aux["tc_dir"],
                    R_gt,
                    t_gt,
                    pose_t_alpha=cfg.pose_t_alpha,
                    pred_t_frame=aux.get("t_local_frame", "A"),
                    t_sample_weight=dt_t_weight,
                )
            else:
                L_pose_coarse = torch.zeros((), device=dev)

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
                + float(getattr(cfg, "w_coarse_pose_aux", 0.0)) * L_pose_coarse
                + cfg.w_x * L_x
                + cfg.w_cyc * L_cyc
                + cfg.w_rel * L_rel
                + epi_w * L_epi
                + float(getattr(cfg, "w_coarse_epi_aux", 0.0)) * epi_ramp * L_epi_coarse
                + photo_w * L_photo
                + smooth_w * L_smooth
            )
            L_scaled = L / float(cfg.grad_accum)

        t_fwd = time.perf_counter()

        forward_ok = all(
            bool(torch.isfinite(x).all())
            for x in [R_pred, t_pred, L_pose, L_pose_coarse, L_x, L_cyc, L_rel, L_epi, L_epi_coarse, L_photo, L_smooth, L]
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
                        joint_eligible = (rot < float(cfg.joint_rot_thresh_deg)) and (tdir_abs < float(cfg.joint_tdir_thresh_deg))

                        metrics = {
                            # test metrics: checkpoint selection still uses these
                            "rot": rot,
                            "tdir": tdir,
                            "tdir_abs": tdir_abs,
                            "tdir_local_A": tdir_local_A,
                            "tdir_local_A_abs": tdir_local_A_abs,
                            "joint_score": joint_score,
                            "joint_eligible": int(joint_eligible),
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
                            f"joint_abs={joint_score:.4f} | joint_ok={int(joint_eligible)} | time={t_eval:.2f}s"
                        )
                        print(tdir_msg)
                        print(tdir_local_msg)
                        for _bucket_msg in bucket_msgs:
                            print(_bucket_msg)

                        _save_ckpt(os.path.join(ckpt_root, "last_eval.pt"), model, optimizer, scaler, scheduler, cfg, step, upd, metrics)
                        if rot < best_rot:
                            best_rot = rot
                            _save_ckpt(os.path.join(ckpt_root, "best_rot.pt"), model, optimizer, scaler, scheduler, cfg, step, upd, metrics)
                        if tdir < best_tdir_raw:
                            best_tdir_raw = tdir
                            _save_ckpt(os.path.join(ckpt_root, "best_tdir_raw.pt"), model, optimizer, scaler, scheduler, cfg, step, upd, metrics)
                        if tdir_abs < best_tdir_abs:
                            best_tdir_abs = tdir_abs
                            _save_ckpt(os.path.join(ckpt_root, "best_tdir_abs.pt"), model, optimizer, scaler, scheduler, cfg, step, upd, metrics)
                        if tdir_local_A < best_tdir_local_A:
                            best_tdir_local_A = tdir_local_A
                            _save_ckpt(os.path.join(ckpt_root, "best_tdir_local_A.pt"), model, optimizer, scaler, scheduler, cfg, step, upd, metrics)
                        if tdir_local_A_abs < best_tdir_local_A_abs:
                            best_tdir_local_A_abs = tdir_local_A_abs
                            _save_ckpt(os.path.join(ckpt_root, "best_tdir_local_A_abs.pt"), model, optimizer, scaler, scheduler, cfg, step, upd, metrics)
                        if joint_eligible and joint_score < best_joint:
                            best_joint = joint_score
                            _save_ckpt(os.path.join(ckpt_root, "best_joint.pt"), model, optimizer, scaler, scheduler, cfg, step, upd, metrics)

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
                            "joint_eligible": int(joint_eligible),
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
                            **{k: float(v) for k, v in tdiag.items()},
                            **{f"train_{k}": float(v) for k, v in train_tdiag.items()},
                        }
                        eval_history.append(eval_rec)
                        if bool(cfg.save_eval_history):
                            _save_json(os.path.join(ckpt_root, "eval_history.json"), {"history": eval_history})
                        if bool(getattr(cfg, 'save_eval_bucket_history', True)):
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
                            _save_json(os.path.join(ckpt_root, f"eval_buckets_upd{upd:05d}.json"), bucket_payload)
                            _save_json(os.path.join(ckpt_root, 'eval_buckets_latest.json'), bucket_payload)

                        if want_vis and vis_payload is not None:
                            tag = f"upd{upd:05d}"
                            _save_vis_payload_npz(os.path.join(vis_root, f"vis_example_{tag}.npz"), vis_payload)
                            _save_vis_payload_npz(os.path.join(vis_root, "vis_example_latest.npz"), vis_payload)
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
    }
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
        f"best_joint={best_joint:.4f} | ckpt_dir={ckpt_root}"
    )


if __name__ == "__main__":
    main()
