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
from losses import depth_smoothness_loss, epipolar_simplified_loss, erp_photometric_loss, pose_loss
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
    n = 0
    n_local = 0
    vis_payload = None

    acc = {
        "raw": 0.0,
        "flip": 0.0,
        "R@t": 0.0,
        "R@(-t)": 0.0,
        "Rt@t": 0.0,
        "Rt@(-t)": 0.0,
        "cnt": 0,
    }
    variant_desc = {
        "raw": "output frame B, baseline B->A",
        "flip": "output frame B, baseline A->B",
        "R@t": "compare to R*t_gt (algebraic diagnostic only)",
        "R@(-t)": "compare to -R*t_gt",
        "Rt@t": "compare to R^T*t_gt (B->A mapped to A-local)",
        "Rt@(-t)": "compare to -R^T*t_gt",
    }

    for bi, batch in enumerate(loader):
        if cfg.max_eval_batches and bi >= cfg.max_eval_batches:
            break

        IA = batch["IA"].to(device, non_blocking=True)
        IB = batch["IB"].to(device, non_blocking=True)
        R_gt = batch["R_gt"].to(device, non_blocking=True)
        t_gt = batch["t_gt_dir"].to(device, non_blocking=True)

        R_pred, t_pred, aux = model(IA, IB, enable_depth_fusion=True)

        if collect_vis and vis_payload is None and bi == int(vis_index):
            depth_key = f"inv_depth_s{int(cfg.depth_loss_scale)}"
            if depth_key in aux and aux.get(depth_key) is not None:
                inv_depth = aux[depth_key]
                IA_small = _downsample_to_like(IA, inv_depth)
                IB_small = _downsample_to_like(IB, inv_depth)
                t_warp = aux.get("t_dir", t_pred)
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
                aux["IA_s"] = IA_small.detach()
                aux["IB_s"] = IB_small.detach()
                aux["Iwarp_s"] = Iwarp.detach()
                aux["warp_valid_s"] = valid.detach()
                aux["photo_err_s"] = (photo_err * valid.float()).detach()
            vis_payload = _extract_vis_payload(batch, aux, cfg)

        rot_rad = matrix_geodesic_distance(R_pred.float(), R_gt.float())
        rot_deg = rot_rad * (180.0 / math.pi)

        tp = F.normalize(t_pred.float(), dim=-1, eps=1e-6)
        tg = F.normalize(t_gt.float(), dim=-1, eps=1e-6)
        cos = torch.sum(tp * tg, dim=-1).clamp(-1.0, 1.0)
        ang = torch.acos(cos) * (180.0 / math.pi)
        ang_abs = torch.minimum(ang, 180.0 - ang)

        bsz = IA.shape[0]
        rot_sum += float(rot_deg.sum().cpu())
        tdir_sum += float(ang.sum().cpu())
        tdir_abs_sum += float(ang_abs.sum().cpu())
        n += bsz

        tg_R = F.normalize(torch.matmul(R_gt.float(), tg.unsqueeze(-1)).squeeze(-1), dim=-1, eps=1e-6)
        tg_Rt = F.normalize(torch.matmul(R_gt.float().transpose(-1, -2), tg.unsqueeze(-1)).squeeze(-1), dim=-1, eps=1e-6)

        t_local_pred = aux.get("t_dir_local", None) if isinstance(aux, dict) else None
        t_local_frame = aux.get("t_local_frame", None) if isinstance(aux, dict) else None
        if t_local_pred is not None and t_local_frame == "A":
            tp_local = F.normalize(t_local_pred.float(), dim=-1, eps=1e-6)
            ang_local = torch.acos(torch.sum(tp_local * tg_Rt, dim=-1).clamp(-1.0, 1.0)) * (180.0 / math.pi)
            ang_local_abs = torch.minimum(ang_local, 180.0 - ang_local)
            tdir_local_A_sum += float(ang_local.sum().cpu())
            tdir_local_A_abs_sum += float(ang_local_abs.sum().cpu())
            n_local += bsz

        def ang_deg(a, b):
            c = torch.sum(a * b, dim=-1).clamp(-1.0, 1.0)
            return torch.acos(c) * (180.0 / math.pi)

        acc["raw"] += float(ang_deg(tp, tg).sum().cpu())
        acc["flip"] += float(ang_deg(tp, -tg).sum().cpu())
        acc["R@t"] += float(ang_deg(tp, tg_R).sum().cpu())
        acc["R@(-t)"] += float(ang_deg(tp, -tg_R).sum().cpu())
        acc["Rt@t"] += float(ang_deg(tp, tg_Rt).sum().cpu())
        acc["Rt@(-t)"] += float(ang_deg(tp, -tg_Rt).sum().cpu())
        acc["cnt"] += bsz

    rot = rot_sum / max(n, 1)
    tdir = tdir_sum / max(n, 1)
    tdir_abs = tdir_abs_sum / max(n, 1)
    tdir_local_A = tdir_local_A_sum / max(n_local, 1) if n_local > 0 else float("nan")
    tdir_local_A_abs = tdir_local_A_abs_sum / max(n_local, 1) if n_local > 0 else float("nan")

    if acc["cnt"] > 0:
        denom = float(acc["cnt"])
        mean_map = {k: acc[k] / denom for k in ["raw", "flip", "R@t", "R@(-t)", "Rt@t", "Rt@(-t)"]}
        ranked = sorted(mean_map.items(), key=lambda kv: kv[1])
        best_key, best_val = ranked[0]
        gap2 = ranked[1][1] - ranked[0][1] if len(ranked) > 1 else float("nan")
        msg = (
            f"[TDIR-CHK] best={best_key} ({variant_desc[best_key]})={best_val:.2f}° | gap2={gap2:.2f}° | "
            f"raw={mean_map['raw']:.2f} flip={mean_map['flip']:.2f} "
            f"R@t={mean_map['R@t']:.2f} R@(-t)={mean_map['R@(-t)']:.2f} "
            f"Rt@t={mean_map['Rt@t']:.2f} Rt@(-t)={mean_map['Rt@(-t)']:.2f} (n={int(denom)})"
        )
    else:
        mean_map = {}
        msg = "[TDIR-CHK] n=0"

    if n_local > 0:
        msg_local = (
            f"[TDIR-LOCAL] local_A={tdir_local_A:.2f}° | local_A_abs={tdir_local_A_abs:.2f}° | "
            f"gt_local_A=R^T*t_gt | n={n_local}"
        )
    else:
        msg_local = "[TDIR-LOCAL] unavailable"

    model.train()
    return rot, tdir, tdir_abs, tdir_local_A, tdir_local_A_abs, mean_map, msg, msg_local, vis_payload


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

    train_keys = set(getattr(train_ds, "sequence_keys", []))
    test_keys = set(getattr(test_ds, "sequence_keys", []))
    overlap = train_keys & test_keys
    if overlap:
        raise RuntimeError(f"train/test split leakage detected: {len(overlap)} overlapping groups, e.g. {sorted(list(overlap))[:8]}")

    for tag, ds in [("train", train_ds), ("test", test_ds)]:
        groups = getattr(ds, "sequence_keys", [])
        print(f"[Split] {tag} by {cfg.split_by} | groups={len(groups)} | seqs={len(groups)} | pairs={len(ds)} | preview={groups[:4]}")
    print(f"[Eval ] protocol={'fixed_pairs' if bool(cfg.eval_use_fixed_pairs) else 'mixed_k_random'} | max_eval_batches={cfg.max_eval_batches}")

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
            )

            W_ab = _first_not_none(aux.get("Wf_ab", None), aux.get("Wc_ab", None))
            W_ba = _first_not_none(aux.get("Wf_ba", None), aux.get("Wc_ba", None))
            cA = _first_not_none(aux.get("cA_f", None), aux.get("cA_c", None))
            cB = _first_not_none(aux.get("cB_f", None), aux.get("cB_c", None))
            bearingA = _first_not_none(aux.get("bearingA_f", None), aux.get("bearingA_c", None))
            bearingB = _first_not_none(aux.get("bearingB_f", None), aux.get("bearingB_c", None))

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
                L_epi = epipolar_simplified_loss(
                    W_ab=W_ab,
                    W_ba=W_ba,
                    bearing_a=bearingA,
                    bearing_b=bearingB,
                    R_gt=R_gt,
                    t_gt=t_gt,
                    allowed_mask=aux.get("allowed_mask", None),
                    angle_thresh_deg=cfg.epi_angle_thresh_deg,
                    use_bidir=cfg.epi_loss_use_bidir,
                )
            else:
                L_epi = torch.zeros((), device=dev)

            depth_key = f"inv_depth_s{int(cfg.depth_loss_scale)}"
            depth_ramp = _depth_weight_ramp(upd, cfg)
            if bool(cfg.use_depth_branch) and (depth_ramp > 0.0) and depth_key in aux and cfg.w_photo > 0:
                inv_depth = aux[depth_key]
                IA_small = _downsample_to_like(IA, inv_depth)
                IB_small = _downsample_to_like(IB, inv_depth)
                R_warp = R_pred.detach() if bool(cfg.depth_use_detached_pose) else R_pred
                t_warp = aux.get("t_dir", t_pred)
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
            L = (
                cfg.w_pose * L_pose
                + cfg.w_x * L_x
                + cfg.w_cyc * L_cyc
                + cfg.w_rel * L_rel
                + cfg.w_epi * L_epi
                + photo_w * L_photo
                + smooth_w * L_smooth
            )
            L_scaled = L / float(cfg.grad_accum)

        t_fwd = time.perf_counter()

        forward_ok = all(
            bool(torch.isfinite(x).all())
            for x in [R_pred, t_pred, L_pose, L_x, L_cyc, L_rel, L_epi, L_photo, L_smooth, L]
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
                        t_eval0 = time.perf_counter()
                        want_vis = bool(cfg.save_vis_examples) and (bool(cfg.vis_dump_every_eval) or not vis_dumped)
                        rot, tdir, tdir_abs, tdir_local_A, tdir_local_A_abs, tdiag, tdir_msg, tdir_local_msg, vis_payload = eval_model(
                            model,
                            test_loader,
                            dev,
                            cfg,
                            collect_vis=want_vis,
                            vis_index=int(cfg.vis_eval_index),
                        )
                        t_eval = time.perf_counter() - t_eval0
                        joint_score = tdir + float(cfg.joint_rot_weight) * rot
                        joint_eligible = (rot < float(cfg.joint_rot_thresh_deg)) and (tdir < float(cfg.joint_tdir_thresh_deg))

                        metrics = {
                            "rot": rot,
                            "tdir": tdir,
                            "tdir_abs": tdir_abs,
                            "tdir_local_A": tdir_local_A,
                            "tdir_local_A_abs": tdir_local_A_abs,
                            "joint_score": joint_score,
                            "joint_eligible": int(joint_eligible),
                            **{f"tdir_diag_{k}": v for k, v in tdiag.items()},
                        }

                        print(
                            f"[Eval ] upd {upd:05d} (step {step:05d}) | rot={rot:.4f}° | "
                            f"tdir={tdir:.4f}° | tdir_abs={tdir_abs:.4f}° | "
                            f"tdir_local_A={tdir_local_A:.4f}° | tdir_local_A_abs={tdir_local_A_abs:.4f}° | "
                            f"joint={joint_score:.4f} | joint_ok={int(joint_eligible)} | time={t_eval:.2f}s"
                        )
                        print(tdir_msg)
                        print(tdir_local_msg)

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
                            "rot": float(rot),
                            "tdir": float(tdir),
                            "tdir_abs": float(tdir_abs),
                            "tdir_local_A": float(tdir_local_A),
                            "tdir_local_A_abs": float(tdir_local_A_abs),
                            "joint_score": float(joint_score),
                            "joint_eligible": int(joint_eligible),
                            "bad_forward": int(bad_forward),
                            "skip_updates": int(skip_updates),
                        }
                        eval_history.append(eval_rec)
                        if bool(cfg.save_eval_history):
                            _save_json(os.path.join(ckpt_root, "eval_history.json"), {"history": eval_history})

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
                f"x={float(L_x.detach().cpu()):.4f} | cyc={float(L_cyc.detach().cpu()):.4f} | "
                f"rel={float(L_rel.detach().cpu()):.4f} | epi={float(L_epi.detach().cpu()):.4f} | "
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
