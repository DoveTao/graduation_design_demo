from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import torch
from losses import translation_direction_loss, translation_magnitude_loss
from pose_head import matrix_geodesic_distance


def build_k_step_weights(
    k_tensor: torch.Tensor,
    *,
    weight_k1: float = 1.0,
    weight_k2: float = 1.0,
    weight_k3: float = 0.9,
    weight_k5: float = 0.8,
    mode: str = "explicit",
    adjacent_weight: Optional[float] = None,
    non_adjacent_weight: Optional[float] = None,
    decay: str = "none",
    min_weight: float = 0.1,
) -> torch.Tensor:
    k = k_tensor.view(-1).to(dtype=torch.int64)
    if str(mode).lower() == "adjacent_vs_nonadjacent":
        adj = 1.0 if adjacent_weight is None else float(adjacent_weight)
        nonadj = 1.0 if non_adjacent_weight is None else float(non_adjacent_weight)
        out = torch.where(k == 1, torch.full_like(k, adj, dtype=torch.float32), torch.full_like(k, nonadj, dtype=torch.float32))
        decay_key = str(decay).lower()
        kf = k.to(dtype=torch.float32)
        if decay_key == "sqrt":
            factor = torch.where(k == 1, torch.ones_like(kf), 1.0 / torch.sqrt(kf))
            out = torch.where(k == 1, out, out * factor)
        elif decay_key == "log":
            factor = torch.where(k == 1, torch.ones_like(kf), 1.0 / torch.log2(kf + 1.0))
            out = torch.where(k == 1, out, out * factor)
        elif decay_key == "clipped_linear":
            factor = torch.where(k == 1, torch.ones_like(kf), torch.clamp(1.25 - 0.1 * (kf - 1.0), min=float(min_weight), max=1.0))
            out = torch.where(k == 1, out, out * factor)
        return out.clamp_min(float(min_weight))
    out = torch.ones_like(k, dtype=torch.float32)
    out = torch.where(k == 1, torch.full_like(out, float(weight_k1)), out)
    out = torch.where(k == 2, torch.full_like(out, float(weight_k2)), out)
    out = torch.where(k == 3, torch.full_like(out, float(weight_k3)), out)
    out = torch.where(k == 5, torch.full_like(out, float(weight_k5)), out)
    return out


def rotation_geodesic_loss(
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    *,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    rot = matrix_geodesic_distance(R_pred.float(), R_gt.float())
    if sample_weight is None:
        return rot.mean()
    w = sample_weight.float().view(-1).to(rot.device).clamp_min(1.0e-6)
    return (rot.view(-1) * w).sum() / w.sum().clamp_min(1.0e-6)


def build_observability_weight(
    *,
    tmag_gt: torch.Tensor,
    k_tensor: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    eps = float(config.get("tmag_epsilon", 1.0e-6))
    near_zero_thresh = float(config.get("near_zero_tmag", 0.02))
    low_thresh = float(config.get("low_tmag", 0.05))
    medium_thresh = float(config.get("medium_tmag", 0.15))
    high_thresh = float(config.get("high_tmag", 0.4))
    min_weight = float(config.get("min_weight", 0.2))
    max_weight = float(config.get("max_weight", 2.0))
    near_zero_weight = float(config.get("near_zero_weight", min_weight))
    low_weight = float(config.get("low_weight", max(min_weight, 0.55)))
    medium_weight = float(config.get("medium_weight", config.get("moderate_baseline_boost", 1.15)))
    high_weight = float(config.get("high_weight", max(float(config.get("moderate_baseline_boost", 1.15)), 1.35)))
    very_high_weight = float(config.get("very_high_weight", 1.05))

    tmag = tmag_gt.float().view(-1).clamp_min(eps)
    k = k_tensor.view(-1).to(dtype=torch.int64, device=tmag.device)
    w = torch.ones_like(tmag, dtype=torch.float32)

    w = torch.where(tmag <= near_zero_thresh, torch.full_like(w, near_zero_weight), w)
    w = torch.where((tmag > near_zero_thresh) & (tmag <= low_thresh), torch.full_like(w, low_weight), w)
    w = torch.where((tmag > low_thresh) & (tmag <= medium_thresh), torch.full_like(w, medium_weight), w)
    w = torch.where((tmag > medium_thresh) & (tmag <= high_thresh), torch.full_like(w, high_weight), w)
    w = torch.where(tmag > high_thresh, torch.full_like(w, very_high_weight), w)

    k_decay = {
        1: float(config.get("k1_factor", 1.05)),
        2: float(config.get("k2_factor", 1.0)),
        3: float(config.get("k3_factor", 0.95)),
        5: float(config.get("k5_factor", 0.85)),
    }
    for kk, factor in k_decay.items():
        w = torch.where(k == int(kk), w * float(factor), w)
    return w.clamp(min=min_weight, max=max_weight)


def scale_stability_loss(
    tmag_pred: torch.Tensor,
    tmag_gt: torch.Tensor,
    *,
    sample_weight: Optional[torch.Tensor] = None,
    eps: float = 1.0e-6,
    collapse_ratio: float = 0.1,
    explosion_ratio: float = 10.0,
    mean_log_bias_weight: float = 1.0,
) -> torch.Tensor:
    pred = tmag_pred.float().view(-1).clamp_min(float(eps))
    gt = tmag_gt.float().view(-1).clamp_min(float(eps))
    log_ratio = torch.log(pred) - torch.log(gt)
    low = math.log(max(float(collapse_ratio), float(eps)))
    high = math.log(max(float(explosion_ratio), 1.0 + float(eps)))
    collapse_pen = torch.relu(torch.full_like(log_ratio, low) - log_ratio)
    explosion_pen = torch.relu(log_ratio - torch.full_like(log_ratio, high))
    sample_term = collapse_pen + explosion_pen

    if sample_weight is not None:
        w = sample_weight.float().view(-1).to(sample_term.device).clamp_min(1.0e-6)
        sample_term = (sample_term * w).sum() / w.sum().clamp_min(1.0e-6)
        mean_log_ratio = (log_ratio * w).sum() / w.sum().clamp_min(1.0e-6)
    else:
        sample_term = sample_term.mean()
        mean_log_ratio = log_ratio.mean()
    return sample_term + float(mean_log_bias_weight) * mean_log_ratio.pow(2)


def train360d_pose_loss(
    *,
    R_pred: torch.Tensor,
    tdir_pred_B: torch.Tensor,
    tmag_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt_vec_B: torch.Tensor,
    tmag_gt: torch.Tensor,
    k_tensor: torch.Tensor,
    k_step_config: Mapping[str, Any],
    observability_config: Mapping[str, Any],
    scale_config: Mapping[str, Any],
    rot_weight: float = 1.0,
    tdir_weight: float = 1.0,
    tmag_weight: float = 0.5,
    scale_stability_weight: float = 0.1,
    tmag_loss_type: str = "log_smooth_l1",
    tmag_epsilon: float = 1.0e-6,
    enable_observability: bool = True,
    enable_k_step_balancing: bool = True,
    enable_scale_stabilization: bool = True,
) -> Dict[str, torch.Tensor]:
    k_weights = build_k_step_weights(
        k_tensor,
        weight_k1=float(k_step_config.get("weight_k1", 1.0)),
        weight_k2=float(k_step_config.get("weight_k2", 1.0)),
        weight_k3=float(k_step_config.get("weight_k3", 0.9)),
        weight_k5=float(k_step_config.get("weight_k5", 0.8)),
        mode=str(k_step_config.get("mode", "explicit")),
        adjacent_weight=k_step_config.get("adjacent_weight"),
        non_adjacent_weight=k_step_config.get("non_adjacent_weight"),
        decay=str(k_step_config.get("decay", "none")),
        min_weight=float(k_step_config.get("min_weight", 0.1)),
    ).to(R_pred.device)
    obs_weight = build_observability_weight(
        tmag_gt=tmag_gt,
        k_tensor=k_tensor,
        config=observability_config,
    ).to(R_pred.device)
    if not bool(enable_k_step_balancing):
        k_weights = torch.ones_like(k_weights)
    if not bool(enable_observability):
        obs_weight = torch.ones_like(obs_weight)
    pose_weight = k_weights
    translation_weight = k_weights * obs_weight

    loss_rot = rotation_geodesic_loss(
        R_pred,
        R_gt,
        sample_weight=pose_weight,
    )
    loss_tdir = translation_direction_loss(
        tdir_pred_B,
        t_gt_vec_B,
        R_gt,
        pred_t_frame="B",
        oriented_weight=1.0,
        axis_weight=0.0,
        sample_weight=translation_weight,
    )
    loss_tmag = translation_magnitude_loss(
        tmag_pred,
        tmag_gt,
        loss_type=tmag_loss_type,
        eps=tmag_epsilon,
        sample_weight=translation_weight,
    )
    loss_scale_stability = scale_stability_loss(
        tmag_pred,
        tmag_gt,
        sample_weight=translation_weight,
        eps=tmag_epsilon,
        collapse_ratio=float(scale_config.get("collapse_ratio", 0.1)),
        explosion_ratio=float(scale_config.get("explosion_ratio", 10.0)),
        mean_log_bias_weight=float(scale_config.get("mean_log_bias_weight", 1.0)),
    )
    if not bool(enable_scale_stabilization):
        loss_scale_stability = loss_scale_stability * 0.0
    total = (
        float(rot_weight) * loss_rot
        + float(tdir_weight) * loss_tdir
        + float(tmag_weight) * loss_tmag
        + float(scale_stability_weight) * loss_scale_stability
    )
    return {
        "loss_total": total,
        "loss_rot": loss_rot,
        "loss_tdir": loss_tdir,
        "loss_tmag": loss_tmag,
        "loss_scale_stability": loss_scale_stability,
        "k_weights": k_weights.detach(),
        "obs_weights": obs_weight.detach(),
        "translation_weights": translation_weight.detach(),
    }
