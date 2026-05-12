from __future__ import annotations

from typing import Dict, Optional

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
) -> torch.Tensor:
    k = k_tensor.view(-1).to(dtype=torch.int64)
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


def train360_pose_loss(
    *,
    R_pred: torch.Tensor,
    tdir_pred_B: torch.Tensor,
    tmag_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt_vec_B: torch.Tensor,
    tmag_gt: torch.Tensor,
    k_weights: Optional[torch.Tensor] = None,
    rot_weight: float = 1.0,
    tdir_weight: float = 1.0,
    tmag_weight: float = 0.5,
    tmag_loss_type: str = "log_smooth_l1",
    tmag_epsilon: float = 1.0e-6,
) -> Dict[str, torch.Tensor]:
    loss_rot = rotation_geodesic_loss(
        R_pred,
        R_gt,
        sample_weight=k_weights,
    )
    loss_tdir = translation_direction_loss(
        tdir_pred_B,
        t_gt_vec_B,
        R_gt,
        pred_t_frame="B",
        oriented_weight=1.0,
        axis_weight=0.0,
        sample_weight=k_weights,
    )
    loss_tmag = translation_magnitude_loss(
        tmag_pred,
        tmag_gt,
        loss_type=tmag_loss_type,
        eps=tmag_epsilon,
        sample_weight=k_weights,
    )
    total = (
        float(rot_weight) * loss_rot
        + float(tdir_weight) * loss_tdir
        + float(tmag_weight) * loss_tmag
    )
    return {
        "loss_total": total,
        "loss_rot": loss_rot,
        "loss_tdir": loss_tdir,
        "loss_tmag": loss_tmag,
    }
