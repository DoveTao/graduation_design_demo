import math
from typing import Optional

import torch
import torch.nn.functional as F

from pose_head import matrix_geodesic_distance


def _normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(v.float(), dim=-1, eps=eps)


def _gt_translation_in_pred_frame(
    t_gt: torch.Tensor,
    R_gt: torch.Tensor,
    *,
    pred_t_frame: str = "B",
) -> torch.Tensor:
    pred_t_frame = str(pred_t_frame).upper()
    t_gt = _normalize(t_gt)
    R_gt = R_gt.float()

    if pred_t_frame == "B":
        return t_gt
    if pred_t_frame == "A":
        return _normalize(torch.matmul(R_gt.transpose(-1, -2), t_gt.unsqueeze(-1)).squeeze(-1))
    raise ValueError(f"Unsupported pred_t_frame={pred_t_frame!r}, expected 'A' or 'B'.")


def pose_loss(
    R_pred: torch.Tensor,
    t_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    *,
    pose_t_alpha: float = 1.0,
    pred_t_frame: str = "B",
    t_sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    R_pred = R_pred.float()
    R_gt = R_gt.float()

    t_pred = _normalize(t_pred)
    t_gt_use = _gt_translation_in_pred_frame(t_gt, R_gt, pred_t_frame=pred_t_frame)

    rot_ang = matrix_geodesic_distance(R_pred, R_gt)
    cos_t = torch.sum(t_pred * t_gt_use, dim=-1).clamp(-1.0, 1.0)
    t_loss = 1.0 - cos_t
    if t_sample_weight is not None:
        w = t_sample_weight.float().view(-1).to(t_loss.device)
        w = w.clamp_min(1e-6)
        t_term = (t_loss.view(-1) * w).sum() / w.sum().clamp_min(1e-6)
    else:
        t_term = t_loss.mean()
    return rot_ang.mean() + float(pose_t_alpha) * t_term


def epipolar_simplified_loss(
    W_ab: torch.Tensor,
    bearing_a: torch.Tensor,
    bearing_b: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    *,
    W_ba: Optional[torch.Tensor] = None,
    allowed_mask: Optional[torch.Tensor] = None,
    angle_thresh_deg: float = 30.0,
    use_bidir: bool = True,
) -> torch.Tensor:
    W_ab = W_ab.float()
    bearing_a = _normalize(bearing_a)
    bearing_b = _normalize(bearing_b)
    R_gt = R_gt.float()
    t_gt = _normalize(t_gt)

    bA_in_B = torch.matmul(bearing_a, R_gt.transpose(-1, -2))
    t_expand = t_gt[:, None, :].expand_as(bA_in_B)
    plane_n = torch.cross(t_expand, bA_in_B, dim=-1)
    plane_n = _normalize(plane_n)
    residual = torch.abs(torch.einsum("bnc,bmc->bnm", plane_n, bearing_b)).clamp(0.0, 1.0)

    band = math.sin(math.radians(float(angle_thresh_deg)))
    if band > 0:
        cost = (residual - band).clamp_min(0.0) / max(band, 1e-6)
    else:
        cost = residual

    if allowed_mask is not None:
        allowed = allowed_mask.to(cost.dtype)
        row_sum = allowed.sum(dim=-1, keepdim=True).clamp_min(1.0)
        loss_ab = (W_ab * cost * allowed).sum() / row_sum.sum().clamp_min(1.0)
    else:
        loss_ab = (W_ab * cost).sum(dim=-1).mean()

    if (not use_bidir) or (W_ba is None):
        return loss_ab

    cost_ba = cost.transpose(-1, -2).contiguous()
    if allowed_mask is not None:
        mask_ba = allowed_mask.transpose(-1, -2).contiguous().to(cost_ba.dtype)
        row_sum = mask_ba.sum(dim=-1, keepdim=True).clamp_min(1.0)
        loss_ba = (W_ba.float() * cost_ba * mask_ba).sum() / row_sum.sum().clamp_min(1.0)
    else:
        loss_ba = (W_ba.float() * cost_ba).sum(dim=-1).mean()
    return 0.5 * (loss_ab + loss_ba)


def _avg_pool3(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)


def ssim_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = _avg_pool3(x)
    mu_y = _avg_pool3(y)
    sigma_x = _avg_pool3(x * x) - mu_x * mu_x
    sigma_y = _avg_pool3(y * y) - mu_y * mu_y
    sigma_xy = _avg_pool3(x * y) - mu_x * mu_y
    num = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)
    ssim = num / den.clamp_min(1e-6)
    return ssim.clamp(0.0, 1.0)


def erp_photometric_loss(
    I_ref: torch.Tensor,
    I_warp: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    I_tgt_identity: Optional[torch.Tensor] = None,
    alpha: float = 0.85,
    use_automask: bool = True,
) -> torch.Tensor:
    I_ref = I_ref.float()
    I_warp = I_warp.float()
    valid = valid_mask.float()

    l1 = torch.abs(I_ref - I_warp).mean(dim=1, keepdim=True)
    ssim = ssim_map(I_ref, I_warp)
    ssim_term = (1.0 - ssim).mean(dim=1, keepdim=True) * 0.5
    photometric = alpha * ssim_term + (1.0 - alpha) * l1

    if use_automask and I_tgt_identity is not None:
        with torch.no_grad():
            l1_id = torch.abs(I_ref - I_tgt_identity.float()).mean(dim=1, keepdim=True)
            ssim_id = ssim_map(I_ref, I_tgt_identity.float())
            ssim_term_id = (1.0 - ssim_id).mean(dim=1, keepdim=True) * 0.5
            identity = alpha * ssim_term_id + (1.0 - alpha) * l1_id
            auto = (photometric < identity).to(photometric.dtype)
        valid = valid * auto

    denom = valid.sum().clamp_min(1.0)
    return (photometric * valid).sum() / denom


def _gradient_x(img: torch.Tensor) -> torch.Tensor:
    return img[..., :, 1:] - img[..., :, :-1]


def _gradient_y(img: torch.Tensor) -> torch.Tensor:
    return img[..., 1:, :] - img[..., :-1, :]


def depth_smoothness_loss(inv_depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    inv_depth = inv_depth.float()
    image = image.float()
    # normalize depth scale to reduce sensitivity to global inverse-depth magnitude
    mean_inv = inv_depth.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    inv_depth = inv_depth / mean_inv
    dx = _gradient_x(inv_depth)
    dy = _gradient_y(inv_depth)
    img_dx = _gradient_x(image).abs().mean(dim=1, keepdim=True)
    img_dy = _gradient_y(image).abs().mean(dim=1, keepdim=True)
    loss_x = (dx.abs() * torch.exp(-img_dx)).mean()
    loss_y = (dy.abs() * torch.exp(-img_dy)).mean()
    return loss_x + loss_y
