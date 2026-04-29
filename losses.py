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
    pose_t_oriented_weight: float = 1.0,
    pose_t_axis_weight: float = 0.0,
    pred_t_frame: str = "B",
    rot_sample_weight: Optional[torch.Tensor] = None,
    t_sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    R_pred = R_pred.float()
    R_gt = R_gt.float()

    t_pred = _normalize(t_pred)
    t_gt_use = _gt_translation_in_pred_frame(t_gt, R_gt, pred_t_frame=pred_t_frame)

    rot_ang = matrix_geodesic_distance(R_pred, R_gt)
    if rot_sample_weight is not None:
        rw = rot_sample_weight.float().view(-1).to(rot_ang.device)
        rw = rw.clamp_min(1e-6)
        rot_term = (rot_ang.view(-1) * rw).sum() / rw.sum().clamp_min(1e-6)
    else:
        rot_term = rot_ang.mean()

    cos_t = torch.sum(t_pred * t_gt_use, dim=-1).clamp(-1.0, 1.0)
    oriented_loss = 1.0 - cos_t
    axis_loss = 1.0 - cos_t.abs()
    t_loss = float(pose_t_oriented_weight) * oriented_loss + float(pose_t_axis_weight) * axis_loss
    if t_sample_weight is not None:
        w = t_sample_weight.float().view(-1).to(t_loss.device)
        w = w.clamp_min(1e-6)
        t_term = (t_loss.view(-1) * w).sum() / w.sum().clamp_min(1e-6)
    else:
        t_term = t_loss.mean()
    return rot_term + float(pose_t_alpha) * t_term


def epipolar_simplified_loss(
    W_ab: torch.Tensor,
    bearing_a: torch.Tensor,
    bearing_b: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    *,
    W_ba: Optional[torch.Tensor] = None,
    allowed_mask: Optional[torch.Tensor] = None,
    angle_thresh_deg: float = 10.0,
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


def epipolar_gt_matching_loss(
    W_ab: torch.Tensor,
    bearing_a: torch.Tensor,
    bearing_b: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    *,
    W_ba: Optional[torch.Tensor] = None,
    allowed_mask: Optional[torch.Tensor] = None,
    angle_thresh_deg: float = 10.0,
    use_bidir: bool = True,
    temperature: float = 1.0,
    min_plane_norm: float = 1e-4,
) -> torch.Tensor:
    """GT-pose epipolar matching loss for soft correspondence matrices.

    It builds a soft target distribution from GT epipolar geometry and minimizes
    cross entropy between that target and W_ab/W_ba.  Gradients only update the
    predicted matching distribution.
    """
    assert W_ab.ndim == 3, f"W_ab must have shape [B, Na, Nb], got {tuple(W_ab.shape)}"
    assert bearing_a.ndim == 3, f"bearing_a must have shape [B, Na, C], got {tuple(bearing_a.shape)}"
    assert bearing_b.ndim == 3, f"bearing_b must have shape [B, Nb, C], got {tuple(bearing_b.shape)}"
    assert W_ab.shape[0] == bearing_a.shape[0] == bearing_b.shape[0], (
        f"Batch mismatch: W_ab={tuple(W_ab.shape)}, "
        f"bearing_a={tuple(bearing_a.shape)}, bearing_b={tuple(bearing_b.shape)}"
    )
    assert W_ab.shape[1] == bearing_a.shape[1], (
        f"W_ab rows must match bearing_a count: W_ab={tuple(W_ab.shape)}, "
        f"bearing_a={tuple(bearing_a.shape)}"
    )
    assert W_ab.shape[2] == bearing_b.shape[1], (
        f"W_ab cols must match bearing_b count: W_ab={tuple(W_ab.shape)}, "
        f"bearing_b={tuple(bearing_b.shape)}"
    )
    if W_ba is not None:
        assert W_ba.ndim == 3, f"W_ba must have shape [B, Nb, Na], got {tuple(W_ba.shape)}"
        assert W_ba.shape[0] == W_ab.shape[0], (
            f"W_ba batch must match W_ab: W_ba={tuple(W_ba.shape)}, W_ab={tuple(W_ab.shape)}"
        )
        assert W_ba.shape[1] == W_ab.shape[2] and W_ba.shape[2] == W_ab.shape[1], (
            f"W_ba shape must be transpose-compatible with W_ab: "
            f"W_ba={tuple(W_ba.shape)}, W_ab={tuple(W_ab.shape)}"
        )
    if allowed_mask is not None:
        assert allowed_mask.shape == W_ab.shape, (
            f"allowed_mask must match W_ab shape: allowed_mask={tuple(allowed_mask.shape)}, "
            f"W_ab={tuple(W_ab.shape)}"
        )

    W_ab = W_ab.float().clamp_min(1e-9)
    bearing_a = _normalize(bearing_a)
    bearing_b = _normalize(bearing_b)
    R_gt = R_gt.float()
    t_gt = _normalize(t_gt)

    # A bearing expressed in B frame: row-vector version of R_gt @ a.
    bA_in_B = torch.matmul(bearing_a, R_gt.transpose(-1, -2))
    t_expand = t_gt[:, None, :].expand_as(bA_in_B)
    plane_n_raw = torch.cross(t_expand, bA_in_B, dim=-1)
    plane_norm = torch.linalg.norm(plane_n_raw, dim=-1)
    valid_plane = plane_norm > float(min_plane_norm)
    plane_n = _normalize(plane_n_raw)

    residual = torch.abs(torch.einsum("bnc,bmc->bnm", plane_n, bearing_b)).clamp(0.0, 1.0)

    band = max(math.sin(math.radians(float(angle_thresh_deg))), 1e-4)
    sigma = max(0.5 * band, 1e-4)
    temp = max(float(temperature), 1e-4)

    target_logits = -((residual / sigma) ** 2) / temp
    if allowed_mask is not None:
        allowed = allowed_mask.to(dtype=torch.bool)
        allowed = allowed & valid_plane.unsqueeze(-1)
        masked_logits = target_logits.masked_fill(~allowed, float("-inf"))
        no_valid = ~allowed.any(dim=-1, keepdim=True)
        masked_logits = torch.where(no_valid, target_logits, masked_logits)
        target_ab = torch.softmax(masked_logits, dim=-1).detach()
        target_ab = target_ab * allowed.to(target_ab.dtype)
        target_ab = target_ab / target_ab.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        row_valid = allowed.any(dim=-1).to(W_ab.dtype)
        loss_ab_rows = -(target_ab * W_ab.log()).sum(dim=-1)
        loss_ab = (loss_ab_rows * row_valid).sum() / row_valid.sum().clamp_min(1.0)
    else:
        target_ab = torch.softmax(target_logits, dim=-1).detach()
        row_valid = valid_plane.to(W_ab.dtype)
        loss_ab_rows = -(target_ab * W_ab.log()).sum(dim=-1)
        loss_ab = (loss_ab_rows * row_valid).sum() / row_valid.sum().clamp_min(1.0)

    if (not use_bidir) or (W_ba is None):
        return loss_ab

    W_ba = W_ba.float().clamp_min(1e-9)
    if allowed_mask is not None:
        allowed_ba = allowed_mask.transpose(-1, -2).contiguous().to(dtype=torch.bool)
        allowed_ba = allowed_ba & valid_plane.unsqueeze(1)
        logits_ba = target_logits.transpose(-1, -2).contiguous()
        masked_logits_ba = logits_ba.masked_fill(~allowed_ba, float("-inf"))
        no_valid_ba = ~allowed_ba.any(dim=-1, keepdim=True)
        masked_logits_ba = torch.where(no_valid_ba, logits_ba, masked_logits_ba)
        target_ba = torch.softmax(masked_logits_ba, dim=-1).detach()
        target_ba = target_ba * allowed_ba.to(target_ba.dtype)
        target_ba = target_ba / target_ba.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        row_valid_ba = allowed_ba.any(dim=-1).to(W_ba.dtype)
        loss_ba_rows = -(target_ba * W_ba.log()).sum(dim=-1)
        loss_ba = (loss_ba_rows * row_valid_ba).sum() / row_valid_ba.sum().clamp_min(1.0)
    else:
        target_ba = torch.softmax(target_logits.transpose(-1, -2).contiguous(), dim=-1).detach()
        row_valid_ba = valid_plane.any(dim=-1, keepdim=True).expand(-1, W_ba.shape[1]).to(W_ba.dtype)
        loss_ba_rows = -(target_ba * W_ba.log()).sum(dim=-1)
        loss_ba = (loss_ba_rows * row_valid_ba).sum() / row_valid_ba.sum().clamp_min(1.0)
    return 0.5 * (loss_ab + loss_ba)


def epipolar_gt_band_nll_loss(
    W_ab: torch.Tensor,
    bearing_a: torch.Tensor,
    bearing_b: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    *,
    W_ba: Optional[torch.Tensor] = None,
    allowed_mask: Optional[torch.Tensor] = None,
    angle_thresh_deg: float = 10.0,
    use_bidir: bool = True,
    min_plane_norm: float = 1e-4,
) -> torch.Tensor:
    """Maximize predicted probability mass inside the GT epipolar band.

    This is intentionally coarser than a soft CE target along the whole epipolar
    line. At coarse resolution, many cells are equally plausible along the line,
    so the first useful signal is simply to put mass inside the valid band.
    """
    assert W_ab.ndim == 3, f"W_ab must have shape [B, Na, Nb], got {tuple(W_ab.shape)}"
    assert bearing_a.ndim == 3 and bearing_b.ndim == 3
    assert W_ab.shape[0] == bearing_a.shape[0] == bearing_b.shape[0]
    assert W_ab.shape[1] == bearing_a.shape[1]
    assert W_ab.shape[2] == bearing_b.shape[1]
    if W_ba is not None:
        assert W_ba.shape[0] == W_ab.shape[0]
        assert W_ba.shape[1] == W_ab.shape[2] and W_ba.shape[2] == W_ab.shape[1]
    if allowed_mask is not None:
        assert allowed_mask.shape == W_ab.shape

    W_ab = W_ab.float().clamp_min(1e-9)
    bearing_a = _normalize(bearing_a)
    bearing_b = _normalize(bearing_b)
    R_gt = R_gt.float()
    t_gt = _normalize(t_gt)

    bA_in_B = torch.matmul(bearing_a, R_gt.transpose(-1, -2))
    plane_n_raw = torch.cross(t_gt[:, None, :].expand_as(bA_in_B), bA_in_B, dim=-1)
    plane_norm = torch.linalg.norm(plane_n_raw, dim=-1)
    valid_plane = plane_norm > float(min_plane_norm)
    plane_n = _normalize(plane_n_raw)
    residual = torch.abs(torch.einsum("bnc,bmc->bnm", plane_n, bearing_b)).clamp(0.0, 1.0)

    band = max(math.sin(math.radians(float(angle_thresh_deg))), 1e-6)
    gt_band = (residual <= band) & valid_plane.unsqueeze(-1)
    if allowed_mask is not None:
        gt_band = gt_band & allowed_mask.to(dtype=torch.bool)

    row_valid = gt_band.any(dim=-1)
    mass_ab = (W_ab * gt_band.to(W_ab.dtype)).sum(dim=-1).clamp_min(1e-9)
    loss_ab_rows = -mass_ab.log()
    loss_ab = (loss_ab_rows * row_valid.to(W_ab.dtype)).sum() / row_valid.to(W_ab.dtype).sum().clamp_min(1.0)

    if (not use_bidir) or (W_ba is None):
        return loss_ab

    W_ba = W_ba.float().clamp_min(1e-9)
    gt_band_ba = gt_band.transpose(-1, -2).contiguous()
    row_valid_ba = gt_band_ba.any(dim=-1)
    mass_ba = (W_ba * gt_band_ba.to(W_ba.dtype)).sum(dim=-1).clamp_min(1e-9)
    loss_ba_rows = -mass_ba.log()
    loss_ba = (loss_ba_rows * row_valid_ba.to(W_ba.dtype)).sum() / row_valid_ba.to(W_ba.dtype).sum().clamp_min(1.0)
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
