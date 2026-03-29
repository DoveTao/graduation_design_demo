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
        # t_gt is stored in frame B as the B->A baseline. If the head predicts
        # the same baseline direction in frame A, the correct supervision target
        # is R^T t_gt.
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
) -> torch.Tensor:
    """
    Pose loss with explicit translation-frame handling.

    Dataset convention:
      - R_gt is R_{B<-A} (rotation A -> B)
      - t_gt is t_{BA} expressed in frame B (baseline B -> A, origin(A) in B)

    The translation head is anchored in frame A because the fused tokens are
    A-centric. We keep the dataset baseline direction (B->A) unchanged and only
    change its coordinate frame, so the A-frame supervision target is R^T t_gt.
    """
    R_pred = R_pred.float()
    R_gt = R_gt.float()

    t_pred = _normalize(t_pred)
    t_gt_use = _gt_translation_in_pred_frame(t_gt, R_gt, pred_t_frame=pred_t_frame)

    rot_ang = matrix_geodesic_distance(R_pred, R_gt)
    cos_t = torch.sum(t_pred * t_gt_use, dim=-1).clamp(-1.0, 1.0)
    t_loss = 1.0 - cos_t
    return rot_ang.mean() + float(pose_t_alpha) * t_loss.mean()


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
    """
    Weighted epipolar loss with *effective gradients* through the soft matching map.

    Instead of a constant rotation-only regularizer, this loss computes the expected
    epipolar violation under the predicted correspondence distribution W_ab (and
    optionally W_ba). This makes the loss depend directly on the matching logits.

    Inputs:
      - W_ab: [B, Na, Nb], row-stochastic soft correspondences A -> B
      - W_ba: optional [B, Nb, Na], soft correspondences B -> A
      - bearing_a / bearing_b: unit rays
      - R_gt, t_gt: ground-truth relative pose in the dataset convention
    """
    W_ab = W_ab.float()
    bearing_a = _normalize(bearing_a)
    bearing_b = _normalize(bearing_b)
    R_gt = R_gt.float()
    t_gt = _normalize(t_gt)

    # Rotate A-rays into B-frame: b_a^B = R_{B<-A} b_a^A
    bA_in_B = torch.matmul(bearing_a, R_gt.transpose(-1, -2))  # [B, Na, 3]

    # Unit epipolar plane normal for each A-ray: n = normalize(t x (R a))
    t_expand = t_gt[:, None, :].expand_as(bA_in_B)
    plane_n = torch.cross(t_expand, bA_in_B, dim=-1)
    plane_n = _normalize(plane_n)

    # Residual = sine of angle from b_B to the epipolar plane, in [0,1]
    residual = torch.abs(torch.einsum("bnc,bmc->bnm", plane_n, bearing_b)).clamp(0.0, 1.0)

    # Optional soft band: no penalty inside threshold, linear penalty outside.
    band = math.sin(math.radians(float(angle_thresh_deg)))
    if band > 0:
        cost = (residual - band).clamp_min(0.0) / max(band, 1e-6)
    else:
        cost = residual

    def _weighted_expectation(W: torch.Tensor, C: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            mask = mask.to(torch.bool)
            W = W * mask.to(W.dtype)
            W = W / W.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return (W * C).sum(dim=-1).mean()

    loss_ab = _weighted_expectation(W_ab, cost, allowed_mask)

    if W_ba is None or not bool(use_bidir):
        return loss_ab

    mask_ba = None if allowed_mask is None else allowed_mask.transpose(-1, -2).contiguous()
    loss_ba = _weighted_expectation(W_ba.float(), cost.transpose(-1, -2).contiguous(), mask_ba)
    return 0.5 * (loss_ab + loss_ba)
