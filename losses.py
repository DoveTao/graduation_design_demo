import math
from typing import Optional

import torch
import torch.nn.functional as F

from pose_head import matrix_geodesic_distance


def pose_loss(
    R_pred: torch.Tensor,
    t_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    *,
    pose_t_alpha: float = 1.0,
) -> torch.Tensor:
    """
    Pose loss (stable under AMP).

    Shapes:
      - R_pred, R_gt: [B,3,3]
      - t_pred, t_gt: [B,3]   (direction vectors, not necessarily normalized)

    Convention (matches dataset_pano_only.py):
      - R_gt is R_{B<-A} (rotation A -> B)
      - t_gt is t_{BA} expressed in frame B (baseline B -> A, i.e., origin(A) in B)

    Loss:
      - rotation: geodesic angle in radians
      - translation direction: (1 - dot) in [0,2]  (monotonic to angular error, no acos)
    """
    # Force float32 for numerical stability (especially under fp16 AMP)
    R_pred = R_pred.float()
    R_gt = R_gt.float()

    t_pred = F.normalize(t_pred.float(), dim=-1, eps=1e-6)
    t_gt = F.normalize(t_gt.float(), dim=-1, eps=1e-6)

    rot_ang = matrix_geodesic_distance(R_pred, R_gt)  # [B], rad

    cos_t = torch.sum(t_pred * t_gt, dim=-1).clamp(-1.0, 1.0)  # [B]
    t_loss = 1.0 - cos_t                                       # [B]

    return rot_ang.mean() + float(pose_t_alpha) * t_loss.mean()


def epipolar_simplified_loss(
    bearing_a: torch.Tensor,
    bearing_b: torch.Tensor,
    R_gt: torch.Tensor,
    *,
    angle_thresh_deg: float = 30.0,
) -> torch.Tensor:
    """
    A cheap *rotation-only* epipolar-consistency penalty, used as a soft regularizer.

    bearing_a: [B,Na,3] unit rays in frame A
    bearing_b: [B,Nb,3] unit rays in frame B
    R_gt     : [B,3,3]  (A -> B)

    We rotate A-rays into B-frame and penalize pairs whose angular deviation exceeds
    `angle_thresh_deg`. Implemented without acos for AMP stability.

    Returns: scalar
    """
    # float32 for stability
    bearing_a = bearing_a.float()
    bearing_b = bearing_b.float()
    R_gt = R_gt.float()

    # Rotate A rays into B: row-vector convention => bA_rot = bA @ R^T
    bA_rot = torch.matmul(bearing_a, R_gt.transpose(-1, -2))  # [B,Na,3]

    # broadcast to [B,Na,Nb,3]
    bA = bA_rot.unsqueeze(2)
    bB = bearing_b.unsqueeze(1)

    cos = torch.sum(bA * bB, dim=-1).clamp(-1.0, 1.0)  # [B,Na,Nb]

    cos_thresh = math.cos(math.radians(float(angle_thresh_deg)))
    # angle > thresh  <=>  cos < cos_thresh
    loss = (cos_thresh - cos).clamp(min=0.0).mean()
    return loss
