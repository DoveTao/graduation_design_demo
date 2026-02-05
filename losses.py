# losses.py
from __future__ import annotations
import torch
import torch.nn.functional as F

from pose_head import matrix_geodesic_distance, normalize_vec


def pose_loss(R_pred: torch.Tensor, t_pred: torch.Tensor, R_gt: torch.Tensor, t_gt: torch.Tensor) -> torch.Tensor:
    """
    R_pred,R_gt: [B,3,3]
    t_pred,t_gt: [B,3] (direction, unit)
    """
    t_pred = normalize_vec(t_pred)
    t_gt = normalize_vec(t_gt)
    rot_ang = matrix_geodesic_distance(R_pred, R_gt)     # [B]
    trans = 1.0 - torch.sum(t_pred * t_gt, dim=-1)       # [B]
    return rot_ang.mean() + trans.mean()


def xlevel_loss(Wc_tilde: torch.Tensor, Wc: torch.Tensor) -> torch.Tensor:
    # both [B,NcA,NcB]
    return F.mse_loss(Wc_tilde, Wc)


def cycle_loss(W_ab: torch.Tensor, W_ba: torch.Tensor) -> torch.Tensor:
    """
    W_ab: [B,NA,NB], W_ba: [B,NB,NA]
    Ideal: W_ab @ W_ba ≈ I_NA
    """
    B, NA, NB = W_ab.shape
    P = torch.matmul(W_ab, W_ba)  # [B,NA,NA]
    I = torch.eye(NA, device=W_ab.device, dtype=W_ab.dtype).unsqueeze(0).expand(B, NA, NA)
    return F.mse_loss(P, I)


'''def reliability_weighted(
    base_loss: torch.Tensor,
    cA: torch.Tensor,  # [B,NA,1]
    cB: torch.Tensor,  # [B,NB,1]
) -> torch.Tensor:
    """
    Simple reliability weighting: multiply loss by mean confidence.
    """
    w = 0.5 * (cA.mean() + cB.mean())
    return w * base_loss'''

def reliability_weighted(base_loss: torch.Tensor, cA: torch.Tensor, cB: torch.Tensor) -> torch.Tensor:
    # cA/cB 是 logits，所以权重要用 sigmoid 转成 (0,1)
    w = 0.5 * (torch.sigmoid(cA).mean() + torch.sigmoid(cB).mean())
    return w * base_loss



'''def reliability_reg_loss(cA: torch.Tensor, cB: torch.Tensor) -> torch.Tensor:
    """
    Simple regularizer: encourage confidence toward 1 (can be replaced with better calibration).
    """
    target = torch.ones_like(cA)
    la = F.binary_cross_entropy(cA, target)
    lb = F.binary_cross_entropy(cB, target)
    return 0.5 * (la + lb)'''

def reliability_reg_loss(cA: torch.Tensor, cB: torch.Tensor) -> torch.Tensor:
    targetA = torch.ones_like(cA)
    targetB = torch.ones_like(cB)
    la = F.binary_cross_entropy_with_logits(cA, targetA)
    lb = F.binary_cross_entropy_with_logits(cB, targetB)
    return 0.5 * (la + lb)



def epipolar_simplified_loss(
    R: torch.Tensor,           # [B,3,3]
    W_ab: torch.Tensor,        # [B,NA,NB]
    bearing_a: torch.Tensor,   # [B,NA,3]
    bearing_b: torch.Tensor,   # [B,NB,3]
) -> torch.Tensor:
    """
    Simplified epipolar consistency placeholder:
      compute expected bearing in B per A token: bB_exp = sum_j W_ab[i,j] * bB[j]
      penalize angular difference between R*bA and bB_exp.

    TODO: Replace with strict epipolar (R,t_dir) band constraint on sphere.
    """
    bA_rot = torch.matmul(bearing_a, R.transpose(-1, -2))  # [B,NA,3]
    bA_rot = F.normalize(bA_rot, dim=-1, eps=1e-6)

    bB_exp = torch.matmul(W_ab, bearing_b)                # [B,NA,3]
    bB_exp = F.normalize(bB_exp, dim=-1, eps=1e-6)

    cos = torch.sum(bA_rot * bB_exp, dim=-1)
    cos = cos.clamp(-1.0 + 1e-4, 1.0 - 1e-4)  # avoid acos grad blow-up at ±1
    ang = torch.acos(cos)  # radians
    return ang.mean()
