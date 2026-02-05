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


def reliability_weighted(base_loss: torch.Tensor, cA: torch.Tensor, cB: torch.Tensor) -> torch.Tensor:
    # cA/cB are logits -> convert to probability for weighting
    w = 0.5 * (torch.sigmoid(cA).mean() + torch.sigmoid(cB).mean())
    return w * base_loss


def reliability_reg_loss(cA: torch.Tensor, cB: torch.Tensor) -> torch.Tensor:
    targetA = torch.ones_like(cA)
    targetB = torch.ones_like(cB)
    la = F.binary_cross_entropy_with_logits(cA, targetA)
    lb = F.binary_cross_entropy_with_logits(cB, targetB)
    return 0.5 * (la + lb)


def epipolar_strict_loss(
    R: torch.Tensor,           # [B,3,3]
    t_dir: torch.Tensor,       # [B,3]
    W_ab: torch.Tensor,        # [B,NA,NB]
    bearing_a: torch.Tensor,   # [B,NA,3]
    bearing_b: torch.Tensor,   # [B,NB,3]
) -> torch.Tensor:
    """
    Strict spherical epipolar consistency.

    For each A-ray, rotate into B: r = R*bA.
    Epipolar plane normal: n = t x r.
    Great-circle constraint for candidate bB: n_unit · bB = 0.

    Loss: expected squared violation under soft correspondence W_ab:
      E_{j~W_ab[i,:]} [ (n_unit(i) · bB_j)^2 ].
    """
    B, NA, _ = bearing_a.shape
    NB = bearing_b.shape[1]

    t = F.normalize(t_dir.float(), dim=-1, eps=1e-6)                             # [B,3]
    r = torch.matmul(bearing_a.float(), R.transpose(-1, -2).float())             # [B,NA,3]
    r = F.normalize(r, dim=-1, eps=1e-6)
    bB = F.normalize(bearing_b.float(), dim=-1, eps=1e-6)                        # [B,NB,3]

    t_exp = t[:, None, :].expand(-1, NA, -1)                                     # [B,NA,3]
    n = torch.cross(t_exp, r, dim=-1)                                            # [B,NA,3]
    n_norm = n.norm(dim=-1, keepdim=True)                                        # [B,NA,1]
    valid = (n_norm > 1e-3).float()                                              # [B,NA,1]
    n_unit = n / (n_norm + 1e-6)

    s = torch.matmul(n_unit, bB.transpose(-1, -2))                               # [B,NA,NB]
    viol = s * s                                                                 # squared distance to epipolar plane
    loss_a = (W_ab.float() * viol).sum(dim=-1)                                   # [B,NA]

    loss = (loss_a * valid.squeeze(-1)).sum() / (valid.sum() + 1e-6)
    return loss
