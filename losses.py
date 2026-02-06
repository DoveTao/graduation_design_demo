# losses.py
from __future__ import annotations
import torch
import torch.nn.functional as F

from pose_head import matrix_geodesic_distance


# =========================
# Tunables
# =========================
# 优先 tdir：加大平移方向项的权重（建议 4~8）
POSE_T_ALPHA: float = 6.0

# reliability weighting 防“逃逸”：
# - detach：不让网络通过降低 w 来缩小主监督梯度
# - clamp：避免主监督被乘到太小
REL_W_MIN: float = 0.70
REL_W_MAX: float = 1.00


def pose_loss(
    R_pred: torch.Tensor,
    t_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
) -> torch.Tensor:
    """
    R_pred,R_gt: [B,3,3]
    t_pred,t_gt: [B,3] 方向向量（不要求已归一）

    目标：优先优化 tdir，同时训练更稳：
      - rotation: geodesic angle (rad)
      - translation direction: (1 - dot)  (smooth, monotonic to angle)
    """
    # safer normalize (avoid NaN when norm ~ 0)
    t_pred = F.normalize(t_pred.float(), dim=-1, eps=1e-6)
    t_gt = F.normalize(t_gt.float(), dim=-1, eps=1e-6)

    rot_ang = matrix_geodesic_distance(R_pred, R_gt)  # [B], rad

    # smooth direction loss in [0,2]
    cos_t = torch.sum(t_pred * t_gt, dim=-1).clamp(-1.0, 1.0)  # [B]
    t_loss = 1.0 - cos_t                                       # [B]

    return rot_ang.mean() + POSE_T_ALPHA * t_loss.mean()


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
    """
    Prevent "confidence escape":
      - w is detached: network can't reduce w to shrink supervision gradients
      - clamp w: supervision won't be multiplied to near zero
    """
    w = 0.5 * (torch.sigmoid(cA).mean() + torch.sigmoid(cB).mean())  # scalar
    w = w.detach()
    w = torch.clamp(w, REL_W_MIN, REL_W_MAX)
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
    viol = s * s                                                                 # [B,NA,NB]
    loss_a = (W_ab.float() * viol).sum(dim=-1)                                   # [B,NA]

    loss = (loss_a * valid.squeeze(-1)).sum() / (valid.sum() + 1e-6)
    return loss
