# pose_head.py
from __future__ import annotations
import torch
import torch.nn.functional as F


def normalize_vec(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # eps 对 AMP 友好一点（1e-9 在 fp16 里意义不大）
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    6D rotation representation (Zhou et al.):
    x: [B,6] -> R: [B,3,3]
    """
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]
    b1 = normalize_vec(a1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = normalize_vec(b2)
    b3 = torch.cross(b1, b2, dim=-1)
    R = torch.stack([b1, b2, b3], dim=-1)  # columns
    return R


def matrix_geodesic_distance(R1: torch.Tensor, R2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Stable geodesic angle (radians) using atan2, AMP-friendly.

    angle = atan2( ||vee(R - R^T)|| / 2, (trace(R)-1)/2 )
    where R = R1^T R2
    """
    # 强制 float32 计算，避免 autocast 把 acos/边界搞炸
    R1 = R1.float()
    R2 = R2.float()

    R = torch.matmul(R1.transpose(-1, -2), R2)  # [B,3,3]

    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    c = (tr - 1.0) * 0.5  # cos(theta)
    c = torch.clamp(c, -1.0, 1.0)

    # s = ||vee(R - R^T)|| / 2  (equivalent to sin(theta) magnitude)
    vx = R[..., 2, 1] - R[..., 1, 2]
    vy = R[..., 0, 2] - R[..., 2, 0]
    vz = R[..., 1, 0] - R[..., 0, 1]
    s = 0.5 * torch.sqrt(vx * vx + vy * vy + vz * vz + eps)

    angle = torch.atan2(s, c.clamp(-1.0 + eps, 1.0 - eps))
    return angle
