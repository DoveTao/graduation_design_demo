# pose_head.py
from __future__ import annotations
import torch
import torch.nn.functional as F


def normalize_vec(v: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
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


def matrix_geodesic_distance(R1: torch.Tensor, R2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    R1,R2: [B,3,3]
    returns angle [B] in radians
    """
    R = torch.matmul(R1.transpose(-1, -2), R2)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = (tr - 1.0) / 2.0
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    return torch.acos(cos)
