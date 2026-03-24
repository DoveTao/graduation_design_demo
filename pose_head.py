import torch
import torch.nn.functional as F


def normalize_vec(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def rot6d_to_matrix(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert 6D rotation representation to a proper rotation matrix.

    This implementation forces float32 internally for AMP stability.
    Input:  x [...,6]
    Output: R [...,3,3] (float32)
    """
    x = x.float()
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]

    b1 = normalize_vec(a1, eps=eps)
    proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = normalize_vec(a2 - proj, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)

    R = torch.stack([b1, b2, b3], dim=-1)  # [...,3,3] (columns)
    return R


def matrix_geodesic_distance(R1: torch.Tensor, R2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Geodesic distance on SO(3): angle between two rotation matrices (radians).
    Stable under AMP (float32 internally).
    """
    R1 = R1.float()
    R2 = R2.float()

    R = torch.matmul(R1.transpose(-1, -2), R2)  # [...,3,3]
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = (trace - 1.0) / 2.0
    cos = cos.clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos)
