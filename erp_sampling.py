# erp_sampling.py
# bearing -> ERP (u,v) -> grid_sample utilities

from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn.functional as F


def bearing_to_erp_uv(b: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    b: [..., 3] unit vectors
    Returns uv in pixel coordinates (float): [..., 2] where
      u in [0, W), v in [0, H)
    ERP convention:
      lon in [-pi, pi], lat in [-pi/2, pi/2]
      u = (lon + pi)/(2pi) * W
      v = (pi/2 - lat)/pi * H
    """
    x, y, z = b[..., 0], b[..., 1], b[..., 2]
    lon = torch.atan2(x, z)  # [-pi, pi]
    lat = torch.asin(torch.clamp(y, -1.0, 1.0))  # [-pi/2, pi/2]
    u = (lon + math.pi) / (2.0 * math.pi) * W
    v = (math.pi / 2.0 - lat) / math.pi * H
    uv = torch.stack([u, v], dim=-1)
    return uv


def uv_to_grid_normalized(uv: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    uv: [...,2] in pixel coords with u in [0,W), v in [0,H)
    returns grid: [...,2] normalized to [-1,1] for grid_sample
    """
    u = uv[..., 0]
    v = uv[..., 1]
    x = 2.0 * (u / (W - 1.0)) - 1.0
    y = 2.0 * (v / (H - 1.0)) - 1.0
    return torch.stack([x, y], dim=-1)


def build_level_grid_from_patch_bearings(
    patch_bearing: torch.Tensor,  # [N,p,p,3]
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Returns grid for grid_sample: [1, N*p, p, 2] normalized.
    Uses width wrap by sampling from concatenated image [B,3,H,3W] and shifting u by +W.

    TODO: For strict ERP wrap + pole handling, consider custom sampling or padding schemes.
    """
    device = patch_bearing.device
    N, p, _, _ = patch_bearing.shape

    uv = bearing_to_erp_uv(patch_bearing, H=H, W=W)  # [N,p,p,2]
    # clamp v to [0,H-1]
    uv[..., 1] = torch.clamp(uv[..., 1], 0.0, float(H - 1))

    # wrap u by shifting to center copy of 3W image
    W3 = 3 * W
    u = uv[..., 0]
    u = torch.remainder(u, float(W)) + float(W)  # now in [W,2W)
    uv3 = torch.stack([u, uv[..., 1]], dim=-1)

    grid = uv_to_grid_normalized(uv3, H=H, W=W3)  # [N,p,p,2]
    grid = grid.view(1, N * p, p, 2).contiguous()
    return grid


def sample_patches_erp(
    img: torch.Tensor,         # [B,3,H,W]
    grid: torch.Tensor,        # [1, N*p, p, 2]
    N: int,
    p: int,
) -> torch.Tensor:
    """
    Returns patches: [B, N, 3, p, p]
    """
    B, C, H, W = img.shape
    # wrap width by tiling 3 times
    img3 = torch.cat([img, img, img], dim=-1)  # [B,3,H,3W]
    gridB = grid.expand(B, -1, -1, -1).contiguous()  # [B,N*p,p,2]
    out = F.grid_sample(
        img3,
        gridB,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # [B,3,N*p,p]
    out = out.view(B, C, N, p, p).permute(0, 2, 1, 3, 4).contiguous()  # [B,N,3,p,p]
    return out
