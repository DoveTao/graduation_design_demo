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
    lon = torch.atan2(x, z)
    lat = torch.asin(torch.clamp(y, -1.0, 1.0))
    u = (lon + math.pi) / (2.0 * math.pi) * W
    v = (math.pi / 2.0 - lat) / math.pi * H
    uv = torch.stack([u, v], dim=-1)
    return uv


def uv_to_grid_normalized(uv: torch.Tensor, H: int, W: int) -> torch.Tensor:
    u = uv[..., 0]
    v = uv[..., 1]
    x = 2.0 * (u / (W - 1.0)) - 1.0
    y = 2.0 * (v / (H - 1.0)) - 1.0
    return torch.stack([x, y], dim=-1)


def build_level_grid_from_patch_bearings(
    patch_bearing: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    uv = bearing_to_erp_uv(patch_bearing, H=H, W=W)
    v = torch.clamp(uv[..., 1], 0.0, float(H - 1))
    u = torch.remainder(uv[..., 0], float(W)) + float(W)
    uv3 = torch.stack([u, v], dim=-1)
    grid = uv_to_grid_normalized(uv3, H=H, W=3 * W)
    N, p, _, _ = patch_bearing.shape
    return grid.view(1, N * p, p, 2).contiguous()


def sample_patches_erp(
    img: torch.Tensor,
    grid: torch.Tensor,
    N: int,
    p: int,
) -> torch.Tensor:
    B, C, H, W = img.shape
    img3 = torch.cat([img, img, img], dim=-1)
    gridB = grid.expand(B, -1, -1, -1).contiguous()
    out = F.grid_sample(
        img3,
        gridB,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    out = out.view(B, C, N, p, p).permute(0, 2, 1, 3, 4).contiguous()
    return out


@torch.no_grad()
def erp_pixel_bearing_grid(H: int, W: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return per-pixel ERP bearing grid with shape [1,H,W,3]. Uses pixel centers."""
    u = torch.arange(W, device=device, dtype=dtype) + 0.5
    v = torch.arange(H, device=device, dtype=dtype) + 0.5
    vv, uu = torch.meshgrid(v, u, indexing='ij')
    lon = (uu / float(W)) * (2.0 * math.pi) - math.pi
    lat = math.pi / 2.0 - (vv / float(H)) * math.pi
    x = torch.cos(lat) * torch.sin(lon)
    y = torch.sin(lat)
    z = torch.cos(lat) * torch.cos(lon)
    b = torch.stack([x, y, z], dim=-1)
    b = b / (torch.linalg.norm(b, dim=-1, keepdim=True) + 1e-9)
    return b.unsqueeze(0)


def sample_dense_features_from_bearings(feat_map: torch.Tensor, bearing: torch.Tensor) -> torch.Tensor:
    B, C, H, W = feat_map.shape
    uv = bearing_to_erp_uv(bearing, H=H, W=W)
    v = torch.clamp(uv[..., 1], 0.0, float(H - 1))
    u = torch.remainder(uv[..., 0], float(W)) + float(W)
    uv3 = torch.stack([u, v], dim=-1)
    grid = uv_to_grid_normalized(uv3, H=H, W=3 * W).view(B, -1, 1, 2)
    fmap3 = torch.cat([feat_map, feat_map, feat_map], dim=-1)
    samp = F.grid_sample(fmap3, grid, mode='bilinear', padding_mode='border', align_corners=True)
    samp = samp.squeeze(-1).transpose(1, 2).contiguous()
    return samp


def warp_erp_with_depth_pose(
    ref_inv_depth: torch.Tensor,
    R_BA: torch.Tensor,
    t_BA: torch.Tensor,
    target_img: torch.Tensor,
    *,
    translation_scale: float = 1.0,
    min_depth: float = 0.1,
    max_depth: float = 80.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Warp target_img into the reference view using ERP inverse depth and relative pose.
    t_BA is the B->A baseline expressed in frame B, matching the dataset convention.
    """
    B, _, H, W = ref_inv_depth.shape
    device, dtype = ref_inv_depth.device, ref_inv_depth.dtype
    bearing = erp_pixel_bearing_grid(H, W, device=device, dtype=dtype).expand(B, -1, -1, -1)

    inv = ref_inv_depth.clamp_min(1e-6)
    depth = inv.reciprocal().clamp(min=min_depth, max=max_depth)
    X_A = depth.permute(0, 2, 3, 1) * bearing
    X_B = torch.matmul(X_A.view(B, -1, 3), R_BA.transpose(-1, -2).float())
    X_B = X_B + float(translation_scale) * t_BA[:, None, :].float()

    z_ok = X_B[..., 2:3] > 1e-6
    norm = torch.linalg.norm(X_B, dim=-1, keepdim=True)
    valid = (norm > 1e-6) & z_ok
    b_B = X_B / norm.clamp_min(1e-6)

    uv = bearing_to_erp_uv(b_B.view(B, H, W, 3), H=H, W=W)
    v = torch.clamp(uv[..., 1], 0.0, float(H - 1))
    u = torch.remainder(uv[..., 0], float(W)) + float(W)
    uv3 = torch.stack([u, v], dim=-1)
    grid = uv_to_grid_normalized(uv3, H=H, W=3 * W)

    tgt3 = torch.cat([target_img, target_img, target_img], dim=-1)
    warped = F.grid_sample(tgt3, grid, mode='bilinear', padding_mode='border', align_corners=True)
    valid = valid.view(B, H, W, 1).permute(0, 3, 1, 2).to(warped.dtype)
    return warped, valid
