"""
File: depth_branch.py
Description:
    Lightweight ERP depth branch for optional self-supervised depth and pose
    fusion experiments. The branch predicts inverse depth maps and exposes
    intermediate features for token-level fusion.

Main Components:
    - Convolution, normalization, and activation blocks
    - LightERPDepthNet for multi-scale inverse-depth prediction
    - Feature outputs used by optional translation/depth fusion

Usage / Role:
    Provides an auxiliary depth module for ablation and future fine-stage
    extensions.

Notes:
    This module is optional in the MVP and is kept lightweight for limited GPU
    memory experiments.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm_groups(ch: int) -> int:
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        g = _group_norm_groups(out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpFuse(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.fuse = ConvGNAct(in_ch + skip_ch, out_ch, stride=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class LightERPDepthNet(nn.Module):
    """
    Tuned lightweight auxiliary depth branch for the current experiment setting:
      - batch_size = 1  -> GroupNorm instead of BatchNorm
      - only return s16 / s32 heads to reduce memory and training noise
      - depth is auxiliary, so the branch stays intentionally small
    """
    def __init__(
        self,
        in_ch: int = 3,
        feat_dim: int = 32,
        inv_depth_min: float = 1.0 / 80.0,
        inv_depth_max: float = 1.0 / 0.1,
    ):
        super().__init__()
        self.inv_depth_min = float(inv_depth_min)
        self.inv_depth_max = float(inv_depth_max)

        self.stem = ConvGNAct(in_ch, 24, stride=2)     # 1/2
        self.enc2 = ConvGNAct(24, 48, stride=2)        # 1/4
        self.enc3 = ConvGNAct(48, 64, stride=2)        # 1/8
        self.enc4 = ConvGNAct(64, 96, stride=2)        # 1/16
        self.enc5 = ConvGNAct(96, 128, stride=2)       # 1/32

        self.up16 = UpFuse(128, 96, 96)

        self.feat32 = nn.Conv2d(128, feat_dim, kernel_size=1)
        self.feat16 = nn.Conv2d(96, feat_dim, kernel_size=1)

        self.inv32 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.inv16 = nn.Conv2d(96, 1, kernel_size=3, padding=1)

    def _bounded_inv_depth(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(x)
        return self.inv_depth_min + (self.inv_depth_max - self.inv_depth_min) * s

    def forward(self, img: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        x1 = self.stem(img)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        d16 = self.up16(x5, x4)

        inv_depths = {
            's32': self._bounded_inv_depth(self.inv32(x5)),
            's16': self._bounded_inv_depth(self.inv16(d16)),
        }
        depth_feats = {
            's32': self.feat32(x5),
            's16': self.feat16(d16),
        }
        return inv_depths, depth_feats
