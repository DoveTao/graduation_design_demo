"""
File: healpix_utils.py
Description:
    Practical spherical hierarchy utilities for panorama matching. This file
    builds deterministic coarse and fine spherical cells with explicit parent
    relationships for coarse-to-fine routing.

Main Components:
    - LevelSpec metadata for hierarchy levels
    - HealpixHierarchy for coarse/fine bearing and patch-bearing generation
    - Deterministic fine-child assignment for each coarse cell

Usage / Role:
    Provides the spherical token layout used by the model sampler.

Notes:
    This is a lightweight in-repo hierarchy for the graduation design MVP, not
    the official HEALPix library. It keeps exact parent ids for fine-stage and
    routing experiments.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class LevelSpec:
    name: str
    N: int


def fibonacci_sphere(N: int, device: torch.device) -> torch.Tensor:
    """
    Returns unit vectors on S^2 with roughly uniform distribution.
    Output: [N, 3]
    """
    i = torch.arange(N, device=device, dtype=torch.float32)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    ga = 2.0 * math.pi * (1.0 - 1.0 / phi)  # golden angle
    y = 1.0 - 2.0 * (i + 0.5) / N
    r = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))
    theta = ga * i
    x = r * torch.cos(theta)
    z = r * torch.sin(theta)
    v = torch.stack([x, y, z], dim=-1)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-9)
    return v


def tangent_basis(b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    b: [N, 3] unit vectors
    returns orthonormal tangents t1,t2: [N,3],[N,3]
    """
    ref = torch.tensor([0.0, 1.0, 0.0], device=b.device, dtype=b.dtype).view(1, 3)
    parallel = (torch.abs((b * ref).sum(dim=-1, keepdim=True)) > 0.9).to(b.dtype)
    ref2 = torch.tensor([1.0, 0.0, 0.0], device=b.device, dtype=b.dtype).view(1, 3)
    ref = ref * (1.0 - parallel) + ref2 * parallel
    t1 = torch.cross(ref.expand_as(b), b, dim=-1)
    t1 = t1 / (t1.norm(dim=-1, keepdim=True) + 1e-9)
    t2 = torch.cross(b, t1, dim=-1)
    t2 = t2 / (t2.norm(dim=-1, keepdim=True) + 1e-9)
    return t1, t2


def build_patch_bearings(
    centers: torch.Tensor,
    p: int,
    scale: float,
) -> torch.Tensor:
    """
    centers: [N,3] unit vectors
    Returns per-cell regular grid bearings: [N, p, p, 3]
    """
    N = centers.shape[0]
    t1, t2 = tangent_basis(centers)

    lin = torch.linspace(-1.0, 1.0, p, device=centers.device, dtype=centers.dtype)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    off1 = (xx * scale).view(1, p, p, 1)
    off2 = (yy * scale).view(1, p, p, 1)

    c = centers.view(N, 1, 1, 3)
    t1v = t1.view(N, 1, 1, 3)
    t2v = t2.view(N, 1, 1, 3)
    b = c + off1 * t1v + off2 * t2v
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-9)
    return b


def approx_cell_angular_scale(N: int) -> float:
    """
    Rough per-cell angular size for equal-area partition.
    area ~ 4pi/N, linear scale ~ sqrt(area) in radians.
    """
    return math.sqrt(4.0 * math.pi / float(N))


def _local_child_offsets(
    children_per_parent: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Deterministic local offsets in the tangent plane.
    For 4 children this yields a balanced 2x2 split; for other values, it fills
    a compact grid from the center outward.
    Output: [M, 2]
    """
    if children_per_parent <= 0:
        raise ValueError(f"children_per_parent must be positive, got {children_per_parent}")

    side = int(math.ceil(math.sqrt(children_per_parent)))
    lin = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    # Prefer compact symmetric layouts.
    radius2 = (coords ** 2).sum(dim=-1)
    order = torch.argsort(radius2)
    coords = coords[order][:children_per_parent]

    # Avoid the exact center when multiple children exist; spread them in a small
    # balanced stencil so siblings are distinguishable.
    if children_per_parent > 1:
        zero = (coords.abs().sum(dim=-1) < 1e-9)
        if zero.any():
            coords[zero] = torch.tensor([1.0, 0.0], device=device, dtype=dtype)

    return coords


def _hierarchical_children(
    coarse: torch.Tensor,
    Nf: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build explicit fine children around each coarse center.

    Returns:
      fine: [Nf,3]
      parent_id: [Nf]
    """
    Nc = coarse.shape[0]
    if Nf < Nc:
        raise ValueError(f"Nf must be >= Nc, got Nc={Nc}, Nf={Nf}")

    children_per_parent = Nf // Nc
    remainder = Nf - children_per_parent * Nc
    if children_per_parent <= 0:
        raise ValueError(f"Invalid hierarchy: Nc={Nc}, Nf={Nf}")

    t1, t2 = tangent_basis(coarse)
    base_scale = 0.32 * approx_cell_angular_scale(Nc)

    offsets = _local_child_offsets(children_per_parent, coarse.device, coarse.dtype)
    offsets = offsets * base_scale

    all_fine = []
    all_parent = []
    for pid in range(Nc):
        c = coarse[pid:pid + 1]
        u = t1[pid:pid + 1]
        v = t2[pid:pid + 1]

        # Rotate each parent's local stencil to avoid global alignment artifacts.
        ang = (2.0 * math.pi * pid) / max(Nc, 1)
        ca = math.cos(ang)
        sa = math.sin(ang)
        rot = torch.tensor([[ca, -sa], [sa, ca]], device=coarse.device, dtype=coarse.dtype)
        offs = offsets @ rot.transpose(0, 1)

        child = c.view(1, 3) + offs[:, 0:1] * u.view(1, 3) + offs[:, 1:2] * v.view(1, 3)
        child = child / (child.norm(dim=-1, keepdim=True) + 1e-9)
        all_fine.append(child)
        all_parent.append(torch.full((children_per_parent,), pid, device=coarse.device, dtype=torch.long))

    fine = torch.cat(all_fine, dim=0)
    parent_id = torch.cat(all_parent, dim=0)

    # If Nf is not an exact multiple of Nc, add remaining cells by taking the
    # earliest parents and placing one extra child closer to the center.
    if remainder > 0:
        extra_parent = torch.arange(remainder, device=coarse.device, dtype=torch.long)
        c = coarse[extra_parent]
        u = t1[extra_parent]
        extra = c + (0.12 * base_scale) * u
        extra = extra / (extra.norm(dim=-1, keepdim=True) + 1e-9)
        fine = torch.cat([fine, extra], dim=0)
        parent_id = torch.cat([parent_id, extra_parent], dim=0)

    if fine.shape[0] != Nf:
        raise RuntimeError(f"Hierarchy construction mismatch: expected Nf={Nf}, got {fine.shape[0]}")
    return fine, parent_id


class HealpixHierarchy(torch.nn.Module):
    """
    Lightweight hierarchical spherical partition used by the model.

    Compared with the previous placeholder version:
      - fine cells are no longer assigned to parents via nearest coarse center;
      - parent_id is exact by construction;
      - the fine level is an explicit split of the coarse level.

    This keeps the same public interface expected by the rest of the codebase.
    """

    def __init__(self, Nc: int, Nf: int, p: int, device: torch.device):
        super().__init__()
        self.Nc = Nc
        self.Nf = Nf
        self.p = p

        coarse = fibonacci_sphere(Nc, device=device)               # [Nc,3]
        fine, parent_id = _hierarchical_children(coarse, Nf=Nf)    # [Nf,3], [Nf]

        sc_c = 0.5 * approx_cell_angular_scale(Nc)
        sc_f = 0.5 * approx_cell_angular_scale(Nf)
        patch_c = build_patch_bearings(coarse, p=p, scale=sc_c)    # [Nc,p,p,3]
        patch_f = build_patch_bearings(fine, p=p, scale=sc_f)      # [Nf,p,p,3]

        coarse_id = torch.arange(Nc, device=device, dtype=torch.long)
        fine_id = torch.arange(Nf, device=device, dtype=torch.long)

        self.register_buffer("coarse_bearing", coarse, persistent=True)
        self.register_buffer("fine_bearing", fine, persistent=True)
        self.register_buffer("coarse_patch_bearing", patch_c, persistent=True)
        self.register_buffer("fine_patch_bearing", patch_f, persistent=True)
        self.register_buffer("coarse_id", coarse_id, persistent=True)
        self.register_buffer("fine_id", fine_id, persistent=True)
        self.register_buffer("fine_parent_id", parent_id, persistent=True)

    @torch.no_grad()
    def level_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "coarse": {
                "bearing": self.coarse_bearing,
                "patch_bearing": self.coarse_patch_bearing,
                "id": self.coarse_id,
                "parent_id": None,
            },
            "fine": {
                "bearing": self.fine_bearing,
                "patch_bearing": self.fine_patch_bearing,
                "id": self.fine_id,
                "parent_id": self.fine_parent_id,
            },
        }
