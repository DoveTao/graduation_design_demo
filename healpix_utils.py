# healpix_utils.py
# Placeholder "HEALPix-like" equal-area cells using Fibonacci sphere.
# Provides: bearings, ids, parent_id (fine->coarse), and per-cell patch bearing grid.

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
    # golden angle
    ga = 2.0 * math.pi * (1.0 - 1.0 / phi)
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
    # pick a reference axis not parallel to b
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
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")  # [p,p]
    # offsets in tangent plane
    off1 = (xx * scale).view(1, p, p, 1)  # [1,p,p,1]
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


class HealpixHierarchy(torch.nn.Module):
    """
    Placeholder hierarchy:
      - coarse: Nc points on sphere
      - fine: Nf points on sphere, each assigned to nearest coarse center as parent_id

    TODO: Replace with real HEALPix (healpy) indexing, neighbors, exact child mapping.
    """

    def __init__(self, Nc: int, Nf: int, p: int, device: torch.device):
        super().__init__()
        self.Nc = Nc
        self.Nf = Nf
        self.p = p

        coarse = fibonacci_sphere(Nc, device=device)  # [Nc,3]
        fine = fibonacci_sphere(Nf, device=device)    # [Nf,3]

        # parent_id: nearest coarse by cosine similarity
        sim = fine @ coarse.t()  # [Nf,Nc]
        parent_id = torch.argmax(sim, dim=-1).to(torch.long)  # [Nf]

        # per-level patch bearings
        sc_c = 0.5 * approx_cell_angular_scale(Nc)
        sc_f = 0.5 * approx_cell_angular_scale(Nf)
        patch_c = build_patch_bearings(coarse, p=p, scale=sc_c)  # [Nc,p,p,3]
        patch_f = build_patch_bearings(fine, p=p, scale=sc_f)    # [Nf,p,p,3]

        # ids
        coarse_id = torch.arange(Nc, device=device, dtype=torch.long)
        fine_id = torch.arange(Nf, device=device, dtype=torch.long)

        # Register as buffers so they move with .to(device)
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
                "bearing": self.coarse_bearing,                # [Nc,3]
                "patch_bearing": self.coarse_patch_bearing,    # [Nc,p,p,3]
                "id": self.coarse_id,                          # [Nc]
                "parent_id": None,
            },
            "fine": {
                "bearing": self.fine_bearing,                  # [Nf,3]
                "patch_bearing": self.fine_patch_bearing,      # [Nf,p,p,3]
                "id": self.fine_id,                            # [Nf]
                "parent_id": self.fine_parent_id,              # [Nf]
            },
        }
