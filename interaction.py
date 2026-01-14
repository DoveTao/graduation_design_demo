# interaction.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pose_head import rot6d_to_matrix, normalize_vec


@dataclass
class Tokens:
    feat: torch.Tensor      # [B,N,D]
    bearing: torch.Tensor   # [B,N,3]
    level: str              # "coarse" / "fine"
    id: torch.Tensor        # [B,N] long
    parent_id: torch.Tensor | None  # [B,N] long or None


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: [B,NA,D], b: [B,NB,D] -> sim [B,NA,NB]
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.transpose(-1, -2)


class FuseMLP(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * D, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

    def forward(self, a: torch.Tensor, b_att: torch.Tensor) -> torch.Tensor:
        # a,b_att: [B,N,D]
        x = torch.cat([a, b_att, a - b_att, a * b_att], dim=-1)
        return self.net(x)


class CoarsePoseHead(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),
            nn.GELU(),
        )
        self.rot = nn.Linear(D, 6)
        self.t = nn.Linear(D, 3)

    def forward(self, Fc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fc: [B,N,D] -> Rc [B,3,3], tc_dir [B,3]
        """
        z = Fc.mean(dim=1)  # [B,D]
        z = self.mlp(z)
        r6 = self.rot(z)
        t = self.t(z)
        R = rot6d_to_matrix(r6)          # [B,3,3]
        t_dir = normalize_vec(t)         # [B,3]
        return R, t_dir


class FinePoseHead(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),
            nn.GELU(),
        )
        self.rot = nn.Linear(D, 6)
        self.t = nn.Linear(D, 3)

    def forward(self, Ff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = Ff.mean(dim=1)  # [B,D]
        z = self.mlp(z)
        r6 = self.rot(z)
        t = self.t(z)
        R = rot6d_to_matrix(r6)
        t_dir = normalize_vec(t)
        return R, t_dir


class ReliabilityHead(nn.Module):
    """
    Simple learned confidence per token: c = sigmoid(MLP(feat)).
    """
    def __init__(self, D: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, 1),
        )

    #   def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B,N,D] -> c: [B,N,1]
        # return torch.sigmoid(self.net(feat))
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B,N,D] -> logits: [B,N,1]
        # 需要概率时在外面用 torch.sigmoid(logits)
        return self.net(feat)


def build_routing_mask_from_coarse_topk(
    Wc_ab: torch.Tensor,         # [B,NcA,NcB]
    fine_parent_a: torch.Tensor,  # [B,NfA] long
    fine_parent_b: torch.Tensor,  # [B,NfB] long
    topk: int,
) -> torch.Tensor:
    """
    Build fine pair mask via parent mapping:
      allow (a_f, b_f) if (parent_a, parent_b) is within top-k coarse correspondences.

    Returns mask: [B,NfA,NfB] bool (True = allowed)
    """
    B, NcA, NcB = Wc_ab.shape
    device = Wc_ab.device
    _, topk_idx = torch.topk(Wc_ab, k=min(topk, NcB), dim=-1)  # [B,NcA,topk]
    allowed_parent = torch.zeros((B, NcA, NcB), device=device, dtype=torch.bool)
    allowed_parent.scatter_(dim=-1, index=topk_idx, value=True)

    # Build fine mask per batch (loop acceptable for MVP sizes)
    masks = []
    for b in range(B):
        ap = fine_parent_a[b]  # [NfA]
        bp = fine_parent_b[b]  # [NfB]
        allowed_ap = allowed_parent[b, ap]          # [NfA,NcB]
        mask = allowed_ap[:, bp]                    # [NfA,NfB]
        masks.append(mask)
    return torch.stack(masks, dim=0)  # [B,NfA,NfB]


def epipolar_band_bias_or_mask(
    R: torch.Tensor,             # [B,3,3]
    bearing_a: torch.Tensor,     # [B,NA,3]
    bearing_b: torch.Tensor,     # [B,NB,3]
    angle_thresh_deg: float,
    bias_strength: float,
    mode: str = "bias",
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Module 3.5: Epipolar-guided candidate band (placeholder).

    Simplified mechanism:
      rotate bearing_a by R and keep candidates in B whose angular distance is within threshold.
      This is NOT the strict epipolar band; it is a structural placeholder.

    Returns:
      bias: [B,NA,NB] float (added to similarity), or None
      mask: [B,NA,NB] bool (True allowed), or None

    TODO: Replace with strict epipolar band derived from (R, t_dir) and spherical geometry.
    """
    B, NA, _ = bearing_a.shape
    NB = bearing_b.shape[1]
    # rotate A bearings into B frame (approx)
    # bA_rot = R * bA  (choose convention; placeholder)
    bA_rot = torch.matmul(bearing_a, R.transpose(-1, -2))  # [B,NA,3]
    bA_rot = F.normalize(bA_rot, dim=-1)

    bB = F.normalize(bearing_b, dim=-1)
    cos = torch.matmul(bA_rot, bB.transpose(-1, -2))  # [B,NA,NB]
    cos = torch.clamp(cos, -1.0, 1.0)
    ang = torch.acos(cos)  # radians
    thresh = math.radians(angle_thresh_deg)
    allowed = ang <= thresh  # [B,NA,NB] bool

    if mode == "mask":
        return None, allowed
    # bias: outside band -> negative penalty
    bias = torch.zeros_like(cos)
    bias = bias - (~allowed).to(bias.dtype) * bias_strength
    return bias, None


class CoarseInteraction(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.fuse = FuseMLP(D)
        self.pose_head = CoarsePoseHead(D)
        self.rel_head = ReliabilityHead(D)

    def forward(self, TokA: Tokens, TokB: Tokens) -> Dict[str, torch.Tensor]:
        """
        Coarse global interaction:
          Wc_ab, Wc_ba, Fc, Rc, tc_dir, confidences
        """
        sim = cosine_sim(TokA.feat, TokB.feat)          # [B,NcA,NcB]
        Wc_ab = F.softmax(sim, dim=-1)                  # [B,NcA,NcB]
        Wc_ba = F.softmax(sim.transpose(-1, -2), dim=-1)  # [B,NcB,NcA]

        B_att = torch.matmul(Wc_ab, TokB.feat)          # [B,NcA,D]
        Fc = self.fuse(TokA.feat, B_att)                # [B,NcA,D]

        Rc, tc_dir = self.pose_head(Fc)                 # [B,3,3], [B,3]

        cA = self.rel_head(TokA.feat)                   # [B,NcA,1]
        cB = self.rel_head(TokB.feat)                   # [B,NcB,1]

        return {
            "Wc_ab": Wc_ab,
            "Wc_ba": Wc_ba,
            "Fc": Fc,
            "Rc": Rc,
            "tc_dir": tc_dir,
            "cA_c": cA,
            "cB_c": cB,
        }


class FineInteraction(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.fuse = FuseMLP(D)
        self.pose_head = FinePoseHead(D)
        self.rel_head = ReliabilityHead(D)

    def forward(
        self,
        TokA_f: Tokens,
        TokB_f: Tokens,
        Wc_ab: torch.Tensor,     # [B,NcA,NcB]
        Rc: torch.Tensor,        # [B,3,3]
        topk_coarse: int,
        epi_angle_thresh_deg: float,
        epi_bias_strength: float,
        epi_mode: str = "bias",  # "bias" or "mask"
    ) -> Dict[str, torch.Tensor]:
        """
        Fine local refinement with routing + epipolar-guided bias/mask (placeholder).
        """
        assert TokA_f.parent_id is not None and TokB_f.parent_id is not None

        # routing mask from coarse top-k
        routing_mask = build_routing_mask_from_coarse_topk(
            Wc_ab=Wc_ab,
            fine_parent_a=TokA_f.parent_id,
            fine_parent_b=TokB_f.parent_id,
            topk=topk_coarse,
        )  # [B,NfA,NfB] bool

        # epipolar band placeholder
        epi_bias, epi_mask = epipolar_band_bias_or_mask(
            R=Rc,
            bearing_a=TokA_f.bearing,
            bearing_b=TokB_f.bearing,
            angle_thresh_deg=epi_angle_thresh_deg,
            bias_strength=epi_bias_strength,
            mode=epi_mode,
        )

        # similarity
        sim = cosine_sim(TokA_f.feat, TokB_f.feat)  # [B,NfA,NfB]

        # apply routing + epi constraints
        allowed = routing_mask
        if epi_mask is not None:
            allowed = allowed & epi_mask

        #   sim = sim.masked_fill(~allowed, -1e9)
        neg = torch.finfo(sim.dtype).min  # float16时约为 -65504
        sim = sim.masked_fill(~allowed, neg)


        if epi_bias is not None:
            # bias only meaningful where routing allows; outside routing already -inf
            sim = sim + epi_bias

        Wf_ab = F.softmax(sim, dim=-1)  # [B,NfA,NfB]
        Wf_ba = F.softmax(sim.transpose(-1, -2), dim=-1)  # [B,NfB,NfA]

        B_att = torch.matmul(Wf_ab, TokB_f.feat)         # [B,NfA,D]
        Ff = self.fuse(TokA_f.feat, B_att)               # [B,NfA,D]

        R, t_dir = self.pose_head(Ff)

        cA = self.rel_head(TokA_f.feat)                  # [B,NfA,1]
        cB = self.rel_head(TokB_f.feat)                  # [B,NfB,1]

        return {
            "Wf_ab": Wf_ab,
            "Wf_ba": Wf_ba,
            "Ff": Ff,
            "R": R,
            "t_dir": t_dir,
            "routing_mask": routing_mask,
            "epi_bias": epi_bias if epi_bias is not None else torch.zeros_like(sim),
            "allowed_mask": allowed,
            "cA_f": cA,
            "cB_f": cB,
        }


def aggregate_fine_to_coarse(
    Wf_ab: torch.Tensor,             # [B,NfA,NfB]
    parent_a: torch.Tensor,          # [B,NfA]
    parent_b: torch.Tensor,          # [B,NfB]
    NcA: int,
    NcB: int,
) -> torch.Tensor:
    """
    Aggregate fine correspondence to coarse via parent ids.
    Returns Wc_tilde: [B,NcA,NcB]
    """
    B, NfA, NfB = Wf_ab.shape
    device = Wf_ab.device
    Wc_tilde = torch.zeros((B, NcA, NcB), device=device, dtype=Wf_ab.dtype)
    # batch loop for clarity (MVP sizes)
    for b in range(B):
        pa = parent_a[b].view(NfA, 1).expand(NfA, NfB)  # [NfA,NfB]
        pb = parent_b[b].view(1, NfB).expand(NfA, NfB)  # [NfA,NfB]
        flat_idx = pa * NcB + pb                         # [NfA,NfB]
        Wflat = Wf_ab[b].reshape(-1)
        idx = flat_idx.reshape(-1)
        Wacc = torch.zeros((NcA * NcB,), device=device, dtype=Wf_ab.dtype)
        Wacc.scatter_add_(0, idx, Wflat)
        Wc_tilde[b] = Wacc.view(NcA, NcB)

    # normalize row-wise to match Wc_ab scale (optional, improves stability)
    Wc_tilde = Wc_tilde / (Wc_tilde.sum(dim=-1, keepdim=True) + 1e-9)
    return Wc_tilde
