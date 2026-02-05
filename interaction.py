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
    a = F.normalize(a, dim=-1, eps=1e-6)
    b = F.normalize(b, dim=-1, eps=1e-6)
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
    Simple learned confidence per token: logits = MLP(feat).
    Use sigmoid(logits) externally when probability is needed.
    """
    def __init__(self, D: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B,N,D] -> logits: [B,N,1]
        return self.net(feat)


def build_routing_mask_from_coarse_topk(
    Wc_ab: torch.Tensor,          # [B,NcA,NcB]
    fine_parent_a: torch.Tensor,  # [B,NfA] long
    fine_parent_b: torch.Tensor,  # [B,NfB] long
    topk: int,
) -> torch.Tensor:
    """
    Build fine pair mask via parent mapping:
      allow (a_f, b_f) if (parent_a, parent_b) is within top-k coarse correspondences.

    Returns mask: [B,NfA,NfB] bool (True = allowed)

    Note: This is a dense mask builder (kept for debugging / visualization).
    FineInteraction below uses a sparse / blockwise implementation and does NOT call this.
    """
    B, NcA, NcB = Wc_ab.shape
    device = Wc_ab.device
    _, topk_idx = torch.topk(Wc_ab, k=min(topk, NcB), dim=-1)  # [B,NcA,topk]
    allowed_parent = torch.zeros((B, NcA, NcB), device=device, dtype=torch.bool)
    allowed_parent.scatter_(dim=-1, index=topk_idx, value=True)

    masks = []
    for b in range(B):
        ap = fine_parent_a[b]  # [NfA]
        bp = fine_parent_b[b]  # [NfB]
        allowed_ap = allowed_parent[b, ap]          # [NfA,NcB]
        mask = allowed_ap[:, bp]                    # [NfA,NfB]
        masks.append(mask)
    return torch.stack(masks, dim=0)  # [B,NfA,NfB]


def strict_spherical_epipolar_band_mask_or_bias(
    R: torch.Tensor,             # [B,3,3]
    t_dir: torch.Tensor,         # [B,3]
    bearing_a: torch.Tensor,     # [B,NA,3]
    bearing_b: torch.Tensor,     # [B,NB,3]
    angle_thresh_deg: float,
    bias_strength: float,
    mode: str = "bias",
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Strict spherical epipolar band derived from (R, t_dir).

    For each A-ray bA, rotate into B: r = R*bA.
    Epipolar plane normal: n = t x r.
    Great-circle constraint: n · bB = 0.
    Band of half-width δ: |n_unit · bB| <= sin(δ).

    Returns:
      bias: [B,NA,NB] float (added to similarity), or None
      mask: [B,NA,NB] bool (True allowed), or None

    This is a dense helper (for debugging). FineInteraction uses a sparse / blockwise version.
    """
    B, NA, _ = bearing_a.shape
    NB = bearing_b.shape[1]
    device = bearing_a.device

    # Normalize (fp32)
    t = F.normalize(t_dir.float(), dim=-1, eps=1e-6)  # [B,3]
    r = torch.matmul(bearing_a.float(), R.transpose(-1, -2).float())  # [B,NA,3]
    r = F.normalize(r, dim=-1, eps=1e-6)
    bB = F.normalize(bearing_b.float(), dim=-1, eps=1e-6)  # [B,NB,3]

    t_exp = t[:, None, :].expand(-1, NA, -1)               # [B,NA,3]
    n = torch.cross(t_exp, r, dim=-1)                      # [B,NA,3]
    n_norm = n.norm(dim=-1, keepdim=True)                  # [B,NA,1]
    valid = (n_norm > 1e-3)                                # [B,NA,1]
    n_unit = n / (n_norm + 1e-6)

    s = torch.abs(torch.matmul(n_unit, bB.transpose(-1, -2)))  # [B,NA,NB]
    sin_th = math.sin(math.radians(angle_thresh_deg))
    allowed = (s <= sin_th) | (~valid).expand(-1, -1, NB)

    if mode == "mask":
        return None, allowed

    bias = torch.zeros((B, NA, NB), device=device, dtype=torch.float32)
    bias = bias - (~allowed).to(bias.dtype) * float(bias_strength)
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
        sim = cosine_sim(TokA.feat, TokB.feat)               # [B,NcA,NcB]
        Wc_ab = F.softmax(sim, dim=-1)                       # [B,NcA,NcB]
        Wc_ba = F.softmax(sim.transpose(-1, -2), dim=-1)     # [B,NcB,NcA]

        B_att = torch.matmul(Wc_ab, TokB.feat)               # [B,NcA,D]
        Fc = self.fuse(TokA.feat, B_att)                     # [B,NcA,D]

        Rc, tc_dir = self.pose_head(Fc)                      # [B,3,3], [B,3]

        cA = self.rel_head(TokA.feat)                        # [B,NcA,1]
        cB = self.rel_head(TokB.feat)                        # [B,NcB,1]

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
    """
    Fine local refinement using *true sparse / blockwise* similarity computation.

    Instead of building dense sim [B,Nf,Nf] via a full matmul, we:
      1) Route fine tokens by coarse top-k (parent mapping).
      2) (Optional) Apply strict spherical epipolar band (R, t_dir) inside each block.
      3) Compute similarities only within each routed candidate block.
      4) Compute softmax row-wise over the *candidate list* (not over all NfB).
      5) Build B_att using candidate-only weights (no dense W @ feat matmul).

    We still return dense Wf_ab/Wf_ba for downstream losses and aggregation,
    but they are constructed via sparse fills (zeros outside candidate support).
    """
    def __init__(self, D: int):
        super().__init__()
        self.fuse = FuseMLP(D)
        self.pose_head = FinePoseHead(D)
        self.rel_head = ReliabilityHead(D)

    @staticmethod
    def _build_buckets(parent: torch.Tensor, n_parent: int) -> list[torch.Tensor]:
        """
        parent: [N] long in [0, n_parent-1]
        returns list of tensors, where buckets[p] = indices i with parent[i]==p
        """
        buckets: list[torch.Tensor] = []
        for p in range(n_parent):
            idx = torch.nonzero(parent == p, as_tuple=False).squeeze(1)
            buckets.append(idx)
        return buckets

    def forward(
        self,
        TokA_f: Tokens,
        TokB_f: Tokens,
        Wc_ab: torch.Tensor,      # [B,NcA,NcB]
        Rc: torch.Tensor,         # [B,3,3]
        tc_dir: torch.Tensor,     # [B,3]
        topk_coarse: int,
        epi_angle_thresh_deg: float,
        epi_bias_strength: float,
        epi_mode: str = "bias",   # "bias" or "mask"
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
          Wf_ab: [B,NfA,NfB]
          Wf_ba: [B,NfB,NfA]
          Ff:    [B,NfA,D]
          R,t_dir predicted at fine
          routing_mask, allowed_mask, epi_bias (dense, for debugging/visualization)
          cA_f, cB_f
        """
        assert TokA_f.parent_id is not None and TokB_f.parent_id is not None

        B, NfA, D = TokA_f.feat.shape
        NfB = TokB_f.feat.shape[1]
        device = TokA_f.feat.device

        # We assume symmetric coarse sizes in this MVP (NcA==NcB==cfg.Nc)
        NcA = Wc_ab.shape[1]
        NcB = Wc_ab.shape[2]

        # Normalize features in float32 for stable dot products under AMP
        a_norm = F.normalize(TokA_f.feat.float(), dim=-1, eps=1e-6)  # [B,NfA,D] fp32
        b_norm = F.normalize(TokB_f.feat.float(), dim=-1, eps=1e-6)  # [B,NfB,D] fp32

        # Bearings (fp32)
        bearing_a = F.normalize(TokA_f.bearing.float(), dim=-1, eps=1e-6)  # [B,NfA,3]
        bearing_b = F.normalize(TokB_f.bearing.float(), dim=-1, eps=1e-6)  # [B,NfB,3]

        # Outputs (dense containers, filled sparsely)
        Wf_ab = torch.zeros((B, NfA, NfB), device=device, dtype=torch.float32)
        # store final logits used for matching (needed to build Wf_ba like softmax(sim^T))
        sim_dense = torch.full((B, NfA, NfB), -1e9, device=device, dtype=torch.float32)

        routing_mask = torch.zeros((B, NfA, NfB), device=device, dtype=torch.bool)
        allowed_mask = torch.zeros((B, NfA, NfB), device=device, dtype=torch.bool)

        epi_bias = torch.zeros((B, NfA, NfB), device=device, dtype=torch.float32)

        # Also build B-attention output sparsely to avoid dense W @ feat
        B_att_out = torch.zeros((B, NfA, D), device=device, dtype=TokA_f.feat.dtype)

        sin_th = math.sin(math.radians(epi_angle_thresh_deg))

        for b in range(B):
            parent_a = TokA_f.parent_id[b].long()  # [NfA]
            parent_b = TokB_f.parent_id[b].long()  # [NfB]

            # Top-k coarse indices per coarse-A parent
            Wc = Wc_ab[b].float()
            _, topk_idx = torch.topk(Wc, k=min(topk_coarse, NcB), dim=-1)  # [NcA,k]

            # Buckets: fine indices grouped by coarse parent
            b_buckets = self._build_buckets(parent_b, NcB)
            a_buckets = self._build_buckets(parent_a, NcA)

            # Pose in B-frame (coarse estimate)
            Rb = Rc[b].float()          # [3,3]
            tb = tc_dir[b].float()      # [3]
            tb = F.normalize(tb, dim=-1, eps=1e-6)

            for pA in range(NcA):
                a_idx = a_buckets[pA]
                if a_idx.numel() == 0:
                    continue

                cand_parents = topk_idx[pA]  # [k]
                # concatenate fine B indices from these parents
                cand_list = []
                for pj in cand_parents.tolist():
                    if b_buckets[pj].numel() > 0:
                        cand_list.append(b_buckets[pj])
                if len(cand_list) == 0:
                    # degenerate: no candidates (should be rare); fall back to all B
                    cand_idx = torch.arange(NfB, device=device)
                else:
                    cand_idx = torch.cat(cand_list, dim=0)

                # Mark routing support
                routing_mask[b][a_idx[:, None], cand_idx[None, :]] = True

                # --- Compute block logits: sim(a,b) for candidates only
                sim_block = a_norm[b, a_idx] @ b_norm[b, cand_idx].T  # [nA,nC]
                sim_block = sim_block.float()  # ✅ 强制 fp32，避免 autocast 变 half

                # --- Strict spherical epipolar band inside this block (optional mask or bias)
                # r = R*bA (row-vector convention -> bA @ R^T)
                r = bearing_a[b, a_idx] @ Rb.transpose(0, 1)          # [nA,3]
                r = F.normalize(r, dim=-1, eps=1e-6)
                t_exp = tb[None, :].expand(r.shape[0], -1)            # [nA,3]
                n = torch.cross(t_exp, r, dim=-1)                     # [nA,3]
                n_norm = n.norm(dim=-1, keepdim=True)                 # [nA,1]
                valid = (n_norm > 1e-3)                               # [nA,1]
                n_unit = n / (n_norm + 1e-6)

                bB = bearing_b[b, cand_idx]                           # [nC,3]
                s = torch.abs(n_unit @ bB.transpose(0, 1))            # [nA,nC]
                epi_allowed = (s <= sin_th) | (~valid).expand(-1, s.shape[1])

                if epi_mode == "mask":
                    sim_block = sim_block.masked_fill(~epi_allowed, -1e9)
                    allowed_here = epi_allowed
                else:
                    # bias mode: all routed candidates remain allowed; penalize outside band
                    penalty = (~epi_allowed).to(torch.float32) * float(epi_bias_strength)  # ✅ fp32
                    sim_block = sim_block - penalty
                    allowed_here = torch.ones_like(epi_allowed, dtype=torch.bool)

                    # record bias for visualization/debug (dtype must match for index_put)
                    epi_bias[b][a_idx[:, None], cand_idx[None, :]] = (-penalty).to(epi_bias.dtype)

                # record allowed_mask (for debug)
                allowed_mask[b][a_idx[:, None], cand_idx[None, :]] = allowed_here

                # store logits for Wf_ba construction (dtype-safe)
                sim_dense[b][a_idx[:, None], cand_idx[None, :]] = sim_block.to(sim_dense.dtype)

                # --- Row-wise softmax over candidates (sparse)
                W_block = F.softmax(sim_block, dim=-1).to(Wf_ab.dtype)  # ✅ fp32

                # fill dense Wf_ab support
                Wf_ab[b][a_idx[:, None], cand_idx[None, :]] = W_block

                # candidate-only attention to TokB features (avoid dense W @ feat)
                b_feat_raw = TokB_f.feat[b, cand_idx]               # [nC,D]
                B_att_block = (W_block @ b_feat_raw.float()).to(TokA_f.feat.dtype)  # [nA,D]
                B_att_out[b, a_idx] = B_att_block

        # Build Wf_ba like original: softmax(sim^T) over A for each B (dense softmax is cheap)
        Wf_ba = F.softmax(sim_dense.transpose(-1, -2), dim=-1)  # [B,NfB,NfA] fp32

        # Fuse + fine pose
        Ff = self.fuse(TokA_f.feat, B_att_out)  # [B,NfA,D]
        R, t_dir = self.pose_head(Ff)

        cA = self.rel_head(TokA_f.feat)  # [B,NfA,1]
        cB = self.rel_head(TokB_f.feat)  # [B,NfB,1]

        return {
            "Wf_ab": Wf_ab,
            "Wf_ba": Wf_ba,
            "Ff": Ff,
            "R": R,
            "t_dir": t_dir,
            "routing_mask": routing_mask,
            "epi_bias": epi_bias,
            "allowed_mask": allowed_mask if epi_mode == "mask" else routing_mask,
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
    for b in range(B):
        pa = parent_a[b].view(NfA, 1).expand(NfA, NfB)  # [NfA,NfB]
        pb = parent_b[b].view(1, NfB).expand(NfA, NfB)  # [NfA,NfB]
        flat_idx = pa * NcB + pb                        # [NfA,NfB]
        Wflat = Wf_ab[b].reshape(-1)
        idx = flat_idx.reshape(-1)
        Wacc = torch.zeros((NcA * NcB,), device=device, dtype=Wf_ab.dtype)
        Wacc.scatter_add_(0, idx, Wflat)
        Wc_tilde[b] = Wacc.view(NcA, NcB)

    Wc_tilde = Wc_tilde / (Wc_tilde.sum(dim=-1, keepdim=True) + 1e-9)
    return Wc_tilde
