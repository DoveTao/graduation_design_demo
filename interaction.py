"""
File: interaction.py
Description:
    Matching and interaction modules for coarse and fine spherical token sets.
    This file computes similarity matrices, soft correspondence weights,
    feature fusion, pose heads, reliability heads, and epipolar guidance.

Main Components:
    - Tokens dataclass for token features and spherical metadata
    - CoarseInteraction and FineInteraction modules
    - Fusion, reliability, rotation, and translation heads
    - Epipolar band bias/mask utilities and fine-to-coarse aggregation

Usage / Role:
    Implements the core matching and pose-regression logic used by the model.

Notes:
    Supports the MVP coarse matching path, fine-stage routing, GT epipolar
    matching diagnostics, lightweight translation-specific feature fusion, and
    odometry-oriented translation magnitude prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pose_head import normalize_vec, rot6d_to_matrix


@dataclass
class Tokens:
    feat: torch.Tensor           # [B,N,D]
    bearing: torch.Tensor        # [B,N,3]
    level: str                   # "coarse" / "fine"
    id: torch.Tensor             # [B,N]
    parent_id: Optional[torch.Tensor]  # [B,N] or None


class PairPoseHead(nn.Module):
    """Baseline head for the ablation: encoder only + regression."""

    def __init__(self, D: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(4 * D),
            nn.Linear(4 * D, 2 * D),
            nn.GELU(),
            nn.Linear(2 * D, D),
            nn.GELU(),
        )
        self.rot = nn.Linear(D, 6)
        self.t = nn.Linear(D, 3)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        za = feat_a.mean(dim=1)
        zb = feat_b.mean(dim=1)
        z = torch.cat([za, zb, za - zb, za * zb], dim=-1)
        z = self.net(z)
        R = rot6d_to_matrix(self.rot(z))
        t_dir = normalize_vec(self.t(z))
        return R, t_dir


class FuseMLP(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(4 * D),
            nn.Linear(4 * D, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

    def forward(self, a: torch.Tensor, b_att: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, b_att, a - b_att, a * b_att], dim=-1)
        return self.net(x)


class TokenReliabilityHead(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class BearingFuse(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(12),
            nn.Linear(12, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

    def forward(self, bearing_a: torch.Tensor, bearing_b_att: torch.Tensor) -> torch.Tensor:
        a = F.normalize(bearing_a.float(), dim=-1, eps=1e-6)
        b = F.normalize(bearing_b_att.float(), dim=-1, eps=1e-6)
        x = torch.cat([a, b, a - b, a * b], dim=-1)
        return self.net(x)


class CoarsePoseHead(nn.Module):
    def __init__(self, D: int, *, use_stats_pool: bool = False):
        super().__init__()
        self.use_stats_pool = bool(use_stats_pool)
        in_dim = 3 * D if self.use_stats_pool else D
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, D),
            nn.GELU(),
            nn.Linear(D, D),
            nn.GELU(),
        )
        self.rot = nn.Linear(D, 6)
        self.t = nn.Linear(D, 3)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = feat.float()
        if self.use_stats_pool:
            z_mean = feat.mean(dim=1)
            z_max = feat.max(dim=1).values
            z_std = torch.sqrt(feat.var(dim=1, unbiased=False).clamp_min(1e-8))
            z = torch.cat([z_mean, z_max, z_std], dim=-1)
        else:
            z = feat.mean(dim=1)
        z = self.mlp(z)
        R = rot6d_to_matrix(self.rot(z))
        t_dir = normalize_vec(self.t(z))
        return R, t_dir



class FinePoseHead(CoarsePoseHead):
    pass


class TranslationOnlyHead(nn.Module):
    """A slightly stronger translation head than plain mean-pooling.

    It pools token features with confidence-weighted mean + max + std statistics,
    which makes the translation branch more sensitive to a few highly informative
    fine correspondences.
    """

    def __init__(self, D: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(3 * D),
            nn.Linear(3 * D, 2 * D),
            nn.GELU(),
            nn.Linear(2 * D, D),
            nn.GELU(),
        )
        self.t = nn.Linear(D, 3)

    def forward(self, feat: torch.Tensor, token_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = feat.float()
        if token_weight is None:
            z_mean = feat.mean(dim=1)
            z_var = feat.var(dim=1, unbiased=False)
        else:
            w = token_weight.float().clamp_min(1e-6)
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
            z_mean = torch.sum(feat * w.unsqueeze(-1), dim=1)
            diff2 = (feat - z_mean.unsqueeze(1)).pow(2)
            z_var = torch.sum(diff2 * w.unsqueeze(-1), dim=1)
        z_std = torch.sqrt(z_var.clamp_min(1e-8))
        z_max = feat.max(dim=1).values
        z = torch.cat([z_mean, z_max, z_std], dim=-1)
        z = self.net(z)
        return normalize_vec(self.t(z))


class TranslationMagnitudeHead(nn.Module):
    """Predict log relative translation scale from fused token features."""

    def __init__(self, D: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(3 * D),
            nn.Linear(3 * D, D),
            nn.GELU(),
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, 1),
        )

    def forward(self, feat: torch.Tensor, token_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = feat.float()
        if token_weight is None:
            z_mean = feat.mean(dim=1)
            z_var = feat.var(dim=1, unbiased=False)
        else:
            w = token_weight.float().clamp_min(1e-6)
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
            z_mean = torch.sum(feat * w.unsqueeze(-1), dim=1)
            diff2 = (feat - z_mean.unsqueeze(1)).pow(2)
            z_var = torch.sum(diff2 * w.unsqueeze(-1), dim=1)
        z_std = torch.sqrt(z_var.clamp_min(1e-8))
        z_max = feat.max(dim=1).values
        z = torch.cat([z_mean, z_max, z_std], dim=-1)
        return self.net(z).squeeze(-1)


def positive_translation_magnitude(
    log_t_mag: torch.Tensor,
    *,
    clamp_min: float,
    clamp_max: float,
    mag_min: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    log_t_mag = log_t_mag.float().view(-1).clamp(min=float(clamp_min), max=float(clamp_max))
    t_mag = torch.exp(log_t_mag).clamp_min(float(mag_min))
    return t_mag, log_t_mag


def cosine_logits(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    temperature: float,
    logits_clip: float,
) -> torch.Tensor:
    a = F.normalize(a.float(), dim=-1, eps=1e-6)
    b = F.normalize(b.float(), dim=-1, eps=1e-6)
    logits = torch.matmul(a, b.transpose(-1, -2)) / max(float(temperature), 1e-6)
    if logits_clip is not None and logits_clip > 0:
        logits = logits.clamp(min=-float(logits_clip), max=float(logits_clip))
    return logits


def stable_softmax(logits: torch.Tensor, dim: int = -1, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    logits = logits.float()
    if mask is not None:
        mask = mask.to(torch.bool)
        valid = mask.any(dim=dim, keepdim=True)
        if not torch.all(valid):
            fallback = torch.zeros_like(mask)
            fallback_idx = logits.argmax(dim=dim, keepdim=True)
            fallback.scatter_(dim, fallback_idx, True)
            mask = torch.where(valid, mask, fallback)
        logits = logits.masked_fill(~mask, -1e4)
        logits = logits - logits.max(dim=dim, keepdim=True).values
        ex = torch.exp(logits) * mask.to(logits.dtype)
        denom = ex.sum(dim=dim, keepdim=True).clamp_min(1e-9)
        return ex / denom

    logits = logits - logits.max(dim=dim, keepdim=True).values
    ex = torch.exp(logits)
    return ex / ex.sum(dim=dim, keepdim=True).clamp_min(1e-9)


def build_routing_mask_from_coarse_topk(
    Wc_ab: torch.Tensor,
    fine_parent_a: torch.Tensor,
    fine_parent_b: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    B, NcA, NcB = Wc_ab.shape
    device = Wc_ab.device
    k = min(int(topk), int(NcB))
    _, topk_idx = torch.topk(Wc_ab, k=k, dim=-1)
    allowed_parent = torch.zeros((B, NcA, NcB), device=device, dtype=torch.bool)
    allowed_parent.scatter_(dim=-1, index=topk_idx, value=True)

    masks = []
    for b in range(B):
        pa = fine_parent_a[b]
        pb = fine_parent_b[b]
        mask = allowed_parent[b, pa][:, pb]
        masks.append(mask)
    return torch.stack(masks, dim=0)


def epipolar_band_bias_or_mask(
    R: torch.Tensor,
    t_dir: torch.Tensor,
    bearing_a: torch.Tensor,
    bearing_b: torch.Tensor,
    angle_thresh_deg: float,
    bias_strength: float,
    mode: str = "bias",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor, float]:
    """
    Translation-aware spherical epipolar prior.

    For a correspondence (a,b), ideal geometry satisfies:
        b^T [t]_x R a = 0
    We convert that into a soft band around the epipolar great circle and use it
    either as a continuous logit bias or as a hard mask.

    Returns:
      bias, allowed_mask, residual, normalized_cost, band_sin
    """
    bA_rot = torch.matmul(bearing_a.float(), R.float().transpose(-1, -2))
    bA_rot = F.normalize(bA_rot, dim=-1, eps=1e-6)
    bB = F.normalize(bearing_b.float(), dim=-1, eps=1e-6)
    t_dir = F.normalize(t_dir.float(), dim=-1, eps=1e-6)

    t_expand = t_dir[:, None, :].expand_as(bA_rot)
    plane_n = torch.cross(t_expand, bA_rot, dim=-1)                 # [B,Na,3]
    plane_norm = plane_n.norm(dim=-1, keepdim=True)                 # [B,Na,1]
    safe_plane = plane_norm > 1e-6
    plane_n = plane_n / plane_norm.clamp_min(1e-6)

    residual = torch.abs(torch.einsum("bnc,bmc->bnm", plane_n, bB)).clamp(0.0, 1.0)
    residual = torch.where(safe_plane.expand_as(residual), residual, torch.zeros_like(residual))

    band = math.sin(math.radians(float(angle_thresh_deg)))
    band = max(band, 1e-6)
    allowed = residual <= band
    allowed = allowed | (~safe_plane).expand_as(allowed)

    over = (residual - band).clamp_min(0.0) / band
    over = torch.where(safe_plane.expand_as(over), over, torch.zeros_like(over))

    if mode == "mask":
        return None, allowed, residual, over, float(band)

    bias = -float(bias_strength) * over
    bias = torch.where(safe_plane.expand_as(bias), bias, torch.zeros_like(bias))
    return bias, None, residual, over, float(band)


class CoarseInteraction(nn.Module):
    def __init__(
        self,
        D: int,
        *,
        temperature: float,
        logits_clip: float,
        pose_use_stats_pool: bool = False,
        use_bearing_fuse: bool = False,
        use_translation_feature_branch: bool = False,
        translation_branch_detach_match: bool = True,
        use_translation_magnitude_head: bool = True,
        tmag_pred_source: str = "translation_branch",
        tmag_detach_features: bool = False,
        tmag_min: float = 1.0e-3,
        log_tmag_clamp_min: float = -6.0,
        log_tmag_clamp_max: float = 6.0,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.logits_clip = float(logits_clip)
        self.use_bearing_fuse = bool(use_bearing_fuse)
        self.use_translation_feature_branch = bool(use_translation_feature_branch)
        self.translation_branch_detach_match = bool(translation_branch_detach_match)
        self.use_translation_magnitude_head = bool(use_translation_magnitude_head)
        self.tmag_pred_source = str(tmag_pred_source)
        if self.tmag_pred_source not in ("translation_branch", "pose_feat"):
            raise ValueError(f"Unsupported tmag_pred_source: {self.tmag_pred_source}")
        self.tmag_detach_features = bool(tmag_detach_features)
        self.tmag_min = float(tmag_min)
        self.log_tmag_clamp_min = float(log_tmag_clamp_min)
        self.log_tmag_clamp_max = float(log_tmag_clamp_max)
        self.fuse = FuseMLP(D)
        self.bearing_fuse = BearingFuse(D) if self.use_bearing_fuse else None
        self.pose_head = CoarsePoseHead(D, use_stats_pool=pose_use_stats_pool)
        self.t_fuse = FuseMLP(D) if self.use_translation_feature_branch else None
        self.t_head = TranslationOnlyHead(D) if self.use_translation_feature_branch else None
        self.mag_head = TranslationMagnitudeHead(D) if self.use_translation_magnitude_head else None
        self.rel_head = TokenReliabilityHead(D)

    def forward(
        self,
        TokA: Tokens,
        TokB: Tokens,
        *,
        tokens_a_t: Optional[Tokens] = None,
        tokens_b_t: Optional[Tokens] = None,
    ) -> Dict[str, torch.Tensor]:
        sim = cosine_logits(
            TokA.feat,
            TokB.feat,
            temperature=self.temperature,
            logits_clip=self.logits_clip,
        )
        Wc_ab = stable_softmax(sim, dim=-1)
        Wc_ba = stable_softmax(sim.transpose(-1, -2), dim=-1)

        feat_b_att = torch.matmul(Wc_ab, TokB.feat.float())
        Fc = self.fuse(TokA.feat.float(), feat_b_att)
        if self.bearing_fuse is not None:
            bearing_b_att = torch.matmul(Wc_ab, TokB.bearing.float())
            Fc = Fc + self.bearing_fuse(TokA.bearing, bearing_b_att)
        Rc, tc_dir = self.pose_head(Fc)
        Fc_t = None
        tc_dir_pose = tc_dir
        if self.t_fuse is not None and self.t_head is not None and tokens_a_t is not None and tokens_b_t is not None:
            W_t = Wc_ab.detach() if self.translation_branch_detach_match else Wc_ab
            feat_b_att_t = torch.matmul(W_t, tokens_b_t.feat.float())
            Fc_t = self.t_fuse(tokens_a_t.feat.float(), feat_b_att_t)
            match_conf = W_t.max(dim=-1).values
            tc_dir = self.t_head(Fc_t, token_weight=match_conf)
        else:
            match_conf = Wc_ab.max(dim=-1).values
        if self.mag_head is not None:
            mag_feat = Fc if self.tmag_pred_source == "pose_feat" else (Fc_t if Fc_t is not None else Fc)
            mag_weight = match_conf
            if self.tmag_detach_features:
                mag_feat = mag_feat.detach()
                mag_weight = mag_weight.detach()
            log_tc_mag = self.mag_head(mag_feat, token_weight=mag_weight)
            tc_mag, log_tc_mag = positive_translation_magnitude(
                log_tc_mag,
                clamp_min=self.log_tmag_clamp_min,
                clamp_max=self.log_tmag_clamp_max,
                mag_min=self.tmag_min,
            )
        else:
            tc_mag = torch.ones((TokA.feat.shape[0],), device=TokA.feat.device, dtype=torch.float32)
            log_tc_mag = torch.zeros_like(tc_mag)

        out = {
            "Wc_ab": Wc_ab,
            "Wc_ba": Wc_ba,
            "Fc": Fc,
            "Rc": Rc,
            "tc_dir": tc_dir,
            "tc_mag": tc_mag,
            "log_tc_mag": log_tc_mag,
            "tc_dir_pose": tc_dir_pose,
            "cA_c": self.rel_head(TokA.feat.float()),
            "cB_c": self.rel_head(TokB.feat.float()),
            "logits_c": sim,
        }
        if Fc_t is not None:
            out["Fc_t"] = Fc_t
        return out


class FineInteraction(nn.Module):
    def __init__(
        self,
        D: int,
        *,
        temperature: float,
        logits_clip: float,
        use_depth_fusion: bool = False,
        depth_fuse_strength: float = 1.0,
        depth_fuse_detach_feature: bool = False,
        use_geometric_t_fusion: bool = False,
        geometric_t_fuse_strength: float = 0.0,
        pose_use_stats_pool: bool = False,
        use_bearing_fuse: bool = False,
        use_translation_magnitude_head: bool = True,
        tmag_detach_features: bool = False,
        tmag_min: float = 1.0e-3,
        log_tmag_clamp_min: float = -6.0,
        log_tmag_clamp_max: float = 6.0,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.logits_clip = float(logits_clip)
        self.use_bearing_fuse = bool(use_bearing_fuse)
        self.fuse = FuseMLP(D)
        self.bearing_fuse = BearingFuse(D) if self.use_bearing_fuse else None
        self.pose_head = FinePoseHead(D, use_stats_pool=pose_use_stats_pool)
        self.rel_head = TokenReliabilityHead(D)
        self.use_depth_fusion = bool(use_depth_fusion)
        self.depth_fuse_strength = float(depth_fuse_strength)
        self.depth_fuse_detach_feature = bool(depth_fuse_detach_feature)
        self.use_geometric_t_fusion = bool(use_geometric_t_fusion)
        self.geometric_t_fuse_strength = float(geometric_t_fuse_strength)
        self.use_translation_magnitude_head = bool(use_translation_magnitude_head)
        self.tmag_detach_features = bool(tmag_detach_features)
        self.tmag_min = float(tmag_min)
        self.log_tmag_clamp_min = float(log_tmag_clamp_min)
        self.log_tmag_clamp_max = float(log_tmag_clamp_max)
        if self.use_depth_fusion:
            self.depth_fuse = nn.Sequential(
                nn.LayerNorm(2 * D),
                nn.Linear(2 * D, D),
                nn.GELU(),
                nn.Linear(D, D),
            )
            self.depth_gate = nn.Sequential(
                nn.LayerNorm(2 * D),
                nn.Linear(2 * D, D),
                nn.GELU(),
                nn.Linear(D, D),
                nn.Sigmoid(),
            )
        self.t_head = TranslationOnlyHead(D)
        self.mag_head = TranslationMagnitudeHead(D) if self.use_translation_magnitude_head else None

    @staticmethod
    def _geometric_translation_from_matches(
        W_ab: torch.Tensor,
        bearing_a: torch.Tensor,
        bearing_b: torch.Tensor,
        R: torch.Tensor,
        token_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        W_ab = W_ab.float()
        bearing_a = F.normalize(bearing_a.float(), dim=-1, eps=1e-6)
        bearing_b = F.normalize(bearing_b.float(), dim=-1, eps=1e-6)
        R = R.float()

        a_in_b = torch.matmul(bearing_a, R.transpose(-1, -2))
        b_match = torch.matmul(W_ab, bearing_b)
        b_match = F.normalize(b_match, dim=-1, eps=1e-6)

        plane_n = torch.cross(a_in_b, b_match, dim=-1)
        plane_norm = torch.linalg.norm(plane_n, dim=-1)
        plane_n = F.normalize(plane_n, dim=-1, eps=1e-6)

        if token_weight is None:
            w = plane_norm
        else:
            w = token_weight.float() * plane_norm
        w = w.clamp_min(1e-6)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        cov = torch.einsum("bn,bni,bnj->bij", w, plane_n, plane_n).float()
        eye = torch.eye(3, device=cov.device, dtype=torch.float32).unsqueeze(0)
        cov = cov + 1e-5 * eye
        if cov.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                _, vec = torch.linalg.eigh(cov)
        else:
            _, vec = torch.linalg.eigh(cov)
        t_out = F.normalize(vec[..., 0], dim=-1, eps=1e-6)
        t_local = F.normalize(torch.matmul(R.transpose(-1, -2), t_out.unsqueeze(-1)).squeeze(-1), dim=-1, eps=1e-6)
        return t_out, t_local

    def forward(
        self,
        TokA_f: Tokens,
        TokB_f: Tokens,
        Wc_ab: torch.Tensor,
        Rc: torch.Tensor,
        tc_dir: torch.Tensor,
        *,
        depth_tok_a: Optional[torch.Tensor] = None,
        topk_coarse: int,
        use_epipolar_bias: bool,
        epi_angle_thresh_deg: float,
        epi_bias_strength: float,
        epi_mode: str = "bias",
    ) -> Dict[str, torch.Tensor]:
        assert TokA_f.parent_id is not None and TokB_f.parent_id is not None

        routing_mask = build_routing_mask_from_coarse_topk(
            Wc_ab=Wc_ab,
            fine_parent_a=TokA_f.parent_id,
            fine_parent_b=TokB_f.parent_id,
            topk=topk_coarse,
        )

        epi_bias, epi_mask, epi_residual, epi_cost, epi_band_sin = None, None, None, None, None
        if use_epipolar_bias:
            epi_bias, epi_mask, epi_residual, epi_cost, epi_band_sin = epipolar_band_bias_or_mask(
                R=Rc,
                t_dir=tc_dir,
                bearing_a=TokA_f.bearing,
                bearing_b=TokB_f.bearing,
                angle_thresh_deg=epi_angle_thresh_deg,
                bias_strength=epi_bias_strength,
                mode=epi_mode,
            )

        allowed = routing_mask
        if epi_mask is not None:
            allowed = allowed & epi_mask
            row_has_valid = allowed.any(dim=-1, keepdim=True)
            allowed = torch.where(row_has_valid, allowed, routing_mask)

        sim = cosine_logits(
            TokA_f.feat,
            TokB_f.feat,
            temperature=self.temperature,
            logits_clip=self.logits_clip,
        )

        # Before geometry bias: useful for vis/ablation plots.
        Wf_ab_raw = stable_softmax(sim, dim=-1, mask=routing_mask)
        Wf_ba_raw = stable_softmax(sim.transpose(-1, -2).contiguous(), dim=-1, mask=routing_mask.transpose(-1, -2).contiguous())

        sim_ab = sim.clone()
        if epi_bias is not None:
            sim_ab = sim_ab + epi_bias.float()
        Wf_ab = stable_softmax(sim_ab, dim=-1, mask=allowed)

        sim_ba = sim.transpose(-1, -2).contiguous()
        allowed_ba = allowed.transpose(-1, -2).contiguous()
        if epi_bias is not None:
            sim_ba = sim_ba + epi_bias.transpose(-1, -2).contiguous().float()
        Wf_ba = stable_softmax(sim_ba, dim=-1, mask=allowed_ba)


        feat_b_att = torch.matmul(Wf_ab, TokB_f.feat.float())
        Ff = self.fuse(TokA_f.feat.float(), feat_b_att)
        if self.bearing_fuse is not None:
            bearing_b_att = torch.matmul(Wf_ab, TokB_f.bearing.float())
            Ff = Ff + self.bearing_fuse(TokA_f.bearing, bearing_b_att)
        R, _ = self.pose_head(Ff)

        if self.use_depth_fusion and depth_tok_a is not None:
            depth_feat = depth_tok_a.detach() if self.depth_fuse_detach_feature else depth_tok_a
            fuse_in = torch.cat([Ff, depth_feat.float()], dim=-1)
            depth_delta = self.depth_fuse(fuse_in)
            depth_gate = self.depth_gate(fuse_in)
            Ff_t = Ff + self.depth_fuse_strength * (depth_gate * depth_delta)
        else:
            Ff_t = Ff

        match_conf = Wf_ab.max(dim=-1).values
        if epi_cost is not None:
            geom_conf = torch.exp(-epi_cost.detach().min(dim=-1).values)
            token_weight = match_conf * geom_conf
        else:
            token_weight = match_conf
        t_dir_raw = self.t_head(Ff_t, token_weight=token_weight)
        if self.mag_head is not None:
            mag_feat = Ff_t.detach() if self.tmag_detach_features else Ff_t
            mag_weight = token_weight.detach() if self.tmag_detach_features else token_weight
            log_t_mag = self.mag_head(mag_feat, token_weight=mag_weight)
            t_mag, log_t_mag = positive_translation_magnitude(
                log_t_mag,
                clamp_min=self.log_tmag_clamp_min,
                clamp_max=self.log_tmag_clamp_max,
                mag_min=self.tmag_min,
            )
        else:
            t_mag = torch.ones((TokA_f.feat.shape[0],), device=TokA_f.feat.device, dtype=torch.float32)
            log_t_mag = torch.zeros_like(t_mag)
        t_geo_out, t_geo_local = self._geometric_translation_from_matches(
            Wf_ab,
            TokA_f.bearing,
            TokB_f.bearing,
            R,
            token_weight=token_weight,
        )
        align = torch.sign(torch.sum(t_geo_local.detach() * t_dir_raw.detach(), dim=-1, keepdim=True))
        align = torch.where(align == 0, torch.ones_like(align), align)
        t_geo_local = t_geo_local * align
        t_geo_out = t_geo_out * align
        if self.use_geometric_t_fusion and self.geometric_t_fuse_strength > 0.0:
            t_dir = F.normalize(t_dir_raw + self.geometric_t_fuse_strength * t_geo_local.detach(), dim=-1, eps=1e-6)
        else:
            t_dir = t_dir_raw

        return {
            "Wf_ab": Wf_ab,
            "Wf_ba": Wf_ba,
            "Wf_ab_raw": Wf_ab_raw,
            "Wf_ba_raw": Wf_ba_raw,
            "Ff": Ff,
            "Ff_t": Ff_t,
            "R": R,
            "t_dir": t_dir,
            "t_mag": t_mag,
            "log_t_mag": log_t_mag,
            "t_dir_raw": t_dir_raw,
            "t_geo_out": t_geo_out,
            "t_geo_local": t_geo_local,
            "routing_mask": routing_mask,
            "allowed_mask": allowed,
            "epi_bias": epi_bias if epi_bias is not None else torch.zeros_like(sim_ab),
            "epi_residual": epi_residual if epi_residual is not None else torch.zeros_like(sim_ab),
            "epi_cost": epi_cost if epi_cost is not None else torch.zeros_like(sim_ab),
            "epi_band_sin": torch.tensor(float(epi_band_sin if epi_band_sin is not None else 0.0), device=sim_ab.device),
            "cA_f": self.rel_head(TokA_f.feat.float()),
            "cB_f": self.rel_head(TokB_f.feat.float()),
            "logits_f": sim,
            "logits_f_biased": sim_ab,
            "allowed_mask_ba": allowed_ba,
            "token_weight_f": token_weight,
        }


def aggregate_fine_to_coarse(
    Wf_ab: torch.Tensor,
    parent_a: torch.Tensor,
    parent_b: torch.Tensor,
    NcA: int,
    NcB: int,
) -> torch.Tensor:
    B, NfA, NfB = Wf_ab.shape
    Wc_tilde = torch.zeros((B, NcA, NcB), device=Wf_ab.device, dtype=Wf_ab.dtype)
    for b in range(B):
        pa = parent_a[b].view(NfA, 1).expand(NfA, NfB)
        pb = parent_b[b].view(1, NfB).expand(NfA, NfB)
        flat_idx = (pa * NcB + pb).reshape(-1)
        flat_val = Wf_ab[b].reshape(-1)
        acc = torch.zeros((NcA * NcB,), device=Wf_ab.device, dtype=Wf_ab.dtype)
        acc.scatter_add_(0, flat_idx, flat_val)
        Wc_tilde[b] = acc.view(NcA, NcB)
    return Wc_tilde / Wc_tilde.sum(dim=-1, keepdim=True).clamp_min(1e-9)
