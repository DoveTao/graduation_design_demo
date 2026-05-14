"""
File: model.py
Description:
    Main model definition for panoramic relative pose estimation. This file
    builds sampled spherical tokens, applies token encoders, runs coarse and
    optional fine interactions, and returns rotation, translation direction,
    translation magnitude, and relative transform predictions.

Main Components:
    - Module2Sampler for ERP patch sampling and token construction
    - PanoramaRelPoseModel for end-to-end relative pose prediction
    - Coarse-only and coarse-to-fine inference paths
    - Optional depth fusion and lightweight translation feature branch

Usage / Role:
    Serves as the model assembly layer used by the training and evaluation
    pipeline.

Notes:
    The current MVP emphasizes coarse matching and local-frame translation
    supervision while keeping optional fine-stage, epipolar-bias, odometry-scale
    translation, and depth branches available for ablation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import Config
from .depth_branch import LightERPDepthNet
from .erp_sampling import (
    build_level_grid_from_patch_bearings,
    sample_dense_features_from_bearings,
    sample_patches_erp,
)
from .healpix_utils import HealpixHierarchy
from .interaction import (
    CoarseInteraction,
    CoupledPoseResidualHead,
    FineInteraction,
    PairPoseHead,
    Tokens,
    aggregate_fine_to_coarse,
)
from .transformer_encoder import BearingPosEnc, CrossContextEncoder, PatchEmbed, TokenEncoder


class Module2Sampler(nn.Module):
    def __init__(self, cfg: Config, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.hier = HealpixHierarchy(cfg.Nc, cfg.Nf, cfg.p, device=device)

        level_data = self.hier.level_data()
        grid_c = build_level_grid_from_patch_bearings(level_data["coarse"]["patch_bearing"], cfg.H, cfg.W)
        grid_f = build_level_grid_from_patch_bearings(level_data["fine"]["patch_bearing"], cfg.H, cfg.W)
        self.register_buffer("grid_coarse", grid_c, persistent=True)
        self.register_buffer("grid_fine", grid_f, persistent=True)

        self.patch_embed_c = PatchEmbed(
            cfg.p,
            cfg.D,
            in_ch=cfg.in_ch,
            use_coords=bool(getattr(cfg, "patch_embed_use_coords", False)),
            use_avgmax_pool=bool(getattr(cfg, "patch_embed_avgmax_pool", False)),
            pool_mode=str(getattr(cfg, "patch_embed_pool_mode", "avg")),
            pool_gate_init=float(getattr(cfg, "patch_embed_pool_gate_init", -2.0)),
        )
        self.patch_embed_f = PatchEmbed(
            cfg.p,
            cfg.D,
            in_ch=cfg.in_ch,
            use_coords=bool(getattr(cfg, "patch_embed_use_coords", False)),
            use_avgmax_pool=bool(getattr(cfg, "patch_embed_avgmax_pool", False)),
            pool_mode=str(getattr(cfg, "patch_embed_pool_mode", "avg")),
            pool_gate_init=float(getattr(cfg, "patch_embed_pool_gate_init", -2.0)),
        )
        self.pos_enc = BearingPosEnc(cfg.D)
        self.enc_c = TokenEncoder(cfg.D, cfg.n_layers, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
        self.enc_f = TokenEncoder(cfg.D, cfg.n_layers, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
        self.use_translation_feature_branch = bool(getattr(cfg, "use_translation_feature_branch", False))
        self.patch_embed_c_t = None
        self.enc_c_t = None
        if self.use_translation_feature_branch:
            t_layers = int(getattr(cfg, "translation_branch_encoder_layers", 0))
            self.patch_embed_c_t = PatchEmbed(
                cfg.p,
                cfg.D,
                in_ch=cfg.in_ch,
                use_coords=bool(getattr(cfg, "patch_embed_use_coords", False)),
                pool_mode=str(getattr(cfg, "translation_patch_pool_mode", "gated_avgmax")),
                pool_gate_init=float(getattr(cfg, "translation_patch_pool_gate_init", -2.0)),
            )
            if t_layers > 0:
                self.enc_c_t = TokenEncoder(cfg.D, t_layers, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            else:
                self.enc_c_t = nn.LayerNorm(cfg.D)
        self.use_cross_context = bool(getattr(cfg, "use_cross_context", False))
        self.cross_c = None
        if self.use_cross_context:
            self.cross_c = CrossContextEncoder(
                cfg.D,
                n_layers=int(getattr(cfg, "cross_context_layers", 1)),
                n_heads=cfg.n_heads,
                mlp_ratio=2.0,
                dropout=cfg.dropout,
                strength=float(getattr(cfg, "cross_context_strength", 0.25)),
            )

    def _make_tokens(self, img: torch.Tensor, level: str, *, branch: str = "pose") -> Tokens:
        level_data = self.hier.level_data()
        B = img.shape[0]

        if level == "coarse":
            N = self.cfg.Nc
            grid = self.grid_coarse
            bearing = level_data["coarse"]["bearing"]
            ids = level_data["coarse"]["id"]
            parent = None
            patches = sample_patches_erp(img, grid, N=N, p=self.cfg.p)
            if branch == "translation":
                assert self.patch_embed_c_t is not None and self.enc_c_t is not None
                feat = self.patch_embed_c_t(patches)
                enc = self.enc_c_t
            else:
                feat = self.patch_embed_c(patches)
                enc = self.enc_c
        else:
            N = self.cfg.Nf
            grid = self.grid_fine
            bearing = level_data["fine"]["bearing"]
            ids = level_data["fine"]["id"]
            parent = level_data["fine"]["parent_id"]
            patches = sample_patches_erp(img, grid, N=N, p=self.cfg.p)
            feat = self.patch_embed_f(patches)
            enc = self.enc_f

        bearing_b = bearing.view(1, N, 3).expand(B, -1, -1)
        pos_scale = 1.0
        if branch == "translation":
            pos_scale = float(getattr(self.cfg, "translation_pos_enc_scale", 1.0))
        feat = feat + self.pos_enc(bearing_b) * pos_scale
        feat = enc(feat)

        ids_b = ids.view(1, N).expand(B, -1)
        parent_b = None if parent is None else parent.view(1, N).expand(B, -1)
        return Tokens(feat=feat, bearing=bearing_b, level=level, id=ids_b, parent_id=parent_b)

    def forward(self, IA: torch.Tensor, IB: torch.Tensor) -> Dict[str, Tokens]:
        TokA_c = self._make_tokens(IA, "coarse")
        TokB_c = self._make_tokens(IB, "coarse")
        if self.cross_c is not None:
            feat_a, feat_b = self.cross_c(TokA_c.feat, TokB_c.feat)
            TokA_c = Tokens(feat=feat_a, bearing=TokA_c.bearing, level=TokA_c.level, id=TokA_c.id, parent_id=TokA_c.parent_id)
            TokB_c = Tokens(feat=feat_b, bearing=TokB_c.bearing, level=TokB_c.level, id=TokB_c.id, parent_id=TokB_c.parent_id)
        out = {
            "TokA_c": TokA_c,
            "TokB_c": TokB_c,
        }
        if bool(self.cfg.use_fine_stage):
            out["TokA_f"] = self._make_tokens(IA, "fine")
            out["TokB_f"] = self._make_tokens(IB, "fine")
        if self.use_translation_feature_branch:
            out["TokA_c_t"] = self._make_tokens(IA, "coarse", branch="translation")
            out["TokB_c_t"] = self._make_tokens(IB, "coarse", branch="translation")
        return out


def _local_t_to_output_frame(R: torch.Tensor, t_local: torch.Tensor) -> torch.Tensor:
    t_local = nn.functional.normalize(t_local.float(), dim=-1, eps=1e-6)
    t_out = torch.matmul(R.float(), t_local.unsqueeze(-1)).squeeze(-1)
    return nn.functional.normalize(t_out, dim=-1, eps=1e-6)


def _project_to_rotation(M: torch.Tensor) -> torch.Tensor:
    M = M.float()
    U, _, Vh = torch.linalg.svd(M)
    R = torch.matmul(U, Vh)
    det = torch.det(R)
    fix = torch.ones((*M.shape[:-2], 3), device=M.device, dtype=M.dtype)
    fix[..., -1] = torch.where(det < 0.0, -1.0, 1.0)
    return torch.matmul(U * fix.unsqueeze(-2), Vh)


def _hat(v: torch.Tensor) -> torch.Tensor:
    vx, vy, vz = v.unbind(dim=-1)
    O = torch.zeros_like(vx)
    return torch.stack(
        [
            torch.stack([O, -vz, vy], dim=-1),
            torch.stack([vz, O, -vx], dim=-1),
            torch.stack([-vy, vx, O], dim=-1),
        ],
        dim=-2,
    )


def _so3_exp_map(omega: torch.Tensor) -> torch.Tensor:
    omega = omega.float()
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    K = _hat(omega)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).view(1, 3, 3).expand(omega.shape[0], -1, -1)
    theta2 = theta * theta
    theta_safe = theta.clamp_min(1.0e-6)
    theta2_safe = theta2.clamp_min(1.0e-8)
    sin_term = torch.sin(theta_safe) / theta_safe
    cos_term = (1.0 - torch.cos(theta_safe)) / theta2_safe
    sin_over_theta = torch.where(
        theta > 1.0e-6,
        sin_term,
        1.0 - theta2 / 6.0,
    )
    one_minus_cos_over_theta2 = torch.where(
        theta > 1.0e-6,
        cos_term,
        0.5 - theta2 / 24.0,
    )
    return I + sin_over_theta.unsqueeze(-1) * K + one_minus_cos_over_theta2.unsqueeze(-1) * torch.matmul(K, K)


def _blend_rotation(R_base: torch.Tensor, R_update: torch.Tensor, strength: float) -> torch.Tensor:
    a = float(max(0.0, min(1.0, strength)))
    if a <= 0.0:
        return R_base.float()
    if a >= 1.0:
        return R_update.float()
    return _project_to_rotation((1.0 - a) * R_base.float() + a * R_update.float())


def _blend_direction(t_base: torch.Tensor, t_update: torch.Tensor, strength: float) -> torch.Tensor:
    a = float(max(0.0, min(1.0, strength)))
    if a <= 0.0:
        return nn.functional.normalize(t_base.float(), dim=-1, eps=1e-6)
    if a >= 1.0:
        return nn.functional.normalize(t_update.float(), dim=-1, eps=1e-6)
    return nn.functional.normalize((1.0 - a) * t_base.float() + a * t_update.float(), dim=-1, eps=1e-6)


def _blend_magnitude(m_base: torch.Tensor, m_update: torch.Tensor, strength: float) -> torch.Tensor:
    a = float(max(0.0, min(1.0, strength)))
    m_base = m_base.float().view(-1)
    m_update = m_update.float().view(-1)
    if a <= 0.0:
        return m_base.clamp_min(1e-6)
    if a >= 1.0:
        return m_update.clamp_min(1e-6)
    return ((1.0 - a) * m_base + a * m_update).clamp_min(1e-6)


def _resolve_fine_fuse_strength(cfg: Config, attr_name: str) -> float:
    value = float(getattr(cfg, attr_name, -1.0))
    if value >= 0.0:
        return value
    return float(getattr(cfg, "fine_pose_fuse_strength", 1.0))


def _apply_log_tmag_bias(
    t_mag: torch.Tensor,
    log_t_mag: torch.Tensor,
    log_bias: Optional[torch.Tensor],
    affine_scale: Optional[torch.Tensor],
    affine_bias: Optional[torch.Tensor],
    *,
    min_mag: float,
    clamp_min: float,
    clamp_max: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    log_t_mag = log_t_mag.float().view(-1)
    has_affine = affine_scale is not None and affine_bias is not None
    if has_affine:
        log_t_mag = log_t_mag * affine_scale.float().view(()) + affine_bias.float().view(())
    if log_bias is None:
        if not has_affine:
            return t_mag.float().view(-1), log_t_mag
        log_t_mag = log_t_mag.clamp(min=float(clamp_min), max=float(clamp_max))
    else:
        log_t_mag = (log_t_mag + log_bias.float().view(())).clamp(
            min=float(clamp_min),
            max=float(clamp_max),
        )
    return torch.exp(log_t_mag).clamp_min(float(min_mag)), log_t_mag


def _set_transform_outputs(
    aux: Dict[str, torch.Tensor],
    R: torch.Tensor,
    t_local: torch.Tensor,
    t_mag: torch.Tensor,
    log_t_mag: Optional[torch.Tensor] = None,
) -> None:
    t_out = _local_t_to_output_frame(R, t_local)
    t_mag = t_mag.float().view(-1).clamp_min(1e-6)
    if log_t_mag is None:
        log_t_mag = torch.log(t_mag)
    else:
        log_t_mag = log_t_mag.float().view(-1)
    aux["t_dir_local"] = nn.functional.normalize(t_local.float(), dim=-1, eps=1e-6)
    aux["t_dir"] = aux["t_dir_local"]
    aux["t_dir_out"] = t_out
    aux["t_mag"] = t_mag
    aux["log_t_mag"] = log_t_mag
    aux["t_vec"] = t_out * t_mag.unsqueeze(-1)
    aux["t_vec_out"] = aux["t_vec"]
    aux["t_vec_local"] = aux["t_dir_local"] * t_mag.unsqueeze(-1)


def _dt_bucket_scale_anchor_bucket(dt: float) -> Optional[str]:
    if 0.1 <= dt < 0.3:
        return "[0.1,0.3)"
    if 0.3 <= dt < 0.5:
        return "[0.3,0.5)"
    if 0.5 <= dt < 1.0:
        return "[0.5,1)"
    return None


def _dt_bucket_scale_anchor_factor(cfg: Config, dt_world: Optional[torch.Tensor]) -> float:
    if not bool(getattr(cfg, "dt_bucket_scale_anchor_apply", False)):
        return 1.0
    if dt_world is None:
        return 1.0
    dt_val = float(dt_world.detach().float().view(-1)[0].cpu().item())
    bucket = _dt_bucket_scale_anchor_bucket(dt_val)
    if bucket == "[0.1,0.3)":
        return float(getattr(cfg, "dt_bucket_scale_anchor_factor_0p1_0p3", 1.0))
    if bucket == "[0.3,0.5)":
        return float(getattr(cfg, "dt_bucket_scale_anchor_factor_0p3_0p5", 1.0))
    if bucket == "[0.5,1)":
        return float(getattr(cfg, "dt_bucket_scale_anchor_factor_0p5_1p0", 1.0))
    return 1.0


def _apply_dt_bucket_scale_anchor(
    cfg: Config,
    aux: Dict[str, torch.Tensor],
    *,
    dt_world: Optional[torch.Tensor],
) -> None:
    factor = _dt_bucket_scale_anchor_factor(cfg, dt_world)
    fac_device = dt_world.device if torch.is_tensor(dt_world) else torch.device("cpu")
    for value in aux.values():
        if torch.is_tensor(value):
            fac_device = value.device
            break
    fac_t = torch.tensor(factor, device=fac_device, dtype=torch.float32)
    aux["dt_bucket_scale_anchor_factor"] = fac_t
    if abs(factor - 1.0) < 1.0e-12:
        return
    for mag_key in (
        "t_mag",
        "t_mag_unbiased",
        "tmag_before_coupled",
        "tmag_after_coupled",
    ):
        if mag_key in aux and torch.is_tensor(aux[mag_key]):
            aux[mag_key] = aux[mag_key].float() * fac_t
    for log_key, src in (
        ("log_t_mag", "t_mag"),
        ("log_t_mag_unbiased", "t_mag_unbiased"),
        ("log_tmag_before_coupled", "tmag_before_coupled"),
        ("log_tmag_after_coupled", "tmag_after_coupled"),
    ):
        if src in aux and torch.is_tensor(aux[src]):
            aux[log_key] = torch.log(aux[src].clamp_min(float(getattr(cfg, "tmag_min", 1.0e-3))))
    for vec_key in ("t_vec", "t_vec_out", "t_vec_local", "t_vec_fine_raw"):
        if vec_key in aux and torch.is_tensor(aux[vec_key]):
            aux[vec_key] = aux[vec_key].float() * fac_t


class RotationCompensatedMotionToken(nn.Module):
    """Experimental ARCH2 helper that fuses soft correspondence geometry."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(feature_dim + 3 + 3 + 3 + 1 + 1 + 1),
            nn.Linear(feature_dim + 3 + 3 + 3 + 1 + 1 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        interaction_features: torch.Tensor,
        bearing_a: torch.Tensor,
        bearing_b: torch.Tensor,
        W_ab: torch.Tensor,
        R_coarse: torch.Tensor,
        *,
        W_ba: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        feat = interaction_features.float()
        ba = nn.functional.normalize(bearing_a.float(), dim=-1, eps=1e-6)
        bb = nn.functional.normalize(bearing_b.float(), dim=-1, eps=1e-6)
        W_ab = W_ab.float()
        bA_rot = torch.matmul(R_coarse.float(), ba.transpose(1, 2)).transpose(1, 2)
        bB_match = torch.matmul(W_ab, bb)
        bB_match = nn.functional.normalize(bB_match, dim=-1, eps=1e-6)
        residual_flow = bB_match - nn.functional.normalize(bA_rot, dim=-1, eps=1e-6)
        confidence = W_ab.max(dim=-1).values.unsqueeze(-1)
        probs = W_ab.clamp_min(1.0e-9)
        entropy = -(probs * probs.log()).sum(dim=-1, keepdim=True)
        if W_ba is not None:
            cyc = torch.matmul(W_ab, W_ba.float())
            eye = torch.eye(cyc.shape[-1], device=cyc.device, dtype=cyc.dtype).view(1, cyc.shape[-2], cyc.shape[-1])
            cycle_error = torch.mean(torch.abs(cyc - eye), dim=-1, keepdim=True)
        else:
            cycle_error = torch.zeros_like(confidence)
        fused = torch.cat([feat, bA_rot, bB_match, residual_flow, confidence, entropy, cycle_error], dim=-1)
        motion_tokens = self.mlp(fused)
        return {
            "motion_tokens": motion_tokens,
            "bA_rot": bA_rot,
            "matched_bearing_b": bB_match,
            "residual_flow": residual_flow,
            "confidence": confidence.squeeze(-1),
            "entropy": entropy.squeeze(-1),
            "cycle_error": cycle_error.squeeze(-1),
        }


class CorrespondenceGeometryTDirHead(nn.Module):
    """Experimental ARCH2 head for conservative direction residuals."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, max_alpha: float = 0.05):
        super().__init__()
        self.max_alpha = float(max_alpha)
        self.backbone = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(hidden_dim, 3)
        self.obs_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, motion_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(motion_tokens.float())
        delta_tdir = torch.tanh(self.delta_head(h))
        observability_score = torch.sigmoid(self.obs_head(h)).squeeze(-1)
        delta_norm = torch.linalg.norm(delta_tdir.float(), dim=-1)
        return {
            "delta_tdir": delta_tdir,
            "delta_tdir_norm": delta_norm,
            "observability_score": observability_score,
            "gate_value": observability_score.clamp(0.0, 1.0),
            "max_alpha": torch.full_like(observability_score, self.max_alpha),
        }


class FineScaleHeadWithS5E15Guard(nn.Module):
    """Experimental bounded scale residual head with train-prior guard."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64, delta_clip: float = 0.25):
        super().__init__()
        self.delta_clip = float(delta_clip)
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled_features: torch.Tensor, base_log_tmag: torch.Tensor) -> Dict[str, torch.Tensor]:
        delta_log_tmag = torch.tanh(self.net(pooled_features.float()).view(-1)) * self.delta_clip
        final_log_tmag = base_log_tmag.float().view(-1) + delta_log_tmag
        final_tmag = torch.exp(final_log_tmag).clamp_min(1.0e-6)
        return {
            "delta_log_tmag": delta_log_tmag,
            "final_log_tmag": final_log_tmag,
            "final_tmag": final_tmag,
        }


class SphericalGeometryTokenBackbone(nn.Module):
    """STRUCT1 token encoder with optional spherical geometry channels."""

    def __init__(
        self,
        appearance_dim: int,
        hidden_dim: int,
        *,
        use_local_tangent_coords: bool = True,
        use_bearing_xyz_channels: bool = True,
        use_latlon_sincos_channels: bool = True,
        token_encoder_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_local_tangent_coords = bool(use_local_tangent_coords)
        self.use_bearing_xyz_channels = bool(use_bearing_xyz_channels)
        self.use_latlon_sincos_channels = bool(use_latlon_sincos_channels)
        extra_dim = 0
        if self.use_local_tangent_coords:
            extra_dim += 2
        if self.use_bearing_xyz_channels:
            extra_dim += 3
        if self.use_latlon_sincos_channels:
            extra_dim += 4
        in_dim = int(appearance_dim) + extra_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos_enc = BearingPosEnc(hidden_dim)
        self.encoder = TokenEncoder(
            hidden_dim,
            n_layers=max(1, int(token_encoder_layers)),
            n_heads=max(1, int(n_heads)),
            mlp_ratio=2.0,
            dropout=float(dropout),
        )

    @staticmethod
    def _bearing_to_geometry_channels(bearing: torch.Tensor) -> Dict[str, torch.Tensor]:
        bearing = nn.functional.normalize(bearing.float(), dim=-1, eps=1e-6)
        x, y, z = bearing.unbind(dim=-1)
        lon = torch.atan2(x, z.clamp(min=-1.0, max=1.0))
        lat = torch.asin(y.clamp(-1.0, 1.0))
        tangent_x = x / z.abs().clamp_min(1.0e-3)
        tangent_y = y / z.abs().clamp_min(1.0e-3)
        return {
            "bearing_xyz": torch.stack([x, y, z], dim=-1),
            "local_tangent_coords": torch.stack([tangent_x, tangent_y], dim=-1),
            "latlon_sincos": torch.stack([torch.sin(lon), torch.cos(lon), torch.sin(lat), torch.cos(lat)], dim=-1),
        }

    def forward(self, appearance_tokens: torch.Tensor, bearing: torch.Tensor) -> Dict[str, torch.Tensor]:
        geom = self._bearing_to_geometry_channels(bearing)
        pieces = [appearance_tokens.float()]
        if self.use_local_tangent_coords:
            pieces.append(geom["local_tangent_coords"])
        if self.use_bearing_xyz_channels:
            pieces.append(geom["bearing_xyz"])
        if self.use_latlon_sincos_channels:
            pieces.append(geom["latlon_sincos"])
        x = torch.cat(pieces, dim=-1)
        feat = self.input_proj(x)
        feat = feat + self.pos_enc(nn.functional.normalize(bearing.float(), dim=-1, eps=1e-6))
        feat = self.encoder(feat)
        return {
            "token_features": feat,
            "bearing": nn.functional.normalize(bearing.float(), dim=-1, eps=1e-6),
            "local_tangent_coords": geom["local_tangent_coords"],
            "bearing_xyz_channels": geom["bearing_xyz"],
            "latlon_sincos_channels": geom["latlon_sincos"],
        }


class SoftCorrespondenceGeometryLayer(nn.Module):
    """STRUCT1 soft correspondence layer that emits geometry tokens."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, temperature: float = 0.05) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.proj = nn.Linear(feature_dim, hidden_dim)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim + 3 + 3 + 3 + 1 + 1 + 1 + 1),
            nn.Linear(hidden_dim + 3 + 3 + 3 + 1 + 1 + 1 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        tokens_a: torch.Tensor,
        tokens_b: torch.Tensor,
        bearing_a: torch.Tensor,
        bearing_b: torch.Tensor,
        *,
        interaction_feature: Optional[torch.Tensor] = None,
        precomputed_W_ab: Optional[torch.Tensor] = None,
        precomputed_W_ba: Optional[torch.Tensor] = None,
        coarse_pose_R: Optional[torch.Tensor] = None,
        coarse_pose_tdir: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        ta = nn.functional.normalize(self.proj(tokens_a.float()), dim=-1, eps=1e-6)
        tb = nn.functional.normalize(self.proj(tokens_b.float()), dim=-1, eps=1e-6)
        sim = torch.matmul(ta, tb.transpose(1, 2))
        if precomputed_W_ab is None:
            W_ab = torch.softmax(sim / max(self.temperature, 1.0e-6), dim=-1)
        else:
            W_ab = precomputed_W_ab.float()
        if precomputed_W_ba is None:
            W_ba = torch.softmax(sim.transpose(1, 2) / max(self.temperature, 1.0e-6), dim=-1)
        else:
            W_ba = precomputed_W_ba.float()
        ba = nn.functional.normalize(bearing_a.float(), dim=-1, eps=1e-6)
        bb = nn.functional.normalize(bearing_b.float(), dim=-1, eps=1e-6)
        matched_bearing_b = torch.matmul(W_ab, bb)
        matched_bearing_b = nn.functional.normalize(matched_bearing_b, dim=-1, eps=1e-6)
        matched_bearing_a = torch.matmul(W_ba, ba)
        matched_bearing_a = nn.functional.normalize(matched_bearing_a, dim=-1, eps=1e-6)
        residual_flow = matched_bearing_b - ba
        probs = W_ab.clamp_min(1.0e-9)
        entropy = -(probs * probs.log()).sum(dim=-1, keepdim=True)
        entropy_norm = entropy / max(float(torch.log(torch.tensor(float(W_ab.shape[-1]))).item()), 1.0e-6)
        confidence = (1.0 - entropy_norm).clamp(0.0, 1.0)
        cyc = torch.matmul(W_ab, W_ba)
        eye = torch.eye(cyc.shape[-1], device=cyc.device, dtype=cyc.dtype).view(1, cyc.shape[-2], cyc.shape[-1])
        cycle_error = torch.mean(torch.abs(cyc - eye), dim=-1, keepdim=True)
        if coarse_pose_R is None or coarse_pose_tdir is None:
            epipolar_residual = torch.zeros_like(confidence)
        else:
            bA_rot = torch.matmul(coarse_pose_R.float(), ba.transpose(1, 2)).transpose(1, 2)
            tdir = nn.functional.normalize(coarse_pose_tdir.float(), dim=-1, eps=1e-6)
            cross = torch.cross(
                tdir[:, None, :].expand_as(matched_bearing_b),
                matched_bearing_b,
                dim=-1,
            )
            epipolar_residual = torch.abs(torch.sum(cross * bA_rot, dim=-1, keepdim=True))
        if interaction_feature is None:
            interaction_feature = 0.5 * (tokens_a.float() + torch.matmul(W_ab, tokens_b.float()))
        fused = torch.cat(
            [
                ba,
                matched_bearing_b,
                residual_flow,
                confidence,
                entropy,
                cycle_error,
                epipolar_residual,
                interaction_feature.float(),
            ],
            dim=-1,
        )
        geometry_tokens = self.out_proj(fused)
        return {
            "W_ab": W_ab,
            "W_ba": W_ba,
            "matched_bearing_b": matched_bearing_b,
            "matched_bearing_a": matched_bearing_a,
            "residual_flow": residual_flow,
            "confidence": confidence.squeeze(-1),
            "entropy": entropy.squeeze(-1),
            "cycle_error": cycle_error.squeeze(-1),
            "epipolar_residual": epipolar_residual.squeeze(-1),
            "interaction_feature": interaction_feature.float(),
            "geometry_tokens": geometry_tokens,
            "bearing_a": ba,
            "bearing_b": bb,
        }


class GeometryTokenPoseSolver(nn.Module):
    """STRUCT1 geometry-aware decoder with coarse-to-fine aggregation."""

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 128,
        *,
        topk: int = 16,
        scale_delta_clip: float = 0.25,
        rotation_refine_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.topk = int(max(1, topk))
        self.rotation_refine_scale = float(rotation_refine_scale)
        self.summary_net = nn.Sequential(
            nn.LayerNorm(token_dim * 3),
            nn.Linear(token_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.rotation_refine = nn.Linear(hidden_dim, 3)
        self.tdir_head = nn.Linear(hidden_dim, 3)
        self.scale_head = FineScaleHeadWithS5E15Guard(hidden_dim, hidden_dim=hidden_dim, delta_clip=scale_delta_clip)
        nn.init.zeros_(self.rotation_refine.weight)
        nn.init.zeros_(self.rotation_refine.bias)

    def _aggregate(self, geometry_tokens: torch.Tensor, confidence: torch.Tensor) -> Dict[str, torch.Tensor]:
        conf = confidence.float().clamp_min(1.0e-6)
        conf_sum = conf.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        mean = torch.sum(geometry_tokens.float() * conf.unsqueeze(-1), dim=1) / conf_sum
        topk = min(self.topk, geometry_tokens.shape[1])
        topk_val, topk_idx = torch.topk(conf, k=topk, dim=-1)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, geometry_tokens.shape[-1])
        topk_tokens = torch.gather(geometry_tokens.float(), 1, gather_idx)
        topk_conf = topk_val / topk_val.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        topk_pool = torch.sum(topk_tokens * topk_conf.unsqueeze(-1), dim=1)
        centered = geometry_tokens.float() - mean[:, None, :]
        second_moment = torch.sum(centered.abs() * conf.unsqueeze(-1), dim=1) / conf_sum
        summary = torch.cat([mean, topk_pool, second_moment], dim=-1)
        return {
            "summary": summary,
            "confidence_weighted_mean": mean,
            "robust_topk_pool": topk_pool,
            "histogram_or_moments": second_moment,
            "topk_indices": topk_idx,
        }

    def forward(
        self,
        geometry_tokens: torch.Tensor,
        confidence: torch.Tensor,
        *,
        coarse_rotation: torch.Tensor,
        coarse_tdir: torch.Tensor,
        base_log_tmag: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        agg = self._aggregate(geometry_tokens, confidence)
        h = self.summary_net(agg["summary"])
        rot_update = torch.tanh(self.rotation_refine(h)) * self.rotation_refine_scale
        R_update = _so3_exp_map(rot_update)
        R_BA = torch.matmul(R_update, coarse_rotation.float())
        delta_tdir = torch.tanh(self.tdir_head(h))
        tdir_B = nn.functional.normalize(delta_tdir, dim=-1, eps=1e-6)
        scale = self.scale_head(h, base_log_tmag)
        return {
            **agg,
            **scale,
            "rotation_update_vec": rot_update,
            "R_BA": R_BA,
            "tdir_B": tdir_B,
            "tmag": scale["final_tmag"],
            "final_tmag": scale["final_tmag"],
            "final_log_tmag": scale["final_log_tmag"],
            "tdir_from_geometry_tokens": True,
            "W_ab_used_in_pose_solver": True,
        }


class PanoramaRelPoseModel(nn.Module):
    def __init__(self, cfg: Config, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.module2 = Module2Sampler(cfg, device=device)
        self.direct_head = PairPoseHead(cfg.D)
        self.coarse = CoarseInteraction(
            cfg.D,
            temperature=cfg.coarse_temperature,
            logits_clip=cfg.logits_clip,
            pose_use_stats_pool=bool(getattr(cfg, "pose_use_stats_pool", False)),
            use_bearing_fuse=bool(getattr(cfg, "use_bearing_fuse", False)),
            use_translation_feature_branch=bool(getattr(cfg, "use_translation_feature_branch", False)),
            translation_branch_detach_match=bool(getattr(cfg, "translation_branch_detach_match", True)),
            use_translation_magnitude_head=bool(getattr(cfg, "use_translation_magnitude_head", True)),
            tmag_pred_source=str(getattr(cfg, "tmag_pred_source", "translation_branch")),
            tmag_detach_features=bool(getattr(cfg, "tmag_detach_features", False)),
            tmag_min=float(getattr(cfg, "tmag_min", 1.0e-3)),
            log_tmag_clamp_min=float(getattr(cfg, "log_tmag_clamp_min", -6.0)),
            log_tmag_clamp_max=float(getattr(cfg, "log_tmag_clamp_max", 6.0)),
            tmag_condition_on_dt=bool(getattr(cfg, "tmag_condition_on_dt", False)),
            # T57b: multiscale tmag head
            tmag_head_mode=str(getattr(cfg, "tmag_head_mode", "scalar")),
            tmag_num_bins=int(getattr(cfg, "tmag_multiscale_num_bins", 4)),
            tmag_log_centers=str(getattr(cfg, "tmag_multiscale_log_centers", "-3.5,-1.7,-0.9,-0.3")),
            tmag_residual_scale=float(getattr(cfg, "tmag_multiscale_residual_scale", 1.0)),
            # T57c: ridge_linear head
            tmag_ridge_head_path=str(getattr(cfg, "tmag_ridge_head_path", "")),
            tmag_ridge_head_trainable=bool(getattr(cfg, "tmag_ridge_head_trainable", True)),
            tmag_ridge_head_scale=float(getattr(cfg, "tmag_ridge_head_scale", 1.0)),
            # T57e: ridge_calib head
            tmag_ridge_calib_init_path=str(getattr(cfg, "tmag_ridge_calib_init_path", "")),
            tmag_ridge_calib_gamma_max=float(getattr(cfg, "tmag_ridge_calib_gamma_max", 0.50)),
            tmag_ridge_calib_gamma_init=float(getattr(cfg, "tmag_ridge_calib_gamma_init", 0.20)),
            tmag_ridge_calib_train_gamma=bool(getattr(cfg, "tmag_ridge_calib_train_gamma", True)),
            tmag_ridge_calib_train_bias=bool(getattr(cfg, "tmag_ridge_calib_train_bias", True)),
            tmag_ridge_calib_raw_center=float(getattr(cfg, "tmag_ridge_calib_raw_center", 0.0)),
            tmag_ridge_calib_log_base=float(getattr(cfg, "tmag_ridge_calib_log_base", 0.0)),
            tmag_ridge_calib_blend_init=float(getattr(cfg, "tmag_ridge_calib_blend_init", 1.0)),
            tmag_ridge_calib_train_blend=bool(getattr(cfg, "tmag_ridge_calib_train_blend", False)),
        )
        self.fine = FineInteraction(
            cfg.D,
            temperature=cfg.fine_temperature,
            logits_clip=cfg.logits_clip,
            use_depth_fusion=bool(cfg.use_depth_branch and cfg.depth_fuse_to_translation_only),
            depth_fuse_strength=float(getattr(cfg, "depth_fuse_strength", 1.0)),
            depth_fuse_detach_feature=bool(getattr(cfg, "depth_fuse_detach_feature", False)),
            use_geometric_t_fusion=bool(getattr(cfg, "use_geometric_t_fusion", False)),
            geometric_t_fuse_strength=float(getattr(cfg, "geometric_t_fuse_strength", 0.0)),
            pose_use_stats_pool=bool(getattr(cfg, "pose_use_stats_pool", False)),
            use_bearing_fuse=bool(getattr(cfg, "use_bearing_fuse", False)),
            use_translation_magnitude_head=bool(getattr(cfg, "use_translation_magnitude_head", True)),
            tmag_detach_features=bool(getattr(cfg, "tmag_detach_features", False)),
            tmag_min=float(getattr(cfg, "tmag_min", 1.0e-3)),
            log_tmag_clamp_min=float(getattr(cfg, "log_tmag_clamp_min", -6.0)),
            log_tmag_clamp_max=float(getattr(cfg, "log_tmag_clamp_max", 6.0)),
        )

        self.depth = None
        self.depth_token_proj = None
        if bool(cfg.use_depth_branch):
            self.depth = LightERPDepthNet(
                in_ch=cfg.in_ch,
                feat_dim=cfg.depth_feat_dim,
                inv_depth_min=cfg.depth_inv_min,
                inv_depth_max=cfg.depth_inv_max,
            )
            self.depth_token_proj = nn.Linear(cfg.depth_feat_dim, cfg.D)
        self.use_tmag_global_bias = bool(getattr(cfg, "use_tmag_global_bias", False))
        self.use_tmag_affine_calib = bool(getattr(cfg, "use_tmag_affine_calib", False))
        if self.use_tmag_global_bias:
            self.log_tmag_bias = nn.Parameter(torch.tensor(float(getattr(cfg, "tmag_global_bias_init", 0.0)), dtype=torch.float32))
        else:
            self.register_parameter("log_tmag_bias", None)
        if self.use_tmag_affine_calib:
            self.tmag_affine_scale = nn.Parameter(torch.tensor(float(getattr(cfg, "tmag_affine_init_scale", 1.0)), dtype=torch.float32))
            self.tmag_affine_bias = nn.Parameter(torch.tensor(float(getattr(cfg, "tmag_affine_init_bias", 0.0)), dtype=torch.float32))
        else:
            self.register_parameter("tmag_affine_scale", None)
            self.register_parameter("tmag_affine_bias", None)
        self.coupled_pose_head = CoupledPoseResidualHead(
            cfg.D,
            hidden_dim=int(getattr(cfg, "coupled_pose_residual_hidden_dim", 128)),
            dropout=float(getattr(cfg, "coupled_pose_residual_dropout", 0.0)),
            enable_rot=bool(getattr(cfg, "coupled_pose_residual_enable_rot", True)),
            enable_tdir=bool(getattr(cfg, "coupled_pose_residual_enable_tdir", False)),
            rot_scale=float(getattr(cfg, "coupled_pose_residual_rot_scale", 0.05)),
            tdir_scale=float(getattr(cfg, "coupled_pose_residual_tdir_scale", 0.05)),
            gate_init=float(getattr(cfg, "coupled_pose_residual_gate_init", -4.0)),
            gate_max=float(getattr(cfg, "coupled_pose_residual_gate_max", 0.05)),
            force_tdir_zero=bool(getattr(cfg, "coupled_pose_residual_force_tdir_zero", True)),
            use_dt_embed=bool(getattr(cfg, "coupled_pose_residual_use_dt_embed", True)),
            use_confidence=bool(getattr(cfg, "coupled_pose_residual_use_confidence", True)),
        )

    def _bias_magnitude(self, t_mag: torch.Tensor, log_t_mag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _apply_log_tmag_bias(
            t_mag,
            log_t_mag,
            self.log_tmag_bias,
            self.tmag_affine_scale if self.use_tmag_affine_calib else None,
            self.tmag_affine_bias if self.use_tmag_affine_calib else None,
            min_mag=float(getattr(self.cfg, "tmag_min", 1.0e-3)),
            clamp_min=float(getattr(self.cfg, "log_tmag_clamp_min", -6.0)),
            clamp_max=float(getattr(self.cfg, "log_tmag_clamp_max", 6.0)),
        )

    def forward(
        self,
        IA: torch.Tensor,
        IB: torch.Tensor,
        *,
        enable_depth_fusion: Optional[bool] = None,
        dt_world: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        tokens = self.module2(IA, IB)
        TokA_c = tokens["TokA_c"]
        TokB_c = tokens["TokB_c"]
        TokA_f = tokens.get("TokA_f")
        TokB_f = tokens.get("TokB_f")

        inv_depths: Dict[str, torch.Tensor] = {}
        depth_feats: Dict[str, torch.Tensor] = {}
        if self.depth is not None:
            inv_depths, depth_feats = self.depth(IA)

        aux: Dict[str, torch.Tensor] = {
            "bearingA_c": TokA_c.bearing,
            "bearingB_c": TokB_c.bearing,
            "t_local_frame": self.cfg.translation_local_frame,
            "t_output_frame": self.cfg.translation_output_frame,
        }
        if TokA_f is not None and TokB_f is not None:
            aux["bearingA_f"] = TokA_f.bearing
            aux["bearingB_f"] = TokB_f.bearing
        if inv_depths:
            for k, v in inv_depths.items():
                aux[f"inv_depth_{k}"] = v
            for k, v in depth_feats.items():
                aux[f"depth_feat_{k}"] = v

        if not self.cfg.use_coarse_interaction:
            R, t_local = self.direct_head(TokA_c.feat, TokB_c.feat)
            t_mag = torch.ones((IA.shape[0],), device=IA.device, dtype=torch.float32)
            _set_transform_outputs(aux, R, t_local, t_mag)
            _apply_dt_bucket_scale_anchor(self.cfg, aux, dt_world=dt_world)
            aux["stage"] = "encoder_only"
            return R, aux["t_dir"], aux

        out_c = self.coarse(
            TokA_c, TokB_c,
            tokens_a_t=tokens.get("TokA_c_t"),
            tokens_b_t=tokens.get("TokB_c_t"),
            dt_world=dt_world,
            tmag_dt_clamp_min=float(getattr(self.cfg, "tmag_dt_clamp_min", 0.01)),
        )
        aux.update(out_c)
        aux["tc_dir_local"] = out_c["tc_dir"]
        aux["tc_dir_out"] = _local_t_to_output_frame(out_c["Rc"], out_c["tc_dir"])
        aux["tc_dir"] = out_c["tc_dir"]
        aux["tc_mag"] = out_c["tc_mag"]
        aux["log_tc_mag"] = out_c["log_tc_mag"]
        aux["tc_vec_out"] = aux["tc_dir_out"] * aux["tc_mag"].unsqueeze(-1)
        aux["tc_vec_local"] = aux["tc_dir_local"] * aux["tc_mag"].unsqueeze(-1)
        t_mag_final, log_t_mag_final = self._bias_magnitude(out_c["tc_mag"], out_c["log_tc_mag"])
        aux["t_mag_unbiased"] = out_c["tc_mag"]
        aux["log_t_mag_unbiased"] = out_c["log_tc_mag"]
        aux["log_tmag_bias"] = self.log_tmag_bias.detach().view(()) if self.log_tmag_bias is not None else torch.zeros((), device=IA.device)
        _set_transform_outputs(aux, out_c["Rc"], out_c["tc_dir"], t_mag_final, log_t_mag_final)
        _apply_dt_bucket_scale_anchor(self.cfg, aux, dt_world=dt_world)
        aux["stage"] = "coarse_only"

        if not self.cfg.use_fine_stage:
            return out_c["Rc"], aux["t_dir"], aux

        assert TokA_f is not None and TokB_f is not None

        if enable_depth_fusion is None:
            enable_depth_fusion = bool(self.cfg.use_depth_branch and self.cfg.depth_fuse_to_translation_only)

        depth_tok_a = None
        if enable_depth_fusion and self.depth is not None and self.depth_token_proj is not None:
            scale_key = f"s{int(self.cfg.depth_fuse_scale)}"
            if scale_key in depth_feats:
                depth_tok_a = sample_dense_features_from_bearings(depth_feats[scale_key], TokA_f.bearing)
                depth_tok_a = self.depth_token_proj(depth_tok_a.float())
                aux["depth_tok_a"] = depth_tok_a

        out_f = self.fine(
            TokA_f=TokA_f,
            TokB_f=TokB_f,
            Wc_ab=out_c["Wc_ab"].detach(),
            Rc=out_c["Rc"].detach(),
            tc_dir=aux["tc_dir_out"].detach(),
            depth_tok_a=depth_tok_a,
            topk_coarse=self.cfg.topk_coarse,
            use_epipolar_bias=self.cfg.use_epipolar_bias,
            epi_angle_thresh_deg=self.cfg.epi_angle_thresh_deg,
            epi_bias_strength=self.cfg.epi_bias_strength,
            epi_mode=self.cfg.epi_mode,
        )
        aux.update(out_f)
        fine_rot_strength = _resolve_fine_fuse_strength(self.cfg, "fine_rot_fuse_strength")
        fine_tdir_strength = _resolve_fine_fuse_strength(self.cfg, "fine_tdir_fuse_strength")
        fine_tmag_strength = _resolve_fine_fuse_strength(self.cfg, "fine_tmag_fuse_strength")
        R_final = _blend_rotation(out_c["Rc"], out_f["R"], fine_rot_strength)
        t_local_final = _blend_direction(aux["tc_dir_local"], out_f["t_dir"], fine_tdir_strength)
        t_mag_final = _blend_magnitude(aux["tc_mag"], out_f["t_mag"], fine_tmag_strength)
        log_t_mag_final = torch.log(t_mag_final.clamp_min(float(getattr(self.cfg, "tmag_min", 1.0e-3))))
        t_mag_final_unbiased = t_mag_final
        t_mag_final, log_t_mag_final = self._bias_magnitude(t_mag_final, log_t_mag_final)
        aux["Rf_raw"] = out_f["R"]
        aux["t_dir_fine_raw"] = out_f["t_dir"]
        aux["t_mag_fine_raw"] = out_f["t_mag"]
        aux["log_t_mag_fine_raw"] = out_f["log_t_mag"]
        aux["t_vec_fine_raw"] = _local_t_to_output_frame(out_f["R"], out_f["t_dir"]) * out_f["t_mag"].float().view(-1, 1)
        aux["t_mag_unbiased"] = t_mag_final_unbiased
        aux["log_t_mag_unbiased"] = torch.log(t_mag_final_unbiased.clamp_min(float(getattr(self.cfg, "tmag_min", 1.0e-3))))
        aux["log_tmag_bias"] = self.log_tmag_bias.detach().view(()) if self.log_tmag_bias is not None else torch.zeros((), device=IA.device)
        aux["fine_pose_fuse_strength"] = torch.tensor(float(getattr(self.cfg, "fine_pose_fuse_strength", 1.0)), device=R_final.device)
        aux["fine_rot_fuse_strength"] = torch.tensor(fine_rot_strength, device=R_final.device)
        aux["fine_tdir_fuse_strength"] = torch.tensor(fine_tdir_strength, device=R_final.device)
        aux["fine_tmag_fuse_strength"] = torch.tensor(fine_tmag_strength, device=R_final.device)
        R_before_coupled = R_final
        tdir_before_coupled = t_local_final
        tmag_before_coupled = t_mag_final
        log_tmag_before_coupled = log_t_mag_final
        delta_rot_vec = torch.zeros((R_final.shape[0], 3), device=R_final.device, dtype=R_final.dtype)
        delta_tdir_vec = torch.zeros((R_final.shape[0], 3), device=R_final.device, dtype=R_final.dtype)
        gate = torch.zeros((R_final.shape[0], 1), device=R_final.device, dtype=R_final.dtype)
        if bool(getattr(self.cfg, "use_coupled_pose_residual_head", False)):
            coupled_out = self.coupled_pose_head(
                out_f.get("Ff_t", out_f["Ff"]),
                base_R=R_final,
                base_tdir_local=t_local_final,
                token_weight=out_f.get("token_weight_f"),
                dt_world=dt_world,
            )
            delta_rot_vec = coupled_out["delta_rot_vec"]
            delta_tdir_vec = coupled_out["delta_tdir_vec"]
            gate = coupled_out["gate"]
            delta_R = _so3_exp_map(gate * delta_rot_vec)
            R_final = torch.matmul(delta_R, R_final.float())
            tdir_enabled = bool(getattr(self.cfg, "coupled_pose_residual_enable_tdir", False))
            force_tdir_zero = bool(getattr(self.cfg, "coupled_pose_residual_force_tdir_zero", True))
            if tdir_enabled and not force_tdir_zero:
                t_local_final = nn.functional.normalize(
                    t_local_final.float() + gate * delta_tdir_vec.float(),
                    dim=-1,
                    eps=1.0e-6,
                )
        aux["R_before_coupled"] = R_before_coupled
        aux["tdir_before_coupled"] = tdir_before_coupled
        aux["tmag_before_coupled"] = tmag_before_coupled
        aux["log_tmag_before_coupled"] = log_tmag_before_coupled
        aux["R_after_coupled"] = R_final
        aux["tdir_after_coupled"] = t_local_final
        aux["tmag_after_coupled"] = tmag_before_coupled
        aux["log_tmag_after_coupled"] = log_tmag_before_coupled
        aux["delta_rot_vec"] = delta_rot_vec
        aux["delta_tdir_vec"] = delta_tdir_vec
        aux["coupled_gate"] = gate
        aux["delta_rot_norm"] = torch.linalg.norm(delta_rot_vec.float(), dim=-1)
        aux["delta_tdir_norm"] = torch.linalg.norm(delta_tdir_vec.float(), dim=-1)
        aux["coupled_delta_rot_norm"] = torch.linalg.norm(delta_rot_vec.float(), dim=-1)
        aux["coupled_delta_tdir_norm"] = torch.linalg.norm(delta_tdir_vec.float(), dim=-1)
        aux["coupled_gate_mean"] = gate.float().mean()
        aux["gate_mean"] = aux["coupled_gate_mean"]
        aux["tdir_before_after_max_diff"] = (
            aux["tdir_after_coupled"].float() - aux["tdir_before_coupled"].float()
        ).abs().max()
        aux["tmag_before_after_max_diff"] = (
            aux["tmag_after_coupled"].float().view(-1) - aux["tmag_before_coupled"].float().view(-1)
        ).abs().max()
        _set_transform_outputs(aux, R_final, t_local_final, tmag_before_coupled, log_tmag_before_coupled)
        _apply_dt_bucket_scale_anchor(self.cfg, aux, dt_world=dt_world)
        aux["stage"] = "coarse_to_fine"
        aux["Wc_tilde"] = aggregate_fine_to_coarse(
            Wf_ab=out_f["Wf_ab"],
            parent_a=TokA_f.parent_id,
            parent_b=TokB_f.parent_id,
            NcA=self.cfg.Nc,
            NcB=self.cfg.Nc,
        )
        return R_final, aux["t_dir"], aux
