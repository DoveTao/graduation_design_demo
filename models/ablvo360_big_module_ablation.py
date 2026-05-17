from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from train360.core.config import Config
from train360.core.erp_sampling import build_level_grid_from_patch_bearings, sample_patches_erp
from train360.core.healpix_utils import HealpixHierarchy
from train360.core.interaction import CoarseInteraction, Tokens
from train360.core.model import _apply_dt_bucket_scale_anchor, _local_t_to_output_frame, _set_transform_outputs, _so3_exp_map
from train360.core.transformer_encoder import BearingPosEnc, PatchEmbed, TokenEncoder


def _erp_uv_to_bearing(u: torch.Tensor, v: torch.Tensor, H: int, W: int) -> torch.Tensor:
    lon = (u / float(W)) * (2.0 * math.pi) - math.pi
    lat = math.pi / 2.0 - (v / float(H)) * math.pi
    x = torch.cos(lat) * torch.sin(lon)
    y = torch.sin(lat)
    z = torch.cos(lat) * torch.cos(lon)
    return F.normalize(torch.stack([x, y, z], dim=-1), dim=-1, eps=1.0e-6)


def _build_planar_level_metadata(
    token_rows: int,
    token_cols: int,
    patch_size: int,
    H: int,
    W: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if token_rows <= 0 or token_cols <= 0:
        raise ValueError("token_rows and token_cols must be positive")
    step_u = float(W) / float(token_cols)
    step_v = float(H) / float(token_rows)
    center_u = (torch.arange(token_cols, dtype=torch.float32) + 0.5) * step_u
    center_v = (torch.arange(token_rows, dtype=torch.float32) + 0.5) * step_v
    vv, uu = torch.meshgrid(center_v, center_u, indexing="ij")

    delta_u = (torch.arange(patch_size, dtype=torch.float32) + 0.5) - (patch_size / 2.0)
    delta_v = (torch.arange(patch_size, dtype=torch.float32) + 0.5) - (patch_size / 2.0)
    dv, du = torch.meshgrid(delta_v, delta_u, indexing="ij")
    du = du * (step_u / float(patch_size))
    dv = dv * (step_v / float(patch_size))

    patch_u = torch.remainder(uu[..., None, None] + du[None, None, ...], float(W))
    patch_v = (vv[..., None, None] + dv[None, None, ...]).clamp(0.0, float(H - 1))

    grid_u = patch_u.reshape(-1, patch_size, patch_size)
    grid_v = patch_v.reshape(-1, patch_size, patch_size)
    grid_x = 2.0 * ((grid_u + float(W)) / float(3 * W - 1)) - 1.0
    grid_y = 2.0 * (grid_v / float(H - 1)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, token_rows * token_cols * patch_size, patch_size, 2)

    center_u_flat = uu.reshape(-1)
    center_v_flat = vv.reshape(-1)
    bearing = _erp_uv_to_bearing(center_u_flat, center_v_flat, H=H, W=W)
    grid_pos = torch.stack(
        [
            (center_u_flat / max(float(W - 1), 1.0)) * 2.0 - 1.0,
            (center_v_flat / max(float(H - 1), 1.0)) * 2.0 - 1.0,
        ],
        dim=-1,
    )
    return grid, bearing, grid_pos


class GridPositionalEncoding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, grid_pos: torch.Tensor) -> torch.Tensor:
        x = grid_pos[..., 0]
        y = grid_pos[..., 1]
        feat = torch.stack(
            [
                x,
                y,
                torch.sin(math.pi * x),
                torch.cos(math.pi * x),
                torch.sin(math.pi * y),
                torch.cos(math.pi * y),
            ],
            dim=-1,
        )
        return self.net(feat.float())


class PairPoseScaleHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        in_dim = 12 * dim
        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.rot = nn.Linear(hidden_dim, 6)
        self.tdir = nn.Linear(hidden_dim, 3)
        self.log_tmag = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _stats_pool(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)
        std = torch.sqrt(x.var(dim=1, unbiased=False).clamp_min(1.0e-8))
        maxv = x.max(dim=1).values
        return torch.cat([mean, maxv, std], dim=-1)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        za = self._stats_pool(feat_a.float())
        zb = self._stats_pool(feat_b.float())
        z = torch.cat([za, zb, torch.abs(za - zb), za * zb], dim=-1)
        h = self.backbone(z)
        rot6d = self.rot(h)
        a1 = F.normalize(rot6d[:, 0:3], dim=-1, eps=1.0e-6)
        a2 = rot6d[:, 3:6]
        b2 = F.normalize(a2 - (a1 * a2).sum(dim=-1, keepdim=True) * a1, dim=-1, eps=1.0e-6)
        b3 = F.normalize(torch.cross(a1, b2, dim=-1), dim=-1, eps=1.0e-6)
        R = torch.stack([a1, b2, b3], dim=-1)
        tdir_out = F.normalize(self.tdir(h), dim=-1, eps=1.0e-6)
        log_tmag = self.log_tmag(h).squeeze(-1)
        return R, tdir_out, log_tmag


class PoseEmbeddingMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        in_dim = 9 + 3 + 1 + 3 * dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
        )

    def forward(
        self,
        coarse_rotation: torch.Tensor,
        coarse_tdir_out: torch.Tensor,
        coarse_log_tmag: torch.Tensor,
        coarse_pair_context: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                coarse_rotation.float().reshape(coarse_rotation.shape[0], -1),
                F.normalize(coarse_tdir_out.float(), dim=-1, eps=1.0e-6),
                coarse_log_tmag.float().view(-1, 1),
                coarse_pair_context.float(),
            ],
            dim=-1,
        )
        return self.net(x)


class PlanarPoseConditionedFineTokenRefiner(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        dropout: float = 0.05,
        mlp_ratio: float = 2.0,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.pos_enc = GridPositionalEncoding(dim)
        self.pose_to_film_a = nn.Linear(dim, 2 * dim)
        self.pose_to_film_b = nn.Linear(dim, 2 * dim)
        self.pose_to_token = nn.Linear(dim, dim)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.cross_attn_ab = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_ba = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_a = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.ffn_b = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.update_gate = nn.Linear(dim, 1)
        nn.init.zeros_(self.update_gate.weight)
        nn.init.constant_(self.update_gate.bias, float(gate_bias))

    @staticmethod
    def _apply_film(x: torch.Tensor, film: torch.Tensor) -> torch.Tensor:
        gamma, beta = film.chunk(2, dim=-1)
        return (1.0 + 0.1 * torch.tanh(gamma)).unsqueeze(1) * x + beta.unsqueeze(1)

    @staticmethod
    def _pool_stats(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)
        std = torch.sqrt(x.var(dim=1, unbiased=False).clamp_min(1.0e-8))
        maxv = x.max(dim=1).values
        return torch.cat([mean, maxv, std], dim=-1)

    def forward(
        self,
        tokens_a: Tokens,
        tokens_b: Tokens,
        pose_embed: torch.Tensor,
        grid_pos_a: torch.Tensor,
        grid_pos_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pose_token = self.pose_to_token(pose_embed).unsqueeze(1)
        xa = tokens_a.feat.float() + self.pos_enc(grid_pos_a.float()) + pose_token
        xb = tokens_b.feat.float() + self.pos_enc(grid_pos_b.float()) + pose_token
        xa = self._apply_film(self.norm_a(xa), self.pose_to_film_a(pose_embed))
        xb = self._apply_film(self.norm_b(xb), self.pose_to_film_b(pose_embed))

        ab_ctx, _ = self.cross_attn_ab(xa, xb, xb, need_weights=False)
        ba_ctx, _ = self.cross_attn_ba(xb, xa, xa, need_weights=False)
        gate = torch.sigmoid(self.update_gate(pose_embed))

        xa = xa + gate.unsqueeze(1) * ab_ctx
        xb = xb + gate.unsqueeze(1) * ba_ctx
        xa = xa + gate.unsqueeze(1) * self.ffn_a(xa)
        xb = xb + gate.unsqueeze(1) * self.ffn_b(xb)

        pooled_a = self._pool_stats(xa)
        pooled_b = self._pool_stats(xb)
        pooled_cross = self._pool_stats(0.5 * (ab_ctx + ba_ctx))
        return {
            "tokens_a": xa,
            "tokens_b": xb,
            "pair_summary": torch.cat([pooled_a, pooled_b, pooled_cross, pose_embed], dim=-1),
            "fine_gate": gate.view(-1),
        }


class FineResidualPoseHead(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: int = 256,
        rot_scale: float = 0.08,
        tdir_scale: float = 0.25,
        log_tmag_scale: float = 0.35,
        gate_bias: float = -2.5,
        gate_max: float = 0.5,
    ) -> None:
        super().__init__()
        self.rot_scale = float(rot_scale)
        self.tdir_scale = float(tdir_scale)
        self.log_tmag_scale = float(log_tmag_scale)
        self.gate_max = float(max(0.0, gate_max))
        in_dim = 10 * dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.rot_head = nn.Linear(hidden_dim, 3)
        self.tdir_head = nn.Linear(hidden_dim, 3)
        self.log_tmag_head = nn.Linear(hidden_dim, 1)
        self.gate_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.rot_head.weight)
        nn.init.zeros_(self.rot_head.bias)
        nn.init.zeros_(self.tdir_head.weight)
        nn.init.zeros_(self.tdir_head.bias)
        nn.init.zeros_(self.log_tmag_head.weight)
        nn.init.zeros_(self.log_tmag_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(gate_bias))

    def forward(self, pair_summary: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.net(pair_summary.float())
        delta_rot_vec = self.rot_scale * torch.tanh(self.rot_head(h))
        delta_tdir = self.tdir_scale * torch.tanh(self.tdir_head(h))
        delta_log_tmag = self.log_tmag_scale * torch.tanh(self.log_tmag_head(h).squeeze(-1))
        gate = torch.sigmoid(self.gate_head(h))
        if self.gate_max < 1.0:
            gate = gate * self.gate_max
        return {
            "delta_rot_vec": delta_rot_vec,
            "delta_tdir": delta_tdir,
            "delta_log_tmag": delta_log_tmag,
            "residual_gate": gate,
        }


class ResidualMagnitudeRegularizer(nn.Module):
    def forward(
        self,
        delta_rot_vec: torch.Tensor,
        delta_tdir: torch.Tensor,
        delta_log_tmag: torch.Tensor,
        gate: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        gate = gate.float().view(-1, 1)
        applied_rot = gate * delta_rot_vec.float()
        applied_tdir = gate * delta_tdir.float()
        applied_log = gate.view(-1) * delta_log_tmag.float().view(-1)
        rot_norm = torch.linalg.norm(applied_rot, dim=-1)
        tdir_norm = torch.linalg.norm(applied_tdir, dim=-1)
        log_abs = torch.abs(applied_log)
        loss = rot_norm.mean() + tdir_norm.mean() + log_abs.mean()
        return {
            "loss": loss,
            "delta_rot_mean_deg": rot_norm.mean() * (180.0 / torch.pi),
            "delta_tdir_norm_mean": tdir_norm.mean(),
            "delta_log_tmag_abs_mean": log_abs.mean(),
        }


class AblationTokenSampler(nn.Module):
    def __init__(
        self,
        cfg: Config,
        device: torch.device,
        *,
        geometry_mode: str,
        use_fine_tokens: bool,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.geometry_mode = str(geometry_mode)
        self.use_fine_tokens = bool(use_fine_tokens)
        self.patch_embed_c = PatchEmbed(cfg.p, cfg.D, in_ch=cfg.in_ch)
        self.patch_embed_f = PatchEmbed(cfg.p, cfg.D, in_ch=cfg.in_ch)
        self.enc_c = TokenEncoder(cfg.D, cfg.n_layers, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
        self.enc_f = TokenEncoder(cfg.D, cfg.n_layers, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
        if self.geometry_mode == "spherical":
            self.hier = HealpixHierarchy(cfg.Nc, cfg.Nf, cfg.p, device=device)
            level_data = self.hier.level_data()
            self.register_buffer(
                "grid_coarse",
                build_level_grid_from_patch_bearings(level_data["coarse"]["patch_bearing"], cfg.H, cfg.W),
                persistent=True,
            )
            self.register_buffer(
                "bearing_coarse",
                level_data["coarse"]["bearing"],
                persistent=True,
            )
            self.register_buffer(
                "id_coarse",
                level_data["coarse"]["id"],
                persistent=True,
            )
            if self.use_fine_tokens:
                self.register_buffer(
                    "grid_fine",
                    build_level_grid_from_patch_bearings(level_data["fine"]["patch_bearing"], cfg.H, cfg.W),
                    persistent=True,
                )
                self.register_buffer("bearing_fine", level_data["fine"]["bearing"], persistent=True)
                self.register_buffer("id_fine", level_data["fine"]["id"], persistent=True)
                self.register_buffer("parent_fine", level_data["fine"]["parent_id"], persistent=True)
            self.pos_enc = BearingPosEnc(cfg.D)
            self.register_buffer("gridpos_coarse", torch.zeros(cfg.Nc, 2, dtype=torch.float32), persistent=False)
            if self.use_fine_tokens:
                self.register_buffer("gridpos_fine", torch.zeros(cfg.Nf, 2, dtype=torch.float32), persistent=False)
        elif self.geometry_mode == "planar":
            grid_c, bearing_c, gridpos_c = _build_planar_level_metadata(
                int(getattr(cfg, "planar_coarse_rows", 12)),
                int(getattr(cfg, "planar_coarse_cols", 16)),
                cfg.p,
                cfg.H,
                cfg.W,
            )
            self.register_buffer("grid_coarse", grid_c.to(device=device), persistent=True)
            self.register_buffer("bearing_coarse", bearing_c.to(device=device), persistent=True)
            self.register_buffer("id_coarse", torch.arange(bearing_c.shape[0], device=device, dtype=torch.long), persistent=True)
            self.register_buffer("gridpos_coarse", gridpos_c.to(device=device), persistent=True)
            if self.use_fine_tokens:
                grid_f, bearing_f, gridpos_f = _build_planar_level_metadata(
                    int(getattr(cfg, "planar_fine_rows", 24)),
                    int(getattr(cfg, "planar_fine_cols", 32)),
                    cfg.p,
                    cfg.H,
                    cfg.W,
                )
                self.register_buffer("grid_fine", grid_f.to(device=device), persistent=True)
                self.register_buffer("bearing_fine", bearing_f.to(device=device), persistent=True)
                self.register_buffer("id_fine", torch.arange(bearing_f.shape[0], device=device, dtype=torch.long), persistent=True)
                self.register_buffer("parent_fine", (torch.arange(bearing_f.shape[0], device=device, dtype=torch.long) // 4), persistent=True)
                self.register_buffer("gridpos_fine", gridpos_f.to(device=device), persistent=True)
            self.pos_enc = GridPositionalEncoding(cfg.D)
        else:
            raise ValueError(f"Unsupported geometry_mode: {self.geometry_mode}")

    def _encode_level(
        self,
        img: torch.Tensor,
        *,
        level: str,
    ) -> Tuple[Tokens, torch.Tensor]:
        if level == "coarse":
            grid = self.grid_coarse
            bearing = self.bearing_coarse
            token_id = self.id_coarse
            parent_id = None
            patch_embed = self.patch_embed_c
            encoder = self.enc_c
            gridpos = self.gridpos_coarse
            N = int(bearing.shape[0])
        else:
            grid = self.grid_fine
            bearing = self.bearing_fine
            token_id = self.id_fine
            parent_id = self.parent_fine
            patch_embed = self.patch_embed_f
            encoder = self.enc_f
            gridpos = self.gridpos_fine
            N = int(bearing.shape[0])
        patches = sample_patches_erp(img, grid, N=N, p=self.cfg.p)
        feat = patch_embed(patches)
        B = img.shape[0]
        bearing_b = bearing.view(1, N, 3).expand(B, -1, -1)
        gridpos_b = gridpos.view(1, N, 2).expand(B, -1, -1)
        if self.geometry_mode == "spherical":
            feat = feat + self.pos_enc(bearing_b)
        else:
            feat = feat + self.pos_enc(gridpos_b)
        feat = encoder(feat)
        token_id_b = token_id.view(1, N).expand(B, -1)
        parent_id_b = None if parent_id is None else parent_id.view(1, N).expand(B, -1)
        return Tokens(feat=feat, bearing=bearing_b, level=level, id=token_id_b, parent_id=parent_id_b), gridpos_b

    def forward(self, IA: torch.Tensor, IB: torch.Tensor) -> Dict[str, torch.Tensor | Tokens]:
        tok_a_c, grid_a_c = self._encode_level(IA, level="coarse")
        tok_b_c, grid_b_c = self._encode_level(IB, level="coarse")
        out: Dict[str, torch.Tensor | Tokens] = {
            "TokA_c": tok_a_c,
            "TokB_c": tok_b_c,
            "gridA_c": grid_a_c,
            "gridB_c": grid_b_c,
        }
        if self.use_fine_tokens:
            tok_a_f, grid_a_f = self._encode_level(IA, level="fine")
            tok_b_f, grid_b_f = self._encode_level(IB, level="fine")
            out["TokA_f"] = tok_a_f
            out["TokB_f"] = tok_b_f
            out["gridA_f"] = grid_a_f
            out["gridB_f"] = grid_b_f
        return out


class AblationBaseModel(nn.Module):
    def __init__(self, cfg: Config, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.log_tmag_bias = None
        self.tmag_affine_scale = None
        self.tmag_affine_bias = None

    @staticmethod
    def _stats_pool(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)
        std = torch.sqrt(x.var(dim=1, unbiased=False).clamp_min(1.0e-8))
        maxv = x.max(dim=1).values
        return torch.cat([mean, maxv, std], dim=-1)

    def _zero_aux(self, aux: Dict[str, torch.Tensor], batch_size: int, device: torch.device) -> None:
        z3 = torch.zeros((batch_size, 3), device=device, dtype=torch.float32)
        z1 = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        zg = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)
        aux["delta_rot_vec_raw"] = z3
        aux["delta_tdir_vec_raw"] = z3
        aux["delta_log_tmag_raw"] = z1
        aux["delta_rot_vec"] = z3
        aux["delta_tdir_vec"] = z3
        aux["delta_log_tmag"] = z1
        aux["delta_rot_norm"] = z1
        aux["delta_tdir_norm"] = z1
        aux["delta_log_tmag_abs"] = z1
        aux["residual_gate"] = zg
        aux["fine_gate"] = z1
        aux["alpha"] = torch.tensor(0.0, device=device, dtype=torch.float32)
        aux["beta"] = torch.tensor(0.0, device=device, dtype=torch.float32)

    def residual_regularization(self, aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        zero = aux["t_mag"].float().mean() * 0.0
        return {
            "loss": zero,
            "delta_rot_mean_deg": zero,
            "delta_tdir_norm_mean": zero,
            "delta_log_tmag_abs_mean": zero,
        }


class AblationPairRegressionModel(AblationBaseModel):
    def __init__(self, cfg: Config, device: torch.device, *, geometry_mode: str, model_name: str) -> None:
        super().__init__(cfg, device)
        self.model_name = str(model_name)
        self.module2 = AblationTokenSampler(cfg, device, geometry_mode=geometry_mode, use_fine_tokens=False)
        self.pair_head = PairPoseScaleHead(cfg.D, hidden_dim=int(getattr(cfg, "abl_pair_hidden_dim", 256)))

    def forward(
        self,
        IA: torch.Tensor,
        IB: torch.Tensor,
        *,
        enable_depth_fusion: Optional[bool] = None,
        dt_world: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        del enable_depth_fusion
        tokens = self.module2(IA, IB)
        tok_a = tokens["TokA_c"]
        tok_b = tokens["TokB_c"]
        grid_a = tokens["gridA_c"]
        grid_b = tokens["gridB_c"]
        R, tdir_out, log_tmag = self.pair_head(tok_a.feat, tok_b.feat)
        log_tmag = log_tmag.clamp(
            min=float(getattr(self.cfg, "log_tmag_clamp_min", -6.0)),
            max=float(getattr(self.cfg, "log_tmag_clamp_max", 6.0)),
        )
        t_mag = torch.exp(log_tmag).clamp_min(float(getattr(self.cfg, "tmag_min", 1.0e-3)))
        tdir_local = F.normalize(torch.matmul(R.transpose(-1, -2), tdir_out.unsqueeze(-1)).squeeze(-1), dim=-1, eps=1.0e-6)
        aux: Dict[str, torch.Tensor] = {
            "bearingA_c": tok_a.bearing,
            "bearingB_c": tok_b.bearing,
            "gridA_c": grid_a,
            "gridB_c": grid_b,
            "coarse_R": R,
            "coarse_t_dir_out": tdir_out,
            "coarse_t_mag": t_mag,
            "coarse_log_t_mag": log_tmag,
            "coarse_pair_context": self._stats_pool(0.5 * (tok_a.feat.float() + tok_b.feat.float())),
            "stage": self.model_name,
        }
        _set_transform_outputs(aux, R, tdir_local, t_mag, log_tmag)
        self._zero_aux(aux, IA.shape[0], IA.device)
        _apply_dt_bucket_scale_anchor(self.cfg, aux, dt_world=dt_world)
        return R, aux["t_dir"], aux


class AblationSingleStageCrossModel(AblationBaseModel):
    def __init__(self, cfg: Config, device: torch.device) -> None:
        super().__init__(cfg, device)
        self.module2 = AblationTokenSampler(cfg, device, geometry_mode="spherical", use_fine_tokens=False)
        self.coarse = CoarseInteraction(
            cfg.D,
            temperature=cfg.coarse_temperature,
            logits_clip=cfg.logits_clip,
            pose_use_stats_pool=bool(getattr(cfg, "pose_use_stats_pool", True)),
            use_bearing_fuse=bool(getattr(cfg, "use_bearing_fuse", False)),
            use_translation_feature_branch=False,
            use_translation_magnitude_head=True,
            tmag_pred_source="pose_feat",
            tmag_detach_features=bool(getattr(cfg, "tmag_detach_features", False)),
            tmag_min=float(getattr(cfg, "tmag_min", 1.0e-3)),
            log_tmag_clamp_min=float(getattr(cfg, "log_tmag_clamp_min", -6.0)),
            log_tmag_clamp_max=float(getattr(cfg, "log_tmag_clamp_max", 6.0)),
            tmag_condition_on_dt=bool(getattr(cfg, "tmag_condition_on_dt", False)),
        )

    def forward(
        self,
        IA: torch.Tensor,
        IB: torch.Tensor,
        *,
        enable_depth_fusion: Optional[bool] = None,
        dt_world: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        del enable_depth_fusion
        tokens = self.module2(IA, IB)
        tok_a = tokens["TokA_c"]
        tok_b = tokens["TokB_c"]
        out_c = self.coarse(
            tok_a,
            tok_b,
            tokens_a_t=None,
            tokens_b_t=None,
            dt_world=dt_world,
            tmag_dt_clamp_min=float(getattr(self.cfg, "tmag_dt_clamp_min", 0.01)),
        )
        coarse_tdir_out = _local_t_to_output_frame(out_c["Rc"], out_c["tc_dir"])
        aux: Dict[str, torch.Tensor] = {
            "bearingA_c": tok_a.bearing,
            "bearingB_c": tok_b.bearing,
            "coarse_R": out_c["Rc"],
            "coarse_t_dir_out": coarse_tdir_out,
            "coarse_t_mag": out_c["tc_mag"],
            "coarse_log_t_mag": out_c["log_tc_mag"],
            "coarse_pair_context": self._stats_pool(out_c["Fc"]),
            "stage": "ABLVO360_SingleStagePoseRegression",
        }
        aux.update(out_c)
        _set_transform_outputs(aux, out_c["Rc"], out_c["tc_dir"], out_c["tc_mag"], out_c["log_tc_mag"])
        self._zero_aux(aux, IA.shape[0], IA.device)
        _apply_dt_bucket_scale_anchor(self.cfg, aux, dt_world=dt_world)
        return out_c["Rc"], aux["t_dir"], aux


class AblationNoSphericalGeometryModel(AblationBaseModel):
    def __init__(self, cfg: Config, device: torch.device) -> None:
        super().__init__(cfg, device)
        self.module2 = AblationTokenSampler(cfg, device, geometry_mode="planar", use_fine_tokens=True)
        self.coarse = CoarseInteraction(
            cfg.D,
            temperature=cfg.coarse_temperature,
            logits_clip=cfg.logits_clip,
            pose_use_stats_pool=bool(getattr(cfg, "pose_use_stats_pool", True)),
            use_bearing_fuse=False,
            use_translation_feature_branch=False,
            use_translation_magnitude_head=True,
            tmag_pred_source="pose_feat",
            tmag_detach_features=bool(getattr(cfg, "tmag_detach_features", False)),
            tmag_min=float(getattr(cfg, "tmag_min", 1.0e-3)),
            log_tmag_clamp_min=float(getattr(cfg, "log_tmag_clamp_min", -6.0)),
            log_tmag_clamp_max=float(getattr(cfg, "log_tmag_clamp_max", 6.0)),
            tmag_condition_on_dt=bool(getattr(cfg, "tmag_condition_on_dt", False)),
        )
        self.struct360b_pose_embed = PoseEmbeddingMLP(cfg.D)
        self.struct360b_fine_refiner = PlanarPoseConditionedFineTokenRefiner(
            cfg.D,
            num_heads=int(getattr(cfg, "struct360b_fine_heads", max(1, cfg.n_heads // 2))),
            dropout=float(getattr(cfg, "struct360b_fine_dropout", cfg.dropout)),
            mlp_ratio=float(getattr(cfg, "struct360b_fine_mlp_ratio", 2.0)),
            gate_bias=float(getattr(cfg, "struct360b_refiner_gate_bias", -2.0)),
        )
        self.struct360b_residual_head = FineResidualPoseHead(
            cfg.D,
            hidden_dim=int(getattr(cfg, "struct360b_residual_hidden_dim", 256)),
            rot_scale=float(getattr(cfg, "struct360b_delta_rot_scale", 0.08)),
            tdir_scale=float(getattr(cfg, "struct360b_delta_tdir_scale", 0.25)),
            log_tmag_scale=float(getattr(cfg, "struct360b_delta_log_tmag_scale", 0.35)),
            gate_bias=float(getattr(cfg, "struct360b_residual_gate_bias", -2.5)),
            gate_max=float(getattr(cfg, "struct360b_residual_gate_max", 0.5)),
        )
        self.struct360b_residual_regularizer = ResidualMagnitudeRegularizer()

    def residual_regularization(self, aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.struct360b_residual_regularizer(
            aux["delta_rot_vec_raw"],
            aux["delta_tdir_vec_raw"],
            aux["delta_log_tmag_raw"],
            aux["residual_gate"],
        )

    def forward(
        self,
        IA: torch.Tensor,
        IB: torch.Tensor,
        *,
        enable_depth_fusion: Optional[bool] = None,
        dt_world: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        del enable_depth_fusion
        tokens = self.module2(IA, IB)
        tok_a_c = tokens["TokA_c"]
        tok_b_c = tokens["TokB_c"]
        tok_a_f = tokens["TokA_f"]
        tok_b_f = tokens["TokB_f"]
        out_c = self.coarse(
            tok_a_c,
            tok_b_c,
            tokens_a_t=None,
            tokens_b_t=None,
            dt_world=dt_world,
            tmag_dt_clamp_min=float(getattr(self.cfg, "tmag_dt_clamp_min", 0.01)),
        )
        coarse_tdir_out = _local_t_to_output_frame(out_c["Rc"], out_c["tc_dir"])
        coarse_pair_context = self._stats_pool(out_c["Fc"])
        pose_embed = self.struct360b_pose_embed(
            out_c["Rc"],
            coarse_tdir_out,
            out_c["log_tc_mag"],
            coarse_pair_context,
        )
        refine_out = self.struct360b_fine_refiner(
            tok_a_f,
            tok_b_f,
            pose_embed,
            tokens["gridA_f"],
            tokens["gridB_f"],
        )
        residual_out = self.struct360b_residual_head(refine_out["pair_summary"])
        gate = residual_out["residual_gate"].float()
        delta_rot_applied = gate * residual_out["delta_rot_vec"].float()
        delta_tdir_applied = gate * residual_out["delta_tdir"].float()
        delta_log_tmag_applied = gate.view(-1) * residual_out["delta_log_tmag"].float().view(-1)

        alpha = float(getattr(self.cfg, "struct360b_alpha", 0.25))
        beta = float(getattr(self.cfg, "struct360b_beta", 0.25))
        delta_R = _so3_exp_map(delta_rot_applied)
        R_final = torch.matmul(delta_R, out_c["Rc"].float())
        tdir_out_final = F.normalize(coarse_tdir_out.float() + alpha * delta_tdir_applied, dim=-1, eps=1.0e-6)
        tdir_local_final = F.normalize(
            torch.matmul(R_final.transpose(-1, -2), tdir_out_final.unsqueeze(-1)).squeeze(-1),
            dim=-1,
            eps=1.0e-6,
        )
        log_tmag_final = (out_c["log_tc_mag"].float() + beta * delta_log_tmag_applied).clamp(
            min=float(getattr(self.cfg, "log_tmag_clamp_min", -6.0)),
            max=float(getattr(self.cfg, "log_tmag_clamp_max", 6.0)),
        )
        tmag_final = torch.exp(log_tmag_final).clamp_min(float(getattr(self.cfg, "tmag_min", 1.0e-3)))
        aux: Dict[str, torch.Tensor] = {
            "bearingA_c": tok_a_c.bearing,
            "bearingB_c": tok_b_c.bearing,
            "bearingA_f": tok_a_f.bearing,
            "bearingB_f": tok_b_f.bearing,
            "gridA_c": tokens["gridA_c"],
            "gridB_c": tokens["gridB_c"],
            "gridA_f": tokens["gridA_f"],
            "gridB_f": tokens["gridB_f"],
            "coarse_R": out_c["Rc"],
            "coarse_t_dir_out": coarse_tdir_out,
            "coarse_t_mag": out_c["tc_mag"],
            "coarse_log_t_mag": out_c["log_tc_mag"],
            "coarse_pair_context": coarse_pair_context,
            "pose_embed": pose_embed,
            "fine_gate": refine_out["fine_gate"],
            "residual_gate": gate,
            "delta_rot_vec_raw": residual_out["delta_rot_vec"],
            "delta_tdir_vec_raw": residual_out["delta_tdir"],
            "delta_log_tmag_raw": residual_out["delta_log_tmag"],
            "delta_rot_vec": delta_rot_applied,
            "delta_tdir_vec": delta_tdir_applied,
            "delta_log_tmag": delta_log_tmag_applied,
            "delta_rot_norm": torch.linalg.norm(delta_rot_applied, dim=-1),
            "delta_tdir_norm": torch.linalg.norm(delta_tdir_applied, dim=-1),
            "delta_log_tmag_abs": torch.abs(delta_log_tmag_applied),
            "alpha": torch.tensor(alpha, device=IA.device, dtype=torch.float32),
            "beta": torch.tensor(beta, device=IA.device, dtype=torch.float32),
            "stage": "ABLVO360_NoSphericalGeometry",
        }
        aux.update(out_c)
        _set_transform_outputs(aux, R_final, tdir_local_final, tmag_final, log_tmag_final)
        _apply_dt_bucket_scale_anchor(self.cfg, aux, dt_world=dt_world)
        return R_final, aux["t_dir"], aux


def build_ablvo360_model(variant: str, cfg: Config, device: torch.device) -> nn.Module:
    variant = str(variant)
    if variant == "ABLVO360_PlainPairVO":
        return AblationPairRegressionModel(cfg, device, geometry_mode="planar", model_name=variant)
    if variant == "ABLVO360_NoCrossImageInteraction":
        return AblationPairRegressionModel(cfg, device, geometry_mode="spherical", model_name=variant)
    if variant == "ABLVO360_SingleStagePoseRegression":
        return AblationSingleStageCrossModel(cfg, device)
    if variant == "ABLVO360_NoSphericalGeometry":
        return AblationNoSphericalGeometryModel(cfg, device)
    raise ValueError(f"Unsupported ABLVO360 variant: {variant}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
