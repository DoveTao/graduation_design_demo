from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from train360.core.config import Config
from train360.core.interaction import Tokens
from train360.core.model import PanoramaRelPoseModel, _apply_dt_bucket_scale_anchor, _local_t_to_output_frame, _set_transform_outputs, _so3_exp_map


class SphericalPositionalEncoding(nn.Module):
    """ERP-aware spherical encoding for latent fine-stage refinement."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, bearing: torch.Tensor) -> torch.Tensor:
        bearing = F.normalize(bearing.float(), dim=-1, eps=1e-6)
        x, y, z = bearing.unbind(dim=-1)
        lon = torch.atan2(x, z)
        lat = torch.asin(y.clamp(-1.0, 1.0))
        cos_lat = torch.cos(lat).clamp_min(1.0e-3)
        pole_bias = 1.0 - cos_lat
        feat = torch.stack(
            [
                x,
                y,
                z,
                torch.sin(lon),
                torch.cos(lon),
                torch.sin(lat),
                torch.cos(lat),
                cos_lat,
                pole_bias,
                lat / 1.5707963267948966,
            ],
            dim=-1,
        )
        return self.net(feat)


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
                F.normalize(coarse_tdir_out.float(), dim=-1, eps=1e-6),
                coarse_log_tmag.float().view(-1, 1),
                coarse_pair_context.float(),
            ],
            dim=-1,
        )
        return self.net(x)


class PoseConditionedFineTokenRefiner(nn.Module):
    """Latent cross-image refinement that never emits explicit match lists."""

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
        self.pos_enc = SphericalPositionalEncoding(dim)
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
        std = torch.sqrt(x.var(dim=1, unbiased=False).clamp_min(1e-8))
        maxv = x.max(dim=1).values
        return torch.cat([mean, maxv, std], dim=-1)

    def forward(
        self,
        tokens_a: Tokens,
        tokens_b: Tokens,
        pose_embed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pose_token = self.pose_to_token(pose_embed).unsqueeze(1)
        xa = tokens_a.feat.float() + self.pos_enc(tokens_a.bearing) + pose_token
        xb = tokens_b.feat.float() + self.pos_enc(tokens_b.bearing) + pose_token
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


class STRUCT360BMatchFreeCoarseToFineModel(PanoramaRelPoseModel):
    def __init__(self, cfg: Config, device: torch.device):
        super().__init__(cfg, device)
        self.struct360b_pose_embed = PoseEmbeddingMLP(cfg.D)
        self.struct360b_fine_refiner = PoseConditionedFineTokenRefiner(
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
        for frozen_module_name in ("fine", "direct_head", "coupled_pose_head"):
            module = getattr(self, frozen_module_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False

    @staticmethod
    def _stats_pool(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)
        std = torch.sqrt(x.var(dim=1, unbiased=False).clamp_min(1e-8))
        maxv = x.max(dim=1).values
        return torch.cat([mean, maxv, std], dim=-1)

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
        del enable_depth_fusion  # STRUCT360B keeps a pure match-free pose refinement path.
        tokens = self.module2(IA, IB)
        tok_a_c = tokens["TokA_c"]
        tok_b_c = tokens["TokB_c"]
        tok_a_f = tokens["TokA_f"]
        tok_b_f = tokens["TokB_f"]

        aux: Dict[str, torch.Tensor] = {
            "bearingA_c": tok_a_c.bearing,
            "bearingB_c": tok_b_c.bearing,
            "bearingA_f": tok_a_f.bearing,
            "bearingB_f": tok_b_f.bearing,
            "t_local_frame": self.cfg.translation_local_frame,
            "t_output_frame": self.cfg.translation_output_frame,
        }

        out_c = self.coarse(
            tok_a_c,
            tok_b_c,
            tokens_a_t=tokens.get("TokA_c_t"),
            tokens_b_t=tokens.get("TokB_c_t"),
            dt_world=dt_world,
            tmag_dt_clamp_min=float(getattr(self.cfg, "tmag_dt_clamp_min", 0.01)),
        )
        aux.update(out_c)
        coarse_tdir_out = _local_t_to_output_frame(out_c["Rc"], out_c["tc_dir"])
        coarse_tmag_eval, coarse_log_tmag_eval = self._bias_magnitude(out_c["tc_mag"], out_c["log_tc_mag"])
        aux["tc_dir_local"] = out_c["tc_dir"]
        aux["tc_dir_out"] = coarse_tdir_out
        aux["tc_mag_eval"] = coarse_tmag_eval
        aux["log_tc_mag_eval"] = coarse_log_tmag_eval
        aux["coarse_R"] = out_c["Rc"]
        aux["coarse_t_dir_out"] = coarse_tdir_out
        aux["coarse_t_mag"] = coarse_tmag_eval
        aux["coarse_log_t_mag"] = coarse_log_tmag_eval
        aux["coarse_pair_context"] = self._stats_pool(out_c["Fc"])

        pose_embed = self.struct360b_pose_embed(
            out_c["Rc"],
            coarse_tdir_out,
            out_c["log_tc_mag"],
            aux["coarse_pair_context"],
        )
        refine_out = self.struct360b_fine_refiner(tok_a_f, tok_b_f, pose_embed)
        residual_out = self.struct360b_residual_head(refine_out["pair_summary"])

        gate = residual_out["residual_gate"].float()
        delta_rot_applied = gate * residual_out["delta_rot_vec"].float()
        delta_tdir_applied = gate * residual_out["delta_tdir"].float()
        delta_log_tmag_applied = gate.view(-1) * residual_out["delta_log_tmag"].float().view(-1)

        alpha = float(getattr(self.cfg, "struct360b_alpha", 0.25))
        beta = float(getattr(self.cfg, "struct360b_beta", 0.25))
        delta_R = _so3_exp_map(delta_rot_applied)
        R_final = torch.matmul(delta_R, out_c["Rc"].float())
        tdir_out_final = F.normalize(coarse_tdir_out.float() + alpha * delta_tdir_applied, dim=-1, eps=1e-6)
        tdir_local_final = F.normalize(
            torch.matmul(R_final.transpose(-1, -2), tdir_out_final.unsqueeze(-1)).squeeze(-1),
            dim=-1,
            eps=1e-6,
        )
        log_tmag_final_unbiased = out_c["log_tc_mag"].float() + beta * delta_log_tmag_applied
        tmag_final_unbiased = torch.exp(
            log_tmag_final_unbiased.clamp(
                min=float(getattr(self.cfg, "log_tmag_clamp_min", -6.0)),
                max=float(getattr(self.cfg, "log_tmag_clamp_max", 6.0)),
            )
        ).clamp_min(float(getattr(self.cfg, "tmag_min", 1.0e-3)))
        tmag_final, log_tmag_final = self._bias_magnitude(tmag_final_unbiased, log_tmag_final_unbiased)

        aux["pose_embed"] = pose_embed
        aux["fine_gate"] = refine_out["fine_gate"]
        aux["residual_gate"] = gate
        aux["delta_rot_vec_raw"] = residual_out["delta_rot_vec"]
        aux["delta_tdir_vec_raw"] = residual_out["delta_tdir"]
        aux["delta_log_tmag_raw"] = residual_out["delta_log_tmag"]
        aux["delta_rot_vec"] = delta_rot_applied
        aux["delta_tdir_vec"] = delta_tdir_applied
        aux["delta_log_tmag"] = delta_log_tmag_applied
        aux["delta_rot_norm"] = torch.linalg.norm(delta_rot_applied, dim=-1)
        aux["delta_tdir_norm"] = torch.linalg.norm(delta_tdir_applied, dim=-1)
        aux["delta_log_tmag_abs"] = torch.abs(delta_log_tmag_applied)
        aux["residual_gate_mean"] = gate.mean()
        aux["residual_gate_median"] = gate.view(-1).median()
        aux["residual_gate_max"] = gate.max()
        aux["alpha"] = torch.tensor(alpha, device=R_final.device, dtype=torch.float32)
        aux["beta"] = torch.tensor(beta, device=R_final.device, dtype=torch.float32)
        aux["R_before_residual"] = out_c["Rc"]
        aux["tdir_before_residual"] = coarse_tdir_out
        aux["tmag_before_residual"] = coarse_tmag_eval
        aux["log_tmag_before_residual"] = coarse_log_tmag_eval
        aux["R_after_residual"] = R_final
        aux["tdir_after_residual"] = tdir_out_final
        aux["tmag_after_residual_unbiased"] = tmag_final_unbiased
        aux["log_tmag_after_residual_unbiased"] = log_tmag_final_unbiased

        _set_transform_outputs(aux, R_final, tdir_local_final, tmag_final, log_tmag_final)
        _apply_dt_bucket_scale_anchor(self.cfg, aux, dt_world=dt_world)
        aux["stage"] = "match_free_coarse_to_fine_pose_residual_refinement"
        return R_final, aux["t_dir"], aux


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
