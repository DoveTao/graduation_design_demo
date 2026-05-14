from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from train360.core.config import Config
from train360.core.interaction import Tokens
from train360.core.model import (
    PanoramaRelPoseModel,
    _apply_dt_bucket_scale_anchor,
    _local_t_to_output_frame,
    _set_transform_outputs,
    _so3_exp_map,
)
from models.struct360b_match_free_coarse_to_fine import (
    FineResidualPoseHead,
    PoseEmbeddingMLP,
    ResidualMagnitudeRegularizer,
    SphericalPositionalEncoding,
)


class RotationAwareSphericalBias(nn.Module):
    """Coarse-rotation-aware latent attention bias on the sphere.

    The module rotates A-image sphere bearings into the B frame using coarse
    `R_BA`, then turns angular distance into an additive attention bias.
    The output is latent only and never materializes explicit correspondences.
    """

    def __init__(self, gamma: float = 1.0, clamp_min: float = -4.0, clamp_max: float = 0.0) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def forward(self, bearing_a: torch.Tensor, bearing_b: torch.Tensor, coarse_R_ba: torch.Tensor) -> torch.Tensor:
        a = F.normalize(bearing_a.float(), dim=-1, eps=1.0e-6)
        b = F.normalize(bearing_b.float(), dim=-1, eps=1.0e-6)
        rotated_a = torch.matmul(coarse_R_ba.float(), a.transpose(1, 2)).transpose(1, 2)
        cosine = torch.matmul(rotated_a, b.transpose(1, 2)).clamp(-1.0, 1.0)
        theta = torch.acos(cosine)
        bias = -float(self.gamma) * theta
        return bias.clamp(min=self.clamp_min, max=self.clamp_max)


class RotationAwarePoseConditionedFineTokenRefiner(nn.Module):
    """STRUCT360C latent refiner with coarse-rotation-aware attention bias."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        dropout: float = 0.05,
        mlp_ratio: float = 2.0,
        gate_bias: float = -2.0,
        bias_gamma: float = 1.0,
        bias_clamp_min: float = -4.0,
        bias_clamp_max: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.pos_enc = SphericalPositionalEncoding(dim)
        self.pose_to_film_a = nn.Linear(dim, 2 * dim)
        self.pose_to_film_b = nn.Linear(dim, 2 * dim)
        self.pose_to_token = nn.Linear(dim, dim)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.cross_attn_ab = nn.MultiheadAttention(dim, self.num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_ba = nn.MultiheadAttention(dim, self.num_heads, dropout=dropout, batch_first=True)
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
        self.rotation_bias = RotationAwareSphericalBias(
            gamma=float(bias_gamma),
            clamp_min=float(bias_clamp_min),
            clamp_max=float(bias_clamp_max),
        )

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

    @staticmethod
    def _entropy_from_attn(attn: torch.Tensor) -> torch.Tensor:
        # attn: [B,H,L,S]
        p = attn.float().clamp_min(1.0e-8)
        return (-(p * p.log()).sum(dim=-1)).mean(dim=(-1, -2))

    def _mask_from_bias(self, bias: torch.Tensor) -> torch.Tensor:
        # nn.MultiheadAttention additive mask shape: [B*H, L, S]
        return bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1).reshape(
            bias.shape[0] * self.num_heads,
            bias.shape[1],
            bias.shape[2],
        )

    def forward(
        self,
        tokens_a: Tokens,
        tokens_b: Tokens,
        pose_embed: torch.Tensor,
        coarse_R_ba: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pose_token = self.pose_to_token(pose_embed).unsqueeze(1)
        xa = tokens_a.feat.float() + self.pos_enc(tokens_a.bearing) + pose_token
        xb = tokens_b.feat.float() + self.pos_enc(tokens_b.bearing) + pose_token
        xa = self._apply_film(self.norm_a(xa), self.pose_to_film_a(pose_embed))
        xb = self._apply_film(self.norm_b(xb), self.pose_to_film_b(pose_embed))

        bias_ab = self.rotation_bias(tokens_a.bearing, tokens_b.bearing, coarse_R_ba)
        bias_ba = self.rotation_bias(tokens_b.bearing, tokens_a.bearing, coarse_R_ba.transpose(-1, -2))

        ab_ctx, ab_attn = self.cross_attn_ab(
            xa,
            xb,
            xb,
            attn_mask=self._mask_from_bias(bias_ab),
            need_weights=True,
            average_attn_weights=False,
        )
        ba_ctx, ba_attn = self.cross_attn_ba(
            xb,
            xa,
            xa,
            attn_mask=self._mask_from_bias(bias_ba),
            need_weights=True,
            average_attn_weights=False,
        )
        gate = torch.sigmoid(self.update_gate(pose_embed))

        xa = xa + gate.unsqueeze(1) * ab_ctx
        xb = xb + gate.unsqueeze(1) * ba_ctx
        xa = xa + gate.unsqueeze(1) * self.ffn_a(xa)
        xb = xb + gate.unsqueeze(1) * self.ffn_b(xb)

        pooled_a = self._pool_stats(xa)
        pooled_b = self._pool_stats(xb)
        pooled_cross = self._pool_stats(0.5 * (ab_ctx + ba_ctx))
        entropy_ab = self._entropy_from_attn(ab_attn)
        entropy_ba = self._entropy_from_attn(ba_attn)
        return {
            "tokens_a": xa,
            "tokens_b": xb,
            "pair_summary": torch.cat([pooled_a, pooled_b, pooled_cross, pose_embed], dim=-1),
            "fine_gate": gate.view(-1),
            "attention_bias_ab": bias_ab,
            "attention_bias_ba": bias_ba,
            "attention_entropy_ab": entropy_ab,
            "attention_entropy_ba": entropy_ba,
        }


class STRUCT360CRotationAwareFineRefinementModel(PanoramaRelPoseModel):
    def __init__(self, cfg: Config, device: torch.device):
        super().__init__(cfg, device)
        self.struct360b_pose_embed = PoseEmbeddingMLP(cfg.D)
        self.struct360c_fine_refiner = RotationAwarePoseConditionedFineTokenRefiner(
            cfg.D,
            num_heads=int(getattr(cfg, "struct360b_fine_heads", max(1, cfg.n_heads // 2))),
            dropout=float(getattr(cfg, "struct360b_fine_dropout", cfg.dropout)),
            mlp_ratio=float(getattr(cfg, "struct360b_fine_mlp_ratio", 2.0)),
            gate_bias=float(getattr(cfg, "struct360b_refiner_gate_bias", -2.0)),
            bias_gamma=float(getattr(cfg, "struct360c_bias_gamma", 1.0)),
            bias_clamp_min=float(getattr(cfg, "struct360c_bias_clamp_min", -4.0)),
            bias_clamp_max=float(getattr(cfg, "struct360c_bias_clamp_max", 0.0)),
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
        std = torch.sqrt(x.var(dim=1, unbiased=False).clamp_min(1.0e-8))
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
        del enable_depth_fusion
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
        refine_out = self.struct360c_fine_refiner(tok_a_f, tok_b_f, pose_embed, out_c["Rc"])
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
        log_tmag_final_unbiased = out_c["log_tc_mag"].float() + beta * delta_log_tmag_applied
        tmag_final_unbiased = torch.exp(
            log_tmag_final_unbiased.clamp(
                min=float(getattr(self.cfg, "log_tmag_clamp_min", -6.0)),
                max=float(getattr(self.cfg, "log_tmag_clamp_max", 6.0)),
            )
        ).clamp_min(float(getattr(self.cfg, "tmag_min", 1.0e-3)))
        tmag_final, log_tmag_final = self._bias_magnitude(tmag_final_unbiased, log_tmag_final_unbiased)

        bias_ab = refine_out["attention_bias_ab"]
        bias_ba = refine_out["attention_bias_ba"]
        entropy_ab = refine_out["attention_entropy_ab"]
        entropy_ba = refine_out["attention_entropy_ba"]
        bias_all = torch.cat([bias_ab.reshape(bias_ab.shape[0], -1), bias_ba.reshape(bias_ba.shape[0], -1)], dim=-1)

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
        aux["struct360c_rotation_attention_bias_mean"] = bias_all.mean()
        aux["struct360c_rotation_attention_bias_min"] = bias_all.min()
        aux["struct360c_rotation_attention_bias_max"] = bias_all.max()
        aux["struct360c_rotation_attention_bias_ab_mean"] = bias_ab.mean()
        aux["struct360c_rotation_attention_bias_ba_mean"] = bias_ba.mean()
        aux["struct360c_attention_entropy_mean"] = 0.5 * (entropy_ab.mean() + entropy_ba.mean())
        aux["struct360c_attention_entropy_ab_mean"] = entropy_ab.mean()
        aux["struct360c_attention_entropy_ba_mean"] = entropy_ba.mean()
        aux["struct360c_attention_bias_is_latent_only"] = torch.tensor(True, device=R_final.device)
        aux["struct360c_outputs_correspondences"] = torch.tensor(False, device=R_final.device)

        _set_transform_outputs(aux, R_final, tdir_local_final, tmag_final, log_tmag_final)
        _apply_dt_bucket_scale_anchor(self.cfg, aux, dt_world=dt_world)
        aux["stage"] = "rotation_aware_match_free_fine_refinement"
        return R_final, aux["t_dir"], aux


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
