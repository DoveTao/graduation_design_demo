from __future__ import annotations

from typing import Dict, Tuple

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
    PoseConditionedFineTokenRefiner,
    PoseEmbeddingMLP,
    ResidualMagnitudeRegularizer,
)


class FINAL360LFactorizedTranslationResidualHead(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        shared_hidden_dim: int = 256,
        tdir_branch_hidden_dim: int = 192,
        conf_branch_hidden_dim: int = 128,
        log_tmag_branch_hidden_dim: int = 128,
        rot_scale: float = 0.08,
        tdir_scale: float = 0.25,
        log_tmag_scale: float = 0.35,
        gate_bias: float = -2.5,
        gate_max: float = 0.5,
        confidence_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.rot_scale = float(rot_scale)
        self.tdir_scale = float(tdir_scale)
        self.log_tmag_scale = float(log_tmag_scale)
        self.gate_max = float(max(0.0, gate_max))
        in_dim = 10 * dim

        self.shared = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, shared_hidden_dim),
            nn.GELU(),
            nn.Linear(shared_hidden_dim, shared_hidden_dim),
            nn.GELU(),
        )
        self.rot_head = nn.Linear(shared_hidden_dim, 3)
        self.gate_head = nn.Linear(shared_hidden_dim, 1)

        self.tdir_branch = nn.Sequential(
            nn.LayerNorm(shared_hidden_dim),
            nn.Linear(shared_hidden_dim, tdir_branch_hidden_dim),
            nn.GELU(),
            nn.Linear(tdir_branch_hidden_dim, 3),
        )
        self.conf_branch = nn.Sequential(
            nn.LayerNorm(shared_hidden_dim),
            nn.Linear(shared_hidden_dim, conf_branch_hidden_dim),
            nn.GELU(),
            nn.Linear(conf_branch_hidden_dim, 1),
        )
        self.log_tmag_branch = nn.Sequential(
            nn.LayerNorm(shared_hidden_dim),
            nn.Linear(shared_hidden_dim, log_tmag_branch_hidden_dim),
            nn.GELU(),
            nn.Linear(log_tmag_branch_hidden_dim, 1),
        )

        nn.init.zeros_(self.rot_head.weight)
        nn.init.zeros_(self.rot_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(gate_bias))
        nn.init.xavier_uniform_(self.tdir_branch[1].weight)
        nn.init.zeros_(self.tdir_branch[1].bias)
        nn.init.zeros_(self.tdir_branch[-1].weight)
        nn.init.zeros_(self.tdir_branch[-1].bias)
        nn.init.xavier_uniform_(self.conf_branch[1].weight)
        nn.init.zeros_(self.conf_branch[1].bias)
        nn.init.zeros_(self.conf_branch[-1].weight)
        nn.init.constant_(self.conf_branch[-1].bias, float(confidence_bias))
        nn.init.xavier_uniform_(self.log_tmag_branch[1].weight)
        nn.init.zeros_(self.log_tmag_branch[1].bias)
        nn.init.zeros_(self.log_tmag_branch[-1].weight)
        nn.init.zeros_(self.log_tmag_branch[-1].bias)

    def forward(self, pair_summary: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.shared(pair_summary.float())
        delta_rot_vec = self.rot_scale * torch.tanh(self.rot_head(h))
        raw_tdir = self.tdir_branch(h)
        delta_tdir = self.tdir_scale * torch.tanh(raw_tdir)
        delta_log_tmag = self.log_tmag_scale * torch.tanh(self.log_tmag_branch(h).squeeze(-1))
        conf_logit = self.conf_branch(h).squeeze(-1)
        confidence = torch.sigmoid(conf_logit)
        gate = torch.sigmoid(self.gate_head(h))
        if self.gate_max < 1.0:
            gate = gate * self.gate_max
        return {
            "delta_rot_vec": delta_rot_vec,
            "delta_tdir": delta_tdir,
            "delta_log_tmag": delta_log_tmag,
            "residual_gate": gate,
            "tdir_confidence_logit": conf_logit,
            "tdir_confidence": confidence,
        }


class FINAL360LTranslationHeadFactorizationModel(PanoramaRelPoseModel):
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
        self.struct360l_residual_head = FINAL360LFactorizedTranslationResidualHead(
            cfg.D,
            shared_hidden_dim=int(getattr(cfg, "struct360b_residual_hidden_dim", 256)),
            tdir_branch_hidden_dim=int(getattr(cfg, "final360l_tdir_branch_hidden_dim", 192)),
            conf_branch_hidden_dim=int(getattr(cfg, "final360l_conf_branch_hidden_dim", 128)),
            log_tmag_branch_hidden_dim=int(getattr(cfg, "final360l_log_tmag_branch_hidden_dim", 128)),
            rot_scale=float(getattr(cfg, "struct360b_delta_rot_scale", 0.08)),
            tdir_scale=float(getattr(cfg, "struct360b_delta_tdir_scale", 0.25)),
            log_tmag_scale=float(getattr(cfg, "struct360b_delta_log_tmag_scale", 0.35)),
            gate_bias=float(getattr(cfg, "struct360b_residual_gate_bias", -2.5)),
            gate_max=float(getattr(cfg, "struct360b_residual_gate_max", 0.5)),
            confidence_bias=float(getattr(cfg, "final360l_confidence_bias", 0.0)),
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
        enable_depth_fusion: bool | None = None,
        dt_world: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        del enable_depth_fusion
        tokens = self.module2(IA, IB)
        tok_a_c: Tokens = tokens["TokA_c"]
        tok_b_c: Tokens = tokens["TokB_c"]
        tok_a_f: Tokens = tokens["TokA_f"]
        tok_b_f: Tokens = tokens["TokB_f"]

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
        residual_out = self.struct360l_residual_head(refine_out["pair_summary"])

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
        aux["tdir_confidence_logit"] = residual_out["tdir_confidence_logit"]
        aux["tdir_confidence"] = residual_out["tdir_confidence"]
        aux["tdir_confidence_mean"] = residual_out["tdir_confidence"].mean()
        aux["tdir_confidence_std"] = residual_out["tdir_confidence"].std(unbiased=False)

        _set_transform_outputs(aux, R_final, tdir_local_final, tmag_final, log_tmag_final)
        _apply_dt_bucket_scale_anchor(self.cfg, aux, dt_world=dt_world)
        aux["stage"] = "translation_head_factorization_direction_confidence_scale"
        return R_final, aux["t_dir"], aux


FINAL360LModel = FINAL360LTranslationHeadFactorizationModel
