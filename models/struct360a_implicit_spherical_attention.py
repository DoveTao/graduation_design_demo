from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from train360.core.config import Config
from train360.core.interaction import Tokens
from train360.core.model import Module2Sampler, PanoramaRelPoseModel
from train360.core.transformer_encoder import MLP, BearingPosEnc, CrossContextEncoder


class SphericalPositionalEncoding(nn.Module):
    """Encode ERP-aware spherical coordinates without producing explicit matches."""

    def __init__(
        self,
        dim: int,
        *,
        use_xyz: bool = True,
        use_latlon_sincos: bool = True,
        use_erp_latitude_distortion: bool = True,
    ) -> None:
        super().__init__()
        self.use_xyz = bool(use_xyz)
        self.use_latlon_sincos = bool(use_latlon_sincos)
        self.use_erp_latitude_distortion = bool(use_erp_latitude_distortion)
        in_dim = 0
        if self.use_xyz:
            in_dim += 3
        if self.use_latlon_sincos:
            in_dim += 4
        if self.use_erp_latitude_distortion:
            in_dim += 3
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, bearing: torch.Tensor) -> torch.Tensor:
        bearing = F.normalize(bearing.float(), dim=-1, eps=1e-6)
        x, y, z = bearing.unbind(dim=-1)
        pieces: List[torch.Tensor] = []
        if self.use_xyz:
            pieces.append(torch.stack([x, y, z], dim=-1))
        if self.use_latlon_sincos:
            lon = torch.atan2(x, z)
            lat = torch.asin(y.clamp(-1.0, 1.0))
            pieces.append(torch.stack([torch.sin(lon), torch.cos(lon), torch.sin(lat), torch.cos(lat)], dim=-1))
        if self.use_erp_latitude_distortion:
            lat = torch.asin(y.clamp(-1.0, 1.0))
            cos_lat = torch.cos(lat).clamp_min(1.0e-3)
            pole_bias = 1.0 - cos_lat
            pieces.append(torch.stack([cos_lat, pole_bias, lat / 1.5707963267948966], dim=-1))
        return self.net(torch.cat(pieces, dim=-1))


class ImplicitSphericalCrossAttentionBlock(nn.Module):
    """Bidirectional latent association block over spherical tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
        pair_context_strength: float = 0.25,
    ) -> None:
        super().__init__()
        self.pair_context_strength = float(pair_context_strength)
        self.pos_enc = SphericalPositionalEncoding(dim)
        self.norm_q_a = nn.LayerNorm(dim)
        self.norm_q_b = nn.LayerNorm(dim)
        self.norm_ctx_a = nn.LayerNorm(dim)
        self.norm_ctx_b = nn.LayerNorm(dim)
        self.cross_attn_ab = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_ba = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm_ffn_a = nn.LayerNorm(dim)
        self.norm_ffn_b = nn.LayerNorm(dim)
        self.ffn_a = MLP(dim, int(dim * mlp_ratio), dropout=dropout)
        self.ffn_b = MLP(dim, int(dim * mlp_ratio), dropout=dropout)
        self.context_mlp = nn.Sequential(
            nn.LayerNorm(4 * dim),
            nn.Linear(4 * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(4 * dim),
            nn.Linear(4 * dim, dim),
            nn.Sigmoid(),
        )

    @staticmethod
    def _attention_stats(weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        # weights: [B, H, N, N]
        probs = weights.float().clamp_min(1.0e-9)
        entropy = -(probs * probs.log()).sum(dim=-1)
        entropy = entropy / max(float(torch.log(torch.tensor(float(weights.shape[-1]))).item()), 1.0e-6)
        return {
            "entropy_mean": entropy.mean(dim=(1, 2)),
            "max_mean": probs.max(dim=-1).values.mean(dim=(1, 2)),
            "concentration_mean": (probs.square().sum(dim=-1)).mean(dim=(1, 2)),
        }

    def forward(
        self,
        tokens_a: torch.Tensor,
        tokens_b: torch.Tensor,
        bearing_a: torch.Tensor,
        bearing_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        pos_a = self.pos_enc(bearing_a)
        pos_b = self.pos_enc(bearing_b)
        q_a = self.norm_q_a(tokens_a + pos_a)
        q_b = self.norm_q_b(tokens_b + pos_b)
        ctx_a = self.norm_ctx_a(tokens_a + pos_a)
        ctx_b = self.norm_ctx_b(tokens_b + pos_b)

        ab_ctx, w_ab = self.cross_attn_ab(q_a, ctx_b, ctx_b, need_weights=True, average_attn_weights=False)
        ba_ctx, w_ba = self.cross_attn_ba(q_b, ctx_a, ctx_a, need_weights=True, average_attn_weights=False)

        enhanced_a = tokens_a + self.drop(ab_ctx)
        enhanced_b = tokens_b + self.drop(ba_ctx)
        enhanced_a = enhanced_a + self.drop(self.ffn_a(self.norm_ffn_a(enhanced_a)))
        enhanced_b = enhanced_b + self.drop(self.ffn_b(self.norm_ffn_b(enhanced_b)))

        pooled_a = enhanced_a.mean(dim=1)
        pooled_b = enhanced_b.mean(dim=1)
        pooled_ab = ab_ctx.mean(dim=1)
        pooled_ba = ba_ctx.mean(dim=1)
        context_input = torch.cat([pooled_a, pooled_b, pooled_ab, pooled_ba], dim=-1)
        pair_context = self.context_mlp(context_input)
        relation_gate = self.gate(context_input)
        enhanced_a = enhanced_a + self.pair_context_strength * relation_gate.unsqueeze(1) * pair_context.unsqueeze(1)
        enhanced_b = enhanced_b + self.pair_context_strength * relation_gate.unsqueeze(1) * pair_context.unsqueeze(1)

        stats_ab = self._attention_stats(w_ab)
        stats_ba = self._attention_stats(w_ba)
        diagnostics = {
            "pair_context": pair_context,
            "relation_gate_mean": relation_gate.mean(dim=-1),
            "attention_affinity_entropy_ab": stats_ab["entropy_mean"],
            "attention_affinity_entropy_ba": stats_ba["entropy_mean"],
            "attention_affinity_max_ab": stats_ab["max_mean"],
            "attention_affinity_max_ba": stats_ba["max_mean"],
            "attention_affinity_concentration_ab": stats_ab["concentration_mean"],
            "attention_affinity_concentration_ba": stats_ba["concentration_mean"],
        }
        return enhanced_a, enhanced_b, diagnostics


class ImplicitSphericalCrossAttentionEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
        pair_context_strength: float = 0.25,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ImplicitSphericalCrossAttentionBlock(
                    dim,
                    n_heads,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    pair_context_strength=pair_context_strength,
                )
                for _ in range(max(1, int(n_layers)))
            ]
        )
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)

    def forward(
        self,
        tokens_a: torch.Tensor,
        tokens_b: torch.Tensor,
        bearing_a: torch.Tensor,
        bearing_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        layer_stats: List[Dict[str, torch.Tensor]] = []
        for block in self.blocks:
            tokens_a, tokens_b, stats = block(tokens_a, tokens_b, bearing_a, bearing_b)
            layer_stats.append(stats)
        merged: Dict[str, torch.Tensor] = {}
        for key in layer_stats[0].keys():
            values = [x[key] for x in layer_stats]
            merged[key] = torch.stack(values, dim=0).mean(dim=0)
        merged["num_attention_layers"] = torch.tensor(float(len(self.blocks)), device=tokens_a.device)
        return self.norm_a(tokens_a), self.norm_b(tokens_b), merged


class Struct360AModule2Sampler(Module2Sampler):
    def __init__(self, cfg: Config, device: torch.device):
        super().__init__(cfg, device=device)
        self.use_cross_context = False
        self.cross_c = None
        self.struct360a_cross = ImplicitSphericalCrossAttentionEncoder(
            cfg.D,
            n_layers=int(getattr(cfg, "struct360a_attention_layers", 1)),
            n_heads=int(getattr(cfg, "struct360a_attention_heads", getattr(cfg, "n_heads", 4))),
            dropout=float(getattr(cfg, "struct360a_attention_dropout", getattr(cfg, "dropout", 0.0))),
            mlp_ratio=float(getattr(cfg, "struct360a_attention_mlp_ratio", 2.0)),
            pair_context_strength=float(getattr(cfg, "struct360a_pair_context_strength", 0.25)),
        )
        self.struct360a_translation_cross = None
        if self.use_translation_feature_branch and bool(getattr(cfg, "struct360a_share_cross_for_translation_branch", True)) is False:
            self.struct360a_translation_cross = ImplicitSphericalCrossAttentionEncoder(
                cfg.D,
                n_layers=int(getattr(cfg, "struct360a_attention_layers", 1)),
                n_heads=int(getattr(cfg, "struct360a_attention_heads", getattr(cfg, "n_heads", 4))),
                dropout=float(getattr(cfg, "struct360a_attention_dropout", getattr(cfg, "dropout", 0.0))),
                mlp_ratio=float(getattr(cfg, "struct360a_attention_mlp_ratio", 2.0)),
                pair_context_strength=float(getattr(cfg, "struct360a_pair_context_strength", 0.25)),
            )
        self.last_relation_stats: Dict[str, torch.Tensor] = {}

    def forward(self, IA: torch.Tensor, IB: torch.Tensor) -> Dict[str, Tokens]:
        TokA_c = self._make_tokens(IA, "coarse")
        TokB_c = self._make_tokens(IB, "coarse")
        feat_a, feat_b, stats = self.struct360a_cross(TokA_c.feat, TokB_c.feat, TokA_c.bearing, TokB_c.bearing)
        TokA_c = Tokens(feat=feat_a, bearing=TokA_c.bearing, level=TokA_c.level, id=TokA_c.id, parent_id=TokA_c.parent_id)
        TokB_c = Tokens(feat=feat_b, bearing=TokB_c.bearing, level=TokB_c.level, id=TokB_c.id, parent_id=TokB_c.parent_id)
        self.last_relation_stats = dict(stats)
        out = {"TokA_c": TokA_c, "TokB_c": TokB_c}
        if bool(self.cfg.use_fine_stage):
            out["TokA_f"] = self._make_tokens(IA, "fine")
            out["TokB_f"] = self._make_tokens(IB, "fine")
        if self.use_translation_feature_branch:
            tok_a_t = self._make_tokens(IA, "coarse", branch="translation")
            tok_b_t = self._make_tokens(IB, "coarse", branch="translation")
            if self.struct360a_translation_cross is not None:
                feat_a_t, feat_b_t, _ = self.struct360a_translation_cross(
                    tok_a_t.feat,
                    tok_b_t.feat,
                    tok_a_t.bearing,
                    tok_b_t.bearing,
                )
                tok_a_t = Tokens(feat=feat_a_t, bearing=tok_a_t.bearing, level=tok_a_t.level, id=tok_a_t.id, parent_id=tok_a_t.parent_id)
                tok_b_t = Tokens(feat=feat_b_t, bearing=tok_b_t.bearing, level=tok_b_t.level, id=tok_b_t.id, parent_id=tok_b_t.parent_id)
            out["TokA_c_t"] = tok_a_t
            out["TokB_c_t"] = tok_b_t
        return out


class STRUCT360AImplicitSphericalAttentionModel(PanoramaRelPoseModel):
    def __init__(self, cfg: Config, device: torch.device):
        super().__init__(cfg, device)
        self.module2 = Struct360AModule2Sampler(cfg, device=device)

    def forward(self, *args, **kwargs):
        R, t_dir, aux = super().forward(*args, **kwargs)
        if getattr(self.module2, "last_relation_stats", None):
            for key, value in self.module2.last_relation_stats.items():
                aux[f"struct360a_{key}"] = value
        return R, t_dir, aux


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
