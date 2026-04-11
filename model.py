from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from config import Config
from erp_sampling import build_level_grid_from_patch_bearings, sample_patches_erp
from healpix_utils import HealpixHierarchy
from interaction import (
    CoarseInteraction,
    FineInteraction,
    PairPoseHead,
    Tokens,
    aggregate_fine_to_coarse,
)
from transformer_encoder import BearingPosEnc, PatchEmbed, TokenEncoder


class Module2Sampler(nn.Module):
    """
    Week-4 style spherical feature extraction / encoding:
      ERP -> sphere-aware patch sampling -> token embed -> positional encoding -> transformer encoder.
    """

    def __init__(self, cfg: Config, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.hier = HealpixHierarchy(cfg.Nc, cfg.Nf, cfg.p, device=device)

        level_data = self.hier.level_data()
        grid_c = build_level_grid_from_patch_bearings(level_data["coarse"]["patch_bearing"], cfg.H, cfg.W)
        grid_f = build_level_grid_from_patch_bearings(level_data["fine"]["patch_bearing"], cfg.H, cfg.W)
        self.register_buffer("grid_coarse", grid_c, persistent=True)
        self.register_buffer("grid_fine", grid_f, persistent=True)

        self.patch_embed_c = PatchEmbed(cfg.p, cfg.D, in_ch=cfg.in_ch)
        self.patch_embed_f = PatchEmbed(cfg.p, cfg.D, in_ch=cfg.in_ch)
        self.pos_enc = BearingPosEnc(cfg.D)
        self.enc_c = TokenEncoder(cfg.D, cfg.n_layers, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
        self.enc_f = TokenEncoder(cfg.D, cfg.n_layers, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)

    def _make_tokens(self, img: torch.Tensor, level: str) -> Tokens:
        level_data = self.hier.level_data()
        B = img.shape[0]

        if level == "coarse":
            N = self.cfg.Nc
            grid = self.grid_coarse
            bearing = level_data["coarse"]["bearing"]
            ids = level_data["coarse"]["id"]
            parent = None
            patches = sample_patches_erp(img, grid, N=N, p=self.cfg.p)
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
        feat = feat + self.pos_enc(bearing_b)
        feat = enc(feat)

        ids_b = ids.view(1, N).expand(B, -1)
        parent_b = None if parent is None else parent.view(1, N).expand(B, -1)
        return Tokens(feat=feat, bearing=bearing_b, level=level, id=ids_b, parent_id=parent_b)

    def forward(self, IA: torch.Tensor, IB: torch.Tensor) -> Dict[str, Tokens]:
        return {
            "TokA_c": self._make_tokens(IA, "coarse"),
            "TokB_c": self._make_tokens(IB, "coarse"),
            "TokA_f": self._make_tokens(IA, "fine"),
            "TokB_f": self._make_tokens(IB, "fine"),
        }


def _local_t_to_output_frame(R: torch.Tensor, t_local: torch.Tensor) -> torch.Tensor:
    """
    Heads regress translation in frame A (feature-anchor frame), using the same
    baseline direction as the dataset target (B->A).

    Dataset convention:
      - R = R_{B<-A}
      - t_gt is the B->A baseline expressed in frame B

    Therefore the same baseline expressed in frame A is:
      t_local_A = R^T t_gt

    Mapping that local A-frame direction back to the output B-frame uses R:
      t_out_B = R t_local_A
    """
    t_local = nn.functional.normalize(t_local.float(), dim=-1, eps=1e-6)
    t_out = torch.matmul(R.float(), t_local.unsqueeze(-1)).squeeze(-1)
    return nn.functional.normalize(t_out, dim=-1, eps=1e-6)


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
        )
        self.fine = FineInteraction(
            cfg.D,
            temperature=cfg.fine_temperature,
            logits_clip=cfg.logits_clip,
        )

    def forward(self, IA: torch.Tensor, IB: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        tokens = self.module2(IA, IB)
        TokA_c = tokens["TokA_c"]
        TokB_c = tokens["TokB_c"]
        TokA_f = tokens["TokA_f"]
        TokB_f = tokens["TokB_f"]

        aux: Dict[str, torch.Tensor] = {
            "bearingA_c": TokA_c.bearing,
            "bearingB_c": TokB_c.bearing,
            "bearingA_f": TokA_f.bearing,
            "bearingB_f": TokB_f.bearing,
            "t_local_frame": self.cfg.translation_local_frame,
            "t_output_frame": self.cfg.translation_output_frame,
        }

        # Ablation: encoder only + regression, no cross-image interaction.
        if not self.cfg.use_coarse_interaction:
            R, t_local = self.direct_head(TokA_c.feat, TokB_c.feat)
            t_dir = _local_t_to_output_frame(R, t_local)
            aux["t_dir_local"] = t_local
            aux["stage"] = "encoder_only"
            return R, t_dir, aux

        # Week-5 core chain: encoding -> coarse interaction -> coarse pose
        out_c = self.coarse(TokA_c, TokB_c)
        aux.update(out_c)
        aux["tc_dir_local"] = out_c["tc_dir"]
        aux["tc_dir"] = _local_t_to_output_frame(out_c["Rc"], out_c["tc_dir"])
        aux["stage"] = "coarse_only"

        if not self.cfg.use_fine_stage:
            return out_c["Rc"], aux["tc_dir"], aux

        # Routed fine stage with translation-aware epipolar prior.
        # Important: detach coarse geometric guidance before feeding it into the
        # fine routing prior. This keeps the prior useful at inference time but
        # prevents unstable self-reinforcement when coarse translation flips.
        out_f = self.fine(
            TokA_f=TokA_f,
            TokB_f=TokB_f,
            Wc_ab=out_c["Wc_ab"].detach(),
            Rc=out_c["Rc"].detach(),
            tc_dir=aux["tc_dir"].detach(),
            topk_coarse=self.cfg.topk_coarse,
            use_epipolar_bias=self.cfg.use_epipolar_bias,
            epi_angle_thresh_deg=self.cfg.epi_angle_thresh_deg,
            epi_bias_strength=self.cfg.epi_bias_strength,
            epi_mode=self.cfg.epi_mode,
        )
        aux.update(out_f)
        aux["t_dir_local"] = out_f["t_dir"]
        aux["t_dir"] = _local_t_to_output_frame(out_f["R"], out_f["t_dir"])
        aux["stage"] = "coarse_to_fine"
        aux["Wc_tilde"] = aggregate_fine_to_coarse(
            Wf_ab=out_f["Wf_ab"],
            parent_a=TokA_f.parent_id,
            parent_b=TokB_f.parent_id,
            NcA=self.cfg.Nc,
            NcB=self.cfg.Nc,
        )
        return out_f["R"], aux["t_dir"], aux
