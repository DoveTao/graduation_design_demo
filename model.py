# model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from config import Config
from healpix_utils import HealpixHierarchy
from erp_sampling import build_level_grid_from_patch_bearings, sample_patches_erp
from transformer_encoder import PatchEmbed, BearingPosEnc, TokenEncoder
from interaction import Tokens, CoarseInteraction, FineInteraction, aggregate_fine_to_coarse


class Module2Sampler(nn.Module):
    """
    Module 2: HEALPix cells -> ERP resampling patch -> ViT tokens
    """
    def __init__(self, cfg: Config, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.hier = HealpixHierarchy(cfg.Nc, cfg.Nf, cfg.p, device=device)

        # Prebuild grids for coarse/fine patches (buffers)
        ld = self.hier.level_data()
        grid_c = build_level_grid_from_patch_bearings(ld["coarse"]["patch_bearing"], cfg.H, cfg.W)  # [1,Nc*p,p,2]
        grid_f = build_level_grid_from_patch_bearings(ld["fine"]["patch_bearing"], cfg.H, cfg.W)    # [1,Nf*p,p,2]
        self.register_buffer("grid_coarse", grid_c, persistent=True)
        self.register_buffer("grid_fine", grid_f, persistent=True)

        # Embedding + encoder per level (can be shared; kept separate for clarity)
        self.patch_embed_c = PatchEmbed(cfg.p, cfg.D, in_ch=cfg.in_ch)
        self.patch_embed_f = PatchEmbed(cfg.p, cfg.D, in_ch=cfg.in_ch)
        self.pos_enc = BearingPosEnc(cfg.D)

        self.enc_c = TokenEncoder(cfg.D, cfg.n_layers, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
        self.enc_f = TokenEncoder(cfg.D, cfg.n_layers, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)

    def _make_tokens(
        self,
        img: torch.Tensor,        # [B,3,H,W]
        level: str,
    ) -> Tokens:
        cfg = self.cfg
        ld = self.hier.level_data()

        if level == "coarse":
            N = cfg.Nc
            grid = self.grid_coarse
            bearing = ld["coarse"]["bearing"]  # [Nc,3]
            ids = ld["coarse"]["id"]           # [Nc]
            parent = None
            patch = sample_patches_erp(img, grid, N=N, p=cfg.p)  # [B,N,3,p,p]
            feat0 = self.patch_embed_c(patch)                    # [B,N,D]
            b = bearing.view(1, N, 3).expand(img.size(0), -1, -1)
            feat = feat0 + self.pos_enc(b)                       # [B,N,D]
            feat = self.enc_c(feat)                              # [B,N,D]
        else:
            N = cfg.Nf
            grid = self.grid_fine
            bearing = ld["fine"]["bearing"]      # [Nf,3]
            ids = ld["fine"]["id"]               # [Nf]
            parent = ld["fine"]["parent_id"]     # [Nf]
            patch = sample_patches_erp(img, grid, N=N, p=cfg.p)  # [B,N,3,p,p]
            feat0 = self.patch_embed_f(patch)                    # [B,N,D]
            b = bearing.view(1, N, 3).expand(img.size(0), -1, -1)
            feat = feat0 + self.pos_enc(b)                       # [B,N,D]
            feat = self.enc_f(feat)                              # [B,N,D]

        idB = ids.view(1, N).expand(img.size(0), -1)
        if parent is None:
            parentB = None
        else:
            parentB = parent.view(1, N).expand(img.size(0), -1)

        return Tokens(
            feat=feat,
            bearing=b,
            level=level,
            id=idB,
            parent_id=parentB,
        )

    def forward(self, IA: torch.Tensor, IB: torch.Tensor) -> Dict[str, Tokens]:
        TokA_c = self._make_tokens(IA, "coarse")
        TokB_c = self._make_tokens(IB, "coarse")
        TokA_f = self._make_tokens(IA, "fine")
        TokB_f = self._make_tokens(IB, "fine")
        return {
            "TokA_c": TokA_c,
            "TokB_c": TokB_c,
            "TokA_f": TokA_f,
            "TokB_f": TokB_f,
        }


class PanoramaRelPoseModel(nn.Module):
    """
    End-to-end MVP model:
      Module2 -> Module3 coarse -> routing+epi -> Module3 fine -> Module4 pose decoder
      (included in fine head)
    """
    def __init__(self, cfg: Config, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.module2 = Module2Sampler(cfg, device=device)
        self.coarse = CoarseInteraction(cfg.D)
        self.fine = FineInteraction(cfg.D)

    def forward(self, IA: torch.Tensor, IB: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
          R: [B,3,3]
          t_dir: [B,3]
          aux: dict with Wc/Wf/Fc/Ff/masks/confidences/etc.
        """
        tokens = self.module2(IA, IB)
        TokA_c, TokB_c = tokens["TokA_c"], tokens["TokB_c"]
        TokA_f, TokB_f = tokens["TokA_f"], tokens["TokB_f"]

        # Module3 coarse
        out_c = self.coarse(TokA_c, TokB_c)
        Wc_ab = out_c["Wc_ab"]
        Rc = out_c["Rc"]
        tc_dir = out_c["tc_dir"]

        # Module3 fine (routing + strict spherical epipolar band + sparse matching)
        out_f = self.fine(
            TokA_f=TokA_f,
            TokB_f=TokB_f,
            Wc_ab=Wc_ab,
            Rc=Rc,
            tc_dir=tc_dir,
            topk_coarse=self.cfg.topk_coarse,
            epi_angle_thresh_deg=self.cfg.epi_angle_thresh_deg,
            epi_bias_strength=self.cfg.epi_bias_strength,
            epi_mode="bias",
        )

        # Cross-level consistency aggregation
        Wc_tilde = aggregate_fine_to_coarse(
            Wf_ab=out_f["Wf_ab"],
            parent_a=TokA_f.parent_id,
            parent_b=TokB_f.parent_id,
            NcA=self.cfg.Nc,
            NcB=self.cfg.Nc,  # symmetrical here (A/B coarse both Nc)
        )

        aux = {}
        aux.update(out_c)
        aux.update(out_f)
        aux["Wc_tilde"] = Wc_tilde
        aux["bearingA_f"] = TokA_f.bearing   # [B,Nf,3]
        aux["bearingB_f"] = TokB_f.bearing   # [B,Nf,3]

        # Final prediction from fine head (Module4 in MVP)
        R = out_f["R"]
        t_dir = out_f["t_dir"]
        return R, t_dir, aux
