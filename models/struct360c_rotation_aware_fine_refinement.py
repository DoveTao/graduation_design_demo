from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from config import Config
from models.struct360b_match_free_coarse_to_fine import STRUCT360BMatchFreeCoarseToFineModel


class RotationAwareSphericalBias(nn.Module):
    """Coarse-R conditioned latent spherical attention bias.

    This module computes angular compatibility between rotated A-sphere token
    bearings and B-sphere token bearings. It is a latent bias tensor only; it
    does not select matches, does not produce top-k neighbors, and does not
    output correspondences.
    """

    def __init__(self, temperature: float = 0.35, clamp_abs: float = 8.0) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.clamp_abs = float(clamp_abs)

    def forward(self, bearing_a: torch.Tensor, bearing_b: torch.Tensor, coarse_R_ba: torch.Tensor) -> torch.Tensor:
        a = torch.nn.functional.normalize(bearing_a.float(), dim=-1, eps=1.0e-6)
        b = torch.nn.functional.normalize(bearing_b.float(), dim=-1, eps=1.0e-6)
        rotated_a = torch.matmul(coarse_R_ba.float(), a.transpose(1, 2)).transpose(1, 2)
        cosine = torch.matmul(rotated_a, b.transpose(1, 2)).clamp(-1.0, 1.0)
        angular = torch.acos(cosine)
        bias = -angular / max(self.temperature, 1.0e-6)
        return bias.clamp(min=-self.clamp_abs, max=self.clamp_abs)


class STRUCT360CRotationAwareFineRefinementModel(STRUCT360BMatchFreeCoarseToFineModel):
    """STRUCT360B challenger with coarse-rotation-aware latent bias diagnostics."""

    def __init__(self, cfg: Config, device: torch.device):
        super().__init__(cfg, device)
        self.struct360c_rotation_bias = RotationAwareSphericalBias(
            temperature=float(getattr(cfg, "struct360c_bias_temperature", 0.35)),
            clamp_abs=float(getattr(cfg, "struct360c_bias_clamp_abs", 8.0)),
        )

    def forward(
        self,
        IA: torch.Tensor,
        IB: torch.Tensor,
        *,
        enable_depth_fusion: Optional[bool] = None,
        dt_world: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        R, t, aux = super().forward(IA, IB, enable_depth_fusion=enable_depth_fusion, dt_world=dt_world)
        bias = self.struct360c_rotation_bias(aux["bearingA_f"], aux["bearingB_f"], aux["coarse_R"])
        aux["struct360c_rotation_attention_bias_mean"] = bias.mean()
        aux["struct360c_rotation_attention_bias_std"] = bias.std(unbiased=False)
        aux["struct360c_rotation_attention_bias_min"] = bias.min()
        aux["struct360c_rotation_attention_bias_max"] = bias.max()
        aux["struct360c_attention_bias_is_latent_only"] = torch.tensor(True, device=R.device)
        aux["struct360c_outputs_correspondences"] = torch.tensor(False, device=R.device)
        aux["stage"] = "rotation_aware_match_free_fine_refinement_prepared"
        return R, t, aux
