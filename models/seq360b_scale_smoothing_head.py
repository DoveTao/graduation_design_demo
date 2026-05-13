from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq360BScaleSmoothingHead(nn.Module):
    """Lightweight sequence-context log-scale smoother.

    The head consumes per-adjacent-pair context and log_tmag values, then emits a
    bounded additive correction. It does not alter rotation or translation
    direction and does not create explicit correspondences.
    """

    def __init__(self, context_dim: int, hidden_dim: int = 128, delta_clamp: float = 0.3) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.delta_clamp = float(delta_clamp)
        in_dim = self.context_dim + 4
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        pair_context: torch.Tensor,
        log_tmag: torch.Tensor,
        *,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if pair_context.ndim != 3:
            raise ValueError("pair_context must be [B, A, C]")
        if log_tmag.ndim != 2:
            raise ValueError("log_tmag must be [B, A]")
        B, A, _ = pair_context.shape
        mask = torch.ones(B, A, dtype=torch.bool, device=pair_context.device) if valid_mask is None else valid_mask.bool()
        denom = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_log = (log_tmag * mask.float()).sum(dim=1, keepdim=True) / denom
        centered = log_tmag - mean_log
        left = torch.zeros_like(centered)
        right = torch.zeros_like(centered)
        if A > 1:
            left[:, 1:] = centered[:, :-1]
            right[:, :-1] = centered[:, 1:]
        pos = torch.linspace(-1.0, 1.0, A, device=pair_context.device, dtype=pair_context.dtype).view(1, A).expand(B, A)
        features = torch.cat(
            [
                pair_context.float(),
                log_tmag.float().unsqueeze(-1),
                centered.float().unsqueeze(-1),
                (centered - left).float().unsqueeze(-1),
                (right - centered).float().unsqueeze(-1),
            ],
            dim=-1,
        )
        raw_delta = self.net(features).squeeze(-1)
        delta = self.delta_clamp * torch.tanh(raw_delta)
        delta = delta * mask.float()
        corrected_log_tmag = log_tmag.float() + delta
        return {
            "corrected_log_tmag": corrected_log_tmag,
            "corrected_tmag": torch.exp(corrected_log_tmag.clamp(-6.0, 6.0)),
            "delta_log_tmag": delta,
            "delta_log_tmag_abs_mean": torch.abs(delta[mask]).mean() if mask.any() else delta.sum() * 0.0,
            "valid_mask": mask,
            "position": pos,
        }


def seq360b_scale_losses(
    corrected_log_tmag: torch.Tensor,
    gt_tmag: torch.Tensor,
    *,
    delta_log_tmag: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1.0e-6,
    log_weight: float = 1.0,
    path_weight: float = 0.5,
    smooth_weight: float = 0.1,
    delta_weight: float = 0.05,
) -> Dict[str, torch.Tensor]:
    mask = torch.ones_like(corrected_log_tmag, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    pred_tmag = torch.exp(corrected_log_tmag.clamp(-6.0, 6.0)).clamp_min(eps)
    gt_tmag = gt_tmag.float().clamp_min(eps)
    log_gt = torch.log(gt_tmag)
    per_pair = F.smooth_l1_loss(corrected_log_tmag[mask], log_gt[mask]) if mask.any() else corrected_log_tmag.sum() * 0.0
    pred_path = (pred_tmag * mask.float()).sum(dim=1)
    gt_path = (gt_tmag * mask.float()).sum(dim=1)
    path_ratio_loss = torch.abs(torch.log((pred_path + eps) / (gt_path + eps))).mean()
    if delta_log_tmag is None:
        delta_log_tmag = corrected_log_tmag.sum() * 0.0 + torch.zeros_like(corrected_log_tmag)
    if corrected_log_tmag.shape[1] > 1:
        smooth_loss = torch.abs(delta_log_tmag[:, 1:] - delta_log_tmag[:, :-1]).mean()
    else:
        smooth_loss = corrected_log_tmag.sum() * 0.0
    delta_loss = torch.abs(delta_log_tmag[mask]).mean() if mask.any() else corrected_log_tmag.sum() * 0.0
    return {
        "log_tmag_loss": per_pair,
        "path_ratio_loss": path_ratio_loss,
        "scale_smoothness_loss": smooth_loss,
        "delta_regularization_loss": delta_loss,
        "total": float(log_weight) * per_pair
        + float(path_weight) * path_ratio_loss
        + float(smooth_weight) * smooth_loss
        + float(delta_weight) * delta_loss,
    }
