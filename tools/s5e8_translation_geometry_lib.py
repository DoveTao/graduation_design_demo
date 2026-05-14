#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn

from s5e7_direction_scale_lib import (
    build_eval_seq_frames,
    build_numeric_features,
    load_npz_model,
    load_ordered_image_pair,
    make_pair_samples_with_buckets,
    predict_s5e2,
    predict_s5e3_head,
    rotvec_to_matrix,
)


class DirectionOnlyCandidate(nn.Module):
    def __init__(self, numeric_dim: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.numeric = nn.Sequential(
            nn.Linear(numeric_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.shared = nn.Sequential(
            nn.Linear(48 + 64, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
        )
        self.dir_head = nn.Linear(64, 3)
        self.sign_head = nn.Linear(64, 1)
        self.conf_head = nn.Linear(64, 1)
        self.bucket_head = nn.Linear(64, 3)

    def forward(self, image_pair: torch.Tensor, numeric_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        visual = self.cnn(image_pair).flatten(1)
        numeric = self.numeric(numeric_feats)
        fused = self.shared(torch.cat([visual, numeric], dim=1))
        return {
            "dir_delta": self.dir_head(fused),
            "sign_score": torch.tanh(self.sign_head(fused)),
            "confidence": torch.sigmoid(self.conf_head(fused)),
            "motion_bucket_logits": self.bucket_head(fused),
        }


class NumericOnlyDirectionModel(nn.Module):
    def __init__(self, numeric_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(numeric_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.dir_head = nn.Linear(32, 3)
        self.sign_head = nn.Linear(32, 1)
        self.bucket_head = nn.Linear(32, 3)

    def forward(self, numeric_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.net(numeric_feats)
        return {
            "dir_delta": self.dir_head(feat),
            "sign_score": torch.tanh(self.sign_head(feat)),
            "motion_bucket_logits": self.bucket_head(feat),
        }


def compose_direction(prior_dir: torch.Tensor, out: Dict[str, torch.Tensor]) -> torch.Tensor:
    raw = prior_dir + out["dir_delta"] + prior_dir * out["sign_score"]
    return raw / raw.norm(dim=1, keepdim=True).clamp_min(1.0e-6)


def save_direction_checkpoint(path: Path, model: nn.Module, image_size: Tuple[int, int], numeric_dim: int, bucket_stats: Dict[str, Any], model_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_size": list(image_size),
            "numeric_dim": int(numeric_dim),
            "bucket_stats": bucket_stats,
            "model_type": model_type,
        },
        path,
    )


def load_direction_checkpoint(path: Path, device: torch.device) -> Tuple[nn.Module, Tuple[int, int], int, Dict[str, Any], str]:
    payload = torch.load(path, map_location=device)
    image_size = tuple(int(x) for x in payload["image_size"])
    numeric_dim = int(payload["numeric_dim"])
    model_type = payload.get("model_type", "direction_only_candidate")
    if model_type == "numeric_only_direction":
        model = NumericOnlyDirectionModel(numeric_dim=numeric_dim)
    else:
        model = DirectionOnlyCandidate(numeric_dim=numeric_dim)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, image_size, numeric_dim, payload["bucket_stats"], model_type
