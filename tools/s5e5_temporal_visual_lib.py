#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch import nn

from s5e2_adjacent_dense_lib import pair_features, scan_frames, split_sequence_keys, adjacent_pairs, relative_pose_A_to_B_in_B


@dataclass
class PairSample:
    frame_i: Any
    frame_j: Any
    edge_index: int
    total_edges: int
    gt_direction: np.ndarray
    gt_logmag: float
    gt_magnitude: float
    gt_rotvec: np.ndarray


def load_npz_model(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def predict_s5e2(model: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    return ((x - model["x_mean"]) / model["x_std"]) @ model["W"] * model["y_std"] + model["y_mean"]


def predict_s5e3_head(model: Dict[str, np.ndarray], prefix: str, x: np.ndarray) -> np.ndarray:
    return ((x - model[f"{prefix}_x_mean"]) / model[f"{prefix}_x_std"]) @ model[f"{prefix}_W"] * model[f"{prefix}_y_std"] + model[f"{prefix}_y_mean"]


def load_ordered_image_pair(frame_i: Any, frame_j: Any, size: Tuple[int, int]) -> torch.Tensor:
    h, w = size
    img_i = Image.open(frame_i.image_path).convert("RGB").resize((w, h), resample=Image.BILINEAR)
    img_j = Image.open(frame_j.image_path).convert("RGB").resize((w, h), resample=Image.BILINEAR)
    arr_i = np.asarray(img_i, dtype=np.float32) / 255.0
    arr_j = np.asarray(img_j, dtype=np.float32) / 255.0
    diff = arr_j - arr_i
    abs_diff = np.abs(diff)
    stacked = np.concatenate([arr_i, arr_j, diff, abs_diff], axis=2)
    stacked = np.transpose(stacked, (2, 0, 1))
    return torch.from_numpy(stacked)


def make_pair_samples(train_only: bool = True) -> Tuple[List[PairSample], List[PairSample], Dict[str, Any]]:
    frames = scan_frames(Path("data"))
    train_keys, test_keys, split_summary = split_sequence_keys(list(frames.keys()))
    train_pairs: List[Tuple[Any, Any]] = []
    for key in train_keys:
        train_pairs.extend(adjacent_pairs(frames.get(key, [])))
    val_count = max(1, int(round(len(train_pairs) * 0.1))) if train_pairs else 0
    core_train, core_val = train_pairs[:-val_count], train_pairs[-val_count:]

    def _pack(pairs: Sequence[Tuple[Any, Any]]) -> List[PairSample]:
        out: List[PairSample] = []
        total = max(len(pairs), 1)
        for idx, (a, b) in enumerate(pairs):
            R, t = relative_pose_A_to_B_in_B(a, b)
            mag = float(np.linalg.norm(t))
            direction = t / max(mag, 1.0e-12)
            rotvec = rotation_matrix_to_rotvec(R)
            out.append(PairSample(a, b, idx, total, direction.astype(np.float32), float(np.log(max(mag, 1.0e-12))), mag, rotvec.astype(np.float32)))
        return out

    return _pack(core_train), _pack(core_val), split_summary


def rotation_matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    cos_theta = (float(np.trace(R)) - 1.0) / 2.0
    theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    if theta < 1.0e-10:
        return np.zeros(3, dtype=np.float64)
    v = np.asarray([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    return theta / (2.0 * np.sin(theta)) * v


def rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(v))
    if theta < 1.0e-10:
        return np.eye(3, dtype=np.float64)
    k = v / theta
    K = np.asarray([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


class TemporalVisualBackboneCandidate(nn.Module):
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
        self.mag_head = nn.Linear(64, 1)

    def forward(self, image_pair: torch.Tensor, numeric_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        visual = self.cnn(image_pair).flatten(1)
        numeric = self.numeric(numeric_feats)
        fused = self.shared(torch.cat([visual, numeric], dim=1))
        dir_raw = self.dir_head(fused)
        sign_logit = self.sign_head(fused)
        logmag_residual = self.mag_head(fused)
        return {"dir_raw": dir_raw, "sign_logit": sign_logit, "logmag_residual": logmag_residual}


def build_numeric_features(sample: PairSample, cache: Dict[str, np.ndarray], s5e2: Dict[str, np.ndarray], s5e3: Dict[str, np.ndarray]) -> np.ndarray:
    x = pair_features(sample.frame_i, sample.frame_j, sample.edge_index, sample.total_edges, cache)
    prior = predict_s5e2(s5e2, x)
    prior_dir = prior[3:6]
    prior_dir = prior_dir / max(float(np.linalg.norm(prior_dir)), 1.0e-12)
    prior_logmag = float(predict_s5e3_head(s5e3, "mag", x).reshape(-1)[0])
    dt = float(sample.frame_j.timestamp - sample.frame_i.timestamp)
    ratio_feats = x[:10] if x.size >= 10 else np.pad(x, (0, max(0, 10 - x.size)))
    return np.concatenate(
        [
            prior_dir.astype(np.float32),
            np.asarray([prior_logmag, dt, sample.edge_index / max(sample.total_edges, 1)], dtype=np.float32),
            ratio_feats.astype(np.float32),
        ]
    )


def build_eval_seq_frames(scene: str, seq: str) -> List[Any]:
    return scan_frames(Path("data"), scene=scene, seq=seq).get((scene, seq), [])


def save_training_checkpoint(
    path: Path,
    model: TemporalVisualBackboneCandidate,
    image_size: Tuple[int, int],
    numeric_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_size": list(image_size),
            "numeric_dim": numeric_dim,
            "model_type": "temporal_visual_backbone",
            "visual_backbone_used": True,
        },
        path,
    )


def load_training_checkpoint(path: Path, device: torch.device) -> Tuple[TemporalVisualBackboneCandidate, Tuple[int, int], int]:
    payload = torch.load(path, map_location=device)
    image_size = tuple(int(x) for x in payload["image_size"])
    numeric_dim = int(payload["numeric_dim"])
    model = TemporalVisualBackboneCandidate(numeric_dim=numeric_dim)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, image_size, numeric_dim
