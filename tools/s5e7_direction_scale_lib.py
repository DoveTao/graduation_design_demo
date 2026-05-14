#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch import nn

from s5e2_adjacent_dense_lib import pair_features, scan_frames, split_sequence_keys, adjacent_pairs, relative_pose_A_to_B_in_B


@dataclass
class DirectionScaleSample:
    frame_i: Any
    frame_j: Any
    edge_index: int
    total_edges: int
    gt_direction: np.ndarray
    gt_logmag: float
    gt_magnitude: float
    gt_rotvec: np.ndarray
    bucket_name: str
    bucket_index: int


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


def compute_motion_bucket_stats(magnitudes: Sequence[float]) -> Dict[str, Any]:
    mags = np.asarray(list(magnitudes), dtype=np.float64)
    small_thr = float(np.percentile(mags, 25))
    large_thr = float(np.percentile(mags, 75))

    def _stats(sub: np.ndarray) -> Dict[str, float]:
        if sub.size == 0:
            return {"count": 0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "clip_lo": 1.0e-6, "clip_hi": 1.0e-6}
        return {
            "count": int(sub.size),
            "median": float(np.percentile(sub, 50)),
            "p90": float(np.percentile(sub, 90)),
            "p95": float(np.percentile(sub, 95)),
            "max": float(np.max(sub)),
            "clip_lo": float(max(np.percentile(sub, 5), 1.0e-6)),
            "clip_hi": float(np.percentile(sub, 95)),
        }

    small = mags[mags <= small_thr]
    large = mags[mags >= large_thr]
    normal = mags[(mags > small_thr) & (mags < large_thr)]
    return {
        "small_threshold": small_thr,
        "large_threshold": large_thr,
        "global": _stats(mags),
        "small_motion": _stats(small),
        "normal_motion": _stats(normal),
        "large_motion": _stats(large),
    }


def assign_motion_bucket(magnitude: float, stats: Dict[str, Any]) -> Tuple[str, int]:
    if magnitude <= stats["small_threshold"]:
        return "small_motion", 0
    if magnitude >= stats["large_threshold"]:
        return "large_motion", 2
    return "normal_motion", 1


def make_pair_samples_with_buckets() -> Tuple[List[DirectionScaleSample], List[DirectionScaleSample], Dict[str, Any], Dict[str, Any]]:
    frames = scan_frames(Path("data"))
    train_keys, test_keys, split_summary = split_sequence_keys(list(frames.keys()))
    train_pairs: List[Tuple[Any, Any]] = []
    for key in train_keys:
        train_pairs.extend(adjacent_pairs(frames.get(key, [])))
    val_count = max(1, int(round(len(train_pairs) * 0.1))) if train_pairs else 0
    core_train, core_val = train_pairs[:-val_count], train_pairs[-val_count:]

    def _basic(pairs: Sequence[Tuple[Any, Any]]) -> List[Tuple[Any, Any, np.ndarray, np.ndarray, float, np.ndarray]]:
        out = []
        for a, b in pairs:
            R, t = relative_pose_A_to_B_in_B(a, b)
            mag = float(np.linalg.norm(t))
            direction = t / max(mag, 1.0e-12)
            rotvec = rotation_matrix_to_rotvec(R)
            out.append((a, b, direction.astype(np.float32), t.astype(np.float32), mag, rotvec.astype(np.float32)))
        return out

    train_basic = _basic(core_train)
    val_basic = _basic(core_val)
    stats = compute_motion_bucket_stats([x[4] for x in train_basic + val_basic])

    def _pack(rows) -> List[DirectionScaleSample]:
        total = max(len(rows), 1)
        packed: List[DirectionScaleSample] = []
        for idx, (a, b, direction, _t, mag, rotvec) in enumerate(rows):
            bucket_name, bucket_index = assign_motion_bucket(mag, stats)
            packed.append(
                DirectionScaleSample(
                    frame_i=a,
                    frame_j=b,
                    edge_index=idx,
                    total_edges=total,
                    gt_direction=direction,
                    gt_logmag=float(np.log(max(mag, 1.0e-12))),
                    gt_magnitude=mag,
                    gt_rotvec=rotvec,
                    bucket_name=bucket_name,
                    bucket_index=bucket_index,
                )
            )
        return packed

    return _pack(train_basic), _pack(val_basic), split_summary, stats


def build_numeric_features(
    sample: DirectionScaleSample,
    cache: Dict[str, np.ndarray],
    s5e2: Dict[str, np.ndarray],
    s5e3: Dict[str, np.ndarray],
    bucket_stats: Dict[str, Any],
) -> np.ndarray:
    x = pair_features(sample.frame_i, sample.frame_j, sample.edge_index, sample.total_edges, cache)
    prior = predict_s5e2(s5e2, x)
    prior_dir = prior[3:6]
    prior_dir = prior_dir / max(float(np.linalg.norm(prior_dir)), 1.0e-12)
    prior_logmag = float(predict_s5e3_head(s5e3, "mag", x).reshape(-1)[0])
    dt = float(sample.frame_j.timestamp - sample.frame_i.timestamp)
    bucket_priors = bucket_stats[sample.bucket_name]
    ratio_feats = x[:12] if x.size >= 12 else np.pad(x, (0, max(0, 12 - x.size)))
    return np.concatenate(
        [
            prior_dir.astype(np.float32),
            np.asarray(
                [
                    prior_logmag,
                    dt,
                    sample.edge_index / max(sample.total_edges, 1),
                    float(bucket_priors["median"]),
                    float(bucket_priors["p95"]),
                    float(sample.bucket_index),
                ],
                dtype=np.float32,
            ),
            ratio_feats.astype(np.float32),
        ]
    )


class DirectionScaleCalibratedModel(nn.Module):
    def __init__(self, numeric_dim: int, log_scale_clip: float) -> None:
        super().__init__()
        self.log_scale_clip = float(log_scale_clip)
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
        self.scale_head = nn.Linear(64, 1)
        self.conf_head = nn.Linear(64, 1)
        self.bucket_head = nn.Linear(64, 3)

    def forward(self, image_pair: torch.Tensor, numeric_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        visual = self.cnn(image_pair).flatten(1)
        numeric = self.numeric(numeric_feats)
        fused = self.shared(torch.cat([visual, numeric], dim=1))
        dir_delta = self.dir_head(fused)
        sign_score = torch.tanh(self.sign_head(fused))
        scale_residual_raw = self.scale_head(fused)
        clamped_log_scale_delta = torch.tanh(scale_residual_raw) * self.log_scale_clip
        confidence = torch.sigmoid(self.conf_head(fused))
        motion_bucket_logits = self.bucket_head(fused)
        return {
            "dir_delta": dir_delta,
            "sign_score": sign_score,
            "scale_residual_raw": scale_residual_raw,
            "clamped_log_scale_delta": clamped_log_scale_delta,
            "confidence": confidence,
            "motion_bucket_logits": motion_bucket_logits,
        }


def factorized_translation(
    out: Dict[str, torch.Tensor],
    prior_dir: torch.Tensor,
    prior_tmag: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    dir_raw = prior_dir + out["dir_delta"] + prior_dir * out["sign_score"]
    pred_dir_unit = dir_raw / dir_raw.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
    pred_log_scale_delta = out["clamped_log_scale_delta"]
    pred_tmag = prior_tmag * torch.exp(pred_log_scale_delta)
    pred_t = pred_dir_unit * pred_tmag
    pred_bucket = torch.softmax(out["motion_bucket_logits"], dim=1)
    return {
        "pred_dir_unit": pred_dir_unit,
        "pred_log_scale_delta": pred_log_scale_delta,
        "pred_tmag": pred_tmag,
        "pred_t": pred_t,
        "pred_confidence": out["confidence"],
        "pred_motion_bucket_probs": pred_bucket,
    }


def save_training_checkpoint(
    path: Path,
    model: DirectionScaleCalibratedModel,
    image_size: Tuple[int, int],
    numeric_dim: int,
    bucket_stats: Dict[str, Any],
    log_scale_clip: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_size": list(image_size),
            "numeric_dim": int(numeric_dim),
            "bucket_stats": bucket_stats,
            "log_scale_clip": float(log_scale_clip),
            "model_type": "direction_scale_calibrated_geometry",
        },
        path,
    )


def load_training_checkpoint(path: Path, device: torch.device) -> Tuple[DirectionScaleCalibratedModel, Tuple[int, int], int, Dict[str, Any], float]:
    payload = torch.load(path, map_location=device)
    image_size = tuple(int(x) for x in payload["image_size"])
    numeric_dim = int(payload["numeric_dim"])
    log_scale_clip = float(payload["log_scale_clip"])
    model = DirectionScaleCalibratedModel(numeric_dim=numeric_dim, log_scale_clip=log_scale_clip)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, image_size, numeric_dim, payload["bucket_stats"], log_scale_clip


def build_eval_seq_frames(scene: str, seq: str) -> List[Any]:
    return scan_frames(Path("data"), scene=scene, seq=seq).get((scene, seq), [])

