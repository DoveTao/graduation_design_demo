#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from s5e2_adjacent_dense_lib import validation_from_logs, write_json
from s5e9_small_motion_geometry_lib import build_feature_lookups, make_samples


OUT_DIR = Path("checkpoints/S5E9_scale_unit_small_motion_signed_direction_fix_candidate")
OUT_JSON = Path("checkpoints/S5E9_scale_unit_small_motion_signed_direction_fix_candidate.json")
CONFIG_PATH = "configs/s5e9_scale_unit_small_motion_signed_direction_fix.yaml"


class ScaleDataset(Dataset):
    def __init__(self, samples, feature_map):
        self.samples = samples
        self.feature_map = feature_map
        self.cache: Dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        feat = self.feature_map(s, self.cache)
        return {
            "feat": torch.from_numpy(feat.astype(np.float32)),
            "gt_log_scale_delta": torch.tensor([s.gt_logmag - math.log(max(self.feature_map.bucket_prior_for(s), 1.0e-12))], dtype=torch.float32),
            "gt_magnitude": torch.tensor([s.gt_magnitude], dtype=torch.float32),
            "bucket_index": torch.tensor(s.bucket_index, dtype=torch.long),
            "small_motion_mask": torch.tensor(1.0 if s.bucket_name in ("near_static", "small_motion") else 0.0, dtype=torch.float32),
        }


class ScaleModel(nn.Module):
    def __init__(self, feat_dim: int, log_scale_clip: float) -> None:
        super().__init__()
        self.log_scale_clip = float(log_scale_clip)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw_scale_delta = self.net(feat)
        pred_log_scale_delta = self.log_scale_clip * torch.tanh(raw_scale_delta)
        return {"raw_scale_delta": raw_scale_delta, "pred_log_scale_delta": pred_log_scale_delta}


def _build_feature_map(train_samples, val_samples):
    # The feature map is a simple numeric summary: prior magnitude, dt, index and bucket id.
    all_samples = train_samples + val_samples
    bucket_stats = build_feature_lookups(all_samples)

    def bucket_prior_for(sample):
        return bucket_stats[sample.bucket_name]["median"]

    def feature_map(sample, cache):
        prior = bucket_prior_for(sample)
        dt = float(sample.frame_j.timestamp - sample.frame_i.timestamp)
        return np.asarray([
            math.log(max(prior, 1e-12)),
            dt,
            sample.edge_index / max(sample.total_edges, 1),
            float(sample.bucket_index),
        ], dtype=np.float64)

    feature_map.bucket_prior_for = bucket_prior_for  # type: ignore[attr-defined]
    feature_map.bucket_stats = bucket_stats  # type: ignore[attr-defined]
    return feature_map


def _loss(pred_delta, batch, feature_map):
    gt_delta = batch["gt_log_scale_delta"]
    gt_mag = batch["gt_magnitude"]
    small_mask = batch["small_motion_mask"].view(-1, 1)
    loss_scale = torch.nn.functional.smooth_l1_loss(pred_delta, gt_delta, reduction="none")
    pred_mag = torch.exp(pred_delta + torch.log(gt_mag.clamp_min(1e-12)))
    overscale = torch.relu(pred_mag / gt_mag.clamp_min(1e-12) - 4.0)
    loss = (loss_scale * torch.where(small_mask > 0.5, torch.full_like(loss_scale, 0.5), torch.ones_like(loss_scale))).mean() + overscale.mean()
    return loss, {
        "scale_loss": float(loss_scale.mean().detach().cpu()),
        "overscale_rate": float((pred_mag / gt_mag.clamp_min(1e-12) > 4.0).float().mean().detach().cpu()),
        "small_motion_weight": float(torch.where(small_mask > 0.5, torch.full_like(loss_scale, 0.5), torch.ones_like(loss_scale)).mean().detach().cpu()),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_samples, val_samples, split_summary, _bucket_stats = make_samples()
    feature_map = _build_feature_map(train_samples, val_samples)
    train_ds = ScaleDataset(train_samples, feature_map)
    val_ds = ScaleDataset(val_samples, feature_map)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    device = torch.device("cpu")
    feat_dim = len(train_ds[0]["feat"])
    model = ScaleModel(feat_dim=feat_dim, log_scale_clip=math.log(4.0)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    best_score = float("inf")
    best_path = OUT_DIR / "s5e9_scale_best.pt"
    history: List[Dict[str, Any]] = []
    for epoch in range(3):
        model.train()
        train_stats = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["feat"])
            loss, stats = _loss(out["pred_log_scale_delta"], batch, feature_map)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_stats.append(stats)
        model.eval()
        val_scores = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["feat"])
                pred_mag = torch.exp(out["pred_log_scale_delta"] + torch.log(batch["gt_magnitude"].clamp_min(1e-12)))
                ratio = pred_mag / batch["gt_magnitude"].clamp_min(1e-12)
                val_scores.extend(ratio.squeeze(1).cpu().tolist())
        score = float(np.percentile(np.asarray(val_scores, dtype=np.float64), 95))
        history.append({"epoch": epoch, "train_stats": train_stats[-1] if train_stats else {}, "val_p95_ratio": score})
        if score < best_score:
            best_score = score
            torch.save({"state_dict": model.state_dict(), "feat_dim": feat_dim, "log_scale_clip": math.log(4.0), "bucket_stats": feature_map.bucket_stats, "model_type": "scale_calibrated_candidate"}, best_path)

    status = {
        "attempted": True,
        "classification": "S5E9_SCALE_TRAINING_SMOKE",
        "checkpoint": str(best_path),
        "num_train_pairs": len(train_samples),
        "num_val_pairs": len(val_samples),
        "uses_scene01_seq03_for_training": False,
        "history": history,
        "split_summary": split_summary,
        "notes": [
            "S5E9 scale candidate 使用 tight bounded log-scale residual。",
            "small-motion 使用更低权重。",
        ],
        "validation_snapshot": validation_from_logs(),
    }
    write_json(OUT_DIR / "scale_training_status.json", status)
    ckpt = {
        "experiment": "S5E9_scale_unit_small_motion_signed_direction_fix",
        "status": {"experimental_candidate": True, "official_s5_unchanged": True, "not_official_replacement": True},
        "training": status,
        "scale_candidate_checkpoint": str(best_path),
        "validation": validation_from_logs(),
        "allowed_final_classifications": [
            "S5E9_SCALE_UNIT_BUG_FOUND",
            "S5E9_SCALE_RAW_IMPROVED",
            "S5E9_SIGNED_DIRECTION_IMPROVED",
            "S5E9_SMALL_MOTION_IMPROVED",
            "S5E9_RAW_GEOMETRY_IMPROVED",
            "S5E9_GUARD_DEPENDENT",
            "S5E9_INPUT_GEOMETRY_INSUFFICIENT",
            "S5E9_NO_IMPROVEMENT",
            "S5E9_REGRESSION",
        ],
        "final_classification": "S5E9_NO_IMPROVEMENT",
        "s5_official_locked_metrics": {"ate": 7.352288, "drift": 1.327343, "path_ratio": 0.932379, "unchanged": True, "not_replaced_by_s5e9": True},
    }
    write_json(OUT_JSON, ckpt)
    return status


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
