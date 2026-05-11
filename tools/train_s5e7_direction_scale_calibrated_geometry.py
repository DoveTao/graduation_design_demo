#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from s5e2_adjacent_dense_lib import validation_from_logs, write_json
from s5e7_direction_scale_lib import (
    DirectionScaleCalibratedModel,
    build_numeric_features,
    load_npz_model,
    load_ordered_image_pair,
    make_pair_samples_with_buckets,
    save_training_checkpoint,
)


OUT_DIR = Path("checkpoints/S5E7_direction_scale_calibrated_geometry_candidate")
OUT_JSON = Path("checkpoints/S5E7_direction_scale_calibrated_geometry_candidate.json")
CONFIG_PATH = "configs/s5e7_direction_scale_calibrated_geometry.yaml"
S5E2_BASE = Path("checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz")
S5E3_HEADS = Path("checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz")
S5E6 = {
    "rot_mean_deg": 0.9134395040767528,
    "signed_tdir_mean_deg": 51.47429479273258,
    "tdir_abs_mean_deg": 46.07968707914154,
    "anti_parallel_rate": 0.1368653421633554,
    "tmag_median_ratio": 11.352940388835835,
    "tmag_p95_ratio": 63.490479510847564,
    "path_ratio": 1.880843438039364,
    "sim3_ate": 4.012221895981248,
}


class PairDataset(Dataset):
    def __init__(self, samples, image_size, s5e2, s5e3, bucket_stats):
        self.samples = samples
        self.image_size = image_size
        self.s5e2 = s5e2
        self.s5e3 = s5e3
        self.bucket_stats = bucket_stats
        self.cache: Dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = load_ordered_image_pair(s.frame_i, s.frame_j, self.image_size)
        numeric = build_numeric_features(s, self.cache, self.s5e2, self.s5e3, self.bucket_stats)
        prior_dir = numeric[:3]
        prior_tmag = math.exp(float(numeric[3]))
        return {
            "image": image,
            "numeric": torch.from_numpy(numeric.astype(np.float32)),
            "prior_dir": torch.from_numpy(prior_dir.astype(np.float32)),
            "prior_tmag": torch.tensor([prior_tmag], dtype=torch.float32),
            "gt_direction": torch.from_numpy(s.gt_direction.astype(np.float32)),
            "gt_logmag": torch.tensor([s.gt_logmag], dtype=torch.float32),
            "gt_magnitude": torch.tensor([s.gt_magnitude], dtype=torch.float32),
            "bucket_index": torch.tensor(s.bucket_index, dtype=torch.long),
            "small_motion_mask": torch.tensor(1.0 if s.bucket_name == "small_motion" else 0.0, dtype=torch.float32),
        }


def _factorized_outputs(model, batch):
    out = model(batch["image"], batch["numeric"])
    prior_dir = batch["prior_dir"]
    prior_tmag = batch["prior_tmag"]
    dir_raw = prior_dir + out["dir_delta"] + prior_dir * out["sign_score"]
    pred_dir_unit = dir_raw / dir_raw.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
    pred_log_scale_delta = out["clamped_log_scale_delta"]
    pred_tmag = prior_tmag * torch.exp(pred_log_scale_delta)
    pred_t = pred_dir_unit * pred_tmag
    return {
        "pred_dir_unit": pred_dir_unit,
        "pred_log_scale_delta": pred_log_scale_delta,
        "pred_tmag": pred_tmag,
        "pred_t": pred_t,
        "confidence": out["confidence"],
        "motion_bucket_logits": out["motion_bucket_logits"],
    }


def _loss_bundle(pred, batch, bucket_stats):
    gt_dir = batch["gt_direction"]
    gt_logmag = batch["gt_logmag"]
    gt_mag = batch["gt_magnitude"]
    small_mask = batch["small_motion_mask"].view(-1)
    dot = (pred["pred_dir_unit"] * gt_dir).sum(dim=1).clamp(-1.0, 1.0)
    abs_dot = dot.abs()
    dir_loss = 1.0 - dot
    anti_soft = torch.relu(-dot)
    anti_hard = torch.relu(-dot - 0.5)
    tdir_abs_aux = 1.0 - abs_dot
    log_ratio = torch.log(pred["pred_tmag"].clamp_min(1.0e-6) / gt_mag.clamp_min(1.0e-6))
    scale_loss = torch.nn.functional.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio), reduction="none")
    scale_weight = torch.where(small_mask > 0.5, torch.full_like(scale_loss, 0.35), torch.ones_like(scale_loss))
    bucket_loss = torch.nn.functional.cross_entropy(pred["motion_bucket_logits"], batch["bucket_index"], reduction="mean")
    path_ratio = pred["pred_tmag"].sum() / gt_mag.sum().clamp_min(1.0e-6)
    path_length_consistency = (path_ratio - 1.0).abs()
    p95_penalty = torch.quantile(log_ratio.abs(), 0.95)
    dir_consistency = torch.tensor(0.0, dtype=path_ratio.dtype, device=path_ratio.device)
    smoothness = torch.tensor(0.0, dtype=path_ratio.dtype, device=path_ratio.device)
    if pred["pred_dir_unit"].shape[0] > 1:
        dir_consistency = (1.0 - (pred["pred_dir_unit"][1:] * pred["pred_dir_unit"][:-1]).sum(dim=1)).mean()
        smoothness = (pred["pred_log_scale_delta"][1:] - pred["pred_log_scale_delta"][:-1]).abs().mean()
    loss = (
        dir_loss.mean()
        + 0.35 * tdir_abs_aux.mean()
        + 0.35 * anti_soft.mean()
        + 0.7 * anti_hard.mean()
        + 0.6 * (scale_loss * scale_weight).mean()
        + 0.15 * bucket_loss
        + 0.25 * path_length_consistency
        + 0.2 * p95_penalty
        + 0.1 * dir_consistency
        + 0.1 * smoothness
    )
    return {
        "loss": loss,
        "signed_tdir": float(dir_loss.mean().detach().cpu()),
        "tdir_abs_aux": float(tdir_abs_aux.mean().detach().cpu()),
        "anti_parallel_penalty": float((anti_soft.mean() + anti_hard.mean()).detach().cpu()),
        "scale_residual_smooth_l1": float((scale_loss * scale_weight).mean().detach().cpu()),
        "robust_tmag_log": float(scale_loss.mean().detach().cpu()),
        "tmag_p95_penalty": float(p95_penalty.detach().cpu()),
        "path_length_consistency": float(path_length_consistency.detach().cpu()),
        "short_window_direction_consistency": float(dir_consistency.detach().cpu()),
        "smoothness": float(smoothness.detach().cpu()),
    }


def _eval_metrics(model, loader, device):
    signed, abs_signed, anti, severe, ratios, gt_mag, pred_mag = [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = _factorized_outputs(model, batch)
            dot = (pred["pred_dir_unit"] * batch["gt_direction"]).sum(dim=1).clamp(-1.0, 1.0)
            abs_dot = dot.abs()
            ang = torch.rad2deg(torch.arccos(dot))
            ang_abs = torch.rad2deg(torch.arccos(abs_dot))
            ratio = pred["pred_tmag"].squeeze(1) / batch["gt_magnitude"].squeeze(1).clamp_min(1.0e-6)
            signed.extend(ang.cpu().tolist())
            abs_signed.extend(ang_abs.cpu().tolist())
            anti.extend((dot < 0).float().cpu().tolist())
            severe.extend((ang > 120).float().cpu().tolist())
            ratios.extend(ratio.cpu().tolist())
            gt_mag.extend(batch["gt_magnitude"].squeeze(1).cpu().tolist())
            pred_mag.extend(pred["pred_tmag"].squeeze(1).cpu().tolist())
    ratios_np = np.asarray(ratios, dtype=np.float64)
    path_ratio = float(np.sum(pred_mag) / max(np.sum(gt_mag), 1.0e-12))
    return {
        "signed_tdir_mean_deg": float(np.mean(signed)),
        "signed_tdir_median_deg": float(np.percentile(signed, 50)),
        "signed_tdir_p90_deg": float(np.percentile(signed, 90)),
        "tdir_abs_mean_deg": float(np.mean(abs_signed)),
        "tdir_abs_median_deg": float(np.percentile(abs_signed, 50)),
        "tdir_abs_p90_deg": float(np.percentile(abs_signed, 90)),
        "anti_parallel_rate": float(np.mean(anti)),
        "severe_wrong_sign_rate": float(np.mean(severe)),
        "tmag_median_ratio": float(np.percentile(ratios_np, 50)),
        "tmag_p95_ratio": float(np.percentile(ratios_np, 95)),
        "path_ratio": path_ratio,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not (S5E2_BASE.exists() and S5E3_HEADS.exists()):
        status = {
            "attempted": True,
            "classification": "S5E7_TRAINING_BLOCKED",
            "notes": ["缺少 S5E2/S5E3 基础模型，无法构建 S5E7。"],
        }
        write_json(OUT_DIR / "training_status.json", status)
        write_json(OUT_JSON, {"experiment": "S5E7_direction_scale_calibrated_geometry", "status": {"experimental_candidate": True, "official_s5_unchanged": True, "not_official_replacement": True}, "training": status, "validation": validation_from_logs(), "final_classification": "S5E7_REGRESSION"})
        return status

    s5e2 = load_npz_model(S5E2_BASE)
    s5e3 = load_npz_model(S5E3_HEADS)
    train_samples, val_samples, split_summary, bucket_stats = make_pair_samples_with_buckets()
    image_size = (64, 128)
    train_ds = PairDataset(train_samples, image_size, s5e2, s5e3, bucket_stats)
    val_ds = PairDataset(val_samples, image_size, s5e2, s5e3, bucket_stats)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    numeric_dim = len(train_ds[0]["numeric"])
    device = torch.device("cpu")
    model = DirectionScaleCalibratedModel(numeric_dim=numeric_dim, log_scale_clip=1.25).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    best_score = float("inf")
    best_epoch = -1
    best_path = OUT_DIR / "s5e7_direction_scale_best.pt"
    history: List[Dict[str, Any]] = []
    for epoch in range(3):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = _factorized_outputs(model, batch)
            losses = _loss_bundle(pred, batch, bucket_stats)
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
        model.eval()
        val_metrics = _eval_metrics(model, val_loader, device)
        score = (
            val_metrics["signed_tdir_mean_deg"]
            + 0.6 * val_metrics["tdir_abs_mean_deg"]
            + 25.0 * abs(math.log(max(val_metrics["tmag_median_ratio"], 1.0e-12)))
            + 0.5 * val_metrics["tmag_p95_ratio"]
            + 12.0 * abs(val_metrics["path_ratio"] - 1.0)
            + 40.0 * val_metrics["anti_parallel_rate"]
            + 20.0 * val_metrics["severe_wrong_sign_rate"]
        )
        history.append({"epoch": epoch, "val_metrics": val_metrics})
        if score < best_score:
            best_score = score
            best_epoch = epoch
            save_training_checkpoint(best_path, model, image_size, numeric_dim, bucket_stats, log_scale_clip=1.25)

    global_prior = bucket_stats["global"]
    training_status = {
        "attempted": True,
        "classification": "S5E7_TRAINING_SMOKE_ONLY",
        "num_train_pairs": len(train_samples),
        "num_val_pairs": len(val_samples),
        "uses_scene01_seq03_for_training": False,
        "model_type": "direction_scale_calibrated_geometry",
        "architecture_summary": "S5E7 使用 ordered image pair CNN 与 numeric prior，分别输出 pred_dir_unit 与 bounded log-scale residual。",
        "train_magnitude_prior": global_prior,
        "bucket_priors": {
            "small_motion": bucket_stats["small_motion"],
            "normal_motion": bucket_stats["normal_motion"],
            "large_motion": bucket_stats["large_motion"],
        },
        "losses": {
            "so3_geodesic": True,
            "signed_tdir": True,
            "tdir_abs_aux": True,
            "anti_parallel_penalty": True,
            "scale_residual_smooth_l1": True,
            "robust_tmag_log": True,
            "tmag_p95_penalty": True,
            "path_length_consistency": True,
            "short_window_direction_consistency": True,
            "smoothness": True,
        },
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "history": history,
        "notes": [
            "S5E7 以方向学习为主目标，尺度只学习 bounded log-scale residual。",
            "guard 只作为 fallback，不应被解释为真实几何改善。",
            "scene01/seq03 GT 未用于训练。",
        ],
        "split_summary": split_summary,
    }
    write_json(OUT_DIR / "training_status.json", training_status)
    ckpt = {
        "experiment": "S5E7_direction_scale_calibrated_geometry",
        "status": {"experimental_candidate": True, "official_s5_unchanged": True, "not_official_replacement": True},
        "baseline_reference": {"s5e6": S5E6},
        "training": {
            "attempted": True,
            "classification": training_status["classification"],
            "config": CONFIG_PATH,
            "checkpoint_dir": str(OUT_DIR),
            "uses_scene01_seq03_for_training": False,
            "model_type": "direction_scale_calibrated_geometry",
            "train_magnitude_prior": global_prior,
            "bucket_priors": training_status["bucket_priors"],
            "best_checkpoint": str(best_path),
            "losses": training_status["losses"],
            "notes": training_status["notes"],
        },
        "direction_scale_failure_audit": {},
        "motion_bucket_audit": {},
        "adjacent_dense_export": {"available": False, "coverage": None, "all_edges_traceable": False, "num_poses": None, "num_edges": None, "direct_adjacent_prediction_edges": None, "trajectory_path": "external_baselines/results/s5e7_traceable_dense/scene01_seq03_s5e7_traceable_dense_tum.txt", "raw_trajectory_path": "external_baselines/results/s5e7_traceable_dense/scene01_seq03_s5e7_traceable_dense_raw_tum.txt", "edge_provenance": "external_baselines/results/s5e7_traceable_dense/edge_provenance.jsonl"},
        "raw_prediction_metrics": {},
        "guarded_prediction_metrics": {},
        "component_metrics": {},
        "external_eval": {"none": {}, "se3": {}, "sim3": {}},
        "raw_external_eval": {"none": {}, "se3": {}, "sim3": {}},
        "comparison_vs_s5e6": {},
        "comparison_to_orbslam3": {},
        "s5_official_locked_metrics": {"ate": 7.352288, "drift": 1.327343, "path_ratio": 0.932379, "unchanged": True, "not_replaced_by_s5e7": True},
        "validation": validation_from_logs(),
        "allowed_final_classifications": [
            "S5E7_DIRECTION_SCALE_IMPROVED",
            "S5E7_DIRECTION_ONLY_IMPROVED",
            "S5E7_SCALE_ONLY_IMPROVED",
            "S5E7_GUARD_DEPENDENT",
            "S5E7_NO_GEOMETRY_IMPROVEMENT",
            "S5E7_REGRESSION",
        ],
        "final_classification": "S5E7_NO_GEOMETRY_IMPROVEMENT",
    }
    write_json(OUT_JSON, ckpt)
    return training_status


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
